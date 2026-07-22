"""Phase 0 Plan B driver: planar-complex BF16 cublasLt probe (review §7)."""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np


def reference_complex_matmul(ar, ai, br, bi):
    """Float32 reference for C = (A)(B), A=ar+j*ai, B=br+j*bi. Returns (cr, ci) float32.

    Inputs are expected to already be in the comparison dtype (e.g. bf16 view as
    float32). We cast to float32 here WITHOUT regenerating from any earlier source,
    so the comparison against the cublasLt path is apples-to-apples on the same
    rounded BF16 values.
    """
    A = ar.astype(np.float32) + 1j * ai.astype(np.float32)
    B = br.astype(np.float32) + 1j * bi.astype(np.float32)
    C = A @ B
    return C.real.astype(np.float32), C.imag.astype(np.float32)


def judge_capability(
    max_rel_err,
    perf_ratio_vs_c64,
    algo_count,
    workspace_bytes,
    output_bytes,
    has_four_real_temps,
    max_abs_err=0.0,
):
    """§7.5 capability judgment for the planar-complex BF16 planar probe.

    The accuracy gate keys on max-RELATIVE-error: BF16 output inherently rounds
    to ~8 mantissa bits (~0.4% relative error on standard-normal inputs), so an
    absolute-error gate mis-flags the spec-compliant BF16-output path. The
    absolute error is carried as a diagnostic via ``max_abs_err`` (reported in
    the failure reason only).
    """
    reasons = []
    if algo_count == 0:
        return {
            "status": "NOT_SUPPORTED",
            "reason": "SM120 returned no algorithm for planar C16BF",
        }
    if max_rel_err > 1e-2:
        reasons.append(
            f"accuracy fail (max_rel_err={max_rel_err:.2e}, "
            f"max_abs_err={max_abs_err:.2e})"
        )
    if perf_ratio_vs_c64 < 1.3:
        reasons.append(f"no speedup vs c64 (perf_ratio={perf_ratio_vs_c64:.2f} < 1.3)")
    if has_four_real_temps:
        reasons.append(
            "four full-size real temp outputs observed (no compression benefit)"
        )
    if workspace_bytes > output_bytes:
        reasons.append(
            f"workspace ({workspace_bytes}) exceeds output ({output_bytes}) "
            "— cancels compression"
        )
    if reasons:
        return {"status": "NOT_SUPPORTED", "reason": "; ".join(reasons)}
    return {
        "status": "SUPPORTED",
        "reason": "usable algo + correct + >=1.3x vs c64 + compression net positive",
    }


def load_ext():
    """Load (and cache) the pybind11 extension built by _phase0_cublaslt_build."""
    from results._phase0_cublaslt_build import load_ext as _le

    return _le()


def load_c1_c2_shapes(
    csv_path="results/phase0/contraction_shapes.csv", min_bytes=64 << 20
):
    """Read contraction_shapes.csv rows whose tensor ``bytes`` >= ``min_bytes``.

    Returns a list of dicts {M,N,K,bytes,node_id}; malformed rows (missing or
    non-int fields) are silently skipped so a dirty CSV never aborts the probe.
    """
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                if int(r["bytes"]) >= min_bytes:
                    rows.append(
                        {
                            "M": int(r["M"]),
                            "N": int(r["N"]),
                            "K": int(r["K"]),
                            "bytes": int(r["bytes"]),
                            "node_id": r["node_id"],
                        }
                    )
            except (KeyError, ValueError):
                continue
    return rows


# --------------------------------------------------------------------------- #
# BF16 bit-format helpers (numpy has no native bfloat16; torch does).
# These replace the brief's ``ar.astype(float16).view(uint16)`` proxy, which
# produces FP16 bit patterns that decode to wrong values when cublasLt reads
# them as BF16. torch.bfloat16 rounds the float32 source the way real BF16
# storage would, so the cublasLt path and the numpy reference see identical
# BF16-rounded input values (apples-to-apples per reference_complex_matmul's
# documented contract).
# --------------------------------------------------------------------------- #
def _f32_to_bf16_bits_and_upcast(f32):
    """Round float32 numpy -> BF16. Returns (uint16 BF16 bits, float32 upcast)."""
    import torch

    t = torch.from_numpy(f32).to(torch.bfloat16)
    bits = t.view(torch.int16).numpy().astype(np.uint16)
    upcast = t.to(torch.float32).numpy()
    return bits, upcast


def _bf16_bits_to_f32(u16):
    """Decode uint16 BF16-bit numpy array -> float32 (torch reinterpret)."""
    import torch

    return (
        torch.from_numpy(u16.view(np.int16))
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )


def _time_c64_gpu_matmul(ar, ai, br, bi, n_time=5):
    """§7.3 complex64 production baseline: GPU torch.complex64 matmul kernel time.

    Builds A=ar+j*ai, B=br+j*bi as torch.complex64 on cuda and times ``A @ B``
    (warmup + median of ``n_time`` with torch.cuda.synchronize). Data is
    GPU-resident across the timed iterations, matching how a production c64 path
    amortizes host<->device transfer — i.e. this is the kernel cost the planar
    BF16 path would have to beat to justify the BF16 compression.
    """
    import torch

    A = torch.complex(torch.from_numpy(ar).cuda(), torch.from_numpy(ai).cuda())
    B = torch.complex(torch.from_numpy(br).cuda(), torch.from_numpy(bi).cuda())
    _ = A @ B
    torch.cuda.synchronize()
    times = []
    for _ in range(n_time):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = A @ B
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    c64_ms = float(np.median(times))
    del A, B
    torch.cuda.empty_cache()
    return c64_ms


def _time_c64_full_roundtrip(ar, ai, br, bi, n_time=5):
    """c64 cost matched in scope to the planar probe call (H2D+kernel+D2H).

    Per iteration: upload numpy float32 re/im planes to cuda, build complex64
    A/B, run ``A @ B``, download the result. The planar BF16 probe call exposed
    by the extension has the same host<->device round-trip shape, so this is the
    apples-to-apples baseline (BF16 moves half the bytes, so it can win here on
    bandwidth-bound shapes even though the kernel-only c64 baseline in
    ``_time_c64_gpu_matmul`` is always faster). Reported as a diagnostic; the
    capability gate uses the kernel-only ratio per the controller's §7.3 spec.
    """
    import torch

    times = []
    for _ in range(n_time):
        t0 = time.perf_counter()
        Ar = torch.from_numpy(ar).cuda()
        Ai = torch.from_numpy(ai).cuda()
        Br = torch.from_numpy(br).cuda()
        Bi = torch.from_numpy(bi).cuda()
        A = torch.complex(Ar, Ai)
        B = torch.complex(Br, Bi)
        C = A @ B
        torch.cuda.synchronize()
        _ = C.cpu().numpy()
        times.append((time.perf_counter() - t0) * 1e3)
    full_ms = float(np.median(times))
    del A, B, C, Ar, Ai, Br, Bi
    torch.cuda.empty_cache()
    return full_ms


def run_matrix(shapes, out_dir="results/phase0"):
    """Run the §7 capability + performance matrix on ``shapes``.

    For each shape:
      * probe_planar_capability (algo enumeration, no execution)
      * planar BF16-output matmul (spec-compliant) timed warmup+median(5)
      * correctness vs numpy float32 reference on BF16-rounded inputs
        (max-abs + signal-floored max-rel; gate is max-rel < 1e-2)
      * GPU complex64 matmul baseline (§7.3) timed warmup+median(5)

    Writes cublaslt_planar_{capability.json,bench.csv,accuracy.csv} and
    returns {capability, best_ratio, worst_rel, worst_abs}. Shapes that would
    exceed the ~8 GB device budget are recorded as ``oom`` rather than aborting
    the matrix.
    """
    import torch  # noqa: F401  (availability guard; _time_c64_* imports it too)

    os.makedirs(out_dir, exist_ok=True)
    ext = load_ext()
    bench_rows, acc_rows = [], []
    perf_ratios, fair_ratios, max_rels, max_abss, algo_counts, workspaces = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    oom_bytes = 8 << 30
    n_time = 5
    signal_floor = 0.5  # well below |C|~sqrt(K); only stops near-zero ref inflation
    dash6 = ["-"] * 6  # padding for non-ok bench rows (6 numeric cols after status)

    for s in shapes:
        m, n, k = s["M"], s["N"], s["K"]

        # OOM guard: c64 = A+B+C complex64 ; bf16 = A,B in (re+im) + C out (re+im)
        c64_bytes = (m * k + k * n + m * n) * 8
        bf16_bytes = (m * k + k * n) * 2 * 2 + m * n * 2 * 2
        if c64_bytes > oom_bytes or bf16_bytes > oom_bytes:
            bench_rows.append([m, n, k, "oom", f"alloc>{oom_bytes >> 30}GB", *dash6])
            continue

        info = ext.probe_planar_capability(m, n, k)
        algo_counts.append(info.get("algo_count", 0))
        workspaces.append(info.get("workspace_bytes", 0))
        if info.get("algo_count", 0) == 0:
            bench_rows.append([m, n, k, "no-algo", *dash6])
            continue

        rng = np.random.default_rng(1)
        ar = rng.standard_normal((m, k)).astype(np.float32)
        ai = rng.standard_normal((m, k)).astype(np.float32)
        br = rng.standard_normal((k, n)).astype(np.float32)
        bi = rng.standard_normal((k, n)).astype(np.float32)
        ar_bf, ar_f = _f32_to_bf16_bits_and_upcast(ar)
        ai_bf, ai_f = _f32_to_bf16_bits_and_upcast(ai)
        br_bf, br_f = _f32_to_bf16_bits_and_upcast(br)
        bi_bf, bi_f = _f32_to_bf16_bits_and_upcast(bi)

        try:
            # warmup (first call may init algo state) then median-of-n timing;
            # each call is the full H2D+kernel+D2H round-trip the extension exposes.
            ext.planar_complex_matmul_bf16(
                ar_bf, ai_bf, br_bf, bi_bf, m, n, k, out_dtype="bf16"
            )
            times = []
            for _ in range(n_time):
                t0 = time.perf_counter()
                cr_u16, ci_u16 = ext.planar_complex_matmul_bf16(
                    ar_bf, ai_bf, br_bf, bi_bf, m, n, k, out_dtype="bf16"
                )
                times.append((time.perf_counter() - t0) * 1e3)
            bf_ms = float(np.median(times))
        except Exception as e:  # noqa: BLE001  (record exec-fail, keep going)
            bench_rows.append([m, n, k, "exec-fail", str(e)[:60], *dash6])
            continue

        cr_ref, ci_ref = reference_complex_matmul(ar_f, ai_f, br_f, bi_f)
        cr = _bf16_bits_to_f32(cr_u16)
        ci = _bf16_bits_to_f32(ci_u16)

        err_r = np.abs(cr - cr_ref)
        err_i = np.abs(ci - ci_ref)
        max_abs = max(float(np.max(err_r)), float(np.max(err_i)))
        denom_r = np.maximum(np.abs(cr_ref), signal_floor)
        denom_i = np.maximum(np.abs(ci_ref), signal_floor)
        max_rel = max(float(np.max(err_r / denom_r)), float(np.max(err_i / denom_i)))
        max_rels.append(max_rel)
        max_abss.append(max_abs)

        # Controller §7.3 baseline: GPU c64 kernel-only (data resident, warmup).
        c64_gpu_ms = _time_c64_gpu_matmul(ar, ai, br, bi, n_time=n_time)
        # Diagnostic: c64 full round-trip matched to the planar probe's scope.
        c64_full_ms = _time_c64_full_roundtrip(ar, ai, br, bi, n_time=n_time)
        ratio = c64_gpu_ms / bf_ms if bf_ms > 0 else 0.0
        fair_ratio = c64_full_ms / bf_ms if bf_ms > 0 else 0.0
        perf_ratios.append(ratio)
        fair_ratios.append(fair_ratio)
        bench_rows.append(
            [
                m,
                n,
                k,
                "ok",
                f"{bf_ms:.3f}",
                f"{c64_gpu_ms:.3f}",
                f"{c64_full_ms:.3f}",
                f"{ratio:.3f}",
                f"{fair_ratio:.3f}",
                info.get("algo_count", 0),
            ]
        )
        acc_rows.append([m, n, k, f"{max_abs:.2e}", f"{max_rel:.2e}"])

    best_ratio = max(perf_ratios) if perf_ratios else 0.0
    best_fair_ratio = max(fair_ratios) if fair_ratios else 0.0
    worst_rel = max(max_rels) if max_rels else 1e9
    worst_abs = max(max_abss) if max_abss else 0.0
    max_algo = max(algo_counts) if algo_counts else 0
    max_ws = max(workspaces) if workspaces else 0
    # Capability gate keys on the controller's kernel-only c64 ratio (§7.3 spec).
    cap = judge_capability(
        max_rel_err=worst_rel,
        perf_ratio_vs_c64=best_ratio,
        algo_count=max_algo,
        workspace_bytes=max_ws,
        output_bytes=max((s.get("bytes", 0) for s in shapes), default=0),
        has_four_real_temps=False,
        max_abs_err=worst_abs,
    )
    summary = {
        "capability": cap,
        "best_perf_ratio_vs_c64_gpu": best_ratio,
        "best_perf_ratio_vs_c64_full": best_fair_ratio,
        "worst_max_rel_err": worst_rel,
        "worst_max_abs_err": worst_abs,
        "max_algo_count": max_algo,
        "max_workspace_bytes": max_ws,
        "shapes_tested": len(shapes),
        "shapes_ok": sum(1 for r in bench_rows if r[3] == "ok"),
        "c64_kernel_baseline": "torch.complex64 GPU kernel (warmup+median of 5)",
        "c64_full_baseline": "torch.complex64 H2D+kernel+D2H (matched scope, median of 5)",
        "planar_timing": "BF16-output full call, warmup+median of 5",
        "gate_note": (
            "perf gate uses kernel-only c64 ratio (conservative: planar call "
            "includes H2D+D2H the c64 kernel baseline does not); "
            "best_perf_ratio_vs_c64_full is the scope-matched diagnostic"
        ),
    }
    with open(os.path.join(out_dir, "cublaslt_planar_capability.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_csv(
        os.path.join(out_dir, "cublaslt_planar_bench.csv"),
        [
            "M",
            "N",
            "K",
            "status",
            "bf16_ms",
            "c64_gpu_ms",
            "c64_full_ms",
            "c64gpu_over_bf16",
            "c64full_over_bf16",
            "algo_count",
        ],
        bench_rows,
    )
    _write_csv(
        os.path.join(out_dir, "cublaslt_planar_accuracy.csv"),
        ["M", "N", "K", "max_abs_err", "max_rel_err"],
        acc_rows,
    )
    return {
        "capability": cap,
        "best_ratio": best_ratio,
        "best_fair_ratio": best_fair_ratio,
        "worst_rel": worst_rel,
        "worst_abs": worst_abs,
    }


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    # Square sanity (known-answer) + distinct real contraction shapes. Dedup by
    # (M,N,K): the CSV repeats identical shapes across many node_ids and running
    # duplicates only burns time without adding signal.
    raw = load_c1_c2_shapes()
    seen, real_shapes = set(), []
    for s in raw:
        key = (s["M"], s["N"], s["K"])
        if key not in seen:
            seen.add(key)
            real_shapes.append(s)
    real_shapes = real_shapes[:6]  # cap to bound runtime
    shapes = [
        {"M": 256, "N": 256, "K": 256},
        {"M": 2048, "N": 2048, "K": 2048},
    ] + real_shapes
    result = run_matrix(shapes)
    print(result)
