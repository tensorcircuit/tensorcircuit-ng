"""Task 8 CUTLASS/CuTe SM120 4M probe driver (final-remediation §11).

Replaces the PlanB-T4 compile-only smoke. Compiles CUTLASS kernels in-tree
via torch.utils.cpp_extension with an isolated nvcc_spike CUDA_HOME, runs
them on sm_120, and aggregates a cutlass-sm120-4m-v1 capability verdict.
"""

from __future__ import annotations

import glob
import os

SCHEMA_VERSION = "cutlass-sm120-4m-v1"
_HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(_HERE, "cpp")
SRC = os.path.join(CPP_DIR, "cutlass_4m.cu")
_CONTRACTION_SHAPES_CSV = os.path.join(
    os.path.dirname(_HERE), "phase0", "contraction_shapes.csv"
)


def discover_paths() -> dict:
    """Discover CUTLASS_ROOT, CUDA_HOME, NVCC from env (no hardcoded /home paths)."""
    home = os.path.expanduser("~")
    cutlass_root = os.environ.get("CUTLASS_ROOT", os.path.join(home, "cutlass_spike"))
    cuda_home = os.environ.get(
        "CUDA_HOME", os.path.join(home, "miniconda3", "envs", "nvcc_spike")
    )
    nvcc = os.environ.get("NVCC", "")
    if not nvcc:
        cands = [os.path.join(cuda_home, "bin", "nvcc")]
        nvcc = next((c for c in cands if os.path.exists(c)), "")
    # CUTLASS splits headers: core in <root>/include, util helpers (packed_stride,
    # reference/device/gemm, host_tensor, ...) in <root>/tools/util/include. The
    # Sm100 path uses cutlass::make_cute_packed_stride from the latter.
    cutlass_util_include = os.path.join(cutlass_root, "tools", "util", "include")
    return {
        "cutlass_root": cutlass_root,
        "cutlass_util_include": cutlass_util_include,
        "cuda_home": cuda_home,
        "nvcc": nvcc,
    }


def build_extension(name: str = "cutlass_4m", extra_defines: list[str] | None = None):
    """Compile cpp/cutlass_4m.cu via torch.utils.cpp_extension (ext.cpp build style).

    CUDA_HOME must point at a toolkit whose nvcc targets sm_120 (nvcc_spike env).
    Returns the loadable module. `name` separates the cached sm100 build from
    the sm80 build so a sm100 compile failure never poisons the sm80 cache.
    """
    import torch  # noqa: F401  (ensures torch + its bundled cuda runtime present)
    from torch.utils.cpp_extension import load

    p = discover_paths()
    os.environ.setdefault("CUDA_HOME", p["cuda_home"])
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ["PATH"] = (
        os.path.join(p["cuda_home"], "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    cflags = ["-std=c++17", "-O2"]
    if extra_defines:
        cflags += extra_defines
    include_paths = [os.path.join(p["cutlass_root"], "include")]
    if os.path.isdir(p["cutlass_util_include"]):
        include_paths.append(p["cutlass_util_include"])
    return load(
        name=name,
        sources=[SRC],
        extra_include_paths=include_paths,
        extra_cuda_cflags=cflags,
        verbose=False,
    )


def four_m_coefficients() -> dict:
    """Signs for the 4-real-GEMM complex decomposition (C = A·B)."""
    return {
        "rec_rea_reb": +1.0,
        "rec_ima_imb": -1.0,
        "imc_rea_imb": +1.0,
        "imc_ima_reb": +1.0,
    }


def c64_reference(ReA, ImA, ReB, ImB):
    """Complex64 reference product via numpy (CPU). Returns (ReC, ImC) float32."""
    import numpy as np

    A = (ReA.astype(np.float32) + 1j * ImA.astype(np.float32)).astype(np.complex64)
    B = (ReB.astype(np.float32) + 1j * ImB.astype(np.float32)).astype(np.complex64)
    C = A @ B
    return C.real.astype(np.float32), C.imag.astype(np.float32)


def _bf16_cuda(t):
    import torch

    return torch.as_tensor(t, device="cuda", dtype=torch.bfloat16)


def run_single_4m(kernel_path: str, shapes, seeds=(0, 1, 2)) -> dict:
    """Run CUTLASS single 4M GEMM on `shapes` x `seeds`, compare to c64 reference.

    kernel_path in {"sm80_fallback", "sm100_native", "sm120_native"}.
      * "sm120_native" — native consumer-Blackwell (arch::Sm120) attempt; falls
        back to sm80 on any failure (records `sm120_blocker`).
      * "sm100_native" — datacenter-Blackwell (arch::Sm100) attempt; falls back
        to sm80 on any failure (records `sm100_blocker`).
      * "sm80_fallback" — proven 2.x Ampere-era MMA path (ko_ratio ~2.71x at
        1024^3).

    Returns correctness plus resource and latency measured on the largest shape
    (Task 3).
    """
    if kernel_path == "sm120_native":
        return _attempt_sm120_then_sm80(shapes, seeds)
    if kernel_path == "sm100_native":
        return _attempt_sm100_then_sm80(shapes, seeds)
    return _run_sm80(shapes, seeds)


def _attempt_sm120_then_sm80(shapes, seeds) -> dict:
    """Genuine attempt at the native Sm120 (consumer Blackwell, RTX 5070 Ti)
    4M path; fall back to Sm80 on any failure.

    CUTLASS_ARCH_MMA_SM120_ENABLED fires at __CUDA_ARCH__==1200 (our GPU), so
    unlike Sm100 this is the *correct* native arch tag. However CUTLASS 3.x's
    Sm120 collective builder is documented F8F6F4-only (FP8/FP6/FP4); a BF16
    instantiation typically fails to compile — recorded verbatim as
    sm120_blocker. Built under a SEPARATE torch extension name so failure
    cannot poison the cached sm80 build.
    """
    try:
        mod = build_extension(
            name="cutlass_4m_sm120",
            extra_defines=["-DCUTLASS_ENABLE_SM120_4M=1"],
        )
        if not hasattr(mod, "has_sm120") or not mod.has_sm120():
            raise RuntimeError(
                "HAS_CUTLASS_4M_SM120=0 after build "
                "(CUTLASS_ARCH_MMA_SM120_SUPPORTED undefined)"
            )
        return _run_with_module(mod, "sm120_native", shapes, seeds)
    except Exception as exc:
        # Transparent fallback — record the verbatim compile/run error.
        return _run_sm80(shapes, seeds, sm120_blocker=str(exc))


def _attempt_sm100_then_sm80(shapes, seeds) -> dict:
    """Genuine attempt at the 3.x Sm100 4M path; fall back to Sm80 on any failure.

    Builds the kernel under a SEPARATE torch extension name
    (cutlass_4m_sm100) so a compile failure does not poison the cached 2.x
    Sm80 build. Any exception (compile error, link error, runtime failure,
    has_sm100()==False) is caught and recorded verbatim as sm100_blocker.
    """
    try:
        mod = build_extension(
            name="cutlass_4m_sm100",
            extra_defines=["-DCUTLASS_ENABLE_SM100_4M=1"],
        )
        if not hasattr(mod, "has_sm100") or not mod.has_sm100():
            # Compiled with the guard set, but the inner
            # CUTLASS_ARCH_MMA_SM100_SUPPORTED branch did not fire — Sm100
            # path not actually present in this build.
            raise RuntimeError(
                "HAS_CUTLASS_4M_SM100=0 after build "
                "(CUTLASS_ARCH_MMA_SM100_SUPPORTED undefined)"
            )
        return _run_with_module(mod, "sm100_native", shapes, seeds)
    except Exception as exc:
        # Transparent fallback — 3.x Sm100 does not instantiate/run on this
        # toolchain. Record the verbatim error for the artifact.
        return _run_sm80(shapes, seeds, sm100_blocker=str(exc))


def _run_sm80(shapes, seeds, sm100_blocker=None, sm120_blocker=None) -> dict:
    """Task 2/3 2.x Sm80 path. Optionally records blockers so the artifact can
    explain why a fallback happened (rather than sm80 being the requested
    path)."""
    mod = build_extension()  # default name=cutlass_4m, no extra_defines
    r = _run_with_module(mod, "sm80_fallback", shapes, seeds)
    if sm100_blocker is not None:
        r["sm100_blocker"] = sm100_blocker
    if sm120_blocker is not None:
        r["sm120_blocker"] = sm120_blocker
    return r


def _run_with_module(mod, kernel_path: str, shapes, seeds) -> dict:
    """Shared correctness + resource + latency runner for all kernel paths.

    Picks mod.cutlass_4m_sm80 / _sm100 / _sm120 based on kernel_path;
    everything else is identical.
    """
    import numpy as np
    import torch  # noqa: F401  (ensures torch CUDA tensors are usable below)

    assert kernel_path in ("sm80_fallback", "sm100_native", "sm120_native")
    gemm_fn = {
        "sm80_fallback": mod.cutlass_4m_sm80,
        "sm100_native": mod.cutlass_4m_sm100,
        "sm120_native": mod.cutlass_4m_sm120,
    }[kernel_path]
    r = {
        "kernel_path": kernel_path,
        "compiles": True,
        "runs": True,
    }
    worst = {"max_rel": 0.0, "max_abs": 0.0, "nan_inf": False}
    for M, K, N in shapes:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            ReA = rng.standard_normal((M, K)).astype(np.float32)
            ImA = rng.standard_normal((M, K)).astype(np.float32)
            ReB = rng.standard_normal((K, N)).astype(np.float32)
            ImB = rng.standard_normal((K, N)).astype(np.float32)
            # BF16 CUDA tensors feed the kernel; their BF16-rounded values also
            # feed the reference (apples-to-apples on the same rounded inputs,
            # matching Task 6 cublaslt's reference_complex_matmul convention) so
            # the comparison isolates kernel numerical error rather than BF16
            # input quantization.
            ReA_bf = _bf16_cuda(ReA)
            ImA_bf = _bf16_cuda(ImA)
            ReB_bf = _bf16_cuda(ReB)
            ImB_bf = _bf16_cuda(ImB)
            refRe, refIm = c64_reference(
                ReA_bf.float().cpu().numpy(),
                ImA_bf.float().cpu().numpy(),
                ReB_bf.float().cpu().numpy(),
                ImB_bf.float().cpu().numpy(),
            )
            ReC, ImC = gemm_fn(ReA_bf, ImA_bf, ReB_bf, ImB_bf)
            gotRe = ReC.cpu().numpy()
            gotIm = ImC.cpu().numpy()
            # signal-floored rel-err (per §7, matching Task 6 cublaslt convention):
            # per-element denom = max(|ref|, 1% of peak); floor stops near-zero inflation.
            peak = max(np.abs(refRe).max(), np.abs(refIm).max(), 1e-12)
            floor = peak * 1e-2
            err_r = np.abs(gotRe - refRe)
            err_i = np.abs(gotIm - refIm)
            denom_r = np.maximum(np.abs(refRe), floor)
            denom_i = np.maximum(np.abs(refIm), floor)
            rel = max(float(np.max(err_r / denom_r)), float(np.max(err_i / denom_i)))
            worst["max_rel"] = max(worst["max_rel"], float(rel))
            worst["max_abs"] = max(
                worst["max_abs"],
                float(err_r.max()),
                float(err_i.max()),
            )
            worst["nan_inf"] = (
                worst["nan_inf"]
                or not np.isfinite(gotRe).all()
                or not np.isfinite(gotIm).all()
            )
    worst["gate_pass"] = (worst["max_rel"] < 1e-2) and not worst["nan_inf"]
    worst["seeds"] = list(seeds)
    r["correctness"] = worst

    # resource + latency on the largest shape (Task 3).
    # shapes elements are (M, K, N); use the actual K — NOT N as a stand-in
    # (the brief's `N if len(shapes[0]) == 2 else N` was always-N and wrong).
    M, K, N = max(shapes)
    ws = int(mod.real_gemm_workspace_bytes(M, N, K))
    # registers/occupancy: best-effort via nvcc --res-usage compile log (would be
    # captured by build_extension when extra_cuda_cflags includes "--res-usage");
    # None if not parsed. Acceptable per Task 3 spec.
    regs = getattr(mod, "_res_usage_registers", None)
    r["resource"] = {
        "registers": regs,
        "occupancy": None,
        "workspace_bytes": ws,
        # NOTE for sm100_native: workspace_bytes is reported via the 2.x
        # RealGemm helper (always compiled). If the sm100 path ever actually
        # runs this slightly under-reports; not load-bearing for the verdict.
    }

    def _ko_us(fn, *args):
        # Kernel-only: 3 warmups, then median-of-5 cudaEvent timings. Handles,
        # workspace, device buffers are all reused; H2D / construction is outside
        # the timed region (per §7.3 fair kernel-only convention).
        for _ in range(3):
            fn(*args)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(5):
            ev0.record()
            fn(*args)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return float(sorted(ts)[2]) * 1e3  # median us

    ReA = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ImA = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ReB = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ImB = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    four_us = _ko_us(gemm_fn, ReA, ImA, ReB, ImB)
    cA = (ReA.float() + 1j * ImA.float()).to(torch.complex64)
    cB = (ReB.float() + 1j * ImB.float()).to(torch.complex64)
    c64_us = _ko_us(lambda a, b: a @ b, cA, cB)
    r["latency"] = {
        "kernelonly_median_us": four_us,
        "c64_baseline_us": c64_us,
        "ko_ratio_vs_c64": (c64_us / four_us) if four_us > 0 else 0.0,
    }
    return r


def load_grouped_shapes(max_subset: int = 8) -> list[dict]:
    """Distinct real-gemm (min dim>=16) shapes from contraction_shapes.csv.

    Full set may be hundreds; return a representative heterogeneous subset
    (small/medium/large + non-16-aligned). Coverage (subset/total) is recorded
    by run_grouped - no full-coverage claim.
    """
    import csv as _csv

    rows = []
    with open(_CONTRACTION_SHAPES_CSV, newline="") as fh:
        for r in _csv.DictReader(fh):
            try:
                M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
            except (KeyError, ValueError):
                continue
            if min(M, N, K) >= 16:
                rows.append((M, N, K))
    # distinct, then pick a spread subset
    distinct = sorted(set(rows))
    if len(distinct) <= max_subset:
        sub = distinct
    else:
        n = len(distinct)
        idx = sorted(
            set(round(i * (n - 1) / (max_subset - 1)) for i in range(max_subset))
        )
        sub = [distinct[i] for i in idx]
    return [{"M": M, "N": N, "K": K} for (M, N, K) in sub]


def run_grouped(shapes: list[dict], seeds=(0,)) -> dict:
    """Run CUTLASS 2.x GemmGrouped 4M over the heterogeneous `shapes`.

    Returns the `grouped` verdict block: status SUPPORTED (grouped compiled + ran
    over every shape in the subset + correctness gate passes per group) /
    NOT_SUPPORTED (grouped compiled but a real CUTLASS constraint blocks it) /
    BLOCKED (toolchain/build failure). Always records coverage (subset run /
    subset size) — never claims full coverage of all contraction shapes.
    """
    G = len(shapes)
    try:
        mod = build_extension(
            name="cutlass_4m_grouped",
            extra_defines=["-DCUTLASS_ENABLE_GROUPED_4M=1"],
        )
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "kernel_path": "none",
            "compiles": False,
            "runs": False,
            "coverage": {
                "shapes_run": 0,
                "shapes_total": G,
                "note": f"grouped build blocked: {exc}",
            },
            "correctness": {},
            "latency": {},
            "blocker": str(exc),
        }
    if not mod.has_grouped_4m():
        return {
            "status": "NOT_SUPPORTED",
            "kernel_path": "none",
            "compiles": False,
            "runs": False,
            "coverage": {
                "shapes_run": 0,
                "shapes_total": G,
                "note": "HAS_CUTLASS_GROUPED_4M=0 after build "
                "(2.x GemmGrouped path not compiled in)",
            },
            "correctness": {},
            "latency": {},
        }

    import numpy as np
    import torch  # noqa: F401  (CUDA tensors used below)

    # Per-group correctness over all seeds. Same methodology as run_single_4m:
    # BF16-rounded inputs feed both the kernel and the c64 reference so the
    # comparison isolates kernel numerical error.
    worst = {"max_rel": 0.0, "max_abs": 0.0, "nan_inf": False, "groups_checked": 0}
    for sd in seeds:
        ReA_list, ImA_list, ReB_list, ImB_list = [], [], [], []
        refRe_list, refIm_list = [], []
        for s in shapes:
            M, K, N = int(s["M"]), int(s["K"]), int(s["N"])
            rng = np.random.default_rng(sd + M * 31 + N * 7 + K)
            ReA = rng.standard_normal((M, K)).astype(np.float32)
            ImA = rng.standard_normal((M, K)).astype(np.float32)
            ReB = rng.standard_normal((K, N)).astype(np.float32)
            ImB = rng.standard_normal((K, N)).astype(np.float32)
            ReA_bf = _bf16_cuda(ReA)
            ImA_bf = _bf16_cuda(ImA)
            ReB_bf = _bf16_cuda(ReB)
            ImB_bf = _bf16_cuda(ImB)
            refRe, refIm = c64_reference(
                ReA_bf.float().cpu().numpy(),
                ImA_bf.float().cpu().numpy(),
                ReB_bf.float().cpu().numpy(),
                ImB_bf.float().cpu().numpy(),
            )
            ReA_list.append(ReA_bf)
            ImA_list.append(ImA_bf)
            ReB_list.append(ReB_bf)
            ImB_list.append(ImB_bf)
            refRe_list.append(refRe)
            refIm_list.append(refIm)
        try:
            ReC_list, ImC_list = mod.cutlass_grouped_4m(
                ReA_list, ImA_list, ReB_list, ImB_list
            )
        except Exception as exc:
            return {
                "status": "NOT_SUPPORTED",
                "kernel_path": "sm80_grouped",
                "compiles": True,
                "runs": False,
                "coverage": {
                    "shapes_run": 0,
                    "shapes_total": G,
                    "note": f"cutlass_grouped_4m raised: {exc}",
                },
                "correctness": {},
                "latency": {},
                "blocker": str(exc),
            }
        for g in range(G):
            gotRe = ReC_list[g].cpu().numpy()
            gotIm = ImC_list[g].cpu().numpy()
            refRe = refRe_list[g]
            refIm = refIm_list[g]
            peak = max(np.abs(refRe).max(), np.abs(refIm).max(), 1e-12)
            floor = peak * 1e-2
            err_r = np.abs(gotRe - refRe)
            err_i = np.abs(gotIm - refIm)
            denom_r = np.maximum(np.abs(refRe), floor)
            denom_i = np.maximum(np.abs(refIm), floor)
            rel = max(float(np.max(err_r / denom_r)), float(np.max(err_i / denom_i)))
            worst["max_rel"] = max(worst["max_rel"], float(rel))
            worst["max_abs"] = max(
                worst["max_abs"], float(err_r.max()), float(err_i.max())
            )
            worst["nan_inf"] = (
                worst["nan_inf"]
                or not np.isfinite(gotRe).all()
                or not np.isfinite(gotIm).all()
            )
            worst["groups_checked"] += 1
    worst["gate_pass"] = (worst["max_rel"] < 1e-2) and not worst["nan_inf"]
    worst["seeds"] = list(seeds)

    # Kernel-only latency on the subset: 3 warmups, median of 5. Compares the
    # full grouped-4M call (4 passes over all G groups) against the equivalent
    # c64 baseline (one complex64 matmul per group, looped).
    def _ko_us(fn, *args):
        for _ in range(3):
            fn(*args)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(5):
            ev0.record()
            fn(*args)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return float(sorted(ts)[2]) * 1e3  # median us

    # Build one fixed input set for latency (deterministic seed, largest shapes
    # dominate the timing — matches the single-4m convention of timing the
    # largest shape).
    ReA_list, ImA_list, ReB_list, ImB_list = [], [], [], []
    cA_list, cB_list = [], []
    for s in shapes:
        M, K, N = int(s["M"]), int(s["K"]), int(s["N"])
        ReA = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ImA = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ReB = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        ImB = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        ReA_list.append(ReA)
        ImA_list.append(ImA)
        ReB_list.append(ReB)
        ImB_list.append(ImB)
        cA_list.append((ReA.float() + 1j * ImA.float()).to(torch.complex64))
        cB_list.append((ReB.float() + 1j * ImB.float()).to(torch.complex64))

    def _grouped_4m():
        mod.cutlass_grouped_4m(ReA_list, ImA_list, ReB_list, ImB_list)

    def _c64_loop():
        for g in range(G):
            _ = cA_list[g] @ cB_list[g]

    grouped_us = _ko_us(_grouped_4m)
    c64_us = _ko_us(_c64_loop)

    return {
        "status": "SUPPORTED" if worst["gate_pass"] else "NOT_SUPPORTED",
        "kernel_path": "sm80_grouped",
        "compiles": True,
        "runs": True,
        "coverage": {
            "shapes_run": G,
            "shapes_total": G,
            "note": "representative heterogeneous subset of contraction_shapes.csv",
        },
        "correctness": worst,
        "latency": {
            "kernelonly_median_us": grouped_us,
            "c64_baseline_us": c64_us,
            "ko_ratio_vs_c64": (c64_us / grouped_us) if grouped_us > 0 else 0.0,
        },
    }
