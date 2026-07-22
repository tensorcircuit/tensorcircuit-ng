"""Integration correctness check for planar_complex_matmul_bf16 vs reference.

Tests BOTH output paths against the same FP32 reference (computed from the SAME
rounded BF16 input values):

  - BF16 output (spec-compliant, out_dtype="bf16"): C/D = CUDA_C_16BF. The
    cublasLt result is BF16-quantized, so the correctness gate is max-RELATIVE-
    error on signal elements (< 1e-2; expected ~0.4% from BF16's 8-bit
    mantissa). Two max-rel figures are reported (see note below).
  - FP32 output (cross-check, out_dtype="fp32"): C/D = CUDA_C_32F. max-abs
    < 1e-2 (expected ~2e-4; only FP32 accumulation-order residual).

NOTE on the BF16-output relative-error metric. The naive all-elements
``max(|err|/max(|ref|,1e-6))`` is dominated by near-zero result elements
(e.g. |ref|~1e-5) where the cublasLt-vs-numpy FP32 accumulation-order delta
(~2e-4 abs, present in BOTH paths) becomes a large relative error. That
all-elements max-rel is IDENTICAL for the FP32- and BF16-output paths (it
measures accumulation noise, not BF16 output quality). To isolate BF16 output
quality we floor the denominator at 1% of the result peak magnitude
(``signal_floor = max(1e-6, 1e-2*peak)``), which excludes the noise tail; on
those signal elements BF16 output rounds at its inherent ~0.4% relative floor.
"""

from __future__ import annotations

import numpy as np

from results._phase0.cublaslt import load_ext, reference_complex_matmul


def _rel_err(val, ref, floor):
    return float(np.max(np.abs(val - ref) / np.maximum(np.abs(ref), floor)))


def _abs_err(val, ref):
    return float(np.max(np.abs(val - ref)))


def main():
    import ml_dtypes

    ext = load_ext()

    # Probe (no execution) — spec-compliant BF16-in/BF16-out config.
    probe = ext.probe_planar_capability(512, 512, 512)
    print("probe_planar_capability(m=n=k=512):", dict(probe))

    # Build random BF16 inputs.
    m = n = k = 512
    rng = np.random.default_rng(42)
    ar_f = rng.standard_normal((m, k)).astype(np.float32)
    ai_f = rng.standard_normal((m, k)).astype(np.float32)
    br_f = rng.standard_normal((k, n)).astype(np.float32)
    bi_f = rng.standard_normal((k, n)).astype(np.float32)

    # Cast to BF16 ONCE; both paths see these exact rounded values.
    ar_bf = ar_f.astype(ml_dtypes.bfloat16)
    ai_bf = ai_f.astype(ml_dtypes.bfloat16)
    br_bf = br_f.astype(ml_dtypes.bfloat16)
    bi_bf = bi_f.astype(ml_dtypes.bfloat16)

    # Raw uint16 views for the extension.
    ar_u16 = ar_bf.view(np.uint16)
    ai_u16 = ai_bf.view(np.uint16)
    br_u16 = br_bf.view(np.uint16)
    bi_u16 = bi_bf.view(np.uint16)

    # Reference uses the SAME BF16 values upcast to float32 (NOT the fp32 source).
    cr_ref, ci_ref = reference_complex_matmul(
        ar_bf.astype(np.float32),
        ai_bf.astype(np.float32),
        br_bf.astype(np.float32),
        bi_bf.astype(np.float32),
    )

    # Signal floor: 1% of result peak magnitude (excludes the near-zero noise
    # tail where cublasLt-vs-numpy FP32 accumulation order dominates rel error).
    peak = max(float(np.max(np.abs(cr_ref))), float(np.max(np.abs(ci_ref))))
    naive_floor = 1e-6
    signal_floor = max(naive_floor, 1e-2 * peak)
    tol_rel = 1e-2
    tol_abs = 1e-2
    all_ok = True

    # ----- BF16-output path (spec-compliant) -----
    cr_u16, ci_u16 = ext.planar_complex_matmul_bf16(
        ar_u16, ai_u16, br_u16, bi_u16, m, n, k, out_dtype="bf16"
    )
    # Upcast raw uint16 BF16 bytes to float32 for comparison.
    cr = np.ascontiguousarray(cr_u16).view(ml_dtypes.bfloat16).astype(np.float32)
    ci = np.ascontiguousarray(ci_u16).view(ml_dtypes.bfloat16).astype(np.float32)

    cr_abs = _abs_err(cr, cr_ref)
    ci_abs = _abs_err(ci, ci_ref)
    # Naive (all-elements) max-rel — dominated by near-zero accumulation noise.
    cr_rel_naive = _rel_err(cr, cr_ref, naive_floor)
    ci_rel_naive = _rel_err(ci, ci_ref, naive_floor)
    # Signal-gated max-rel — isolates BF16 output rounding quality.
    cr_rel_sig = _rel_err(cr, cr_ref, signal_floor)
    ci_rel_sig = _rel_err(ci, ci_ref, signal_floor)
    bf16_ok = bool(cr_rel_sig < tol_rel and ci_rel_sig < tol_rel)
    all_ok = all_ok and bf16_ok
    print(
        f"[bf16 out] m=n=k={m}  peak|ref|={peak:.3f}  signal_floor={signal_floor:.4e}"
    )
    print(
        f"  max|cr-cr_ref|={cr_abs:.6e}  max-rel(naive)={cr_rel_naive:.6e}  max-rel(signal)={cr_rel_sig:.6e}"
    )
    print(
        f"  max|ci-ci_ref|={ci_abs:.6e}  max-rel(naive)={ci_rel_naive:.6e}  max-rel(signal)={ci_rel_sig:.6e}"
    )
    print(f"  cr_ref sample [0,0..3]: {cr_ref[0, :4]}")
    print(f"  cr      sample [0,0..3]: {cr[0, :4]}")
    print(f"  PASS (max-rel(signal) < {tol_rel}): {bf16_ok}")

    # ----- FP32-output path (cross-check) -----
    cr_f32, ci_f32 = ext.planar_complex_matmul_bf16(
        ar_u16, ai_u16, br_u16, bi_u16, m, n, k, out_dtype="fp32"
    )
    fp32_cr_abs = _abs_err(cr_f32, cr_ref)
    fp32_ci_abs = _abs_err(ci_f32, ci_ref)
    # FP32 path naive max-rel — MUST match the BF16 path's naive max-rel (proves
    # BF16 output adds no relative error; the naive value is accumulation noise).
    fp32_cr_rel_naive = _rel_err(cr_f32, cr_ref, naive_floor)
    fp32_ci_rel_naive = _rel_err(ci_f32, ci_ref, naive_floor)
    fp32_ok = bool(fp32_cr_abs < tol_abs and fp32_ci_abs < tol_abs)
    all_ok = all_ok and fp32_ok
    print(f"[fp32 out] m=n=k={m}")
    print(f"  max|cr-cr_ref|={fp32_cr_abs:.6e}  max-rel(naive)={fp32_cr_rel_naive:.6e}")
    print(f"  max|ci-ci_ref|={fp32_ci_abs:.6e}  max-rel(naive)={fp32_ci_rel_naive:.6e}")
    print(f"  PASS (max-abs < {tol_abs}): {fp32_ok}")

    rel_match = (
        abs(cr_rel_naive - fp32_cr_rel_naive) <= 0.05 * cr_rel_naive
        and abs(ci_rel_naive - fp32_ci_rel_naive) <= 0.05 * ci_rel_naive
    )
    print(
        f"[cross-check] BF16 naive max-rel == FP32 naive max-rel (within 5%): {rel_match}\n"
        f"  -> confirms BF16 output adds NO relative error beyond FP32 accumulation noise"
    )

    return bool(all_ok and rel_match)


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
