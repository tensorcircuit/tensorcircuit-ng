"""Phase 0 Task 9: numerical validation matrix (final-remediation Task 9).

Aggregates the numerical correctness of all four BF16 contraction routes
(planar / grouped / region_fused / cutlass_4m_single) over actual-large shapes,
>=3 seeds, 3 adversarial dynamic-range levels, {C16BF, C32F} output dtypes, against
a c64 fp32 materialized reference. Produces a fail-closed numerical_validation.csv +
.json consumed by Task 10 (gonogo) and Task 11 (manifest).

Pure functions (compute_metrics, make_inputs, apply_policy, aggregate, writers) are
GPU-free and unit-tested first. GPU route collectors import existing helpers from
cublaslt.py / region_proto.py (zero changes to those modules).
"""

from __future__ import annotations

import numpy as np


def compute_metrics(out, ref, signal_floor: float = 0.5) -> dict:
    """Numerical correctness of ``out`` vs c64 fp32 materialized ``ref``.

    Returns JSON-serializable scalars:
    - relative_l2: ||out-ref||_2 / max(1, ||ref||_2)
    - max_abs:     max |out-ref|
    - max_rel:     max |out-ref| / max(|ref|, signal_floor)   (signal_floor avoids div-by-0)
    - nan_inf:     any non-finite in out
    - n_elems:     out.size
    """
    out = np.asarray(out)
    ref = np.asarray(ref)
    diff = out - ref
    nan_inf = bool(not np.all(np.isfinite(out)))
    denom = np.maximum(np.abs(ref), signal_floor)
    rel_l2 = float(np.linalg.norm(diff) / max(1.0, float(np.linalg.norm(ref))))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    max_rel = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
    return {
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nan_inf": nan_inf,
        "n_elems": int(out.size),
    }


_LEVELS = ("baseline", "mixed_scale", "cancellation")


def make_inputs(level, shape, seed, ref_dtype=np.complex64):
    """Generate (A, B) for C = A @ B at a given dynamic-range level.

    shape = (M, N, K) (matches cublaslt artifacts); A is (M,K), B is (K,N). Deterministic in seed.
    - baseline: real/imag ~ N(0,1)
    - mixed_scale: per-element Bernoulli(0.5) mix of N(0, 1e2^2) and N(0, 1e-2^2)
      -> dynamic range 1e4, exposes bf16 small-magnitude loss (spec §4.2)
    - cancellation: B rows paired +- (B[2j+1] = -B[2j]); reference C has near-zero
      elements -> amplifies max_rel denominator sensitivity (spec §4.3). Requires K even.
    """
    if level not in _LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {_LEVELS}")
    M, N, K = shape
    rng = np.random.default_rng(seed)

    def complex_normal(sz, sigma):
        return (rng.standard_normal(sz) + 1j * rng.standard_normal(sz)).astype(
            ref_dtype
        ) * sigma

    if level == "baseline":
        A = complex_normal((M, K), 1.0)
        B = complex_normal((K, N), 1.0)
    elif level == "mixed_scale":
        mask_a = rng.random((M, K)) < 0.5
        big_a = complex_normal((M, K), 1e2)
        small_a = complex_normal((M, K), 1e-2)
        A = np.where(mask_a, big_a, small_a).astype(ref_dtype)
        mask_b = rng.random((K, N)) < 0.5
        big_b = complex_normal((K, N), 1e2)
        small_b = complex_normal((K, N), 1e-2)
        B = np.where(mask_b, big_b, small_b).astype(ref_dtype)
    else:  # cancellation
        if K % 2 != 0:
            raise ValueError(f"cancellation requires even K, got K={K}")
        A = complex_normal((M, K), 1.0)
        half = complex_normal((K // 2, N), 1.0)
        B = np.empty((K, N), dtype=ref_dtype)
        B[0::2] = half
        B[1::2] = -half
    return A, B
