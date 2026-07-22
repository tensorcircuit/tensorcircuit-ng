"""Phase 0 Plan B driver: planar-complex BF16 cublasLt probe (review §7)."""

from __future__ import annotations

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
    max_abs_err,
    perf_ratio_vs_c64,
    algo_count,
    workspace_bytes,
    output_bytes,
    has_four_real_temps,
):
    """§7.5 capability judgment for the planar-complex BF16 planar probe."""
    reasons = []
    if algo_count == 0:
        return {
            "status": "NOT_SUPPORTED",
            "reason": "SM120 returned no algorithm for planar C16BF",
        }
    if max_abs_err > 1e-2:
        reasons.append(f"accuracy fail (max_abs_err={max_abs_err:.2e})")
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


if __name__ == "__main__":
    ext = load_ext()
    print("ext:", ext)
    print("cublaslt_info:", ext.cublaslt_info())
