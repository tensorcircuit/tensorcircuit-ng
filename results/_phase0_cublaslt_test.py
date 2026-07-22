"""Tests for Phase 0 Plan B Task 2: planar-complex BF16 cublasLt matmul + judge."""

import numpy as np


def test_reference_complex_matmul_matches_numpy():
    from results._phase0_cublaslt import reference_complex_matmul

    m = k = n = 32
    rng = np.random.default_rng(0)
    ar = rng.standard_normal((m, k)).astype(np.float32)
    ai = rng.standard_normal((m, k)).astype(np.float32)
    br = rng.standard_normal((k, n)).astype(np.float32)
    bi = rng.standard_normal((k, n)).astype(np.float32)
    cr, ci = reference_complex_matmul(ar, ai, br, bi)
    A = (ar + 1j * ai).astype(np.complex64)
    B = (br + 1j * bi).astype(np.complex64)
    ref = A @ B
    assert np.allclose(cr, ref.real, atol=1e-3, rtol=1e-3)
    assert np.allclose(ci, ref.imag, atol=1e-3, rtol=1e-3)


def test_judge_capability_supported_when_all_pass():
    from results._phase0_cublaslt import judge_capability

    j = judge_capability(
        max_rel_err=1e-3,
        perf_ratio_vs_c64=1.5,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
    )
    assert j["status"] == "SUPPORTED", j


def test_judge_capability_not_supported_when_slow():
    from results._phase0_cublaslt import judge_capability

    j = judge_capability(
        max_rel_err=1e-3,
        perf_ratio_vs_c64=0.9,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
    )
    assert j["status"] == "NOT_SUPPORTED"
    assert "1.3" in j["reason"] or "speed" in j["reason"].lower()


def test_judge_capability_not_supported_when_no_algo():
    from results._phase0_cublaslt import judge_capability

    j = judge_capability(
        max_rel_err=1e-3,
        perf_ratio_vs_c64=2.0,
        algo_count=0,
        workspace_bytes=0,
        output_bytes=1 << 24,
        has_four_real_temps=False,
    )
    assert j["status"] == "NOT_SUPPORTED"


def test_judge_capability_accuracy_gate_is_max_rel():
    """BF16 output has ~0.4% relative error: a passing rel error (4e-3) must
    NOT be flagged, while a failing rel error (2e-2) must — even though the
    absolute error would look large in BF16-magnitude terms."""
    from results._phase0_cublaslt import judge_capability

    # 0.4% rel error, large abs (BF16-output tail) -> SUPPORTED.
    j_ok = judge_capability(
        max_rel_err=4e-3,
        perf_ratio_vs_c64=1.5,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
        max_abs_err=0.5,
    )
    assert j_ok["status"] == "SUPPORTED", j_ok

    # 2% rel error -> accuracy fail.
    j_bad = judge_capability(
        max_rel_err=2e-2,
        perf_ratio_vs_c64=1.5,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
        max_abs_err=0.5,
    )
    assert j_bad["status"] == "NOT_SUPPORTED"
    assert "max_rel_err" in j_bad["reason"]


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
