"""Unit tests for Probe 1-deferred pure logic. Run: pytest results/_phase0_cublaslt_gap_test.py -v"""

from results._phase0_cublaslt_gap import tflops, has_complex_bf16_dtype


def test_tflops_standard():
    # 2 * M*N*K FLOPs，M=N=K=4096 → 2*4096^3 FLOPs
    assert (
        abs(tflops(m=4096, k=4096, n=4096, seconds=0.01) - (2 * 4096**3 / 1e12 / 0.01))
        < 1e-6
    )


def test_tflops_zero_seconds_safe():
    assert tflops(4096, 4096, 4096, 0.0) == 0.0  # 防除零


def test_has_complex_bf16_dtype_absent_with_evidence():
    from results._phase0_cublaslt_gap import has_complex_bf16_dtype

    for be in ("jax", "pytorch"):
        r = has_complex_bf16_dtype(be)
        assert r["present"] is False, f"{be}: {r}"
        assert r["evidence"]  # non-empty reason


def test_pair_complex_matmul_hlo_has_four_real_dots():
    from results._phase0_cublaslt_gap import pair_complex_matmul_hlo

    r = pair_complex_matmul_hlo(m=64)
    # a complex matmul via the 4-M pair path lowers to 4 real dot_general ops
    assert r["dot_count"] >= 4, r


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
