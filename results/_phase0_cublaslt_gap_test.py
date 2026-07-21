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


def test_has_complex_bf16_dtype_returns_bool():
    assert isinstance(has_complex_bf16_dtype("jax"), bool)
    assert isinstance(has_complex_bf16_dtype("pytorch"), bool)


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
