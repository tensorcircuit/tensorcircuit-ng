import numpy as np
import pytest


def test_compute_metrics_basic():
    from results._phase0.numerical import compute_metrics

    ref = np.ones((4, 4), dtype=np.complex64)
    out = (1 + 1e-3 * np.ones((4, 4))).astype(np.complex64)
    m = compute_metrics(out, ref)
    assert m["nan_inf"] is False
    assert m["n_elems"] == 16
    assert m["max_abs"] == pytest.approx(1e-3, rel=0.02)        # |out-ref| = 1e-3
    assert m["max_rel"] == pytest.approx(1e-3, rel=0.02)        # denom=max(|ref|,0.5)=1.0
    assert m["relative_l2"] == pytest.approx(1e-3, rel=0.02)    # ||diff||/max(1,||ref||)


def test_compute_metrics_detects_nan():
    from results._phase0.numerical import compute_metrics

    ref = np.ones((2, 2), dtype=np.complex64)
    out = np.array([[1, np.nan], [1, 1]], dtype=np.complex64)
    m = compute_metrics(out, ref)
    assert m["nan_inf"] is True


def test_make_inputs_baseline_stats():
    from results._phase0.numerical import make_inputs

    A, B = make_inputs("baseline", (1024, 1024, 64), seed=0)  # (M,N,K)
    assert A.shape == (1024, 64) and B.shape == (64, 1024)    # A=(M,K), B=(K,N)
    assert A.dtype == np.complex64
    # real & imag ~ N(0,1): mean ~0, std ~1
    assert abs(A.real.mean()) < 0.1 and abs(A.real.std() - 1.0) < 0.1


def test_make_inputs_mixed_scale_dynamic_range():
    from results._phase0.numerical import make_inputs

    A, _ = make_inputs("mixed_scale", (512, 32, 512), seed=1)
    mag = np.abs(A)
    # bimodal: some elements ~1e2, some ~1e-2 -> dynamic range ~1e4
    assert mag.max() > 50 and mag.min() < 0.1
    assert (mag > 50).sum() > 0 and (mag < 0.1).sum() > 0


def test_make_inputs_cancellation_paired_rows():
    from results._phase0.numerical import make_inputs

    _, B = make_inputs("cancellation", (64, 64, 64), seed=2)
    K = 64
    # B[2j+1] == -B[2j] for paired rows (cancellation structure, spec §4.3)
    assert np.allclose(B[1], -B[0])
    assert np.allclose(B[K - 1], -B[K - 2])


def test_make_inputs_deterministic_in_seed():
    from results._phase0.numerical import make_inputs

    a1, _ = make_inputs("baseline", (32, 8, 32), seed=5)
    a2, _ = make_inputs("baseline", (32, 8, 32), seed=5)
    assert np.array_equal(a1, a2)
