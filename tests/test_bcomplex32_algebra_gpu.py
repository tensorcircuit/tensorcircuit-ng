"""GPU smoke tests for complex<bf16> pair-algebra across tc GPU backends.
Skipped unless the backend + a CUDA GPU are available. CI runs numpy only; these
run author-side on the supercomputer. Mirrors the numpy bf16 tests."""
import numpy as np
import pytest

import tensorcircuit as tc
from applications.bcomplex32_algebra import bcomplex32


def _gpu_present(name: str) -> bool:
    """True if backend `name` has a CUDA GPU available."""
    try:
        if name == "jax":
            import jax

            return any(d.platform == "gpu" for d in jax.devices())
        if name == "pytorch":
            import torch

            return bool(torch.cuda.is_available())
        if name == "tensorflow":
            import tensorflow as tf

            return bool(tf.config.list_physical_devices("GPU"))
        if name == "cupy":
            import cupy

            return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
    return False


def _backend_available(name: str) -> bool:
    try:
        if not _gpu_present(name):
            return False
        tc.set_backend(name)
        be = tc.backend
        t = be.cast(be.convert_to_tensor(np.array([1.0 + 2.0j], dtype=np.complex64)), "complex64")
        from applications.bcomplex32_algebra import _complex_to_pair, _pair_to_complex

        back = _pair_to_complex(be, _complex_to_pair(be, t))
        _ = be.numpy(back)
        return True
    except Exception:
        return False


GPU_BACKENDS = ["jax", "pytorch", "tensorflow", "cupy"]


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    tc.set_backend("numpy")


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_bf16_end_to_end_matches_complex64_gpu(backend):
    if not _backend_available(backend):
        pytest.skip(f"backend {backend} or GPU unavailable")
    tc.set_backend(backend)

    def build():
        c = tc.Circuit(4)
        c.H(0)
        for i in range(3):
            c.cnot(i, i + 1)
        return np.asarray(c.state())

    ref = build()
    with bcomplex32():
        got = build()
    np.testing.assert_allclose(got, ref, rtol=2e-2)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_pair_einsum_keeps_bfloat16_dtype_gpu(backend):
    if not _backend_available(backend):
        pytest.skip(f"backend {backend} or GPU unavailable")
    tc.set_backend(backend)
    be = tc.backend
    from applications.bcomplex32_algebra import _complex_to_pair, _pair_einsum

    a = np.array([[1.0 + 2.0j, 3.0j], [-1.0j, 2.0 - 1.0j]], dtype=np.complex64)
    b = np.array([[0.5 + 0.5j, 1.0j], [2.0j, -1.0 + 1.0j]], dtype=np.complex64)
    a = be.cast(be.convert_to_tensor(a), "complex64")
    b = be.cast(be.convert_to_tensor(b), "complex64")
    result = _pair_einsum(be, "ij,jk->ik", _complex_to_pair(be, a), _complex_to_pair(be, b))
    re, _ = result.unpack()
    assert str(be.dtype(re)).endswith("bfloat16"), \
        f"_pair_einsum upcast on {backend}: {be.dtype(re)}"
