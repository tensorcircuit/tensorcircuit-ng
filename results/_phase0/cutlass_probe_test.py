import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def test_discover_paths_uses_env_vars(monkeypatch):
    import cutlass_probe

    monkeypatch.setenv("CUTLASS_ROOT", "/fake/cutlass")
    monkeypatch.setenv("CUDA_HOME", "/fake/cuda")
    monkeypatch.setenv("NVCC", "/fake/cuda/bin/nvcc")
    p = cutlass_probe.discover_paths()
    assert p["cutlass_root"] == "/fake/cutlass"
    assert p["cuda_home"] == "/fake/cuda"
    assert p["nvcc"] == "/fake/cuda/bin/nvcc"
    assert (
        os.path.isdir(p["cutlass_root"]) is False
    )  # not validated here; build validates


def test_build_extension_signature_exists():
    import cutlass_probe

    assert callable(cutlass_probe.build_extension)


def _gpu_ready():
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import cutlass_probe

        p = cutlass_probe.discover_paths()
        return os.path.isdir(p["cutlass_root"]) and os.path.exists(p["nvcc"])
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_build_extension_compiles_and_loads():
    import cutlass_probe

    mod = cutlass_probe.build_extension()
    assert mod.probe() == 42


def test_four_m_coefficients():
    import cutlass_probe

    c = cutlass_probe.four_m_coefficients()
    # ReC = +1*ReA.ReB + (-1)*ImA.ImB ; ImC = +1*ReA.ImB + +1*ImA.ReB
    assert c["rec_rea_reb"] == +1.0 and c["rec_ima_imb"] == -1.0
    assert c["imc_rea_imb"] == +1.0 and c["imc_ima_reb"] == +1.0


def test_c64_reference_matches_numpy_complex():
    import cutlass_probe
    import numpy as np

    rng = np.random.default_rng(0)
    ReA = rng.standard_normal((4, 8)).astype(np.float32)
    ImA = rng.standard_normal((4, 8)).astype(np.float32)
    ReB = rng.standard_normal((8, 6)).astype(np.float32)
    ImB = rng.standard_normal((8, 6)).astype(np.float32)
    ReC, ImC = cutlass_probe.c64_reference(ReA, ImA, ReB, ImB)
    A = (ReA + 1j * ImA).astype(np.complex64)
    B = (ReB + 1j * ImB).astype(np.complex64)
    C = A @ B
    np.testing.assert_allclose(ReC, C.real, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ImC, C.imag, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _gpu_ready(), reason="needs GPU + nvcc_spike + CUTLASS_ROOT")
def test_single_4m_sm80_correctness_real_gemm():
    import cutlass_probe

    r = cutlass_probe.run_single_4m(
        "sm80_fallback", shapes=[(128, 128, 128)], seeds=(0,)
    )
    assert r["correctness"]["gate_pass"] is True, r["correctness"]
    assert r["correctness"]["max_rel"] < 1e-2
