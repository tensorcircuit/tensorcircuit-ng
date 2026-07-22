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
