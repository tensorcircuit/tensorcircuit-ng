import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a CUDA GPU (WSL + cublasLt/cutlass ext)"
    )
