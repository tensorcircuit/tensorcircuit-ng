"""Task 8 CUTLASS/CuTe SM120 4M probe driver (final-remediation §11).

Replaces the PlanB-T4 compile-only smoke. Compiles CUTLASS kernels in-tree
via torch.utils.cpp_extension with an isolated nvcc_spike CUDA_HOME, runs
them on sm_120, and aggregates a cutlass-sm120-4m-v1 capability verdict.
"""

from __future__ import annotations

import glob
import os

SCHEMA_VERSION = "cutlass-sm120-4m-v1"
_HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(_HERE, "cpp")
SRC = os.path.join(CPP_DIR, "cutlass_4m.cu")


def discover_paths() -> dict:
    """Discover CUTLASS_ROOT, CUDA_HOME, NVCC from env (no hardcoded /home paths)."""
    home = os.path.expanduser("~")
    cutlass_root = os.environ.get("CUTLASS_ROOT", os.path.join(home, "cutlass_spike"))
    cuda_home = os.environ.get(
        "CUDA_HOME", os.path.join(home, "miniconda3", "envs", "nvcc_spike")
    )
    nvcc = os.environ.get("NVCC", "")
    if not nvcc:
        cands = [os.path.join(cuda_home, "bin", "nvcc")]
        nvcc = next((c for c in cands if os.path.exists(c)), "")
    return {"cutlass_root": cutlass_root, "cuda_home": cuda_home, "nvcc": nvcc}


def build_extension(name: str = "cutlass_4m", extra_defines: list[str] | None = None):
    """Compile cpp/cutlass_4m.cu via torch.utils.cpp_extension (ext.cpp build style).

    CUDA_HOME must point at a toolkit whose nvcc targets sm_120 (nvcc_spike env).
    Returns the loadable module.
    """
    import torch  # noqa: F401  (ensures torch + its bundled cuda runtime present)
    from torch.utils.cpp_extension import load

    p = discover_paths()
    os.environ.setdefault("CUDA_HOME", p["cuda_home"])
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ["PATH"] = (
        os.path.join(p["cuda_home"], "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    cflags = ["-std=c++17", "-O2"]
    if extra_defines:
        cflags += extra_defines
    return load(
        name=name,
        sources=[SRC],
        extra_include_paths=[os.path.join(p["cutlass_root"], "include")],
        extra_cuda_cflags=cflags,
        verbose=False,
    )
