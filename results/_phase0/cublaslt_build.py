"""Build/load the _phase0_cublaslt_ext pybind11 module via torch.utils.cpp_extension.
Uses the user-installed g++ + bundled cublasLt.h / cuda_runtime.h / libcublasLt.so.12.
"""

from __future__ import annotations

import glob
import os
import sys

SP = os.path.join(sys.prefix, "lib", "python3.10", "site-packages")
CUBLAS_INC = os.path.join(SP, "nvidia", "cublas", "include")
CUBLAS_LIB = os.path.join(SP, "nvidia", "cublas", "lib")
CUDA_INC = os.path.join(SP, "nvidia", "cuda_runtime", "include")
# driver_types.h (via cublasLt.h -> cublas_api.h) does #include "crt/host_defines.h";
# the cuda_runtime wheel ships host_defines.h flat (no crt/ subdir), so point at the
# cuda_nvcc wheel whose include/ has the crt/ tree.
CUDA_NVCC_INC = os.path.join(SP, "nvidia", "cuda_nvcc", "include")
# cuda_fp16.h / cuda_bf16.h (also pulled by cublas_api.h) do #include <nv/target>;
# no nvidia wheel here ships it (nvidia-cuda-cccl-cu12 not installed). Fall back to the
# canonical libcu++/CCCL headers vendored by cupy (NVIDIA, Apache-2.0 + LLVM exception).
CCCL_INC = os.path.join(SP, "cupy", "_core", "include", "cupy", "_cccl", "libcudacxx")
EXT_DIR = os.path.join(os.path.dirname(__file__), "cpp")


def load_ext():
    import torch
    from torch.utils.cpp_extension import load

    src = os.path.join(EXT_DIR, "ext.cpp")
    return load(
        name="_phase0_cublaslt_ext",
        sources=[src],
        extra_include_paths=[CUBLAS_INC, CUDA_INC, CUDA_NVCC_INC, CCCL_INC],
        # the cublas wheel ships only the versioned soname (libcublasLt.so.12, no
        # libcublasLt.so symlink), so plain -lcublasLt misses; -l: matches the file.
        extra_ldflags=[
            f"-L{CUBLAS_LIB}",
            "-l:libcublasLt.so.12",
            "-Wl,-rpath," + CUBLAS_LIB,
        ],
        verbose=False,
    )


if __name__ == "__main__":
    ext = load_ext()
    print("smoke_add(2,3) =", ext.smoke_add(2, 3))
    print("cublaslt_info:", ext.cublaslt_info())
