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


def four_m_coefficients() -> dict:
    """Signs for the 4-real-GEMM complex decomposition (C = A·B)."""
    return {
        "rec_rea_reb": +1.0,
        "rec_ima_imb": -1.0,
        "imc_rea_imb": +1.0,
        "imc_ima_reb": +1.0,
    }


def c64_reference(ReA, ImA, ReB, ImB):
    """Complex64 reference product via numpy (CPU). Returns (ReC, ImC) float32."""
    import numpy as np

    A = (ReA.astype(np.float32) + 1j * ImA.astype(np.float32)).astype(np.complex64)
    B = (ReB.astype(np.float32) + 1j * ImB.astype(np.float32)).astype(np.complex64)
    C = A @ B
    return C.real.astype(np.float32), C.imag.astype(np.float32)


def _bf16_cuda(t):
    import torch

    return torch.as_tensor(t, device="cuda", dtype=torch.bfloat16)


def run_single_4m(kernel_path: str, shapes, seeds=(0, 1, 2)) -> dict:
    """Run CUTLASS single 4M GEMM on `shapes` x `seeds`, compare to c64 reference.

    kernel_path in {"sm80_fallback", "sm100_native"}. Returns correctness fields.
    """
    import numpy as np
    import torch  # noqa: F401  (ensures torch CUDA tensors are usable below)

    assert kernel_path == "sm80_fallback"  # Task 2: only sm80; Task 4 adds sm100
    mod = build_extension()
    worst = {"max_rel": 0.0, "max_abs": 0.0, "nan_inf": False}
    for M, K, N in shapes:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            ReA = rng.standard_normal((M, K)).astype(np.float32)
            ImA = rng.standard_normal((M, K)).astype(np.float32)
            ReB = rng.standard_normal((K, N)).astype(np.float32)
            ImB = rng.standard_normal((K, N)).astype(np.float32)
            # BF16 CUDA tensors feed the kernel; their BF16-rounded values also
            # feed the reference (apples-to-apples on the same rounded inputs,
            # matching Task 6 cublaslt's reference_complex_matmul convention) so
            # the comparison isolates kernel numerical error rather than BF16
            # input quantization.
            ReA_bf = _bf16_cuda(ReA)
            ImA_bf = _bf16_cuda(ImA)
            ReB_bf = _bf16_cuda(ReB)
            ImB_bf = _bf16_cuda(ImB)
            refRe, refIm = c64_reference(
                ReA_bf.float().cpu().numpy(),
                ImA_bf.float().cpu().numpy(),
                ReB_bf.float().cpu().numpy(),
                ImB_bf.float().cpu().numpy(),
            )
            ReC, ImC = mod.cutlass_4m_sm80(ReA_bf, ImA_bf, ReB_bf, ImB_bf)
            gotRe = ReC.cpu().numpy()
            gotIm = ImC.cpu().numpy()
            # signal-floored rel-err (per §7, matching Task 6 cublaslt convention):
            # per-element denom = max(|ref|, 1% of peak); floor stops near-zero inflation.
            peak = max(np.abs(refRe).max(), np.abs(refIm).max(), 1e-12)
            floor = peak * 1e-2
            err_r = np.abs(gotRe - refRe)
            err_i = np.abs(gotIm - refIm)
            denom_r = np.maximum(np.abs(refRe), floor)
            denom_i = np.maximum(np.abs(refIm), floor)
            rel = max(float(np.max(err_r / denom_r)), float(np.max(err_i / denom_i)))
            worst["max_rel"] = max(worst["max_rel"], float(rel))
            worst["max_abs"] = max(
                worst["max_abs"],
                float(err_r.max()),
                float(err_i.max()),
            )
            worst["nan_inf"] = (
                worst["nan_inf"]
                or not np.isfinite(gotRe).all()
                or not np.isfinite(gotIm).all()
            )
    worst["gate_pass"] = (worst["max_rel"] < 1e-2) and not worst["nan_inf"]
    worst["seeds"] = list(seeds)
    return {
        "kernel_path": kernel_path,
        "compiles": True,
        "runs": True,
        "correctness": worst,
    }
