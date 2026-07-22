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
    # CUTLASS splits headers: core in <root>/include, util helpers (packed_stride,
    # reference/device/gemm, host_tensor, ...) in <root>/tools/util/include. The
    # Sm100 path uses cutlass::make_cute_packed_stride from the latter.
    cutlass_util_include = os.path.join(cutlass_root, "tools", "util", "include")
    return {
        "cutlass_root": cutlass_root,
        "cutlass_util_include": cutlass_util_include,
        "cuda_home": cuda_home,
        "nvcc": nvcc,
    }


def build_extension(name: str = "cutlass_4m", extra_defines: list[str] | None = None):
    """Compile cpp/cutlass_4m.cu via torch.utils.cpp_extension (ext.cpp build style).

    CUDA_HOME must point at a toolkit whose nvcc targets sm_120 (nvcc_spike env).
    Returns the loadable module. `name` separates the cached sm100 build from
    the sm80 build so a sm100 compile failure never poisons the sm80 cache.
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
    include_paths = [os.path.join(p["cutlass_root"], "include")]
    if os.path.isdir(p["cutlass_util_include"]):
        include_paths.append(p["cutlass_util_include"])
    return load(
        name=name,
        sources=[SRC],
        extra_include_paths=include_paths,
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

    kernel_path in {"sm80_fallback", "sm100_native"}. For "sm100_native" the
    3.x Blackwell Sm100 GEMM is attempted via a separate build (extra define
    CUTLASS_ENABLE_SM100_4M=1); on any failure it transparently falls back to
    the proven 2.x Sm80 path and records `sm100_blocker`. Returns correctness
    plus resource and latency measured on the largest shape (Task 3).
    """
    if kernel_path == "sm100_native":
        return _attempt_sm100_then_sm80(shapes, seeds)
    return _run_sm80(shapes, seeds)


def _attempt_sm100_then_sm80(shapes, seeds) -> dict:
    """Genuine attempt at the 3.x Sm100 4M path; fall back to Sm80 on any failure.

    Builds the kernel under a SEPARATE torch extension name
    (cutlass_4m_sm100) so a compile failure does not poison the cached 2.x
    Sm80 build. Any exception (compile error, link error, runtime failure,
    has_sm100()==False) is caught and recorded verbatim as sm100_blocker.
    """
    try:
        mod = build_extension(
            name="cutlass_4m_sm100",
            extra_defines=["-DCUTLASS_ENABLE_SM100_4M=1"],
        )
        if not hasattr(mod, "has_sm100") or not mod.has_sm100():
            # Compiled with the guard set, but the inner
            # CUTLASS_ARCH_MMA_SM100_SUPPORTED branch did not fire — Sm100
            # path not actually present in this build.
            raise RuntimeError(
                "HAS_CUTLASS_4M_SM100=0 after build "
                "(CUTLASS_ARCH_MMA_SM100_SUPPORTED undefined)"
            )
        return _run_with_module(mod, "sm100_native", shapes, seeds)
    except Exception as exc:
        # Transparent fallback — 3.x Sm100 does not instantiate/run on this
        # toolchain. Record the verbatim error for the artifact.
        return _run_sm80(shapes, seeds, sm100_blocker=str(exc))


def _run_sm80(shapes, seeds, sm100_blocker=None) -> dict:
    """Task 2/3 2.x Sm80 path. Optionally records an sm100_blocker so the
    artifact can explain why a fallback happened (rather than sm80 being the
    requested path)."""
    mod = build_extension()  # default name=cutlass_4m, no extra_defines
    r = _run_with_module(mod, "sm80_fallback", shapes, seeds)
    if sm100_blocker is not None:
        r["sm100_blocker"] = sm100_blocker
    return r


def _run_with_module(mod, kernel_path: str, shapes, seeds) -> dict:
    """Shared correctness + resource + latency runner for both kernel paths.

    Picks mod.cutlass_4m_sm80 (sm80_fallback) or mod.cutlass_4m_sm100
    (sm100_native) based on kernel_path; everything else is identical.
    """
    import numpy as np
    import torch  # noqa: F401  (ensures torch CUDA tensors are usable below)

    assert kernel_path in ("sm80_fallback", "sm100_native")
    gemm_fn = (
        mod.cutlass_4m_sm100 if kernel_path == "sm100_native" else mod.cutlass_4m_sm80
    )
    r = {
        "kernel_path": kernel_path,
        "compiles": True,
        "runs": True,
    }
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
            ReC, ImC = gemm_fn(ReA_bf, ImA_bf, ReB_bf, ImB_bf)
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
    r["correctness"] = worst

    # resource + latency on the largest shape (Task 3).
    # shapes elements are (M, K, N); use the actual K — NOT N as a stand-in
    # (the brief's `N if len(shapes[0]) == 2 else N` was always-N and wrong).
    M, K, N = max(shapes)
    ws = int(mod.real_gemm_workspace_bytes(M, N, K))
    # registers/occupancy: best-effort via nvcc --res-usage compile log (would be
    # captured by build_extension when extra_cuda_cflags includes "--res-usage");
    # None if not parsed. Acceptable per Task 3 spec.
    regs = getattr(mod, "_res_usage_registers", None)
    r["resource"] = {
        "registers": regs,
        "occupancy": None,
        "workspace_bytes": ws,
        # NOTE for sm100_native: workspace_bytes is reported via the 2.x
        # RealGemm helper (always compiled). If the sm100 path ever actually
        # runs this slightly under-reports; not load-bearing for the verdict.
    }

    def _ko_us(fn, *args):
        # Kernel-only: 3 warmups, then median-of-5 cudaEvent timings. Handles,
        # workspace, device buffers are all reused; H2D / construction is outside
        # the timed region (per §7.3 fair kernel-only convention).
        for _ in range(3):
            fn(*args)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(5):
            ev0.record()
            fn(*args)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return float(sorted(ts)[2]) * 1e3  # median us

    ReA = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ImA = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ReB = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ImB = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    four_us = _ko_us(gemm_fn, ReA, ImA, ReB, ImB)
    cA = (ReA.float() + 1j * ImA.float()).to(torch.complex64)
    cB = (ReB.float() + 1j * ImB.float()).to(torch.complex64)
    c64_us = _ko_us(lambda a, b: a @ b, cA, cB)
    r["latency"] = {
        "kernelonly_median_us": four_us,
        "c64_baseline_us": c64_us,
        "ko_ratio_vs_c64": (c64_us / four_us) if four_us > 0 else 0.0,
    }
    return r
