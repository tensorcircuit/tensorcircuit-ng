"""CUTLASS SM120 compile-level probe (review §8). Uses the bundled nvidia-cuda-nvcc wheel.

Deviation from the task brief (documented): the installed
``nvidia-cuda-nvcc-cu12`` 12.9.86 wheel ships only ``ptxas`` + ``nvvm`` +
headers — it does NOT ship the ``nvcc`` driver binary (verified: the
``nvidia/cuda_nvcc/bin/`` directory contains only ``ptxas``). With no ``nvcc``
on PATH either, the subprocess path cannot run. The probe therefore falls
back to NVRTC (``cuda.bindings.nvrtc``), which shares nvcc's frontend
compiler (same ``cicc``), compiles the same ``.cu`` source in-memory for
``-arch=compute_120``, and answers the same §8 question: does the CUDA 12.x
frontend accept BF16 wmma Tensor Core intrinsics for compute capability
12.0? NVRTC reports supported archs via ``nvrtcGetSupportedArchs``. If a
real ``nvcc`` is present it is used directly (brief path).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

SP = os.path.join(sys.prefix, "lib", "python3.10", "site-packages")
NVCC = glob.glob(os.path.join(SP, "nvidia", "cuda_nvcc", "**", "nvcc"), recursive=True)
# cuda_runtime.h lives under cuda_runtime/include; crt/mma.h (pulled in by
# <mma.h>) lives under cuda_nvcc/include, so both include dirs are needed.
CUDA_INC = os.path.join(SP, "nvidia", "cuda_runtime", "include")
NVRTC_INC = os.path.join(SP, "nvidia", "cuda_nvcc", "include")
SRC = os.path.join(
    os.path.dirname(__file__), "_phase0_cublaslt", "minimal_cutlass_sm120.cu"
)

TARGET_ARCH = 120  # compute_120 / sm_120 (Blackwell)


def _read_source() -> str:
    with open(SRC, "r", encoding="utf-8") as fh:
        return fh.read()


def _probe_nvcc() -> dict:
    nvcc = NVCC[0]
    cmd = [
        nvcc,
        "-arch=sm_120",
        "-std=c++17",
        f"-I{CUDA_INC}",
        f"-I{NVRTC_INC}",
        SRC,
        "-o",
        "/tmp/probe_sm120",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ok = p.returncode == 0
    arch_ok = ok  # -arch=sm_120 was accepted iff the compile succeeded
    wmma_ok = ok  # source is wmma BF16; a clean compile means wmma bf16 was accepted
    return {
        "compile_path": "nvcc",
        "nvcc": nvcc,
        "cmd": cmd,
        "returncode": p.returncode,
        "status": "COMPILES" if ok else "COMPILE_FAIL",
        "arch_sm120_ok": arch_ok,
        "wmma_bf16_ok": wmma_ok,
        "stderr_tail": p.stderr[-400:],
    }


def _probe_nvrtc() -> dict:
    from cuda.bindings import nvrtc  # type: ignore

    v_err, major, minor = nvrtc.nvrtcVersion()
    a_res = nvrtc.nvrtcGetSupportedArchs()
    supported = list(a_res[1]) if isinstance(a_res, tuple) else list(a_res)
    src = _read_source()
    inc1 = CUDA_INC.encode()
    inc2 = NVRTC_INC.encode()
    opts = [
        b"-std=c++17",
        b"-arch=compute_120",
        b"-default-device",
        b"-I" + inc1,
        b"-I" + inc2,
    ]
    c_err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"probe.cu", 0, [], [])
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    code = res[0] if isinstance(res, tuple) else res
    _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    buf = bytearray(int(log_size))
    nvrtc.nvrtcGetProgramLog(prog, buf)
    log = bytes(buf).decode(errors="replace")
    ok = int(code) == 0
    arch_ok = (TARGET_ARCH in supported) and ok
    wmma_ok = ok  # source is wmma BF16; a clean compile means wmma bf16 was accepted
    return {
        "compile_path": "nvrtc-fallback",
        "nvrtc_version": f"{int(major)}.{int(minor)}",
        "supported_archs": supported,
        "compute_120_supported": TARGET_ARCH in supported,
        "opts": [o.decode(errors="replace") for o in opts],
        "returncode": int(code),
        "status": "COMPILES" if ok else "COMPILE_FAIL",
        "arch_sm120_ok": arch_ok,
        "wmma_bf16_ok": wmma_ok,
        "stderr_tail": log[-400:],
    }


def probe_cutlass_sm120() -> dict:
    """Compile ``minimal_cutlass_sm120.cu`` for sm_120 / compute_120.

    Prefers the wheel ``nvcc``; falls back to NVRTC (same frontend) when the
    wheel ships no ``nvcc`` binary. Reports build status, arch acceptance and
    whether the BF16 wmma intrinsics compiled.
    """
    if NVCC:
        try:
            return _probe_nvcc()
        except Exception as exc:  # pragma: no cover - environmental guard
            return {
                "compile_path": "nvcc",
                "status": "PROBE_ERROR",
                "detail": f"nvcc present but probe errored: {exc!r}",
            }
    # Wheel ships no nvcc binary (nvidia-cuda-nvcc-cu12 12.9.86 = ptxas+nvvm only).
    # NVRTC shares nvcc's frontend compiler, so it answers the same §8 question.
    try:
        return _probe_nvrtc()
    except Exception as exc:
        return {
            "compile_path": "nvrtc-fallback",
            "status": "PROBE_ERROR",
            "detail": ("wheel ships no nvcc and NVRTC fallback failed: " f"{exc!r}"),
        }


if __name__ == "__main__":
    r = probe_cutlass_sm120()
    print(json.dumps(r, indent=2))
    os.makedirs("results/phase0", exist_ok=True)
    with open("results/phase0/cutlass_sm120_capability.md", "w", encoding="utf-8") as f:
        f.write(
            "# CUTLASS SM120 compile probe (review §8)\n\n"
            "Compile-level probe: does the CUDA frontend accept BF16 wmma "
            "Tensor Core intrinsics for compute capability 12.0?\n\n"
            "```\n" + json.dumps(r, indent=2) + "\n```\n"
        )
