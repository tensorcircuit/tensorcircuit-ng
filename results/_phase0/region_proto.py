"""Region/tile-fusion prototype (rereview §5.2/5.3, Task 3 minimal viable subset).

Proves the 512 MiB C1 anchor producer output C = A@B (c64[4096,16384]) need NOT be
materialized in global memory, via a naive per-element fused kernel compiled with
cupy.RawKernel (nvrtc, sm_120) from cpp/region_proto.cu:

- peak_memory (real shape 4096x16384x1024, no kernel): materialized allocates A+B+C
  (C = 512 MiB); fused allocates A+B only. The delta (~512 MiB) is the avoidable buffer.
- correctness (small shape): fused reduce-in-register == materialized write+reduce ==
  torch A@B reference.

Minimal viable subset (per checkpoint agreement): naive per-element complex GEMM; peak via
raw allocation accounting on the real shape; correctness on a small shape. DEFERRED to the
full Task 3 follow-up: tiled/shared-memory realization, occupancy, pack/recompute/
conversion bytes, and latency vs the c64 baseline.
"""

from __future__ import annotations

import json
import os

import cupy as cp
import numpy as np

OUT_DIR = "results/phase0"
KERNEL_PATH = os.path.join(os.path.dirname(__file__), "cpp", "region_proto.cu")
BLOCK = 256


def _kernel(name: str):
    with open(KERNEL_PATH) as fh:
        code = fh.read()
    return cp.RawKernel(code, name)


def _grid(mn: int):
    return ((mn + BLOCK - 1) // BLOCK,)


def peak_memory(M: int, N: int, K: int) -> dict:
    """Allocation-accounting peak on the real shape (no kernel run)."""
    bytesA = M * K * 8
    bytesB = K * N * 8
    bytesC = M * N * 8
    rt = cp.cuda.runtime
    dev = cp.cuda.Device(0)

    def delta(sizes):
        dev.synchronize()
        f0 = int(rt.memGetInfo()[0])
        ptrs = [rt.malloc(s) for s in sizes]
        dev.synchronize()
        f1 = int(rt.memGetInfo()[0])
        for p in ptrs:
            rt.free(p)
        dev.synchronize()
        return f0 - f1

    mat = delta([bytesA, bytesB, bytesC])
    fused = delta([bytesA, bytesB, 4])
    return {
        "materialized_peak_bytes": mat,
        "fused_peak_bytes": fused,
        "c_buffer_bytes": bytesC,
        "delta_bytes": mat - fused,
    }


def fused_sum(hA, hB, M, N, K) -> float:
    """sum |A@B|^2 with the full C NEVER materialized (reduce in registers)."""
    kr = _kernel("gemm_reduce_kernel")
    dA = cp.asarray(hA, dtype=cp.complex64)
    dB = cp.asarray(hB, dtype=cp.complex64)
    dS = cp.zeros(1, dtype=cp.float32)
    MN = M * N
    kr(
        _grid(MN),
        (BLOCK,),
        (dA, dB, dS, np.int32(M), np.int32(N), np.int32(K), np.int32(MN)),
        shared_mem=BLOCK * 4,
    )
    cp.cuda.Device(0).synchronize()
    return float(dS.get()[0])


def materialized_sum(hA, hB, M, N, K) -> float:
    """sum |A@B|^2 via the full C buffer (write then reduce)."""
    kw = _kernel("gemm_write_kernel")
    kred = _kernel("reduce_sqsum_kernel")
    dA = cp.asarray(hA, dtype=cp.complex64)
    dB = cp.asarray(hB, dtype=cp.complex64)
    dC = cp.empty(M * N, dtype=cp.complex64)
    dS = cp.zeros(1, dtype=cp.float32)
    MN = M * N
    kw(
        _grid(MN),
        (BLOCK,),
        (dA, dB, dC, np.int32(M), np.int32(N), np.int32(K), np.int32(MN)),
    )
    cp.cuda.Device(0).synchronize()
    kred(_grid(MN), (BLOCK,), (dC, dS, np.int32(MN)), shared_mem=BLOCK * 4)
    cp.cuda.Device(0).synchronize()
    return float(dS.get()[0])


def run(
    M: int = 4096,
    N: int = 16384,
    K: int = 1024,
    correctness_shape=(256, 256, 64),
    seed: int = 0,
) -> dict:
    mem = peak_memory(M, N, K)
    cM, cN, cK = correctness_shape
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((cM, cK)) + 1j * rng.standard_normal((cM, cK))).astype(
        np.complex64
    )
    B = (rng.standard_normal((cK, cN)) + 1j * rng.standard_normal((cK, cN))).astype(
        np.complex64
    )
    fs = fused_sum(A, B, cM, cN, cK)
    ms = materialized_sum(A, B, cM, cN, cK)
    ref = float((np.abs(A @ B) ** 2).sum())
    rel_fused = abs(fs - ref) / ref if ref else 0.0
    rel_mat = abs(ms - ref) / ref if ref else 0.0
    memory_feasible = (
        mem["delta_bytes"] > 0
        and mem["fused_peak_bytes"] < mem["materialized_peak_bytes"]
    )
    correct = rel_fused < 1e-3 and rel_mat < 1e-3
    verdict = (
        "TILE_FUSION_MEMORY_FEASIBLE"
        if (memory_feasible and correct)
        else "NOT_FEASIBLE"
    )
    out = {
        "shape": [M, N, K],
        "correctness_shape": list(correctness_shape),
        **mem,
        "fused_sum": fs,
        "materialized_sum": ms,
        "torch_ref_sum": ref,
        "rel_diff_fused_vs_ref": rel_fused,
        "rel_diff_materialized_vs_ref": rel_mat,
        "memory_feasible": memory_feasible,
        "correct": correct,
        "verdict": verdict,
        "basis": "hlo_use_def_anchor_shape",
        "note": (
            "minimal viable subset: naive per-element fused complex-GEMM kernel; peak via raw "
            "allocation accounting on the real shape; correctness on a small shape. Deferred: "
            "tiled/shared-mem realization, occupancy, pack/recompute/conversion bytes, latency vs c64."
        ),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/region_prototype.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
