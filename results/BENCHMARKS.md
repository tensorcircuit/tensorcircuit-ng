# bf16 GPU benchmark results — RTX 5070 Ti (Blackwell sm_120)

**Author-run** on RTX 5070 Ti Laptop GPU (12 GiB, sm_120), WSL2, conda env `tcng` (Python 3.10).
torch 2.11.0+cu128 · jax 0.6.2 · tensorflow 2.21.0 · cupy 14.1.1. **Date:** 2026-07-21.
Plan: `docs/superpowers/plans/2026-07-21-l3-gpu-bf16.md`; env decision: `results/env_decision.md`.

## Headline finding (the plan's premise is inverted)

The complex<bf16> pair-algebra is **correct** and runs **native bf16 Tensor-Core GEMM**
(K3, below) — but for tc.Circuit contraction (cotengra breaks the circuit into many small
contractions) it is **slower than complex64 and uses comparable-or-more peak memory**, on
all four GPU backends. The 4-matmul complex→real decomposition (`cr=ar·br−ai·bi;
ci=ar·bi+ai·br`) materialises 4 intermediates, which outweighs bf16's 2× per-element byte
saving. So bf16's benefit here is the **large-dense-matmul (micro) GEMM and accuracy
preservation**, NOT circuit-contraction memory or speed. (At large n the state vector
eventually dominates and complex64 OOMs first — see `results/e2e_pytorch_bign.csv` — but
the contraction workspace itself is not halved.)

## K3 — native bf16 Tensor-Core GEMM (micro: one 4096² complex-bf16 matmul = 4 bf16 GEMMs)

| backend | micro wall (s) | evidence |
|---|---|---|
| cupy | 0.0367 | cuBLAS bf16 GEMM (ml_dtypes.bfloat16) |
| tensorflow | 0.0455 | XLA bf16 matmul |
| jax | 0.0520 | StableHLO: 4× `dot_general` bf16 (`results/jax_hlo.txt`) |
| pytorch | 0.0575 | `cutlass_80_tensorop_bf16_s16816gemm` kernel, 48 vs 12 TFLOPS (4.06× vs fp32; `results/k3_pytorch.md`) |

K3 substitutes ncu (not installed in `tcng`/WSL) with torch.profiler + StableHLO dump +
TFLOPS — same or stronger evidence. For reference, pytorch micro is 5.9× faster on GPU than
the pre-fix CPU run (0.0575 vs 0.337 s).

## Accuracy (bf16 vs complex64, max-abs-err)

| backend | ghz | brickwork (n=18) | qaoa-ising (n=18) | note |
|---|---|---|---|---|
| pytorch | 7.6e-5 | 2.3e-4 | 4.3e-4 | true fp32 complex64 baseline |
| tensorflow | 7.6e-5 | 2.3e-4 | 4.3e-4 | true fp32 complex64 baseline |
| cupy | 7.6e-5 | 2.3e-4 | 4.3e-4 | true fp32 complex64 baseline |
| jax | **0.0** | 2.4e-4 | 4.0e-4 | DEFAULT matmul precision already reduces complex64 to bf16/TF32 (precision=[DEFAULT,DEFAULT] in the HLO), so ghz is bit-identical; bf16 does 4 dots for the same result |

## Per-backend notes
- **pytorch**: tc-ng's pytorch backend defaults to CPU — harness needed `torch.set_default_device('cuda')` (commit 34449322) else the "GPU" benchmark ran on CPU (peak_alloc=0). `peak_alloc_bytes` (torch.cuda.max_memory_allocated) is the precise per-contraction metric for pytorch; others use nvidia-smi `peak_smi_bytes` only.
- **jax**: `peak_alloc_bytes` null (no equivalent). Big-n (ghz 24/26/28) omitted: XLA JIT-compile of the large pair network + cotengra pathfinding made a single n=28 trial run >7 min (CPU-bound, GPU ~6% util).
- **tensorflow**: 2.21 has no sm_120 kernels → PTX JIT (CUDA module cache keeps small kernels ~fast). Needed LD_LIBRARY_PATH for the nvidia wheel libs + force-reinstall nvidia-nvjitlink-cu12==12.8.93 (the `tensorflow[cuda]` install had left it empty). true-fp32 complex64 baseline.
- **cupy**: needed 3 compat fixes (commit 7dc31d7a) — `cupy.bfloat16` doesn't exist (use ml_dtypes.bfloat16), tensornetwork's cupy backend hasn't implemented einsum (route to `cupy.einsum`), and cupy arrays need explicit `.get()` (use `be.numpy()`).

## End-to-end raw data (bf16 vs complex64, n=14..22)

`mem ratio` = c64 peak_smi / bf16 peak_smi (>1 = bf16 uses less). `speedup` = c64 wall / bf16 wall (<1 = bf16 slower). peak_smi includes the ~1–1.6 GiB CUDA context baseline, so small contractions show ratio ≈1.0 regardless. See `e2e_pytorch_bign.csv` for the large-n memory-inversion evidence.



| backend | circuit | n | c64 mem | bf16 mem | mem ratio | c64 s | bf16 s | speedup | bf16 max-abs-err |
|---|---|---|---|---|---|---|---|---|---|
| cupy | brickwork | 18 | 0.97 GiB | 0.97 GiB | 0.99x | 0.05 | 0.06 | 0.71x | 0.00022731667559128255 |
| cupy | brickwork | 20 | 0.98 GiB | 1.01 GiB | 0.97x | 0.05 | 0.05 | 0.94x | 0.0001309751096414402 |
| cupy | brickwork | 22 | 1.03 GiB | 0.96 GiB | 1.07x | 0.05 | 0.09 | 0.56x | 7.031532004475594e-05 |
| cupy | ghz | 14 | 0.96 GiB | 0.96 GiB | 1.00x | 0.01 | 0.01 | 0.71x | 7.551908493041992e-05 |
| cupy | ghz | 16 | 0.96 GiB | 0.96 GiB | 1.00x | 0.01 | 0.01 | 0.56x | 7.551908493041992e-05 |
| cupy | ghz | 18 | 0.96 GiB | 0.97 GiB | 1.00x | 0.01 | 0.01 | 0.72x | 7.551908493041992e-05 |
| cupy | ghz | 20 | 0.98 GiB | 0.98 GiB | 1.01x | 0.01 | 0.02 | 0.53x | 7.551908493041992e-05 |
| cupy | ghz | 22 | 0.96 GiB | 1.14 GiB | 0.85x | 0.01 | 0.02 | 0.69x | 7.551908493041992e-05 |
| cupy | qaoa-ising | 18 | 0.96 GiB | 0.97 GiB | 0.99x | 0.03 | 0.04 | 0.67x | 0.00043257942888885736 |
| cupy | qaoa-ising | 20 | 0.96 GiB | 0.96 GiB | 1.00x | 0.04 | 0.04 | 0.86x | 0.00027310740551911294 |
| cupy | qaoa-ising | 22 | 0.96 GiB | 1.00 GiB | 0.97x | 0.04 | 0.05 | 0.78x | 0.00018524701590649784 |
| jax | brickwork | 18 | 0.89 GiB | 0.95 GiB | 0.94x | 0.12 | 0.19 | 0.62x | 0.0002415627968730405 |
| jax | brickwork | 20 | 0.90 GiB | 1.01 GiB | 0.89x | 0.13 | 0.20 | 0.65x | 0.0001382041082251817 |
| jax | brickwork | 22 | 0.95 GiB | 1.01 GiB | 0.94x | 0.15 | 0.23 | 0.64x | 7.476388418581337e-05 |
| jax | ghz | 14 | 0.89 GiB | 0.95 GiB | 0.93x | 0.01 | 0.02 | 0.40x | 0.0 |
| jax | ghz | 16 | 0.89 GiB | 0.95 GiB | 0.93x | 0.01 | 0.03 | 0.38x | 0.0 |
| jax | ghz | 18 | 0.89 GiB | 0.95 GiB | 0.94x | 0.01 | 0.03 | 0.41x | 0.0 |
| jax | ghz | 20 | 0.91 GiB | 1.01 GiB | 0.90x | 0.01 | 0.03 | 0.45x | 0.0 |
| jax | ghz | 22 | 1.01 GiB | 1.01 GiB | 1.00x | 0.02 | 0.04 | 0.56x | 0.0 |
| jax | qaoa-ising | 18 | 0.90 GiB | 0.95 GiB | 0.95x | 0.08 | 0.12 | 0.62x | 0.00039975004619918764 |
| jax | qaoa-ising | 20 | 0.91 GiB | 1.01 GiB | 0.90x | 0.09 | 0.13 | 0.70x | 0.0002504627627786249 |
| jax | qaoa-ising | 22 | 1.01 GiB | 1.01 GiB | 1.00x | 0.12 | 0.16 | 0.73x | 0.00016964206588454545 |
| pytorch | brickwork | 18 | 1.53 GiB | 1.53 GiB | 1.00x | 0.05 | 0.05 | 1.04x | 0.00022731667559128255 |
| pytorch | brickwork | 20 | 1.55 GiB | 1.53 GiB | 1.01x | 0.05 | 0.07 | 0.73x | 0.0001309751096414402 |
| pytorch | brickwork | 22 | 1.53 GiB | 1.53 GiB | 1.00x | 0.07 | 0.09 | 0.80x | 7.031532004475594e-05 |
| pytorch | ghz | 14 | 1.51 GiB | 1.53 GiB | 0.98x | 0.01 | 0.01 | 0.79x | 7.551908493041992e-05 |
| pytorch | ghz | 16 | 1.53 GiB | 1.54 GiB | 1.00x | 0.01 | 0.01 | 0.68x | 7.551908493041992e-05 |
| pytorch | ghz | 18 | 1.53 GiB | 1.54 GiB | 1.00x | 0.01 | 0.01 | 0.62x | 7.551908493041992e-05 |
| pytorch | ghz | 20 | 1.55 GiB | 1.54 GiB | 1.01x | 0.01 | 0.02 | 0.57x | 7.551908493041992e-05 |
| pytorch | ghz | 22 | 1.53 GiB | 1.65 GiB | 0.93x | 0.02 | 0.02 | 0.83x | 7.551908493041992e-05 |
| pytorch | qaoa-ising | 18 | 1.54 GiB | 1.54 GiB | 1.00x | 0.03 | 0.04 | 0.73x | 0.00043257942888885736 |
| pytorch | qaoa-ising | 20 | 1.55 GiB | 1.53 GiB | 1.01x | 0.03 | 0.06 | 0.59x | 0.00027310740551911294 |
| pytorch | qaoa-ising | 22 | 1.63 GiB | 1.67 GiB | 0.98x | 0.05 | 0.07 | 0.70x | 0.00018524701590649784 |
| tensorflow | brickwork | 18 | 1.12 GiB | 1.12 GiB | 1.00x | 0.13 | 0.30 | 0.43x | 0.00022731667559128255 |
| tensorflow | brickwork | 20 | 1.12 GiB | 1.04 GiB | 1.08x | 0.14 | 0.33 | 0.43x | 0.0001309751096414402 |
| tensorflow | brickwork | 22 | 1.19 GiB | 1.27 GiB | 0.93x | 0.17 | 0.37 | 0.47x | 7.031532004475594e-05 |
| tensorflow | ghz | 14 | 1.04 GiB | 1.09 GiB | 0.96x | 0.02 | 0.07 | 0.26x | 7.551908493041992e-05 |
| tensorflow | ghz | 16 | 1.04 GiB | 1.12 GiB | 0.94x | 0.02 | 0.05 | 0.42x | 7.551908493041992e-05 |
| tensorflow | ghz | 18 | 1.05 GiB | 1.12 GiB | 0.94x | 0.02 | 0.07 | 0.34x | 7.551908493041992e-05 |
| tensorflow | ghz | 20 | 1.07 GiB | 1.12 GiB | 0.96x | 0.03 | 0.08 | 0.45x | 7.551908493041992e-05 |
| tensorflow | ghz | 22 | 1.17 GiB | 1.37 GiB | 0.85x | 0.04 | 0.09 | 0.44x | 7.551908493041992e-05 |
| tensorflow | qaoa-ising | 18 | 1.03 GiB | 1.04 GiB | 0.99x | 0.09 | 0.22 | 0.43x | 0.00043257942888885736 |
| tensorflow | qaoa-ising | 20 | 1.03 GiB | 1.08 GiB | 0.95x | 0.10 | 0.22 | 0.45x | 0.00027310740551911294 |
| tensorflow | qaoa-ising | 22 | 1.15 GiB | 1.28 GiB | 0.90x | 0.12 | 0.27 | 0.45x | 0.00018524701590649784 |
