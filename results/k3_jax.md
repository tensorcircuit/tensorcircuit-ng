# K3 evidence — jax on RTX 5070 Ti (sm_120)

**Backend:** jax 0.6.2 (jaxlib 0.6.2, jax-cuda12-plugin 0.6.2) · **Device:** CudaDevice(id=0),
default backend `gpu` · **Date:** 2026-07-21 · **Plan ref:** Task 9 Step 2

## Result: the 4M complex-bf16 contraction lowers to 4 bf16 `dot_general` ops ✅

`_pair_tensordot` computes `cr=ar·br−ai·bi; ci=ar·bi+ai·br` (4 real matmuls). Lowering
the identical raw-array computation and dumping StableHLO (`results/jax_hlo.txt`,
probe `results/_jax_hlo_probe.py`) shows exactly four bf16 dot ops:

```
%0 = stablehlo.dot_general %arg0, %arg2, contracting_dims=[1]x[0], precision=[DEFAULT,DEFAULT] : (tensor<1024x1024xbf16>, ...)
%1 = stablehlo.dot_general %arg1, %arg3, ...   (ai, bi)
%3 = stablehlo.dot_general %arg0, %arg3, ...   (ar, bi)
%4 = stablehlo.dot_general %arg1, %arg2, ...   (ai, br)
```

These compile to bf16 cuBLAS GEMM (Tensor Core) on sm_120. → the pair-algebra runs
**native bf16 GEMM**, not an upcast. (The plan's HLO dump is reproduced; PairTensor
isn't jax-traceable so the identical raw-array computation is lowered instead.)

Micro (`results/micro_jax.csv`): 4096² complex-bf16 matmul **0.052 s** on GPU.

## Important jax-specific nuance: DEFAULT matmul precision already reduces complex64

The dumped dots carry `precision = [DEFAULT, DEFAULT]`. On the jax GPU backend,
**DEFAULT precision permits bf16/TF32 Tensor-Core accumulation for fp32/complex64
inputs too** — so the "complex64" reference contraction is *already* reduced-precision.
Direct evidence (ghz state, 1/√2 amplitude):

```
complex64 ref[:1] = 0.70703125+0.j   ← the bf16 rounding of 1/√2, not true fp32 0.70710678
bf16      got[:1] = 0.70703125+0.j
max_abs_diff = 0.000e+00   (identical, for n=8 and n=12)
```

Consequence: under jax default precision, the bf16 pair-algebra yields **identical
accuracy to complex64** (because complex64 was already computed at bf16/TF32), but it
does so with **4 dot ops instead of the 2 real dots a native complex matmul uses** →
strictly more work for the same result. (To force a true-fp32 complex64 baseline one
would set `jax_default_matmul_precision=highest`; out of scope here — the default
behaviour is the more honest "what users actually get" baseline.)

## E2e memory/speed

See `results/e2e_jax.csv` (n=14..22) + `results/BENCHMARKS.md`. Consistent with pytorch:
bf16 pair does more matmuls, so no contraction-level memory/speed win (bf16 ~0.4–0.7×
the speed and ~1.05× the peak_smi of complex64). `peak_alloc_bytes` is null for jax
(no `max_memory_allocated` equivalent) — memory is via nvidia-smi `peak_smi_bytes` only.

**Big-n (ghz 24/26/28) omitted for jax:** the per-contraction cost is dominated by XLA
JIT compilation of the large pair network + cotengra pathfinding (CPU-bound, GPU at ~6%
util), not the contraction itself — a single n=28 trial ran >7 min without completing
(killed). The large-n memory-inversion is already captured by the pytorch big-n data
(`results/e2e_pytorch_bign.csv`); jax at n≤22 shows the same relative pattern.
