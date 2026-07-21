# K3 evidence — pytorch on RTX 5070 Ti (sm_120)

**Backend:** pytorch 2.11.0+cu128 · **Device:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, cap (12,0)
**Date:** 2026-07-21 · **Plan ref:** docs/superpowers/plans/2026-07-21-l3-gpu-bf16.md Task 10 Step 2

> **Method note:** the plan's `ncu --kernel-name regex:mma,gemm` could not be used —
> Nsight Compute (`ncu`) is not installed in the `tcng` env and there is no system CUDA
> toolkit in WSL2 (the torch wheel pulls only runtime libs, not Nsight). Substituted an
> equivalent-or-stronger probe using `torch.profiler` (kernel names = same info as
> `ncu --kernel-name`) plus a bf16-vs-fp32 TFLOPS comparison (proves Tensor Core use).
> Raw probe output: `results/k3_pytorch.log`; probe source: `results/_k3_pytorch_probe.py`.

## Result: bf16 contraction runs a native bf16 Tensor-Core GEMM ✅

The single bf16 matmul (8192×8192) launches exactly one CUDA kernel:

```
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_nn_align8>
```

Dissected: `cutlass` (CUTLASS) · `tensorop` (Tensor Core, not CUDA-core FMA) · `bf16` ·
`gemm` · `s16816` (bf16 MMA instruction shape 16×8×16) · tile `128x128_32x4`. → **bf16
complex contraction's 4 real matmuls execute as native bf16 Tensor-Core GEMMs.**

## TFLOPS confirmation (Tensor-Core engagement)

| dtype | matmul 8192² wall | TFLOPS |
|---|---|---|
| bf16 | 22.76 ms | **48.3** |
| fp32 (TF32 off) | 92.36 ms | 11.9 |

bf16/fp32 = **4.06×**. With `allow_tf32=False`, fp32 runs on CUDA cores; a >4× bf16
throughput advantage is the signature of Tensor Core execution. (cuBLAS picked a
`cutlass_80`-tagged kernel — forward-compatible on sm_120 — still a bf16 tensorop GEMM.)

## Micro benchmark (4M path: one complex-bf16 matmul = 4 bf16 GEMMs, m=4096)

`results/micro_pytorch.csv`: median wall **0.0575 s** over 5 trials **on GPU**
(was 0.337 s on CPU before the device fix → **5.9× moving bf16 GEMM to GPU**).

## End-to-end note (circuit contraction)

Native bf16 Tensor-Core GEMM wins for *large dense matmuls* (micro above). For
end-to-end tc.Circuit contraction (cotengra breaks the circuit into many small
contractions) bf16 is **~0.6–0.8× the speed of complex64 and uses ~1.5× MORE peak
memory** at large n (e2e_pytorch_bign.csv: ghz n=28 complex64 5.0 GiB vs bf16 7.5 GiB
allocated). The 4-matmul complex→real decomposition materializes 4 intermediates,
which outweighs the per-element byte saving. So bf16 pair-algebra's benefit is the
micro GEMM / accuracy, not contraction memory or speed. See `results/BENCHMARKS.md`.

