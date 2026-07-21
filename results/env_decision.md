# L3 GPU env decision — RTX 5070 Ti (Blackwell, sm_120)

**Date:** 2026-07-21
**Plan:** docs/superpowers/plans/2026-07-21-l3-gpu-bf16.md (Task 8)
**Branch:** feat/contraction-algebra-tropical @ 424a6e78

## Hardware / driver
- GPU: **NVIDIA GeForce RTX 5070 Ti Laptop GPU**, 12 GiB (plan assumed 16 GiB → tighter OOM ceiling; mem-ns kept per plan, OOM expected at the top end for complex64).
- Compute capability: **(12, 0) = sm_120 (Blackwell)**. Requires **CUDA 12.8+** wheels (older wheels reject sm_120).
- Driver: 592.47 (Windows) / WSL2 GPU driver (`libcuda.so` in `/usr/lib/wsl/lib`).

## Software stack
- Host: Windows 11 → **WSL2 `ubuntu-24.04`**.
- Conda env: **`tcng`** (user-named; plan's `tcng-l3` name not used — user instruction wins), Python **3.10.20**, conda 26.5.3.
- torch **2.11.0+cu128** (CUDA 12.8 runtime via bundled nvidia-*-cu12 wheels; cuBLAS 12.8.4, cuDNN 9.19).
- ml_dtypes 0.5.4, cotengra 0.8.2, autoray 0.8.11, opt_einsum 3.4.0, numpy 2.2.6, scipy 1.15.3, tensornetwork-ng 0.5.1.
- tensorcircuit-ng 1.7.0 installed editable (`pip install -e .`) from the repo.

## WSL2 decision (plan Task 8 Step 3) — **option (a): all four backends in WSL2**
The plan's jax-on-Windows branch (a) WSL2 vs (b) accept jax-CPU: we run **everything in WSL2**, so jax takes the GPU too. No Windows-native jax CPU fallback. Rationale: single unified env, mirrors the E2 install path; jax added after pytorch is validated.

## Sequencing (user choice: pytorch-first, then expand)
1. **pytorch** (validated first — most reliable on Blackwell): smoke ✓, then micro K3 + e2e.
2. jax, tensorflow, cupy added incrementally (each its own commit, per Tasks 9/11/12).

## Validation so far
- `torch.cuda.is_available()` True; bf16 matmul on GPU returns bf16. (Task 8 Step 2)
- ghz(4) under `bcomplex32()` matches complex64 to **max-abs-diff 7.55e-05** on the 5070 Ti. (author probe)

Not committed (results file; aggregated into the final results commit in Task 13).
