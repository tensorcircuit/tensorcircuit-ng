# CUTLASS SM120 compile probe (review §8)

Compile-level probe: does the CUDA frontend accept BF16 wmma Tensor Core intrinsics for compute capability 12.0?

```
{
  "compile_path": "nvrtc-fallback",
  "nvrtc_version": "12.8",
  "supported_archs": [
    50,
    52,
    53,
    60,
    61,
    62,
    70,
    72,
    75,
    80,
    86,
    87,
    89,
    90,
    100,
    101,
    120
  ],
  "compute_120_supported": true,
  "opts": [
    "-std=c++17",
    "-arch=compute_120",
    "-default-device",
    "-I/home/ubuntu/miniconda3/envs/tcng/lib/python3.10/site-packages/nvidia/cuda_runtime/include",
    "-I/home/ubuntu/miniconda3/envs/tcng/lib/python3.10/site-packages/nvidia/cuda_nvcc/include"
  ],
  "returncode": 0,
  "status": "COMPILES",
  "arch_sm120_ok": true,
  "wmma_bf16_ok": true,
  "stderr_tail": "\u0000"
}
```
