# CUTLASS/CuTe SM120 4M capability (Task 8)

**overall:** `FEASIBLE_WITH_SM80_FALLBACK`  |  **schema:** `cutlass-sm120-4m-v1`

```
{
  "schema_version": "cutlass-sm120-4m-v1",
  "toolchain": {
    "nvcc_version": "12.8.93",
    "cutlass_head": "2802e22",
    "target_arch": "sm_120",
    "compile_path": "torch.utils.cpp_extension",
    "cuda_runtime": "12.8",
    "cuda_home_source": "/home/ubuntu/miniconda3/envs/nvcc_spike"
  },
  "single_4m": {
    "kernel_path": "sm80_fallback",
    "compiles": true,
    "runs": true,
    "correctness": {
      "max_rel": 6.547227530973032e-05,
      "max_abs": 0.0003509521484375,
      "nan_inf": false,
      "gate_pass": true,
      "seeds": [
        0,
        1,
        2
      ]
    },
    "resource": {
      "registers": null,
      "occupancy": null,
      "workspace_bytes": 0
    },
    "latency": {
      "kernelonly_median_us": 3259.6800327301025,
      "c64_baseline_us": 17029.695510864258,
      "ko_ratio_vs_c64": 5.22434574555505
    },
    "sm100_blocker": "Sm100 initialize failed: kErrorInternal \u2014 cudaFuncSetAttribute on device_kernel<Sm100GemmKernel> fails on sm_120 (Sm100 device MMA gated by __CUDA_ARCH__==1000)",
    "sm120_blocker": "Error building extension 'cutlass_4m_sm120': [1/2] /home/ubuntu/miniconda3/envs/tcng/bin/nvcc -MD -MF cutlass_4m.cuda.o.d -ccbin /home/ubuntu/miniconda3/envs/tcng/bin/x86_64-conda-linux-gnu-cc -DTORCH_EXTENSION_NAME=cutlass_4m_sm120 -DTORCH_API_INCLUDE_EXTENSION_H -I/home/ubuntu/cutlass_spike/include -I/home/ubuntu/cutlass_spike/tools/util/include -isystem /home/ubuntu/miniconda3/envs/tcng/lib/python3.10/site-packages/torch/include -isystem /home/ubuntu/miniconda3/envs/tcng/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ubuntu/miniconda3/envs/tcng/include -isystem /home/ubuntu/miniconda3/envs/tcng/include/python3.10 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_120,code=sm_120 --compiler-options '-fPIC' -std=c++17 -O2 -DCUTLASS_ENABLE_SM120_4M=1 -c /mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu -o cutlass_4m.cuda.o \nFAILED: [code=2] cutlass_4m.cuda.o \n/home/ubuntu/miniconda3/envs/tcng/bin/nvcc -MD -MF cutlass_4m.cuda.o.d -ccbin /home/ubuntu/miniconda3/envs/tcng/bin/x86_64-conda-linux-gnu-cc -DTORCH_EXTENSION_NAME=cutlass_4m_sm120 -DTORCH_API_INCLUDE_EXTENSION_H -I/home/ubuntu/cutlass_spike/include -I/home/ubuntu/cutlass_spike/tools/util/include -isystem /home/ubuntu/miniconda3/envs/tcng/lib/python3.10/site-packages/torch/include -isystem /home/ubuntu/miniconda3/envs/tcng/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ubuntu/miniconda3/envs/tcng/include -isystem /home/ubuntu/miniconda3/envs/tcng/include/python3.10 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_120,code=sm_120 --compiler-options '-fPIC' -std=c++17 -O2 -DCUTLASS_ENABLE_SM120_4M=1 -c /mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu -o cutlass_4m.cuda.o \n/home/ubuntu/cutlass_spike/include/cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp(53): warning #20012-D: __device__ annotation is ignored on a function(\"StaticPersistentTileScheduler100\") that is explicitly defaulted on its first declaration\n    __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n                                                             ^\n\nRemark: The warnings can be suppressed with \"-diag-suppress <warning-number>\"\n\n/home/ubuntu/cutlass_spike/include/cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp(53): warning #20012-D: __host__ annotation is ignored on a function(\"StaticPersistentTileScheduler100\") that is explicitly defaulted on its first declaration\n    __inline__ __attribute__((always_inline)) __attribute__((device)) __attribute__((host))\n                                                                                     ^\n\n/home/ubuntu/cutlass_spike/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl(80): error: static assertion failed with \"SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA.\"\n    static_assert(detail::is_sm10x_f8f6f4_element<ElementA>() && detail::is_sm10x_f8f6f4_element<ElementB>(),\n    ^\n          detected during instantiation of class \"cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA, GmemLayoutATag, AlignmentA, ElementB, GmemLayoutBTag, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType, BuilderScheduleTag, std::enable_if_t<<expression>, void>> [with ElementA=Sm120ElementA, GmemLayoutATag=Sm120LayoutA, AlignmentA=8, ElementB=Sm120ElementB, GmemLayoutBTag=Sm120LayoutB, AlignmentB=8, ElementAccumulator=Sm120ElementAcc, TileShape_MNK=Sm120MmaTileShape, ClusterShape_MNK=Sm120ClusterShape, StageCountType=cutlass::gemm::collective::StageCountAutoCarveout<26624>, BuilderScheduleTag=cutlass::gemm::collective::KernelScheduleAuto]\" at line 300 of /mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu\n\n/home/ubuntu/cutlass_spike/include/cute/arch/mma_sm120.hpp(47): error: static assertion failed with \"No MMA matches SM120_16x8x32_TN for given data types.\"\n    static_assert(cutlass::detail::dependent_false<a_type>, \"No MMA matches SM120_16x8x32_TN for given data types.\");\n    ^\n          detected during:\n            instantiation of class \"cute::SM120_16x8x32_TN<a_type, b_type, c_type> [with a_type=Sm120ElementA, b_type=Sm120ElementB, c_type=Sm120ElementAcc]\" at line 3255\n            instantiation of \"auto cute::rr_op_selector_sm120<ElementA,ElementB,ElementC>() [with ElementA=Sm120ElementA, ElementB=Sm120ElementB, ElementC=Sm120ElementAcc]\" at line 108 of /home/ubuntu/cutlass_spike/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl\n            instantiation of class \"cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA, GmemLayoutATag, AlignmentA, ElementB, GmemLayoutBTag, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType, BuilderScheduleTag, std::enable_if_t<<expression>, void>> [with ElementA=Sm120ElementA, GmemLayoutATag=Sm120LayoutA, AlignmentA=8, ElementB=Sm120ElementB, GmemLayoutBTag=Sm120LayoutB, AlignmentB=8, ElementAccumulator=Sm120ElementAcc, TileShape_MNK=Sm120MmaTileShape, ClusterShape_MNK=Sm120ClusterShape, StageCountType=cutlass::gemm::collective::StageCountAutoCarveout<26624>, BuilderScheduleTag=cutlass::gemm::collective::KernelScheduleAuto]\" at line 300 of /mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu\n\n/home/ubuntu/cutlass_spike/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl(115): error: static assertion failed with \"Non-blockscaled collective builder only supports F8F6F4 MMA.\n\"\n    static_assert(UseF8f6f4, \"Non-blockscaled collective builder only supports F8F6F4 MMA.\\n\");\n    ^\n          detected during instantiation of class \"cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA, GmemLayoutATag, AlignmentA, ElementB, GmemLayoutBTag, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType, BuilderScheduleTag, std::enable_if_t<<expression>, void>> [with ElementA=Sm120ElementA, GmemLayoutATag=Sm120LayoutA, AlignmentA=8, ElementB=Sm120ElementB, GmemLayoutBTag=Sm120LayoutB, AlignmentB=8, ElementAccumulator=Sm120ElementAcc, TileShape_MNK=Sm120MmaTileShape, ClusterShape_MNK=Sm120ClusterShape, StageCountType=cutlass::gemm::collective::StageCountAutoCarveout<26624>, BuilderScheduleTag=cutlass::gemm::collective::KernelScheduleAuto]\" at line 300 of /mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu\n\n/mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu(338): error: identifier \"cutlass_status_name\" is undefined\n     \"Sm120 can_implement failed: \", cutlass_status_name(st), \" (M=\", M, \", N=\", N, \", K=\", K, \")\"\n                                     ^\n\n/mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu(342): error: identifier \"cutlass_status_name\" is undefined\n     \"Sm120 initialize failed: \", cutlass_status_name(st)\n                                  ^\n\n/mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu(345): error: identifier \"cutlass_status_name\" is undefined\n     \"Sm120 run failed: \", cutlass_status_name(st)\n                           ^\n\n6 errors detected in the compilation of \"/mnt/e/Study/.Ashare/OneDrive/OneDriveSync/session/tc/tensorcircuit-ng/results/_phase0/cpp/cutlass_4m.cu\".\nninja: build stopped: subcommand failed.\n"
  },
  "grouped": {
    "status": "SUPPORTED",
    "kernel_path": "sm80_grouped",
    "compiles": true,
    "runs": true,
    "coverage": {
      "shapes_run": 8,
      "shapes_total": 8,
      "note": "representative heterogeneous subset of contraction_shapes.csv"
    },
    "correctness": {
      "max_rel": 3.6017430829815567e-05,
      "max_abs": 0.0001373291015625,
      "nan_inf": false,
      "groups_checked": 8,
      "gate_pass": true,
      "seeds": [
        0
      ]
    },
    "latency": {
      "kernelonly_median_us": 4349.887847900391,
      "c64_baseline_us": 2267.008066177368,
      "ko_ratio_vs_c64": 0.5211647162975962
    }
  },
  "overall": "FEASIBLE_WITH_SM80_FALLBACK",
  "blocker": null
}
```

## Toolkit recipe (reproduce)
1. `conda create -n nvcc_spike -c nvidia cuda-nvcc=12.8`
2. `conda install -n nvcc_spike -c nvidia cuda-cudart-dev=12.8 cuda-cccl=12.8`
3. `git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/cutlass_spike`
4. `CUDA_HOME=<nvcc_spike> TORCH_CUDA_ARCH_LIST=12.0 CUTLASS_ROOT=~/cutlass_spike`
