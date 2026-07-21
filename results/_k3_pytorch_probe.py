"""K3 evidence for pytorch on RTX 5070 Ti (sm_120): native bf16 GEMM hits Tensor Cores.

Substitutes for ncu (unavailable in the tcng env / WSL). Uses torch.profiler to
capture CUDA kernel names (same information as `ncu --kernel-name regex:mma,gemm`)
and a bf16-vs-fp32 TFLOPS comparison: with TF32 disabled, fp32 runs on CUDA cores
while bf16 runs on Tensor Cores -> a large bf16 speedup proves Tensor Core use.
"""
import time
import torch
from torch.profiler import profile, ProfilerActivity

torch.backends.cuda.matmul.allow_tf32 = False  # fp32 = pure CUDA-core FMA, not TC
dev = "cuda"
m = 8192
flops = 2 * m ** 3


def time_matmul(dtype, warmup=3, iters=10):
    a = torch.randn(m, m, device=dev, dtype=dtype)
    b = torch.randn(m, m, device=dev, dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    del a, b, c
    torch.cuda.empty_cache()
    return dt


bf = time_matmul(torch.bfloat16)
fp = time_matmul(torch.float32)
bf_tflops = flops / bf / 1e12
fp_tflops = flops / fp / 1e12
print(f"=== matmul {m}x{m} (2*m^3 = {flops:,} FLOPs) ===")
print(f"bf16 wall {bf*1e3:7.2f} ms  -> {bf_tflops:7.1f} TFLOPS")
print(f"fp32 wall {fp*1e3:7.2f} ms  -> {fp_tflops:7.1f} TFLOPS  (TF32 disabled)")
print(f"bf16 speedup vs fp32: {fp/bf:.2f}x   | bf16/fp32 TFLOPS ratio: {bf_tflops/fp_tflops:.2f}x")

print("\n=== torch.profiler CUDA kernels for one bf16 matmul ===")
a = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
b = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    c = a @ b
    torch.cuda.synchronize()
ka = prof.key_averages()
# print top kernels by CUDA time
print(ka.table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=70))
# collect names, scan for bf16/gemm/tensor/mma/cutlass cues
names = [e.key for e in ka if e.device_time_total > 0]
import re

cues = {}
for n in names:
    for cue in ("bf16", "bfloat16", "gemm", "tensor", "mma", "cutlass", "cublas", "splitK", "sm120", "sm_120"):
        if cue.lower() in n.lower():
            cues.setdefault(cue, []).append(n)
print("=== kernel-name cues ===")
for cue, ks in cues.items():
    print(f"  [{cue}] ({len(ks)} kernel(s)) e.g. {ks[0][:90]}")
if not cues:
    print("  (no bf16/gemm/mma cues in kernel names — raw names above are the evidence)")

print("\n=== device ===")
print(torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
print("allow_tf32 =", torch.backends.cuda.matmul.allow_tf32)
print("=== K3 probe done ===")
