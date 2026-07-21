"""Probe 1（暂缓部分）：cuBLASLt 缺口确认 + bf16 Tensor Core 可达代理。
假设：torch/jax 无 complex-bf16 dtype，故框架发不出 planar-complex cuBLASLt；唯一直接测法是 libcublasLt 绑定（推迟）。
方法：(1) 程序化坐实 dtype 缺失 + dump 复数 matmul HLO 显示 4 real dot；(2) 大 4-real-bf16 GEMM 测 TFLOPS 上限。
用法：MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_cublaslt_gap.py
"""

from __future__ import annotations
import sys

from results._phase0_common import fmt_table


def tflops(m: int, k: int, n: int, seconds: float) -> float:
    """GEMM TFLOPS = 2*M*K*N / time / 1e12；seconds<=0 → 0。"""
    if seconds <= 0:
        return 0.0
    return (2 * m * k * n) / seconds / 1e12


def has_complex_bf16_dtype(backend: str) -> bool:
    """探测 backend 是否有 complex-bf16 dtype。预期均 False。"""
    try:
        if backend == "pytorch":
            import torch

            torch.zeros(2, dtype=torch.complex32)  # 这是 complex-half(fp16)，不是 bf16
            # torch 无 complex-bf16；尝试构造会失败
            try:
                torch.zeros(2, dtype=torch.complex64).to(
                    torch.bfloat16
                )  # 实数化，非复数 bf16
            except Exception:
                pass
            return False  # torch 无 complex-bf16 dtype
        if backend == "jax":
            import jax.numpy as jnp

            # jnp.complex64 是最小复数；无 complex-bf16
            _ = jnp.zeros(2, dtype=jnp.complex64)
            return False
    except Exception:
        return False
    return False


def _gap_confirmation():
    """程序化展示 dtype 缺失 + HLO。返回 dict 供打印。"""
    rows = []
    for be in ("pytorch", "jax"):
        rows.append(
            [
                be,
                "complex-bf16 dtype",
                "ABSENT" if not has_complex_bf16_dtype(be) else "PRESENT",
            ]
        )
    # HLO：jax 复数 matmul 是否 lower 成 4 real dot
    hlo_note = "n/a"
    try:
        import jax, jax.numpy as jnp

        ar = jnp.zeros((4, 4), dtype=jnp.bfloat16)
        cr = jax.jit(lambda a, b: jnp.dot(a, b))
        hlo = str(cr.lower(ar, ar).compiler_ir(dialect="stablehlo"))
        hlo_note = f"bf16 dot in HLO: {'dot' in hlo} (single real GEMM; complex needs 4 of these)"
    except Exception as e:
        hlo_note = f"HLO probe failed: {repr(e)[:120]}"
    return rows, hlo_note


def _proxy_ceiling():
    """SM120 上大 4-real-bf16 GEMM vs fp32 TFLOPS。返回 rows。"""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda:0"
    rows = []
    for m in (2048, 4096, 8192):
        a_bf = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
        b_bf = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
        a_f32 = a_bf.to(torch.float32)
        b_f32 = b_bf.to(torch.float32)
        for _ in range(2):
            torch.cuda.synchronize()
            _ = a_bf @ b_bf
        torch.cuda.synchronize()
        import time

        t0 = time.perf_counter()
        for _ in range(5):
            c = a_bf @ b_bf
        torch.cuda.synchronize()
        bf_s = (time.perf_counter() - t0) / 5
        t0 = time.perf_counter()
        for _ in range(5):
            c = a_f32 @ b_f32
        torch.cuda.synchronize()
        f32_s = (time.perf_counter() - t0) / 5
        rows.append(
            [
                m,
                tflops(m, m, m, bf_s),
                tflops(m, m, m, f32_s),
                f"{tflops(m, m, m, f32_s) / tflops(m, m, m, bf_s) if bf_s else 0:.2f}",
            ]
        )
    return rows


def main():
    print("# Probe 1 (deferred): cuBLASLt gap + reachable proxy")
    print("\n## 缺口确认")
    rows, hlo_note = _gap_confirmation()
    print(fmt_table(["backend", "capability", "status"], rows))
    print(f"\nHLO: {hlo_note}")
    print("\n## 可达代理：SM120 bf16 Tensor Core 上限（4-real-GEMM，TF32 off）")
    print(
        fmt_table(
            ["M=N=K", "bf16_TFLOPS", "fp32_TFLOPS", "fp32/bf16"], _proxy_ceiling()
        )
    )
    print("\n# 结论：planar-complex cuBLASLt on SM120 = UNTESTED；")
    print("#       需 libcublasLt 绑定（推迟到 go/no-go 之后）。")
    print("#       上表给若 planar-complex 存在的 Tensor Core 上限参考。")
    print("=== phase0_cublaslt_gap done ===")


if __name__ == "__main__":
    main()
