"""real-BF16 Tensor Core ceiling proxy + framework gap confirmation (NOT Probe 1 capability — planar cuBLASLt is Plan B)。
假设：torch/jax 无 complex-bf16 dtype，故框架发不出 planar-complex cuBLASLt；唯一直接测法是 libcublasLt 绑定（推迟）。
方法：(1) 程序化坐实 dtype 缺失 + 调真实 pair-complex matmul 数 ≥4 real dot_general；(2) 大 4-real-bf16 GEMM 测 TFLOPS 上限。
用法：MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_cublaslt_gap.py
"""

from __future__ import annotations

from results._phase0_common import fmt_table


def tflops(m: int, k: int, n: int, seconds: float) -> float:
    """GEMM TFLOPS = 2*M*K*N / time / 1e12；seconds<=0 → 0。"""
    if seconds <= 0:
        return 0.0
    return (2 * m * k * n) / seconds / 1e12


def has_complex_bf16_dtype(backend):
    """Actually probe whether the backend exposes a complex-bfloat16 dtype. Returns {present, evidence}."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if backend == "pytorch":
                import torch

                dtypes = [d for d in vars(torch).values() if isinstance(d, torch.dtype)]
                names = sorted(str(d) for d in dtypes)
                present = any("bfloat16" in n and "complex" in n for n in names)
                return {
                    "present": present,
                    "evidence": f"torch dtypes: {names}; complex-bf16 absent",
                }
            if backend == "jax":
                import jax.numpy as jnp

                names = sorted(n for n in dir(jnp) if "bfloat" in n or "complex" in n)
                present = any("complex" in n and "bfloat" in n for n in names)
                return {
                    "present": present,
                    "evidence": f"jnp candidates: {names}; complex-bf16 absent",
                }
        except Exception as e:
            return {"present": False, "evidence": f"probe error: {repr(e)[:200]}"}
    return {"present": False, "evidence": "unknown backend"}


def pair_complex_matmul_hlo(m=64):
    """Call the REAL pair complex matmul from bcomplex32_algebra and count real dot_general in its HLO."""
    import warnings, jax, jax.numpy as jnp
    import tensorcircuit as tc
    from applications.bcomplex32_algebra import bcomplex32

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tc.set_backend("jax")
        ar = jnp.ones((m, m), dtype=jnp.bfloat16)
        ai = jnp.ones((m, m), dtype=jnp.bfloat16)
        br = jnp.ones((m, m), dtype=jnp.bfloat16)
        bi = jnp.ones((m, m), dtype=jnp.bfloat16)
        with bcomplex32():

            def cmul(ar, ai, br, bi):
                be = tc.backend
                cr = be.tensordot(ar, br, 1) - be.tensordot(ai, bi, 1)
                ci = be.tensordot(ar, bi, 1) + be.tensordot(ai, br, 1)
                return cr, ci

            lowered = jax.jit(cmul).lower(ar, ai, br, bi)
        hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    dots = hlo.count("dot_general")
    return {"dot_count": dots, "hlo_head": hlo[:500]}


def _gap_confirmation():
    """程序化展示 dtype 缺失 + HLO。返回 dict 供打印。"""
    rows = []
    for be in ("pytorch", "jax"):
        r = has_complex_bf16_dtype(be)
        rows.append(
            [
                be,
                "complex-bf16 dtype",
                "ABSENT" if not r["present"] else "PRESENT",
                r["evidence"],
            ]
        )
    # HLO：调真实 pair-complex matmul，数 stablehlo 中 real dot_general
    try:
        hlo_res = pair_complex_matmul_hlo(m=64)
        hlo_note = (
            f"pair-complex matmul stablehlo: {hlo_res['dot_count']} dot_general "
            f"(complex matmul = 4 real dot_general; head: {hlo_res['hlo_head'][:120]!r})"
        )
    except Exception as e:
        hlo_note = f"pair-HLO probe failed: {repr(e)[:120]}"
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
        for _ in range(2):  # warmup：分摊 cuBLAS fp32 kernel autotuning
            torch.cuda.synchronize()
            _ = a_f32 @ b_f32
        torch.cuda.synchronize()
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
                f"{tflops(m, m, m, bf_s) / tflops(m, m, m, f32_s) if f32_s else 0:.2f}",
            ]
        )
    return rows


def main():
    print(
        "# real-BF16 Tensor Core ceiling proxy + framework gap confirmation (Plan A Task 2)"
    )
    print("\n## 缺口确认（dtype 探测 + 真实 pair-complex matmul HLO）")
    rows, hlo_note = _gap_confirmation()
    print(fmt_table(["backend", "capability", "status", "evidence"], rows))
    print(f"\nHLO: {hlo_note}")
    print("\n## 可达代理：SM120 bf16 Tensor Core 上限（4-real-GEMM，TF32 off）")
    print(
        fmt_table(
            ["M=N=K", "bf16_TFLOPS", "fp32_TFLOPS", "bf16/fp32"], _proxy_ceiling()
        )
    )
    print("\n# 结论：planar-complex cuBLASLt on SM120 = UNTESTED；")
    print("#       需 libcublasLt 绑定（推迟到 go/no-go 之后）。")
    print("#       上表给若 planar-complex 存在的 Tensor Core 上限参考。")
    print("=== phase0_cublaslt_gap done ===")


if __name__ == "__main__":
    main()
