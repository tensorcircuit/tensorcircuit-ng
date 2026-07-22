"""Probe 2: OOM / 物化前沿测绘。
假设：在 12GB sm_120 上，每种输出从 'XLA 融合消掉' → '被迫物化' → 'OOM/崩溃' 有可定位边界；
      该边界决定 bf16 是否存在受益窗口。
方法：扫 (circuit, n, depth, output, backend) 矩阵，每 config 一子进程，测 outcome/peak/peak_ratio/ms。
用法：MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_frontier_probe.py --matrix smoke
"""

from __future__ import annotations
import argparse
import sys

from applications.benchmarks.bench_bf16_gpu import (
    build_circuit,
    GpuSmiPoller,
    _setup_gpu_device,
    reset_backend_mem,
)
from results._phase0.common import (
    orchestrate,
    worker_emit,
    fmt_table,
    median_wall_ms,
)

OUTPUTS = ("state", "expectation", "norm")
CIRCUITS = ("brickwork", "ghz", "qaoa-ising")
FULL_NS = (18, 20, 22, 24, 26)
FULL_DEPTHS = (3, 10, 16, 24)


def build_configs(matrix: str):
    """生成扫描矩阵。smoke = 快速冒烟（< 20 config）；full = spec §4.1 全矩阵。"""
    if matrix == "smoke":
        out = []
        for n in (18, 22):
            for depth in (3, 10):
                for output in ("state", "expectation"):
                    out.append(
                        {
                            "circuit": "brickwork",
                            "n": n,
                            "depth": depth,
                            "output": output,
                            "backend": "jax",
                        }
                    )
        return out
    # full
    cfgs = []
    for circuit in CIRCUITS:
        for n in FULL_NS:
            for depth in FULL_DEPTHS:
                for output in OUTPUTS:
                    for backend in ("jax", "pytorch"):
                        cfgs.append(
                            {
                                "circuit": circuit,
                                "n": n,
                                "depth": depth,
                                "output": output,
                                "backend": backend,
                            }
                        )
    return cfgs


def run_output_kind(output: str):
    """校验 output token；返回其本身或 'state' 兜底（单测用）。"""
    return output if output in OUTPUTS else "state"


def _build_deep(n, depth):
    """带可调深度的 brickwork（F6 族），gate set H/cnot/rz(0.7)。匹配 _leverage_jit_probe._build_deep。"""
    import tensorcircuit as tc

    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for _ in range(depth):
        for i in range(0, n - 1, 2):
            c.cnot(i, i + 1)
        for i in range(1, n - 1, 2):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rz(i, theta=0.7)
    return c


def _make_circuit(circuit, n, depth):
    import tensorcircuit as tc

    if circuit == "brickwork":
        return _build_deep(n, depth)
    # ghz / qaoa-ising 用 harness（depth 固定）
    return build_circuit(circuit, n)


def _compute_output(c, output):
    """返回 (value, is_full_state_materialized)。"""
    import tensorcircuit as tc

    if output == "state":
        return c.state(), True
    if output == "expectation":
        # tc-ng API: 每个 op 为 ``(tc.gates.X(), [qubit])``，见
        # ``tensorcircuit/circuit.py`` 中 ``Circuit.expectation`` 文档与示例。
        return c.expectation((tc.gates.z(), [0])), False
    # norm: 标量 terminal，由 state 表达式导出（测 XLA 是否把 state 融掉）
    st = c.state()
    return tc.backend.sum(tc.backend.abs(st) ** 2), False


def _peak_bytes(backend):
    if backend == "pytorch":
        import torch

        return int(torch.cuda.max_memory_allocated())
    import jax

    return int(jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0))


def _sync_for(backend):
    if backend == "pytorch":
        import torch

        return lambda _r: torch.cuda.synchronize()
    import jax

    return lambda r: jax.block_until_ready(r)


def worker_main(argv):
    """单 config 测量。打印单行 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", required=True)
    a = ap.parse_args(argv)

    import tensorcircuit as tc

    tc.set_backend(a.backend)
    if a.backend == "pytorch":
        import torch

        _setup_gpu_device("pytorch", 0)
        torch.backends.cuda.matmul.allow_tf32 = False

    c = _make_circuit(a.circuit, a.n, a.depth)
    fn = lambda: _compute_output(c, a.output)[0]
    sync = _sync_for(a.backend)

    # warmup（不计入 peak）。故意不 try/except：若 warmup 抛出（如 jax
    # ``RESOURCE_EXHAUSTED`` / pytorch ``CUDA out of memory``），让异常向上传播，
    # Python 打印 traceback 到 stderr 并以非零码退出 —— 这样 ``orchestrate``
    # 走 ``returncode != 0`` 分支并用 ``classify_stderr`` 分类为 ``oom`` 等结局，
    # 而不是被这里吞掉后又被 ``orchestrate`` 误覆盖成成功的 ``run``。
    median_wall_ms(fn, warmup=1, iters=1, sync=sync)

    if a.backend == "pytorch":
        reset_backend_mem(a.backend)
    # 测量窗口。``GpuSmiPoller`` 只暴露 context-manager 接口（见
    # ``bench_bf16_gpu.py``），故用 ``with`` 而非 start/stop；peak 在退出后仍可读。
    poller = GpuSmiPoller(gpu=0, interval_s=0.02)
    with poller:
        ms = median_wall_ms(fn, warmup=0, iters=5, sync=sync)
    peak_alloc = _peak_bytes(a.backend)

    state_bytes = (2**a.n) * 8
    worker_emit(
        {
            "outcome": "run",
            "peak_alloc_B": peak_alloc,
            "peak_smi_B": poller.peak_bytes(),
            "ms": ms,
            "peak_ratio_vs_state": (peak_alloc / state_bytes) if state_bytes else 0.0,
            "is_state_output": a.output == "state",
        }
    )


def summarize_frontier(rows):
    """按 (output, backend) 给出 max_run_n / min_fail_n 边界。"""
    summary = {}
    for r in rows:
        cfg = r["config"]
        key = (cfg["output"], cfg["backend"])
        s = summary.setdefault(key, {"max_run_n": -1, "min_fail_n": 10**9})
        if r["ok"]:
            s["max_run_n"] = max(s["max_run_n"], cfg["n"])
        else:
            s["min_fail_n"] = min(s["min_fail_n"], cfg["n"])
    return summary


def build_worker_argv(cfg):
    return [
        "--circuit",
        cfg["circuit"],
        "--n",
        str(cfg["n"]),
        "--depth",
        str(cfg["depth"]),
        "--output",
        cfg["output"],
        "--backend",
        cfg["backend"],
    ]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    import os

    configs = build_configs(a.matrix)
    print(f"# Probe 2 frontier, matrix={a.matrix}, configs={len(configs)}")
    rows = orchestrate(
        configs, build_worker_argv, os.path.abspath(__file__), timeout=a.timeout
    )

    table_rows = []
    for r in rows:
        cfg = r["config"]
        if r["ok"]:
            res = r["result"]
            table_rows.append(
                [
                    cfg["circuit"],
                    cfg["n"],
                    cfg["depth"],
                    cfg["output"],
                    cfg["backend"],
                    "run",
                    res.get("peak_alloc_B", 0),
                    f"{res.get('peak_ratio_vs_state', 0):.2f}",
                    f"{res.get('ms', 0):.1f}",
                ]
            )
        else:
            table_rows.append(
                [
                    cfg["circuit"],
                    cfg["n"],
                    cfg["depth"],
                    cfg["output"],
                    cfg["backend"],
                    r["outcome"],
                    "-",
                    "-",
                    "-",
                ]
            )
    print(
        fmt_table(
            [
                "circuit",
                "n",
                "depth",
                "output",
                "backend",
                "outcome",
                "peak_alloc_B",
                "peak/state",
                "ms",
            ],
            table_rows,
        )
    )
    print("\n# frontier boundary (output, backend) -> max_run_n / min_fail_n:")
    for (out, be), s in sorted(summarize_frontier(rows).items()):
        print(
            f"  {out:12s} {be:8s} max_run_n={s['max_run_n']}  min_fail_n={s['min_fail_n']}"
        )
    print("=== phase0_frontier done ===")


if __name__ == "__main__":
    main()
