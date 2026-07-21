"""Probe 3: XLA 融合窗口定位。
假设：真实 tc-ng 收缩里，XLA 在某些子图融不掉（被迫物化）；若这些子图 single-consumer/tile-mappable，
      spec §8.1 region fusion 可覆盖 → bf16 窗口可达。
方法：lens1 静态 HLO（dot/fusion 计数）；lens2 融合禁用 A/B（决定性 peak 比）；lens3 nsys 时间线。
用法：MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_fusion_window_probe.py --matrix smoke
注：XLA_FLAGS 是进程级，每配置独立子进程（XLA_FLAGS 由 worker 内 os.environ 设）。
"""

from __future__ import annotations
import argparse
import os
import re
import sys

from results._phase0_common import orchestrate, worker_emit, fmt_table, median_wall_ms


def classify_materialization(peak_default: int, peak_no_fusion: int) -> str:
    """据 '关融合/默认' peak 比分类。peak_no_fusion/peak_default ≈1 → 物化不可避免（窗口存在）；
    大 → 原被融掉（无窗口）。"""
    if peak_default <= 0:
        return "unknown"
    ratio = peak_no_fusion / peak_default
    if ratio < 1.10:
        return "materialized-unavoidable"
    if ratio > 2.00:
        return "fused-away"
    return "materialized-avoidable"


def parse_hlo_counts(hlo_text: str) -> dict:
    """数 HLO 文本里的 dot / fusion 指令（定性信号）。"""
    dots = len(re.findall(r"%dot(?:_general)?\.", hlo_text))
    fusions = len(re.findall(r"%fusion\.", hlo_text))
    return {"dot": dots, "fusion": fusions}


def _build_deep(n, depth):
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


def worker_main(argv):
    """单 (circuit,n,depth,output,disable_fusion) 测量。打印单行 JSON。
    disable_fusion=1 时在 import jax 前设 XLA_FLAGS。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--output", default="state")
    ap.add_argument("--disable-fusion", type=int, default=0)
    a = ap.parse_args(argv)

    if a.disable_fusion:
        # 必须在 jax 初始化前设
        prev = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (prev + " --xla_disable_hlo_passes=fusion").strip()

    import jax
    import tensorcircuit as tc

    tc.set_backend("jax")
    c = _build_deep(a.n, a.depth)

    def run():
        if a.output == "expectation":
            return c.expectation(("z", [0]))
        if a.output == "norm":
            st = c.state()
            return tc.backend.sum(tc.backend.abs(st) ** 2)
        return c.state()

    jf = jax.jit(run)
    try:
        jax.block_until_ready(jf())  # 编译 + 一次运行
    except Exception as e:
        worker_emit(
            {
                "outcome": "crash",
                "disable_fusion": a.disable_fusion,
                "error": repr(e)[:300],
            }
        )
        return

    # 稳态 peak（编译后）
    dev = jax.local_devices()[0]
    peak = int(dev.memory_stats().get("peak_bytes_in_use", 0))
    ms = median_wall_ms(jf, warmup=1, iters=5, sync=lambda r: jax.block_until_ready(r))

    hlo_counts = {"dot": 0, "fusion": 0}
    try:
        hlo_text = str(jf.lower().compiler_ir(dialect="stablehlo"))
        hlo_counts = parse_hlo_counts(hlo_text)
    except Exception:
        pass

    worker_emit(
        {
            "outcome": "run",
            "disable_fusion": a.disable_fusion,
            "peak_B": peak,
            "ms": ms,
            "hlo_dot": hlo_counts["dot"],
            "hlo_fusion": hlo_counts["fusion"],
        }
    )


def build_worker_argv(cfg):
    return [
        "--n",
        str(cfg["n"]),
        "--depth",
        str(cfg["depth"]),
        "--output",
        cfg["output"],
        "--disable-fusion",
        str(cfg["disable_fusion"]),
    ]


def _configs(matrix):
    if matrix == "smoke":
        base = [{"n": 18, "depth": 10, "output": "state"}]
    else:
        base = [
            {"n": n, "depth": d, "output": o}
            for n in (18, 20, 22)
            for d in (10, 16)
            for o in ("state", "expectation")
        ]
    cfgs = []
    for b in base:
        cfgs.append({**b, "disable_fusion": 0})
        cfgs.append({**b, "disable_fusion": 1})  # A/B 配对
    return cfgs


def _calibrate_flag():
    """在已知可融合小 case 上验证 --xla_disable_hlo_passes=fusion 有效（peak 应变化）。返回 bool。"""
    import subprocess, sys as _sys

    # 用一个显然可融合的元素wise 链：默认应大量融合，关融合后 peak 应涨
    calib = [
        {"n": 10, "depth": 3, "output": "norm", "disable_fusion": 0},
        {"n": 10, "depth": 3, "output": "norm", "disable_fusion": 1},
    ]
    rows = orchestrate(calib, build_worker_argv, os.path.abspath(__file__), timeout=300)
    peaks = [r["result"]["peak_B"] for r in rows if r["ok"]]
    if len(peaks) == 2 and peaks[1] != peaks[0]:
        print(
            f"# calibration: fusion flag changes peak ({peaks[0]} -> {peaks[1]}); lens2 VALID"
        )
        return True
    print(
        f"# calibration: fusion flag did NOT change peak {peaks}; lens2 INVALID, fallback to lens1+3"
    )
    return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    lens2_valid = _calibrate_flag()

    configs = _configs(a.matrix)
    rows = orchestrate(
        configs, build_worker_argv, os.path.abspath(__file__), timeout=a.timeout
    )

    # 按 (n,depth,output) 配对 default vs no-fusion
    by_key = {}
    for r in rows:
        if not r["ok"]:
            continue
        cfg = r["config"]
        k = (cfg["n"], cfg["depth"], cfg["output"])
        by_key.setdefault(k, {})[cfg["disable_fusion"]] = r["result"]

    table_rows = []
    for k in sorted(by_key):
        pair = by_key[k]
        dflt = pair.get(0, {})
        nofus = pair.get(1, {})
        cls = (
            classify_materialization(dflt.get("peak_B", 0), nofus.get("peak_B", 0))
            if lens2_valid
            else "lens2-invalid"
        )
        # single-consumer 启发：fusion 计数高 + dot 少 → 多为可融合链；具体子图判断留给 nsys
        table_rows.append(
            [
                k[0],
                k[1],
                k[2],
                dflt.get("peak_B", 0),
                nofus.get("peak_B", 0),
                dflt.get("hlo_fusion", 0),
                dflt.get("hlo_dot", 0),
                cls,
            ]
        )
    print(f"# Probe 3 fusion window, matrix={a.matrix}, lens2_valid={lens2_valid}")
    print(
        fmt_table(
            [
                "n",
                "depth",
                "output",
                "peak_default",
                "peak_nofusion",
                "hlo_fusion",
                "hlo_dot",
                "classification",
            ],
            table_rows,
        )
    )
    print(
        "\n# 注：classification=materialized-unavoidable 的 (n,depth) 即 bf16 受益窗口候选；"
    )
    print(
        "#     single-consumer/tile-mappable 需结合 nsys 时间线人工确认（见 _phase0_setup_note）。"
    )
    print("=== phase0_fusion done ===")


if __name__ == "__main__":
    main()
