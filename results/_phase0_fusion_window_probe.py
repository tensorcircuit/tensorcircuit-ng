"""Probe 3: XLA 融合窗口定位。
假设：真实 tc-ng 收缩里，XLA 在某些子图融不掉（被迫物化）；若这些子图 single-consumer/tile-mappable，
      spec §8.1 region fusion 可覆盖 → bf16 窗口可达。
方法：lens1 静态 stablehlo（dot_general 计数；fusion 不可测，见 parse_hlo_counts 注）；lens2 融合禁用 A/B（决定性 peak 比）；lens3 nsys 时间线。
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
    大 → 原被融掉（无窗口）。任一臂 peak<=0（测量失败/未编译）→ unknown，避免假信号。"""
    if peak_default <= 0:
        return "unknown"
    if peak_no_fusion <= 0:
        return "unknown"
    ratio = peak_no_fusion / peak_default
    if ratio < 1.10:
        return "materialized-unavoidable"
    if ratio > 2.00:
        return "fused-away"
    return "materialized-avoidable"


def parse_hlo_counts(stablehlo_text: str) -> dict:
    """数 stablehlo 文本里的 dot_general 指令（定性收缩信号）。

    注：fusion 是 XLA *优化* 阶段产物，不在 pre-opt stablehlo 里出现。本探针的电路无运行时输入，
    XLA 会常量折叠整图 → optimized HLO 里 fusion 恒为 0（实测 smoke n=18,d=10：stablehlo 385 个
    dot_general，optimized HLO 0 个 fusion）。故 fusion 计数对本 lens-1 既不可得也无意义；
    融合的决定性测量由 lens-2（融合禁用 A/B peak 比）给出，不由本函数计。
    """
    dots = len(re.findall(r"\bdot_general\b", stablehlo_text))
    return {"dot": dots}


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
            # tc-ng API: 每个 op 为 ``(tc.gates.X(), [qubit])``，见
            # ``tensorcircuit/circuit.py`` 中 ``Circuit.expectation`` 签名
            # ``*ops: Tuple[tn.Node, List[int]]``（与 Probe 2 ``_compute_output`` 对齐）。
            # ``tc`` 由外层 ``worker_main`` 的 ``import tensorcircuit as tc`` 提供闭包。
            return c.expectation((tc.gates.z(), [0]))
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

    hlo_counts = {"dot": 0}
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
        # 同时覆盖 state 与 expectation：后者走 ``Circuit.expectation`` 路径，
        # 防 API 漂移（如 ``("z", [0])`` 字符串形式）静默回归。
        base = [
            {"n": 18, "depth": 10, "output": "state"},
            {"n": 18, "depth": 10, "output": "expectation"},
        ]
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
    # 用一个显然可融合的元素wise 链：默认应大量融合，关融合后 peak 应涨
    calib = [
        {"n": 10, "depth": 3, "output": "norm", "disable_fusion": 0},
        {"n": 10, "depth": 3, "output": "norm", "disable_fusion": 1},
    ]
    rows = orchestrate(calib, build_worker_argv, os.path.abspath(__file__), timeout=300)
    peaks = [r["result"]["peak_B"] for r in rows if r["ok"]]
    # 关融合应使 peak 上升；方向反或相等都判 invalid（拒绝错误方向或无信号）
    if len(peaks) == 2 and peaks[1] > peaks[0]:
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
        # single-consumer 启发：dot 计数高 → 收缩链密集；具体子图融合判断留给 lens-2 A/B + nsys
        table_rows.append(
            [
                k[0],
                k[1],
                k[2],
                dflt.get("peak_B", 0),
                nofus.get("peak_B", 0),
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
