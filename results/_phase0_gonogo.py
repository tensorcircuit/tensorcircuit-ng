"""Phase 0 go/no-go 聚合。
假设：套 spec §7 三门槛，由三探针产出判定是否值得投入后续大工程（含 libcublasLt 绑定）。
方法：pure 评估函数 evaluate_criteria(...)；main() 交互式（或读笔记）收集三项输入，出判定写 verdict.md。
用法：MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_gonogo.py
"""

from __future__ import annotations
import argparse
import sys

VERDICT_GO = "GO"
VERDICT_NOGO = "NO-GO"

_CEILING_RATIO_THRESHOLD = 1.3


def evaluate_criteria(
    has_unavoidable_materialization: bool,
    materialization_single_consumer_mappable: bool,
    bf16_ceiling_ratio: float,
) -> dict:
    """套 spec §7。返回 {verdict, reason, criteria}。"""
    criteria = {
        "1_window_exists": has_unavoidable_materialization,
        "2_coverable": materialization_single_consumer_mappable,
        "3_ceiling_real": bf16_ceiling_ratio >= _CEILING_RATIO_THRESHOLD,
    }
    if not criteria["1_window_exists"]:
        return {
            "verdict": VERDICT_NOGO,
            "reason": "no unavoidable-materialization window found — bf16 has nothing to halve",
            "criteria": criteria,
        }
    if not criteria["2_coverable"]:
        return {
            "verdict": VERDICT_NOGO,
            "reason": "window exists but NOT single-consumer/tile-mappable — region fusion (spec §8.1) cannot cover it; open problem",
            "criteria": criteria,
        }
    if not criteria["3_ceiling_real"]:
        return {
            "verdict": VERDICT_NOGO,
            "reason": f"bf16 Tensor Core ceiling not real on SM120 (ratio {bf16_ceiling_ratio:.2f} < {_CEILING_RATIO_THRESHOLD})",
            "criteria": criteria,
        }
    return {
        "verdict": VERDICT_GO,
        "reason": "window exists, coverable, ceiling real — proceed to libcublasLt binding (deferred Probe 1)",
        "criteria": criteria,
    }


def _collect_from_user():
    """交互收集三项（也可改为解析探针输出文件；此处人读笔记驱动）。"""
    print("# 依据三探针产出填写（参考 results/_phase0_*.txt）：")
    has = (
        input("Probe 3 是否存在 materialized-unavoidable 区？(y/n): ").strip().lower()
        == "y"
    )
    cov = (
        input("该区是否 single-consumer/tile-mappable（nsys/HLO 人工判断）？(y/n): ")
        .strip()
        .lower()
        == "y"
    )
    ratio = float(
        input("Probe 1 代理 bf16/fp32 TFLOPS 比的最大值（如 4.5）: ").strip() or "0"
    )
    return has, cov, ratio


def _parse_cli_args(argv):
    """解析 CLI args；三项都给齐返回 (has, cov, ratio)，否则返回 None。"""
    p = argparse.ArgumentParser(
        description="Phase 0 go/no-go aggregator (spec §7 three criteria)"
    )
    p.add_argument(
        "--has",
        choices=["y", "n"],
        help="criterion 1: unavoidable-materialization window exists (y/n)",
    )
    p.add_argument(
        "--coverable",
        choices=["y", "n"],
        help="criterion 2: window is single-consumer/tile-mappable (y/n)",
    )
    p.add_argument(
        "--ratio",
        type=float,
        help="criterion 3: bf16/fp32 TFLOPS ceiling ratio (e.g. 2.7)",
    )
    args = p.parse_args(argv)
    if args.has is not None and args.coverable is not None and args.ratio is not None:
        return (args.has == "y", args.coverable == "y", float(args.ratio))
    return None


def main(argv=None):
    """argv=None 走 sys.argv[1:]；三项 CLI 齐则非交互，否则交互收集。"""
    cli = _parse_cli_args(sys.argv[1:] if argv is None else argv)
    if cli is not None:
        has, cov, ratio = cli
    else:
        has, cov, ratio = _collect_from_user()
    res = evaluate_criteria(has, cov, ratio)
    lines = [
        "# Phase 0 Go/No-Go Verdict",
        "",
        f"**Verdict: {res['verdict']}**",
        "",
        f"Reason: {res['reason']}",
        "",
        "Criteria:",
    ]
    for k, v in res["criteria"].items():
        lines.append(f"- {k}: {v}")
    text = "\n".join(lines) + "\n"
    print(text)
    with open("results/_phase0_gonogo_verdict.md", "w") as f:
        f.write(text)
    print("=== phase0_gonogo done === (written to results/_phase0_gonogo_verdict.md)")


if __name__ == "__main__":
    main()
