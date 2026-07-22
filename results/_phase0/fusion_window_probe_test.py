"""Unit tests for Probe 3 pure logic. Run: pytest results/_phase0_fusion_window_probe_test.py -v"""

from results._phase0.fusion_window_probe import (
    classify_materialization,
    parse_hlo_counts,
)


def test_classify_unavoidable_when_fusion_does_nothing():
    # 关闭融合后 peak 几乎不变 → XLA 本来没在消除 → 物化不可避免 → 窗口存在
    assert (
        classify_materialization(peak_default=1_000_000, peak_no_fusion=1_050_000)
        == "materialized-unavoidable"
    )


def test_classify_fused_away_when_fusion_was_helping():
    # 关闭融合后 peak 翻倍 → XLA 原本把它融掉了 → 无窗口
    assert (
        classify_materialization(peak_default=1_000_000, peak_no_fusion=2_500_000)
        == "fused-away"
    )


def test_classify_avoidable_in_between():
    assert (
        classify_materialization(peak_default=1_000_000, peak_no_fusion=1_500_000)
        == "materialized-avoidable"
    )


def test_classify_unknown_on_zero():
    assert classify_materialization(0, 0) == "unknown"


def test_classify_no_fusion_zero_is_unknown():
    # no-fusion 臂失败（peak=0）绝不能被 0/1e6 < 1.10 误判为 "物化不可避免"（假信号 bf16 窗口存在）
    assert classify_materialization(1_000_000, 0) == "unknown"


def test_parse_hlo_counts_stablehlo_dots():
    # 真实 stablehlo dump 片段：op 写作 stablehlo.dot_general（pre-opt，无 %fusion.）
    shlo = (
        "module @jit_f {\n"
        "  func.func public @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) {\n"
        "    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : "
        "(tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>\n"
        "    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [0] : "
        "(tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>\n"
        "    %2 = stablehlo.add %1, %1 : tensor<4x4xf32>\n"
        "    return %2 : tensor<4x4xf32>\n"
        "  }\n"
        "}\n"
    )
    counts = parse_hlo_counts(shlo)
    assert counts["dot"] == 2  # 两个 dot_general 收缩
    # fusion 不再由 lens-1 计：stablehlo（pre-opt）里没有 fusion；optimized HLO 对本探针的
    # 无输入电路恒为 0（常量折叠）。fusion 的决定性测量走 lens-2 融合禁用 A/B。
    assert "fusion" not in counts


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
