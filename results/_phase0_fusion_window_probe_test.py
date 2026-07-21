"""Unit tests for Probe 3 pure logic. Run: pytest results/_phase0_fusion_window_probe_test.py -v"""

from results._phase0_fusion_window_probe import (
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


def test_parse_hlo_counts_dots_and_fusions():
    hlo = "sample\n%dot.1 = dot(...) %fusion.2 = fusion(...) %dot.3 = dot_general(...)"
    counts = parse_hlo_counts(hlo)
    assert counts["dot"] >= 2  # dot + dot_general
    assert counts["fusion"] >= 1


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
