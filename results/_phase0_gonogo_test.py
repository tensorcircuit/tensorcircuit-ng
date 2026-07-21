"""Unit tests for go/no-go criterion evaluation. Run: pytest results/_phase0_gonogo_test.py -v"""

from results._phase0_gonogo import evaluate_criteria, VERDICT_GO, VERDICT_NOGO


def test_all_yes_is_go():
    res = evaluate_criteria(
        has_unavoidable_materialization=True,
        materialization_single_consumer_mappable=True,
        bf16_ceiling_ratio=2.5,
    )
    assert res["verdict"] == VERDICT_GO


def test_no_window_is_nogo():
    res = evaluate_criteria(
        has_unavoidable_materialization=False,
        materialization_single_consumer_mappable=True,
        bf16_ceiling_ratio=2.5,
    )
    assert res["verdict"] == VERDICT_NOGO
    assert "window" in res["reason"].lower()


def test_window_but_not_coverable_is_open():
    # 窗口存在但不可覆盖 → 非 go，记开放问题
    res = evaluate_criteria(
        has_unavoidable_materialization=True,
        materialization_single_consumer_mappable=False,
        bf16_ceiling_ratio=2.5,
    )
    assert res["verdict"] == VERDICT_NOGO
    assert "cover" in res["reason"].lower() or "region" in res["reason"].lower()


def test_low_ceiling_is_nogo():
    res = evaluate_criteria(
        has_unavoidable_materialization=True,
        materialization_single_consumer_mappable=True,
        bf16_ceiling_ratio=1.1,
    )
    assert res["verdict"] == VERDICT_NOGO


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
