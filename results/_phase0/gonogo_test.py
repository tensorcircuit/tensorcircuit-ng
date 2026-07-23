"""Unit tests for the four-state Phase 0 aggregator (review §9 truth table).

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_gonogo_test.py
"""

from results._phase0.gonogo import aggregate


def test_all_pass_is_go_to_phase1():
    assert aggregate("PASS", "PASS", "PASS", 2.7)["verdict"] == "GO_TO_PHASE1"


def test_c1_fail_is_no_window():
    assert aggregate("FAIL", "PASS", "NOT_RUN", 2.7)["verdict"] == "NO_GO_NO_WINDOW"


def test_c2_fail_is_not_coverable():
    assert aggregate("PASS", "FAIL", "NOT_RUN", 2.7)["verdict"] == "NO_GO_NOT_COVERABLE"


def test_c3_real_only_does_not_make_planar_pass():
    # C3 planar is NOT_RUN (Plan B); real ceiling is auxiliary only
    a = aggregate("PASS", "PASS", "NOT_RUN", 2.7)
    assert a["verdict"] == "INCONCLUSIVE"  # planar not probed


def test_unknown_is_inconclusive():
    assert aggregate("UNKNOWN", "PASS", "NOT_RUN", 2.7)["verdict"] == "INCONCLUSIVE"


def test_c3_planar_from_capability_json(tmp_path):
    import json, os
    from results._phase0.gonogo import _c3_planar_from_capability

    p = tmp_path / "cublaslt_planar_capability.json"
    p.write_text(json.dumps({"capability": {"status": "SUPPORTED", "reason": "ok"}}))
    assert _c3_planar_from_capability(str(p)) == "PASS"
    p.write_text(
        json.dumps({"capability": {"status": "NOT_SUPPORTED", "reason": "slow"}})
    )
    assert _c3_planar_from_capability(str(p)) == "FAIL"
    assert _c3_planar_from_capability(str(tmp_path / "missing.json")) == "NOT_RUN"


def test_normalize_pass_supported_feasible_are_ok():
    from results._phase0.gonogo import _normalize
    for v in ("PASS", "SUPPORTED", "FEASIBLE_WITH_SM80_FALLBACK",
              "FEASIBLE_WITH_RECOMPUTE", "TILE_FUSION_FEASIBLE"):
        assert _normalize(v) == "OK", v


def test_normalize_fail_not_supported_are_not_ok():
    from results._phase0.gonogo import _normalize
    for v in ("FAIL", "NOT_SUPPORTED", "NOT_FEASIBLE"):
        assert _normalize(v) == "NOT_OK", v


def test_normalize_unknown_not_run_blocked_are_undetermined():
    from results._phase0.gonogo import _normalize
    for v in ("UNKNOWN", "NOT_RUN", "BLOCKED", "", "weird-token"):
        assert _normalize(v) == "UNDETERMINED", v


def test_constants_define_routes_and_required_criteria():
    from results._phase0.gonogo import REQUIRED_CRITERIA, ROUTE_CAPABILITY_CRITERIA
    assert set(ROUTE_CAPABILITY_CRITERIA) == {
        "planar", "grouped", "region_fused", "cutlass_4m_single"}
    assert "C2" in REQUIRED_CRITERIA and "NUMERICAL" in REQUIRED_CRITERIA
    # region route depends on the region-kernel sub-criterion (truth-table rule 3)
    assert "C2_REGION_KERNEL" in ROUTE_CAPABILITY_CRITERIA["region_fused"]


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
