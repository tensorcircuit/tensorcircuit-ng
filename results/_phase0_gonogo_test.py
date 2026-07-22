"""Unit tests for the four-state Phase 0 aggregator (review §9 truth table).

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_gonogo_test.py
"""

from results._phase0_gonogo import aggregate


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
    from results._phase0_gonogo import _c3_planar_from_capability

    p = tmp_path / "cublaslt_planar_capability.json"
    p.write_text(json.dumps({"capability": {"status": "SUPPORTED", "reason": "ok"}}))
    assert _c3_planar_from_capability(str(p)) == "PASS"
    p.write_text(
        json.dumps({"capability": {"status": "NOT_SUPPORTED", "reason": "slow"}})
    )
    assert _c3_planar_from_capability(str(p)) == "FAIL"
    assert _c3_planar_from_capability(str(tmp_path / "missing.json")) == "NOT_RUN"


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
