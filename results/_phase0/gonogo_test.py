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


def test_c2_layer_status_reads_sublayer():
    from results._phase0.gonogo import _c2_layer_status
    data = {"n24": {"layers": {"C2_REGION_KERNEL_FEASIBILITY": "PASS",
                               "C2_CANONICAL": "UNKNOWN"}}}
    assert _c2_layer_status(data, "C2_REGION_KERNEL_FEASIBILITY") == "PASS"
    assert _c2_layer_status(data, "C2_CANONICAL") == "UNKNOWN"
    # missing layer or malformed -> UNKNOWN (never default PASS)
    assert _c2_layer_status(data, "C2_NOPE") == "UNKNOWN"
    assert _c2_layer_status({}, "C2_CANONICAL") == "UNKNOWN"


def test_c3_full_matrix_status(tmp_path):
    import json
    from results._phase0.gonogo import _c3_planar_full_matrix_status
    p = tmp_path / "fm.csv"
    p.write_text("M,N,K,status\n1024,1024,1024,ok\n")
    assert _c3_planar_full_matrix_status(str(p)) == "PASS"
    assert _c3_planar_full_matrix_status(str(tmp_path / "missing.csv")) == "NOT_RUN"
    empty = tmp_path / "empty.csv"
    empty.write_text("M,N,K,status\n")
    assert _c3_planar_full_matrix_status(str(empty)) == "UNKNOWN"


def test_c3_grouped_status(tmp_path):
    import json
    from results._phase0.gonogo import _c3_grouped_status
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"capability": {"status": "NOT_SUPPORTED"}}))
    assert _c3_grouped_status(str(p)) == "NOT_SUPPORTED"
    p.write_text(json.dumps({"capability": {"status": "SUPPORTED"}}))
    assert _c3_grouped_status(str(p)) == "SUPPORTED"
    assert _c3_grouped_status(str(tmp_path / "missing.json")) == "NOT_RUN"


def test_cutlass_status_derives_from_single_4m(tmp_path):
    import json
    from results._phase0.gonogo import _cutlass_status
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"single_4m": {
        "kernel_path": "sm80_fallback", "compiles": True, "runs": True,
        "correctness": {"gate_pass": True}}}))
    assert _cutlass_status(str(p)) == "FEASIBLE_WITH_SM80_FALLBACK"
    p.write_text(json.dumps({"single_4m": {
        "kernel_path": "sm120_native", "compiles": True, "runs": True,
        "correctness": {"gate_pass": True}}}))
    assert _cutlass_status(str(p)) == "FEASIBLE"
    p.write_text(json.dumps({"single_4m": {
        "kernel_path": "sm80_fallback", "compiles": True, "runs": False,
        "correctness": {"gate_pass": False}}}))
    assert _cutlass_status(str(p)) == "FAIL"
    assert _cutlass_status(str(tmp_path / "missing.json")) == "NOT_RUN"


def test_region_proto_status(tmp_path):
    import json
    from results._phase0.gonogo import _region_proto_status
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"verdict": "FEASIBLE_WITH_RECOMPUTE"}))
    assert _region_proto_status(str(p)) == "FEASIBLE_WITH_RECOMPUTE"
    p.write_text(json.dumps({"verdict": "NOT_FEASIBLE"}))
    assert _region_proto_status(str(p)) == "NOT_FEASIBLE"
    assert _region_proto_status(str(tmp_path / "missing.json")) == "NOT_RUN"


def test_numerical_status_reads_overall_and_per_route(tmp_path):
    import json
    from results._phase0.gonogo import (
        _numerical_overall_status, _numerical_per_route)
    p = tmp_path / "n.json"
    p.write_text(json.dumps({
        "overall_numerical_status": "FAIL",
        "per_route": [
            {"route": "planar", "criterion": "FAIL", "n_cells": 144},
            {"route": "region_fused", "criterion": "PASS", "n_cells": 9},
        ]}))
    assert _numerical_overall_status(str(p)) == "FAIL"
    per = _numerical_per_route(str(p))
    assert per["planar"] == "FAIL" and per["region_fused"] == "PASS"
    # missing artifact -> overall NOT_RUN, empty per-route map
    assert _numerical_overall_status(str(tmp_path / "missing.json")) == "NOT_RUN"
    assert _numerical_per_route(str(tmp_path / "missing.json")) == {}


def test_numerical_per_route_skips_malformed_row(tmp_path):
    import json
    from results._phase0.gonogo import _numerical_per_route
    p = tmp_path / "n.json"
    # row with valid criterion but no route key must not raise; valid rows kept
    p.write_text(json.dumps({"per_route": [
        {"criterion": "PASS"},                       # malformed: no route
        {"route": "planar", "criterion": "FAIL"},    # valid
    ]}))
    per = _numerical_per_route(str(p))
    assert per == {"planar": "FAIL"}


def test_capability_layer_combines_per_route():
    from results._phase0.gonogo import capability_layer
    criteria = {
        "C3_PLANAR_CORE": "SUPPORTED", "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",
        "REGION_PROTOTYPE": "FEASIBLE_WITH_RECOMPUTE", "C2_REGION_KERNEL": "PASS",
        "CUTLASS_SM120_4M": "FEASIBLE_WITH_SM80_FALLBACK",
    }
    cap = capability_layer(criteria)
    assert cap["planar"] == "OK"            # core OK + full matrix OK
    assert cap["grouped"] == "NOT_OK"       # NOT_SUPPORTED
    assert cap["region_fused"] == "OK"      # region proto OK + region kernel OK
    assert cap["cutlass_4m_single"] == "OK"


def test_capability_layer_undetermined_if_any_dep_not_run():
    from results._phase0.gonogo import capability_layer
    criteria = {"C3_PLANAR_CORE": "SUPPORTED", "C3_PLANAR_FULL_MATRIX": "NOT_RUN",
                "C3_GROUPED": "NOT_SUPPORTED", "REGION_PROTOTYPE": "NOT_RUN",
                "C2_REGION_KERNEL": "PASS", "CUTLASS_SM120_4M": "NOT_RUN"}
    cap = capability_layer(criteria)
    assert cap["planar"] == "UNDETERMINED"  # full matrix NOT_RUN, no NOT_OK
    assert cap["region_fused"] == "UNDETERMINED"


def test_numerical_layer_maps_per_route():
    from results._phase0.gonogo import numerical_layer, ROUTES
    per = {"planar": "FAIL", "grouped": "FAIL", "region_fused": "PASS",
           "cutlass_4m_single": "PASS"}
    num = numerical_layer(per, ROUTES)
    assert num["planar"] == "NOT_OK"
    assert num["region_fused"] == "OK"


def test_numerical_layer_missing_route_is_undetermined():
    from results._phase0.gonogo import numerical_layer, ROUTES
    num = numerical_layer({"region_fused": "PASS"}, ROUTES)
    assert num["planar"] == "UNDETERMINED"  # absent -> UNDETERMINED
    assert num["region_fused"] == "OK"


def test_route_verdict_viable_requires_both_ok():
    from results._phase0.gonogo import route_verdict
    rv = route_verdict(
        {"planar": "OK", "grouped": "NOT_OK", "region_fused": "OK",
         "cutlass_4m_single": "OK"},
        {"planar": "NOT_OK", "grouped": "NOT_OK", "region_fused": "OK",
         "cutlass_4m_single": "OK"})
    assert rv["planar"]["status"] == "NOT_VIABLE"          # num NOT_OK
    assert rv["planar"]["numerical"] == "NOT_OK"
    assert rv["grouped"]["status"] == "NOT_VIABLE"         # both NOT_OK
    assert rv["region_fused"]["status"] == "VIABLE"        # both OK
    assert rv["cutlass_4m_single"]["status"] == "VIABLE"


def test_route_verdict_unknown_when_undetermined_and_no_not_ok():
    from results._phase0.gonogo import route_verdict
    rv = route_verdict(
        {"planar": "OK", "grouped": "UNDETERMINED", "region_fused": "OK",
         "cutlass_4m_single": "OK"},
        {"planar": "UNDETERMINED", "grouped": "NOT_OK", "region_fused": "OK",
         "cutlass_4m_single": "OK"})
    assert rv["planar"]["status"] == "UNKNOWN"   # num UNDETERMINED, no NOT_OK
    assert rv["grouped"]["status"] == "NOT_VIABLE"  # grouped num NOT_OK


def test_route_verdict_rule3_region_kernel_fail_sinks_region():
    # rule 3 encoded structurally: region capability NOT_OK -> NOT_VIABLE
    from results._phase0.gonogo import route_verdict
    rv = route_verdict(
        {"planar": "OK", "grouped": "OK", "region_fused": "NOT_OK",
         "cutlass_4m_single": "OK"},
        {"planar": "OK", "grouped": "OK", "region_fused": "OK",
         "cutlass_4m_single": "OK"})
    assert rv["region_fused"]["status"] == "NOT_VIABLE"


def test_completion_inconclusive_if_any_required_unknown():
    from results._phase0.gonogo import evaluate_completion
    criteria = {c: "PASS" for c in (
        "C1", "C2", "C3_PLANAR_CORE", "C3_PLANAR_FULL_MATRIX", "C3_GROUPED",
        "CUTLASS_SM120_4M", "REGION_PROTOTYPE", "NUMERICAL")}
    criteria["C2"] = "UNKNOWN"  # the real binding constraint
    assert evaluate_completion(criteria) == "INCONCLUSIVE"


def test_completion_complete_when_all_determined_and_numerical_fail_ok():
    # NUMERICAL=FAIL is "determined" -> does NOT sink completion (rule 5 edge)
    from results._phase0.gonogo import evaluate_completion
    criteria = {c: "PASS" for c in (
        "C1", "C2", "C3_PLANAR_CORE", "C3_PLANAR_FULL_MATRIX", "C3_GROUPED",
        "CUTLASS_SM120_4M", "REGION_PROTOTYPE")}
    criteria["NUMERICAL"] = "FAIL"
    criteria["C3_GROUPED"] = "NOT_SUPPORTED"  # determined, not UNKNOWN
    assert evaluate_completion(criteria) == "COMPLETE"


def test_completion_inconclusive_if_c3_subordinate_not_run():
    # rule 4: C3_PLANAR_CORE PASS but FULL_MATRIX NOT_RUN -> INCONCLUSIVE
    from results._phase0.gonogo import evaluate_completion
    criteria = {c: "PASS" for c in (
        "C1", "C2", "C3_PLANAR_CORE", "C3_PLANAR_FULL_MATRIX", "C3_GROUPED",
        "CUTLASS_SM120_4M", "REGION_PROTOTYPE", "NUMERICAL")}
    criteria["C3_PLANAR_FULL_MATRIX"] = "NOT_RUN"
    assert evaluate_completion(criteria) == "INCONCLUSIVE"


def test_authorize_phase1_truth_table():
    from results._phase0.gonogo import authorize_phase1
    viable = {"region_fused": {"status": "VIABLE"}}
    none = {"planar": {"status": "NOT_VIABLE"}, "grouped": {"status": "NOT_VIABLE"},
            "region_fused": {"status": "NOT_VIABLE"}, "cutlass_4m_single": {"status": "NOT_VIABLE"}}
    assert authorize_phase1("COMPLETE", viable) == "GO_TO_PHASE1"
    assert authorize_phase1("COMPLETE", none) == "NO_GO"
    assert authorize_phase1("INCONCLUSIVE", viable) == "NOT_AUTHORIZED"


def test_aggregate_two_layer_end_to_end():
    from results._phase0.gonogo import aggregate_two_layer
    criteria = {
        "C1": "PASS", "C2": "UNKNOWN", "C2_REGION_KERNEL": "PASS",
        "C3_PLANAR_CORE": "SUPPORTED", "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED", "CUTLASS_SM120_4M": "FEASIBLE_WITH_SM80_FALLBACK",
        "REGION_PROTOTYPE": "FEASIBLE_WITH_RECOMPUTE", "NUMERICAL": "FAIL"}
    rv = {
        "planar": {"status": "NOT_VIABLE", "capability": "OK", "numerical": "NOT_OK"},
        "grouped": {"status": "NOT_VIABLE", "capability": "NOT_OK", "numerical": "NOT_OK"},
        "region_fused": {"status": "VIABLE", "capability": "OK", "numerical": "OK"},
        "cutlass_4m_single": {"status": "VIABLE", "capability": "OK", "numerical": "OK"}}
    agg = aggregate_two_layer(criteria, rv, "INCONCLUSIVE", "NOT_AUTHORIZED")
    assert agg["schema_version"] == "gonogo-v2"
    assert agg["phase0_completion"] == "INCONCLUSIVE"
    assert agg["phase1_authorization"] == "NOT_AUTHORIZED"
    assert agg["criteria"]["C2"] == "UNKNOWN"
    assert agg["route_verdict"]["region_fused"]["status"] == "VIABLE"
    assert any("C2" in r for r in agg["reasons"])
    assert "c2_judgment.json" in " ".join(agg["blocking_artifacts"])


def test_render_md_matches_json_object():
    # truth-table rule 7: MD is generated from the same object -> no contradiction
    from results._phase0.gonogo import aggregate_two_layer, _render_md
    agg = aggregate_two_layer({"NUMERICAL": "FAIL"}, {}, "INCONCLUSIVE", "NOT_AUTHORIZED")
    md = _render_md(agg)
    assert "INCONCLUSIVE" in md
    assert "NOT_AUTHORIZED" in md
    # the four phase-level fields appear and agree with the JSON object
    for field in ("phase0_completion", "phase1_authorization"):
        assert agg[field] in md


def test_main_emits_consistent_gonogo_v2(tmp_path, monkeypatch):
    # Drive main() against a staging dir with the real (current) phase0
    # artifacts copied in, then assert the emitted gonogo is schema-valid,
    # honest (INCONCLUSIVE while C2 is UNKNOWN), and JSON/MD agree.
    import json, os, shutil
    from results._phase0 import gonogo as G

    src = "results/phase0"
    stage = tmp_path / "phase0"
    stage.mkdir()
    for name in ("c1_judgment.json", "c2_judgment.json",
                 "cublaslt_planar_capability.json", "cublaslt_grouped_capability.json",
                 "cublaslt_full_matrix.csv", "cutlass_sm120_4m.json",
                 "region_prototype.json", "numerical_validation.json",
                 "cublaslt_gap.txt"):
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy(s, stage / name)

    monkeypatch.setattr(G, "_collect_environment", lambda: {"_stub": True})

    G.main(stage_dir=str(stage))

    agg = json.load(open(stage / "gonogo.json"))
    assert agg["schema_version"] == "gonogo-v2"
    # Honest headline: C2 canonical is UNKNOWN -> INCONCLUSIVE, not GO.
    assert agg["phase0_completion"] == "INCONCLUSIVE"
    assert agg["phase1_authorization"] == "NOT_AUTHORIZED"
    # per-route fail-closed: region_fused VIABLE, planar/grouped NOT_VIABLE
    assert agg["route_verdict"]["region_fused"]["status"] == "VIABLE"
    assert agg["route_verdict"]["planar"]["status"] == "NOT_VIABLE"
    # rule 7: MD rendered from same object
    md = (stage / "gonogo.md").read_text()
    assert agg["phase0_completion"] in md and agg["phase1_authorization"] in md
    # minimal manifest is consistent with the new verdict (no stale GO_TO_PHASE1)
    manifest = json.load(open(stage / "manifest.json"))
    assert manifest["phase0_completion"] == agg["phase0_completion"]
    assert manifest["phase1_authorization"] == agg["phase1_authorization"]


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
