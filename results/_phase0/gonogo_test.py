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


def test_normalize_only_canonical_pass_is_ok():
    """plan §4 验收 (Task 1): ``_normalize`` no longer promotes artifact-native
    detail tokens (SUPPORTED / FEASIBLE* / TILE_FUSION_FEASIBLE) to OK. Only the
    canonical PASS token is "established good"; detail tokens fail closed to
    UNDETERMINED via ``verdict_schema.normalize_criterion`` (the reader must
    re-derive PASS from evidence upstream)."""
    from results._phase0.gonogo import _normalize

    assert _normalize("PASS") == "OK"
    # Detail tokens must NOT be auto-promoted to OK (Task 1 kill of
    # startswith("FEASIBLE") and the SUPPORTED / TILE_FUSION_FEASIBLE shortcuts).
    for v in (
        "SUPPORTED",
        "FEASIBLE_WITH_SM80_FALLBACK",
        "FEASIBLE_WITH_RECOMPUTE",
        "FEASIBLE",
        "TILE_FUSION_FEASIBLE",
    ):
        assert _normalize(v) == "UNDETERMINED", v


def test_normalize_only_canonical_fail_and_not_supported_are_not_ok():
    """plan §4 验收 (Task 1): canonical FAIL / NOT_SUPPORTED are "established
    bad". Artifact-native NOT_FEASIBLE is a detail token -> UNDETERMINED (the
    reader must re-derive canonical FAIL/NOT_SUPPORTED from evidence upstream)."""
    from results._phase0.gonogo import _normalize

    for v in ("FAIL", "NOT_SUPPORTED"):
        assert _normalize(v) == "NOT_OK", v
    # NOT_FEASIBLE is a detail token; it must not be auto-promoted to NOT_OK.
    assert _normalize("NOT_FEASIBLE") == "UNDETERMINED"


def test_normalize_unknown_not_run_blocked_are_undetermined():
    from results._phase0.gonogo import _normalize

    for v in ("UNKNOWN", "NOT_RUN", "BLOCKED", "", "weird-token"):
        assert _normalize(v) == "UNDETERMINED", v


def test_constants_define_routes_and_required_criteria():
    from results._phase0.gonogo import REQUIRED_CRITERIA, ROUTE_CAPABILITY_CRITERIA

    assert set(ROUTE_CAPABILITY_CRITERIA) == {
        "planar",
        "grouped",
        "region_fused",
        "cutlass_4m_single",
    }
    assert "C2" in REQUIRED_CRITERIA and "NUMERICAL" in REQUIRED_CRITERIA
    # region route depends on the region-kernel sub-criterion (truth-table rule 3)
    assert "C2_REGION_KERNEL" in ROUTE_CAPABILITY_CRITERIA["region_fused"]


def test_c2_layer_status_reads_sublayer():
    from results._phase0.gonogo import _c2_layer_status

    data = {
        "n24": {
            "layers": {
                "C2_REGION_KERNEL_FEASIBILITY": "PASS",
                "C2_CANONICAL": "UNKNOWN",
            }
        }
    }
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
    p.write_text(
        json.dumps(
            {
                "single_4m": {
                    "kernel_path": "sm80_fallback",
                    "compiles": True,
                    "runs": True,
                    "correctness": {"gate_pass": True},
                }
            }
        )
    )
    assert _cutlass_status(str(p)) == "FEASIBLE_WITH_SM80_FALLBACK"
    p.write_text(
        json.dumps(
            {
                "single_4m": {
                    "kernel_path": "sm120_native",
                    "compiles": True,
                    "runs": True,
                    "correctness": {"gate_pass": True},
                }
            }
        )
    )
    assert _cutlass_status(str(p)) == "FEASIBLE"
    p.write_text(
        json.dumps(
            {
                "single_4m": {
                    "kernel_path": "sm80_fallback",
                    "compiles": True,
                    "runs": False,
                    "correctness": {"gate_pass": False},
                }
            }
        )
    )
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
    from results._phase0.gonogo import _numerical_overall_status, _numerical_per_route

    p = tmp_path / "n.json"
    p.write_text(
        json.dumps(
            {
                "overall_numerical_status": "FAIL",
                "per_route": [
                    {"route": "planar", "criterion": "FAIL", "n_cells": 144},
                    {"route": "region_fused", "criterion": "PASS", "n_cells": 9},
                ],
            }
        )
    )
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
    p.write_text(
        json.dumps(
            {
                "per_route": [
                    {"criterion": "PASS"},  # malformed: no route
                    {"route": "planar", "criterion": "FAIL"},  # valid
                ]
            }
        )
    )
    per = _numerical_per_route(str(p))
    assert per == {"planar": "FAIL"}


def test_capability_layer_combines_per_route():
    # Task 1 contract: criteria fed to capability_layer are canonical criterion
    # tokens (PASS / FAIL / NOT_SUPPORTED / UNKNOWN / NOT_RUN). Detail tokens
    # are fail-closed to UNDETERMINED by _normalize and must not be promoted.
    from results._phase0.gonogo import capability_layer

    criteria = {
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",
        "REGION_PROTOTYPE": "PASS",
        "C2_REGION_KERNEL": "PASS",
        "CUTLASS_SM120_4M": "PASS",
    }
    cap = capability_layer(criteria)
    assert cap["planar"] == "OK"  # core OK + full matrix OK
    assert cap["grouped"] == "NOT_OK"  # NOT_SUPPORTED
    assert cap["region_fused"] == "OK"  # region proto OK + region kernel OK
    assert cap["cutlass_4m_single"] == "OK"


def test_capability_layer_undetermined_if_any_dep_not_run():
    # Task 1 contract: canonical tokens only (PASS for established-good caps,
    # NOT_RUN for not-yet-run sub-criteria). Any UNDETERMINED dep -> route
    # capability UNDETERMINED (no NOT_OK to sink it).
    from results._phase0.gonogo import capability_layer

    criteria = {
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "NOT_RUN",
        "C3_GROUPED": "NOT_SUPPORTED",
        "REGION_PROTOTYPE": "NOT_RUN",
        "C2_REGION_KERNEL": "PASS",
        "CUTLASS_SM120_4M": "NOT_RUN",
    }
    cap = capability_layer(criteria)
    assert cap["planar"] == "UNDETERMINED"  # full matrix NOT_RUN, no NOT_OK
    assert cap["region_fused"] == "UNDETERMINED"


def test_numerical_layer_maps_per_route():
    from results._phase0.gonogo import numerical_layer, ROUTES

    per = {
        "planar": "FAIL",
        "grouped": "FAIL",
        "region_fused": "PASS",
        "cutlass_4m_single": "PASS",
    }
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
        {
            "planar": "OK",
            "grouped": "NOT_OK",
            "region_fused": "OK",
            "cutlass_4m_single": "OK",
        },
        {
            "planar": "NOT_OK",
            "grouped": "NOT_OK",
            "region_fused": "OK",
            "cutlass_4m_single": "OK",
        },
    )
    assert rv["planar"]["status"] == "NOT_VIABLE"  # num NOT_OK
    assert rv["planar"]["numerical"] == "NOT_OK"
    assert rv["grouped"]["status"] == "NOT_VIABLE"  # both NOT_OK
    assert rv["region_fused"]["status"] == "VIABLE"  # both OK
    assert rv["cutlass_4m_single"]["status"] == "VIABLE"


def test_route_verdict_unknown_when_undetermined_and_no_not_ok():
    from results._phase0.gonogo import route_verdict

    rv = route_verdict(
        {
            "planar": "OK",
            "grouped": "UNDETERMINED",
            "region_fused": "OK",
            "cutlass_4m_single": "OK",
        },
        {
            "planar": "UNDETERMINED",
            "grouped": "NOT_OK",
            "region_fused": "OK",
            "cutlass_4m_single": "OK",
        },
    )
    assert rv["planar"]["status"] == "UNKNOWN"  # num UNDETERMINED, no NOT_OK
    assert rv["grouped"]["status"] == "NOT_VIABLE"  # grouped num NOT_OK


def test_route_verdict_rule3_region_kernel_fail_sinks_region():
    # rule 3 encoded structurally: region capability NOT_OK -> NOT_VIABLE
    from results._phase0.gonogo import route_verdict

    rv = route_verdict(
        {
            "planar": "OK",
            "grouped": "OK",
            "region_fused": "NOT_OK",
            "cutlass_4m_single": "OK",
        },
        {
            "planar": "OK",
            "grouped": "OK",
            "region_fused": "OK",
            "cutlass_4m_single": "OK",
        },
    )
    assert rv["region_fused"]["status"] == "NOT_VIABLE"


def test_completion_inconclusive_if_any_required_unknown():
    from results._phase0.gonogo import evaluate_completion

    criteria = {
        c: "PASS"
        for c in (
            "C1",
            "C2",
            "C3_PLANAR_CORE",
            "C3_PLANAR_FULL_MATRIX",
            "C3_GROUPED",
            "CUTLASS_SM120_4M",
            "REGION_PROTOTYPE",
            "NUMERICAL",
        )
    }
    criteria["C2"] = "UNKNOWN"  # the real binding constraint
    assert evaluate_completion(criteria) == "INCONCLUSIVE"


def test_completion_complete_when_all_determined_and_numerical_fail_ok():
    # NUMERICAL=FAIL is "determined" -> does NOT sink completion (rule 5 edge)
    from results._phase0.gonogo import evaluate_completion

    criteria = {
        c: "PASS"
        for c in (
            "C1",
            "C2",
            "C3_PLANAR_CORE",
            "C3_PLANAR_FULL_MATRIX",
            "C3_GROUPED",
            "CUTLASS_SM120_4M",
            "REGION_PROTOTYPE",
        )
    }
    criteria["NUMERICAL"] = "FAIL"
    criteria["C3_GROUPED"] = "NOT_SUPPORTED"  # determined, not UNKNOWN
    assert evaluate_completion(criteria) == "COMPLETE"


def test_completion_inconclusive_if_c3_subordinate_not_run():
    # rule 4: C3_PLANAR_CORE PASS but FULL_MATRIX NOT_RUN -> INCONCLUSIVE
    from results._phase0.gonogo import evaluate_completion

    criteria = {
        c: "PASS"
        for c in (
            "C1",
            "C2",
            "C3_PLANAR_CORE",
            "C3_PLANAR_FULL_MATRIX",
            "C3_GROUPED",
            "CUTLASS_SM120_4M",
            "REGION_PROTOTYPE",
            "NUMERICAL",
        )
    }
    criteria["C3_PLANAR_FULL_MATRIX"] = "NOT_RUN"
    assert evaluate_completion(criteria) == "INCONCLUSIVE"


def test_authorize_phase1_truth_table():
    from results._phase0.gonogo import authorize_phase1

    viable = {"region_fused": {"status": "VIABLE"}}
    none = {
        "planar": {"status": "NOT_VIABLE"},
        "grouped": {"status": "NOT_VIABLE"},
        "region_fused": {"status": "NOT_VIABLE"},
        "cutlass_4m_single": {"status": "NOT_VIABLE"},
    }
    assert authorize_phase1("COMPLETE", viable) == "GO_TO_PHASE1"
    assert authorize_phase1("COMPLETE", none) == "NO_GO"
    assert authorize_phase1("INCONCLUSIVE", viable) == "NOT_AUTHORIZED"


def test_aggregate_two_layer_end_to_end():
    from results._phase0.gonogo import aggregate_two_layer

    criteria = {
        "C1": "PASS",
        "C2": "UNKNOWN",
        "C2_REGION_KERNEL": "PASS",
        "C3_PLANAR_CORE": "SUPPORTED",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",
        "CUTLASS_SM120_4M": "FEASIBLE_WITH_SM80_FALLBACK",
        "REGION_PROTOTYPE": "FEASIBLE_WITH_RECOMPUTE",
        "NUMERICAL": "FAIL",
    }
    rv = {
        "planar": {"status": "NOT_VIABLE", "capability": "OK", "numerical": "NOT_OK"},
        "grouped": {
            "status": "NOT_VIABLE",
            "capability": "NOT_OK",
            "numerical": "NOT_OK",
        },
        "region_fused": {"status": "VIABLE", "capability": "OK", "numerical": "OK"},
        "cutlass_4m_single": {
            "status": "VIABLE",
            "capability": "OK",
            "numerical": "OK",
        },
    }
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

    agg = aggregate_two_layer(
        {"NUMERICAL": "FAIL"}, {}, "INCONCLUSIVE", "NOT_AUTHORIZED"
    )
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
    for name in (
        "c1_judgment.json",
        "c2_judgment.json",
        "cublaslt_planar_capability.json",
        "cublaslt_grouped_capability.json",
        "cublaslt_full_matrix.csv",
        "cutlass_sm120_4m.json",
        "region_prototype.json",
        "numerical_validation.json",
        "cublaslt_gap.txt",
    ):
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
    # Task 1 fail-closed: REGION_PROTOTYPE reads the canonical
    # region_prototype.json verdict=FEASIBLE_WITH_RECOMPUTE (a DETAIL token, not
    # canonical PASS) and CUTLASS_SM120_4M reads FEASIBLE_WITH_SM80_FALLBACK
    # (also a detail token). With startswith("FEASIBLE") promotion killed in
    # _normalize, region_fused capability -> UNDETERMINED -> route UNKNOWN
    # (was previously VIABLE, the fail-open surface Task 1 closes; Tasks 2a/4
    # will re-derive canonical PASS upstream at the reader level).
    assert agg["route_verdict"]["region_fused"]["status"] == "UNKNOWN"
    assert agg["route_verdict"]["planar"]["status"] == "NOT_VIABLE"
    # rule 7: MD rendered from same object
    md = (stage / "gonogo.md").read_text()
    assert agg["phase0_completion"] in md and agg["phase1_authorization"] in md
    # Task 11 handoff: gonogo.main() no longer writes manifest.json (manifest.py owns it)
    assert not (stage / "manifest.json").exists()


def test_gonogo_main_does_not_write_manifest(tmp_path, monkeypatch):
    import os, shutil
    from results._phase0 import gonogo as G

    src = "results/phase0"
    stage = tmp_path / "phase0"
    stage.mkdir()
    for name in (
        "c1_judgment.json",
        "c2_judgment.json",
        "cublaslt_planar_capability.json",
        "cublaslt_grouped_capability.json",
        "cublaslt_full_matrix.csv",
        "cutlass_sm120_4m.json",
        "region_prototype.json",
        "numerical_validation.json",
    ):
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy(s, stage / name)
    monkeypatch.setattr(G, "_collect_environment", lambda: {"_stub": True})
    G.main(stage_dir=str(stage))
    assert not (stage / "manifest.json").exists()  # handoff to manifest.py
    assert (stage / "gonogo.json").exists()  # gonogo still writes these
    assert (stage / "environment.json").exists()


def test_capability_layer_region_kernel_fail_sinks_region():
    # truth-table rule 3, structurally: region_fused depends on C2_REGION_KERNEL;
    # a canonical FAIL there sinks region capability to NOT_OK even with a PASS
    # region prototype. (Task 1: criteria fed in are canonical tokens.)
    from results._phase0.gonogo import capability_layer

    criteria = {
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",
        "REGION_PROTOTYPE": "PASS",
        "C2_REGION_KERNEL": "FAIL",
        "CUTLASS_SM120_4M": "PASS",
    }
    cap = capability_layer(criteria)
    assert cap["region_fused"] == "NOT_OK"  # rule 3: region-kernel FAIL sinks region


# ---------------------------------------------------------------------------
# Task 0 (SDD plan §3 操作.2): fail-closed RED baseline. The tests below freeze
# the target behavior the gonogo gate must adopt after Tasks 5/6/7 wire the
# canonical verdict_schema in. They FAIL on the current implementation by clean
# assertion (not import, not GPU).
# ---------------------------------------------------------------------------


def test_normalize_does_not_promote_feasible_detail_tokens_to_ok():
    """plan §4 验收: 不再使用 ``startswith('FEASIBLE')`` 无条件提升全部架构/route.

    The current ``_normalize`` does ``verdict.startswith('FEASIBLE') -> OK``
    (gonogo.py), which promotes any FEASIBLE* detail token to capability-OK in
    the canonical criterion layer. Per the fail-closed model those tokens are
    DETAIL tokens (verdict_schema.DETAIL_TOKENS): a canonical criterion field
    carrying them must fail closed to UNKNOWN, not be promoted to OK. This test
    freezes the target: FEASIBLE* / TILE_FUSION_FEASIBLE / SUPPORTED / BLOCKED
    must NOT normalize to OK in the canonical-criterion pipeline."""
    from results._phase0.gonogo import _normalize
    from results._phase0.verdict_schema import normalize_criterion

    # The canonical criterion scrubber (verdict_schema) is the source of truth.
    for t in (
        "FEASIBLE",
        "FEASIBLE_WITH_RECOMPUTE",
        "FEASIBLE_WITH_SM80_FALLBACK",
        "TILE_FUSION_FEASIBLE",
        "SUPPORTED",
        "BLOCKED",
    ):
        # canonical-criterion-pipeline contract: detail tokens -> UNKNOWN (PASS
        # must be re-derived from evidence, not promoted from the detail prefix).
        assert normalize_criterion(t) == "UNKNOWN", t

    # The gonogo route/capability layer (_normalize) currently promotes
    # FEASIBLE* to OK. The canonical criteria feeding it must already be
    # canonical tokens by the time they reach _normalize, so _normalize should
    # never SEE a FEASIBLE* token. This asserts the contract: _normalize does
    # not get to promote detail tokens; the input is canonicalized upstream.
    # (Fails today: _normalize('FEASIBLE_WITH_RECOMPUTE') == 'OK'.)
    for t in ("FEASIBLE_WITH_RECOMPUTE", "FEASIBLE_WITH_SM80_FALLBACK"):
        assert (
            _normalize(t) != "OK"
        ), f"_normalize must not promote detail token {t!r} to OK"


def test_main_emits_two_cutlass_criteria_for_native_blocker_plus_sm80_fallback(
    tmp_path, monkeypatch
):
    """plan §3 操作.2 bullet 8: a cutlass artifact recording BOTH a native SM120
    blocker AND a working SM80 fallback must surface TWO DISTINCT criteria --
    ``CUTLASS_SM120_4M`` (native SM120 -> FAIL / UNKNOWN, never PASS) and
    ``CUTLASS_SM80_FALLBACK_CAPABILITY`` (fallback success -> PASS).

    Today ``_cutlass_status`` merges both outcomes into one
    ``FEASIBLE_WITH_SM80_FALLBACK`` criterion (gonogo.py), so the native SM120
    blocker is invisible behind the fallback's success -- exactly the
    information loss plan §3 操作.2 bullet 8 forbids. This test drives ``main``
    against a synthetic cutlass artifact that records both outcomes and asserts
    the emitted criteria dict carries BOTH canonical criterion keys."""
    import json
    from results._phase0 import gonogo as G

    # Synthetic cutlass artifact: native SM120 blocked + SM80 fallback works.
    (tmp_path / "cutlass_sm120_4m.json").write_text(
        json.dumps(
            {
                "overall": "FEASIBLE_WITH_SM80_FALLBACK",
                "single_4m": {
                    "kernel_path": "sm80_fallback",
                    "compiles": True,
                    "runs": True,
                    "correctness": {"gate_pass": True},
                    "native_sm120_blocker": "F8F6F4 static_assert (BF16 blocked)",
                },
            }
        )
    )
    # Minimal supporting artifacts so main() proceeds without raising.
    (tmp_path / "c1_judgment.json").write_text(
        json.dumps({"n24_d10": {"judgment": {"status": "PASS"}}})
    )
    (tmp_path / "c2_judgment.json").write_text("{}")
    (tmp_path / "cublaslt_planar_capability.json").write_text("{}")
    (tmp_path / "cublaslt_full_matrix.csv").write_text("h\n1\n")
    (tmp_path / "cublaslt_grouped_capability.json").write_text("{}")
    (tmp_path / "region_prototype.json").write_text("{}")
    (tmp_path / "numerical_validation.json").write_text("{}")
    monkeypatch.setattr(G, "_collect_environment", lambda: {"_stub": True})

    G.main(stage_dir=str(tmp_path))
    agg = json.load(open(tmp_path / "gonogo.json"))
    criteria = agg["criteria"]

    # BOTH canonical criteria must be present (today only CUTLASS_SM120_4M).
    assert "CUTLASS_SM120_4M" in criteria, criteria
    assert "CUTLASS_SM80_FALLBACK_CAPABILITY" in criteria, (
        "native SM120 blocker + SM80 fallback success must surface as TWO "
        "distinct criteria, not a single merged criterion; got: " + str(criteria)
    )
    # native SM120 is BLOCKED -> canonical criterion must NOT be PASS
    assert criteria["CUTLASS_SM120_4M"] != "PASS", criteria
    # SM80 fallback succeeds -> canonical criterion is PASS
    assert criteria["CUTLASS_SM80_FALLBACK_CAPABILITY"] == "PASS", criteria


def test_c3_full_matrix_unknown_on_missing_expected_cell(tmp_path):
    """plan §3 操作.2 bullet 9: a full-matrix CSV missing an expected cell ->
    criterion UNKNOWN. Today ``_c3_planar_full_matrix_status`` only requires
    '>=1 data row' (gonogo.py), so any non-empty CSV returns PASS regardless of
    coverage. This test freezes the target: a CSV with only a subset of the
    expected matrix cells yields UNKNOWN."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    # A CSV with ONE data row for an unexpected shape -> does NOT cover the
    # full matrix (which spans the canonical SHAPES, e.g. 16384x1024x1024 +
    # 524288x32x32 + 262144x64x64 + 1048576x16x16 + ...). One subset row
    # cannot be a complete matrix -> UNKNOWN.
    p = tmp_path / "fm.csv"
    p.write_text("M,N,K,status\n1024,1024,1024,ok\n")
    assert _c3_planar_full_matrix_status(str(p)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_duplicate_row(tmp_path):
    """plan §3 操作.2 bullet 9: a full-matrix CSV with a duplicate row ->
    UNKNOWN. Duplicates indicate a broken sweep / re-run contamination, not a
    canonical PASS."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    p = tmp_path / "fm.csv"
    # same shape row twice -> duplicate contamination
    p.write_text("M,N,K,status\n1024,1024,1024,ok\n1024,1024,1024,ok\n")
    assert _c3_planar_full_matrix_status(str(p)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_shape_drift(tmp_path):
    """plan §3 操作.2 bullet 9: a full-matrix CSV with a row whose (M,N,K) is
    OUTSIDE the expected matrix (shape drift) -> UNKNOWN. Today
    ``_c3_planar_full_matrix_status`` accepts any non-empty CSV; the fix
    validates that every row's shape is in the canonical matrix."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    p = tmp_path / "fm.csv"
    # 9999x9999x9999 is not any canonical matrix shape -> drift
    p.write_text("M,N,K,status\n1024,1024,1024,ok\n9999,9999,9999,drift\n")
    assert _c3_planar_full_matrix_status(str(p)) == "UNKNOWN"


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
