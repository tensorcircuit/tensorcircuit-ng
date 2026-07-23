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
    """Task 5 strict validator: the committed 128-cell artifact -> PASS;
    missing artifact -> NOT_RUN; empty/header-only -> UNKNOWN.

    The validator derives the expected 128-cell contract from
    contraction_shapes.csv + the producer matrix grid, so the real artifact
    (cublaslt_full_matrix.csv + contraction_shapes.csv) is the only honest
    PASS fixture. Pure-function (no GPU); tests under this name do not load
    the extension."""
    import os

    from results._phase0.gonogo import _c3_planar_full_matrix_status

    real_csv = "results/phase0/cublaslt_full_matrix.csv"
    if os.path.exists(real_csv):
        # The committed 128-cell artifact (120 ok + 8 policy no-algo) -> PASS.
        assert _c3_planar_full_matrix_status(real_csv) == "PASS"
    # absent artifact -> NOT_RUN (Plan B / Task 6 not yet run)
    assert _c3_planar_full_matrix_status(str(tmp_path / "missing.csv")) == "NOT_RUN"
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "M,N,K,out_dtype,ws_cap,op,aligned,algo_count,first_algo_id,workspace_bytes,status\n"
    )
    assert _c3_planar_full_matrix_status(str(empty)) == "UNKNOWN"


def test_c3_full_matrix_pass_on_synthetic_complete_matrix(tmp_path):
    """Task 5: a synthetic COMPLETE matrix over a tiny shape set -> PASS.
    Confirms the validator is a pure-function (no GPU, no real artifacts):
    given a contraction_shapes.csv whose bytes>=64MiB rows yield N distinct
    shapes, a full-matrix CSV covering all N*16 cells (with no-algo only on
    the explicit policy shape) is PASS."""
    import csv

    from results._phase0.cublaslt import _FULL_MATRIX_HEADER, full_matrix_no_algo_policy
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    # 2 distinct actual-large shapes (>=64 MiB). (262144,64,4) is the policy
    # shape whose OP_T cells legitimately enumerate zero algos; (16384,16,16)
    # is a fully-aligned real-gemm shape where every cell has an algo.
    shapes_csv = tmp_path / "contraction_shapes.csv"
    shapes_csv.write_text(
        "n,depth,output,node_id,M,N,K,bytes\n"
        "24,10,state,0,262144,64,4,134217728\n"
        "24,10,state,1,16384,16,16,134217728\n"
    )
    policy = full_matrix_no_algo_policy()
    rows = []
    for m, n, k, od, ws, op in [
        (262144, 64, 4, od, ws, op)
        for od in ("bf16", "fp32")
        for ws in ("0", "1MiB", "16MiB", "max")
        for op in ("N", "T")
    ] + [
        (16384, 16, 16, od, ws, op)
        for od in ("bf16", "fp32")
        for ws in ("0", "1MiB", "16MiB", "max")
        for op in ("N", "T")
    ]:
        key = (m, n, k, od, ws, op)
        aligned = int(m % 16 == 0 and n % 16 == 0 and k % 16 == 0)
        status = "no-algo" if key in policy else "ok"
        rows.append([m, n, k, od, ws, op, aligned, 1, 21, 0, status])

    fm = tmp_path / "fm.csv"
    with open(fm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FULL_MATRIX_HEADER)
        w.writerows(rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "PASS"


def test_c3_grouped_status(tmp_path):
    import json
    from results._phase0.gonogo import _c3_grouped_status

    p = tmp_path / "g.json"
    p.write_text(json.dumps({"capability": {"status": "NOT_SUPPORTED"}}))
    assert _c3_grouped_status(str(p)) == "NOT_SUPPORTED"
    p.write_text(json.dumps({"capability": {"status": "SUPPORTED"}}))
    assert _c3_grouped_status(str(p)) == "SUPPORTED"
    assert _c3_grouped_status(str(tmp_path / "missing.json")) == "NOT_RUN"


def test_cutlass_status_derives_two_independent_criteria(tmp_path):
    """plan §7 Task 4: ``_cutlass_status`` returns TWO INDEPENDENT canonical
    criteria (``CUTLASS_SM120_4M`` native + ``CUTLASS_SM80_FALLBACK_CAPABILITY``
    fallback), never a single merged token. Native failure and fallback
    success coexist without contradiction."""
    import json
    from results._phase0.gonogo import _cutlass_status

    p = tmp_path / "c.json"
    # sm80 fallback works + native sm120 blocker recorded verbatim
    p.write_text(
        json.dumps(
            {
                "single_4m": {
                    "kernel_path": "sm80_fallback",
                    "compiles": True,
                    "runs": True,
                    "correctness": {"gate_pass": True},
                    "sm120_blocker": "F8F6F4 static_assert (BF16 blocked)",
                }
            }
        )
    )
    c = _cutlass_status(str(p))
    assert c["CUTLASS_SM120_4M"] == "NOT_SUPPORTED", c
    assert c["CUTLASS_SM80_FALLBACK_CAPABILITY"] == "PASS", c
    # The two criteria are INDEPENDENT — one is NOT_SUPPORTED, the other PASS.
    assert c["CUTLASS_SM120_4M"] != c["CUTLASS_SM80_FALLBACK_CAPABILITY"], c

    # Theoretical future: native sm120 path actually landed + passed (no fallback).
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
    c = _cutlass_status(str(p))
    assert c["CUTLASS_SM120_4M"] == "PASS", c
    # No sm80_fallback info -> fallback criterion is UNKNOWN (capability NOT
    # derived from native success).
    assert c["CUTLASS_SM80_FALLBACK_CAPABILITY"] == "UNKNOWN", c

    # sm80 fallback that failed correctness -> fallback FAIL; native NOT_SUPPORTED
    # (the artifact documents landing on the sm80 fallback, so native did not
    # land — independent of whether the fallback itself later passed).
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
    c = _cutlass_status(str(p))
    assert c["CUTLASS_SM80_FALLBACK_CAPABILITY"] == "FAIL", c
    assert c["CUTLASS_SM120_4M"] == "NOT_SUPPORTED", c

    # Missing artifact -> both criteria NOT_RUN
    c = _cutlass_status(str(tmp_path / "missing.json"))
    assert c == {
        "CUTLASS_SM120_4M": "NOT_RUN",
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "NOT_RUN",
    }, c


def test_cutlass_status_reads_new_two_section_structure(tmp_path):
    """plan §7 Task 4: ``_cutlass_status`` reads the regenerated two-section
    artifact (native_sm120_bf16_4m + sm80_fallback_bf16_4m) directly —
    preferred over the legacy single_4m block."""
    import json
    from results._phase0.gonogo import _cutlass_status

    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "native_sm120_bf16_4m": {
                    "capability": "NOT_SUPPORTED",
                    "compile_status": "BLOCKED",
                    "blocker": "F8F6F4 static_assert",
                },
                "sm80_fallback_bf16_4m": {
                    "capability": "PASS",
                    "correctness": {"gate_pass": True},
                },
            }
        )
    )
    c = _cutlass_status(str(p))
    assert c == {
        "CUTLASS_SM120_4M": "NOT_SUPPORTED",
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
    }, c


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
    # Task 4: cutlass_4m_single now depends on CUTLASS_SM80_FALLBACK_CAPABILITY
    # (the path that actually runs), NOT on CUTLASS_SM120_4M (native, BLOCKED).
    from results._phase0.gonogo import capability_layer

    criteria = {
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",
        "REGION_PROTOTYPE": "PASS",
        "C2_REGION_KERNEL": "PASS",
        "CUTLASS_SM120_4M": "NOT_SUPPORTED",  # native BLOCKED
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",  # fallback runs
    }
    cap = capability_layer(criteria)
    assert cap["planar"] == "OK"  # core OK + full matrix OK
    assert cap["grouped"] == "NOT_OK"  # NOT_SUPPORTED
    assert cap["region_fused"] == "OK"  # region proto OK + region kernel OK
    # cutlass_4m_single capability follows the FALLBACK (PASS), independent of
    # CUTLASS_SM120_4M being NOT_SUPPORTED — native failure does not sink the
    # route that actually runs.
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
            "CUTLASS_SM80_FALLBACK_CAPABILITY",
            "REGION_PROTOTYPE",
        )
    }
    criteria["NUMERICAL"] = "FAIL"
    criteria["C3_GROUPED"] = "NOT_SUPPORTED"  # determined, not UNKNOWN
    criteria["CUTLASS_SM120_4M"] = "NOT_SUPPORTED"  # determined, not UNKNOWN
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
        "contraction_shapes.csv",
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
        "contraction_shapes.csv",
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


def _write_synthetic_shapes(tmp_path, shapes):
    """Write a contraction_shapes.csv with the given (M,N,K) shapes all marked
    bytes>=64MiB so load_c1_c2_shapes keeps them. Returns the path."""
    shapes_csv = tmp_path / "contraction_shapes.csv"
    lines = ["n,depth,output,node_id,M,N,K,bytes"]
    for i, (m, n, k) in enumerate(shapes):
        lines.append(f"24,10,state,{i},{m},{n},{k},134217728")
    shapes_csv.write_text("\n".join(lines) + "\n")
    return shapes_csv


def _write_full_matrix(tmp_path, rows, name="fm.csv"):
    """Write a full-matrix CSV with the canonical header + given rows."""
    import csv
    from results._phase0.cublaslt import _FULL_MATRIX_HEADER

    p = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FULL_MATRIX_HEADER)
        w.writerows(rows)
    return p


def _synth_complete_rows(shapes):
    """Build the canonical complete-matrix rows for the given (M,N,K) shape list,
    with no-algo ONLY on the explicit-policy shape's OP_T cells."""
    from results._phase0.cublaslt import full_matrix_no_algo_policy

    policy = full_matrix_no_algo_policy()
    rows = []
    for m, n, k in shapes:
        for od in ("bf16", "fp32"):
            for ws in ("0", "1MiB", "16MiB", "max"):
                for op in ("N", "T"):
                    key = (m, n, k, od, ws, op)
                    aligned = int(m % 16 == 0 and n % 16 == 0 and k % 16 == 0)
                    status = "no-algo" if key in policy else "ok"
                    rows.append([m, n, k, od, ws, op, aligned, 1, 21, 0, status])
    return rows


def test_c3_full_matrix_unknown_on_missing_expected_cell(tmp_path):
    """plan §3 操作.2 bullet 9: a full-matrix CSV with proper schema but only a
    SUBSET of the expected cells -> UNKNOWN (missing coverage). The validator
    derives the expected 128-cell contract per shape; a single row cannot cover
    the full matrix."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(262144, 64, 4), (16384, 16, 16)]
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    # Build the complete matrix, then keep only ONE row -> missing coverage.
    one_row = [_synth_complete_rows(shapes)[0]]
    fm = _write_full_matrix(tmp_path, one_row)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_duplicate_row(tmp_path):
    """plan §3 操作.2 bullet 9: a full-matrix CSV with a duplicate cell key ->
    UNKNOWN. Duplicates indicate a broken sweep / re-run contamination."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]  # one shape: 16 expected cells
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    # Duplicate the first row -> 17 rows, one cell key appears twice.
    rows.append(rows[0])
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_shape_drift(tmp_path):
    """plan §3 操作.2 bullet 9: a full-matrix CSV containing a row whose (M,N,K)
    is OUTSIDE the expected matrix (shape drift) -> UNKNOWN. The validator
    binds every cell's shape to contraction_shapes.csv."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    # 9999x9999x9999 is not any canonical contraction shape -> drift.
    rows.append([9999, 9999, 9999, "bf16", "0", "N", 0, 1, 21, 0, "ok"])
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_illegal_status_token(tmp_path):
    """plan §3 操作.2 bullet 9: a full-matrix CSV with a status token outside
    the producer's legal set ('ok'/'no-algo') -> UNKNOWN. An unknown status is
    a sweep/parser regression, not a canonical PASS."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    # 'error' is not a producer status token -> illegal.
    rows[0][10] = "error"
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_no_algo_outside_policy(tmp_path):
    """Task 5 explicit no-algo policy: a no-algo cell OUTSIDE the 8-cell policy
    set -> UNKNOWN. The 8 legitimate no-algo cells are all OP_T on shape
    (262144,64,4); a no-algo on (16384,16,16) (a fully-aligned real-gemm shape)
    is a real coverage gap, not a PASS."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]  # not the policy shape
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    # Flip one (16384,16,16) cell to no-algo -> outside the policy -> UNKNOWN.
    rows[0][10] = "no-algo"
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
