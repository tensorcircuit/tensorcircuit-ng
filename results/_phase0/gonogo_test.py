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


# Task 7: the gonogo-local truth-table duplicate (_normalize, _combine_tri,
# capability_layer, numerical_layer, route_verdict, evaluate_completion,
# authorize_phase1, REQUIRED_CRITERIA, ROUTE_CAPABILITY_CRITERIA) has been
# KILLED. The §5 truth table now lives solely in verdict_schema
# (recompute_derived_state, tested by verdict_schema_test.py). gonogo is the
# CRITERIA PRODUCER; derivation goes through the shared helper. The tests that
# pinned the gonogo-local duplicate are removed; the shared helper's contract
# is pinned in verdict_schema_test.py.


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
        if key in policy:
            # no-algo rows carry the producer's zero-algo sentinel values.
            rows.append([m, n, k, od, ws, op, aligned, 0, -1, 0, "no-algo"])
        else:
            rows.append([m, n, k, od, ws, op, aligned, 1, 21, 0, "ok"])

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
    # Task 4 (nongpu-rereview §3.5.1): a bare SUPPORTED with no backing API/run
    # evidence is an unconfirmable positive claim -> UNKNOWN (NOT the raw
    # SUPPORTED detail token, which tri_normalize would also downgrade).
    p.write_text(json.dumps({"capability": {"status": "SUPPORTED"}}))
    assert _c3_grouped_status(str(p)) == "UNKNOWN"
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
                    "kernel_path": "sm80_fallback",
                    "runs": True,
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
    # Task 4 (nongpu-rereview §3.5.2): artifact-native detail tokens are NEVER
    # returned directly. FEASIBLE_WITH_RECOMPUTE without full-anchor evidence
    # -> UNKNOWN (the detail token would be downgraded anyway).
    p.write_text(json.dumps({"verdict": "FEASIBLE_WITH_RECOMPUTE"}))
    assert _region_proto_status(str(p)) == "UNKNOWN"
    # NOT_FEASIBLE is a definitive negative -> canonical FAIL.
    p.write_text(json.dumps({"verdict": "NOT_FEASIBLE"}))
    assert _region_proto_status(str(p)) == "FAIL"
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


def test_aggregate_two_layer_uses_shared_helper():
    """Task 7: aggregate_two_layer delegates to verdict_schema.recompute_derived_state
    (the shared §5 truth table). The derived state (route_verdict / completion /
    authorization / reasons / blocking) is RECOMPUTED, not passed in."""
    from results._phase0.gonogo import aggregate_two_layer
    from results._phase0.verdict_schema import recompute_derived_state

    criteria = {
        "C1": "PASS",
        "C2": "UNKNOWN",
        "C2_REGION_KERNEL_FEASIBILITY": "PASS",
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",
        "CUTLASS_SM120_4M": "NOT_SUPPORTED",
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
        "REGION_PROTOTYPE": "UNKNOWN",
        "NUMERICAL": "FAIL",
    }
    per_route = {"planar": "FAIL", "grouped": "FAIL"}
    agg = aggregate_two_layer(criteria, per_route)
    # The derived state matches the shared helper exactly (no divergence).
    expected = recompute_derived_state(criteria, per_route)
    assert agg["route_verdict"] == expected["route_verdict"]
    assert agg["phase0_completion"] == expected["phase0_completion"]
    assert agg["phase1_authorization"] == expected["phase1_authorization"]
    assert agg["reasons"] == expected["reasons"]
    assert agg["blocking_artifacts"] == expected["blocking_artifacts"]
    # Honest headline: C2 canonical undetermined -> INCONCLUSIVE -> NOT_AUTHORIZED.
    assert agg["schema_version"] == "gonogo-v2"
    assert agg["phase0_completion"] == "INCONCLUSIVE"
    assert agg["phase1_authorization"] == "NOT_AUTHORIZED"
    # Task 1: C2 compat alias tracks C2_CANONICAL (not in REQUIRED_CRITERIA /
    # gates). The alias is set by validate_criteria after computing the rollup.
    assert agg["criteria"]["C2"] == agg["criteria"]["C2_CANONICAL"]
    # planar: capability OK (C3 core+full PASS) but numerical FAIL -> NOT_VIABLE.
    assert agg["route_verdict"]["planar"]["status"] == "NOT_VIABLE"
    # region_fused: REGION_PROTOTYPE UNKNOWN -> capability UNDETERMINED -> UNKNOWN.
    assert agg["route_verdict"]["region_fused"]["status"] == "UNKNOWN"
    # reasons name the undetermined criterion.
    assert any("C2" in r for r in agg["reasons"])
    assert "c2_judgment.json" in " ".join(agg["blocking_artifacts"])


def test_render_md_matches_json_object():
    """truth-table rule 7: MD is generated from the same object -> no contradiction.
    Task 7: aggregate_two_layer uses the new (criteria, per_route_numerical) signature.
    """
    from results._phase0.gonogo import aggregate_two_layer, _render_md

    agg = aggregate_two_layer({"NUMERICAL": "FAIL"}, {})
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
    # Task 7: gonogo emits canonical CRITERIA_NAMES keys (NOT the abbreviated
    # C2_REGION_KERNEL). The region prototype verdict=UNKNOWN (canonical, from
    # region_prototype.json) and C2_REGION_KERNEL_FEASIBILITY=UNKNOWN (from
    # c2_judgment layers) -> region_fused capability UNDETERMINED -> route
    # UNKNOWN. CUTLASS_SM120_4M=NOT_SUPPORTED + CUTLASS_SM80_FALLBACK_CAPABILITY
    # =PASS (split criteria, Task 4); cutlass_4m_single capability follows the
    # fallback (PASS) but numerical UNKNOWN -> route UNKNOWN.
    assert "C2_REGION_KERNEL_FEASIBILITY" in agg["criteria"]
    assert "C2_REGION_KERNEL" not in agg["criteria"]
    assert "CUTLASS_SM120_4M" in agg["criteria"]
    assert "CUTLASS_SM80_FALLBACK_CAPABILITY" in agg["criteria"]
    assert agg["route_verdict"]["region_fused"]["status"] == "UNKNOWN"
    assert agg["route_verdict"]["planar"]["status"] == "NOT_VIABLE"
    assert agg["route_verdict"]["grouped"]["status"] == "NOT_VIABLE"
    assert agg["route_verdict"]["cutlass_4m_single"]["status"] == "UNKNOWN"
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


def test_normalize_does_not_promote_feasible_detail_tokens_to_ok():
    """plan §4 验收: 不再使用 ``startswith('FEASIBLE')`` 无条件提升全部架构/route.

    Task 7: gonogo's local _normalize duplicate has been KILLED. The canonical
    criterion scrubber (verdict_schema.normalize_criterion) is the single source
    of truth. FEASIBLE* / TILE_FUSION_FEASIBLE / SUPPORTED / BLOCKED detail tokens
    must fail closed to UNKNOWN (PASS must be re-derived from evidence), never
    promoted to OK. The truth table (recompute_derived_state) consumes criteria
    that have already been canonicalized by the criterion producers."""
    from results._phase0.verdict_schema import normalize_criterion

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
    with no-algo ONLY on the explicit-policy shape's OP_T cells.

    Algorithm columns match the producer (cublaslt.run_full_matrix): ok rows
    carry algo_count=1/first_algo_id=21/workspace_bytes=0, no-algo rows carry
    algo_count=0/first_algo_id=-1/workspace_bytes=0."""
    from results._phase0.cublaslt import full_matrix_no_algo_policy

    policy = full_matrix_no_algo_policy()
    rows = []
    for m, n, k in shapes:
        for od in ("bf16", "fp32"):
            for ws in ("0", "1MiB", "16MiB", "max"):
                for op in ("N", "T"):
                    key = (m, n, k, od, ws, op)
                    aligned = int(m % 16 == 0 and n % 16 == 0 and k % 16 == 0)
                    if key in policy:
                        rows.append([m, n, k, od, ws, op, aligned, 0, -1, 0, "no-algo"])
                    else:
                        rows.append([m, n, k, od, ws, op, aligned, 1, 21, 0, "ok"])
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
    # Flip one (16384,16,16) cell to no-algo with producer-consistent algo
    # columns (algo_count=0, first_algo_id=-1) so the row reaches the no-algo
    # POLICY check rather than tripping the algo-consistency check first;
    # (16384,16,16) is outside the 8-cell policy set -> UNKNOWN.
    rows[0][7] = 0  # algo_count
    rows[0][8] = -1  # first_algo_id
    rows[0][10] = "no-algo"
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Task 5 algorithm-column legality (review Important): each test mutates
# exactly ONE algorithm field on a valid-baseline CSV and asserts UNKNOWN.
# algo_count / first_algo_id / workspace_bytes must be integers, in range, and
# consistent with the row's status (ok<->algo_count>=1; no-algo<->algo_count==0
# + first_algo_id==-1). Any violation is fail-closed -> UNKNOWN.
# ---------------------------------------------------------------------------


def test_c3_full_matrix_unknown_on_ok_with_zero_algo_count(tmp_path):
    """status='ok' but algo_count=0 is contradictory ('ok' means >=1 algorithm
    was found) -> UNKNOWN."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]  # all-ok shape
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    rows[0][7] = 0  # algo_count=0 contradicts status="ok"
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_non_integer_algo_count(tmp_path):
    """A non-integer algo_count ('x') fails to parse -> UNKNOWN."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    rows[0][7] = "x"  # non-integer algo_count
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_negative_algo_count(tmp_path):
    """A negative algo_count is out of range -> UNKNOWN."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    rows[0][7] = -1  # negative algo_count
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_no_algo_with_nonzero_algo_count(tmp_path):
    """status='no-algo' but algo_count=2 is contradictory (no-algo must be
    zero-algo) -> UNKNOWN."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(262144, 64, 4)]  # policy shape -> has legitimate no-algo cells
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    idx = next(i for i, r in enumerate(rows) if r[10] == "no-algo")
    rows[idx][7] = 2  # algo_count=2 contradicts status="no-algo"
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_no_algo_with_non_sentinel_algo_id(tmp_path):
    """status='no-algo' but first_algo_id=5 (not the -1 sentinel) is
    contradictory -> UNKNOWN."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(262144, 64, 4)]  # policy shape -> has legitimate no-algo cells
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    idx = next(i for i, r in enumerate(rows) if r[10] == "no-algo")
    rows[idx][8] = 5  # first_algo_id=5 contradicts status="no-algo" (must be -1)
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_header_drift(tmp_path):
    """Task 5 (review Minor): a CSV whose header row != the canonical
    _FULL_MATRIX_HEADER -> UNKNOWN (schema/header drift). The check existed but
    had no dedicated test."""
    import csv

    from results._phase0.cublaslt import _FULL_MATRIX_HEADER
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    # Tamper the header: rename 'algo_count' -> 'algo_cnt' (non-canonical).
    drifted_header = list(_FULL_MATRIX_HEADER)
    drifted_header[7] = "algo_cnt"
    p = tmp_path / "fm.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(drifted_header)
        w.writerows(rows)
    assert _c3_planar_full_matrix_status(str(p), str(shapes_csv)) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Task 7: gonogo↔manifest alignment + canonical keys + honest state + JSON==MD.
# These tests pin the Task 7 contract: gonogo reuses the shared §5 truth table
# (recompute_derived_state), emits canonical CRITERIA_NAMES keys, and the
# regenerated gonogo.json AGREES with manifest.json on route verdicts +
# completion + authorization.
# ---------------------------------------------------------------------------


def test_gonogo_emits_canonical_criteria_keys(tmp_path, monkeypatch):
    """Task 7: gonogo emits the verdict_schema.CRITERIA_NAMES keys -- in
    particular C2_REGION_KERNEL_FEASIBILITY (NOT the abbreviated C2_REGION_KERNEL)
    and BOTH cutlass criteria (CUTLASS_SM120_4M + CUTLASS_SM80_FALLBACK_CAPABILITY)."""
    import json, os, shutil
    from results._phase0 import gonogo as G
    from results._phase0.verdict_schema import CRITERIA_NAMES

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
    agg = json.load(open(stage / "gonogo.json"))
    criteria = agg["criteria"]
    # The abbreviated key must NOT appear.
    assert "C2_REGION_KERNEL" not in criteria, criteria
    # The canonical key must appear.
    assert "C2_REGION_KERNEL_FEASIBILITY" in criteria, criteria
    # Both cutlass criteria must be present (Task 4 split).
    assert "CUTLASS_SM120_4M" in criteria, criteria
    assert "CUTLASS_SM80_FALLBACK_CAPABILITY" in criteria, criteria
    # Every CRITERIA_NAMES key that gonogo is responsible for producing is
    # present (C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK / C2_JOINT_EXECUTABLE_LEVERAGE
    # / C2_CANONICAL are sub-layers consumed by the C2 roll-up, not top-level
    # gonogo criteria; the top-level criteria are the ones main() emits).
    for key in (
        "C1",
        "C2",
        "C2_REGION_KERNEL_FEASIBILITY",
        "C3_PLANAR_CORE",
        "C3_PLANAR_FULL_MATRIX",
        "C3_GROUPED",
        "CUTLASS_SM120_4M",
        "CUTLASS_SM80_FALLBACK_CAPABILITY",
        "REGION_PROTOTYPE",
        "NUMERICAL",
    ):
        assert key in criteria, f"missing canonical criterion key: {key}"
    # Sanity: the canonical key set is a subset of CRITERIA_NAMES + the
    # top-level roll-up keys (C2, C3_*, REGION_PROTOTYPE, NUMERICAL) that
    # gonogo emits as criterion-producer outputs.
    assert "C2_REGION_KERNEL_FEASIBILITY" in CRITERIA_NAMES


def test_gonogo_json_matches_expected_honest_state(tmp_path, monkeypatch):
    """Task 7 plan §10: the regenerated gonogo.json must match the expected
    honest state (no pre-written PASS). region_fused / cutlass_4m_single are
    UNKNOWN (not yet measured); planar / grouped are NOT_VIABLE; completion
    INCONCLUSIVE; authorization NOT_AUTHORIZED."""
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
    ):
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy(s, stage / name)
    monkeypatch.setattr(G, "_collect_environment", lambda: {"_stub": True})
    G.main(stage_dir=str(stage))
    agg = json.load(open(stage / "gonogo.json"))
    rv = agg["route_verdict"]
    assert rv["planar"]["status"] == "NOT_VIABLE", rv["planar"]
    assert rv["grouped"]["status"] == "NOT_VIABLE", rv["grouped"]
    assert rv["region_fused"]["status"] == "UNKNOWN", rv["region_fused"]
    assert rv["cutlass_4m_single"]["status"] == "UNKNOWN", rv["cutlass_4m_single"]
    assert agg["phase0_completion"] == "INCONCLUSIVE"
    assert agg["phase1_authorization"] == "NOT_AUTHORIZED"
    # No route is VIABLE (no pre-written PASS).
    assert all(v["status"] != "VIABLE" for v in rv.values()), rv
    # reasons precisely name the undetermined criteria.
    assert any("C2" in r for r in agg["reasons"]), agg["reasons"]
    # blocking_artifacts lists only real blockers.
    assert all(isinstance(a, str) and a for a in agg["blocking_artifacts"])


def test_gonogo_json_derived_state_matches_manifest(tmp_path, monkeypatch):
    """Task 7: gonogo.json and manifest.json AGREE on route_verdict /
    phase0_completion / phase1_authorization. Both use the same shared helper
    (recompute_derived_state) on the same gate artifacts, so they CANNOT
    diverge."""
    import json, os, shutil
    from results._phase0 import gonogo as G, manifest as M

    src = "results/phase0"
    stage = tmp_path / "phase0"
    stage.mkdir()
    for name in os.listdir(src):
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(stage, name))
        elif os.path.isdir(s) and name in (
            "c1_optimized_hlo",
            "c1_buffer_assignment",
            "c1_xla_dump",
        ):
            shutil.copytree(s, os.path.join(stage, name))
    monkeypatch.setattr(G, "_collect_environment", lambda: {"_stub": True})
    G.main(stage_dir=str(stage))
    manifest = M.build_manifest(str(stage), generated_at="2026-07-23T00:00:00Z")
    gonogo = json.load(open(stage / "gonogo.json"))
    # Derived state must match.
    assert gonogo["phase0_completion"] == manifest["phase0_completion"], (
        gonogo["phase0_completion"],
        manifest["phase0_completion"],
    )
    assert gonogo["phase1_authorization"] == manifest["phase1_authorization"], (
        gonogo["phase1_authorization"],
        manifest["phase1_authorization"],
    )
    assert gonogo["route_verdict"] == manifest["route_verdict"], (
        gonogo["route_verdict"],
        manifest["route_verdict"],
    )
    # reasons + blocking also match (both from the same shared helper).
    assert gonogo["reasons"] == manifest["reasons"], (
        gonogo["reasons"],
        manifest["reasons"],
    )
    assert gonogo["blocking_artifacts"] == manifest["blocking_artifacts"], (
        gonogo["blocking_artifacts"],
        manifest["blocking_artifacts"],
    )


def test_json_and_md_render_from_same_object(tmp_path, monkeypatch):
    """Task 7 plan §10 验收: JSON 与 Markdown 从同一对象生成. gonogo.md is
    rendered FROM the gonogo.json object (via _render_md), so the two never
    contradict each other on any phase-level field or route verdict."""
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
    ):
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy(s, stage / name)
    monkeypatch.setattr(G, "_collect_environment", lambda: {"_stub": True})
    G.main(stage_dir=str(stage))
    agg = json.load(open(stage / "gonogo.json"))
    md = (stage / "gonogo.md").read_text()
    # Every phase-level field in the JSON appears in the MD.
    for field in ("phase0_completion", "phase1_authorization"):
        assert agg[field] in md, (field, agg[field])
    # Every route verdict status in the JSON appears in the MD.
    for route, rv in agg["route_verdict"].items():
        assert rv["status"] in md, (route, rv["status"])
        assert f"`{route}`" in md, route
    # Every reason in the JSON appears in the MD.
    for reason in agg["reasons"]:
        assert reason in md, reason


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.3: REQUIRED_CRITERIA uses old C2 alias; a UNKNOWN
# C2 sub-layer must block COMPLETE even when the old "C2" alias is PASS.
# ---------------------------------------------------------------------------


def test_completion_inconclusive_when_c2_region_unknown_even_if_c2_alias_pass():
    """Nongpu rereview finding 3.3: ``C2=PASS`` (old alias) but
    ``C2_REGION_KERNEL_FEASIBILITY=UNKNOWN`` -> ``phase0_completion`` must be
    ``INCONCLUSIVE``, ``phase1_authorization`` ``NOT_AUTHORIZED``.

    Current ``REQUIRED_CRITERIA`` (verdict_schema.py:193-201) includes ``"C2"``
    (the alias) but NOT the four C2 layers, so a UNKNOWN sub-layer does not
    block completion -> false COMPLETE / GO_TO_PHASE1."""
    from results._phase0.gonogo import aggregate_two_layer

    criteria = {
        "C1": "PASS",
        "C2": "PASS",  # old alias, determined
        "C2_REGION_KERNEL_FEASIBILITY": "UNKNOWN",  # sub-layer undetermined
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",
        "CUTLASS_SM120_4M": "NOT_SUPPORTED",
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
        "REGION_PROTOTYPE": "PASS",
        "NUMERICAL": "PASS",
    }
    per_route = {"planar": "PASS"}
    agg = aggregate_two_layer(criteria, per_route)
    assert agg["phase0_completion"] == "INCONCLUSIVE", agg["phase0_completion"]
    assert agg["phase1_authorization"] == "NOT_AUTHORIZED", agg["phase1_authorization"]


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.5: canonical capability readers return detail tokens
# -> false negative (real success can't be PASS).
# ---------------------------------------------------------------------------


def test_c3_grouped_status_recomputes_pass_from_supported_evidence(tmp_path):
    """Nongpu rereview finding 3.5.1: grouped raw ``SUPPORTED`` + complete
    evidence -> canonical ``PASS``. Current ``_c3_grouped_status``
    (gonogo.py:406-409) returns raw ``"SUPPORTED"`` (a detail token) which
    ``tri_normalize`` maps to UNKNOWN -> real grouped success can never be PASS
    (false negative)."""
    import json

    from results._phase0.gonogo import _c3_grouped_status

    p = tmp_path / "g.json"
    p.write_text(
        json.dumps(
            {
                "capability": {"status": "SUPPORTED"},
                # complete evidence fields (the fix recomputes from these)
                "batched_route": {"status": "SUPPORTED"},
                "grouped_api_probe": {"cublaslt_grouped3gemm": True},
            }
        )
    )
    assert _c3_grouped_status(str(p)) == "PASS"


def test_region_proto_status_recomputes_pass_from_full_anchor_evidence(tmp_path):
    """Nongpu rereview finding 3.5.2: region canonical ``PASS`` + complete
    full-anchor evidence -> canonical ``PASS``. Current ``_region_proto_status``
    (gonogo.py:505-522) doesn't accept canonical ``PASS`` (only ``FEASIBLE*`` /
    ``NOT_FEASIBLE`` / ``BLOCKED``) -> returns UNKNOWN -> full-anchor region
    success can never be PASS (false negative).

    Task 4: the reader now recomputes via the C2 ``_recompute_conditions``
    helper (shared peak gate). PASS requires a success verdict + a fused
    full-anchor run + recomputed accuracy/resource pass + a MEASURED runtime
    peak (peak_evidence_class=MEASURED + required measured fields) -- the same
    standard C2_REGION_KERNEL_FEASIBILITY uses."""
    import json

    from results._phase0.gonogo import _region_proto_status

    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                # Real P->T->E prototype fields (Task 4 review fix finding 4):
                # the region reader now gates PASS on the SAME intrinsic standard
                # as c2._is_real_pte_prototype -- schema version, region
                # producer/consumer MNK, full-E consumer, no full P/T
                # materialized, non-reduction math. Without these the strict
                # reader returns UNKNOWN (a GEMM->norm artifact cannot PASS).
                "schema_version": "region-prototype-v2",
                "region": {
                    "producer": [4096, 16384, 1024],
                    "consumer": [64, 1048576, 64],
                    "dtype": "c64",
                },
                "math": "E = D @ transform(A@B); transform = reshape->transpose->reshape",
                "no_full_P_materialized": True,
                "no_full_T_materialized": True,
                "fused_full_anchor_run": True,
                "relative_l2": 1e-7,
                "max_rel": 1e-7,
                "registers_per_thread": 40,
                "occupancy_pct": 100.0,
                # MEASURED runtime peak (shared peak gate with C2): a full-anchor
                # fused run measured the runtime allocator peak -> canonical gain.
                "peak_evidence_class": "MEASURED",
                "materialized_runtime_allocator_peak_bytes": 2000000000,
                "fused_runtime_allocator_peak_bytes": 1000000000,
                "runtime_peak_measurement_method": "cuda allocator high-watermark",
                "runtime_peak_scope": "full_anchor",
                "runtime_peak_sample_count": 3,
            }
        )
    )
    assert _region_proto_status(str(p)) == "PASS"


def test_region_proto_status_unknown_when_feasible_without_full_anchor(tmp_path):
    """Nongpu rereview finding 3.5.2 (complementary GREEN pin): region detail
    ``FEASIBLE*`` without full-anchor evidence -> UNKNOWN. This should already
    pass on current code (normalize maps FEASIBLE* to UNKNOWN at the criterion
    level). Included to pin the honest-negative path alongside the false-
    negative RED test above."""
    import json

    from results._phase0.gonogo import _region_proto_status
    from results._phase0.verdict_schema import normalize_criterion

    p = tmp_path / "r.json"
    p.write_text(json.dumps({"verdict": "FEASIBLE_WITH_RECOMPUTE"}))
    raw = _region_proto_status(str(p))
    # The criterion-level value (after normalize) must be UNKNOWN.
    assert normalize_criterion(raw) == "UNKNOWN", raw


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.6: CUTLASS trusts self-reported capability.
# ---------------------------------------------------------------------------


def test_cutlass_native_rejects_self_reported_pass_without_evidence():
    """Nongpu rereview finding 3.6: CUTLASS native ``section.capability=PASS``
    but ``runs=false`` / ``gate_pass=false`` -> must be UNKNOWN/FAIL, not PASS.
    Current ``_cutlass_native_sm120_criterion`` (gonogo.py:461-463) returns
    ``sec.get("capability")`` directly -> PASS leaks through."""
    from results._phase0.gonogo import _cutlass_native_sm120_criterion

    data = {
        "native_sm120_bf16_4m": {
            "capability": "PASS",
            "runs": False,
            "correctness": {"gate_pass": False},
        }
    }
    result = _cutlass_native_sm120_criterion(data)
    assert result != "PASS", (
        f"self-reported PASS without runs/gate evidence must not be PASS, "
        f"got {result!r}"
    )
    assert result in ("UNKNOWN", "FAIL"), result


def test_cutlass_fallback_rejects_self_reported_pass_with_wrong_path():
    """Nongpu rereview finding 3.6: CUTLASS fallback ``section.capability=PASS``
    but actual path is native/unknown -> must be UNKNOWN. Current
    ``_cutlass_sm80_fallback_criterion`` (gonogo.py:492-494) returns
    ``sec.get("capability")`` directly -> PASS leaks through."""
    from results._phase0.gonogo import _cutlass_sm80_fallback_criterion

    data = {
        "sm80_fallback_bf16_4m": {
            "capability": "PASS",
            "kernel_path": "sm120_native",  # actual path is native, not fallback
        }
    }
    result = _cutlass_sm80_fallback_criterion(data)
    assert (
        result == "UNKNOWN"
    ), f"self-reported PASS with wrong kernel_path must be UNKNOWN, got {result!r}"


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.8: full-matrix algorithm/workspace constraints
# incomplete.
# ---------------------------------------------------------------------------


def test_c3_full_matrix_unknown_on_ok_with_sentinel_algo_id(tmp_path):
    """Nongpu rereview finding 3.8: ``status='ok'`` + ``first_algo_id=-1`` ->
    UNKNOWN. An 'ok' row must have a real algo id (>= 0), not the -1 sentinel.
    Current reader (gonogo.py) checks ``algo_count >= 1`` for ok rows but not
    ``first_algo_id >= 0`` -> PASS leaks through."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]  # all-ok shape
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    rows[0][8] = -1  # first_algo_id=-1 on an ok row (should be >= 0)
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_workspace_exceeding_cap(tmp_path):
    """Nongpu rereview finding 3.8: ``workspace_bytes > ws_cap`` bytes ->
    UNKNOWN. The workspace must not exceed the selected workspace cap. Current
    reader checks ``workspace_bytes >= 0`` but not ``<= cap`` -> PASS leaks
    through."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(16384, 16, 16)]
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    # Find the ws_cap="0" (cap=0 bytes) row and set workspace_bytes=1000 (>0).
    for r in rows:
        if r[4] == "0":  # ws_cap name "0" -> cap 0 bytes
            r[9] = 1000  # workspace_bytes=1000 > cap 0
            break
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


def test_c3_full_matrix_unknown_on_no_algo_with_nonzero_workspace(tmp_path):
    """Nongpu rereview finding 3.8: ``status='no-algo'`` + ``workspace_bytes>0``
    -> UNKNOWN. A no-algo row must have ``workspace_bytes=0``. Current reader
    checks ``algo_count==0`` + ``first_algo_id==-1`` for no-algo rows but not
    ``workspace_bytes==0`` -> PASS leaks through."""
    from results._phase0.gonogo import _c3_planar_full_matrix_status

    shapes = [(262144, 64, 4)]  # policy shape -> has legitimate no-algo cells
    shapes_csv = _write_synthetic_shapes(tmp_path, shapes)
    rows = _synth_complete_rows(shapes)
    idx = next(i for i, r in enumerate(rows) if r[10] == "no-algo")
    rows[idx][9] = 100  # workspace_bytes=100 on a no-algo row (should be 0)
    fm = _write_full_matrix(tmp_path, rows)
    assert _c3_planar_full_matrix_status(str(fm), str(shapes_csv)) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.10: blocking_artifacts semantics wrong.
# ---------------------------------------------------------------------------


def test_blocking_artifacts_lists_real_blockers_not_determined_grouped():
    """Nongpu rereview finding 3.10: ``blocking_artifacts`` must contain C2,
    REGION_PROTOTYPE, NUMERICAL (the undetermined required criteria) and must
    NOT contain the determined grouped NOT_SUPPORTED (a single-route blocker
    that doesn't affect completion).

    Current ``_build_blocking_artifacts`` (verdict_schema.py:330-340) lists
    only C2 + grouped NOT_SUPPORTED (wrong): it misses REGION_PROTOTYPE and
    NUMERICAL (undetermined), and wrongly includes grouped (determined)."""
    from results._phase0.gonogo import aggregate_two_layer

    criteria = {
        "C1": "PASS",
        "C2": "UNKNOWN",
        "C2_REGION_KERNEL_FEASIBILITY": "UNKNOWN",
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",  # determined, sinks grouped route only
        "CUTLASS_SM120_4M": "NOT_SUPPORTED",
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
        "REGION_PROTOTYPE": "UNKNOWN",  # undetermined -> must be in blocking
        "NUMERICAL": "UNKNOWN",  # undetermined -> must be in blocking
    }
    per_route = {"planar": "FAIL", "grouped": "FAIL"}
    agg = aggregate_two_layer(criteria, per_route)
    blocking = " ".join(agg["blocking_artifacts"]).lower()
    # Must list C2 (undetermined).
    assert "c2" in blocking, agg["blocking_artifacts"]
    # Must list REGION_PROTOTYPE (undetermined) -- currently missing.
    assert "region" in blocking, agg["blocking_artifacts"]
    # Must list NUMERICAL (undetermined) -- currently missing.
    assert "numerical" in blocking, agg["blocking_artifacts"]
    # Must NOT list grouped NOT_SUPPORTED (determined, single-route blocker).
    assert "grouped" not in blocking, agg["blocking_artifacts"]


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
