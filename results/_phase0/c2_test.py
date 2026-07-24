"""Unit tests for C2.

Two paths are exercised:
- INFORMATIONAL cotengra-state heuristic (``classify_tileability`` / ``judge_c2``): unchanged,
  demoted, non-canonical. Kept tests below.
- CANONICAL fail-closed C2 v2 gate (``judge_c2_canonical``): final-remediation Task 5,
  spec ``2026-07-22-phase0-final-review-spec.md`` §5.4 / §8 / plan §5.

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
python -m pytest results/_phase0/c2_test.py -v
"""

from results._phase0.c2 import classify_tileability, judge_c2


def test_large_regular_gemm_is_direct_tileable():
    s = {
        "M": 4096,
        "N": 4096,
        "K": 4096,
        "consumer_count": 1,
        "transpose": False,
        "bytes": 4096 * 4096 * 8,
    }
    c = classify_tileability(s)
    assert c["class"] in ("direct-gemm-tileable", "tileable-with-pack")
    assert c["global_bytes_eliminated"] > 0


def test_multi_consumer_is_not_tileable():
    s = {
        "M": 2048,
        "N": 2048,
        "K": 2048,
        "consumer_count": 4,
        "transpose": False,
        "bytes": 2048**2 * 8,
    }
    assert classify_tileability(s)["class"] == "not-tileable"


def test_judge_c2_pass_with_one_tileable_large_buffer():
    shapes = [
        {
            "M": 4096,
            "N": 4096,
            "K": 4096,
            "consumer_count": 1,
            "transpose": False,
            "bytes": 4096**2 * 8,
        }
    ]
    j = judge_c2(shapes)
    assert j["status"] == "PASS"


def test_judge_c2_unknown_when_all_unknown():
    shapes = [
        {"M": 0, "N": 0, "K": 0, "consumer_count": 1, "transpose": False, "bytes": 0}
    ]
    assert judge_c2(shapes)["status"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Canonical fail-closed C2 v2 gate (Task 5, spec §5.1 matrix)
# ---------------------------------------------------------------------------

import copy  # noqa: E402

from results._phase0.c2 import judge_c2_canonical  # noqa: E402

HLO_H = "a2dba7afeae3a3bfe16dc645d44c0b1b2da4eb2623e5ac65ca5c9042fe9849be"
AUD_H = "29004fd786ff1302ba00399602ac9e2145229898a4eba61bb70ef993997a35a2"
EDGE_H = "9dc930781a3e5074eb2ee6b4d8c9329ee9d5a58f96c174e36122735054414e78"
PEAK_H = "14bb79810b7a1a461a3f211529ecb2409d7463be0222c4e2da3f06534330da59"
PROTO_H = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
BA_H = "035d52a92f49cb540a3762edab9632a723dc4fde1d720d0194f2ef6c3e78a79a"


def _good_case():
    return {"n": 24, "depth": 10, "fusion": "default", "case_id": "n24_d10_default"}


def _good_edge():
    return {
        "schema_version": "c1-c2-edge-v2",
        "case_id": "n24_d10_default",
        "n": 24,
        "depth": 10,
        "fusion": "default",
        "producer": {
            "hlo_value_id": "%custom-call.497",
            "result_index": 0,
            "dtype": "c64",
            "shape": [4096, 16384],
            "layout": [1, 0],
            "M": 4096,
            "N": 16384,
            "K": 1024,
            "bytes": 536870912,
        },
        "transform": {
            "hlo_ids": [
                "get-tuple-element.246.0",
                "loop_transpose_fusion.2",
                "bitcast.1317.0",
            ],
            "steps": [
                {
                    "op": "bitcast",
                    "shape_in": [4096, 16384],
                    "shape_out": [64, 1048576],
                },
                {
                    "op": "transpose",
                    "dimensions": [2, 1, 0, 4, 6, 3, 5, 7],
                    "shape_in": [64, 1048576],
                    "shape_out": [64, 1048576],
                },
                {
                    "op": "bitcast",
                    "shape_in": [64, 1048576],
                    "shape_out": [64, 1048576],
                },
            ],
            "forward_index_map": "fwd",
            "inverse_index_map": "inv",
            "output_shape": [64, 1048576],
            "output_layout": [1, 0],
        },
        "consumer": {
            "hlo_value_id": "%custom-call.498",
            "result_index": 0,
            "dtype": "c64",
            "shape": [64, 1048576],
            "layout": [1, 0],
            "M": 64,
            "N": 1048576,
            "K": 64,
            "bytes": 536870912,
        },
        "consumer_count": 1,
        "trace_status": "EXACT",
        "source_hlo": {"path": "p", "sha256": HLO_H},
        "allocation_audit": {"path": "p", "sha256": AUD_H},
    }


def _good_peak():
    return {
        "schema_version": "c2-peak-frontier-v1",
        "case_id": "n24_d10_default",
        "n": 24,
        "depth": 10,
        "fusion": "default",
        "source_hlo_sha256": HLO_H,
        "edge_map_sha256": EDGE_H,
        "buffer_assignment_path": "ba.txt",
        "base_peak_bytes": 1107390736,
        "base_peak_t": 1514,
        "anchor_window": {
            "producer_id": "%custom-call.497",
            "consumer_id": "%custom-call.498",
            "single_reduction_bytes": 31872,
            "peak_after_single_elimination": 1107358864,
        },
        "joint_model": {
            "max_joint_reduction_bytes": 704736544,
            "min_cover_by_target": {},
        },
        "diagnostics": {
            "single_anchor_patch_status": "peak_reduction_below_threshold",
            "single_anchor_reduction_bytes": 31872,
            "joint_model_status": "joint_reduction_meets_threshold",
            "max_joint_reduction_bytes": 704736544,
            "kernel_feasibility_status": "UNKNOWN",
        },
        "model_assumptions": ["counterfactual only"],
    }


def _good_prototype():
    return {
        "schema_version": "region-prototype-v2",
        "case_id": "n24_d10_default",
        "region": {
            "producer": [4096, 16384, 1024],
            "consumer": [64, 1048576, 64],
            "dtype": "c64",
        },
        "math": "E = D @ transform(A@B); transform = reshape->transpose->reshape (Task 2)",
        "no_full_P_materialized": True,
        "no_full_T_materialized": True,
        "correctness_contract": {"PM": 2, "PN": 16, "K1": 4, "TM": 4, "TN": 8},
        "n_seeds": 3,
        "relative_l2": 1.35e-7,
        "max_rel": 2.4e-7,
        "correct": True,
        "device": "RTX 5070 Ti Laptop",
        "num_sm": 46,
        "threads_per_block": 256,
        "registers_per_thread": 40,
        "occupancy_blocks_per_sm": 6,
        "occupancy_pct": 100.0,
        "materialized_peak_bytes": 1778384896,
        "fused_peak_bytes": 704643072,
        "peak_saved_bytes": 1073741824,
        # MEASURED runtime allocator peak (plan §5 2.1 / Task 3): these are the
        # canonical peak fields the gate reads via ``_normalize_region_peak``.
        # Task 3 errata: the normalizer reads the committed artifact's REAL
        # field names (``peak_measurement_method``, ``peak_evidence_class``,
        # ``materialized_peak_bytes``, ``fused_peak_bytes``, ``n_seeds``).
        # The approved method is ``cuda_allocator_high_watermark_v1`` (the
        # ONLY entry in normative_policy.json's approved_methods); the
        # canonical full-anchor scope is ``full_anchor_pte_v1``.
        "peak_evidence_class": "MEASURED",
        "peak_measurement_method": "cuda_allocator_high_watermark_v1",
        "runtime_peak_scope": "full_anchor_pte_v1",
        "p_buffer_bytes": 536870912,
        "t_buffer_bytes": 536870912,
        "producer_recompute_factor": 64,
        "producer_recompute_flops": 8796093022208,
        "materialized_latency_ms": 161.6,
        "fused_full_anchor_run": False,
        "fused_latency_note": "compute-bound, not timed",
        "memory_policy_met": True,
        "verdict": "FEASIBLE_WITH_RECOMPUTE",
        "note": "real two-stage P->T->E prototype",
    }


def _good_audit():
    return {
        "schema_version": "c1-buffer-audit-v2",
        "case_id": "n24_d10_default",
        "n": 24,
        "depth": 10,
        "fusion": "default",
        "source_hlo_sha256": HLO_H,
        "buffer_assignment_sha256": BA_H,
        "allocation_source": "xla_buffer_assignment",
        "live_range_source": "xla_buffer_assignment",
        "audit_status": "COMPLETE",
        "missing_fields": [],
        "buffer_count": 251,
        "anchor_count": 1,
        "buffers": [],
    }


def _good_file_hashes():
    return {
        "source_hlo": HLO_H,
        "allocation_audit": AUD_H,
        "edge_map": EDGE_H,
        "peak_frontier": PEAK_H,
        "prototype": PROTO_H,
        "buffer_assignment": BA_H,
    }


def _good():
    """A fully-consistent n24_d10_default input set (all hashes/cases agree)."""
    return (
        _good_edge(),
        _good_peak(),
        _good_prototype(),
        _good_audit(),
        _good_case(),
        _good_file_hashes(),
    )


# --- the real n24_d10_default shape: region PASS, single FAIL, joint UNKNOWN ---


def test_canonical_baseline_region_pass_single_fail_joint_unknown():
    """The honest n24 verdict: region UNKNOWN (full-anchor fused run NOT executed,
    so the kernel-feasibility leverage is unmeasured -> fail-closed UNKNOWN),
    single-patch peak FAIL (structural, route-local), joint UNKNOWN (model-only,
    no executable joint impl) -> canonical UNKNOWN. Single-pair FAIL must NOT
    propagate to canonical FAIL."""
    edge, peak, proto, audit, case, fh = _good()
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    L = j["layers"]
    assert L["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j
    assert L["C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK"] == "FAIL", j
    assert L["C2_JOINT_EXECUTABLE_LEVERAGE"] == "UNKNOWN", j
    assert L["C2_CANONICAL"] == "UNKNOWN", j
    assert j["status"] == "UNKNOWN"
    # single-pair FAIL is labeled as the route-local counterfactual it is
    assert "counterfactual" in j["reason"].lower() or "single" in j["reason"].lower(), j


# --- §5.1: every incomplete/mismatched/stale case -> UNKNOWN ---


def test_canonical_unknown_when_case_id_mismatch():
    edge, peak, proto, audit, case, fh = _good()
    peak["case_id"] = "n22_d10_default"  # differs from edge/case
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_n_depth_fusion_mismatch():
    edge, peak, proto, audit, case, fh = _good()
    edge["n"] = 22  # differs from the case being judged (n=24)
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_producer_shape_mismatch_edge_vs_prototype():
    edge, peak, proto, audit, case, fh = _good()
    proto["region"]["producer"] = [9999, 16384, 1024]  # M != edge producer M (4096)
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_transform_inverse_missing():
    edge, peak, proto, audit, case, fh = _good()
    del edge["transform"]["inverse_index_map"]
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_trace_status_not_exact():
    edge, peak, proto, audit, case, fh = _good()
    edge["trace_status"] = "AMBIGUOUS"  # multiple terminal consumers / unpierceable
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_cross_artifact_hash_mismatch():
    edge, peak, proto, audit, case, fh = _good()
    peak["source_hlo_sha256"] = "dead" * 16  # != edge.source_hlo.sha256
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_on_disk_hash_mismatch():
    edge, peak, proto, audit, case, fh = _good()
    fh = copy.deepcopy(fh)
    fh["edge_map"] = "stale" * 16  # on-disk != recorded peak.edge_map_sha256
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_still_gemm_norm_artifact():
    """A GEMM->norm/reduction prototype must NOT be accepted as a real P->T->E region,
    even if it claims a verdict (the rejected Task C artifact shape)."""
    edge, peak, proto, audit, case, fh = _good()
    proto["math"] = "s = sum(|P|^2)"  # reduction, not E = D @ transform(A@B)
    proto["no_full_P_materialized"] = False
    proto["verdict"] = "NOT_FEASIBLE"
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_unknown_when_E_not_full_output():
    """Consumer must be a full-E GEMM output, not a scalar/degenerate reduction."""
    edge, peak, proto, audit, case, fh = _good()
    proto["region"]["consumer"] = [64, 1, 64]  # N=1 -> 512 B, not a full E tensor
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_correctness_fields_missing():
    edge, peak, proto, audit, case, fh = _good()
    del proto["relative_l2"]  # cannot recompute accuracy_pass
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_resource_fields_missing():
    edge, peak, proto, audit, case, fh = _good()
    del proto["registers_per_thread"]  # cannot recompute resource_pass
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_unknown_schema_version():
    edge, peak, proto, audit, case, fh = _good()
    proto["schema_version"] = "region-prototype-???"  # unrecognized schema
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_only_single_anchor_fail_no_prototype():
    """The core bug fix: single-pair peak FAIL with NO kernel/joint evidence must be
    canonical UNKNOWN, never canonical FAIL (the old gate's over-generalization)."""
    edge, peak, _proto, audit, case, fh = _good()
    j = judge_c2_canonical(
        edge, peak, {}, audit, case=case, file_hashes=fh
    )  # no prototype
    assert j["layers"]["C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK"] == "FAIL", j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_unknown_when_audit_not_real_allocation():
    edge, peak, proto, audit, case, fh = _good()
    audit["allocation_source"] = "hlo_shape_only"  # not a real XLA allocation
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


# --- §5.1: definitive negatives may return FAIL ---


def test_canonical_region_fail_when_real_prototype_not_feasible():
    """A REAL P->T->E prototype that definitively fails correctness/policy ->
    C2_REGION_KERNEL_FEASIBILITY = FAIL, and canonical FAIL (definitive blocker)."""
    edge, peak, proto, audit, case, fh = _good()
    proto["verdict"] = "NOT_FEASIBLE"
    proto["correct"] = False
    proto["relative_l2"] = 0.9  # fails the recomputed accuracy policy
    proto["max_rel"] = 0.9
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "FAIL", j
    assert j["layers"]["C2_CANONICAL"] == "FAIL", j


def test_canonical_joint_fail_when_joint_model_below_threshold():
    """Joint model proven below threshold (complete coverage) -> joint layer FAIL.
    Canonical stays UNKNOWN here because region is PASS but joint is FAIL-without-
    executable-proof is still not a canonical PASS; it is a route-local negative."""
    edge, peak, proto, audit, case, fh = _good()
    peak["joint_model"]["max_joint_reduction_bytes"] = 1024  # << threshold
    peak["diagnostics"]["joint_model_status"] = "joint_reduction_below_threshold"
    peak["diagnostics"]["max_joint_reduction_bytes"] = 1024
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_JOINT_EXECUTABLE_LEVERAGE"] == "FAIL", j


# --- §5.1: PASS only when every layer is complete and positive ---


def test_canonical_pass_when_all_layers_pass():
    """canonical PASS requires region PASS + joint executable PASS (all hashes bound).

    Task 2a: region PASS now requires fused_full_anchor_run=True (plan §5 2.1); the
    good fixture's prototype carries fused_full_anchor_run=False (mirrors the
    committed canonical artifact), so this test sets it True to exercise the
    all-pass composition path."""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True  # exercise the full-anchor-measured PASS path
    # recognize an executable joint implementation (absent in the real frontier, which
    # stays UNKNOWN; this exercises the PASS composition path).
    peak["diagnostics"]["joint_executable_status"] = "PASS"
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "PASS", j
    assert j["layers"]["C2_JOINT_EXECUTABLE_LEVERAGE"] == "PASS", j
    assert j["layers"]["C2_CANONICAL"] == "PASS", j
    assert j["status"] == "PASS"


# --- §5.2: the gate self-recomputes from raw fields ---


def test_canonical_self_recomputes_region_peak_gain_and_single_reduction():
    """Self-recompute (not trusting self-reported booleans): region_peak_gain =
    materialized_runtime_allocator_peak - fused_runtime_allocator_peak (MEASURED
    fields only, plan §5 2.2); single_reduction = base_peak - peak_after_single.
    The gate must NOT trust the self-reported ``runtime_peak_gain_bytes`` /
    ``peak_saved_bytes`` fields."""
    edge, peak, proto, audit, case, fh = _good()
    # sabotage the self-reported saved bytes; the gate must recompute from raw
    # MEASURED runtime allocator peaks, not the self-reported gain fields.
    proto["peak_saved_bytes"] = 0
    proto["runtime_peak_gain_bytes"] = 0
    peak["anchor_window"]["single_reduction_bytes"] = 0
    peak["diagnostics"]["single_anchor_reduction_bytes"] = 0
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    rc = j["recomputed"]
    assert rc["region_peak_gain_bytes"] == 1778384896 - 704643072, j
    assert rc["single_reduction_bytes"] == 1107390736 - 1107358864, j


# ---------------------------------------------------------------------------
# Task 0 (SDD plan §3 操作.2): fail-closed RED baseline. The tests below
# freeze the target behavior the v2 gate must adopt after Tasks 2a/3a wire the
# canonical verdict_schema in. They FAIL on the current implementation by clean
# assertion (not import, not GPU) — that is the point of the RED baseline.
# ---------------------------------------------------------------------------


def test_canonical_region_unknown_when_fused_full_anchor_run_false():
    """plan §3 操作.2 bullet 1: ``fused_full_anchor_run=false`` -> the fused
    kernel was NOT timed/measured at the full anchor. The fail-closed region
    criterion must be UNKNOWN until the full-anchor run is actually executed.

    The current gate returns PASS with a 'not measured' scope note (c2.py
    ``_region_layer``), which leaks an unmeasured-evidence state into a canonical
    PASS — exactly the fail-open pattern plan §3 操作.2 bullet 1 forbids. This
    test freezes the target (UNKNOWN)."""
    from results._phase0.verdict_schema import CRITERION_TOKENS

    edge, peak, proto, audit, case, fh = _good()
    # _good_prototype() carries fused_full_anchor_run=False (mirrors the
    # committed canonical region_prototype.json)
    assert proto["fused_full_anchor_run"] is False, proto
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    region = j["layers"]["C2_REGION_KERNEL_FEASIBILITY"]
    assert (
        region == "UNKNOWN"
    ), f"fused_full_anchor_run=False must yield region UNKNOWN, got {region!r}"
    assert region in CRITERION_TOKENS, region
    assert j["layers"]["C2_CANONICAL"] == "UNKNOWN", j


def test_canonical_region_unknown_when_actual_peak_missing():
    """plan §3 操作.2 bullet 2 / finding 3.1: measured peak fields
    (``materialized_peak_bytes`` /
    ``fused_peak_bytes``) missing -> region UNKNOWN. The gate
    self-recomputes ``region_peak_gain_bytes`` from those fields; if
    either is absent the peak benefit is unconfirmable, so the region criterion
    must fail closed to UNKNOWN. (Task 3: the normalizer reads the committed
    artifact's REAL field names ``materialized_peak_bytes`` /
    ``fused_peak_bytes``, not the plan's stale ``runtime_*`` variants.)"""
    edge, peak, proto, audit, case, fh = _good()
    # Isolate the peak-missing path: declare the full-anchor run done so bullet 1
    # does not independently force UNKNOWN, then strip the peak fields.
    proto["fused_full_anchor_run"] = True
    del proto["materialized_peak_bytes"]
    del proto["fused_peak_bytes"]
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["region_peak_gain_bytes"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


# ---------------------------------------------------------------------------
# Task 2a (plan §3 操作.2 bullet 2 -- M1 fold-in): pin EACH of the four
# missing-field conditions that must sink C2_REGION_KERNEL_FEASIBILITY to
# UNKNOWN. Task 0 only pinned actual-peak (test above); the four tests below
# pin registers / occupancy / actual-peak / full-E correctness explicitly and
# in isolation. They are GPU-free (synthetic fixtures via _good()).
# ---------------------------------------------------------------------------


def test_canonical_region_unknown_m1_when_registers_missing():
    """M1 condition 1/4: ``registers_per_thread`` missing -> resource_pass None ->
    region UNKNOWN. The full-anchor run is declared done so the only UNKNOWN
    driver is the missing register count."""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    del proto["registers_per_thread"]  # M1 #1: registers unmeasured
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["resource_pass"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_region_unknown_m1_when_occupancy_missing():
    """M1 condition 2/4: ``occupancy_pct`` missing -> resource_pass None ->
    region UNKNOWN."""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    del proto["occupancy_pct"]  # M1 #2: occupancy unmeasured
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["resource_pass"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_region_unknown_m1_when_actual_peak_missing():
    """M1 condition 3/4: ``materialized_peak_bytes`` /
    ``fused_peak_bytes`` missing -> region_peak_gain_bytes
    None -> region UNKNOWN. (Parallel to the Task 0 RED test above, named here
    to pin M1 condition 3 explicitly. Task 3: the normalizer reads the
    committed artifact's REAL field names.)"""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    del proto["materialized_peak_bytes"]
    del proto["fused_peak_bytes"]  # M1 #3: peak unmeasured
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["region_peak_gain_bytes"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_region_unknown_m1_when_full_E_correctness_missing():
    """M1 condition 4/4: ``fused_full_anchor_run`` is False -> full-E correctness on
    the real anchor shape was never measured -> region UNKNOWN. (Parallel to the
    Task 0 RED test ``test_canonical_region_unknown_when_fused_full_anchor_run_false``,
    named here to pin M1 condition 4 explicitly.)"""
    from results._phase0.verdict_schema import CRITERION_TOKENS

    edge, peak, proto, audit, case, fh = _good()
    # _good_prototype() carries fused_full_anchor_run=False (the only honest value
    # until Task 2b's full-anchor kernel runs).
    assert proto["fused_full_anchor_run"] is False, proto
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    region = j["layers"]["C2_REGION_KERNEL_FEASIBILITY"]
    assert region == "UNKNOWN", j
    assert region in CRITERION_TOKENS, region


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.1: MODEL_ONLY peak must not yield region PASS.
# ---------------------------------------------------------------------------


def test_canonical_region_unknown_when_peak_evidence_model_only():
    """Nongpu rereview finding 3.1: ``peak_evidence_class=MODEL_ONLY`` must NOT
    yield ``C2_REGION_KERNEL_FEASIBILITY=PASS``. Even with
    ``fused_full_anchor_run=True``, complete accuracy/resource, and legacy
    ``materialized_peak_bytes``/``fused_peak_bytes`` present, a MODEL_ONLY peak
    is an analytical/allocation upper bound -- not a measured runtime allocator
    peak. The gate must fail closed to UNKNOWN.

    Task 2 fix: ``_recompute_conditions`` now gates on
    ``peak_evidence_class == MEASURED``; MODEL_ONLY / missing -> region_peak_gain
    None -> region UNKNOWN. This test was RED before the fix (the old gate read
    legacy ``materialized_peak_bytes``/``fused_peak_bytes`` with no evidence-class
    check) and is GREEN after."""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    proto["peak_evidence_class"] = "MODEL_ONLY"
    # Legacy raw-allocation fields are present (the _good fixture carries them).
    assert proto["materialized_peak_bytes"] is not None
    assert proto["fused_peak_bytes"] is not None
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    region = j["layers"]["C2_REGION_KERNEL_FEASIBILITY"]
    assert (
        region == "UNKNOWN"
    ), f"MODEL_ONLY peak must yield region UNKNOWN, got {region!r}"


# ---------------------------------------------------------------------------
# Task 2 acceptance (plan §5 验收): the MEASURED gate must reject every
# incomplete / fake / absent evidence-class combination, and only a complete
# MEASURED fixture can reach region PASS.
# ---------------------------------------------------------------------------


def test_canonical_region_unknown_when_peak_evidence_class_deleted():
    """plan §5 验收: delete ``peak_evidence_class`` entirely -> region UNKNOWN.
    No evidence class means the gate cannot confirm the peak was measured."""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    del proto["peak_evidence_class"]
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["region_peak_gain_bytes"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_region_unknown_when_measured_but_scope_missing():
    """plan §5 验收: ``peak_evidence_class=MEASURED`` but ``runtime_peak_scope``
    missing -> region UNKNOWN. Fake MEASURED without the required scope/method/
    sample_count metadata cannot produce a canonical gain."""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    proto["peak_evidence_class"] = "MEASURED"
    del proto["runtime_peak_scope"]
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["region_peak_gain_bytes"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_region_unknown_when_measured_but_method_missing():
    """plan §5 验收: ``peak_evidence_class=MEASURED`` but
    ``peak_measurement_method`` missing -> region UNKNOWN. (Task 3: the
    normalizer reads the committed artifact's REAL field name
    ``peak_measurement_method``, not the plan's stale
    ``runtime_peak_measurement_method``.)"""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    proto["peak_evidence_class"] = "MEASURED"
    del proto["peak_measurement_method"]
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["region_peak_gain_bytes"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_region_unknown_when_measured_but_sample_count_missing():
    """plan §5 验收: ``peak_evidence_class=MEASURED`` but
    ``n_seeds`` missing -> region UNKNOWN. (Task 3: the normalizer reads
    ``n_seeds``, the committed artifact's REAL field name, not the plan's
    stale ``runtime_peak_sample_count``.)"""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    proto["peak_evidence_class"] = "MEASURED"
    del proto["n_seeds"]
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["region_peak_gain_bytes"] is None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", j


def test_canonical_region_pass_only_with_complete_measured_fixture():
    """plan §5 验收: a COMPLETE measured fixture (MEASURED + full-anchor +
    all required fields + no P/T evidence) -> region PASS. This is the sole
    path to canonical region PASS; the _good fixture already carries all
    required MEASURED fields (Task 3: uses the committed artifact's REAL
    field names -- ``peak_measurement_method``, ``materialized_peak_bytes``,
    ``fused_peak_bytes``, ``n_seeds``, ``runtime_peak_scope``)."""
    edge, peak, proto, audit, case, fh = _good()
    proto["fused_full_anchor_run"] = True
    assert proto["peak_evidence_class"] == "MEASURED", proto
    assert proto["materialized_peak_bytes"] is not None
    assert proto["fused_peak_bytes"] is not None
    assert proto["peak_measurement_method"] is not None
    assert proto["runtime_peak_scope"] is not None
    assert proto["n_seeds"] is not None
    j = judge_c2_canonical(edge, peak, proto, audit, case=case, file_hashes=fh)
    assert j["recomputed"]["region_peak_gain_bytes"] is not None, j
    assert j["layers"]["C2_REGION_KERNEL_FEASIBILITY"] == "PASS", j


# ---------------------------------------------------------------------------
# Task 3 (evidence-integrity plan v3 finding 3.3): shared region normalizer +
# GateContract in BOTH c2 and gonogo. The tests below pin the shared
# ``_normalize_region_peak`` + ``_classify_peak_v2`` + the 12-field
# region_peak contract (with accuracy_state + resource_state).
# ---------------------------------------------------------------------------


def test_classify_peak_strict():
    """Plan Task 3 Step 1: ``_classify_peak_v2`` strict peak-value classifier."""
    from results._phase0.c2 import _classify_peak_v2

    for v, exp in [
        (float("nan"), "NAN_INF"),
        (float("inf"), "NAN_INF"),
        (-5, "NEGATIVE"),
        (True, "BOOL"),
        (None, "MISSING"),
        (3.5, "NON_INTEGER"),
        (100, "OK"),
    ]:
        assert _classify_peak_v2(v) == exp, (v, exp, _classify_peak_v2(v))


def test_region_missing_case_binding_not_pass():
    """Task 3 errata #2: ``case_binding_state=MISSING`` (no binding verification)
    -> not PASS. Uses the committed artifact's REAL field names
    (``schema_version=region-prototype-v2``, ``peak_measurement_method``,
    ``materialized_peak_bytes``, ``fused_peak_bytes``, ``n_seeds``)."""
    from results._phase0.c2 import _normalize_region_peak
    from results._phase0.gate_contracts import GATE_CONTRACTS, evaluate_gate

    proto = {
        "schema_version": "region-prototype-v2",
        "verdict": "FEASIBLE_WITH_RECOMPUTE",
        "peak_evidence_class": "MEASURED",
        "peak_measurement_method": "cuda_allocator_high_watermark_v1",
        "runtime_peak_scope": "full_anchor_pte_v1",
        "n_seeds": 3,
        "materialized_peak_bytes": 400,
        "fused_peak_bytes": 100,
        "fused_full_anchor_run": True,
        "relative_l2": 1e-7,
        "max_rel": 1e-7,
        "registers_per_thread": 40,
        "occupancy_pct": 100.0,
    }
    # Default case_binding_state=MISSING (no binding context) -> not PASS.
    raw = _normalize_region_peak(proto)
    assert raw["case_binding_state"] == "MISSING"
    token, _ = evaluate_gate(raw, GATE_CONTRACTS["region_peak"])
    assert token != "PASS", (token, raw)


def test_region_scope_mismatch_conflict():
    """Task 3 errata #3: scope claims full_anchor AND ``fused_full_anchor_run``
    is False -> ``scope_state=MISMATCH`` -> contradiction -> UNKNOWN. Checked
    BEFORE ``FULL_ANCHOR_PTE`` (the MISMATCH ordering)."""
    from results._phase0.c2 import _normalize_region_peak

    proto = {
        "peak_evidence_class": "MEASURED",
        "peak_measurement_method": "cuda_allocator_high_watermark_v1",
        "runtime_peak_scope": "full_anchor_pte_v1",
        "n_seeds": 3,
        "materialized_peak_bytes": 400,
        "fused_peak_bytes": 100,
        "fused_full_anchor_run": False,  # scope full_anchor but run False -> MISMATCH
    }
    raw = _normalize_region_peak(proto)
    assert raw["scope_state"] == "MISMATCH", raw


def test_region_full_positive_pass():
    """Task 3 errata #7: a full positive fixture with ALL 12 conditions OK ->
    PASS, using REAL recomputed accuracy/resource (not self-reported booleans).
    Uses the committed artifact's REAL field names. Requires
    ``case_binding_state=MATCH`` (the c2 reader's binding-verified path)."""
    from results._phase0.c2 import _normalize_region_peak
    from results._phase0.gate_contracts import GATE_CONTRACTS, evaluate_gate

    proto = {
        "schema_version": "region-prototype-v2",
        "verdict": "FEASIBLE_WITH_RECOMPUTE",
        "peak_evidence_class": "MEASURED",
        "peak_measurement_method": "cuda_allocator_high_watermark_v1",
        "runtime_peak_scope": "full_anchor_pte_v1",
        "n_seeds": 3,
        "materialized_peak_bytes": 2000000000,
        "fused_peak_bytes": 1000000000,
        "fused_full_anchor_run": True,
        "relative_l2": 1e-7,
        "max_rel": 1e-7,
        "registers_per_thread": 40,
        "occupancy_pct": 100.0,
    }
    # c2 reader's binding-verified path: case_binding_state=MATCH.
    raw = _normalize_region_peak(proto, case_binding_state="MATCH")
    # All 12 states must be at their PASS values.
    assert raw["schema_state"] == "VALID"
    assert raw["evidence_class_state"] == "MEASURED"
    assert raw["method_state"] == "APPROVED"
    assert raw["scope_state"] == "FULL_ANCHOR_PTE"
    assert raw["sample_state"] == "OK"
    assert raw["peak_state"] == "OK"
    assert raw["gain_state"] == "OK"
    assert raw["full_anchor_run_state"] == "TRUE"
    assert raw["case_binding_state"] == "MATCH"
    assert raw["accuracy_state"] == "PASSED"
    assert raw["resource_state"] == "OK"
    token, _ = evaluate_gate(raw, GATE_CONTRACTS["region_peak"])
    assert token == "PASS", (token, raw)


def test_region_committed_artifact_is_unknown():
    """Task 3: the committed ``results/phase0/region_prototype.json`` (MODEL_ONLY,
    no full-anchor run, resource null, unapproved method) -> reader returns
    UNKNOWN (honest). No regen needed; the committed artifact is honestly
    UNKNOWN and the shared normalizer + GateContract must reflect that."""
    import json

    from results._phase0.c2 import _normalize_region_peak
    from results._phase0.gate_contracts import GATE_CONTRACTS, evaluate_gate

    with open("results/phase0/region_prototype.json") as f:
        proto = json.load(f)
    # The committed artifact has the REAL fields the normalizer maps.
    assert proto["schema_version"] == "region-prototype-v2"
    assert proto["peak_evidence_class"] == "MODEL_ONLY"
    assert proto["peak_measurement_method"] == "raw_allocation_size_delta"
    assert proto["fused_full_anchor_run"] is False
    assert proto["registers_per_thread"] is None
    assert proto["occupancy_pct"] is None
    raw = _normalize_region_peak(proto, case_binding_state="MATCH")
    token, _ = evaluate_gate(raw, GATE_CONTRACTS["region_peak"])
    # Bidirectional consistency: verdict=UNKNOWN -> expected=UNKNOWN; candidate
    # is FAIL (resource_state=MISSING) -> CONFLICT -> UNKNOWN. Either way UNKNOWN.
    from results._phase0.c2 import _REGION_SELF_REPORT_MAP

    expected = _REGION_SELF_REPORT_MAP.get(proto.get("verdict"))
    if expected is not None and token != expected:
        raw["consistency_state"] = "CONFLICT"
        token, _ = evaluate_gate(raw, GATE_CONTRACTS["region_peak"])
    assert token == "UNKNOWN", (token, raw)


# ---------------------------------------------------------------------------
# Task 8 Step 4 concrete: region negative gain -> FAIL
# ---------------------------------------------------------------------------


def test_region_negative_gain_fails():
    """A region proto with materialized_peak < fused_peak -> gain negative ->
    gain_state=NEGATIVE -> evaluate_gate returns FAIL. Uses the shared
    _normalize_region_peak + evaluate_gate(GATE_CONTRACTS["region_peak"])."""
    from results._phase0.c2 import _normalize_region_peak
    from results._phase0.gate_contracts import GATE_CONTRACTS, evaluate_gate

    proto = {
        "schema_version": "region-prototype-v2",
        "verdict": "FEASIBLE_WITH_RECOMPUTE",
        "peak_evidence_class": "MEASURED",
        "peak_measurement_method": "cuda_allocator_high_watermark_v1",
        "runtime_peak_scope": "full_anchor_pte_v1",
        "n_seeds": 3,
        "materialized_peak_bytes": 100,
        "fused_peak_bytes": 400,
        "fused_full_anchor_run": True,
        "relative_l2": 1e-7,
        "max_rel": 1e-7,
        "registers_per_thread": 40,
        "occupancy_pct": 100.0,
    }
    raw = _normalize_region_peak(proto, case_binding_state="MATCH")
    assert raw["gain_state"] == "NEGATIVE", raw
    token, reason = evaluate_gate(raw, GATE_CONTRACTS["region_peak"])
    assert token == "FAIL", (token, reason, raw)


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
