"""TDD tests for ``gate_contracts.py`` (plan Task 0).

These tests pin the empty-safe / OR-of-AND fail / dominance / default-UNKNOWN
behavior of :func:`evaluate_gate` and the four frozen contracts in
``GATE_CONTRACTS``. They are the RED step of the TDD loop (the brief's Step 1).

The plan's original ``test_real_multi_determination_conflict`` was muddy
(grouped with a contradiction, which short-circuits via the contradiction
path rather than exercising a true multi-determination double-hit). Per the
v3-review errata it is split into TWO tests:

  * ``test_contradiction_field_yields_unknown`` -- contradiction path
    (grouped raw with ``consistency_state=CONFLICT``).
  * ``test_real_multi_determination_double_hit`` -- a structurally-real
    double-hit on ``cutlass_native`` (all 7 pass fields at PASS values AND
    ``blocker_state=PRESENT`` + ``blocker_source_state=RECOGNIZED`` so
    ``pass_clause`` and ``not_supported_clause`` both hit -> hits=2 ->
    UNKNOWN). A real double-hit is structurally impossible for ``grouped``
    (its fail/not_supported clauses are value-contradictions of the pass
    fields), so the test uses ``cutlass_native`` whose not_supported fields
    (``blocker_state`` / ``blocker_source_state``) are disjoint from the
    pass fields.
"""

from results._phase0.gate_contracts import (
    GATE_CONTRACTS,
    GateContract,
    evaluate_gate,
    load_normative_policy,
)


def test_empty_not_supported_does_not_hit():
    full_region = {
        "schema_state": "VALID",
        "evidence_class_state": "MEASURED",
        "method_state": "APPROVED",
        "scope_state": "FULL_ANCHOR_PTE",
        "sample_state": "OK",
        "peak_state": "OK",
        "gain_state": "OK",
        "full_anchor_run_state": "TRUE",
        "case_binding_state": "MATCH",
        "consistency_state": "CONSISTENT",
    }
    assert evaluate_gate(full_region, GATE_CONTRACTS["region_peak"])[0] == "PASS"


def test_fail_or_of_and():
    base = {
        "schema_state": "VALID",
        "api_state": "PRESENT",
        "attempt_state": "ATTEMPTED",
        "compile_state": "SUCCEEDED",
        "run_state": "FAILED",
        "correctness_state": "PASSED",
        "coverage_state": "COMPLETE",
        "consistency_state": "CONSISTENT",
    }
    assert evaluate_gate(base, GATE_CONTRACTS["grouped"])[0] == "FAIL"


def test_allowlist_dominance_per_condition_flip():
    base = {
        "schema_state": "VALID",
        "api_state": "PRESENT",
        "attempt_state": "ATTEMPTED",
        "compile_state": "SUCCEEDED",
        "run_state": "SUCCEEDED",
        "correctness_state": "PASSED",
        "coverage_state": "COMPLETE",
        "consistency_state": "CONSISTENT",
    }
    for k, v in [
        ("schema_state", "MISSING"),
        ("api_state", "ABSENT_DEFINITIVE"),
        ("attempt_state", "NOT_ATTEMPTED"),
        ("compile_state", "FAILED"),
        ("run_state", "FAILED"),
        ("correctness_state", "FAILED"),
        ("coverage_state", "INCOMPLETE"),
        ("consistency_state", "CONFLICT"),
    ]:
        assert evaluate_gate({**base, k: v}, GATE_CONTRACTS["grouped"])[0] != "PASS"


def test_default_unknown_empty_input():
    assert evaluate_gate({}, GATE_CONTRACTS["grouped"])[0] == "UNKNOWN"


def test_contradiction_field_yields_unknown():
    # grouped raw with consistency_state=CONFLICT (all other pass fields OK)
    # -> contradiction path short-circuits to UNKNOWN (NOT PASS, NOT FAIL).
    raw = {
        "schema_state": "VALID",
        "api_state": "PRESENT",
        "attempt_state": "ATTEMPTED",
        "compile_state": "SUCCEEDED",
        "run_state": "SUCCEEDED",
        "correctness_state": "PASSED",
        "coverage_state": "COMPLETE",
        "consistency_state": "CONFLICT",
    }
    assert evaluate_gate(raw, GATE_CONTRACTS["grouped"])[0] == "UNKNOWN"


def test_real_multi_determination_double_hit():
    # cutlass_native: all 7 pass fields at PASS values AND not_supported
    # fields (blocker_state=PRESENT + blocker_source_state=RECOGNIZED) both
    # satisfied. pass_clause AND not_supported_clause both hit -> hits=2 ->
    # UNKNOWN (multi-determination, NOT contradiction early-exit). This is
    # the structurally-real double-hit the v3-review errata requires.
    raw = {
        "schema_state": "VALID",
        "attempt_state": "ATTEMPTED",
        "compile_state": "SUCCEEDED",
        "run_state": "SUCCEEDED",
        "correctness_state": "PASSED",
        "coverage_state": "COMPLETE",
        "consistency_state": "CONSISTENT",
        "blocker_state": "PRESENT",
        "blocker_source_state": "RECOGNIZED",
    }
    assert evaluate_gate(raw, GATE_CONTRACTS["cutlass_native"])[0] == "UNKNOWN"


def test_normative_policy_constants_only():
    pol = load_normative_policy()
    assert pol["region_policy"]["approved_methods"] == [
        "cuda_allocator_high_watermark_v1"
    ]
    assert pol["region_policy"]["min_gain_bytes"] == 268435456
    assert "pass_clause" not in pol  # rules in GateContract, not JSON
