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

import pytest

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
        # v3-review errata (Task 3): accuracy_state + resource_state added
        # to the pass clause (12 fields total). Without these, a region with
        # relative_l2 above threshold or missing registers/occupancy would PASS.
        "accuracy_state": "PASSED",
        "resource_state": "OK",
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
        "probe_source_state": "RECOGNIZED",
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
        ("probe_source_state", "UNRECOGNIZED"),
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
        "probe_source_state": "RECOGNIZED",
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
    # P1 #2 fix: raw_allocation_size_delta (stale analytical) removed;
    # cuda_allocator_highwatermark (real runtime method) added.
    assert pol["region_policy"]["approved_methods"] == [
        "cuda_allocator_high_watermark_v1",
        "cuda_allocator_highwatermark",
    ]
    assert pol["region_policy"]["min_gain_bytes"] == 268435456
    assert "pass_clause" not in pol  # rules in GateContract, not JSON


# ---------------------------------------------------------------------------
# Task 8 Step 4 concrete tests: per-condition flip for region (12), native (7),
# fallback (6). For each pass condition, flip it to a non-PASS value while
# keeping the other N-1 at PASS -> assert evaluate_gate != PASS.
# ---------------------------------------------------------------------------


def _flip(contract, flip_map):
    """Yield (field_name, flipped_value) for each pass condition, using the
    appropriate non-PASS flip value from *flip_map* (or ``"FLIPPED"`` default)."""
    for field_name, pass_value in contract.pass_clause:
        yield field_name, flip_map.get(field_name, "FLIPPED")


_REGION_FLIP = {
    "schema_state": "BROKEN",
    "evidence_class_state": "MODEL_ONLY",
    "method_state": "UNAPPROVED",
    "scope_state": "PARTIAL",
    "sample_state": "NAN",
    "peak_state": "NAN",
    "gain_state": "NEGATIVE",
    "full_anchor_run_state": "FALSE",
    "case_binding_state": "MISSING",
    "consistency_state": "CONFLICT",
    "accuracy_state": "FAILED",
    "resource_state": "MISSING",
}


def test_region_12_condition_flip():
    """Flip each of the 12 region_peak pass conditions one at a time ->
    evaluate_gate != PASS."""
    contract = GATE_CONTRACTS["region_peak"]
    base = {
        f: v
        for f, v in contract.pass_clause
        if f not in {c[0] for c in contract.contradiction_fields}
    }
    # Build the full PASS raw with the 10 non-contradiction fields at PASS
    # values, then add the 2 contradiction-capable fields one at a time
    # (avoiding the contradiction value in the base so the base itself is PASS).
    n = 0
    for field_name, flip_value in _flip(contract, _REGION_FLIP):
        raw = dict(base)
        # Set the flipped field to its non-PASS value.
        raw[field_name] = flip_value
        # Set all other pass fields that might not be in base to their PASS values.
        for f, pv in contract.pass_clause:
            if f not in raw:
                raw[f] = pv
        token, _ = evaluate_gate(raw, contract)
        assert token != "PASS", (
            f"region flipped {field_name}={flip_value} still got PASS; "
            f"raw={ {k: v for k, v in raw.items() if k == field_name or k in ('schema_state',)} }"
        )
        n += 1
    assert n == 12, n  # all 12 pass conditions were exercised


_NATIVE_FLIP = {
    "schema_state": "BROKEN",
    "attempt_state": "NOT_ATTEMPTED",
    "compile_state": "FAILED",
    "run_state": "FAILED",
    "correctness_state": "FAILED",
    "coverage_state": "INCOMPLETE",
    "consistency_state": "CONFLICT",
}


def test_cutlass_native_7_condition_flip():
    """Flip each of the 7 cutlass_native pass conditions one at a time ->
    evaluate_gate != PASS."""
    contract = GATE_CONTRACTS["cutlass_native"]
    base = dict(contract.pass_clause)
    n = 0
    for field_name, flip_value in _flip(contract, _NATIVE_FLIP):
        raw = dict(base)
        raw[field_name] = flip_value
        token, _ = evaluate_gate(raw, contract)
        assert token != "PASS", f"native flipped {field_name}={flip_value} got PASS"
        n += 1
    assert n == 7, n


_FALLBACK_FLIP = {
    "schema_state": "UNRECOGNIZED",
    "attempt_state": "NOT_ATTEMPTED",
    "compile_state": "BLOCKED",
    "run_state": "FAILED",
    "correctness_state": "FAILED",
    "coverage_state": "INCOMPLETE",
}


def test_cutlass_fallback_6_condition_flip():
    """Flip each of the 6 cutlass_fallback pass conditions one at a time ->
    evaluate_gate != PASS."""
    contract = GATE_CONTRACTS["cutlass_fallback"]
    base = dict(contract.pass_clause)
    n = 0
    for field_name, flip_value in _flip(contract, _FALLBACK_FLIP):
        raw = dict(base)
        raw[field_name] = flip_value
        token, _ = evaluate_gate(raw, contract)
        assert token != "PASS", f"fallback flipped {field_name}={flip_value} got PASS"
        n += 1
    assert n == 6, n


# ---------------------------------------------------------------------------
# F1 fail-open fix: EXHAUSTIVE truth-table test (the anti-reactive part). For
# EACH of the 4 contracts, flip EACH pass field to EACH invalid value (None,
# "" empty, "WRONG" generic token, and a field-specific invalid) and assert
# evaluate_gate != PASS. This proves NO single-field invalid value yields PASS
# -- the coverage the prior reactive (one-value-per-field) flips missed.
# ---------------------------------------------------------------------------

# Field-specific invalid value per pass field (the "natural" failure token for
# that field). Every pass field across all 4 contracts is covered so each
# field gets a 4th, semantically-meaningful invalid value alongside the
# generic None / "" / "WRONG".
_FIELD_SPECIFIC_INVALID = {
    "schema_state": "UNRECOGNIZED",
    "api_state": "ABSENT_INCONCLUSIVE",
    "attempt_state": "NOT_ATTEMPTED",
    "probe_source_state": "UNRECOGNIZED",
    "compile_state": "FAILED",
    "run_state": "FAILED",
    "correctness_state": "FAILED",
    "coverage_state": "INCOMPLETE",
    "consistency_state": "CONFLICT",
    "evidence_class_state": "MODEL_ONLY",
    "method_state": "UNAPPROVED",
    "scope_state": "PARTIAL",
    "sample_state": "NAN",
    "peak_state": "NAN",
    "gain_state": "NEGATIVE",
    "full_anchor_run_state": "FALSE",
    "case_binding_state": "MISSING",
    "accuracy_state": "FAILED",
    "resource_state": "MISSING",
}


def _exhaustive_no_pass_cases():
    """Build (contract_name, field_name, invalid_value) cases: for each
    contract, for each pass field, for each of {None, "", "WRONG",
    field-specific invalid}."""
    cases = []
    for contract_name in (
        "grouped",
        "region_peak",
        "cutlass_native",
        "cutlass_fallback",
    ):
        contract = GATE_CONTRACTS[contract_name]
        for field_name, _pass_value in contract.pass_clause:
            specific = _FIELD_SPECIFIC_INVALID[field_name]
            for invalid in (None, "", "WRONG", specific):
                cases.append((contract_name, field_name, invalid))
    return cases


@pytest.mark.parametrize(
    "contract_name,field_name,invalid_value",
    _exhaustive_no_pass_cases(),
)
def test_no_invalid_value_yields_PASS(contract_name, field_name, invalid_value):
    """EXHAUSTIVE truth-table: flipping any single pass field to any invalid
    value must NOT yield PASS (anti-reactive coverage for the F1 fail-open
    fix). The all-PASS base is first asserted PASS so the test is never
    vacuous; then one field is flipped and the result must differ from PASS."""
    contract = GATE_CONTRACTS[contract_name]
    base = dict(contract.pass_clause)
    # Sanity: the all-PASS base itself must be PASS (else the flip test is
    # vacuous -- a broken base would mask a real fail-open).
    base_token, _ = evaluate_gate(base, contract)
    assert base_token == "PASS", (
        f"{contract_name} all-PASS base is {base_token!r}, not PASS; "
        f"test harness is broken"
    )
    raw = dict(base)
    raw[field_name] = invalid_value
    token, reason = evaluate_gate(raw, contract)
    assert token != "PASS", (
        f"{contract_name} field {field_name}={invalid_value!r} yielded PASS; "
        f"reason={reason}"
    )
