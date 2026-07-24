"""Unit tests for the canonical verdict vocabulary (plan Task 0 / §4).

The schema module is the single source of truth for canonical status tokens.
These tests pin the exact token sets, the criteria / numerical-route name lists,
and the ``normalize_criterion`` mappings. They must stay GREEN — they encode the
contract every reader is rewired onto in Tasks 1-7.

Run (no GPU required):
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
    python -m pytest results/_phase0/verdict_schema_test.py -v
"""

from results._phase0.verdict_schema import (
    AUTHORIZATION_TOKENS,
    COMPLETION_TOKENS,
    CRITERIA_NAMES,
    CRITERION_TOKENS,
    DETAIL_TOKENS,
    NUMERICAL_ROUTES,
    REQUIRED_CRITERIA,
    ROUTE_CAPABILITY_CRITERIA,
    ROUTE_TOKENS,
    TRI_NOT_OK,
    TRI_OK,
    TRI_UNDETERMINED,
    _combine_tri,
    normalize_criterion,
    recompute_authorization,
    recompute_completion,
    recompute_route_verdict,
    tri_normalize,
)

# --- canonical token sets are exactly the plan §4 set, no more, no less ---


def test_criterion_tokens_match_plan_section4():
    assert CRITERION_TOKENS == frozenset(
        {"PASS", "FAIL", "UNKNOWN", "NOT_RUN", "NOT_SUPPORTED"}
    )


def test_route_tokens_match_plan_section4():
    assert ROUTE_TOKENS == frozenset({"VIABLE", "NOT_VIABLE", "UNKNOWN"})


def test_completion_tokens_match_plan_section4():
    assert COMPLETION_TOKENS == frozenset({"COMPLETE", "INCONCLUSIVE"})


def test_authorization_tokens_match_plan_section4():
    assert AUTHORIZATION_TOKENS == frozenset(
        {"GO_TO_PHASE1", "NO_GO", "NOT_AUTHORIZED"}
    )


# --- criteria names: every plan §4 name present; no extras ---


def test_criteria_names_match_plan_section4():
    expected = {
        "C1",
        "C2_REGION_KERNEL_FEASIBILITY",
        "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK",
        "C2_JOINT_EXECUTABLE_LEVERAGE",
        "C2_CANONICAL",
        "C3_PLANAR_CORE",
        "C3_PLANAR_FULL_MATRIX",
        "C3_GROUPED",
        "CUTLASS_SM120_4M",
        "CUTLASS_SM80_FALLBACK_CAPABILITY",
        "REGION_PROTOTYPE",
        "NUMERICAL",
    }
    assert set(CRITERIA_NAMES) == expected
    assert len(CRITERIA_NAMES) == len(expected)  # no duplicates


def test_cutlass_criteria_split_native_and_fallback():
    """Plan §4 freezes two distinct CUTLASS criteria — native SM120 capability
    is tracked separately from the SM80 fallback capability. This split is the
    schema-level hook for plan §3 操作.2 bullet 8 (the two-must-not-merge rule)."""
    assert "CUTLASS_SM120_4M" in CRITERIA_NAMES
    assert "CUTLASS_SM80_FALLBACK_CAPABILITY" in CRITERIA_NAMES
    assert "CUTLASS_SM120_4M" != "CUTLASS_SM80_FALLBACK_CAPABILITY"


# --- numerical routes: canonical route keys, incl. cutlass_4m_single ---


def test_numerical_routes_match_plan_section4():
    """The numerical route list uses the canonical route KEY ``cutlass_4m_single``
    (matching ``ROUTE_CAPABILITY_CRITERIA``), the route actually measured by the
    numerical matrix. The real per_route data (numerical_validation.json) uses
    ``cutlass_4m_single`` -- NOT a separate ``cutlass_sm80_fallback`` name."""
    assert set(NUMERICAL_ROUTES) == {
        "planar",
        "grouped",
        "region_fused",
        "cutlass_4m_single",
    }
    assert "cutlass_sm80_fallback" not in NUMERICAL_ROUTES


def test_numerical_routes_use_route_keys_not_criterion_names():
    """Numerical routes are ROUTE KEYS (not criterion names). The cutlass route
    key ``cutlass_4m_single`` is distinct from its capability CRITERION name
    ``CUTLASS_SM80_FALLBACK_CAPABILITY``."""
    assert "cutlass_4m_single" in NUMERICAL_ROUTES
    assert "cutlass_4m_single" not in CRITERIA_NAMES
    assert "CUTLASS_SM80_FALLBACK_CAPABILITY" in CRITERIA_NAMES


# --- detail tokens are non-canonical (no leakage into criterion fields) ---


def test_detail_tokens_disjoint_from_canonical_criterion():
    """Detail tokens (FEASIBLE*, BLOCKED, SUPPORTED, NOT_FEASIBLE, ...) must
    NOT be canonical criterion tokens. NOT_SUPPORTED is intentionally canonical
    (cubasLt emits it verbatim) and therefore excluded from DETAIL_TOKENS."""
    assert DETAIL_TOKENS.isdisjoint(CRITERION_TOKENS), DETAIL_TOKENS & CRITERION_TOKENS


def test_detail_tokens_includes_plan_section4_examples():
    for t in (
        "FEASIBLE",
        "FEASIBLE_WITH_RECOMPUTE",
        "FEASIBLE_WITH_SM80_FALLBACK",
        "TILE_FUSION_FEASIBLE",
        "BLOCKED",
        "SUPPORTED",
        "NOT_FEASIBLE",
    ):
        assert t in DETAIL_TOKENS, t


# --- normalize_criterion: the fail-closed scrubber ---


def test_normalize_criterion_passthrough_canonical_tokens():
    for t in ("PASS", "FAIL", "UNKNOWN", "NOT_RUN", "NOT_SUPPORTED"):
        assert normalize_criterion(t) == t, t


def test_normalize_criterion_blocked_maps_to_unknown_not_pass():
    """Plan §4 验收: BLOCKED normalize 为 UNKNOWN，不是 PASS."""
    assert normalize_criterion("BLOCKED") == "UNKNOWN"


def test_normalize_criterion_inconclusive_maps_to_unknown():
    """INCONCLUSIVE is a canonical COMPLETION token; in a criterion field it is
    non-canonical and must fail closed to UNKNOWN."""
    assert normalize_criterion("INCONCLUSIVE") == "UNKNOWN"


def test_normalize_criterion_feasible_family_does_not_auto_promote():
    """Plan §4 验收: 不再使用 startswith('FEASIBLE') 无条件提升全部架构/route.
    The FEASIBLE* family in a canonical criterion field -> UNKNOWN (reader must
    re-derive PASS from evidence), NOT PASS."""
    for t in (
        "FEASIBLE",
        "FEASIBLE_WITH_RECOMPUTE",
        "FEASIBLE_WITH_SM80_FALLBACK",
        "TILE_FUSION_FEASIBLE",
    ):
        assert normalize_criterion(t) == "UNKNOWN", t


def test_normalize_criterion_supported_is_not_canonical_pass():
    """SUPPORTED is the cublasLt artifact-native success token; in a canonical
    criterion field it must NOT be auto-promoted to PASS (the reader must
    re-derive). Fail closed -> UNKNOWN."""
    assert normalize_criterion("SUPPORTED") == "UNKNOWN"


def test_normalize_criterion_not_feasible_is_not_canonical_fail():
    """NOT_FEASIBLE is artifact-native; canonical FAIL or NOT_SUPPORTED must be
    re-derived by the reader. In a canonical criterion field -> UNKNOWN."""
    assert normalize_criterion("NOT_FEASIBLE") == "UNKNOWN"


def test_normalize_criterion_missing_empty_or_unknown_token():
    for t in (
        "",
        None,
        "weird-token",
        "GO_TO_PHASE1",
    ):  # GO_TO_PHASE1 is auth, not criterion
        assert normalize_criterion(t) == "UNKNOWN", t


def test_normalize_criterion_never_returns_detail_token():
    """Contract: normalize_criterion's output is always a canonical criterion
    token. Detail tokens can never leak through."""
    cases = (
        "PASS",
        "FAIL",
        "UNKNOWN",
        "NOT_RUN",
        "NOT_SUPPORTED",
        "BLOCKED",
        "FEASIBLE",
        "FEASIBLE_WITH_RECOMPUTE",
        "FEASIBLE_WITH_SM80_FALLBACK",
        "TILE_FUSION_FEASIBLE",
        "SUPPORTED",
        "NOT_FEASIBLE",
        "INCONCLUSIVE",
        "VIABLE",
        "GO_TO_PHASE1",
        "",
        None,
    )
    for t in cases:
        out = normalize_criterion(t)
        assert out in CRITERION_TOKENS, (t, out)


# --- §5 truth table: tri_normalize / _combine_tri (NON-tautological logic) ---


def test_tri_normalize_pass_is_ok():
    assert tri_normalize("PASS") == TRI_OK


def test_tri_normalize_fail_and_not_supported_are_not_ok():
    assert tri_normalize("FAIL") == TRI_NOT_OK
    assert tri_normalize("NOT_SUPPORTED") == TRI_NOT_OK


def test_tri_normalize_unknown_and_not_run_are_undetermined():
    assert tri_normalize("UNKNOWN") == TRI_UNDETERMINED
    assert tri_normalize("NOT_RUN") == TRI_UNDETERMINED


def test_tri_normalize_blocked_feeds_undetermined():
    """BLOCKED normalizes to UNKNOWN (plan §4), which feeds UNDETERMINED in the
    truth table -- never OK (no startswith('FEASIBLE') promotion)."""
    assert tri_normalize("BLOCKED") == TRI_UNDETERMINED


def test_combine_tri_any_not_ok_is_not_ok():
    assert _combine_tri([TRI_OK, TRI_NOT_OK, TRI_OK]) == TRI_NOT_OK


def test_combine_tri_undetermined_when_no_not_ok_but_any_undetermined():
    assert _combine_tri([TRI_OK, TRI_UNDETERMINED]) == TRI_UNDETERMINED


def test_combine_tri_all_ok_is_ok():
    assert _combine_tri([TRI_OK, TRI_OK]) == TRI_OK


def test_combine_tri_empty_is_undetermined():
    assert _combine_tri([]) == TRI_UNDETERMINED


# --- §5 truth table: recompute_route_verdict (rule 3 + rule 8) ---


def test_route_verdict_viable_when_capability_and_numerical_pass():
    criteria = {"C3_PLANAR_CORE": "PASS", "C3_PLANAR_FULL_MATRIX": "PASS"}
    rv = recompute_route_verdict(criteria, {"planar": "PASS"})
    assert rv["planar"]["status"] == "VIABLE"
    assert rv["planar"]["capability"] == TRI_OK
    assert rv["planar"]["numerical"] == TRI_OK


def test_route_verdict_not_viable_when_capability_fail():
    criteria = {"C3_PLANAR_CORE": "FAIL", "C3_PLANAR_FULL_MATRIX": "PASS"}
    rv = recompute_route_verdict(criteria, {"planar": "PASS"})
    assert rv["planar"]["status"] == "NOT_VIABLE"


def test_route_verdict_not_viable_when_capability_not_supported():
    """NOT_SUPPORTED normalizes to NOT_OK (rule 3) -> NOT_VIABLE."""
    criteria = {"C3_PLANAR_CORE": "NOT_SUPPORTED", "C3_PLANAR_FULL_MATRIX": "PASS"}
    rv = recompute_route_verdict(criteria, {"planar": "PASS"})
    assert rv["planar"]["status"] == "NOT_VIABLE"


def test_route_verdict_not_viable_when_numerical_fail():
    criteria = {"C3_PLANAR_CORE": "PASS", "C3_PLANAR_FULL_MATRIX": "PASS"}
    rv = recompute_route_verdict(criteria, {"planar": "FAIL"})
    assert rv["planar"]["status"] == "NOT_VIABLE"


def test_route_verdict_unknown_when_capability_unknown():
    criteria = {"C3_PLANAR_CORE": "UNKNOWN", "C3_PLANAR_FULL_MATRIX": "PASS"}
    rv = recompute_route_verdict(criteria, {"planar": "PASS"})
    assert rv["planar"]["status"] == "UNKNOWN"


def test_route_verdict_unknown_when_numerical_absent_not_run():
    """A route absent from per_route_numerical defaults to NOT_RUN ->
    UNDETERMINED -> status UNKNOWN (fail-closed)."""
    criteria = {"C3_PLANAR_CORE": "PASS", "C3_PLANAR_FULL_MATRIX": "PASS"}
    rv = recompute_route_verdict(criteria, {})
    assert rv["planar"]["status"] == "UNKNOWN"
    assert rv["planar"]["numerical"] == TRI_UNDETERMINED


def test_route_verdict_region_fused_viable_stale_key_catcher():
    """CATCHES the Finding-1 stale-key bug. region_fused depends on
    ``C2_REGION_KERNEL_FEASIBILITY`` (canonical, matching gonogo output at
    gonogo.py:670). Before the fix ``ROUTE_CAPABILITY_CRITERIA["region_fused"]``
    used the abbreviated ``C2_REGION_KERNEL``, so the lookup
    ``criteria.get("C2_REGION_KERNEL")`` missed -> None -> UNDETERMINED ->
    region_fused UNKNOWN even with both deps PASS. After the fix this SYNTHETIC
    fixture (REGION_PROTOTYPE=PASS + C2_REGION_KERNEL_FEASIBILITY=PASS +
    numerical PASS) -> region_fused VIABLE. This test FAILS before the fix and
    PASSES after."""
    criteria = {
        "REGION_PROTOTYPE": "PASS",
        "C2_REGION_KERNEL_FEASIBILITY": "PASS",
    }
    rv = recompute_route_verdict(criteria, {"region_fused": "PASS"})
    assert rv["region_fused"]["status"] == "VIABLE", rv["region_fused"]
    assert rv["region_fused"]["capability"] == TRI_OK, rv["region_fused"]
    assert rv["region_fused"]["numerical"] == TRI_OK, rv["region_fused"]


# --- §5 truth table: recompute_completion (rules 1/4/5) ---


def _all_determined_criteria():
    """Every REQUIRED_CRITERION + every route-capability dep set to PASS
    (determined). Useful baseline for completion tests."""
    criteria = {c: "PASS" for c in REQUIRED_CRITERIA}
    for deps in ROUTE_CAPABILITY_CRITERIA.values():
        for d in deps:
            criteria[d] = "PASS"
    return criteria


def test_completion_complete_when_all_required_determined():
    assert recompute_completion(_all_determined_criteria()) == "COMPLETE"


def test_completion_inconclusive_when_any_unknown():
    criteria = _all_determined_criteria()
    criteria["C2_CANONICAL"] = "UNKNOWN"
    assert recompute_completion(criteria) == "INCONCLUSIVE"


def test_completion_inconclusive_when_any_not_run():
    criteria = _all_determined_criteria()
    criteria["NUMERICAL"] = "NOT_RUN"
    assert recompute_completion(criteria) == "INCONCLUSIVE"


def test_completion_numerical_fail_does_not_sink():
    """NUMERICAL=FAIL is determined (NOT_OK, not UNDETERMINED) -- it does NOT
    alone make completion INCONCLUSIVE (§5 truth table rule 1/4: only
    UNDETERMINED criteria sink completion)."""
    criteria = _all_determined_criteria()
    criteria["NUMERICAL"] = "FAIL"
    assert recompute_completion(criteria) == "COMPLETE"


# --- §5 truth table: recompute_authorization (rule 6) ---


def test_authorization_not_authorized_when_inconclusive():
    rv = {"planar": {"status": "VIABLE"}}
    assert recompute_authorization("INCONCLUSIVE", rv) == "NOT_AUTHORIZED"


def test_authorization_go_to_phase1_when_complete_and_any_viable():
    rv = {"planar": {"status": "VIABLE"}, "grouped": {"status": "NOT_VIABLE"}}
    assert recompute_authorization("COMPLETE", rv) == "GO_TO_PHASE1"


def test_authorization_no_go_when_complete_and_none_viable():
    rv = {"planar": {"status": "NOT_VIABLE"}, "grouped": {"status": "UNKNOWN"}}
    assert recompute_authorization("COMPLETE", rv) == "NO_GO"


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.3: each C2 sub-layer UNKNOWN must block COMPLETE.
# Current REQUIRED_CRITERIA (verdict_schema.py:193-201) uses the old "C2"
# alias, not the four C2 layers, so a UNKNOWN sub-layer doesn't block
# completion -> false COMPLETE.
# ---------------------------------------------------------------------------


def test_completion_inconclusive_when_c2_single_anchor_unknown():
    """Nongpu rereview finding 3.3: ``C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK``
    = UNKNOWN must block COMPLETE. Current ``REQUIRED_CRITERIA`` uses the old
    ``"C2"`` alias; the four C2 layers are not in ``REQUIRED_CRITERIA``, so a
    UNKNOWN sub-layer doesn't block completion."""
    from results._phase0.verdict_schema import REQUIRED_CRITERIA, recompute_completion

    criteria = {c: "PASS" for c in REQUIRED_CRITERIA}
    criteria["C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK"] = "UNKNOWN"
    assert recompute_completion(criteria) == "INCONCLUSIVE"


def test_completion_inconclusive_when_c2_joint_leverage_unknown():
    """Nongpu rereview finding 3.3: ``C2_JOINT_EXECUTABLE_LEVERAGE`` = UNKNOWN
    must block COMPLETE."""
    from results._phase0.verdict_schema import REQUIRED_CRITERIA, recompute_completion

    criteria = {c: "PASS" for c in REQUIRED_CRITERIA}
    criteria["C2_JOINT_EXECUTABLE_LEVERAGE"] = "UNKNOWN"
    assert recompute_completion(criteria) == "INCONCLUSIVE"


def test_completion_inconclusive_when_c2_canonical_unknown():
    """Nongpu rereview finding 3.3: ``C2_CANONICAL`` = UNKNOWN must block
    COMPLETE."""
    from results._phase0.verdict_schema import REQUIRED_CRITERIA, recompute_completion

    criteria = {c: "PASS" for c in REQUIRED_CRITERIA}
    criteria["C2_CANONICAL"] = "UNKNOWN"
    assert recompute_completion(criteria) == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.10: blocking_artifacts semantics wrong.
# Must list ALL undetermined required criteria (C2, REGION_PROTOTYPE,
# NUMERICAL), not just C2. Must NOT list determined single-route blockers
# (grouped NOT_SUPPORTED).
# ---------------------------------------------------------------------------


def test_blocking_artifacts_contains_all_undetermined_not_determined_single_route():
    """Nongpu rereview finding 3.10: ``blocking_artifacts`` lists artifacts for
    ALL undetermined required criteria (C2, REGION_PROTOTYPE, NUMERICAL), not
    just C2. And does NOT list determined single-route blockers (grouped
    NOT_SUPPORTED).

    Current ``_build_blocking_artifacts`` (verdict_schema.py:330-340) lists
    only C2 + grouped NOT_SUPPORTED (wrong): misses REGION_PROTOTYPE and
    NUMERICAL (undetermined), wrongly includes grouped (determined)."""
    from results._phase0.verdict_schema import recompute_derived_state

    criteria = {
        "C1": "PASS",
        "C2": "UNKNOWN",
        "C3_PLANAR_CORE": "PASS",
        "C3_PLANAR_FULL_MATRIX": "PASS",
        "C3_GROUPED": "NOT_SUPPORTED",  # determined, sinks grouped route only
        "CUTLASS_SM120_4M": "NOT_SUPPORTED",
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
        "REGION_PROTOTYPE": "UNKNOWN",  # undetermined -> must be in blocking
        "NUMERICAL": "UNKNOWN",  # undetermined -> must be in blocking
    }
    per_route = {"planar": "FAIL", "grouped": "FAIL"}
    derived = recompute_derived_state(criteria, per_route)
    blocking = " ".join(derived["blocking_artifacts"]).lower()
    # Must list REGION_PROTOTYPE (undetermined) -- currently missing.
    assert "region" in blocking, derived["blocking_artifacts"]
    # Must list NUMERICAL (undetermined) -- currently missing.
    assert "numerical" in blocking, derived["blocking_artifacts"]
    # Must NOT list grouped NOT_SUPPORTED (determined, single-route blocker).
    assert "grouped" not in blocking, derived["blocking_artifacts"]


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
