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
    ROUTE_TOKENS,
    normalize_criterion,
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


# --- numerical routes: canonical names, incl. cutlass_sm80_fallback ---


def test_numerical_routes_match_plan_section4():
    """The numerical route list uses ``cutlass_sm80_fallback`` (the route
    actually measured by the numerical matrix), NOT ``cutlass_4m_single``
    (the capability-route name). The two are deliberately distinct."""
    assert set(NUMERICAL_ROUTES) == {
        "planar",
        "grouped",
        "region_fused",
        "cutlass_sm80_fallback",
    }
    assert "cutlass_4m_single" not in NUMERICAL_ROUTES


def test_numerical_routes_distinct_from_cutlass_capability_name():
    assert "cutlass_sm80_fallback" in NUMERICAL_ROUTES
    assert "cutlass_sm80_fallback" not in CRITERIA_NAMES


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


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
