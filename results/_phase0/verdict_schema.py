"""Canonical verdict vocabulary for the Phase 0 fail-closed gates (plan Task 1 / §4).

Stdlib-only (no numpy / cupy / torch) so every reader / gate / test can import it
without GPU or heavy-dep pull-downs. This module is the SINGLE SOURCE OF TRUTH for
the canonical status tokens; readers import these sets instead of open-coding them
and never promote artifact-native detail tokens (``FEASIBLE*``,
``TILE_FUSION_FEASIBLE``, ``BLOCKED``, ``SUPPORTED``, ``NOT_FEASIBLE``) into
canonical criterion fields.

Status model (plan §4 状态模型)::

    criterion:     PASS | FAIL | UNKNOWN | NOT_RUN | NOT_SUPPORTED
    route:         VIABLE | NOT_VIABLE | UNKNOWN
    completion:    COMPLETE | INCONCLUSIVE
    authorization: GO_TO_PHASE1 | NO_GO | NOT_AUTHORIZED

artifact-native detail tokens may live in ``detail_status`` / raw artifact fields;
canonical fields use only the tokens above. ``normalize_criterion`` is the helper
that maps any artifact-native token landing in a canonical criterion field back
OUT of the canonical set (to ``UNKNOWN`` — the safe default while the reader has
not yet re-derived the canonical token from evidence).

This module is introduced in Task 0 (SDD plan). Task 0 only adds it plus its own
unit tests; the readers (c2 / numerical / manifest / gonogo / region_proto) are
rewired to actually use it in Tasks 1, 2a, 3a, 4-7.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical status token sets (plan §4 状态模型)
# ---------------------------------------------------------------------------

#: Canonical criterion-layer tokens. Any other string in a ``criterion`` field
#: means the reader has leaked an artifact-native detail token through.
CRITERION_TOKENS = frozenset(
    {
        "PASS",
        "FAIL",
        "UNKNOWN",
        "NOT_RUN",
        "NOT_SUPPORTED",
    }
)

#: Canonical route-layer tokens.
ROUTE_TOKENS = frozenset(
    {
        "VIABLE",
        "NOT_VIABLE",
        "UNKNOWN",
    }
)

#: Canonical phase0-completion tokens.
COMPLETION_TOKENS = frozenset(
    {
        "COMPLETE",
        "INCONCLUSIVE",
    }
)

#: Canonical phase1-authorization tokens.
AUTHORIZATION_TOKENS = frozenset(
    {
        "GO_TO_PHASE1",
        "NO_GO",
        "NOT_AUTHORIZED",
    }
)

# ---------------------------------------------------------------------------
# Criteria names (plan §4 criteria list)
# ---------------------------------------------------------------------------

#: Ordered list of canonical criteria names. A criterion field keyed by any of
#: these names must carry a value from ``CRITERION_TOKENS`` (after
#: normalization), never an artifact-native detail token.
CRITERIA_NAMES = (
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
)

# ---------------------------------------------------------------------------
# Numerical routes (plan §4 numerical list)
# ---------------------------------------------------------------------------

#: Canonical numerical-route names. Note the cutlass numerical route is named
#: ``cutlass_sm80_fallback`` (distinct from the capability-route names) because
#: the numerical matrix measures the SM80-fallback kernel's BF16 output, not the
#: native-SM120 kernel (which is BLOCKED for BF16 on consumer Blackwell).
NUMERICAL_ROUTES = (
    "planar",
    "grouped",
    "region_fused",
    "cutlass_sm80_fallback",
)

# ---------------------------------------------------------------------------
# Artifact-native detail tokens (must NOT appear in canonical criterion fields)
# ---------------------------------------------------------------------------

#: Tokens that may appear in raw artifact JSON or a ``detail_status`` sidecar
#: field but must NOT appear verbatim in a canonical criterion field. These are
#: the tokens that the old fail-open gates used to ``startswith("FEASIBLE")`` /
#: equality-promote into PASS; the fail-closed model maps them to UNKNOWN (the
#: reader must re-derive the canonical PASS / FAIL from evidence).
DETAIL_TOKENS = frozenset(
    {
        "FEASIBLE",
        "FEASIBLE_WITH_RECOMPUTE",
        "FEASIBLE_WITH_SM80_FALLBACK",
        "TILE_FUSION_FEASIBLE",
        "BLOCKED",
        "SUPPORTED",
        "NOT_FEASIBLE",
    }
)

#: Tokens explicitly mandated to normalize as UNKNOWN (never PASS) per plan §4
#: 验收: ``BLOCKED`` normalize 为 UNKNOWN，不是 PASS.
_UNKNOWN_DETAIL_TOKENS = frozenset(
    {
        "BLOCKED",
        "INCONCLUSIVE",
    }
)


def normalize_criterion(token):
    """Map an artifact-native token to a canonical criterion token (plan §4).

    Used by readers (Tasks 1-7) to scrub artifact-native detail tokens out of
    canonical criterion fields before they reach the gate layer.

    Mapping rules:
    - Canonical criterion tokens (``CRITERION_TOKENS``) pass through unchanged.
      In particular ``NOT_SUPPORTED`` is canonical and is preserved (cublasLt
      artifact emits ``NOT_SUPPORTED`` verbatim; the canonical criterion is the
      same string).
    - ``BLOCKED`` and ``INCONCLUSIVE`` -> ``UNKNOWN`` (plan §4 验收: BLOCKED
      is UNKNOWN, never PASS).
    - Empty string / None (missing field) -> ``UNKNOWN`` (reader for a missing
      canonical field should have produced ``NOT_RUN`` upstream, but if a raw
      missing value lands here it fails closed to UNKNOWN).
    - Any other artifact-native detail token (``FEASIBLE*``,
      ``TILE_FUSION_FEASIBLE``, ``SUPPORTED``, ``NOT_FEASIBLE``, ...) -> ``UNKNOWN``.
      These tokens belong in ``detail_status``; a canonical field carrying one
      means the reader has not done its fail-closed derivation, so the safe
      canonical value is UNKNOWN (NOT a promotion to PASS, which was the old
      ``startswith("FEASIBLE")`` fail-open behavior).
    """
    if token in CRITERION_TOKENS:
        return token
    if token in _UNKNOWN_DETAIL_TOKENS:
        return "UNKNOWN"
    if not token:  # "", None
        return "UNKNOWN"
    # All other artifact-native detail tokens: the reader must derive the
    # canonical PASS / FAIL / NOT_SUPPORTED from evidence. Promoting the detail
    # directly is exactly what plan §4 forbids. Fail-closed -> UNKNOWN.
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# §5 truth table: route / completion / authorization recompute (plan §9 Task 6)
# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for the §5 truth table. manifest.py (Task 6) uses this
# to recompute derived state from validated criteria instead of copying stale
# gonogo.json values. gonogo.py (Task 7) will be refactored to reuse this too;
# until then the gonogo-local copies are the temporary duplication (noted).

_TRI_OK = "OK"
_TRI_NOT_OK = "NOT_OK"
_TRI_UNDETERMINED = "UNDETERMINED"

#: Criteria whose determined-ness gates phase0_completion (§5 truth table).
#: NUMERICAL=FAIL is "determined" (NOT_OK) and does NOT sink completion.
#: CUTLASS_SM120_4M (native, NOT_SUPPORTED) and CUTLASS_SM80_FALLBACK_CAPABILITY
#: (fallback, PASS) are SPLIT into two independent criteria (plan §7 Task 4).
REQUIRED_CRITERIA = (
    "C1",
    "C2",
    "C3_PLANAR_CORE",
    "C3_PLANAR_FULL_MATRIX",
    "C3_GROUPED",
    "CUTLASS_SM120_4M",
    "CUTLASS_SM80_FALLBACK_CAPABILITY",
    "REGION_PROTOTYPE",
    "NUMERICAL",
)

#: Route -> capability criteria dependencies (§5 truth table rule 8 + rule 3).
#: A route is VIABLE only if every listed capability criterion normalizes to OK
#: AND its numerical criterion normalizes to OK.
#:
#: ``cutlass_4m_single`` depends on the FALLBACK capability
#: (CUTLASS_SM80_FALLBACK_CAPABILITY), NOT on CUTLASS_SM120_4M: on consumer
#: Blackwell sm_120 the route's actual kernel is the 2.x Ampere fallback (the
#: native SM120 path is architecturally BLOCKED), so the route's capability
#: tracks the path that really runs. Native failure is recorded as a separate
#: CUTLASS_SM120_4M criterion but does NOT sink the route by itself.
ROUTE_CAPABILITY_CRITERIA = {
    "planar": ("C3_PLANAR_CORE", "C3_PLANAR_FULL_MATRIX"),
    "grouped": ("C3_GROUPED",),
    "region_fused": ("REGION_PROTOTYPE", "C2_REGION_KERNEL"),
    "cutlass_4m_single": ("CUTLASS_SM80_FALLBACK_CAPABILITY",),
}

#: Ordered route names (matches ROUTE_CAPABILITY_CRITERIA keys).
RECOMPUTE_ROUTES = tuple(ROUTE_CAPABILITY_CRITERIA)


def tri_normalize(verdict):
    """Map a canonical criterion token to a gating tri-state (§5 truth table).

    OK          -> the canonical criterion is PASS (the only "established good"
                   token; plan §4 forbids promoting FEASIBLE* / SUPPORTED /
                   TILE_FUSION_FEASIBLE detail tokens to OK)
    NOT_OK      -> established as bad (canonical FAIL or NOT_SUPPORTED)
    UNDETERMINED-> not established (UNKNOWN, NOT_RUN, BLOCKED, INCONCLUSIVE,
                   any artifact-native detail token, unrecognized strings)

    Plan §4 验收: no ``startswith('FEASIBLE')`` unconditional promotion. Every
    incoming token is first scrubbed by ``normalize_criterion``, which
    fail-closes artifact-native detail tokens to canonical UNKNOWN.
    """
    canonical = normalize_criterion(verdict)
    if canonical == "PASS":
        return _TRI_OK
    if canonical in ("FAIL", "NOT_SUPPORTED"):
        return _TRI_NOT_OK
    return _TRI_UNDETERMINED


def _combine_tri(states):
    """AND-combine tri-states: any NOT_OK -> NOT_OK; else any UNDETERMINED ->
    UNDETERMINED; else OK. Empty -> UNDETERMINED."""
    if not states:
        return _TRI_UNDETERMINED
    if any(s == _TRI_NOT_OK for s in states):
        return _TRI_NOT_OK
    if any(s == _TRI_UNDETERMINED for s in states):
        return _TRI_UNDETERMINED
    return _TRI_OK


def recompute_route_verdict(criteria, per_route_numerical):
    """Per-route {route: {status, capability, numerical}} from validated
    criteria + per-route numerical map (§5 truth table rule 8).

    status is VIABLE / NOT_VIABLE / UNKNOWN; capability and numerical carry
    the raw tri-states for transparency. A route absent from
    per_route_numerical is UNDETERMINED (its numerical criterion was not
    produced or cannot be trusted).
    """
    out = {}
    for route, deps in ROUTE_CAPABILITY_CRITERIA.items():
        cap = _combine_tri([tri_normalize(criteria.get(c)) for c in deps])
        num = tri_normalize(per_route_numerical.get(route, "NOT_RUN"))
        if _TRI_NOT_OK in (cap, num):
            status = "NOT_VIABLE"
        elif _TRI_UNDETERMINED in (cap, num):
            status = "UNKNOWN"
        else:
            status = "VIABLE"
        out[route] = {"status": status, "capability": cap, "numerical": num}
    return out


def recompute_completion(criteria):
    """§5 truth table rules 1/4/5: COMPLETE iff every REQUIRED_CRITERION is
    determined (normalizes to OK or NOT_OK). Any UNKNOWN/NOT_RUN (i.e.
    UNDETERMINED) -> INCONCLUSIVE. NUMERICAL=FAIL is determined and does NOT
    sink completion."""
    for c in REQUIRED_CRITERIA:
        if tri_normalize(criteria.get(c)) == _TRI_UNDETERMINED:
            return "INCONCLUSIVE"
    return "COMPLETE"


def recompute_authorization(completion, route_verdict_map):
    """§5 truth table rule 6: GO_TO_PHASE1 iff COMPLETE and >=1 route VIABLE;
    NO_GO if COMPLETE with no viable route; NOT_AUTHORIZED if INCONCLUSIVE."""
    if completion != "COMPLETE":
        return "NOT_AUTHORIZED"
    if any(rv["status"] == "VIABLE" for rv in route_verdict_map.values()):
        return "GO_TO_PHASE1"
    return "NO_GO"


def _build_reasons(criteria, route_verdict_map, completion):
    """Human-readable explanation lines, kept in sync with the verdict."""
    reasons = []
    if completion == "INCONCLUSIVE":
        undetermined = [
            c
            for c in REQUIRED_CRITERIA
            if tri_normalize(criteria.get(c)) == _TRI_UNDETERMINED
        ]
        if undetermined:
            reasons.append(
                "canonical criteria undetermined -> phase0_completion INCONCLUSIVE: "
                + ", ".join(undetermined)
            )
    for r, rv in route_verdict_map.items():
        if rv["status"] == "NOT_VIABLE":
            reasons.append(
                f"{r} NOT_VIABLE: capability={rv['capability']} numerical={rv['numerical']}"
            )
        elif rv["status"] == "UNKNOWN":
            reasons.append(
                f"{r} UNKNOWN: capability={rv['capability']} numerical={rv['numerical']}"
            )
    return reasons


def _build_blocking_artifacts(criteria, route_verdict_map):
    """Artifact paths whose undetermined/failed state blocks a clean GO."""
    blocking = []
    if tri_normalize(criteria.get("C2")) == _TRI_UNDETERMINED:
        blocking.append("c2_judgment.json (C2_CANONICAL undetermined)")
    if tri_normalize(criteria.get("NUMERICAL")) == _TRI_NOT_OK:
        blocking.append("numerical_validation.json (overall=FAIL)")
    for r, rv in route_verdict_map.items():
        if rv["capability"] == _TRI_NOT_OK and r == "grouped":
            blocking.append("cublaslt_grouped_capability.json (NOT_SUPPORTED)")
    return blocking


def recompute_derived_state(criteria, per_route_numerical):
    """Recompute route_verdict / phase0_completion / phase1_authorization /
    reasons / blocking_artifacts from validated criteria + per-route numerical
    using the §5 truth table.

    SINGLE SOURCE OF TRUTH for the truth table. manifest.py (Task 6) uses this
    instead of copying stale gonogo.json derived state. gonogo.py (Task 7) will
    be refactored to reuse this too.

    ``per_route_numerical`` is {route: PASS|FAIL|...}. If the numerical binding
    chain is broken, the caller should pass an empty dict so every route's
    numerical tri-state is UNDETERMINED (fail-closed: cannot trust per-route
    data whose input binding is unconfirmable).
    """
    rv = recompute_route_verdict(criteria, per_route_numerical)
    completion = recompute_completion(criteria)
    authorization = recompute_authorization(completion, rv)
    reasons = _build_reasons(criteria, rv, completion)
    blocking = _build_blocking_artifacts(criteria, rv)
    return {
        "route_verdict": rv,
        "phase0_completion": completion,
        "phase1_authorization": authorization,
        "reasons": reasons,
        "blocking_artifacts": blocking,
    }


__all__ = [
    "CRITERION_TOKENS",
    "ROUTE_TOKENS",
    "COMPLETION_TOKENS",
    "AUTHORIZATION_TOKENS",
    "CRITERIA_NAMES",
    "NUMERICAL_ROUTES",
    "DETAIL_TOKENS",
    "normalize_criterion",
    # §5 truth table (plan §9 Task 6)
    "REQUIRED_CRITERIA",
    "ROUTE_CAPABILITY_CRITERIA",
    "RECOMPUTE_ROUTES",
    "TRI_OK",
    "TRI_NOT_OK",
    "TRI_UNDETERMINED",
    "tri_normalize",
    "recompute_route_verdict",
    "recompute_completion",
    "recompute_authorization",
    "recompute_derived_state",
]

# Public tri-state token aliases (used by gonogo.py which will be refactored
# in Task 7 to import these instead of defining its own).
TRI_OK = _TRI_OK
TRI_NOT_OK = _TRI_NOT_OK
TRI_UNDETERMINED = _TRI_UNDETERMINED
