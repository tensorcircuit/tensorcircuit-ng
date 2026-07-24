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

#: Ordered list of canonical criteria names (plan §4.1 / §1.1 -- SINGLE SOURCE OF
#: TRUTH). CRITERIA_NAMES, REQUIRED_CRITERIA, gonogo output, manifest required
#: map, and Markdown ALL derive from this one set. A criterion field keyed by
#: any of these names must carry a value from ``CRITERION_TOKENS`` (after
#: normalization), never an artifact-native detail token.
#:
#: The old ``"C2"`` alias is NOT in this list -- it is a compat-only alias for
#: ``C2_CANONICAL`` (see ``C2_COMPAT_ALIAS``) and must NOT participate in
#: completion / route / authorization gates.
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
    "REGION_PROTOTYPE",
    "NUMERICAL",
)

#: Compat alias for the old ``"C2"`` criterion key (plan §1.2). Kept for
#: backward compatibility with gonogo/manifest output consumers that read the
#: ``"C2"`` key. After ``validate_criteria`` it always equals
#: ``C2_CANONICAL``. It must NOT participate in completion / route /
#: authorization gates (only the 4 real C2 layers do).
C2_COMPAT_ALIAS = "C2"

#: The 3 C2 input layers that roll up into ``C2_CANONICAL`` (plan §1.4).
C2_INPUT_LAYERS = (
    "C2_REGION_KERNEL_FEASIBILITY",
    "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK",
    "C2_JOINT_EXECUTABLE_LEVERAGE",
)

#: All recognized criterion keys: canonical names + the C2 compat alias.
RECOGNIZED_CRITERIA_KEYS = frozenset(CRITERIA_NAMES) | {C2_COMPAT_ALIAS}

# ---------------------------------------------------------------------------
# Numerical routes (plan §4 numerical list)
# ---------------------------------------------------------------------------

#: Canonical numerical-route names. These are ROUTE KEYS (matching
#: ``ROUTE_CAPABILITY_CRITERIA`` keys), not criterion names. The numerical
#: matrix measures BF16 output per route; the cutlass numerical route is keyed
#: ``cutlass_4m_single`` (the same key as the capability route) -- the route's
#: actual kernel on consumer Blackwell sm_120 is the 2.x Ampere (SM80) fallback
#: (the native SM120 path is BLOCKED for BF16), but the route KEY is
#: ``cutlass_4m_single`` to match the capability route, NOT a separate
#: ``cutlass_sm80_fallback`` name.
NUMERICAL_ROUTES = (
    "planar",
    "grouped",
    "region_fused",
    "cutlass_4m_single",
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
# C2 canonical rollup + schema-v3 criteria validation (plan §1.3 / §1.4)
# ---------------------------------------------------------------------------


def rollup_c2_canonical(criteria):
    """Roll up the 3 C2 input layers into a canonical C2 status (plan §1.4).

    ``C2_CANONICAL`` must equal this rollup. The rollup rules mirror
    ``gonogo._roll_up_statuses``: any FAIL -> FAIL; any UNKNOWN -> UNKNOWN;
    any NOT_RUN -> NOT_RUN; all PASS -> PASS; anything else (e.g.
    NOT_SUPPORTED in a layer) -> UNKNOWN (the canonical cannot be determined).

    Each input layer is first scrubbed by ``normalize_criterion`` so that
    artifact-native detail tokens (which should never reach this point after
    ``validate_criteria``) are treated as UNKNOWN.
    """
    statuses = [
        normalize_criterion(criteria.get(layer, "NOT_RUN")) for layer in C2_INPUT_LAYERS
    ]
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "UNKNOWN" for s in statuses):
        return "UNKNOWN"
    if any(s == "NOT_RUN" for s in statuses):
        return "NOT_RUN"
    if all(s == "PASS" for s in statuses):
        return "PASS"
    return "UNKNOWN"


def validate_criteria(criteria):
    """Schema-v3 validation of a criteria dict before it reaches the truth
    table (plan §1.3 / §1.4).

    Returns ``(validated, reasons)`` where ``validated`` is a new dict and
    ``reasons`` is a list of human-readable validation notes.

    Steps:
      1. **Unknown keys** -- keys not in ``CRITERIA_NAMES`` and not the
         ``C2_COMPAT_ALIAS`` are dropped (a reason is recorded).
      2. **Missing required** -- every ``REQUIRED_CRITERION`` absent from the
         input is added as ``NOT_RUN``.
      3. **Token validation** -- each value is scrubbed by
         ``normalize_criterion``; detail tokens (SUPPORTED / FEASIBLE* /
         TILE_FUSION_FEASIBLE / NOT_FEASIBLE / BLOCKED / INCONCLUSIVE) are
         downgraded to ``UNKNOWN`` (a reason is recorded).
      4. **C2_CANONICAL rollup** -- ``C2_CANONICAL`` is validated against
         ``rollup_c2_canonical``; if inconsistent (or was missing), it is set
         to ``UNKNOWN`` (a reason is recorded).
      5. **C2 compat alias** -- ``C2`` is set to ``C2_CANONICAL`` so the alias
         always tracks the canonical value. The alias is NOT in
         ``REQUIRED_CRITERIA`` and does NOT participate in gates.
    """
    validated = {}
    reasons = []

    for key, value in criteria.items():
        if key not in RECOGNIZED_CRITERIA_KEYS:
            reasons.append(f"unknown criterion key '{key}' removed from output")
            continue
        normalized = normalize_criterion(value)
        if normalized != value and value not in (None, ""):
            reasons.append(
                f"criterion '{key}' detail token '{value}' downgraded to UNKNOWN"
            )
        validated[key] = normalized

    # Add missing required criteria as NOT_RUN.
    for c in REQUIRED_CRITERIA:
        if c not in validated:
            validated[c] = "NOT_RUN"

    # Validate C2_CANONICAL against the rollup of the 3 C2 input layers.
    c2_rollup = rollup_c2_canonical(validated)
    provided = validated.get("C2_CANONICAL")
    if provided != c2_rollup:
        reasons.append(
            f"C2_CANONICAL={provided} != rollup={c2_rollup} -> downgraded to UNKNOWN"
        )
        validated["C2_CANONICAL"] = "UNKNOWN"

    # Set the C2 compat alias = C2_CANONICAL (must NOT participate in gates).
    validated[C2_COMPAT_ALIAS] = validated["C2_CANONICAL"]

    return validated, reasons


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

#: Criteria whose determined-ness gates phase0_completion (§5 truth table /
#: plan §1.4). Derived from the SINGLE SOURCE OF TRUTH (``CRITERIA_NAMES``) --
#: every canonical criterion is required. The old ``"C2"`` alias is NOT here
#: (plan §1.2: alias must NOT participate in gates; only the 4 real C2 layers
#: do). NUMERICAL=FAIL is "determined" (NOT_OK) and does NOT sink completion.
#: CUTLASS_SM120_4M (native, NOT_SUPPORTED) and CUTLASS_SM80_FALLBACK_CAPABILITY
#: (fallback, PASS) are SPLIT into two independent criteria (plan §7 Task 4).
REQUIRED_CRITERIA = CRITERIA_NAMES  # all 12 canonical criteria are required

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
    "region_fused": ("REGION_PROTOTYPE", "C2_REGION_KERNEL_FEASIBILITY"),
    "cutlass_4m_single": ("CUTLASS_SM80_FALLBACK_CAPABILITY",),
}

#: Ordered route names (matches ROUTE_CAPABILITY_CRITERIA keys).
RECOMPUTE_ROUTES = tuple(ROUTE_CAPABILITY_CRITERIA)

#: Required-criterion -> blocking-artifact reporting map (plan §11). Used by
#: ``_build_blocking_artifacts`` to list the artifact(s) behind each required
#: criterion. The 4 C2 layers (``C2_INPUT_LAYERS`` + ``C2_CANONICAL``) share the
#: same c2_judgment.json + c2_checkpoint_manifest.json chain, so they are
#: collapsed into a SINGLE ``C2_CANONICAL`` entry -- otherwise an undetermined
#: C2 family would duplicate the shared chain 4x. Non-C2 criteria each map to
#: their own artifact.
#:
#: Each entry is ``(label, undetermined_triggers, not_ok_criterion, artifact)``:
#:   * ``undetermined_triggers`` -- criteria checked for UNKNOWN/NOT_RUN (rule 1).
#:     For the C2 family, ANY layer undetermined lists the shared chain once.
#:   * ``not_ok_criterion`` -- the single criterion checked for determined
#:     NOT_OK at COMPLETE + all-routes-NOT_VIABLE (rule 2). For the C2 family
#:     this is ``C2_CANONICAL`` (the rollup), whose FAIL at COMPLETE means the
#:     whole C2 chain is definitively bad.
#:
#: This is the canonical criterion->artifact map for BLOCKING reporting. Both
#: gonogo (``aggregate_two_layer``) and manifest (``build_manifest``) derive
#: blocking through ``recompute_derived_state`` -> ``_build_blocking_artifacts``
#: using THIS map, so they report identical blockers (DRY -- single helper,
#: single map). manifest's ``REQUIRED_ARTIFACTS`` is a richer superset (full
#: per-criterion file lists for presence-gating) and stays in sync because both
#: derive from the canonical ``CRITERIA_NAMES``.
CRITERION_BLOCKING_ARTIFACTS = (
    (
        "C2_CANONICAL",
        C2_INPUT_LAYERS + ("C2_CANONICAL",),
        "C2_CANONICAL",
        "c2_judgment.json / c2_checkpoint_manifest.json",
    ),
    ("C1", ("C1",), "C1", "c1_judgment.json"),
    (
        "C3_PLANAR_CORE",
        ("C3_PLANAR_CORE",),
        "C3_PLANAR_CORE",
        "cublaslt_planar_capability.json",
    ),
    (
        "C3_PLANAR_FULL_MATRIX",
        ("C3_PLANAR_FULL_MATRIX",),
        "C3_PLANAR_FULL_MATRIX",
        "cublaslt_full_matrix.csv",
    ),
    ("C3_GROUPED", ("C3_GROUPED",), "C3_GROUPED", "cublaslt_grouped_capability.json"),
    (
        "CUTLASS_SM120_4M",
        ("CUTLASS_SM120_4M",),
        "CUTLASS_SM120_4M",
        "cutlass_sm120_4m.json",
    ),
    (
        "CUTLASS_SM80_FALLBACK_CAPABILITY",
        ("CUTLASS_SM80_FALLBACK_CAPABILITY",),
        "CUTLASS_SM80_FALLBACK_CAPABILITY",
        "cutlass_sm120_4m.json",
    ),
    (
        "REGION_PROTOTYPE",
        ("REGION_PROTOTYPE",),
        "REGION_PROTOTYPE",
        "region_prototype.json",
    ),
    ("NUMERICAL", ("NUMERICAL",), "NUMERICAL", "numerical_validation.json"),
)


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
    """Artifact paths whose undetermined/failed state blocks a clean GO
    (plan §11 / nongpu-rereview §3.10).

    Lists ONLY real blockers:

      1. Artifacts causing a required criterion to be UNKNOWN/NOT_RUN
         (UNDETERMINED) -- these keep ``phase0_completion`` INCONCLUSIVE and
         are the real completion blockers. In the current honest state this is
         C2 (C2_CANONICAL undetermined), REGION_PROTOTYPE and NUMERICAL.
      2. At COMPLETE (no UNDETERMINED required criterion), if EVERY route is
         NOT_VIABLE, the deterministic global blockers (determined NOT_OK
         criteria) that make all routes NOT_VIABLE.

    Does NOT list:

      * A determined result that makes only a SINGLE route NOT_VIABLE (e.g.
        grouped NOT_SUPPORTED sinks the grouped route but doesn't block other
        routes or completion) -- unless COMPLETE + all routes NOT_VIABLE.
      * A determined capability that doesn't affect completion (e.g. NUMERICAL
        FAIL is determined and does NOT sink completion; it only sinks routes
        via per-route numerical, so it is not a completion blocker).

    The 4 C2 layers share the c2_judgment.json + c2_checkpoint_manifest.json
    chain and are collapsed into a single C2_CANONICAL entry (via
    ``CRITERION_BLOCKING_ARTIFACTS``) so the shared chain is not duplicated 4x.
    """
    blocking = []
    # Rule 1: artifacts for required criteria that are UNKNOWN/NOT_RUN.
    for label, triggers, _not_ok, artifact in CRITERION_BLOCKING_ARTIFACTS:
        if any(tri_normalize(criteria.get(c)) == _TRI_UNDETERMINED for c in triggers):
            blocking.append(f"{artifact} ({label} undetermined)")
    # Rule 2: at COMPLETE (no undetermined required criteria), if ALL routes
    # are NOT_VIABLE, list the deterministic global blockers (determined
    # NOT_OK). A single route's NOT_VIABLE is NOT a global blocker -- it
    # doesn't block other routes or completion, so it is only surfaced here
    # when every route is sunk.
    if (
        not blocking
        and route_verdict_map
        and all(rv["status"] == "NOT_VIABLE" for rv in route_verdict_map.values())
    ):
        for (
            label,
            _triggers,
            not_ok_criterion,
            artifact,
        ) in CRITERION_BLOCKING_ARTIFACTS:
            if tri_normalize(criteria.get(not_ok_criterion)) == _TRI_NOT_OK:
                blocking.append(f"{artifact} ({label} NOT_OK)")
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
    "REQUIRED_CRITERIA",
    "C2_COMPAT_ALIAS",
    "C2_INPUT_LAYERS",
    "RECOGNIZED_CRITERIA_KEYS",
    "NUMERICAL_ROUTES",
    "DETAIL_TOKENS",
    "normalize_criterion",
    "rollup_c2_canonical",
    "validate_criteria",
    # §5 truth table (plan §9 Task 6)
    "ROUTE_CAPABILITY_CRITERIA",
    "RECOMPUTE_ROUTES",
    "CRITERION_BLOCKING_ARTIFACTS",
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
