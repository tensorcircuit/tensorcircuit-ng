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


__all__ = [
    "CRITERION_TOKENS",
    "ROUTE_TOKENS",
    "COMPLETION_TOKENS",
    "AUTHORIZATION_TOKENS",
    "CRITERIA_NAMES",
    "NUMERICAL_ROUTES",
    "DETAIL_TOKENS",
    "normalize_criterion",
]
