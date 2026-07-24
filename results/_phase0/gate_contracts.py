"""Single executable semantic source for every canonical reader gate
(plan Task 0 / §6).

A :class:`GateContract` is the ONE decision rule for one criterion family
(``grouped`` / ``region_peak`` / ``cutlass_native`` / ``cutlass_fallback``).
Downstream normalizers (Tasks 2 / 3 / 5) build a normalized ``raw`` dict --
whose field names ARE the cross-task interface defined by
:data:`GATE_CONTRACTS` -- and call :func:`evaluate_gate`. Readers MUST NOT
retain any undeclared PASS branch (plan §6); every PASS/FAIL/NOT_SUPPORTED
flows through this engine.

Decision model (plan Task 0 Step 4 prose + v3-review errata)::

    parse_error      -> UNKNOWN          (raw not a dict, or carries
                                          a ``parse_error`` marker)
    contradiction    -> UNKNOWN          (any contradiction_fields cond hit)
    pass_ok   = bool(c.pass_clause) and _clause_ok(c.pass_clause, raw)
    fail_ok   = any(_clause_ok(cl, raw) for cl in c.fail_clauses)        # empty -> False
    ns_ok     = any(_clause_ok(cl, raw) for cl in c.not_supported_clauses)  # empty -> False
    hits = pass_ok + fail_ok + ns_ok
    hits > 1          -> UNKNOWN          (multi-determination)
    hits == 1         -> that token (PASS / FAIL / NOT_SUPPORTED)
    hits == 0         -> UNKNOWN          (fail-closed default)

Returned ``token`` is always one of ``verdict_schema.CRITERION_TOKENS``
(``PASS`` / ``FAIL`` / ``UNKNOWN`` / ``NOT_SUPPORTED``).

``normative_policy.json`` (loaded by :func:`load_normative_policy`) stores
ONLY shared constants (``region_policy``, ``numerical_required_input_profiles``,
``cell_key_fields``); gate decision rules live in :class:`GateContract`
instances, NOT in the JSON. ``test_normative_policy_constants_only`` enforces
``"pass_clause" not in pol``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

#: A single condition ``(field_name, expected_value)``. A field absent from
#: ``raw`` is NOT satisfied (empty-safe: missing evidence cannot satisfy any
#: condition).
Cond = Tuple[str, str]

#: A clause is an AND of conditions: every condition must hold for the clause
#: to be satisfied. An empty clause is vacuously True under :func:`_clause_ok`,
#: but :func:`evaluate_gate` treats empty clause-LISTS as never-hitting
#: (empty-safe per the v3-review errata).
Clause = Tuple[Cond, ...]


# ---------------------------------------------------------------------------
# GateContract (frozen -- the single decision rule for one gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateContract:
    """Frozen executable semantic source for a single canonical gate.

    Fields:
      name: canonical gate name (key into :data:`GATE_CONTRACTS`).
      pass_clause: AND of conditions; all must hold for PASS. Empty tuple
        -> never PASS (empty-safe via
        ``bool(c.pass_clause) and _clause_ok(...)``).
      fail_clauses: OR-of-AND; any clause fully satisfied -> FAIL. Empty
        tuple -> never FAIL (empty-safe).
      not_supported_clauses: OR-of-AND; any clause fully satisfied ->
        NOT_SUPPORTED. Empty tuple -> never NOT_SUPPORTED (empty-safe).
      contradiction_fields: conditions (field, value); any matched in
        ``raw`` -> short-circuit to UNKNOWN (the raw is self-contradictory
        and cannot be trusted to carry a single canonical token).
    """

    name: str
    pass_clause: Clause
    fail_clauses: Tuple[Clause, ...]
    not_supported_clauses: Tuple[Clause, ...]
    contradiction_fields: Tuple[Cond, ...]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _clause_ok(clause, raw):
    """Return True iff every condition in the AND-clause is satisfied by raw.

    A condition ``(field, expected)`` is satisfied iff
    ``raw.get(field) == expected`` -- a field absent from ``raw`` is NOT
    satisfied (empty-safe: missing evidence cannot satisfy any condition).
    An empty clause is vacuously True; callers gate empty clause-lists
    separately (see :func:`evaluate_gate`).
    """
    for field_name, expected in clause:
        if raw.get(field_name) != expected:
            return False
    return True


def evaluate_gate(raw, c):
    """Evaluate ``raw`` against contract ``c`` and return ``(token, reason)``.

    Decision order (plan Task 0 Step 4 prose + v3-review errata):

      1. **parse_error** -- ``raw`` is not a dict, or carries a
         ``parse_error`` marker (a downstream normalizer failed to parse
         its artifact and emitted ``{"parse_error": "..."}`` instead of
         the normalized fields) -> UNKNOWN.
      2. **contradiction** -- any ``contradiction_fields`` condition
         matched in ``raw`` -> UNKNOWN (the raw is self-contradictory).
      3. Compute three booleans (empty-safe per the errata):
           * ``pass_ok = bool(c.pass_clause) and _clause_ok(c.pass_clause, raw)``
           * ``fail_ok = any(_clause_ok(cl, raw) for cl in c.fail_clauses)``
             (empty list -> False)
           * ``ns_ok = any(_clause_ok(cl, raw) for cl in c.not_supported_clauses)``
             (empty list -> False)
      4. **multi-determination** -- count hits among {pass_ok, fail_ok, ns_ok};
         ``hits > 1`` -> UNKNOWN (over-determined raw cannot be trusted to
         carry a single canonical token).
      5. **unique hit** -> that token (PASS / FAIL / NOT_SUPPORTED).
      6. **zero hits** -> default UNKNOWN (fail-closed).

    ``token`` is always one of ``verdict_schema.CRITERION_TOKENS``.
    """
    # 1. parse_error -> UNKNOWN (cannot trust input shape).
    if not isinstance(raw, dict):
        return "UNKNOWN", "parse_error: raw is not a dict"
    if raw.get("parse_error"):
        return "UNKNOWN", f"parse_error: {raw['parse_error']}"

    # 2. contradiction -> UNKNOWN (short-circuit before any PASS/FAIL check).
    for field_name, expected in c.contradiction_fields:
        if raw.get(field_name) == expected:
            return "UNKNOWN", f"contradiction: {field_name}={expected}"

    # 3. empty-safe boolean hits.
    pass_ok = bool(c.pass_clause) and _clause_ok(c.pass_clause, raw)
    fail_ok = any(_clause_ok(cl, raw) for cl in c.fail_clauses)
    ns_ok = any(_clause_ok(cl, raw) for cl in c.not_supported_clauses)

    # 4. multi-determination -> UNKNOWN.
    hits = int(pass_ok) + int(fail_ok) + int(ns_ok)
    if hits > 1:
        hit_names = []
        if pass_ok:
            hit_names.append("pass")
        if fail_ok:
            hit_names.append("fail")
        if ns_ok:
            hit_names.append("not_supported")
        return "UNKNOWN", "multi-determination: " + "+".join(hit_names) + " both hit"

    # 5. unique hit -> token.  6. zero hits -> default UNKNOWN (fail-closed).
    if pass_ok:
        return "PASS", "pass_clause satisfied"
    if fail_ok:
        return "FAIL", "fail_clause satisfied"
    if ns_ok:
        return "NOT_SUPPORTED", "not_supported_clause satisfied"
    return "UNKNOWN", "default: no clause hit (fail-closed)"


# ---------------------------------------------------------------------------
# The 4 frozen contracts (cross-task interface -- downstream normalizers
# emit exactly these field names).
# ---------------------------------------------------------------------------

#: Grouped (C3_GROUPED) gate. A real PASS needs every stage from schema
#: validation through consistency all green; FAIL is OR-of-AND over any
#: single failed stage; NOT_SUPPORTED requires authoritative API absence
#: (``api_state=ABSENT_DEFINITIVE``) WITH a recognized probe source AND an
#: actual attempt (the API was confirmed absent, not merely unprobed).
#: ``consistency_state=CONFLICT`` is a contradiction (not a FAIL): the raw
#: disagrees with itself and cannot be trusted.
GROUPED = GateContract(
    name="grouped",
    pass_clause=(
        ("schema_state", "VALID"),
        ("api_state", "PRESENT"),
        ("attempt_state", "ATTEMPTED"),
        ("compile_state", "SUCCEEDED"),
        ("run_state", "SUCCEEDED"),
        ("correctness_state", "PASSED"),
        ("coverage_state", "COMPLETE"),
        ("consistency_state", "CONSISTENT"),
    ),
    fail_clauses=(
        (("schema_state", "MISSING"),),
        (("api_state", "ABSENT_INCONCLUSIVE"),),
        (("attempt_state", "NOT_ATTEMPTED"),),
        (("compile_state", "FAILED"),),
        (("run_state", "FAILED"),),
        (("correctness_state", "FAILED"),),
        (("coverage_state", "INCOMPLETE"),),
    ),
    not_supported_clauses=(
        (
            ("api_state", "ABSENT_DEFINITIVE"),
            ("attempt_state", "ATTEMPTED"),
            ("probe_source_state", "RECOGNIZED"),
        ),
    ),
    contradiction_fields=(("consistency_state", "CONFLICT"),),
)

#: Region peak (C2 region kernel feasibility / REGION_PROTOTYPE) gate.
#: A real PASS requires MEASURED (not model-only) evidence, an approved
#: method, full-anchor PTE scope, OK sample/peak/gain, a real full-anchor
#: run, matched case binding, and consistency. FAIL is substantive only:
#: peak reduction negative or below the 256 MiB policy threshold
#: (``min_gain_bytes`` from ``normative_policy.json``). NOT_SUPPORTED is
#: empty (region has no NOT_SUPPORTED path -- empty-safe). Scope mismatch
#: and consistency conflict are contradictions. ``case_binding_state=MISSING``
#: yields no hit -> default UNKNOWN (unverified binding cannot PASS).
REGION_PEAK = GateContract(
    name="region_peak",
    pass_clause=(
        ("schema_state", "VALID"),
        ("evidence_class_state", "MEASURED"),
        ("method_state", "APPROVED"),
        ("scope_state", "FULL_ANCHOR_PTE"),
        ("sample_state", "OK"),
        ("peak_state", "OK"),
        ("gain_state", "OK"),
        ("full_anchor_run_state", "TRUE"),
        ("case_binding_state", "MATCH"),
        ("consistency_state", "CONSISTENT"),
    ),
    fail_clauses=(
        (("gain_state", "NEGATIVE"),),
        (("gain_state", "BELOW_POLICY"),),
    ),
    not_supported_clauses=(),
    contradiction_fields=(
        ("scope_state", "MISMATCH"),
        ("consistency_state", "CONFLICT"),
    ),
)

#: Native cutlass SM120 4M gate. PASS needs every stage green; FAIL is
#: OR-of-AND over compile/run/correctness/coverage failure. NOT_SUPPORTED
#: requires a REAL captured native blocker (``blocker_state=PRESENT``) WITH
#: a recognized source (``blocker_source_state=RECOGNIZED``) -- the two
#: not_supported fields are DISJOINT from the pass fields, so a real
#: double-hit (pass AND not_supported simultaneously) is structurally
#: possible and exercised by ``test_real_multi_determination_double_hit``.
#: ``consistency_state=CONFLICT`` is a contradiction.
CUTLASS_NATIVE = GateContract(
    name="cutlass_native",
    pass_clause=(
        ("schema_state", "VALID"),
        ("attempt_state", "ATTEMPTED"),
        ("compile_state", "SUCCEEDED"),
        ("run_state", "SUCCEEDED"),
        ("correctness_state", "PASSED"),
        ("coverage_state", "COMPLETE"),
        ("consistency_state", "CONSISTENT"),
    ),
    fail_clauses=(
        (("compile_state", "FAILED"),),
        (("run_state", "FAILED"),),
        (("correctness_state", "FAILED"),),
        (("coverage_state", "INCOMPLETE"),),
    ),
    not_supported_clauses=(
        (
            ("blocker_state", "PRESENT"),
            ("blocker_source_state", "RECOGNIZED"),
        ),
    ),
    contradiction_fields=(("consistency_state", "CONFLICT"),),
)

#: SM80 fallback gate. The fallback path is the one that actually runs on
#: consumer Blackwell sm_120 (the native SM120 path is BLOCKED for BF16),
#: so its PASS is the load-bearing capability for the
#: ``cutlass_4m_single`` route. Note ``compile_state=OK`` here (not
#: ``SUCCEEDED``) per Task 5's test ``compile_status=="OK"`` -- the
#: fallback compile is a softer check. NOT_SUPPORTED is empty (empty-safe).
CUTLASS_FALLBACK = GateContract(
    name="cutlass_fallback",
    pass_clause=(
        ("attempt_state", "ATTEMPTED"),
        ("compile_state", "OK"),
        ("run_state", "SUCCEEDED"),
        ("correctness_state", "PASSED"),
        ("coverage_state", "COMPLETE"),
    ),
    fail_clauses=(
        (("run_state", "FAILED"),),
        (("correctness_state", "FAILED"),),
    ),
    not_supported_clauses=(),
    contradiction_fields=(),
)

#: SINGLE SOURCE OF TRUTH -- the four canonical gate contracts, keyed by
#: the names downstream normalizers (Tasks 2 / 3 / 5) pass to
#: :func:`evaluate_gate`.
GATE_CONTRACTS = {
    "grouped": GROUPED,
    "region_peak": REGION_PEAK,
    "cutlass_native": CUTLASS_NATIVE,
    "cutlass_fallback": CUTLASS_FALLBACK,
}


# ---------------------------------------------------------------------------
# normative_policy.json loader (shared constants only -- no gate rules)
# ---------------------------------------------------------------------------

_POLICY_PATH = Path(__file__).resolve().parent / "normative_policy.json"


def load_normative_policy():
    """Load shared normative constants from ``normative_policy.json``.

    The JSON stores ONLY shared constants (``region_policy``,
    ``numerical_required_input_profiles``, ``cell_key_fields``); gate
    decision rules (``pass_clause`` / ``fail_clauses`` /
    ``not_supported_clauses``) live in :class:`GateContract` instances, NOT
    in the JSON. ``test_normative_policy_constants_only`` enforces
    ``"pass_clause" not in pol``.

    Returns a dict. Loaded fresh each call (the file is small and this
    keeps tests that monkeypatch the path honest).
    """
    with _POLICY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "Cond",
    "Clause",
    "GateContract",
    "evaluate_gate",
    "GATE_CONTRACTS",
    "GROUPED",
    "REGION_PEAK",
    "CUTLASS_NATIVE",
    "CUTLASS_FALLBACK",
    "load_normative_policy",
]
