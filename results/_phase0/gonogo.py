"""Two-layer Phase 0 aggregator (review §10 / final-remediation plan §13).

route_verdict (per-route capability AND numerical) + phase0_completion (all
canonical criteria determined) -> phase1_authorization. Emits gonogo.json /
gonogo.md / environment.json under results/phase0/. md is generated FROM the
json object, never hand-overwritten.

Task 7: gonogo is the CRITERIA PRODUCER (reads gate artifacts -> native
canonical criteria). The route/completion/authorization derivation goes through
``verdict_schema.recompute_derived_state`` -- the SINGLE SOURCE OF TRUTH for the
§5 truth table -- so gonogo and manifest derivation cannot diverge (the
derivation logic is shared; inputs/binding-validation differ -- gonogo trusts
per_route directly while manifest fail-closes on binding break). JSON + Markdown
render from one canonical gonogo-v2 object (no divergent code paths).
"""

from __future__ import annotations

import csv
import json
import os

from results._phase0.gate_contracts import GATE_CONTRACTS, evaluate_gate
from results._phase0.verdict_schema import recompute_derived_state, validate_criteria

VERDICTS = (
    "GO_TO_PHASE1",
    "NO_GO_NO_WINDOW",
    "NO_GO_NOT_COVERABLE",
    "NO_GO_KERNEL",
    "INCONCLUSIVE",
)

# Canonical criterion tokens used by the criterion producers (gate-artifact
# readers). These are string aliases for the canonical CRITERION_TOKENS set
# (verdict_schema.CRITERION_TOKENS); they are NOT truth-table logic (the truth
# table lives in verdict_schema.recompute_derived_state, reused below).
_OK = "PASS"
_BAD = "FAIL"
_UNKNOWN = "UNKNOWN"
_NOT_RUN = "NOT_RUN"


def aggregate_two_layer(criteria, per_route_numerical):
    """Compose the full gonogo-v2 verdict object from criteria + per-route
    numerical via the shared §5 truth table
    (``verdict_schema.recompute_derived_state``).

    Task 1 (plan §1.3): before reaching the truth table, ``criteria`` is
    validated by ``verdict_schema.validate_criteria`` -- unknown keys are
    dropped, missing required criteria are added as NOT_RUN, detail tokens are
    downgraded to UNKNOWN, and ``C2_CANONICAL`` is validated against the rollup
    of the 3 C2 input layers. The ``C2`` compat alias is set to
    ``C2_CANONICAL``. The validated criteria (not the raw input) appear in the
    output and feed the truth-table derivation.

    Task 7: the route_verdict / phase0_completion / phase1_authorization /
    reasons / blocking_artifacts derivation goes through the shared helper so
    gonogo and manifest derivation cannot diverge (shared derivation logic;
    inputs/binding-validation differ -- gonogo trusts per_route directly while
    manifest fail-closes on binding break). JSON + Markdown render from THIS
    object (truth-table rule 7). ``per_route_numerical`` is {route: PASS|FAIL|...};
    a route absent from it is UNDETERMINED (fail-closed).
    """
    validated, validation_reasons = validate_criteria(criteria)
    derived = recompute_derived_state(validated, per_route_numerical)
    return {
        "schema_version": "gonogo-v2",
        "criteria": dict(validated),
        "route_verdict": derived["route_verdict"],
        "phase0_completion": derived["phase0_completion"],
        "phase1_authorization": derived["phase1_authorization"],
        "reasons": derived["reasons"],
        "blocking_artifacts": derived["blocking_artifacts"],
        "validation_notes": validation_reasons,
    }


def _render_md(agg):
    """Render gonogo.md FROM the same object as gonogo.json (truth-table rule 7)."""
    lines = [
        "# Phase 0 Go/No-Go (two-layer, §10 / plan §13)",
        "",
        f"**phase0_completion: {agg['phase0_completion']}**",
        f"**phase1_authorization: {agg['phase1_authorization']}**",
        "",
        "## Route verdict",
        "",
    ]
    for r, rv in agg["route_verdict"].items():
        lines.append(
            f"- `{r}`: **{rv['status']}** (capability={rv['capability']}, numerical={rv['numerical']})"
        )
    lines += [
        "",
        "## Criteria",
        "```json",
        json.dumps(agg["criteria"], indent=2),
        "```",
    ]
    if agg["reasons"]:
        lines += ["", "## Reasons"]
        lines += [f"- {x}" for x in agg["reasons"]]
    if agg["blocking_artifacts"]:
        lines += ["", "## Blocking artifacts"]
        lines += [f"- {x}" for x in agg["blocking_artifacts"]]
    return "\n".join(lines) + "\n"


def aggregate(c1, c2, c3_planar, c3_real_ceiling_ratio=None):
    """§9 four-state truth table.

    c3_planar is authoritative for criterion 3; c3_real_ceiling_ratio is auxiliary
    (a real-BF16 GEMM ceiling proxy only — it cannot stand in for the planar-complex
    libcublasLt probe, which is Plan B).

    A definitive FAIL on C1 or C2 short-circuits to a NO_GO even if C3_planar is
    still NOT_RUN (a hard fail is not masked by an unprobed later criterion). Only
    UNKNOWN, or NOT_RUN with no upstream FAIL, defers to INCONCLUSIVE.
    """
    c3 = c3_planar
    criteria = {
        "C1": c1,
        "C2": c2,
        "C3_planar": c3,
        "C3_real_ceiling_ratio": c3_real_ceiling_ratio,
    }
    if c1 == _BAD:
        v = "NO_GO_NO_WINDOW"
    elif c1 == _OK and c2 == _BAD:
        v = "NO_GO_NOT_COVERABLE"
    elif c1 == _OK and c2 == _OK and c3 == _BAD:
        v = "NO_GO_KERNEL"
    elif c1 == _OK and c2 == _OK and c3 == _OK:
        v = "GO_TO_PHASE1"
    elif c1 == _UNKNOWN or c2 == _UNKNOWN or c3 == _UNKNOWN or c3 == _NOT_RUN:
        v = "INCONCLUSIVE"
    else:
        v = "INCONCLUSIVE"
    note = _verdict_note(v, c1, c2, c3)
    return {
        "verdict": v,
        "criteria": criteria,
        "note": note,
    }


def _verdict_note(verdict, c1, c2, c3):
    """Human-readable explanation of why the truth table produced `verdict`.

    Kept in sync with the §9 truth table so gonogo.{json,md} never contradict
    the verdict (the prior static note asserted C3_planar=NOT_RUN regardless
    of the actual C3_planar status, which became stale once Plan B wired the
    cublasLt capability artifact into main()).
    """
    if verdict == "GO_TO_PHASE1":
        return "C1 PASS + C2 PASS + C3_planar PASS (cublasLt planar-complex SUPPORTED)"
    if verdict == "NO_GO_KERNEL":
        return "C3_planar FAIL (cublasLt planar-complex NOT_SUPPORTED) — kernel path infeasible"
    if verdict == "NO_GO_NOT_COVERABLE":
        return (
            "C1 PASS but C2 FAIL (large buffers not tile-coverable with net byte gain)"
        )
    if verdict == "NO_GO_NO_WINDOW":
        return "C1 FAIL (no BF16 materialization window found)"
    # INCONCLUSIVE: a criterion is UNKNOWN or (C3_planar) NOT_RUN without any hard FAIL.
    if c3 == _NOT_RUN and c1 == _OK and c2 == _OK:
        return (
            "C3_planar=NOT_RUN (cublasLt capability artifact absent) — pending Plan B"
        )
    return "A criterion is UNKNOWN or NOT_RUN; pending a definitive PASS/FAIL"


def _roll_up_statuses(statuses):
    """Combine per-case statuses into one criterion status.

    Any FAIL  -> FAIL; else any UNKNOWN -> UNKNOWN; else any NOT_RUN -> NOT_RUN;
    else all PASS -> PASS; empty -> NOT_RUN.
    """
    if not statuses:
        return _NOT_RUN
    if any(s == _BAD for s in statuses):
        return _BAD
    if any(s == _UNKNOWN for s in statuses):
        return _UNKNOWN
    if any(s == _NOT_RUN for s in statuses):
        return _NOT_RUN
    if all(s == _OK for s in statuses):
        return _OK
    return _UNKNOWN


def _c1_status_from_judgment(data):
    """c1_judgment.json is {case_key: {..., "judgment": {"status": ...}}}."""
    if not isinstance(data, dict) or not data:
        return _NOT_RUN
    statuses = []
    for case in data.values():
        if isinstance(case, dict):
            statuses.append(case.get("judgment", {}).get("status", _UNKNOWN))
        else:
            statuses.append(_UNKNOWN)
    return _roll_up_statuses(statuses)


def _c2_status_from_judgment(data):
    """c2_judgment.json is {case_key: {..., "status": ...}} (Task 7 integration verdict)."""
    if not isinstance(data, dict) or not data:
        return _NOT_RUN
    statuses = []
    for case in data.values():
        if isinstance(case, dict):
            statuses.append(case.get("status", _UNKNOWN))
        else:
            statuses.append(_UNKNOWN)
    return _roll_up_statuses(statuses)


def _c3_planar_from_capability(path):
    """Read the cublasLt planar-complex capability artifact (Plan B Task 2).

    Artifact shape: {"capability": {"status": "SUPPORTED"|"NOT_SUPPORTED", ...}}.
    Returns PASS for SUPPORTED, FAIL for NOT_SUPPORTED, NOT_RUN if the artifact
    is absent (Plan B not yet run). Any unparseable / malformed artifact is
    treated as UNKNOWN-deferred (NOT_RUN) so the gonogo falls back to
    INCONCLUSIVE rather than masking a hard C1/C2 FAIL.
    """
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _NOT_RUN
    status = (data.get("capability") or {}).get("status")
    if status == "SUPPORTED":
        return _OK
    if status == "NOT_SUPPORTED":
        return _BAD
    return _UNKNOWN


def _c2_layer_status(data, layer):
    """Read one C2 sub-layer status from c2_judgment.json's nested cases.

    Each case is {layers: {<LAYER>: "PASS"|"FAIL"|"UNKNOWN", ...}}. We roll up
    across cases (any FAIL -> FAIL, any UNKNOWN -> UNKNOWN, else PASS) so the
    region-kernel sub-criterion (rule 3) reflects the worst case. Empty or
    malformed -> UNKNOWN (never default PASS).
    """
    if not isinstance(data, dict) or not data:
        return _UNKNOWN
    statuses = []
    for case in data.values():
        if isinstance(case, dict):
            layers = case.get("layers") or {}
            statuses.append(layers.get(layer, _UNKNOWN))
        else:
            statuses.append(_UNKNOWN)
    return _roll_up_statuses(statuses)


def _c3_planar_full_matrix_status(path, contraction_shapes_path=None):
    """C3 planar full-matrix completeness (Task 6 sweep artifact) -- STRICT
    128-cell validator (Task 5: plan §8 / spec §3.5).

    Derives the expected (M,N,K,out_dtype,ws_cap,op) cell key set from
    contraction_shapes.csv + the producer's matrix grid
    (cublaslt.FULL_MATRIX_*/full_matrix_expected_keys), then enforces the full
    matrix contract:

      * exact header (cublaslt._FULL_MATRIX_HEADER)
      * no duplicate (M,N,K,out_dtype,ws_cap,op) cell key
      * every expected cell present, no unexpected cell
      * legal dtype / ws_cap / op / aligned / status tokens
      * aligned matches the recomputed ``m%16==n%16==k%16==0`` invariant
      * (M,N,K) bound to contraction_shapes.csv (no shape drift)
      * algorithm-column legality: algo_count / first_algo_id /
        workspace_bytes are integers, in range, and consistent with status
        (ok: algo_count>=1 + first_algo_id>=0 + 0<=workspace_bytes<=cap;
        no-algo: algo_count==0 + first_algo_id==-1 + workspace_bytes==0)
      * status='no-algo' allowed ONLY on cublaslt.full_matrix_no_algo_policy()
        cells (explicit 8-cell policy, not error-swallowing)

    Any violation -> UNKNOWN (never PASS). NOT_RUN only when the artifact itself
    is absent. Pure-function: no GPU, no extension load.

    The 8 legitimate no-algo cells (OP_T on shape 262144x64x4 across 2 dtypes x
    4 workspace caps) are the cuBLASLt sweep's genuine zero-algorithm results;
    a no-algo anywhere else is a real coverage gap -> UNKNOWN.
    """
    if not os.path.exists(path):
        return _NOT_RUN
    # Resolve the contraction-shapes source for expected-cell derivation. Both
    # the producer (run_full_matrix) and the committed artifact live alongside
    # contraction_shapes.csv under results/phase0/.
    if contraction_shapes_path is None:
        contraction_shapes_path = os.path.join(
            os.path.dirname(path), "contraction_shapes.csv"
        )
    try:
        from results._phase0 import cublaslt as _cublaslt
    except Exception:
        # Any import-time failure (missing module, missing numpy, etc.) is
        # fail-closed: we cannot derive the expected-cell contract -> UNKNOWN.
        return _UNKNOWN

    # Derive the expected shape set with the SAME filter the producer uses
    # (load_c1_c2_shapes: bytes >= 64 MiB), deduped by (M,N,K).
    try:
        raw_shapes = _cublaslt.load_c1_c2_shapes(contraction_shapes_path)
    except (OSError, ValueError):
        # contraction_shapes.csv absent/unreadable -> cannot derive the
        # expected-cell contract -> fail-closed UNKNOWN.
        return _UNKNOWN
    if not raw_shapes:
        return _UNKNOWN
    seen, expected_shapes = set(), []
    expected_shape_keys = set()
    for s in raw_shapes:
        key = (s["M"], s["N"], s["K"])
        if key not in seen:
            seen.add(key)
            expected_shapes.append(s)
            expected_shape_keys.add(key)
    expected_keys = set(_cublaslt.full_matrix_expected_keys(expected_shapes))
    no_algo_policy = _cublaslt.full_matrix_no_algo_policy()
    header = list(_cublaslt._FULL_MATRIX_HEADER)
    ws_cap_bytes = dict(_cublaslt.FULL_MATRIX_WS_CAPS)

    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except (OSError, ValueError):
        return _UNKNOWN
    if len(rows) < 2:
        return _UNKNOWN  # header-only / empty
    if rows[0] != header:
        return _UNKNOWN  # schema / header drift

    actual_keys = set()
    for row in rows[1:]:
        if len(row) != len(header):
            return _UNKNOWN  # malformed row (wrong column count)
        rec = dict(zip(header, row))
        try:
            m, n, k = int(rec["M"]), int(rec["N"]), int(rec["K"])
        except ValueError:
            return _UNKNOWN  # non-integer M/N/K
        od, ws, op = rec["out_dtype"], rec["ws_cap"], rec["op"]
        key = (m, n, k, od, ws, op)
        # duplicate cell key -> broken sweep / re-run contamination
        if key in actual_keys:
            return _UNKNOWN
        actual_keys.add(key)
        # shape binding: (M,N,K) must be one of the expected contraction shapes
        if (m, n, k) not in expected_shape_keys:
            return _UNKNOWN  # shape drift
        # dtype / workspace-cap / op token legality
        if od not in _cublaslt.FULL_MATRIX_OUT_DTYPES:
            return _UNKNOWN
        if ws not in ws_cap_bytes:
            return _UNKNOWN
        if op not in _cublaslt.FULL_MATRIX_OPS:
            return _UNKNOWN
        # aligned must match the recomputed producer invariant
        try:
            aligned = int(rec["aligned"])
        except ValueError:
            return _UNKNOWN
        if aligned not in (0, 1):
            return _UNKNOWN
        if aligned != int(m % 16 == 0 and n % 16 == 0 and k % 16 == 0):
            return _UNKNOWN
        # status token legality
        status = rec["status"]
        if status not in _cublaslt.FULL_MATRIX_STATUS_TOKENS:
            return _UNKNOWN
        # algorithm-column legality (Task 5 algorithm-status check +
        # nongpu-rereview §3.8 residual checks): algo_count / first_algo_id /
        # workspace_bytes must be integers, in range, and consistent with the
        # row's status. Any violation -> UNKNOWN (fail-closed, never PASS).
        # The producer (run_full_matrix) writes:
        #   ok      -> algo_count>=1, first_algo_id>=0, 0<=workspace<=cap
        #   no-algo -> algo_count==0, first_algo_id==-1, workspace==0
        # The reader enforces that contract here, INCLUDING the three §3.8
        # residual checks the prior reader missed: ok rows must carry a real
        # (non-sentinel) first_algo_id and a workspace within the selected
        # ws_cap; no-algo rows must carry workspace==0 (a no-algo cell with
        # nonzero workspace is a forged/swallowed error, not a PASS).
        try:
            algo_count = int(rec["algo_count"])
            first_algo_id = int(rec["first_algo_id"])
            workspace_bytes = int(rec["workspace_bytes"])
        except ValueError:
            return _UNKNOWN  # non-integer algorithm column
        if algo_count < 0 or workspace_bytes < 0:
            return _UNKNOWN  # out-of-range algorithm column
        if status == "ok":
            if algo_count < 1:
                return _UNKNOWN  # "ok" must have found >=1 algorithm
            if first_algo_id < 0:
                return _UNKNOWN  # ok must carry a real algo id, not the -1 sentinel
            # workspace must not exceed the selected ws_cap (§3.8)
            if workspace_bytes > ws_cap_bytes[ws]:
                return _UNKNOWN  # workspace exceeds the selected cap
        elif status == "no-algo":
            if algo_count != 0 or first_algo_id != -1:
                return _UNKNOWN  # no-algo must be zero-algo with sentinel id
            if workspace_bytes != 0:
                return _UNKNOWN  # no-algo must report zero workspace
        # explicit no-algo policy: a no-algo OUTSIDE the policy set is a real
        # coverage gap (broken sweep / cuBLASLt regression), not a PASS.
        if status == "no-algo" and key not in no_algo_policy:
            return _UNKNOWN

    # expected keys == actual keys: every expected cell present, no extra cell.
    if actual_keys != expected_keys:
        return _UNKNOWN
    return _OK


#: Exact schema-version allowlist for the grouped capability artifact (Task 2 /
#: evidence-integrity plan v3 finding 3.2). A v1 (or any other) schema_version
#: is UNRECOGNIZED -- never silently accepted.
_GROUPED_SCHEMA_VERSIONS = frozenset({"c3-grouped-v2"})

#: Exact probe_source allowlist for the grouped API probe (Task 2). Only the
#: compile-header probe (``#ifdef`` against the real ``cublasLt.h``) is a
#: RECOGNIZED authority for API absence/presence; any other source is
#: UNRECOGNIZED -> the not_supported clause cannot hit -> UNKNOWN.
_GROUPED_PROBE_SOURCES = frozenset({"compiled_header_probe"})

#: Frozen self-report -> canonical-token map (v3-review errata). The artifact's
#: ``capability.status`` is a SELF-REPORT; the canonical token is recomputed via
#: :func:`evaluate_gate`. If the two disagree, the artifact is internally
#: inconsistent -> consistency_state=CONFLICT -> contradiction -> UNKNOWN.
_GROUPED_SELF_REPORT_MAP = {
    "SUPPORTED": "PASS",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
    "BLOCKED": "FAIL",
}


def _grouped_normalized(data):
    """Build the normalized ``raw`` dict for the grouped gate contract (Task 2).

    Reads the v2 artifact and emits the cross-task field names defined by
    :data:`gate_contracts.GATE_CONTRACTS` ``["grouped"]``:

      * ``schema_state``: VALID if ``schema_version`` in the allowlist, MISSING
        if absent, else UNRECOGNIZED.
      * ``api_state``: PRESENT / ABSENT_DEFINITIVE / ABSENT_INCONCLUSIVE from
        ``grouped_api_probe.cublaslt_grouped3gemm`` (True / False / absent).
      * ``attempt_state``: ATTEMPTED / NOT_ATTEMPTED from the API PROBE attempt
        (``grouped_api_probe.attempted``), NOT the execution attempt.
      * ``probe_source_state``: RECOGNIZED / UNRECOGNIZED / MISSING from
        ``grouped_api_probe.probe_source``.
      * ``compile_state`` / ``run_state`` / ``correctness_state`` /
        ``coverage_state``: set ONLY when ``grouped_execution.attempted`` is
        True. When the execution block is absent or not attempted, these fields
        are LEFT ABSENT -- so fail clauses needing them don't hit, and
        authoritative API absence yields NOT_SUPPORTED (via the not_supported
        clause), NOT FAIL.
      * ``consistency_state``: CONSISTENT (tentative; the caller may flip it to
        CONFLICT via the bidirectional consistency check).
    """
    probe = data.get("grouped_api_probe") or {}
    if not isinstance(probe, dict):
        probe = {}

    sv = data.get("schema_version")
    if sv is None:
        schema_state = "MISSING"
    elif sv in _GROUPED_SCHEMA_VERSIONS:
        schema_state = "VALID"
    else:
        schema_state = "UNRECOGNIZED"

    g3 = probe.get("cublaslt_grouped3gemm")
    if g3 is True:
        api_state = "PRESENT"
    elif g3 is False:
        api_state = "ABSENT_DEFINITIVE"
    else:
        api_state = "ABSENT_INCONCLUSIVE"

    attempt_state = "ATTEMPTED" if probe.get("attempted") is True else "NOT_ATTEMPTED"

    ps = probe.get("probe_source")
    if ps is None:
        probe_source_state = "MISSING"
    elif ps in _GROUPED_PROBE_SOURCES:
        probe_source_state = "RECOGNIZED"
    else:
        probe_source_state = "UNRECOGNIZED"

    raw = {
        "schema_state": schema_state,
        "api_state": api_state,
        "attempt_state": attempt_state,
        "probe_source_state": probe_source_state,
        "consistency_state": "CONSISTENT",
    }

    # Execution states are set ONLY when the execution was actually attempted.
    # When absent (the API-absent toolchain reality), they stay out of ``raw``
    # so the fail clauses needing them don't hit -- authoritative API absence
    # then routes to NOT_SUPPORTED via the not_supported clause, NOT FAIL.
    exec_block = data.get("grouped_execution")
    if isinstance(exec_block, dict) and exec_block.get("attempted") is True:
        compiles = exec_block.get("compiles")
        if compiles is True:
            raw["compile_state"] = "SUCCEEDED"
        elif compiles is False:
            raw["compile_state"] = "FAILED"
        else:
            raw["compile_state"] = "UNKNOWN"
        runs = exec_block.get("runs")
        if runs is True:
            raw["run_state"] = "SUCCEEDED"
        elif runs is False:
            raw["run_state"] = "FAILED"
        else:
            raw["run_state"] = "UNKNOWN"
        corr = exec_block.get("correctness")
        corr = corr if isinstance(corr, dict) else {}
        gate_pass = corr.get("gate_pass")
        if gate_pass is True:
            raw["correctness_state"] = "PASSED"
        elif gate_pass is False:
            raw["correctness_state"] = "FAILED"
        else:
            raw["correctness_state"] = "UNKNOWN"
        raw["coverage_state"] = (
            "COMPLETE" if exec_block.get("coverage_complete") is True else "INCOMPLETE"
        )

    return raw


def _c3_grouped_status(path):
    """cublasLt grouped capability verdict (Task 2 / evidence-integrity plan v3
    finding 3.2 -- a P1 fail-open fix).

    Recomputes a CANONICAL token from the raw v2 artifact via
    :func:`evaluate_gate` over :data:`GATE_CONTRACTS` ``["grouped"]`` -- the
    SINGLE decision rule. The reader retains NO undeclared PASS branch: every
    PASS/FAIL/NOT_SUPPORTED flows through the gate engine.

    The prior reader returned PASS for ``capability.status=SUPPORTED`` + API
    presence alone, without checking schema/execution/coverage -- a fail-open
    that trusted the self-reported status. The v2 reader enforces:

      * exact schema-version allowlist (``c3-grouped-v2``); any other schema ->
        UNRECOGNIZED (never PASS).
      * exact probe_source allowlist (``compiled_header_probe``); an
        unrecognized source cannot back a NOT_SUPPORTED claim.
      * execution states (compile/run/correctness/coverage) are checked ONLY
        when the execution was attempted; an API-absent toolchain with no
        execution yields NOT_SUPPORTED (via the not_supported clause), NOT FAIL.
      * bidirectional self-report consistency: the recomputed token is compared
        to the self-reported ``capability.status`` (frozen map: SUPPORTED->PASS,
        NOT_SUPPORTED->NOT_SUPPORTED, BLOCKED->FAIL); any disagreement ->
        ``consistency_state=CONFLICT`` -> contradiction -> UNKNOWN.

    Returns ``PASS`` / ``FAIL`` / ``UNKNOWN`` / ``NOT_SUPPORTED`` / ``NOT_RUN``
    (NOT_RUN only when the artifact itself is absent).
    """
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _UNKNOWN
    if not isinstance(data, dict):
        return _UNKNOWN

    raw = _grouped_normalized(data)

    # 1. Tentative candidate with consistency=CONSISTENT.
    candidate = evaluate_gate(raw, GATE_CONTRACTS["grouped"])[0]

    # 2. Bidirectional self-report consistency: compare the recomputed token to
    #    what the self-reported capability.status maps to. Any disagreement ->
    #    CONFLICT -> re-evaluate -> contradiction -> UNKNOWN.
    status = (data.get("capability") or {}).get("status")
    expected_from_self = _GROUPED_SELF_REPORT_MAP.get(status)
    if expected_from_self is not None and candidate != expected_from_self:
        raw["consistency_state"] = "CONFLICT"
        candidate = evaluate_gate(raw, GATE_CONTRACTS["grouped"])[0]

    return candidate


def _cutlass_status(path):
    """CUTLASS SM120 4M feasibility (Task 8), split into TWO independent
    canonical criteria (plan §7 Task 4):

      * ``CUTLASS_SM120_4M`` — native consumer-Blackwell sm_120 BF16 4M
        capability. NOT_SUPPORTED (CUTLASS 3.x Sm120 collective is
        F8F6F4-only + Sm100 gated by __CUDA_ARCH__==1000), or PASS only if
        a native sm120 path genuinely landed and passed (theoretical future).
      * ``CUTLASS_SM80_FALLBACK_CAPABILITY`` — the 2.x Ampere Sm80 fallback
        path that actually compiles+runs on sm_120. PASS iff the artifact
        records the fallback running + passing the BF16 correctness gate.

    Returns ``{"CUTLASS_SM120_4M": <token>, "CUTLASS_SM80_FALLBACK_CAPABILITY":
    <token>}``. Native failure and fallback success are INDEPENDENT — one
    does NOT derive from the other (plan §7 验收: native failure and fallback
    success coexist without contradiction). Numerical is a SEPARATE Task 3
    criterion (CUTLASS_SM80_FALLBACK_NUMERICAL) and is NOT touched here.

    Absent artifact -> both NOT_RUN; malformed -> both UNKNOWN.
    """
    if not os.path.exists(path):
        return {
            "CUTLASS_SM120_4M": _NOT_RUN,
            "CUTLASS_SM80_FALLBACK_CAPABILITY": _NOT_RUN,
        }
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {
            "CUTLASS_SM120_4M": _UNKNOWN,
            "CUTLASS_SM80_FALLBACK_CAPABILITY": _UNKNOWN,
        }
    return {
        "CUTLASS_SM120_4M": _cutlass_native_sm120_criterion(data),
        "CUTLASS_SM80_FALLBACK_CAPABILITY": _cutlass_sm80_fallback_criterion(data),
    }


#: Exact schema-version allowlist for the cutlass SM120 4M artifact (Task 5 /
#: evidence-integrity plan v3 finding 3.5). A different schema_version is
#: UNRECOGNIZED -- never silently accepted.
_CUTLASS_SCHEMA_VERSIONS = frozenset({"cutlass-sm120-4m-v1"})

#: Exact blocker_source allowlist for the native SM120 gate (Task 5). Only a
#: REAL captured blocker WITH a RECOGNIZED source backs NOT_SUPPORTED;
#: fallback-only without a captured blocker+source -> UNKNOWN. Mirrors
#: :data:`cutlass_probe._CUTLASS_BLOCKER_SOURCES` (single semantic source).
_CUTLASS_BLOCKER_SOURCES = frozenset({"compiler", "header_probe", "static_assert"})


#: Frozen self-report -> canonical-token map for the native SM120 section
#: (v3-review errata). The section's ``capability`` is a SELF-REPORT; the
#: canonical token is recomputed via :func:`evaluate_gate`. If the two
#: disagree, the artifact is internally inconsistent ->
#: ``consistency_state=CONFLICT`` -> contradiction -> UNKNOWN.
_CUTLASS_NATIVE_SELF_REPORT_MAP = {
    "PASS": "PASS",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
    "FAIL": "FAIL",
    "UNKNOWN": "UNKNOWN",
}


#: Frozen self-report -> canonical-token map for the SM80 fallback section.
#: The fallback contract has empty ``contradiction_fields``, so CONFLICT does
#: not trigger a contradiction re-evaluate; the bidirectional check is
#: informational (the recompute via evaluate_gate is the single decision rule).
_CUTLASS_FALLBACK_SELF_REPORT_MAP = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "UNKNOWN": "UNKNOWN",
}


#: Fields the cutlass readers consume from the merged (section + single_4m)
#: execution source (F3b). The committed artifact splits these across the
#: dedicated section (attempted/coverage_complete/compile_status/correctness)
#: and single_4m (kernel_path/runs/compiles). The old pick-one reader discarded
#: one section's evidence; the merge recombines them.
_CUTLASS_MERGE_FIELDS = (
    "attempted",
    "compiles",
    "compile_status",
    "runs",
    "correctness",
    "coverage_complete",
    "kernel_path",
)


def _cutlass_merge_sections(sec, s4, kernel_filter):
    """Merge a dedicated cutlass section with ``single_4m`` (F3b).

    Each field is read from ``sec`` first; if ``sec``'s value is ``None``, the
    field falls back to ``single_4m`` -- but ONLY when
    ``single_4m.kernel_path == kernel_filter`` (cross-promo prevention: a
    ``single_4m`` that ran a DIFFERENT kernel path must NOT contribute its
    execution evidence to this section). Returns a flat dict of the merged
    fields (absent fields map to ``None``).
    """
    s4_ok = s4.get("kernel_path") == kernel_filter
    out = {}
    for k in _CUTLASS_MERGE_FIELDS:
        v = sec.get(k)
        if v is None and s4_ok:
            v = s4.get(k)
        out[k] = v
    return out


def _cutlass_native_normalized(data):
    """Build the normalized ``raw`` dict for the native SM120 gate contract
    (Task 5 / evidence-integrity plan v3 finding 3.5).

    Reads the artifact and emits the cross-task field names defined by
    :data:`gate_contracts.GATE_CONTRACTS` ``["cutlass_native"]``:

      * ``schema_state``: VALID if ``schema_version`` in the allowlist, MISSING
        if absent, else UNRECOGNIZED.
      * ``attempt_state``: ATTEMPTED if ``attempted is True``, else
        NOT_ATTEMPTED.
      * ``compile_state`` / ``run_state`` / ``correctness_state`` /
        ``coverage_state``: set ONLY when the native path actually compiled
        (``compiles is True`` -- F3a: ``compile_status`` alone does NOT
        substitute). When the native path was blocked (compile failed /
        blocker captured), these fields are LEFT ABSENT -- so fail clauses
        needing them do not multi-hit
        alongside the ``not_supported`` clause (blocker + source). This
        mirrors the grouped reader pattern of setting execution states
        only when attempted.
      * ``blocker_state``: PRESENT if a real ``sm120_blocker`` (non-empty
        string) exists in the section or ``single_4m``, else MISSING.
      * ``blocker_source_state``: RECOGNIZED if ``blocker_source`` is in
        the :data:`cutlass_probe._CUTLASS_BLOCKER_SOURCES` allowlist, else
        UNRECOGNIZED / MISSING.
      * ``consistency_state``: CONSISTENT (tentative; the caller may flip
        it to CONFLICT via the bidirectional consistency check).

    Cross-promo prevention: execution fields are read from the native
    section, OR from ``single_4m`` only when ``single_4m.kernel_path ==
    "sm120_native"`` (the native path landed, so ``single_4m`` carries the
    native evidence). When ``kernel_path == "sm80_fallback"``, the
    ``single_4m`` execution fields belong to the FALLBACK and are NOT
    cross-promoted into the native raw dict.
    """
    sec = data.get("native_sm120_bf16_4m")
    sec = sec if isinstance(sec, dict) else {}
    s4 = data.get("single_4m")
    s4 = s4 if isinstance(s4, dict) else {}

    # Schema state (top-level artifact field).
    sv = data.get("schema_version")
    if sv is None:
        schema_state = "MISSING"
    elif sv in _CUTLASS_SCHEMA_VERSIONS:
        schema_state = "VALID"
    else:
        schema_state = "UNRECOGNIZED"

    # Attempt state (from section or single_4m).
    attempted = sec.get("attempted")
    if attempted is None:
        attempted = s4.get("attempted")
    attempt_state = "ATTEMPTED" if attempted is True else "NOT_ATTEMPTED"

    # Blocker state (native-specific; safe to read from section or single_4m).
    sm120_blocker = (
        sec.get("sm120_blocker")
        or sec.get("blocker")
        or s4.get("sm120_blocker")
        or s4.get("native_sm120_blocker")
    )
    blocker_state = (
        "PRESENT" if (isinstance(sm120_blocker, str) and sm120_blocker) else "MISSING"
    )

    # Blocker source state (gates NOT_SUPPORTED on a RECOGNIZED authority).
    blocker_source = sec.get("blocker_source") or s4.get("blocker_source")
    if blocker_source is None:
        blocker_source_state = "MISSING"
    elif blocker_source in _CUTLASS_BLOCKER_SOURCES:
        blocker_source_state = "RECOGNIZED"
    else:
        blocker_source_state = "UNRECOGNIZED"

    raw = {
        "schema_state": schema_state,
        "attempt_state": attempt_state,
        "blocker_state": blocker_state,
        "blocker_source_state": blocker_source_state,
        "consistency_state": "CONSISTENT",
    }

    # F3b: MERGE sec + single_4m execution fields (cross-promo respect). The
    # old reader picked ONE section (exec_src = sec if sec.kernel_path else
    # (s4 if s4.kernel_path=="sm120_native" else {})), discarding the other
    # section's evidence. Now each field is read from sec first, falling back to
    # single_4m ONLY when single_4m.kernel_path == "sm120_native" (the native
    # path landed -> single_4m carries native evidence; a fallback single_4m is
    # NOT cross-promoted into the native raw dict).
    exec_src = _cutlass_merge_sections(sec, s4, "sm120_native")

    # F3a: native compile success requires ``compiles is True``. compile_status
    # alone does NOT substitute (compiles=False + compile_status="OK" is a
    # contradiction). Execution states (compile/run/correctness/coverage) are
    # set ONLY when compiles is True -- when the native path was blocked
    # (compiles False/None), they are LEFT ABSENT so fail clauses do not
    # multi-hit alongside the not_supported clause (blocker+source).
    compiles = exec_src.get("compiles")
    if compiles is True:
        raw["compile_state"] = "SUCCEEDED"
        runs = exec_src.get("runs")
        if runs is True:
            raw["run_state"] = "SUCCEEDED"
        elif runs is False:
            raw["run_state"] = "FAILED"
        else:
            raw["run_state"] = "UNKNOWN"
        corr = exec_src.get("correctness")
        corr = corr if isinstance(corr, dict) else {}
        gate_pass = corr.get("gate_pass")
        if gate_pass is True:
            raw["correctness_state"] = "PASSED"
        elif gate_pass is False:
            raw["correctness_state"] = "FAILED"
        else:
            raw["correctness_state"] = "UNKNOWN"
        coverage_complete = exec_src.get("coverage_complete")
        raw["coverage_state"] = (
            "COMPLETE" if coverage_complete is True else "INCOMPLETE"
        )

    return raw


def _cutlass_fallback_normalized(data):
    """Build the normalized ``raw`` dict for the SM80 fallback gate contract
    (Task 5 / evidence-integrity plan v3 finding 3.5).

    Reads the ``sm80_fallback_bf16_4m`` section (or ``single_4m`` when
    ``kernel_path == "sm80_fallback"``) and emits the field names defined by
    :data:`gate_contracts.GATE_CONTRACTS` ``["cutlass_fallback"]``:

      * ``schema_state``: VALID if ``schema_version`` in the allowlist, MISSING
        if absent, else UNRECOGNIZED (F1 fail-open fix -- the fallback pass
        clause now requires schema_state=VALID, mirroring the native reader;
        an unrecognized schema can never back a fallback PASS).
      * ``attempt_state``: ATTEMPTED if ``attempted is True``, else
        NOT_ATTEMPTED.
      * ``compile_state``: ``"OK"`` (not ``"SUCCEEDED"``) if ``compiles is
        True`` (F3a: ``compile_status`` alone does NOT substitute --
        ``compiles=False`` + ``compile_status="OK"`` is a contradiction and
        yields ``compile_state=FAILED``; ``compiles=None`` yields UNKNOWN).
        The fallback contract uses ``OK`` per Task 5 test
        ``compile_status=="OK"``.
      * ``run_state`` / ``correctness_state`` / ``coverage_state``: mapped
        from the section ``runs`` / ``correctness.gate_pass`` /
        ``coverage_complete``.

    No blocker fields (the fallback contract has empty not_supported and
    empty contradiction). No bidirectional consistency check (the contract
    has empty ``contradiction_fields``, so CONFLICT cannot trigger a
    contradiction -- the recompute via evaluate_gate is the single rule).
    """
    sec = data.get("sm80_fallback_bf16_4m")
    sec = sec if isinstance(sec, dict) else {}
    s4 = data.get("single_4m")
    s4 = s4 if isinstance(s4, dict) else {}

    # F3b: MERGE sec + single_4m fields (cross-promo respect). The committed
    # artifact splits fields: sm80_fallback_bf16_4m carries
    # attempted/coverage_complete/compile_status/correctness; single_4m carries
    # kernel_path/runs/compiles. The old reader picked ONE section (exec_src =
    # sec if sec.kernel_path else s4), discarding the other's evidence. Now each
    # field is read from sec first, falling back to single_4m ONLY when
    # single_4m.kernel_path == "sm80_fallback" (the fallback path landed ->
    # single_4m carries fallback evidence; a native single_4m is NOT
    # cross-promoted).
    exec_src = _cutlass_merge_sections(sec, s4, "sm80_fallback")

    attempted = exec_src.get("attempted")
    attempt_state = "ATTEMPTED" if attempted is True else "NOT_ATTEMPTED"

    # Schema state (top-level artifact field; F1 fail-open fix -- the fallback
    # pass clause now requires schema_state=VALID, mirroring the native reader).
    sv = data.get("schema_version")
    if sv is None:
        schema_state = "MISSING"
    elif sv in _CUTLASS_SCHEMA_VERSIONS:
        schema_state = "VALID"
    else:
        schema_state = "UNRECOGNIZED"

    compiles = exec_src.get("compiles")

    raw = {
        "schema_state": schema_state,
        "attempt_state": attempt_state,
        "consistency_state": "CONSISTENT",
    }

    # F3a: compile_state requires ``compiles is True``. compile_status alone
    # does NOT substitute -- compiles=False + compile_status="OK" is a
    # contradiction and must NOT pass. compiles is True -> OK; compiles is
    # False -> FAILED; else (None/absent) -> UNKNOWN (cannot confirm).
    if compiles is True:
        raw["compile_state"] = "OK"
    elif compiles is False:
        raw["compile_state"] = "FAILED"
    else:
        raw["compile_state"] = "UNKNOWN"

    runs = exec_src.get("runs")
    if runs is True:
        raw["run_state"] = "SUCCEEDED"
    elif runs is False:
        raw["run_state"] = "FAILED"
    else:
        raw["run_state"] = "UNKNOWN"

    corr = exec_src.get("correctness")
    corr = corr if isinstance(corr, dict) else {}
    gate_pass = corr.get("gate_pass")
    if gate_pass is True:
        raw["correctness_state"] = "PASSED"
    elif gate_pass is False:
        raw["correctness_state"] = "FAILED"
    else:
        raw["correctness_state"] = "UNKNOWN"

    coverage_complete = exec_src.get("coverage_complete")
    raw["coverage_state"] = "COMPLETE" if coverage_complete is True else "INCOMPLETE"

    return raw


def _cutlass_native_sm120_criterion(data):
    """Native SM120 BF16 4M capability (evidence-integrity plan v3 finding 3.5).

    Recomputes a CANONICAL token from the raw artifact via
    :func:`evaluate_gate` over :data:`GATE_CONTRACTS` ``["cutlass_native"]``
    -- the SINGLE decision rule. The reader retains NO undeclared PASS branch:
    every PASS/FAIL/NOT_SUPPORTED flows through the gate engine.

    NOT_SUPPORTED requires a REAL captured blocker (``blocker_state=PRESENT``)
    WITH a RECOGNIZED source (``blocker_source_state=RECOGNIZED``). Fallback-
    only (``kernel_path == "sm80_fallback"``) without a captured blocker+
    source -> UNKNOWN (no synthesized NOT_SUPPORTED from the fallback alone --
    the native verdict must NOT be DERIVED from the fallback).

    The self-reported ``capability`` is a DIAGNOSTIC consistency check: if it
    disagrees with the recomputed token, ``consistency_state`` flips to
    ``CONFLICT`` -> contradiction -> UNKNOWN.
    """
    if not isinstance(data, dict):
        return _UNKNOWN

    raw = _cutlass_native_normalized(data)
    candidate = evaluate_gate(raw, GATE_CONTRACTS["cutlass_native"])[0]

    # Bidirectional self-report consistency: compare the recomputed token to
    # what the self-reported capability maps to. Any disagreement -> CONFLICT
    # -> re-evaluate -> contradiction -> UNKNOWN.
    sec = data.get("native_sm120_bf16_4m")
    sec = sec if isinstance(sec, dict) else {}
    self_reported = sec.get("capability")
    expected_from_self = _CUTLASS_NATIVE_SELF_REPORT_MAP.get(self_reported)
    if expected_from_self is not None and candidate != expected_from_self:
        raw["consistency_state"] = "CONFLICT"
        candidate = evaluate_gate(raw, GATE_CONTRACTS["cutlass_native"])[0]

    return candidate


def _cutlass_sm80_fallback_criterion(data):
    """SM80 fallback BF16 4M capability (evidence-integrity plan v3 finding 3.5).

    Recomputes a CANONICAL token from the raw artifact via
    :func:`evaluate_gate` over :data:`GATE_CONTRACTS` ``["cutlass_fallback"]``
    -- the SINGLE decision rule. The reader retains NO undeclared PASS branch.

    PASS requires ``attempted`` AND ``compile_state="OK"`` AND ``runs`` AND
    ``gate_pass`` AND ``coverage_complete`` -- all five must be green. Missing
    coverage -> not PASS (UNKNOWN, fail-closed). FAIL only when ``runs`` or
    ``gate_pass`` is explicitly False.

    Cross-promo prevention: when ``kernel_path != "sm80_fallback"``, the
    reader returns UNKNOWN directly (the fallback path was not the one that
    ran -- no evidence cross-promotion from a native path).

    The fallback contract has empty ``contradiction_fields``, so no
    bidirectional consistency check is needed (CONFLICT cannot trigger a
    contradiction re-evaluate; the recompute is the single rule).
    """
    if not isinstance(data, dict):
        return _UNKNOWN

    sec = data.get("sm80_fallback_bf16_4m")
    sec = sec if isinstance(sec, dict) else {}
    s4 = data.get("single_4m")
    s4 = s4 if isinstance(s4, dict) else {}
    kernel_path = sec.get("kernel_path") or s4.get("kernel_path")
    # No cross-promo: if the fallback path was not the one that ran, the
    # fallback criterion is UNKNOWN (a native path evidence cannot confirm
    # the fallback). This is a fail-closed UNKNOWN, not a PASS branch.
    if kernel_path != "sm80_fallback":
        return _UNKNOWN

    raw = _cutlass_fallback_normalized(data)
    candidate = evaluate_gate(raw, GATE_CONTRACTS["cutlass_fallback"])[0]

    return candidate


def _region_proto_is_real_pte(data):
    """Intrinsic P->T->E prototype gate -- shares ONE standard with
    ``c2._is_real_pte_prototype`` (the same checks the C2 region reader gates on):
    schema version, real region producer/consumer MNK, a full-E consumer tensor,
    no full P/T materialized, and non-reduction math. The cross-edge MNK binding
    (an independent edge artifact) is N/A for the single-artifact region reader,
    so it is not duplicated here; the intrinsic fields are what distinguish a
    real P->T->E prototype from the rejected GEMM->norm/reduction artifact
    (final-review §3.2/§7.1). Returns True iff ``data`` is a real prototype.
    """
    from results._phase0 import c2 as _c2

    if not isinstance(data, dict) or not data:
        return False
    if data.get("schema_version") != _c2.PROTO_SCHEMA:
        return False
    region = data.get("region") or {}
    prod = region.get("producer")
    cons = region.get("consumer")
    if not (
        isinstance(prod, list) and len(prod) == 3 and all(int(x) > 0 for x in prod)
    ):
        return False
    if not (
        isinstance(cons, list) and len(cons) == 3 and all(int(x) > 0 for x in cons)
    ):
        return False
    if cons[0] * cons[1] * 8 < _c2.FULL_E_MIN_BYTES:
        return False
    if not (data.get("no_full_P_materialized") and data.get("no_full_T_materialized")):
        return False
    if any(m in str(data.get("math", "")).lower() for m in _c2._REDUCTION_MARKERS):
        return False
    return True


def _region_proto_status(path):
    """Region P->T->E prototype verdict (Task 4 / nongpu-rereview §3.5.2 /
    evidence-integrity plan v3 finding 3.3 -- a P1 fail-open fix).

    Returns ONLY a canonical token (PASS/FAIL/UNKNOWN/NOT_RUN). Recomputes
    from raw evidence via the SHARED ``c2._normalize_region_peak`` +
    ``evaluate_gate(GATE_CONTRACTS["region_peak"])`` -- the SINGLE decision
    rule. The reader retains NO undeclared PASS branch: every PASS/FAIL/
    UNKNOWN flows through the gate engine.

    Task 3 (finding 3.3): the prior ad-hoc acc/res/peak PASS branch is
    replaced by the shared normalizer + GateContract. Both
    ``c2._region_layer`` AND this function call the SAME normalizer +
    contract, so REGION_PROTOTYPE and C2_REGION_KERNEL_FEASIBILITY share
    ONE standard.

    ``case_binding_state=MISSING`` (the default in ``_normalize_region_peak``)
    is the honest value here: gonogo reads a single artifact with no
    canonical case context or hash binding, so the reader CANNOT verify
    binding -> MISSING -> not PASS. Only ``c2._region_layer`` (which runs
    after ``_binding_problems`` verifies case+hash binding at the
    ``judge_c2_canonical`` level) supplies ``case_binding_state=MATCH``.

    The bidirectional self-report consistency check (errata #2) compares the
    recomputed token to ``data["verdict"]`` via
    ``c2._REGION_SELF_REPORT_MAP``; disagreement -> ``consistency_state=
    CONFLICT`` -> contradiction -> UNKNOWN. This routes a FEASIBLE*
    self-report with missing evidence (candidate=FAIL or UNKNOWN) through
    CONFLICT -> UNKNOWN (the honest outcome for an unsubstantiated claim).

    ``NOT_FEASIBLE`` -> FAIL (definitive negative, kept as a direct short-
    circuit before the gate: the artifact itself declares the region
    infeasible). Absent artifact -> NOT_RUN; malformed -> UNKNOWN.
    """
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _UNKNOWN
    if not isinstance(data, dict):
        return _UNKNOWN
    from results._phase0 import c2 as _c2
    from results._phase0.gate_contracts import GATE_CONTRACTS, evaluate_gate

    verdict = data.get("verdict")
    # NOT_FEASIBLE is a definitive negative -> canonical FAIL (short-circuit
    # before the gate; the artifact itself declares the region infeasible).
    if verdict == "NOT_FEASIBLE":
        return _BAD

    # Gate the intrinsic P->T->E prototype standard (same checks c2 gates on).
    if not _region_proto_is_real_pte(data):
        return _UNKNOWN

    # Shared normalizer + GateContract (the SINGLE decision rule).
    # case_binding_state=MISSING: gonogo has no canonical case context ->
    # cannot verify binding -> not PASS (honest).
    raw = _c2._normalize_region_peak(data, case_binding_state="MISSING")
    token, _ = evaluate_gate(raw, GATE_CONTRACTS["region_peak"])

    # Bidirectional self-report consistency: compare recomputed token to the
    # self-reported verdict. Disagreement -> CONFLICT -> contradiction -> UNKNOWN.
    expected_from_self = _c2._REGION_SELF_REPORT_MAP.get(verdict)
    if expected_from_self is not None and token != expected_from_self:
        raw["consistency_state"] = "CONFLICT"
        token, _ = evaluate_gate(raw, GATE_CONTRACTS["region_peak"])

    return token


def _numerical_overall_status(path):
    """Overall numerical status (Task 9). NOT_RUN if absent."""
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _UNKNOWN
    overall = data.get("overall_numerical_status") if isinstance(data, dict) else None
    if overall in ("PASS", "FAIL"):
        return overall
    return _UNKNOWN


def _numerical_per_route(path):
    """Per-route numerical criterion map {route: PASS|FAIL} from Task 9.

    Routes absent from the artifact are omitted (callers treat omission as
    UNDETERMINED). Empty dict if artifact absent/malformed.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    per = {}
    for row in (data.get("per_route") or []) if isinstance(data, dict) else []:
        if isinstance(row, dict) and row.get("criterion") in ("PASS", "FAIL"):
            route = row.get("route")
            if route is not None:
                per[route] = row["criterion"]
    return per


def _collect_environment():
    """Snapshot GPU/SM/driver/CUDA/library versions + TF32 state + theta seeds.

    Best-effort: missing fields are recorded as null rather than raising, so a
    partial env doesn't abort the gonogo emission. GPU queries require a CUDA
    runtime (torch.cuda); pure-CPU runs leave the GPU fields null.
    """
    env = {
        "gpu_name": None,
        "gpu_uuid": None,
        "sm_compute_capability": None,
        "multiprocessor_count": None,
        "total_vram_GB": None,
        "driver_version": None,
        "cuda_version": None,
        "torch_version": None,
        "jax_version": None,
        "cotengra_version": None,
        "tensorcircuit_version": None,
        "tf32_matmul_allowed": None,
        "theta_seeds": None,
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["tf32_matmul_allowed"] = bool(torch.backends.cuda.matmul.allow_tf32)
        if torch.version.cuda:
            env["cuda_version"] = torch.version.cuda
        try:
            p = torch.cuda.get_device_properties(0)
            env["gpu_name"] = p.name
            env["sm_compute_capability"] = f"{p.major}.{p.minor}"
            env["multiprocessor_count"] = p.multi_processor_count
            env["total_vram_GB"] = round(p.total_memory / 1e9, 4)
            env["gpu_uuid"] = str(getattr(p, "uuid", "") or "")
        except Exception:
            pass
    except Exception:
        pass
    try:
        import jax

        env["jax_version"] = jax.__version__
    except Exception:
        pass
    try:
        import importlib.metadata as md

        for pkg in ("cotengra", "tensorcircuit-ng"):
            try:
                if pkg == "tensorcircuit-ng":
                    env["tensorcircuit_version"] = md.version(pkg)
                else:
                    env["cotengra_version"] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            parts = out.split(",")
            if len(parts) >= 1 and parts[0]:
                env["driver_version"] = parts[0].strip()
            if len(parts) >= 2 and parts[1].strip():
                env["gpu_uuid"] = parts[1].strip()
    except Exception:
        pass
    # theta-seed values used by the C1/C2 contraction probes (results/_phase0_c1.py).
    env["theta_seeds"] = [0.7, 0.8, 0.9]
    return env


def main(stage_dir=None):
    """Two-layer Phase 0 aggregator entry point.

    Reads every capability + numerical artifact, runs the two-layer pipeline,
    and writes gonogo.json / gonogo.md / environment.json (manifest.json is
    owned by manifest.py — Task 11 handoff). stage_dir defaults to
    results/phase0.
    """
    base = stage_dir or "results/phase0"
    os.makedirs(base, exist_ok=True)

    def _load_json(name):
        p = os.path.join(base, name)
        if not os.path.exists(p):
            return {}
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    c1_j = _load_json("c1_judgment.json")
    c2_j = _load_json("c2_judgment.json")

    # plan §7 Task 4: _cutlass_status returns BOTH canonical cutlass criteria
    # (native SM120 + SM80 fallback). They are split into independent keys so
    # native failure and fallback success coexist without contradiction.
    cutlass = _cutlass_status(os.path.join(base, "cutlass_sm120_4m.json"))

    criteria = {
        "C1": _c1_status_from_judgment(c1_j),
        # Task 1 (plan §1.1/§1.2): gonogo emits all 4 C2 layers (the 3 input
        # layers + C2_CANONICAL). The old "C2" alias is NOT produced here --
        # aggregate_two_layer's validate_criteria sets it as a compat alias =
        # C2_CANONICAL after validating the rollup.
        "C2_REGION_KERNEL_FEASIBILITY": _c2_layer_status(
            c2_j, "C2_REGION_KERNEL_FEASIBILITY"
        ),
        "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK": _c2_layer_status(
            c2_j, "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK"
        ),
        "C2_JOINT_EXECUTABLE_LEVERAGE": _c2_layer_status(
            c2_j, "C2_JOINT_EXECUTABLE_LEVERAGE"
        ),
        "C2_CANONICAL": _c2_layer_status(c2_j, "C2_CANONICAL"),
        "C3_PLANAR_CORE": _c3_planar_from_capability(
            os.path.join(base, "cublaslt_planar_capability.json")
        ),
        "C3_PLANAR_FULL_MATRIX": _c3_planar_full_matrix_status(
            os.path.join(base, "cublaslt_full_matrix.csv")
        ),
        "C3_GROUPED": _c3_grouped_status(
            os.path.join(base, "cublaslt_grouped_capability.json")
        ),
        "CUTLASS_SM120_4M": cutlass["CUTLASS_SM120_4M"],
        "CUTLASS_SM80_FALLBACK_CAPABILITY": cutlass["CUTLASS_SM80_FALLBACK_CAPABILITY"],
        "REGION_PROTOTYPE": _region_proto_status(
            os.path.join(base, "region_prototype.json")
        ),
        "NUMERICAL": _numerical_overall_status(
            os.path.join(base, "numerical_validation.json")
        ),
    }

    # Task 7: derivation goes through the shared §5 truth table
    # (verdict_schema.recompute_derived_state) so gonogo and manifest derivation
    # cannot diverge (shared derivation logic; inputs/binding-validation differ).
    # JSON + Markdown render from the single ``agg`` object below.
    per_route_num = _numerical_per_route(
        os.path.join(base, "numerical_validation.json")
    )
    agg = aggregate_two_layer(criteria, per_route_num)

    with open(os.path.join(base, "gonogo.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(base, "gonogo.md"), "w") as f:
        f.write(_render_md(agg))

    # environment snapshot (kept; Task 11 manifest references it)
    with open(os.path.join(base, "environment.json"), "w") as f:
        json.dump(_collect_environment(), f, indent=2)

    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
