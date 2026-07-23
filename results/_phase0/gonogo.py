"""Two-layer Phase 0 aggregator (review §10 / final-remediation plan §13).

route_verdict (per-route capability AND numerical) + phase0_completion (all
canonical criteria determined) -> phase1_authorization. Emits gonogo.json /
gonogo.md / environment.json under results/phase0/. md is generated FROM the
json object, never hand-overwritten.
"""

from __future__ import annotations

import csv
import json
import os

from results._phase0.verdict_schema import normalize_criterion

VERDICTS = (
    "GO_TO_PHASE1",
    "NO_GO_NO_WINDOW",
    "NO_GO_NOT_COVERABLE",
    "NO_GO_KERNEL",
    "INCONCLUSIVE",
)

# Status tokens shared across the truth table.
_OK = "PASS"
_BAD = "FAIL"
_UNKNOWN = "UNKNOWN"
_NOT_RUN = "NOT_RUN"

# Tri-state values used by the two-layer gating logic.
_TRI_OK = "OK"
_TRI_NOT_OK = "NOT_OK"
_TRI_UNDETERMINED = "UNDETERMINED"

# Canonical criteria whose determined-ness gates phase0_completion (truth-table
# rules 1/5). NUMERICAL=FAIL is "determined" and does NOT sink completion.
# CUTLASS_SM120_4M (native, NOT_SUPPORTED) and CUTLASS_SM80_FALLBACK_CAPABILITY
# (fallback, PASS) are SPLIT into two independent criteria (plan §7 Task 4):
# native failure and fallback success coexist without contradiction.
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

# Which capability criteria each contraction route depends on (truth-table
# rule 8 + rule 3). A route is VIABLE only if every listed capability criterion
# normalizes to OK AND its numerical criterion normalizes to OK.
#
# ``cutlass_4m_single`` depends on the FALLBACK capability
# (CUTLASS_SM80_FALLBACK_CAPABILITY), NOT on CUTLASS_SM120_4M: on consumer
# Blackwell sm_120 the route's actual kernel is the 2.x Ampere fallback (the
# native SM120 path is architecturally BLOCKED), so the route's capability
# tracks the path that really runs. Native failure is recorded as a separate
# CUTLASS_SM120_4M criterion but does NOT sink the route by itself.
ROUTE_CAPABILITY_CRITERIA = {
    "planar": ("C3_PLANAR_CORE", "C3_PLANAR_FULL_MATRIX"),
    "grouped": ("C3_GROUPED",),
    "region_fused": ("REGION_PROTOTYPE", "C2_REGION_KERNEL"),
    "cutlass_4m_single": ("CUTLASS_SM80_FALLBACK_CAPABILITY",),
}

ROUTES = tuple(ROUTE_CAPABILITY_CRITERIA)


def _normalize(verdict):
    """Map an artifact-native verdict token to a gating tri-state.

    OK          -> the canonical criterion is PASS (the only "established good"
                   token; plan §4 forbids promoting FEASIBLE* / SUPPORTED /
                   TILE_FUSION_FEASIBLE detail tokens to OK)
    NOT_OK      -> established as bad (canonical FAIL or NOT_SUPPORTED)
    UNDETERMINED-> not established (UNKNOWN, NOT_RUN, BLOCKED, INCONCLUSIVE,
                   any artifact-native detail token, unrecognized strings)

    Plan §4 验收: no ``startswith('FEASIBLE')`` unconditional promotion. Every
    incoming token is first scrubbed by ``verdict_schema.normalize_criterion``,
    which fail-closes artifact-native detail tokens (FEASIBLE*, SUPPORTED,
    TILE_FUSION_FEASIBLE, NOT_FEASIBLE, BLOCKED, INCONCLUSIVE) to canonical
    UNKNOWN. The canonical criteria feeding this layer should already be
    canonical tokens by the time they reach it; if a detail token leaks through
    it fails closed to UNDETERMINED rather than being promoted to OK.
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


def capability_layer(criteria):
    """Per-route capability tri-state from the canonical criteria dict.

    A route's capability is the AND of its ROUTE_CAPABILITY_CRITERIA entries
    (each normalized). rule 3 (region depends on C2_REGION_KERNEL) is encoded
    by the route's criteria tuple.
    """
    out = {}
    for route, deps in ROUTE_CAPABILITY_CRITERIA.items():
        out[route] = _combine_tri([_normalize(criteria.get(c)) for c in deps])
    return out


def numerical_layer(per_route_num, routes):
    """Per-route numerical tri-state from Task 9 per_route criterion map.

    A route absent from per_route_num is UNDETERMINED (its numerical criterion
    was not produced).
    """
    out = {}
    for r in routes:
        out[r] = _normalize(per_route_num.get(r, _NOT_RUN))
    return out


def _route_status(cap_tri, num_tri):
    """Truth-table rule 8: VIABLE iff capability OK AND numerical OK;
    NOT_VIABLE if either NOT_OK; else UNKNOWN."""
    if _TRI_NOT_OK in (cap_tri, num_tri):
        return "NOT_VIABLE"
    if _TRI_UNDETERMINED in (cap_tri, num_tri):
        return "UNKNOWN"
    return "VIABLE"


def route_verdict(cap_tri, num_tri):
    """Per-route verdict map {route: {status, capability, numerical}}.

    status is VIABLE / NOT_VIABLE / UNKNOWN per rule 8; capability and numerical
    carry the raw tri-states for transparency.
    """
    out = {}
    for r in ROUTES:
        c, n = cap_tri.get(r, _TRI_UNDETERMINED), num_tri.get(r, _TRI_UNDETERMINED)
        out[r] = {"status": _route_status(c, n), "capability": c, "numerical": n}
    return out


def evaluate_completion(criteria):
    """Truth-table rules 1/4/5: COMPLETE iff every REQUIRED_CRITERION is
    determined (normalizes to OK or NOT_OK). Any UNKNOWN/NOT_RUN (i.e.
    UNDETERMINED) -> INCONCLUSIVE. NUMERICAL=FAIL is determined and does NOT
    sink completion."""
    for c in REQUIRED_CRITERIA:
        if _normalize(criteria.get(c)) == _TRI_UNDETERMINED:
            return "INCONCLUSIVE"
    return "COMPLETE"


def authorize_phase1(completion, route_verdict_map):
    """Truth-table rule 6: GO_TO_PHASE1 iff COMPLETE and >=1 route VIABLE;
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
            if _normalize(criteria.get(c)) == _TRI_UNDETERMINED
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
    if _normalize(criteria.get("C2")) == _TRI_UNDETERMINED:
        blocking.append("c2_judgment.json (C2_CANONICAL undetermined)")
    if _normalize(criteria.get("NUMERICAL")) == _TRI_NOT_OK:
        blocking.append("numerical_validation.json (overall=FAIL)")
    for r, rv in route_verdict_map.items():
        if rv["capability"] == _TRI_NOT_OK and r == "grouped":
            blocking.append("cublaslt_grouped_capability.json (NOT_SUPPORTED)")
    return blocking


def aggregate_two_layer(criteria, route_verdict_map, completion, authorization):
    """Compose the full gonogo-v2 verdict object from the layered results."""
    return {
        "schema_version": "gonogo-v2",
        "criteria": dict(criteria),
        "route_verdict": {r: dict(v) for r, v in route_verdict_map.items()},
        "phase0_completion": completion,
        "phase1_authorization": authorization,
        "reasons": _build_reasons(criteria, route_verdict_map, completion),
        "blocking_artifacts": _build_blocking_artifacts(criteria, route_verdict_map),
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
    ws_cap_names = {c[0] for c in _cublaslt.FULL_MATRIX_WS_CAPS}

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
        if ws not in ws_cap_names:
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
        # explicit no-algo policy: a no-algo OUTSIDE the policy set is a real
        # coverage gap (broken sweep / cuBLASLt regression), not a PASS.
        if status == "no-algo" and key not in no_algo_policy:
            return _UNKNOWN

    # expected keys == actual keys: every expected cell present, no extra cell.
    if actual_keys != expected_keys:
        return _UNKNOWN
    return _OK


def _c3_grouped_status(path):
    """cublasLt grouped capability verdict (Task 7). NOT_RUN if absent."""
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _UNKNOWN
    status = (data.get("capability") or {}).get("status")
    if status in ("SUPPORTED", "NOT_SUPPORTED"):
        return status
    return _UNKNOWN


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


def _cutlass_native_sm120_criterion(data):
    """Read the native SM120 BF16 capability. Prefer the new two-section
    ``native_sm120_bf16_4m.capability`` field; fall back to the legacy
    ``single_4m`` block plus native-sm120 blocker keys so older artifacts
    (and synths that record ``single_4m.native_sm120_blocker``) still load.
    """
    if not isinstance(data, dict):
        return _UNKNOWN
    sec = data.get("native_sm120_bf16_4m")
    if isinstance(sec, dict):
        cap = sec.get("capability")
        if cap in ("PASS", "FAIL", "NOT_SUPPORTED", _UNKNOWN):
            return cap
    s4 = data.get("single_4m")
    if isinstance(s4, dict):
        blocker = s4.get("native_sm120_blocker") or s4.get("sm120_blocker")
        kp = s4.get("kernel_path")
        runs = bool(s4.get("runs"))
        gate = bool((s4.get("correctness") or {}).get("gate_pass"))
        # Theoretical future: native sm120 actually landed + passed.
        if kp == "sm120_native" and runs and gate:
            return _OK
        # Real-world: blocker recorded, OR artifact documents landing on the
        # sm80 fallback (no native path landed) -> NOT_SUPPORTED.
        if blocker or kp == "sm80_fallback":
            return "NOT_SUPPORTED"
        if kp:
            # Attempted but outcome unclear; fail-closed -> UNKNOWN.
            return _UNKNOWN
    return _UNKNOWN


def _cutlass_sm80_fallback_criterion(data):
    """Read the SM80 fallback capability. Prefer the new two-section
    ``sm80_fallback_bf16_4m.capability`` field; fall back to the legacy
    ``single_4m`` block. CAPABILITY only — numerical is Task 3's concern.
    """
    if not isinstance(data, dict):
        return _UNKNOWN
    sec = data.get("sm80_fallback_bf16_4m")
    if isinstance(sec, dict):
        cap = sec.get("capability")
        if cap in ("PASS", "FAIL", "NOT_SUPPORTED", _UNKNOWN):
            return cap
    s4 = data.get("single_4m")
    if isinstance(s4, dict):
        kp = s4.get("kernel_path")
        runs = bool(s4.get("runs"))
        gate = bool((s4.get("correctness") or {}).get("gate_pass"))
        if kp == "sm80_fallback":
            return _OK if (runs and gate) else _BAD
    return _UNKNOWN


def _region_proto_status(path):
    """Region P->T->E prototype verdict (Task 4). NOT_RUN if absent."""
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _UNKNOWN
    verdict = data.get("verdict") if isinstance(data, dict) else None
    if verdict in (
        "TILE_FUSION_FEASIBLE",
        "FEASIBLE_WITH_RECOMPUTE",
        "NOT_FEASIBLE",
        "BLOCKED",
    ):
        return verdict
    return _UNKNOWN


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
        "C2": _c2_status_from_judgment(c2_j),
        "C2_REGION_KERNEL": _c2_layer_status(c2_j, "C2_REGION_KERNEL_FEASIBILITY"),
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

    cap_tri = capability_layer(criteria)
    num_per = _numerical_per_route(os.path.join(base, "numerical_validation.json"))
    num_tri = numerical_layer(num_per, ROUTES)
    rv = route_verdict(cap_tri, num_tri)
    completion = evaluate_completion(criteria)
    authorization = authorize_phase1(completion, rv)
    agg = aggregate_two_layer(criteria, rv, completion, authorization)

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
