"""Two-layer Phase 0 aggregator (review §10 / final-remediation plan §13).

route_verdict (per-route capability AND numerical) + phase0_completion (all
canonical criteria determined) -> phase1_authorization. Emits gonogo.json /
gonogo.md / environment.json under results/phase0/. md is generated FROM the
json object, never hand-overwritten.
"""

from __future__ import annotations

import json
import os

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
REQUIRED_CRITERIA = (
    "C1",
    "C2",
    "C3_PLANAR_CORE",
    "C3_PLANAR_FULL_MATRIX",
    "C3_GROUPED",
    "CUTLASS_SM120_4M",
    "REGION_PROTOTYPE",
    "NUMERICAL",
)

# Which capability criteria each contraction route depends on (truth-table
# rule 8 + rule 3). A route is VIABLE only if every listed capability criterion
# normalizes to OK AND its numerical criterion normalizes to OK.
ROUTE_CAPABILITY_CRITERIA = {
    "planar": ("C3_PLANAR_CORE", "C3_PLANAR_FULL_MATRIX"),
    "grouped": ("C3_GROUPED",),
    "region_fused": ("REGION_PROTOTYPE", "C2_REGION_KERNEL"),
    "cutlass_4m_single": ("CUTLASS_SM120_4M",),
}

ROUTES = tuple(ROUTE_CAPABILITY_CRITERIA)


def _normalize(verdict):
    """Map an artifact-native verdict token to a gating tri-state.

    OK          -> the capability/result is established as good
                   (PASS, SUPPORTED, FEASIBLE*, TILE_FUSION_FEASIBLE)
    NOT_OK      -> established as bad (FAIL, NOT_SUPPORTED, NOT_FEASIBLE)
    UNDETERMINED-> not established (UNKNOWN, NOT_RUN, BLOCKED, unrecognized)
    """
    if verdict in ("PASS", "SUPPORTED", "TILE_FUSION_FEASIBLE"):
        return _TRI_OK
    if isinstance(verdict, str) and verdict.startswith("FEASIBLE"):
        return _TRI_OK
    if verdict in ("FAIL", "NOT_SUPPORTED", "NOT_FEASIBLE"):
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


def _c3_planar_full_matrix_status(path):
    """C3 planar full-matrix completeness (Task 6 sweep artifact).

    PASS   -> cublaslt_full_matrix.csv exists with a header and >=1 data row
              (the sweep produced output; Task 6 vetted its quality separately)
    NOT_RUN-> artifact absent
    UNKNOWN-> present but empty/unparseable
    """
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            rows = [ln for ln in f if ln.strip()]
    except (OSError, ValueError):
        return _UNKNOWN
    # rows[0] is the header; need >=1 data row beyond it
    if len(rows) < 2:
        return _UNKNOWN
    return "PASS"  # sweep produced output; Task 6 vetted its quality separately


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
    """CUTLASS SM120 4M feasibility (Task 8), derived from single_4m.

    Derive from single_4m (compiles+runs+gate_pass); matches the artifact's
    top-level `overall` field. absent -> NOT_RUN;
    attempted-but-not-passing -> FAIL; malformed -> UNKNOWN.
    """
    if not os.path.exists(path):
        return _NOT_RUN
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _UNKNOWN
    s4 = data.get("single_4m") if isinstance(data, dict) else None
    if not isinstance(s4, dict):
        return _UNKNOWN
    kernel_path = s4.get("kernel_path")
    gate_pass = (s4.get("correctness") or {}).get("gate_pass")
    if s4.get("compiles") and s4.get("runs") and gate_pass:
        return (
            "FEASIBLE_WITH_SM80_FALLBACK"
            if kernel_path == "sm80_fallback"
            else "FEASIBLE"
        )
    if kernel_path:
        return _BAD  # attempted a path but it did not pass
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
        "CUTLASS_SM120_4M": _cutlass_status(
            os.path.join(base, "cutlass_sm120_4m.json")
        ),
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
