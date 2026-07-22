"""Four-state Phase 0 aggregator (review §9).

Reads structured artifacts (c1_judgment.json, c2_judgment.json, _phase0_cublaslt_gap.txt)
and emits gonogo.json / gonogo.md / manifest.json / environment.json under results/phase0/.
md is generated FROM json, never hand-overwritten.

用法: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_gonogo.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re

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


def _parse_c3_real_ceiling_ratio(path):
    """Max bf16/fp32 TFLOPS ratio from the cublaslt_gap txt table; None if missing/unparseable.

    Lines look like: '2048   41.35...  15.63...  2.65'
    """
    if not os.path.exists(path):
        return None
    try:
        ratios = []
        with open(path) as f:
            for ln in f:
                parts = ln.split()
                # row of interest: first token is an int M=N=K, last token is the ratio float
                if len(parts) >= 4 and re.fullmatch(r"\d+", parts[0]):
                    try:
                        ratios.append(float(parts[-1]))
                    except ValueError:
                        continue
        return max(ratios) if ratios else None
    except OSError:
        return None


def _file_hash(p):
    return (
        hashlib.sha1(open(p, "rb").read()).hexdigest()[:16]
        if os.path.exists(p)
        else None
    )


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


def main():
    base = "results/phase0"
    os.makedirs(base, exist_ok=True)

    # C1: roll the per-case judgment statuses up into one criterion status.
    c1 = _NOT_RUN
    cj = os.path.join(base, "c1_judgment.json")
    if os.path.exists(cj):
        with open(cj) as f:
            c1 = _c1_status_from_judgment(json.load(f))

    # C2: consume the already-judged c2_judgment.json from the Task 7 integration
    # (it has the C1-large pre-filter applied; do NOT re-run judge_c2 on raw shapes).
    c2 = _NOT_RUN
    c2j = os.path.join(base, "c2_judgment.json")
    if os.path.exists(c2j):
        with open(c2j) as f:
            c2 = _c2_status_from_judgment(json.load(f))

    # C3 planar (authoritative): read the cublasLt planar-complex capability
    # artifact produced by Plan B Task 2 (PASS=SUPPORTED / FAIL=NOT_SUPPORTED /
    # NOT_RUN=artifact absent). Keys the §9 truth table; no longer hard NOT_RUN.
    c3_planar = _c3_planar_from_capability(
        os.path.join(base, "cublaslt_planar_capability.json")
    )

    # C3 real ceiling (auxiliary): parse the cublaslt_gap txt proxy.
    c3_real = _parse_c3_real_ceiling_ratio("results/_phase0_cublaslt_gap.txt")

    agg = aggregate(c1, c2, c3_planar, c3_real)

    with open(os.path.join(base, "gonogo.json"), "w") as f:
        json.dump(agg, f, indent=2)

    md = [
        "# Phase 0 Go/No-Go (four-state, §9 truth table)",
        "",
        f"**Verdict: {agg['verdict']}**",
        "",
        "**Note:** " + agg["note"],
        "",
        "C3_planar is read from `cublaslt_planar_capability.json` (Plan B Task 2): "
        "PASS = SUPPORTED, FAIL = NOT_SUPPORTED, NOT_RUN = artifact absent.",
        "",
        "## Criteria",
        "```json",
        json.dumps(agg["criteria"], indent=2),
        "```",
    ]
    with open(os.path.join(base, "gonogo.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # environment snapshot (GPU/SM/driver/CUDA/library versions, TF32=off, theta seeds)
    env_snapshot = _collect_environment()
    with open(os.path.join(base, "environment.json"), "w") as f:
        json.dump(env_snapshot, f, indent=2)

    # manifest: per-case status + artifact hashes. Written last so it can hash the
    # other emitted files; the manifest does not hash itself (self-reference).
    manifest = {
        "c1": c1,
        "c2": c2,
        "c3_planar": c3_planar,
        "c3_real_ceiling_ratio": c3_real,
        "verdict": agg["verdict"],
        "artifacts": {
            f: _file_hash(os.path.join(base, f))
            for f in (
                "c1_judgment.json",
                "c2_judgment.json",
                "cublaslt_planar_capability.json",
                "contraction_shapes.csv",
                "c2_tileability.csv",
                "c1_default_vs_nofusion.csv",
                "gonogo.json",
                "gonogo.md",
                "environment.json",
            )
        },
    }
    with open(os.path.join(base, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
