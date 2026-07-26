"""Phase 0 run context / reproducibility provenance (final-remediation Task 0).

Records the source commit, dirty-worktree flag, package-version fingerprint, and the
canonical command templates -- the reproducible starting point for a Phase 0 run. Records
NO private absolute paths, usernames, or real conda env names (per remediation plan §1).

Lightweight: uses importlib.metadata (no GPU/CUDA init) + git. Run:
  python results/_phase0/run_context.py
"""

from __future__ import annotations

import json
import os
import subprocess
from importlib import metadata

OUT = "results/phase0/run_context.json"

# The real aggregation command (Task 6 errata #2: a real reproducible script,
# not a nonexistent script or a ``python -c`` one-liner). ``numerical.py
# --regen-no-gpu`` regenerates the numerical validation matrix from existing
# CSV rows without GPU measurement (plan §6 3a / Task 3a regen path).
AGGREGATION_COMMAND = "python results/_phase0/numerical.py --regen-no-gpu"

COMMAND_TEMPLATES = {
    "xla_dump": "python results/_phase0/xla_dump.py",
    "c1_ab": "python results/_phase0/c1.py --ab --n {n} --depth {depth}",
    "edge_map": "python results/_phase0/c1_to_c2_map.py",
    "peak_frontier": "python results/_phase0/c2_peak_analysis.py",
    "region_proto": "python results/_phase0/region_proto.py",
    "c2_gate": "python results/_phase0/c2.py --n {n} --depth {depth}",
    "gonogo": "python results/_phase0/gonogo.py",
    "manifest": "python results/_phase0/manifest.py",
}

_VERSION_PACKAGES = (
    "jax",
    "jaxlib",
    "cupy",
    "cupy-cuda12x",
    "torch",
    "numpy",
    "cotengra",
    "tensorcircuit",
    "tensorcircuit-ng",
    "nvidia-cublas-cu12",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
)


def _git(args):
    try:
        r = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=os.getcwd()
        )
        return r.stdout.strip()
    except Exception:
        return None


def _versions():
    out = {}
    for pkg in _VERSION_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def _preserve_measurement(existing):
    """Read the existing run_context.json dict and preserve its measurement
    role (Task 6 errata #1: v1->v2 migration).

    - v2 nested (has ``measurement`` dict): preserve verbatim.
    - v1 flat (has ``source_commit``): migrate ``source_commit`` ->
      ``measurement.source_commit`` (validate non-empty). Also carry over
      ``run_id`` / ``environment_hash`` if present in the v1 file.
    - missing/empty/malformed: no measurement role to preserve (first run).

    Returns a dict (possibly empty) suitable for the ``measurement`` field
    of the v2 schema. Never raises.
    """
    if not isinstance(existing, dict):
        return {}
    meas = existing.get("measurement")
    if isinstance(meas, dict) and meas.get("source_commit"):
        return dict(meas)  # v2 nested: preserve verbatim
    # v1 flat migration: source_commit -> measurement.source_commit
    flat_commit = existing.get("source_commit")
    if flat_commit:
        migrated = {"source_commit": flat_commit}
        for k in ("run_id", "environment_hash"):
            if existing.get(k):
                migrated[k] = existing[k]
        return migrated
    return {}  # no prior measurement role to preserve


def build():
    """Build the run-context-v2 provenance record and write it to ``OUT``.

    v2 schema (Task 6 / finding 3.6): separates the MEASUREMENT role (the
    commit that produced the GPU evidence) from the AGGREGATION role (the real
    current HEAD + dirty-worktree flag + the real reproducible command that
    re-derives the aggregate artifacts).

    P1 #5 fix (reviewer B): ``measurement.source_commit`` MUST be the current
    HEAD (the commit containing the measurement code), NOT a stale preserved
    value from a prior run. The old ``_preserve_measurement`` carried over
    hardcoded stale commits (e.g. ``20589967`` from evidence-integrity), but
    the GPU measurement code (full-anchor collectors, G1-G5 kernels, current
    policy) only exists at ``976c7892+``. The actual re-measurement with the
    new dual-gate policy waits for B approval; until then, ``build()`` records
    the current HEAD as the measurement commit. ``run_id`` /
    ``environment_hash`` from a prior measurement role are still preserved.

    Lightweight: uses importlib.metadata (no GPU/CUDA init) + git. Run:
      python results/_phase0/run_context.py
    """
    head = _git(["rev-parse", "HEAD"])

    # P1 #5: measurement.source_commit = current HEAD (NOT stale preserved).
    # Preserve only run_id / environment_hash from a prior measurement role.
    measurement = {"source_commit": head}
    if os.path.exists(OUT):
        try:
            with open(OUT) as fh:
                existing = json.load(fh)
            if isinstance(existing, dict):
                prior = existing.get("measurement")
                if isinstance(prior, dict):
                    for k in ("run_id", "environment_hash"):
                        if prior.get(k):
                            measurement[k] = prior[k]
        except (OSError, ValueError):
            pass  # unreadable/missing -> no prior measurement role

    # dirty_worktree reflects TRACKED modifications only (exclude untracked
    # ``??`` scratch, which is pre-existing throwaway not part of the commit and
    # does not affect reproducibility from source_commit). This is the
    # reproducibility signal: were there uncommitted changes to committed files
    # when the aggregation ran? Run run_context.build() BEFORE the regenerating
    # producers (numerical/gonogo) so the tracked tree is clean -> dirty=False.
    porcelain = _git(["status", "--porcelain", "--untracked-files=no"]) or ""
    ctx = {
        "schema_version": "run-context-v2",
        "measurement": measurement,
        "aggregation": {
            "source_commit": head,
            "dirty_worktree": bool(porcelain.strip()),
            "dirty_file_count": len(
                [ln for ln in porcelain.splitlines() if ln.strip()]
            ),
            "command": AGGREGATION_COMMAND,
            "package_versions": _versions(),
        },
        "command_templates": COMMAND_TEMPLATES,
        "runner_note": (
            "All commands run via the project WSL harness in the project conda "
            "env. Machine-specific strings are sanitized in tracked artifacts "
            "(spec §3.7): conda env names -> <env>, toolchain clone dirs -> "
            "<toolchain>, home/repo absolute paths -> <home>/<repo>. Package "
            "versions + source commit are the reproducibility fingerprint."
        ),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(ctx, fh, indent=2)
    return ctx


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
