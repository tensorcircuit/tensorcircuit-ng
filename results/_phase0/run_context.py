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


def build():
    porcelain = _git(["status", "--porcelain"]) or ""
    ctx = {
        "schema_version": "run-context-v1",
        "source_commit": _git(["rev-parse", "HEAD"]),
        "dirty_worktree": bool(porcelain.strip()),
        "dirty_file_count": len([ln for ln in porcelain.splitlines() if ln.strip()]),
        "package_versions": _versions(),
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
