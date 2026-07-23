"""Full reproducibility manifest for Phase 0 (review §11 / plan §14).

Reads run_context.json + gonogo.json + all phase0 artifacts, fail-closed-validates
the criteria (presence + checkpoint-hash), records hashes/cases/provenance, and
writes manifest.json. gonogo.py no longer writes manifest.json (Task 11 handoff).

用法: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
      python results/_phase0/manifest.py
"""

from __future__ import annotations

import hashlib
import os

SCHEMA_VERSION = "manifest-v1"

# criterion -> required artifacts (presence-gating; missing -> NOT_RUN)
REQUIRED_ARTIFACTS = {
    "C1": ["c1_judgment.json", "c1_default_vs_nofusion.csv"],
    "C2": ["c2_judgment.json", "c2_checkpoint_manifest.json"],
    "C3_PLANAR_CORE": ["cublaslt_planar_capability.json"],
    "C3_PLANAR_FULL_MATRIX": ["cublaslt_full_matrix.csv"],
    "C3_GROUPED": ["cublaslt_grouped_capability.json"],
    "CUTLASS_SM120_4M": ["cutlass_sm120_4m.json"],
    "REGION_PROTOTYPE": ["region_prototype.json"],
    "NUMERICAL": ["numerical_validation.json"],
}

# driving artifacts hashed into inputs{} (files + dirs expanded per-file)
INPUT_ARTIFACT_FILES = [
    "c1_judgment.json",
    "c2_judgment.json",
    "c1_c2_edge_map.json",
    "c2_peak_frontier.json",
    "c2_checkpoint_manifest.json",
    "cublaslt_planar_capability.json",
    "cublaslt_full_matrix.csv",
    "cublaslt_grouped_capability.json",
    "cublaslt_grouped.csv",
    "cutlass_sm120_4m.json",
    "numerical_validation.json",
    "numerical_validation.csv",
    "region_prototype.json",
    "contraction_shapes.csv",
    "c1_default_vs_nofusion.csv",
    "c2_tileability.csv",
    "run_context.json",
]
INPUT_ARTIFACT_DIRS = ["c1_optimized_hlo", "c1_buffer_assignment", "c1_xla_dump"]

# generated verdicts hashed into outputs{} (manifest.json excluded — no self-hash)
OUTPUT_ARTIFACTS = ["gonogo.json", "gonogo.md", "environment.json"]

# C2 checkpoint binding keys to re-hash. c2_checkpoint_manifest.artifact_hashes
# records full sha256 (truncate to [:16] for comparison). allocation_audit in the
# checkpoint corresponds to the "audit" key in c2_judgment.artifact_paths.
C2_CHECKPOINT_KEYS = [
    "source_hlo",
    "buffer_assignment",
    "allocation_audit",
    "edge_map",
    "peak_frontier",
    "prototype",
]
C2_PATH_KEY_ALIASES = {"allocation_audit": "audit"}

# NUMERICAL case_binding hashes (sha[:16]): (file under base) -> binding key
NUMERICAL_BINDINGS = {
    "edge_map": ("c1_c2_edge_map.json", "edge_map_hash"),
    "prototype": ("region_prototype.json", "prototype_hash"),
    "contraction_shapes": ("contraction_shapes.csv", "contraction_shapes_hash"),
}


def _hash_file(path):
    """sha256[:16] of file bytes; None if missing."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _hash_dir(dir_path):
    """{relative_path: sha256[:16]} for each file under dir_path (recursive).

    Keys are relative to the dir's PARENT (so 'c1_optimized_hlo/<file>'), '/'-joined.
    """
    out = {}
    if not os.path.isdir(dir_path):
        return out
    parent = os.path.dirname(dir_path)
    entries = []
    for root, _dirs, files in os.walk(dir_path):
        for name in files:
            entries.append(os.path.join(root, name))
    for full in sorted(entries):
        rel = os.path.relpath(full, parent).replace(os.sep, "/")
        out[rel] = _hash_file(full)
    return out


def _resolve_under_base(base, path):
    """artifact_paths in c2_judgment are repo-relative ('results/phase0/...').
    Strip that prefix and join under base so staging-dir tests resolve too."""
    for pfx in ("results/phase0/", "results\\phase0\\"):
        if path.startswith(pfx):
            path = path[len(pfx) :]
            break
    return os.path.join(base, path)


def _presence_check(gonogo_criteria, base):
    """Fail-closed presence validation. For each criterion in REQUIRED_ARTIFACTS,
    if ANY required artifact is missing under base, force the criterion to NOT_RUN
    (no evidence) regardless of what gonogo.json claimed. Never defaults to PASS."""
    validated = dict(gonogo_criteria)
    for criterion, required in REQUIRED_ARTIFACTS.items():
        if criterion not in validated:
            continue
        if any(not os.path.exists(os.path.join(base, r)) for r in required):
            validated[criterion] = "NOT_RUN"
    return validated


def _c2_artifact_paths(c2_judgment):
    """artifact_paths from the first case in c2_judgment.json (case-keyed dict)."""
    if not isinstance(c2_judgment, dict) or not c2_judgment:
        return {}
    first = next(iter(c2_judgment.values()))
    if isinstance(first, dict):
        return first.get("artifact_paths") or {}
    return {}


def _validate_c2_checkpoint(base, c2_judgment, c2_checkpoint):
    """Re-hash C2 binding source files; compare to c2_checkpoint_manifest hashes.

    Returns OK if every resolvable binding matches, MISMATCH if any differs or its
    source file is gone, UNAVAILABLE if the checkpoint artifact is absent/malformed.
    """
    if not isinstance(c2_checkpoint, dict) or not c2_checkpoint.get("artifact_hashes"):
        return "UNAVAILABLE"
    expected = c2_checkpoint["artifact_hashes"]
    paths = _c2_artifact_paths(c2_judgment)
    checked = 0
    for key in C2_CHECKPOINT_KEYS:
        exp_full = expected.get(key)
        path_key = C2_PATH_KEY_ALIASES.get(key, key)
        src = paths.get(path_key)
        if not exp_full or not src:
            continue
        checked += 1
        actual = _hash_file(_resolve_under_base(base, src))
        if actual is None or actual != exp_full[:16]:
            return "MISMATCH"
    return "OK" if checked else "UNAVAILABLE"


def _validate_numerical_binding(base, numerical_json):
    """Re-hash numerical case_binding source files; compare to recorded sha[:16]."""
    if not isinstance(numerical_json, dict):
        return "UNAVAILABLE"
    binding = numerical_json.get("case_binding")
    if not isinstance(binding, dict) or not binding:
        return "UNAVAILABLE"
    checked = 0
    for _name, (rel, hash_key) in NUMERICAL_BINDINGS.items():
        exp = binding.get(hash_key)
        if not exp:
            continue
        checked += 1
        actual = _hash_file(os.path.join(base, rel))
        if actual is None or actual != exp[:16]:
            return "MISMATCH"
    return "OK" if checked else "UNAVAILABLE"


def _apply_checkpoint_validation(criteria, c2_status, num_status):
    """A checkpoint MISMATCH breaks the binding -> the criterion cannot be trusted
    -> force UNKNOWN (covers 'cannot retain PASS' and is fail-closed for FAIL too).
    UNAVAILABLE -> no change (cannot validate, do not downgrade)."""
    out = dict(criteria)
    if c2_status == "MISMATCH" and "C2" in out:
        out["C2"] = "UNKNOWN"
    if num_status == "MISMATCH" and "NUMERICAL" in out:
        out["NUMERICAL"] = "UNKNOWN"
    return out
