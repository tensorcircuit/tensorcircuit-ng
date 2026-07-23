"""Full reproducibility manifest for Phase 0 (review §11 / plan §14).

Reads run_context.json + gonogo.json + all phase0 artifacts, fail-closed-validates
the criteria (presence + checkpoint-hash), records hashes/cases/provenance, and
writes manifest.json. gonogo.py no longer writes manifest.json (Task 11 handoff).

用法: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
      python results/_phase0/manifest.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
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


def _case_artifacts(case_id, base):
    """Case-specific files under the INPUT_ARTIFACT_DIRS whose name or parent dir
    matches the case_id prefix (e.g. 'n24_d10'). Best-effort provenance list."""
    parts = case_id.split("_")
    needle = "_".join(parts[:2]) if len(parts) >= 2 else case_id
    found = []
    for d in INPUT_ARTIFACT_DIRS:
        full_dir = os.path.join(base, d)
        if not os.path.isdir(full_dir):
            continue
        for root, _dirs, files in os.walk(full_dir):
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), base).replace(
                    os.sep, "/"
                )
                if (
                    name.startswith(needle)
                    or ("/" + needle + "_") in rel
                    or name.startswith(case_id)
                ):
                    found.append(rel)
    return sorted(set(found))


def _build_cases(c1_judgment, c2_judgment, base):
    """Merge c1/c2 judgment cases into {case_id: {status, config, artifacts}}."""
    c1 = c1_judgment if isinstance(c1_judgment, dict) else {}
    c2 = c2_judgment if isinstance(c2_judgment, dict) else {}
    cases = {}
    for case_id in sorted(set(c1) | set(c2)):
        entry = {"status": {}, "config": {}, "artifacts": []}
        c1c = c1.get(case_id) if isinstance(c1.get(case_id), dict) else {}
        c2c = c2.get(case_id) if isinstance(c2.get(case_id), dict) else {}
        if c1c:
            entry["status"]["C1"] = (c1c.get("judgment") or {}).get("status")
            entry["config"] = {k: c1c[k] for k in ("n", "depth", "fusion") if k in c1c}
        if c2c:
            entry["status"]["C2"] = c2c.get("status")
            if isinstance(c2c.get("layers"), dict):
                entry["status"]["C2_layers"] = c2c["layers"]
            for k in ("n", "depth", "fusion"):
                if k in c2c and k not in entry["config"]:
                    entry["config"][k] = c2c[k]
        entry["artifacts"] = _case_artifacts(case_id, base)
        cases[case_id] = entry
    return cases


def _collect_inputs_outputs(base):
    """Hash driving artifacts (inputs, incl. dirs) and generated verdicts (outputs).
    manifest.json is never included. Missing files are omitted (no None entries)."""
    inputs = {}
    for rel in INPUT_ARTIFACT_FILES:
        h = _hash_file(os.path.join(base, rel))
        if h is not None:
            inputs[rel] = h
    for d in INPUT_ARTIFACT_DIRS:
        inputs.update(_hash_dir(os.path.join(base, d)))
    outputs = {}
    for rel in OUTPUT_ARTIFACTS:
        h = _hash_file(os.path.join(base, rel))
        if h is not None:
            outputs[rel] = h
    return inputs, outputs


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def build_manifest(base, generated_at=None):
    """Compose the manifest-v1 object from run_context + gonogo + validated
    criteria + cases + inputs/outputs. Deterministic given fixed generated_at."""
    run_ctx = _load_json(os.path.join(base, "run_context.json"))
    gonogo = _load_json(os.path.join(base, "gonogo.json"))
    gonogo = gonogo if isinstance(gonogo, dict) else {}
    run_ctx = run_ctx if isinstance(run_ctx, dict) else {}
    c1_j = _load_json(os.path.join(base, "c1_judgment.json"))
    c2_j = _load_json(os.path.join(base, "c2_judgment.json"))
    c2_ckpt = _load_json(os.path.join(base, "c2_checkpoint_manifest.json"))
    numerical = _load_json(os.path.join(base, "numerical_validation.json"))

    gonogo_criteria = gonogo.get("criteria", {})
    criteria = _presence_check(gonogo_criteria, base)
    c2_status = _validate_c2_checkpoint(base, c2_j, c2_ckpt)
    num_status = _validate_numerical_binding(base, numerical)
    criteria = _apply_checkpoint_validation(criteria, c2_status, num_status)

    inputs, outputs = _collect_inputs_outputs(base)
    cases = _build_cases(c1_j, c2_j, base)

    if generated_at is None:
        generated_at = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_commit": run_ctx.get("source_commit"),
        "dirty_worktree": run_ctx.get("dirty_worktree"),
        "dirty_file_count": run_ctx.get("dirty_file_count"),
        "commands": run_ctx.get("command_templates") or {},
        "environment_hash": _hash_file(os.path.join(base, "environment.json")),
        "criteria": criteria,
        "route_verdict": {
            r: (v.get("status") if isinstance(v, dict) else v)
            for r, v in (gonogo.get("route_verdict") or {}).items()
        },
        "phase0_completion": gonogo.get("phase0_completion"),
        "phase1_authorization": gonogo.get("phase1_authorization"),
        "required_artifacts": {k: list(v) for k, v in REQUIRED_ARTIFACTS.items()},
        "inputs": dict(sorted(inputs.items())),
        "outputs": dict(sorted(outputs.items())),
        "cases": cases,
        "generated_at": generated_at,
    }


def main(stage_dir=None):
    """Write results/phase0/manifest.json (or under stage_dir)."""
    base = stage_dir or "results/phase0"
    os.makedirs(base, exist_ok=True)
    manifest = build_manifest(base)
    with open(os.path.join(base, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "phase0_completion": manifest["phase0_completion"],
                "phase1_authorization": manifest["phase1_authorization"],
                "criteria": manifest["criteria"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
