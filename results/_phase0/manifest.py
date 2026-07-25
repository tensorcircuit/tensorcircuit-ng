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

from results._phase0.verdict_schema import (
    CRITERIA_NAMES,
    recompute_derived_state,
    validate_criteria,
)

SCHEMA_VERSION = "manifest-v1"

# criterion -> required artifacts (presence-gating; missing -> NOT_RUN).
# Task 1 (plan §1.1): derived from the canonical CRITERIA_NAMES single source
# of truth. The old "C2" alias is NOT here -- the 4 real C2 layers each map
# to the shared c2_judgment.json + c2_checkpoint_manifest.json chain.
# Task 5 (finding 3.7): CUTLASS_SM80_FALLBACK_CAPABILITY now maps to the SAME
# cutlass_sm120_4m.json artifact as CUTLASS_SM120_4M -- deleting the shared
# artifact downgrades BOTH criteria together (plus NUMERICAL via the cutlass
# source hash in case_binding), so a stale fallback PASS can no longer survive.
REQUIRED_ARTIFACTS = {
    "C1": ["c1_judgment.json", "c1_default_vs_nofusion.csv"],
    "C2_REGION_KERNEL_FEASIBILITY": [
        "c2_judgment.json",
        "c2_checkpoint_manifest.json",
    ],
    "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK": [
        "c2_judgment.json",
        "c2_checkpoint_manifest.json",
    ],
    "C2_JOINT_EXECUTABLE_LEVERAGE": [
        "c2_judgment.json",
        "c2_checkpoint_manifest.json",
    ],
    "C2_CANONICAL": ["c2_judgment.json", "c2_checkpoint_manifest.json"],
    "C3_PLANAR_CORE": ["cublaslt_planar_capability.json"],
    "C3_PLANAR_FULL_MATRIX": ["cublaslt_full_matrix.csv"],
    "C3_GROUPED": ["cublaslt_grouped_capability.json"],
    "CUTLASS_SM120_4M": ["cutlass_sm120_4m.json"],
    "CUTLASS_SM80_FALLBACK_CAPABILITY": ["cutlass_sm120_4m.json"],
    "REGION_PROTOTYPE": ["region_prototype.json"],
    "NUMERICAL": ["numerical_validation.json", "numerical_validation.csv"],
}

# Task 5 fold-in (I2): REQUIRED_ARTIFACTS must be a subset of the canonical
# CRITERIA_NAMES (plan §1.1 single source of truth). This assertion prevents
# the required-artifact map from drifting to criterion names that don't exist
# in the canonical schema.
assert set(REQUIRED_ARTIFACTS).issubset(set(CRITERIA_NAMES)), (
    "REQUIRED_ARTIFACTS keys must all be canonical CRITERIA_NAMES; "
    f"extra: {set(REQUIRED_ARTIFACTS) - set(CRITERIA_NAMES)}"
)

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
INPUT_ARTIFACT_DIRS = ["c1_optimized_hlo", "c1_buffer_assignment"]

# generated verdicts hashed into outputs{} (manifest.json excluded — no self-hash)
OUTPUT_ARTIFACTS = ["gonogo.json", "gonogo.md", "environment.json"]

# C2 checkpoint binding keys to re-hash (plan §9 6.1 / spec §3.3.1). ALL must
# be present and match for OK. c2_checkpoint_manifest.artifact_hashes records
# full sha256 (64-hex, compared directly -- F6b removed the [:16] truncation).
# allocation_audit in the checkpoint corresponds to the "audit" key in
# c2_judgment.artifact_paths. c2_judgment hashes the c2_judgment.json file
# itself (fixed location, not in artifact_paths).
C2_CHECKPOINT_KEYS = [
    "source_hlo",
    "buffer_assignment",
    "allocation_audit",
    "edge_map",
    "peak_frontier",
    "prototype",
    "c2_judgment",
]
C2_PATH_KEY_ALIASES = {"allocation_audit": "audit"}
# Keys whose source file is a fixed artifact under base (not in artifact_paths).
C2_FIXED_PATH_KEYS = {"c2_judgment": "c2_judgment.json"}

# Numerical case_binding hashes (plan §5.2 / spec §4.4 -- full SHA256 binding
# of ALL 9 route-source files). EVERY entry is hash-checked (no presence-only
# files). The hash_key suffix ``_sha256`` documents the algorithm; the
# ``case_binding["algorithm"]`` field in numerical_validation.json documents
# the full (non-truncated) 64-hex-char length (spec §4.4: no unexplained
# truncation). Missing expected hash / missing file -> UNAVAILABLE; content
# mismatch -> MISMATCH (plan §5.3).
NUMERICAL_BINDINGS = {
    "edge_map": ("c1_c2_edge_map.json", "edge_map_sha256"),
    "region_prototype": ("region_prototype.json", "region_prototype_sha256"),
    "contraction_shapes": ("contraction_shapes.csv", "contraction_shapes_sha256"),
    "cublaslt_planar_capability": (
        "cublaslt_planar_capability.json",
        "cublaslt_planar_capability_sha256",
    ),
    "cublaslt_full_matrix": (
        "cublaslt_full_matrix.csv",
        "cublaslt_full_matrix_sha256",
    ),
    "cublaslt_grouped_capability": (
        "cublaslt_grouped_capability.json",
        "cublaslt_grouped_capability_sha256",
    ),
    "cublaslt_grouped_rows": (
        "cublaslt_grouped.csv",
        "cublaslt_grouped_rows_sha256",
    ),
    "cutlass_4m": ("cutlass_sm120_4m.json", "cutlass_4m_sha256"),
    "numerical_csv": ("numerical_validation.csv", "numerical_csv_sha256"),
}

# The 6 files that were PREVIOUSLY presence-only (finding 3.2 fail-open
# surface). They are now fully hash-bound via NUMERICAL_BINDINGS above; this
# list is kept for the 3.2 mutation-test iteration and documents which files
# were the original fail-open gap.
NUMERICAL_REQUIRED_FILES = [
    "numerical_validation.csv",
    "cublaslt_planar_capability.json",
    "cublaslt_full_matrix.csv",
    "cublaslt_grouped_capability.json",
    "cublaslt_grouped.csv",
    "cutlass_sm120_4m.json",
]


def _hash_file(path):
    """Full sha256 (64 hex chars) of file bytes; None if missing.

    F6b (Scope Reset): was sha256[:16] (16-hex truncation); now full 64-hex
    so ALL manifest provenance hashes (inputs/outputs/environment) are unified
    with the case_binding hashes (which were already full sha256 via
    ``_hash_file_full``). No unexplained truncation remains.
    """
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file_full(path):
    """Full sha256 (64 hex chars) of file bytes; None if missing.

    F6b: ``_hash_file`` now also returns full 64-hex, so this helper is
    functionally identical. It is retained at the numerical case_binding call
    sites (``_validate_numerical_binding``) to document that the full sha256
    is required there (the ``_sha256`` key suffix is literal).
    """
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


#: F6c: evidence-file extensions hashed from INPUT_ARTIFACT_DIRS. Scratch XLA
#: dump byproducts (.ptx, .ll, .debug_options, .pbtxt, .ir-no-opt.ll,
#: .ir-with-opt.ll) are regenerable compiler byproducts NOT bound by
#: c1_judgment -- only .hlo / .txt / .json are evidence.
_EVIDENCE_EXTENSIONS = {".hlo", ".txt", ".json"}


def _hash_dir(dir_path):
    """{relative_path: full sha256} for each evidence file under dir_path.

    Keys are relative to the dir's PARENT (so 'c1_optimized_hlo/<file>'),
    '/'-joined. F6c: only files with extensions in ``_EVIDENCE_EXTENSIONS``
    (.hlo / .txt / .json) are hashed -- scratch XLA dump byproducts
    (.ptx / .ll / .debug_options / .pbtxt) are excluded because they are
    regenerable compiler byproducts, not evidence bound by c1_judgment.
    """
    out = {}
    if not os.path.isdir(dir_path):
        return out
    parent = os.path.dirname(dir_path)
    entries = []
    for root, _dirs, files in os.walk(dir_path):
        for name in files:
            ext = os.path.splitext(name)[1]
            if ext not in _EVIDENCE_EXTENSIONS:
                continue
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


def validate_required_artifacts(base, criterion):
    """Per-criterion presence check (Task 6 errata #5 / finding 3.7).

    Returns True iff ALL required artifacts for ``criterion`` exist under
    ``base``. Unknown criteria return True (vacuously -- no required
    artifacts to check). Content/hash validation stays in
    ``_validate_numerical_binding`` (binding chain); this is the presence
    gate that ensures NUMERICAL requires BOTH the JSON and the CSV before
    the criterion can be evaluated (the CSV was previously missing from
    REQUIRED_ARTIFACTS, allowing a PASS with only the JSON present).
    """
    required = REQUIRED_ARTIFACTS.get(criterion, [])
    return all(os.path.exists(os.path.join(base, r)) for r in required)


def _c2_artifact_paths(c2_judgment):
    """artifact_paths from the first case in c2_judgment.json (case-keyed dict)."""
    if not isinstance(c2_judgment, dict) or not c2_judgment:
        return {}
    first = next(iter(c2_judgment.values()))
    if isinstance(first, dict):
        return first.get("artifact_paths") or {}
    return {}


def _validate_c2_checkpoint(base, c2_judgment, c2_checkpoint):
    """Re-hash C2 binding source files; compare to c2_checkpoint_manifest hashes
    (plan §9 6.1 / spec §3.3.1 -- full required binding, fail-closed).

    Returns:
      OK          -- every required C2_CHECKPOINT_KEYS binding is present
                     (hash recorded + source path/file resolvable) AND matches.
      UNAVAILABLE -- any required binding is missing (hash not recorded, source
                     path not in artifact_paths, or source file absent on disk).
      MISMATCH    -- all required bindings are present but at least one hash
                     differs from the on-disk source file.

    No ``continue``-then-``OK``: every required key must be exercised. A single
    missing binding makes the whole chain UNAVAILABLE (cannot confirm); a single
    hash mismatch makes it MISMATCH (cannot trust).
    """
    if not isinstance(c2_checkpoint, dict) or not c2_checkpoint.get("artifact_hashes"):
        return "UNAVAILABLE"
    expected = c2_checkpoint["artifact_hashes"]
    paths = _c2_artifact_paths(c2_judgment)
    for key in C2_CHECKPOINT_KEYS:
        exp_full = expected.get(key)
        if not exp_full:
            return "UNAVAILABLE"  # missing required binding hash
        if key in C2_FIXED_PATH_KEYS:
            src = C2_FIXED_PATH_KEYS[key]
        else:
            path_key = C2_PATH_KEY_ALIASES.get(key, key)
            src = paths.get(path_key)
        if not src:
            return "UNAVAILABLE"  # missing required binding source path
        actual = _hash_file(_resolve_under_base(base, src))
        if actual is None:
            return "UNAVAILABLE"  # source file absent on disk
        if actual != exp_full:
            return "MISMATCH"  # hash mismatch (F6b: full 64-hex compare)
    return "OK"


def _validate_numerical_binding(base, numerical_json):
    """Re-hash ALL 9 numerical case_binding source files; compare to recorded
    full sha256 (plan §5.2-5.3 / spec §3.2 / §4.4 -- full required binding,
    fail-closed).

    Every entry in ``NUMERICAL_BINDINGS`` (the 3 structural sources + the 6
    route-specific sources + the numerical CSV) is hash-checked. There are NO
    presence-only files (finding 3.2 fix): mutating any of the 6 previously
    presence-only files now produces MISMATCH.

    Returns:
      OK          -- all 9 required hashes present + match AND all 9 source
                     files present.
      UNAVAILABLE -- any required hash missing from case_binding, any required
                     source file absent, or case_binding itself absent/malformed.
      MISMATCH    -- all required hashes present but at least one differs from
                     the on-disk source file.

    No ``continue``-then-``OK``: every required binding must be exercised.
    """
    if not isinstance(numerical_json, dict):
        return "UNAVAILABLE"
    binding = numerical_json.get("case_binding")
    if not isinstance(binding, dict) or not binding:
        return "UNAVAILABLE"
    # Phase 1: every required hash must be recorded. Any missing -> UNAVAILABLE.
    for _name, (_rel, hash_key) in NUMERICAL_BINDINGS.items():
        if not binding.get(hash_key):
            return "UNAVAILABLE"
    # Phase 2: every required source file must exist. Any absent -> UNAVAILABLE.
    for _name, (rel, _hash_key) in NUMERICAL_BINDINGS.items():
        if _hash_file_full(os.path.join(base, rel)) is None:
            return "UNAVAILABLE"
    # Phase 3: every recorded hash must match the on-disk file (full sha256).
    # Any diff -> MISMATCH.
    for _name, (rel, hash_key) in NUMERICAL_BINDINGS.items():
        exp = binding.get(hash_key)
        actual = _hash_file_full(os.path.join(base, rel))
        if actual is None or actual != exp:
            return "MISMATCH"
    return "OK"


def _apply_checkpoint_validation(criteria, c2_status, num_status):
    """Apply checkpoint validation results to the criteria dict (plan §9 6.2 /
    spec §3.3.1 -- fail-closed downgrade).

    UNAVAILABLE or MISMATCH on a binding chain -> the dependent criterion
    cannot be trusted -> force UNKNOWN (covers 'cannot retain PASS' and is
    fail-closed for FAIL too: a FAIL resting on a broken binding chain is
    also unconfirmable). This is the fail-closed fix for the Task 2a-deferred
    'UNAVAILABLE preserves PASS' fail-open surface.

    C2 cascade (F1): the C2 checkpoint validates the SHARED C2 artifact chain
    (C2_CHECKPOINT_KEYS: edge_map / peak_frontier / prototype / c2_judgment /
    source_hlo / buffer_assignment / allocation_audit) that EVERY C2
    sub-criterion rests on -- e.g. C2_REGION_KERNEL depends on the region
    prototype + edge map + peak frontier, the same shared artifacts. So a
    broken C2 binding downgrades the WHOLE C2 family to UNKNOWN, not just the
    top-level "C2". Without this cascade a broken C2 chain + numerical OK
    could leave region_fused VIABLE (its capability C2_REGION_KERNEL still
    PASS) while C2=UNKNOWN -- a fail-open on the spine (spec §3.3.1).

    The C2 family is identified by prefix (== "C2" or startswith "C2_")
    applied to keys PRESENT in the criteria dict, so it survives the Task 7
    criteria-key rename (C2_REGION_KERNEL -> C2_REGION_KERNEL_FEASIBILITY)
    without an exact-name enumeration, and aligns with the C2_* members of
    verdict_schema.CRITERIA_NAMES plus the top-level "C2". Absent C2-family
    keys need no downgrade.

    C1 is never affected (no checkpoint binding for C1).
    """
    out = dict(criteria)
    if c2_status in ("MISMATCH", "UNAVAILABLE"):
        # Cascade to the whole C2 family (spec §3.3.1: UNAVAILABLE or MISMATCH
        # -> dependent criterion UNKNOWN). The prefix test catches "C2" and
        # every C2_* sub-criterion present, robust to the Task 7 rename.
        for k in out:
            if k == "C2" or k.startswith("C2_"):
                out[k] = "UNKNOWN"
    if num_status in ("MISMATCH", "UNAVAILABLE") and "NUMERICAL" in out:
        out["NUMERICAL"] = "UNKNOWN"
    return out


def _extract_per_route_numerical(numerical_json, num_status):
    """Extract {route: PASS|FAIL|...} from numerical_validation.json's per_route
    list. If the numerical binding chain is broken (num_status != OK), return
    an empty dict so every route's numerical tri-state is UNDETERMINED
    (fail-closed: cannot trust per-route data whose input binding is
    unconfirmable)."""
    if num_status != "OK":
        return {}
    if not isinstance(numerical_json, dict):
        return {}
    per = {}
    for row in numerical_json.get("per_route") or []:
        if isinstance(row, dict) and row.get("criterion") in ("PASS", "FAIL"):
            route = row.get("route")
            if route is not None:
                per[route] = row["criterion"]
    return per


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
    criteria + cases + inputs/outputs. Deterministic given fixed generated_at.

    Pipeline (plan §9 6.3):
      load gonogo native criteria
        -> schema-v3 validation (validate_criteria -- Task 5 fold-in I1)
        -> presence validation
        -> binding/hash validation
        -> validated criteria/numerical
        -> recompute routes / completion / authorization / reasons / blocking
        -> render manifest

    Derived state (route_verdict / phase0_completion / phase1_authorization /
    reasons / blocking_artifacts) is RECOMPUTED from validated criteria via the
    §5 truth table (verdict_schema.recompute_derived_state). It is NEVER copied
    from gonogo.json -- the manifest must never present an internally-
    contradictory state (criterion UNKNOWN + dependent route VIABLE, or
    downgraded criteria + completion COMPLETE).
    """
    run_ctx = _load_json(os.path.join(base, "run_context.json"))
    gonogo = _load_json(os.path.join(base, "gonogo.json"))
    gonogo = gonogo if isinstance(gonogo, dict) else {}
    run_ctx = run_ctx if isinstance(run_ctx, dict) else {}
    # Task 6 (finding 3.6): run_context is v2 nested (measurement +
    # aggregation roles). The manifest records BOTH the measurement commit
    # (which commit produced the GPU evidence) and the aggregation commit
    # (which commit produced the aggregate). The flat v1 source_commit /
    # dirty_worktree / dirty_file_count reads are replaced.
    measurement = run_ctx.get("measurement") or {}
    aggregation = run_ctx.get("aggregation") or {}
    c1_j = _load_json(os.path.join(base, "c1_judgment.json"))
    c2_j = _load_json(os.path.join(base, "c2_judgment.json"))
    c2_ckpt = _load_json(os.path.join(base, "c2_checkpoint_manifest.json"))
    numerical = _load_json(os.path.join(base, "numerical_validation.json"))

    # Stage 1: load gonogo native criteria.
    gonogo_criteria = gonogo.get("criteria", {})
    # Stage 2a: schema-v3 validation (Task 5 fold-in I1 -- DRY gap: gonogo's
    # aggregate_two_layer already validates via validate_criteria; manifest
    # didn't). This scrubs detail tokens to UNKNOWN, fills missing required
    # criteria as NOT_RUN, validates C2_CANONICAL against the rollup, and sets
    # the C2 compat alias = C2_CANONICAL. Run BEFORE _apply_checkpoint_validation
    # so the C2 binding cascade (which downgrades the whole C2 family to
    # UNKNOWN on UNAVAILABLE/MISMATCH) takes effect AFTER the alias is set.
    # Verify honest state preserved: current artifacts are UNKNOWN/FAIL ->
    # validate_criteria keeps them UNKNOWN/INCONCLUSIVE (no promotion).
    criteria, _validation_notes = validate_criteria(gonogo_criteria)
    # Stage 2b: presence validation (missing artifacts -> NOT_RUN).
    criteria = _presence_check(criteria, base)
    # Stage 3: binding/hash validation.
    c2_status = _validate_c2_checkpoint(base, c2_j, c2_ckpt)
    num_status = _validate_numerical_binding(base, numerical)
    # Stage 4: validated criteria (downgrade on UNAVAILABLE/MISMATCH).
    criteria = _apply_checkpoint_validation(criteria, c2_status, num_status)
    # Stage 5: recompute derived state from validated criteria + per-route
    # numerical. If the numerical binding is broken, per-route data is not
    # trusted (empty dict -> all UNDETERMINED).
    per_route_num = _extract_per_route_numerical(numerical, num_status)
    derived = recompute_derived_state(criteria, per_route_num)

    inputs, outputs = _collect_inputs_outputs(base)
    cases = _build_cases(c1_j, c2_j, base)

    if generated_at is None:
        generated_at = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_source_commit": measurement.get("source_commit"),
        "aggregation_source_commit": aggregation.get("source_commit"),
        "aggregation_dirty_worktree": aggregation.get("dirty_worktree"),
        "aggregation_dirty_file_count": aggregation.get("dirty_file_count"),
        "commands": run_ctx.get("command_templates") or {},
        "environment_hash": _hash_file(os.path.join(base, "environment.json")),
        "criteria": criteria,
        "route_verdict": derived["route_verdict"],
        "phase0_completion": derived["phase0_completion"],
        "phase1_authorization": derived["phase1_authorization"],
        "reasons": derived["reasons"],
        "blocking_artifacts": derived["blocking_artifacts"],
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
