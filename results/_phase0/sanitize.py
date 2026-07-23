"""Unified privacy sanitizer for Phase 0 tracked artifacts (Task 8, spec §3.7).

Normalizes machine-specific strings in compiler/runtime diagnostics so
tracked artifacts carry no real env names, usernames, host absolute
paths, or private toolchain dirs.

Substitutions (order matters -- longer/more-specific first):

  1. home dir (absolute path)     -> ``<home>``
  2. repo dir  (absolute path)    -> ``<repo>``
  3. ``$HOME`` / ``~/`` / bare ``~`` -> ``<home>``   (existing placeholders / shell shorthand)
  4. ``$REPO``                     -> ``<repo>``
  5. toolchain clone dirs          -> ``<toolchain>``  (e.g. ``cutlass_spike``)
  6. conda env names               -> ``<env>``        (e.g. ``tcng``, ``nvcc_spike``)

PRESERVES diagnostic semantics: CUTLASS source-file references
(``sm120_mma_builder.inl:80``, ``mma_sm120.hpp:47``), relative file
positions, line numbers, and the substantive error text (F8F6F4
collective limit, ``__CUDA_ARCH__==1000`` gate) are NOT touched --
only machine-specific path/name tokens are normalized.

Applied BEFORE writing JSON/Markdown/text artifacts (plan §11).
"""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    """Absolute path to the tensorcircuit-ng repo root (two levels up)."""
    return os.path.dirname(os.path.dirname(_HERE))


# Conda env names used in the project's isolated spike toolchains.
# These are private environment names that must not appear in tracked
# artifacts (spec §3.7).
_ENV_NAMES = ("tcng", "nvcc_spike")

# Toolchain clone dir names (the CUTLASS source checkout used for probing).
_TOOLCHAIN_DIRS = ("cutlass_spike",)


def sanitize_text(
    text: str,
    *,
    home: str | None = None,
    repo: str | None = None,
    env_names: tuple[str, ...] = _ENV_NAMES,
    toolchain_dirs: tuple[str, ...] = _TOOLCHAIN_DIRS,
) -> str:
    """Return *text* with machine-specific strings normalized.

    Parameters
    ----------
    text
        The raw text to sanitize (serialized JSON, Markdown, HLO, etc.).
    home
        The home directory absolute path to replace.  Defaults to
        ``os.path.expanduser("~")`` at call time.
    repo
        The repository root absolute path to replace.  Defaults to
        two levels up from this module.
    env_names
        Conda env names to replace with ``<env>``.
    toolchain_dirs
        Toolchain clone dir names to replace with ``<toolchain>``.

    Returns
    -------
    str
        The sanitized text.  CUTLASS source-file references, line
        numbers, relative paths, and error text are preserved.

    Examples
    --------
    >>> sanitize_text("/home/alice/miniconda3/envs/tcng/bin/nvcc",
    ...               home="/home/alice", repo="/repo")
    '<home>/miniconda3/envs/<env>/bin/nvcc'
    >>> sanitize_text("$HOME/cutlass_spike/include/cutlass/gemm/collective/"
    ...               "builders/sm120_mma_builder.inl(80): error")
    '<home>/<toolchain>/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl(80): error'
    """
    if home is None:
        home = os.path.expanduser("~")
    if repo is None:
        repo = _repo_root()

    # 1. Absolute home dir (most specific -- do first so path fragments
    #    don't leave env-name-looking remnants).
    if home and home not in ("~", "/"):
        text = text.replace(home, "<home>")
    # 2. Absolute repo dir.
    if repo:
        text = text.replace(repo, "<repo>")
    # 3. Shell home shorthand + legacy $HOME placeholder.
    text = text.replace("$HOME", "<home>")
    text = text.replace("~/", "<home>/")
    # 4. Legacy $REPO placeholder.
    text = text.replace("$REPO", "<repo>")
    # 5. Toolchain clone dirs (e.g. cutlass_spike -> <toolchain>).
    for tc in toolchain_dirs:
        text = text.replace(tc, "<toolchain>")
    # 6. Conda env names (e.g. tcng, nvcc_spike -> <env>).
    for env in env_names:
        text = text.replace(env, "<env>")
    return text


def sanitize_file(path: str) -> bool:
    """Sanitize a file in-place, also normalizing CRLF -> LF.

    Returns ``True`` if the file content changed (private strings removed
    or line endings normalized), ``False`` if it was already clean.
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        original = fh.read()
    # Normalize CRLF -> LF (kill OneDrive phantoms) then sanitize.
    normalized = original.replace("\r\n", "\n")
    sanitized = sanitize_text(normalized)
    if sanitized != original:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(sanitized)
        return True
    return False


# --- C2 checkpoint manifest re-hash (post-sanitization) -------------------


def _resolve_under_base(base: str, path: str) -> str:
    """Resolve a repo-relative artifact path under *base*.

    ``artifact_paths`` in ``c2_judgment.json`` are repo-relative
    (``results/phase0/...``); strip that prefix and join under *base*
    so the file resolves regardless of the working directory.
    """
    for pfx in ("results/phase0/", "results\\phase0\\"):
        if path.startswith(pfx):
            path = path[len(pfx) :]
            break
    return os.path.join(base, path)


def rehash_c2_checkpoint(base: str = "results/phase0") -> bool:
    """Re-hash ALL C2 checkpoint binding keys in
    ``c2_checkpoint_manifest.json`` after sanitization.

    Sanitizing the HLO and buffer-assignment files changes their bytes,
    which cascades through the hash chain: the buffer-assignment audit
    (``c1_buffer_assignment/n24_d10_default.json``) records their hashes,
    the edge map (``c1_c2_edge_map.json``) records the audit hash, and
    the peak frontier (``c2_peak_frontier.json``) records the edge-map
    hash.  The C2 checkpoint manifest records all of these; if any go
    stale the manifest's fail-closed hash validation
    (``manifest._validate_c2_checkpoint``) returns MISMATCH and
    downgrades the C2 family to UNKNOWN -- changing verdicts.

    This function re-computes every C2 checkpoint binding hash from the
    on-disk source files so the manifest stays consistent.

    Returns ``True`` if the manifest was modified.
    """
    import hashlib
    import json

    ckpt_path = os.path.join(base, "c2_checkpoint_manifest.json")
    c2j_path = os.path.join(base, "c2_judgment.json")
    with open(ckpt_path) as fh:
        ckpt = json.load(fh)
    with open(c2j_path) as fh:
        c2j = json.load(fh)

    # artifact_paths from the first case.
    first_case = next(iter(c2j.values())) if c2j else {}
    paths = (
        (first_case.get("artifact_paths") or {}) if isinstance(first_case, dict) else {}
    )

    # C2 checkpoint binding keys (must match manifest.C2_CHECKPOINT_KEYS).
    # allocation_audit is stored as "audit" in artifact_paths;
    # c2_judgment is a fixed-path key (not in artifact_paths).
    _PATH_ALIASES = {"allocation_audit": "audit"}
    _FIXED_PATHS = {"c2_judgment": "c2_judgment.json"}
    _ALL_KEYS = (
        "source_hlo",
        "buffer_assignment",
        "allocation_audit",
        "edge_map",
        "peak_frontier",
        "prototype",
        "c2_judgment",
    )

    modified = False
    for key in _ALL_KEYS:
        if key in _FIXED_PATHS:
            src = _FIXED_PATHS[key]
        else:
            src = paths.get(_PATH_ALIASES.get(key, key))
        if not src:
            continue
        full = _resolve_under_base(base, src)
        if not os.path.exists(full):
            continue
        with open(full, "rb") as fh:
            new_hash = hashlib.sha256(fh.read()).hexdigest()
        old_hash = (ckpt.get("artifact_hashes") or {}).get(key)
        if new_hash != old_hash:
            ckpt["artifact_hashes"][key] = new_hash
            modified = True

    if modified:
        with open(ckpt_path, "w", newline="") as fh:
            json.dump(ckpt, fh, indent=2)
    return modified


def rehash_numerical_binding(base: str = "results/phase0") -> bool:
    """Re-hash ``case_binding`` source-file hashes in
    ``numerical_validation.json`` after sanitization.

    The numerical binding (``manifest._validate_numerical_binding``)
    compares ``case_binding`` hashes against the on-disk source files.
    If the edge map (``c1_c2_edge_map.json``) is regenerated by a test
    after sanitization (because the HLO it references was sanitized),
    its hash changes and the binding goes stale -> MISMATCH ->
    ``NUMERICAL`` downgraded to UNKNOWN.

    This function re-computes all three ``case_binding`` hashes from
    the on-disk source files so the binding stays consistent.

    Returns ``True`` if the JSON was modified.
    """
    import hashlib
    import json

    nv_path = os.path.join(base, "numerical_validation.json")
    with open(nv_path) as fh:
        nv = json.load(fh)
    binding = nv.get("case_binding")
    if not isinstance(binding, dict):
        return False

    # (file under base) -> binding key (must match manifest.NUMERICAL_BINDINGS).
    _BINDINGS = {
        "c1_c2_edge_map.json": "edge_map_hash",
        "region_prototype.json": "prototype_hash",
        "contraction_shapes.csv": "contraction_shapes_hash",
    }

    modified = False
    for rel, hash_key in _BINDINGS.items():
        full = os.path.join(base, rel)
        if not os.path.exists(full):
            continue
        with open(full, "rb") as fh:
            new_hash = hashlib.sha256(fh.read()).hexdigest()[:16]
        old_hash = binding.get(hash_key)
        if new_hash != old_hash:
            binding[hash_key] = new_hash
            modified = True

    if modified:
        nv["case_binding"] = binding
        with open(nv_path, "w", newline="") as fh:
            json.dump(nv, fh, indent=2)
    return modified
