"""Unified privacy sanitizer for Phase 0 tracked artifacts (Task 8, spec §3.7).

Normalizes machine-specific strings in compiler/runtime diagnostics so
tracked artifacts carry no real env names, usernames, host absolute
paths, or private toolchain dirs.

Substitutions (order matters -- longer/more-specific first):

  1. home dir (absolute path)     -> ``<home>``
  2. repo dir  (absolute path)    -> ``<repo>``
  3. ``$HOME`` / ``~/`` / bare ``~`` -> ``<home>``   (existing placeholders / shell shorthand)
  4. ``$REPO``                     -> ``<repo>``
  5. toolchain clone dirs          -> ``<toolchain>``
  6. conda env names               -> ``<env>``
  7. caller ``redact`` mapping     -> (extra replacements, applied last)

Private name tokens (env names, toolchain clone dir names) are extracted
DYNAMICALLY at call time from ``CONDA_PREFIX`` / ``CUDA_HOME`` /
``CUTLASS_ROOT`` (basenames), never hardcoded as module constants (spec
§3.9, AGENTS.md: do not list real env/toolchain names in source). An
optional ``redact`` mapping / CLI ``--redact OLD:NEW`` flag adds
caller-supplied replacements on top of the dynamic set.

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


def _conda_env_name(path: str) -> str | None:
    """Return the conda env name when *path* is inside a ``.../envs/<name>`` dir.

    Returns ``None`` for empty paths or paths not under a conda ``envs/``
    directory, so a non-conda ``CUDA_HOME`` (e.g. ``/usr/local/cuda``) is NOT
    mis-redacted as an env name (avoids touching the generic word ``cuda``).
    Both ``/`` and ``\\`` separators are accepted (Windows conda paths).
    """
    if not path:
        return None
    parts = path.replace("\\", "/").split("/")
    for i in range(len(parts) - 1):
        if parts[i] == "envs" and parts[i + 1]:
            return parts[i + 1]
    return None


def _dynamic_private_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract ``(env_names, toolchain_dirs)`` dynamically from the runtime env.

    * env_names: conda env basenames from ``CONDA_PREFIX`` and ``CUDA_HOME``
      (only when ``CUDA_HOME`` points inside a conda ``envs/`` dir) ->
      replaced with ``<env>``.
    * toolchain_dirs: basename of ``CUTLASS_ROOT`` (the CUTLASS source clone)
      -> replaced with ``<toolchain>``.

    No real env/toolchain names are hardcoded (spec §3.9, AGENTS.md);
    extraction is purely structural. Returns empty tuples when the env vars
    are unset (e.g. CI without the toolchain), in which case only the
    home/repo/``$HOME``/``$REPO`` substitutions apply -- re-running sanitize
    on already-placeholdered artifacts stays a no-op (idempotent).
    """
    env_names: list[str] = []
    for var in ("CONDA_PREFIX", "CUDA_HOME"):
        name = _conda_env_name(os.environ.get(var, ""))
        if name and name not in env_names:
            env_names.append(name)
    toolchain_dirs: list[str] = []
    cutlass_root = os.environ.get("CUTLASS_ROOT", "")
    if cutlass_root:
        base = os.path.basename(os.path.normpath(cutlass_root))
        if base and base not in toolchain_dirs:
            toolchain_dirs.append(base)
    return tuple(env_names), tuple(toolchain_dirs)


def sanitize_text(
    text: str,
    *,
    home: str | None = None,
    repo: str | None = None,
    env_names: tuple[str, ...] | None = None,
    toolchain_dirs: tuple[str, ...] | None = None,
    redact: dict[str, str] | None = None,
) -> str:
    """Return *text* with machine-specific strings normalized.

    Private env/toolchain names are extracted dynamically from the runtime
    env when ``env_names`` / ``toolchain_dirs`` are not supplied (spec §3.9);
    callers may pass explicit tuples (tests use fictional names).

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
        Conda env names to replace with ``<env>``.  Defaults to the
        dynamic extraction from ``CONDA_PREFIX`` / ``CUDA_HOME``.
    toolchain_dirs
        Toolchain clone dir names to replace with ``<toolchain>``.
        Defaults to the dynamic extraction from ``CUTLASS_ROOT``.
    redact
        Optional extra ``{old: new}`` replacements applied last
        (wired to the CLI ``--redact OLD:NEW`` flag).

    Returns
    -------
    str
        The sanitized text.  CUTLASS source-file references, line
        numbers, relative paths, and error text are preserved.

    Examples
    --------
    >>> sanitize_text("/home/alice/miniconda3/envs/example-env-alpha/bin/nvcc",
    ...               home="/home/alice", repo="/repo",
    ...               env_names=("example-env-alpha",))
    '<home>/miniconda3/envs/<env>/bin/nvcc'
    >>> sanitize_text("$HOME/example-toolchain-beta/include/cutlass/gemm/"
    ...               "collective/builders/sm120_mma_builder.inl(80): error",
    ...               home="/home/alice", repo="/repo",
    ...               toolchain_dirs=("example-toolchain-beta",))
    '<home>/<toolchain>/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl(80): error'
    """
    if home is None:
        home = os.path.expanduser("~")
    if repo is None:
        repo = _repo_root()
    if env_names is None or toolchain_dirs is None:
        dyn_env, dyn_tc = _dynamic_private_names()
        if env_names is None:
            env_names = dyn_env
        if toolchain_dirs is None:
            toolchain_dirs = dyn_tc

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
    # 5. Toolchain clone dirs (basename of CUTLASS_ROOT) -> <toolchain>. Replace
    #    the already-bracketed form (<name>) FIRST so a pre-wrapped token does
    #    not double-wrap into <<toolchain>>; then the bare form.
    for tc in toolchain_dirs:
        text = text.replace(f"<{tc}>", "<toolchain>")
        text = text.replace(tc, "<toolchain>")
    # 6. Conda env names (basename of CONDA_PREFIX / CUDA_HOME) -> <env>. Same
    #    bracketed-first ordering: <name> -> <env> (not <<env>>), then bare.
    #    This is NOT a blanket << -> < collapse -- only the known private
    #    tokens are touched, so C++ template/shift syntax survives intact.
    for env in env_names:
        text = text.replace(f"<{env}>", "<env>")
        text = text.replace(env, "<env>")
    # 7. Caller-supplied extra redactions (CLI --redact), applied last.
    if redact:
        for old, new in redact.items():
            text = text.replace(old, new)
    return text


def sanitize_file(path: str, *, redact: dict[str, str] | None = None) -> bool:
    """Sanitize a file in-place, also normalizing CRLF -> LF.

    Returns ``True`` if the file content changed (private strings removed
    or line endings normalized), ``False`` if it was already clean.
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        original = fh.read()
    # Normalize CRLF -> LF (kill OneDrive phantoms) then sanitize.
    normalized = original.replace("\r\n", "\n")
    sanitized = sanitize_text(normalized, redact=redact)
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
    """Re-hash ALL 9 ``case_binding`` source-file hashes in
    ``numerical_validation.json`` after privacy sanitization (plan §5.4).

    The numerical binding (``manifest._validate_numerical_binding``) compares
    ``case_binding`` hashes against the on-disk source files. If a source file
    is regenerated or sanitized its hash changes and the binding goes stale ->
    MISMATCH -> ``NUMERICAL`` downgraded to UNKNOWN. This function re-computes
    all 9 full-sha256 hashes from the POST-sanitization on-disk source files
    so the binding stays consistent.

    Ordering invariant (plan §5.4): the canonical pipeline is
    ``sanitize -> generate/recompute numerical CSV/JSON -> compute final
    bindings -> generate gonogo -> generate manifest``. This rehash is the
    privacy-sanitization shortcut: it re-computes hashes WITHOUT re-running
    the numerical producer, which is valid only for non-semantic (path/name)
    sanitization. For changes that affect numerical/structural semantics the
    producer MUST be re-run -- the sanitizer must NOT mask a source content
    change by only rewriting the expected hash.

    Returns ``True`` if the JSON was modified.
    """
    import hashlib
    import json

    from results._phase0.manifest import NUMERICAL_BINDINGS

    nv_path = os.path.join(base, "numerical_validation.json")
    with open(nv_path) as fh:
        nv = json.load(fh)
    binding = nv.get("case_binding")
    if not isinstance(binding, dict):
        return False

    modified = False
    for rel, hash_key in NUMERICAL_BINDINGS.values():
        full = os.path.join(base, rel)
        if not os.path.exists(full):
            continue
        with open(full, "rb") as fh:
            new_hash = hashlib.sha256(fh.read()).hexdigest()
        old_hash = binding.get(hash_key)
        if new_hash != old_hash:
            binding[hash_key] = new_hash
            modified = True

    if modified:
        nv["case_binding"] = binding
        with open(nv_path, "w", newline="") as fh:
            json.dump(nv, fh, indent=2)
    return modified


# --- CLI (optional extra --redact replacements, spec §3.9) -----------------


def _cli(argv: list[str] | None = None) -> int:
    """``python -m results._phase0.sanitize [--redact OLD:NEW]... <file>...``

    Sanitize files in-place using dynamic env extraction plus any
    caller-supplied ``--redact OLD:NEW`` pairs (repeatable). The extra
    redactions are applied after the dynamic home/repo/env/toolchain
    substitutions. Returns 0 on success.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="sanitize",
        description="In-place privacy sanitizer for Phase 0 artifacts.",
    )
    parser.add_argument("files", nargs="+", help="files to sanitize in-place")
    parser.add_argument(
        "--redact",
        action="append",
        default=[],
        metavar="OLD:NEW",
        help="extra OLD->NEW replacement (repeatable); applied after dynamic extraction",
    )
    args = parser.parse_args(argv)
    redact: dict[str, str] = {}
    for pair in args.redact:
        if ":" in pair:
            old, new = pair.split(":", 1)
            redact[old] = new
    for f in args.files:
        sanitize_file(f, redact=redact or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
