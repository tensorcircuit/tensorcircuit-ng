"""Git-tree-X-bound review subject for the self-validating release gate.

Plan Task 8 / v3-review errata. The review subject binds a specific Git tree X
(the reviewed commit) to the file hashes of 5 evidence files:

  * 3 phase0 artifacts (``results/phase0/manifest.json``,
    ``results/phase0/test_report.json``, ``results/phase0/closeout_facts.json``)
    -- read FROM Git tree X via ``git show <X>:<path>`` (the commit object, not
    the working tree).
  * 2 workspace-root docs (the spec + plan Markdown) -- read from the
    filesystem at ``workspace_root`` (docs live OUTSIDE the repo).

:func:`validate_review_subject` RECOMPUTES all 5 hashes from Git tree X (NOT
just checks fields exist) -- the critical errata that prevents a stale / forged
review_subject from passing. If the docs changed since the review_subject was
built, the hash mismatches and validation fails.

When ``dirty_worktree`` is True the validator also recomputes ``patch_sha256``
(sha256 of ``git diff <X>`` output) and ``untracked_hashes`` (per-untracked-file
sha256 of content) and compares them to the rs's values.

The current HEAD may be handoff Y (not required == X): the validator checks Git
tree X (a commit object), not the current checkout.

stdlib only (``hashlib`` / ``json`` / ``subprocess`` / ``pathlib``).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: The review_subject schema version (frozen).
SCHEMA_VERSION = 1

#: Phase0 artifact paths INSIDE the git repo (read from Git tree X via git show).
_PHASE0_MANIFEST_PATH = "results/phase0/manifest.json"
_PHASE0_TEST_REPORT_PATH = "results/phase0/test_report.json"
_PHASE0_CLOSEOUT_FACTS_PATH = "results/phase0/closeout_facts.json"

#: Doc paths OUTSIDE the git repo (read from workspace_root filesystem).
_SPEC_REL = "docs/superpowers/specs/2026-07-24-anti-cycle4-scope-reset-spec.md"
_PLAN_REL = (
    "docs/superpowers/plans/"
    "2026-07-24-phase0-nongpu-evidence-integrity-remediation-plan-v2.md"
)

#: The 5 required file-hash keys in a review_subject.
_REQUIRED_HASH_KEYS = (
    "spec_sha256",
    "plan_sha256",
    "artifact_manifest_sha256",
    "test_report_sha256",
    "closeout_facts_sha256",
)

#: The 3 phase0 (field_key, git_path) pairs recomputed from Git tree X.
_PHASE0_HASH_PAIRS = (
    ("artifact_manifest_sha256", _PHASE0_MANIFEST_PATH),
    ("test_report_sha256", _PHASE0_TEST_REPORT_PATH),
    ("closeout_facts_sha256", _PHASE0_CLOSEOUT_FACTS_PATH),
)

#: The 2 doc (field_key, relative_path) pairs recomputed from workspace_root.
_DOC_HASH_PAIRS = (
    ("spec_sha256", _SPEC_REL),
    ("plan_sha256", _PLAN_REL),
)


def _sha256_bytes(data: bytes) -> str:
    """sha256 hexdigest of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path) -> str:
    """sha256 hexdigest of a file's contents (64 KiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_full_sha(s):
    """True iff *s* is a 40-character string (loose hex-sha check)."""
    return isinstance(s, str) and len(s) == 40


def _git_cat_file_commit_exists(repo_cwd, git_tree_x):
    """True iff ``git_tree_x`` resolves to a valid commit object.

    Uses ``git cat-file -e <X>^{commit}`` (the commit-object peeling syntax).
    """
    try:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{git_tree_x}^{{commit}}"],
            cwd=str(repo_cwd),
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _git_show_path(repo_cwd, git_tree_x, path):
    """Return the bytes of *path* at commit *git_tree_x*, or None on error."""
    try:
        result = subprocess.run(
            ["git", "show", f"{git_tree_x}:{path}"],
            cwd=str(repo_cwd),
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception:
        return None


def _git_diff_output(repo_cwd, git_tree_x):
    """Return the bytes of ``git diff <git_tree_x>`` output, or None on error."""
    try:
        result = subprocess.run(
            ["git", "diff", git_tree_x],
            cwd=str(repo_cwd),
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception:
        return None


def _git_untracked_files(repo_cwd):
    """Return a list of untracked file paths (the ``??`` lines from
    ``git status --porcelain``). Returns None on git error."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_cwd),
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        paths = []
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            if line.startswith("?? "):
                p = line[3:].strip()
                # git status --porcelain quotes paths containing special chars.
                if p.startswith('"') and p.endswith('"'):
                    p = p[1:-1]
                paths.append(p)
        return paths
    except Exception:
        return None


def build_review_subject(
    subject_commit,
    dirty,
    spec_sha256,
    plan_sha256,
    artifact_manifest_sha256,
    test_report_sha256,
    closeout_facts_sha256,
    patch_sha256=None,
    untracked_hashes=None,
):
    """Build a review_subject dict binding Git tree X to 5 evidence file hashes.

    *subject_commit* must be a 40-character sha string (else ValueError).

    Returns a dict with ``schema_version=1``, ``subject_commit``,
    ``dirty_worktree``, the 5 file hashes, ``patch_sha256``, and
    ``untracked_hashes``.

    When *dirty* is True the caller SHOULD supply *patch_sha256* (sha256 of
    ``git diff <subject_commit>`` output) and *untracked_hashes*
    (``{path: sha256(content)}`` for untracked files); the validator will
    recompute and compare them. When *dirty* is False both should be None.
    """
    if not _is_full_sha(subject_commit):
        raise ValueError(
            "subject_commit must be a 40-char hex sha, got: " f"{subject_commit!r}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "subject_commit": subject_commit,
        "dirty_worktree": bool(dirty),
        "spec_sha256": spec_sha256,
        "plan_sha256": plan_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "test_report_sha256": test_report_sha256,
        "closeout_facts_sha256": closeout_facts_sha256,
        "patch_sha256": patch_sha256,
        "untracked_hashes": dict(untracked_hashes) if untracked_hashes else None,
    }


def validate_review_subject(rs, git_tree_x, workspace_root):
    """Validate a review_subject against Git tree X (recomputing 5 file hashes).

    Returns True ONLY if ALL of the following hold:

      1. ``git cat-file -e <git_tree_x>^{commit}`` succeeds (X is a valid
         commit object).
      2. ``rs["subject_commit"] == git_tree_x`` AND is a full 40-char sha.
      3. All 5 file-hash keys are present (non-empty).
      4. The 3 phase0 file hashes are RECOMPUTED from Git tree X via
         ``git show <git_tree_x>:results/phase0/{manifest,test_report,
         closeout_facts}.json`` -> sha256 -> compare to rs's values.
      5. The 2 doc hashes are RECOMPUTED from the workspace_root filesystem
         (``workspace_root/docs/superpowers/specs/...`` +
         ``.../plans/...``) -> sha256 -> compare to rs's values. If the docs
         changed since the rs was built, this fails.
      6. If ``rs["dirty_worktree"]`` is True: ``patch_sha256`` +
         ``untracked_hashes`` MUST be present AND the validator RECOMPUTES them
         (``patch_sha256`` = sha256 of ``git diff <git_tree_x>`` output;
         ``untracked_hashes`` = ``{path: sha256(content)}`` for untracked files)
         and compares to rs's values.

    Any git error (invalid commit, file not in tree, etc.) -> False (not raise).
    The current HEAD may be handoff Y (the validator checks Git tree X, a commit
    object, not the current checkout).
    """
    if not isinstance(rs, dict):
        return False

    repo_cwd = Path.cwd()

    # 1. git_tree_x must be a valid commit object.
    if not _git_cat_file_commit_exists(repo_cwd, git_tree_x):
        return False

    # 2. rs["subject_commit"] == git_tree_x AND is full 40-char sha.
    subject_commit = rs.get("subject_commit")
    if not _is_full_sha(subject_commit):
        return False
    if subject_commit != git_tree_x:
        return False

    # 3. All 5 file-hash keys present.
    for key in _REQUIRED_HASH_KEYS:
        if not rs.get(key):
            return False

    # 4. Recompute the 3 phase0 file hashes FROM Git tree X.
    for key, git_path in _PHASE0_HASH_PAIRS:
        content = _git_show_path(repo_cwd, git_tree_x, git_path)
        if content is None:
            return False
        if _sha256_bytes(content) != rs.get(key):
            return False

    # 5. Recompute the 2 doc hashes FROM workspace_root filesystem.
    root = Path(workspace_root)
    for key, rel in _DOC_HASH_PAIRS:
        doc_path = root / rel
        if not doc_path.is_file():
            return False
        if _sha256_file(doc_path) != rs.get(key):
            return False

    # 6. dirty_worktree handling.
    if rs.get("dirty_worktree"):
        patch_sha = rs.get("patch_sha256")
        untracked = rs.get("untracked_hashes")
        if not patch_sha:
            return False
        if not isinstance(untracked, dict):
            return False
        # Recompute patch_sha256 = sha256 of git diff <git_tree_x> output.
        diff_bytes = _git_diff_output(repo_cwd, git_tree_x)
        if diff_bytes is None:
            return False
        if _sha256_bytes(diff_bytes) != patch_sha:
            return False
        # Recompute untracked_hashes.
        untracked_paths = _git_untracked_files(repo_cwd)
        if untracked_paths is None:
            return False
        recomputed = {}
        for p in untracked_paths:
            fpath = Path(repo_cwd) / p
            try:
                recomputed[p] = _sha256_file(fpath)
            except Exception:
                return False
        if recomputed != untracked:
            return False

    return True


__all__ = [
    "SCHEMA_VERSION",
    "build_review_subject",
    "validate_review_subject",
]
