"""Facts-only closeout + workspace-root-relative doc reference integrity.

Plan Task 7 / finding 3.8 / INV-5/INV-6.

The Phase 0 closeout is deliberately FACTS-ONLY: this module never self-awards
an ACCEPTED / VIABLE / merge-ready verdict. :func:`build_closeout_facts`
assembles the gate results, invariant results, open findings, and a headline
under ``self_verdict="PENDING_EXTERNAL_REVIEW"`` -- the actual release decision
is delegated to an independent reviewer (Task 10) and self-computed by
``derived_status`` (Task 8), never by this producer.

Finding 3.8 (P3): the prior closeout referenced the Spec / Plan via paths
written as ``docs/superpowers/...``, which are unresolvable from inside the
repo because the docs tree lives at the *workspace root* (``tc/``), not inside
``tensorcircuit-ng/``. INV-5/INV-6 require doc references to be
workspace-root-relative, resolvable, and hash-bound. This module provides the
two halves of that contract, kept STRICTLY SEPARATE per the v3-review errata:

  * :func:`compute_doc_hash` -- the GENERATION function. Computes the sha256
    of a file's bytes (the value a producer stores in a ref).
  * :func:`validate_doc_references` -- the VALIDATION function. Checks a list
    of refs against the filesystem WITHOUT modifying the input. Missing hash,
    absolute path, ``../`` escape outside the workspace root, missing file, or
    hash mismatch each cause the whole validation to return False
    (fail-closed; nothing is skipped).

stdlib only (``hashlib`` / ``os`` / ``pathlib``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: The only self-verdict a facts-only closeout may carry. The release verdict
#: is produced elsewhere (``derived_status`` validating an independent review
#: against Git tree X) -- never self-awarded here.
SELF_VERDICT = "PENDING_EXTERNAL_REVIEW"

#: The single permitted ``path_base`` for a doc reference. Doc trees (specs /
#: plans) live at the workspace root, NOT inside the repo, so a ref must
#: declare that its ``path`` is resolved relative to the workspace root.
WORKSPACE_ROOT_BASE = "workspace_root"


def build_closeout_facts(gate_results, headline, open_findings=None):
    """Assemble a facts-only closeout dict.

    Returns a dict with EXACTLY these keys (plan Task 7 Step 3 prose + errata)::

        {
          "self_verdict": "PENDING_EXTERNAL_REVIEW",
          "gate_results": gate_results,
          "invariant_results": {},
          "open_findings": open_findings or [],
          "headline": headline,
        }

    There is **no** ``task9_report_sha256`` field: the v3-review errata
    removes it because there is no producer for it in this module (a future
    task that actually produces a task-9 report would carry its hash through
    ``review_subject`` / ``derived_status`` instead). Adding a dangling
    ``task9_report_sha256`` here would be a self-referential placeholder,
    which is exactly the class of failure finding 3.8 flags.

    This function does NOT compute a verdict, does NOT inspect ``gate_results``
    for PASS/FAIL, and does NOT self-award any release status. It is a pure
    assembler of the caller-supplied facts.
    """
    return {
        "self_verdict": SELF_VERDICT,
        "gate_results": gate_results,
        "invariant_results": {},
        "open_findings": list(open_findings) if open_findings else [],
        "headline": headline,
    }


def compute_doc_hash(path):
    """GENERATION: return the sha256 hexdigest of the file at ``path``.

    This is the producer half of the doc-reference contract: a caller builds a
    ref dict ``{"path_base": "workspace_root", "path": <rel>, "sha256":
    compute_doc_hash(resolved)}`` and stores it. Validation is delegated to
    :func:`validate_doc_references` (kept separate per the errata -- generation
    must not also validate, and validation must not also generate).

    Reads the file in 64 KiB chunks so large specs/plans do not need to be
    held in memory. Raises ``FileNotFoundError`` if the path does not exist --
    callers that need fail-closed behavior should check existence first (as
    :func:`validate_doc_references` does).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_within(child, parent):
    """True iff resolved ``child`` is ``parent`` itself or nested under it.

    Uses :meth:`pathlib.Path.resolve` on both sides (symlinks resolved,
    ``..`` collapsed) and then a proper parent-chain membership test -- NOT
    string matching. ``"../etc/passwd"`` resolves outside the root and is
    rejected here even if the target happens to exist.
    """
    child_r = Path(child).resolve()
    parent_r = Path(parent).resolve()
    return child_r == parent_r or parent_r in child_r.parents


def validate_doc_references(refs, workspace_root):
    """VALIDATION: validate a list of doc references without modifying input.

    Each ref is a dict with ``path_base``, ``path``, ``sha256``. A ref passes
    iff ALL of the following hold (any failure short-circuits the whole
    validation to False -- nothing is skipped, per the errata):

      1. ``path_base == "workspace_root"`` (the only permitted base; doc trees
         live at the workspace root, not inside the repo).
      2. ``path`` is a non-empty RELATIVE string -- absolute paths are rejected
         via :func:`os.path.isabs` (an absolute path could point anywhere on
         the host and bypass the workspace-root confinement).
      3. ``resolved = (workspace_root / path).resolve()`` is WITHIN
         ``workspace_root.resolve()`` (no ``../`` escape). Uses
         :func:`_is_within` (proper parent-chain check, not string matching).
      4. ``sha256`` is present (not None / not empty). A missing hash is a
         FAILURE, not a skip (errata: "缺 hash 失败").
      5. The file EXISTS at ``resolved`` (``is_file``).
      6. The file's actual sha256 (computed fresh via
         :func:`compute_doc_hash`) MATCHES the ref's ``sha256``.

    Returns True only if every ref passes; False if any ref fails or if
    ``refs`` is empty-but-malformed.

    This function does NOT modify ``refs`` or any dict inside it (no mutation,
    per the errata: "验证不修改输入"). It only reads.
    """
    root = Path(workspace_root)
    for ref in refs:
        # Defensive: a ref must be a dict to carry the required keys.
        if not isinstance(ref, dict):
            return False

        # 1. path_base must be the workspace root.
        if ref.get("path_base") != WORKSPACE_ROOT_BASE:
            return False

        # 2. path must be a non-empty relative string.
        path = ref.get("path")
        if not isinstance(path, str) or not path:
            return False
        if os.path.isabs(path):
            return False

        # 3. resolved path must stay within the workspace root (no escape).
        resolved = (root / path).resolve()
        if not _is_within(resolved, root):
            return False

        # 4. sha256 must be present (None / empty -> fail, not skip).
        sha = ref.get("sha256")
        if not sha:
            return False

        # 5. file must exist at the resolved path.
        if not resolved.is_file():
            return False

        # 6. actual sha256 (computed fresh) must match the ref's hash.
        if compute_doc_hash(resolved) != sha:
            return False

    return True


__all__ = [
    "SELF_VERDICT",
    "WORKSPACE_ROOT_BASE",
    "build_closeout_facts",
    "compute_doc_hash",
    "validate_doc_references",
]
