"""Self-validating release status gate (plan Task 8 / v3-review errata).

:func:`derive_release_status` is the capstone of the anti-fail-open architecture:
there is NO path to ``release == "ACCEPTED"`` except through genuine evidence.
It does NOT trust the external review's self-claim -- it SELF-COMPUTES every
condition:

  1. The external review (ext) ``verdict`` must be ``"ACCEPTED"``.
  2. The ext's ``findings`` are self-parsed: any OPEN P0 / P1 / P2 blocks.
  3. ``ext["review_subject_sha256"]`` must equal the sha256 of the rs_path file
     bytes (the ext was reviewing THIS exact review_subject).
  4. :func:`review_subject.validate_review_subject` must pass -- it RECOMPUTES
     the 5 file hashes from Git tree X (not just checks fields exist) and
     verifies the dirty-worktree binding. This validates Git tree X, NOT the
     current HEAD (the current HEAD may be handoff Y).
  5. The test_report must have ``passed is True`` (frozen schema).
  6. ``user_confirms`` must be True.

Any missing / unknown / conflict / fail -> ``NOT_ACCEPTED`` with a reason.
Returns ``{"release": "ACCEPTED"|"NOT_ACCEPTED", "reasons": [...]}``.

stdlib only (``hashlib`` / ``json`` / ``pathlib``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from results._phase0.review_subject import validate_review_subject

#: Severities that block release when OPEN.
_BLOCKING_SEVERITIES = frozenset({"P0", "P1", "P2"})


def derive_release_status(
    ext_path,
    rs_path,
    test_report_path,
    user_confirms,
    git_tree_x,
    workspace_root,
):
    """Self-compute the release status from external review + evidence.

    Parameters:
      ext_path: path to the external review JSON file.
      rs_path: path to the review_subject JSON file.
      test_report_path: path to the test_report JSON file.
      user_confirms: bool -- the human user's explicit confirmation.
      git_tree_x: the reviewed Git commit sha (40-char).
      workspace_root: path to the workspace root (where docs/ lives).

    Returns ``{"release": "ACCEPTED"|"NOT_ACCEPTED", "reasons": [...]}``.
    """
    reasons = []

    # --- 1. Load ext; verdict must be ACCEPTED. ---
    try:
        ext = json.loads(Path(ext_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "release": "NOT_ACCEPTED",
            "reasons": [f"ext load/parse error: {exc}"],
        }

    if not isinstance(ext, dict):
        return {
            "release": "NOT_ACCEPTED",
            "reasons": ["ext is not a JSON object"],
        }

    if ext.get("verdict") != "ACCEPTED":
        reasons.append(f"ext verdict={ext.get('verdict')!r} != ACCEPTED")

    # --- 2. Self-parse findings: open P0/P1/P2 blocks. ---
    findings = ext.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        severity = f.get("severity", "")
        status = f.get("status", "")
        if status == "OPEN" and severity in _BLOCKING_SEVERITIES:
            reasons.append(
                f"open {severity} finding blocks release: " f"{f.get('summary', f)}"
            )

    # --- 3. Compare ext["review_subject_sha256"] to sha256(rs file bytes). ---
    try:
        rs_bytes = Path(rs_path).read_bytes()
        rs_file_sha = hashlib.sha256(rs_bytes).hexdigest()
    except Exception as exc:
        reasons.append(f"rs file read error: {exc}")
        rs_file_sha = None

    ext_rs_sha = ext.get("review_subject_sha256")
    if rs_file_sha is not None and ext_rs_sha != rs_file_sha:
        reasons.append(
            "ext review_subject_sha256 != sha256(rs file bytes): "
            f"{ext_rs_sha!r} != {rs_file_sha!r}"
        )

    # --- 4. Load rs, call validate_review_subject (validates Git tree X). ---
    try:
        rs = json.loads(Path(rs_path).read_text(encoding="utf-8"))
    except Exception as exc:
        reasons.append(f"rs parse error: {exc}")
        rs = None

    if rs is not None:
        if not validate_review_subject(rs, git_tree_x, workspace_root):
            reasons.append("review_subject invalid: Git tree X recompute failed")

    # --- 5. Load test_report; check passed is True (frozen schema). ---
    try:
        tr = json.loads(Path(test_report_path).read_text(encoding="utf-8"))
    except Exception as exc:
        reasons.append(f"test_report load/parse error: {exc}")
        tr = None

    if tr is not None:
        if tr.get("passed") is not True:
            reasons.append(f"test_report passed={tr.get('passed')!r} != True")

    # --- 6. user_confirms must be True. ---
    if not user_confirms:
        reasons.append("user_confirms is False")

    if reasons:
        return {"release": "NOT_ACCEPTED", "reasons": reasons}
    return {"release": "ACCEPTED", "reasons": []}


__all__ = ["derive_release_status"]
