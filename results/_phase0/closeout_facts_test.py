"""TDD tests for ``closeout_facts.py`` (plan Task 7 / finding 3.8 / INV-5/INV-6).

These tests pin the facts-only closeout (``self_verdict=PENDING_EXTERNAL_REVIEW``
with NO ``task9_report_sha256`` -- there is no producer for it, per the v3-review
errata) and the workspace-root-relative doc reference integrity
(:func:`validate_doc_references`): missing hash fails, absolute path rejected,
``../`` escape outside root rejected, correct relative+hash passes.

They are the RED step of the TDD loop (the brief's Step 1). Verbatim from the
frozen Plan v3 Task 7 Step 1.
"""

import hashlib
from pathlib import Path

from results._phase0.closeout_facts import (
    build_closeout_facts,
    compute_doc_hash,
    validate_doc_references,
)


def test_closeout_self_verdict_pending_no_task9_field():
    cf = build_closeout_facts(gate_results={}, headline={})
    assert cf["self_verdict"] == "PENDING_EXTERNAL_REVIEW"
    assert "task9_report_sha256" not in cf  # removed (no producer)


def test_doc_ref_missing_hash_fails(tmp_path):
    f = tmp_path / "spec.md"
    f.write_text("x")
    refs = [{"path_base": "workspace_root", "path": "spec.md", "sha256": None}]
    assert validate_doc_references(refs, workspace_root=tmp_path) is False


def test_doc_ref_correct_relative_hash_passes(tmp_path):
    (tmp_path / "spec.md").write_text("x")
    h = hashlib.sha256(b"x").hexdigest()
    refs = [{"path_base": "workspace_root", "path": "spec.md", "sha256": h}]
    assert validate_doc_references(refs, workspace_root=tmp_path) is True


def test_doc_ref_absolute_path_rejected(tmp_path):
    f = tmp_path / "spec.md"
    f.write_text("x")
    refs = [
        {
            "path_base": "workspace_root",
            "path": str(f),
            "sha256": hashlib.sha256(b"x").hexdigest(),
        }
    ]
    assert (
        validate_doc_references(refs, workspace_root=tmp_path) is False
    )  # absolute rejected


def test_doc_ref_escape_outside_root_rejected(tmp_path):
    (tmp_path / "spec.md").write_text("x")
    h = hashlib.sha256(b"x").hexdigest()
    refs = [{"path_base": "workspace_root", "path": "../etc/passwd", "sha256": h}]
    assert (
        validate_doc_references(refs, workspace_root=tmp_path) is False
    )  # escape rejected
