"""TDD tests for ``derived_status.py`` (plan Task 8).

These tests pin the self-validating release gate: there is NO path to
``release == "ACCEPTED"`` except through genuine evidence. Each negative test
isolates ONE condition (flip one thing -> NOT_ACCEPTED). The ACCEPTED positive
fixture builds a valid review_subject from a temp git repo and verifies all
conditions pass.

The plan's Step 1 tests (4 negative tests) are included verbatim (adapted to
the actual function signature which takes file paths, not dicts).

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
python -m pytest results/_phase0/derived_status_test.py -v
"""

import hashlib
import json
import subprocess

from results._phase0.derived_status import derive_release_status
from results._phase0.review_subject import build_review_subject

_SPEC_REL = "docs/superpowers/specs/2026-07-24-anti-cycle4-scope-reset-spec.md"
_PLAN_REL = (
    "docs/superpowers/plans/"
    "2026-07-24-phase0-nongpu-evidence-integrity-remediation-plan-v2.md"
)
_SPEC_CONTENT = b"# Anti-cycle4 scope reset spec\n"
_PLAN_CONTENT = b"# Phase0 nongpu evidence integrity remediation plan v2\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _w(tmp_path, name, obj):
    """Write a JSON object to tmp_path/name and return the path string."""
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _init_temp_repo(tmp_path, monkeypatch):
    """Create a temp git repo with phase0 artifacts + docs committed.

    Returns (commit_sha, workspace_root_str, file_hashes_dict).
    """
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"],
        check=True,
        capture_output=True,
    )

    phase0 = tmp_path / "results" / "phase0"
    phase0.mkdir(parents=True)
    manifest = json.dumps({"schema_version": "manifest-v1"}).encode()
    test_report = json.dumps(
        {"schema_version": 1, "command": "...", "exit_code": 0, "passed": True}
    ).encode()
    closeout = json.dumps({"self_verdict": "PENDING_EXTERNAL_REVIEW"}).encode()
    (phase0 / "manifest.json").write_bytes(manifest)
    (phase0 / "test_report.json").write_bytes(test_report)
    (phase0 / "closeout_facts.json").write_bytes(closeout)

    specs_dir = tmp_path / "docs" / "superpowers" / "specs"
    plans_dir = tmp_path / "docs" / "superpowers" / "plans"
    specs_dir.mkdir(parents=True)
    plans_dir.mkdir(parents=True)
    (specs_dir / "2026-07-24-anti-cycle4-scope-reset-spec.md").write_bytes(
        _SPEC_CONTENT
    )
    (
        plans_dir / "2026-07-24-phase0-nongpu-evidence-integrity-remediation-plan-v2.md"
    ).write_bytes(_PLAN_CONTENT)

    subprocess.run(["git", "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test"], check=True, capture_output=True
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    commit = result.stdout.strip()
    hashes = {
        "spec_sha256": _sha(_SPEC_CONTENT),
        "plan_sha256": _sha(_PLAN_CONTENT),
        "artifact_manifest_sha256": _sha(manifest),
        "test_report_sha256": _sha(test_report),
        "closeout_facts_sha256": _sha(closeout),
    }
    return commit, str(tmp_path), hashes


# ---------------------------------------------------------------------------
# Plan Step 1 negative tests (each isolates ONE condition -> NOT_ACCEPTED)
# ---------------------------------------------------------------------------


def test_not_accepted_verdict_not_accepted(tmp_path):
    """ext verdict != ACCEPTED -> NOT_ACCEPTED."""
    ext = _w(
        tmp_path,
        "ext.json",
        {"verdict": "NOT_ACCEPTED", "findings": [], "review_subject_sha256": "x"},
    )
    rs = _w(
        tmp_path,
        "rs.json",
        {
            "schema_version": 1,
            "subject_commit": "a" * 40,
            "dirty_worktree": False,
            "spec_sha256": "s",
            "plan_sha256": "p",
            "artifact_manifest_sha256": "m",
            "test_report_sha256": "t",
            "closeout_facts_sha256": "c",
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"


def test_open_p2_not_accepted(tmp_path):
    """ext verdict=ACCEPTED but an OPEN P2 finding -> NOT_ACCEPTED."""
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [{"severity": "P2", "status": "OPEN"}],
            "review_subject_sha256": "x",
        },
    )
    rs = _w(
        tmp_path,
        "rs.json",
        {
            "schema_version": 1,
            "subject_commit": "a" * 40,
            "dirty_worktree": False,
            "spec_sha256": "s",
            "plan_sha256": "p",
            "artifact_manifest_sha256": "m",
            "test_report_sha256": "t",
            "closeout_facts_sha256": "c",
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"


def test_invalid_subject_commit_not_accepted(tmp_path):
    """rs.subject_commit is not a 40-char sha -> NOT_ACCEPTED."""
    ext = _w(
        tmp_path,
        "ext.json",
        {"verdict": "ACCEPTED", "findings": [], "review_subject_sha256": "x"},
    )
    rs = _w(
        tmp_path,
        "rs.json",
        {
            "schema_version": 1,
            "subject_commit": "short",
            "dirty_worktree": False,
            "spec_sha256": "s",
            "plan_sha256": "p",
            "artifact_manifest_sha256": "m",
            "test_report_sha256": "t",
            "closeout_facts_sha256": "c",
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"  # subject_commit not full sha


def test_dirty_without_patch_hash_not_accepted(tmp_path):
    """dirty=True but patch_sha256 is None -> NOT_ACCEPTED."""
    ext = _w(
        tmp_path,
        "ext.json",
        {"verdict": "ACCEPTED", "findings": [], "review_subject_sha256": "x"},
    )
    rs = _w(
        tmp_path,
        "rs.json",
        {
            "schema_version": 1,
            "subject_commit": "a" * 40,
            "dirty_worktree": True,
            "spec_sha256": "s",
            "plan_sha256": "p",
            "artifact_manifest_sha256": "m",
            "test_report_sha256": "t",
            "closeout_facts_sha256": "c",
            "patch_sha256": None,
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"  # dirty but no patch hash


# ---------------------------------------------------------------------------
# Additional per-condition isolation tests
# ---------------------------------------------------------------------------


def test_open_p0_not_accepted(tmp_path):
    """OPEN P0 finding blocks release."""
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [{"severity": "P0", "status": "OPEN"}],
            "review_subject_sha256": "x",
        },
    )
    rs = _w(
        tmp_path,
        "rs.json",
        {
            "schema_version": 1,
            "subject_commit": "a" * 40,
            "dirty_worktree": False,
            "spec_sha256": "s",
            "plan_sha256": "p",
            "artifact_manifest_sha256": "m",
            "test_report_sha256": "t",
            "closeout_facts_sha256": "c",
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"


def test_open_p1_not_accepted(tmp_path):
    """OPEN P1 finding blocks release."""
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [{"severity": "P1", "status": "OPEN"}],
            "review_subject_sha256": "x",
        },
    )
    rs = _w(
        tmp_path,
        "rs.json",
        {
            "schema_version": 1,
            "subject_commit": "a" * 40,
            "dirty_worktree": False,
            "spec_sha256": "s",
            "plan_sha256": "p",
            "artifact_manifest_sha256": "m",
            "test_report_sha256": "t",
            "closeout_facts_sha256": "c",
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"


def test_closed_p2_does_not_block(tmp_path, monkeypatch):
    """A CLOSED P2 finding does NOT block (only OPEN P0/P1/P2 blocks)."""
    commit, ws_root, hashes = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **hashes)
    rs_path = str(tmp_path / "rs.json")
    with open(rs_path, "w") as f:
        json.dump(rs, f)
    rs_file_sha = hashlib.sha256((tmp_path / "rs.json").read_bytes()).hexdigest()
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [{"severity": "P2", "status": "CLOSED"}],
            "review_subject_sha256": rs_file_sha,
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs_path,
        tr,
        user_confirms=True,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] == "ACCEPTED", out["reasons"]


def test_user_confirms_false_not_accepted(tmp_path, monkeypatch):
    """user_confirms=False -> NOT_ACCEPTED (all other conditions met)."""
    commit, ws_root, hashes = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **hashes)
    rs_path = str(tmp_path / "rs.json")
    with open(rs_path, "w") as f:
        json.dump(rs, f)
    rs_file_sha = hashlib.sha256((tmp_path / "rs.json").read_bytes()).hexdigest()
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [],
            "review_subject_sha256": rs_file_sha,
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs_path,
        tr,
        user_confirms=False,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] != "ACCEPTED"


def test_test_report_not_passed_not_accepted(tmp_path, monkeypatch):
    """test_report.passed=False -> NOT_ACCEPTED (all other conditions met)."""
    commit, ws_root, hashes = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **hashes)
    rs_path = str(tmp_path / "rs.json")
    with open(rs_path, "w") as f:
        json.dump(rs, f)
    rs_file_sha = hashlib.sha256((tmp_path / "rs.json").read_bytes()).hexdigest()
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [],
            "review_subject_sha256": rs_file_sha,
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 1, "passed": False},
    )
    out = derive_release_status(
        ext,
        rs_path,
        tr,
        user_confirms=True,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] != "ACCEPTED"


def test_review_subject_sha256_mismatch_not_accepted(tmp_path, monkeypatch):
    """ext.review_subject_sha256 != sha256(rs file) -> NOT_ACCEPTED."""
    commit, ws_root, hashes = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **hashes)
    rs_path = str(tmp_path / "rs.json")
    with open(rs_path, "w") as f:
        json.dump(rs, f)
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [],
            "review_subject_sha256": "0" * 64,  # wrong
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs_path,
        tr,
        user_confirms=True,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] != "ACCEPTED"


# ---------------------------------------------------------------------------
# ACCEPTED positive fixture (all conditions met -> ACCEPTED)
# ---------------------------------------------------------------------------


def test_accepted_positive_all_conditions_met(tmp_path, monkeypatch):
    """The complete ACCEPTED positive fixture:

    ext verdict=ACCEPTED + no open P0/P1/P2 + review_subject_sha256 matches +
    valid rs (Git tree X recompute passes) + test_report passed + user_confirms
    -> release == ACCEPTED.

    Uses a temp git repo so validate_review_subject's ``git show`` / ``git
    cat-file`` calls read from the repo's commit object.
    """
    commit, ws_root, hashes = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **hashes)
    rs_path = str(tmp_path / "rs.json")
    with open(rs_path, "w") as f:
        json.dump(rs, f)
    rs_file_sha = hashlib.sha256((tmp_path / "rs.json").read_bytes()).hexdigest()
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [],
            "review_subject_sha256": rs_file_sha,
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs_path,
        tr,
        user_confirms=True,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] == "ACCEPTED", out["reasons"]


def test_accepted_reasons_empty_on_success(tmp_path, monkeypatch):
    """When release == ACCEPTED, the reasons list is empty."""
    commit, ws_root, hashes = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **hashes)
    rs_path = str(tmp_path / "rs.json")
    with open(rs_path, "w") as f:
        json.dump(rs, f)
    rs_file_sha = hashlib.sha256((tmp_path / "rs.json").read_bytes()).hexdigest()
    ext = _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": [],
            "review_subject_sha256": rs_file_sha,
        },
    )
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": True},
    )
    out = derive_release_status(
        ext,
        rs_path,
        tr,
        user_confirms=True,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] == "ACCEPTED"
    assert out["reasons"] == []


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
