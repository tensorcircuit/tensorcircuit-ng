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

#: F6a: the manifest committed in the temp repo declares ``inputs`` so
#: validate_review_subject's step 7 (input-chain verification) passes. The
#: fixture commits one input file (c1_judgment.json) referenced by the manifest.
_INPUT_CONTENT = b"input-data"
_INPUT_HASH = hashlib.sha256(_INPUT_CONTENT).hexdigest()

#: F8a: the exact bytes of the committed test_report.json in the temp repo.
#: Positive tests must pass a tr file whose byte hash matches
#: rs["test_report_sha256"] (which is the hash of the committed test_report).
_COMMITTED_TR_CONTENT = json.dumps(
    {"schema_version": 1, "command": "...", "exit_code": 0, "passed": True}
).encode()


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
    # Commit an input file referenced by the manifest (F6a input chain).
    (phase0 / "c1_judgment.json").write_bytes(_INPUT_CONTENT)
    manifest = json.dumps(
        {
            "schema_version": "manifest-v1",
            "inputs": {"c1_judgment.json": _INPUT_HASH},
        }
    ).encode()
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


def _committed_tr(tmp_path):
    """Write a tr file matching the committed test_report.json bytes (F8a).

    The F8a fix binds ``test_report_path`` to ``rs["test_report_sha256"]``;
    positive tests must pass a tr file whose byte hash matches the rs's
    recorded hash (which is the hash of the committed test_report.json).
    """
    p = tmp_path / "tr.json"
    p.write_bytes(_COMMITTED_TR_CONTENT)
    return str(p)


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
    tr = _committed_tr(tmp_path)
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
    tr = _committed_tr(tmp_path)
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
    assert any("exit_code" in r for r in out["reasons"]), out["reasons"]


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
    tr = _committed_tr(tmp_path)
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
    tr = _committed_tr(tmp_path)
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
    tr = _committed_tr(tmp_path)
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


# ---------------------------------------------------------------------------
# F5 (evidence-integrity remediation): derive_release_status must fail-closed on
# (a) non-list findings / non-dict finding elements, and (b) a non-frozen
# test_report schema. Previously a dict findings was treated as empty (no open
# P0/P1/P2 detected) and a test_report with exit_code=1 + passed=True was
# accepted (fail-open).
# ---------------------------------------------------------------------------


#: F8a: the exact bytes of ``_valid_tr``'s content. ``_synthetic_rs`` records
#: this hash as ``test_report_sha256`` so the F8a byte-hash binding check
#: passes (isolating the F5 conditions under test).
_VALID_TR_CONTENT = json.dumps(
    {"schema_version": 1, "exit_code": 0, "passed": True}
).encode()
_VALID_TR_HASH = hashlib.sha256(_VALID_TR_CONTENT).hexdigest()


def _synthetic_rs(tmp_path):
    """Write a synthetic rs JSON (fails real Git-tree validation) for negative
    tests. The rs validation will fail-closed on the synthetic data, which is
    fine -- the tests assert NOT_ACCEPTED and check for the specific reason.

    F8a: ``test_report_sha256`` is set to the hash of ``_valid_tr``'s content
    so the F8a byte-hash binding check does NOT fire (isolating the test's
    intended condition)."""
    return _w(
        tmp_path,
        "rs.json",
        {
            "schema_version": 1,
            "subject_commit": "a" * 40,
            "dirty_worktree": False,
            "spec_sha256": "s",
            "plan_sha256": "p",
            "artifact_manifest_sha256": "m",
            "test_report_sha256": _VALID_TR_HASH,
            "closeout_facts_sha256": "c",
        },
    )


def _synthetic_ext(tmp_path, findings):
    """Write an ext JSON with verdict=ACCEPTED and the given findings."""
    return _w(
        tmp_path,
        "ext.json",
        {
            "verdict": "ACCEPTED",
            "findings": findings,
            "review_subject_sha256": "x",
        },
    )


def _valid_tr(tmp_path):
    """Write a valid test_report (frozen schema: v1, exit 0, passed True).

    F8a: writes the exact ``_VALID_TR_CONTENT`` bytes so the F8a byte-hash
    binding check matches ``_synthetic_rs``'s ``test_report_sha256``."""
    p = tmp_path / "tr.json"
    p.write_bytes(_VALID_TR_CONTENT)
    return str(p)


# --- F5a: findings must be a list of dicts ---


def test_findings_dict_not_accepted(tmp_path):
    """F5a: findings is a dict (not a list) -> NOT_ACCEPTED.

    Previously ``findings={"severity":"P0","status":"OPEN"}`` was treated as
    empty (``not isinstance(dict, list)`` -> ``findings = []``) -> no open
    P0/P1/P2 detected -> ACCEPTED (fail-open).
    """
    ext = _synthetic_ext(tmp_path, {"severity": "P0", "status": "OPEN"})
    rs = _synthetic_rs(tmp_path)
    tr = _valid_tr(tmp_path)
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"
    assert any("findings not a list" in r for r in out["reasons"]), out["reasons"]


def test_findings_non_dict_element_not_accepted(tmp_path):
    """F5a: a non-dict finding in the list -> NOT_ACCEPTED (don't skip).

    Previously ``continue`` skipped non-dict elements silently -> the open P0
    in element 0 was detected, but element 1 was silently dropped (no reason).
    Now each non-dict element adds its own reason.
    """
    ext = _synthetic_ext(tmp_path, [{"severity": "P0", "status": "OPEN"}, "not a dict"])
    rs = _synthetic_rs(tmp_path)
    tr = _valid_tr(tmp_path)
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"
    assert any("finding 1 not a dict" in r for r in out["reasons"]), out["reasons"]


def test_findings_string_not_accepted(tmp_path):
    """F5a: findings as a string -> NOT_ACCEPTED (not a list)."""
    ext = _synthetic_ext(tmp_path, "not a list")
    rs = _synthetic_rs(tmp_path)
    tr = _valid_tr(tmp_path)
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"
    assert any("findings not a list" in r for r in out["reasons"]), out["reasons"]


def test_findings_none_not_accepted(tmp_path):
    """F5a: findings=None -> NOT_ACCEPTED (not a list).

    ``ext.get("findings", [])`` returns None when the key is present with value
    None (the default is only used for absent keys). Previously None was treated
    as empty (``not isinstance(None, list)`` -> ``findings = []``) -> ACCEPTED.
    """
    ext = _w(
        tmp_path,
        "ext.json",
        {"verdict": "ACCEPTED", "findings": None, "review_subject_sha256": "x"},
    )
    rs = _synthetic_rs(tmp_path)
    tr = _valid_tr(tmp_path)
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"
    assert any("findings not a list" in r for r in out["reasons"]), out["reasons"]


# --- F5b: test_report frozen-schema check ---


def test_test_report_exit_code_1_not_accepted(tmp_path):
    """F5b: exit_code=1 + passed=True -> NOT_ACCEPTED.

    Previously only ``passed is True`` was checked, so a test_report with
    exit_code=1 (tests failed) but passed=True (stale/forged) was accepted.
    """
    ext = _synthetic_ext(tmp_path, [])
    rs = _synthetic_rs(tmp_path)
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 1, "passed": True},
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
    assert any("exit_code" in r for r in out["reasons"]), out["reasons"]


def test_test_report_missing_schema_version_not_accepted(tmp_path):
    """F5b: missing schema_version -> NOT_ACCEPTED (frozen schema requires v1)."""
    ext = _synthetic_ext(tmp_path, [])
    rs = _synthetic_rs(tmp_path)
    tr = _w(tmp_path, "tr.json", {"exit_code": 0, "passed": True})
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    assert out["release"] != "ACCEPTED"
    assert any("schema_version" in r for r in out["reasons"]), out["reasons"]


def test_test_report_passed_string_not_accepted(tmp_path):
    """F5b: passed='true' (string) -> NOT_ACCEPTED (must be bool True)."""
    ext = _synthetic_ext(tmp_path, [])
    rs = _synthetic_rs(tmp_path)
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": "true"},
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
    assert any("test_report passed" in r for r in out["reasons"]), out["reasons"]


def test_test_report_passed_int_not_accepted(tmp_path):
    """F5b: passed=1 (int) -> NOT_ACCEPTED (must be bool True, not 1)."""
    ext = _synthetic_ext(tmp_path, [])
    rs = _synthetic_rs(tmp_path)
    tr = _w(
        tmp_path,
        "tr.json",
        {"schema_version": 1, "exit_code": 0, "passed": 1},
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
    assert any("test_report passed" in r for r in out["reasons"]), out["reasons"]


def test_test_report_valid_passes_condition(tmp_path):
    """F5b: valid test_report (schema_version=1, exit_code=0, passed=True) ->
    no test_report reason (the frozen-schema check passes this condition).

    Note: the overall result may still be NOT_ACCEPTED from rs validation on
    synthetic data; this test only asserts the test_report check adds no reason.
    """
    ext = _synthetic_ext(tmp_path, [])
    rs = _synthetic_rs(tmp_path)
    tr = _valid_tr(tmp_path)
    out = derive_release_status(
        ext,
        rs,
        tr,
        user_confirms=True,
        git_tree_x="a" * 40,
        workspace_root=str(tmp_path),
    )
    tr_reasons = [r for r in out["reasons"] if r.startswith("test_report ")]
    assert tr_reasons == [], tr_reasons


# ---------------------------------------------------------------------------
# F8a: test_report_path must be bound to rs.test_report_sha256 by byte hash.
# Previously the caller passed a SEPARATE test_report_path; without comparing
# its byte hash to rs["test_report_sha256"], a forged test_report (different
# bytes, same exit_code=0/passed=True/schema_version=1) was accepted.
# ---------------------------------------------------------------------------


def test_f8a_test_report_byte_hash_mismatch_not_accepted(tmp_path, monkeypatch):
    """F8a: a test_report whose byte hash != rs.test_report_sha256 (but
    exit_code=0, passed=True, schema_version=1) -> NOT_ACCEPTED.

    Counter-example: pass a DIFFERENT test_report (different hash, forged
    command="not-the-subject-report") with exit_code=0, passed=true,
    schema_version=1. Without F8a, the frozen-schema check passes and the
    release is ACCEPTED (the test_report_path was never bound to the rs's
    recorded hash)."""
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
    # Forged test_report: different content (different hash) but valid schema.
    forged_tr = _w(
        tmp_path,
        "tr.json",
        {
            "schema_version": 1,
            "command": "not-the-subject-report",
            "exit_code": 0,
            "passed": True,
        },
    )
    out = derive_release_status(
        ext,
        rs_path,
        forged_tr,
        user_confirms=True,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] != "ACCEPTED"
    assert any("byte hash != rs.test_report_sha256" in r for r in out["reasons"]), out[
        "reasons"
    ]


def test_f8a_test_report_byte_hash_match_accepted(tmp_path, monkeypatch):
    """F8a: with the CORRECT test_report (byte hash matches rs's recorded hash)
    -> this condition passes (ACCEPTED if all other conditions met)."""
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
    tr = _committed_tr(tmp_path)
    out = derive_release_status(
        ext,
        rs_path,
        tr,
        user_confirms=True,
        git_tree_x=commit,
        workspace_root=ws_root,
    )
    assert out["release"] == "ACCEPTED", out["reasons"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
