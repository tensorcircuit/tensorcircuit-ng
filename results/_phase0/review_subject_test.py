"""TDD tests for ``review_subject.py`` (plan Task 8).

These tests pin the Git-tree-X recompute behavior of
:func:`validate_review_subject` -- the critical errata that prevents a stale /
forged review_subject from passing. The validator RECOMPUTES the 5 file hashes
from Git tree X (3 phase0 artifacts via ``git show <X>:<path>`` + 2 docs from
the workspace_root filesystem), NOT just checks fields exist.

Each test isolates ONE condition (flip one thing -> validation fails). The
positive fixtures use a temp git repo (via ``monkeypatch.chdir``) so the
validator's ``git show`` / ``git cat-file`` calls read from the temp repo's
commit object.

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
python -m pytest results/_phase0/review_subject_test.py -v
"""

import hashlib
import json
import subprocess

import pytest

from results._phase0.review_subject import (
    SCHEMA_VERSION,
    build_review_subject,
    validate_review_subject,
)

_SPEC_REL = "docs/superpowers/specs/2026-07-24-anti-cycle4-scope-reset-spec.md"
_PLAN_REL = (
    "docs/superpowers/plans/"
    "2026-07-24-phase0-nongpu-evidence-integrity-remediation-plan-v2.md"
)
_SPEC_CONTENT = b"# Anti-cycle4 scope reset spec\n"
_PLAN_CONTENT = b"# Phase0 nongpu evidence integrity remediation plan v2\n"

#: F6a: the manifest committed in the temp repo declares ``inputs`` (a dict
#: of {relative_path: hash}) so validate_review_subject's step 7 (input-chain
#: verification) has something to check. The default fixture commits one
#: input file (c1_judgment.json) referenced by the manifest.
_INPUT_CONTENT = b"input-data"
_INPUT_HASH = hashlib.sha256(_INPUT_CONTENT).hexdigest()
_MANIFEST_CONTENT = json.dumps(
    {
        "schema_version": "manifest-v1",
        "inputs": {"c1_judgment.json": _INPUT_HASH},
    }
).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _init_temp_repo(tmp_path, monkeypatch, manifest_inputs=None, input_files=None):
    """Create a temp git repo at *tmp_path* with phase0 artifacts + docs committed.

    ``monkeypatch.chdir`` switches cwd to the temp repo so the validator's git
    subprocess calls read from this repo. Returns the commit sha.

    *manifest_inputs* overrides the manifest's ``inputs`` dict (default:
    ``{"c1_judgment.json": _INPUT_HASH}``). *input_files* is a dict of
    {relative_path: bytes} for input files committed under results/phase0/
    (default: ``{"c1_judgment.json": _INPUT_CONTENT}``). F6a tests pass a
    manifest_inputs dict that references inputs NOT committed to X.
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

    # phase0 artifacts
    phase0 = tmp_path / "results" / "phase0"
    phase0.mkdir(parents=True)
    # Commit input files referenced by the manifest (F6a input chain).
    if input_files is None:
        input_files = {"c1_judgment.json": _INPUT_CONTENT}
    for rel, content in input_files.items():
        fpath = phase0 / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(content)
    if manifest_inputs is None:
        manifest = _MANIFEST_CONTENT
    else:
        manifest = json.dumps(
            {"schema_version": "manifest-v1", "inputs": manifest_inputs}
        ).encode()
    (phase0 / "manifest.json").write_bytes(manifest)
    test_report = json.dumps(
        {"schema_version": 1, "command": "...", "exit_code": 0, "passed": True}
    ).encode()
    closeout = json.dumps({"self_verdict": "PENDING_EXTERNAL_REVIEW"}).encode()
    (phase0 / "test_report.json").write_bytes(test_report)
    (phase0 / "closeout_facts.json").write_bytes(closeout)

    # docs at workspace_root (= tmp_path for the test)
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
    return result.stdout.strip()


def _good_hashes():
    """Return the 5 file hashes for the temp repo's committed content."""
    test_report = json.dumps(
        {"schema_version": 1, "command": "...", "exit_code": 0, "passed": True}
    ).encode()
    closeout = json.dumps({"self_verdict": "PENDING_EXTERNAL_REVIEW"}).encode()
    return {
        "spec_sha256": _sha(_SPEC_CONTENT),
        "plan_sha256": _sha(_PLAN_CONTENT),
        "artifact_manifest_sha256": _sha(_MANIFEST_CONTENT),
        "test_report_sha256": _sha(test_report),
        "closeout_facts_sha256": _sha(closeout),
    }


# ---------------------------------------------------------------------------
# build_review_subject
# ---------------------------------------------------------------------------


def test_build_review_subject_returns_correct_dict():
    rs = build_review_subject(
        subject_commit="a" * 40,
        dirty=False,
        spec_sha256="s",
        plan_sha256="p",
        artifact_manifest_sha256="m",
        test_report_sha256="t",
        closeout_facts_sha256="c",
    )
    assert rs["schema_version"] == SCHEMA_VERSION
    assert rs["subject_commit"] == "a" * 40
    assert rs["dirty_worktree"] is False
    assert rs["spec_sha256"] == "s"
    assert rs["plan_sha256"] == "p"
    assert rs["artifact_manifest_sha256"] == "m"
    assert rs["test_report_sha256"] == "t"
    assert rs["closeout_facts_sha256"] == "c"
    assert rs["patch_sha256"] is None
    assert rs["untracked_hashes"] is None


def test_build_review_subject_rejects_short_commit():
    with pytest.raises(ValueError):
        build_review_subject(
            subject_commit="short",
            dirty=False,
            spec_sha256="s",
            plan_sha256="p",
            artifact_manifest_sha256="m",
            test_report_sha256="t",
            closeout_facts_sha256="c",
        )


def test_build_review_subject_dirty_with_patch_and_untracked():
    rs = build_review_subject(
        subject_commit="a" * 40,
        dirty=True,
        spec_sha256="s",
        plan_sha256="p",
        artifact_manifest_sha256="m",
        test_report_sha256="t",
        closeout_facts_sha256="c",
        patch_sha256="patch",
        untracked_hashes={"foo.txt": "abc"},
    )
    assert rs["dirty_worktree"] is True
    assert rs["patch_sha256"] == "patch"
    assert rs["untracked_hashes"] == {"foo.txt": "abc"}


# ---------------------------------------------------------------------------
# validate_review_subject -- Git tree X recompute
# ---------------------------------------------------------------------------


def test_validate_invalid_commit_returns_false():
    """git_tree_x is not a valid commit object -> False."""
    rs = build_review_subject(subject_commit="a" * 40, dirty=False, **_good_hashes())
    assert validate_review_subject(rs, "z" * 40, workspace_root=".") is False


def test_validate_subject_commit_mismatch_returns_false(tmp_path, monkeypatch):
    """rs.subject_commit != git_tree_x -> False (even if both are valid shas)."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    # Use a different (but valid-looking) sha as git_tree_x.
    other = "b" * 40
    assert validate_review_subject(rs, other, workspace_root=str(tmp_path)) is False


def test_validate_subject_commit_not_40_chars_returns_false(tmp_path, monkeypatch):
    """rs.subject_commit is not a 40-char sha -> False."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    rs["subject_commit"] = "short"
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_missing_hash_returns_false(tmp_path, monkeypatch):
    """One of the 5 required hashes is missing -> False."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    rs["spec_sha256"] = None
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_phase0_file_hash_mismatch_returns_false(tmp_path, monkeypatch):
    """A phase0 file hash in rs doesn't match the recompute from Git tree X."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    rs["artifact_manifest_sha256"] = "0" * 64  # wrong hash
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_doc_hash_mismatch_returns_false(tmp_path, monkeypatch):
    """A doc hash in rs doesn't match the recompute from workspace_root."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    rs["spec_sha256"] = "0" * 64  # wrong hash
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_doc_file_missing_returns_false(tmp_path, monkeypatch):
    """A doc file is missing from workspace_root -> False."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    # Delete the spec file after commit (it's in the working tree, not git-tracked
    # in the real scenario, but here it's committed -- the validator reads from
    # the filesystem, not git).
    spec_path = tmp_path / _SPEC_REL
    spec_path.unlink()
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_all_valid_clean_returns_true(tmp_path, monkeypatch):
    """All 5 hashes match, dirty=False -> True (the positive clean fixture)."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is True


# ---------------------------------------------------------------------------
# dirty worktree
# ---------------------------------------------------------------------------


def test_validate_dirty_without_patch_returns_false(tmp_path, monkeypatch):
    """dirty=True but patch_sha256 is None -> False."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=True, **_good_hashes())
    # patch_sha256 defaults to None when not supplied.
    assert rs["patch_sha256"] is None
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_dirty_without_untracked_hashes_returns_false(tmp_path, monkeypatch):
    """dirty=True but untracked_hashes is None -> False."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(
        subject_commit=commit,
        dirty=True,
        patch_sha256="x",
        **_good_hashes(),
    )
    assert rs["untracked_hashes"] is None
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_dirty_patch_mismatch_returns_false(tmp_path, monkeypatch):
    """dirty=True with wrong patch_sha256 -> False."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    # Modify a tracked file to create a diff.
    (tmp_path / "results" / "phase0" / "manifest.json").write_bytes(b"modified")
    rs = build_review_subject(
        subject_commit=commit,
        dirty=True,
        patch_sha256="0" * 64,  # wrong
        untracked_hashes={},
        **_good_hashes(),
    )
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_dirty_with_correct_patch_and_untracked_returns_true(
    tmp_path, monkeypatch
):
    """dirty=True with correct patch_sha256 + untracked_hashes -> True.

    After committing, we modify a tracked file and add an untracked file, then
    build rs with the recomputed patch + untracked hashes. The validator
    recomputes them and they match -> True.
    """
    commit = _init_temp_repo(tmp_path, monkeypatch)
    # Modify a tracked file -> creates a diff.
    (tmp_path / "results" / "phase0" / "manifest.json").write_bytes(b"modified")
    # Add an untracked file.
    (tmp_path / "scratch.txt").write_bytes(b"untracked content")

    # Recompute patch_sha256 = sha256 of `git diff <commit>` output.
    diff_result = subprocess.run(
        ["git", "diff", commit], capture_output=True, cwd=str(tmp_path)
    )
    patch_sha = _sha(diff_result.stdout)

    # Recompute untracked_hashes.
    status_result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, cwd=str(tmp_path)
    )
    untracked = {}
    for line in status_result.stdout.decode().splitlines():
        if line.startswith("?? "):
            p = line[3:].strip().strip('"')
            untracked[p] = _sha((tmp_path / p).read_bytes())

    rs = build_review_subject(
        subject_commit=commit,
        dirty=True,
        patch_sha256=patch_sha,
        untracked_hashes=untracked,
        **_good_hashes(),
    )
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is True


def test_validate_dirty_untracked_mismatch_returns_false(tmp_path, monkeypatch):
    """dirty=True with correct patch but wrong untracked_hashes -> False."""
    commit = _init_temp_repo(tmp_path, monkeypatch)
    (tmp_path / "results" / "phase0" / "manifest.json").write_bytes(b"modified")
    (tmp_path / "scratch.txt").write_bytes(b"untracked content")

    diff_result = subprocess.run(
        ["git", "diff", commit], capture_output=True, cwd=str(tmp_path)
    )
    patch_sha = _sha(diff_result.stdout)

    rs = build_review_subject(
        subject_commit=commit,
        dirty=True,
        patch_sha256=patch_sha,
        untracked_hashes={"nonexistent.txt": "0" * 64},  # wrong
        **_good_hashes(),
    )
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


# ---------------------------------------------------------------------------
# rs not a dict / malformed
# ---------------------------------------------------------------------------


def test_validate_non_dict_rs_returns_false():
    assert validate_review_subject("not a dict", "a" * 40, ".") is False


# ---------------------------------------------------------------------------
# F6a: manifest input-chain verification (step 7)
# ---------------------------------------------------------------------------


def test_validate_manifest_input_not_in_x_clean_returns_false(tmp_path, monkeypatch):
    """F6a: manifest references an input NOT in X + dirty=False -> False.

    The manifest (committed in X) declares an input 'extra.json' that was
    never committed. A clean review_subject (dirty=False) claims X is a
    reproducible snapshot, but the evidence chain is broken -> False.
    """
    # manifest references c1_judgment.json (committed) + extra.json (NOT committed)
    commit = _init_temp_repo(
        tmp_path,
        monkeypatch,
        manifest_inputs={
            "c1_judgment.json": _INPUT_HASH,
            "extra.json": "0" * 64,  # never committed
        },
        # only c1_judgment.json is committed; extra.json is absent
        input_files={"c1_judgment.json": _INPUT_CONTENT},
    )
    # Recompute the manifest hash for the custom manifest (step 4 checks it).
    custom_manifest = json.dumps(
        {
            "schema_version": "manifest-v1",
            "inputs": {
                "c1_judgment.json": _INPUT_HASH,
                "extra.json": "0" * 64,
            },
        }
    ).encode()
    hashes = _good_hashes()
    hashes["artifact_manifest_sha256"] = _sha(custom_manifest)
    rs = build_review_subject(subject_commit=commit, dirty=False, **hashes)
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_manifest_all_inputs_in_x_clean_returns_true(tmp_path, monkeypatch):
    """F6a: manifest with all inputs in X + dirty=False -> True (positive).

    The default fixture commits c1_judgment.json (the manifest's sole input),
    so the input chain is fully retrievable from X.
    """
    commit = _init_temp_repo(tmp_path, monkeypatch)
    rs = build_review_subject(subject_commit=commit, dirty=False, **_good_hashes())
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is True


def test_validate_manifest_input_not_in_x_dirty_untracked_covers_returns_true(
    tmp_path, monkeypatch
):
    """F6a: input not in X + dirty=True + untracked_hashes covers it -> True.

    The manifest references 'extra.json' which is NOT in X but IS present as
    an untracked working-tree file. dirty=True with correct patch_sha256 +
    untracked_hashes (covering extra.json) -> step 6 passes, step 7 finds
    extra.json not in X but covered by untracked_hashes -> True.
    """
    # Commit manifest referencing c1_judgment.json (committed) + extra.json
    # (NOT committed). extra.json will be added as untracked after commit.
    commit = _init_temp_repo(
        tmp_path,
        monkeypatch,
        manifest_inputs={
            "c1_judgment.json": _INPUT_HASH,
            "extra.json": "0" * 64,
        },
        input_files={"c1_judgment.json": _INPUT_CONTENT},
    )
    # Add extra.json as an untracked working-tree file.
    extra_content = b"extra-untracked"
    (tmp_path / "results" / "phase0" / "extra.json").write_bytes(extra_content)

    # Recompute patch_sha256 (no tracked changes -> empty diff).
    diff_result = subprocess.run(
        ["git", "diff", commit], capture_output=True, cwd=str(tmp_path)
    )
    patch_sha = _sha(diff_result.stdout)

    # Recompute untracked_hashes.
    status_result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, cwd=str(tmp_path)
    )
    untracked = {}
    for line in status_result.stdout.decode().splitlines():
        if line.startswith("?? "):
            p = line[3:].strip().strip('"')
            untracked[p] = _sha((tmp_path / p).read_bytes())

    custom_manifest = json.dumps(
        {
            "schema_version": "manifest-v1",
            "inputs": {
                "c1_judgment.json": _INPUT_HASH,
                "extra.json": "0" * 64,
            },
        }
    ).encode()
    hashes = _good_hashes()
    hashes["artifact_manifest_sha256"] = _sha(custom_manifest)
    rs = build_review_subject(
        subject_commit=commit,
        dirty=True,
        patch_sha256=patch_sha,
        untracked_hashes=untracked,
        **hashes,
    )
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is True


def test_validate_manifest_input_not_in_x_dirty_not_covering_returns_false(
    tmp_path, monkeypatch
):
    """F6a: input not in X + dirty=True + untracked_hashes does NOT cover it
    -> False.

    The manifest references 'extra.json' which is neither in X nor in the
    working tree. dirty=True with empty patch + empty untracked (clean working
    tree) -> step 6 passes, but step 7 finds extra.json not in X and not in
    untracked_hashes -> False. This isolates step 7 (step 6 passes).
    """
    commit = _init_temp_repo(
        tmp_path,
        monkeypatch,
        manifest_inputs={
            "c1_judgment.json": _INPUT_HASH,
            "extra.json": "0" * 64,  # not in X, not in working tree
        },
        input_files={"c1_judgment.json": _INPUT_CONTENT},
    )
    # No working-tree changes -> empty diff + no untracked.
    diff_result = subprocess.run(
        ["git", "diff", commit], capture_output=True, cwd=str(tmp_path)
    )
    patch_sha = _sha(diff_result.stdout)

    custom_manifest = json.dumps(
        {
            "schema_version": "manifest-v1",
            "inputs": {
                "c1_judgment.json": _INPUT_HASH,
                "extra.json": "0" * 64,
            },
        }
    ).encode()
    hashes = _good_hashes()
    hashes["artifact_manifest_sha256"] = _sha(custom_manifest)
    rs = build_review_subject(
        subject_commit=commit,
        dirty=True,
        patch_sha256=patch_sha,
        untracked_hashes={},  # no untracked files
        **hashes,
    )
    assert validate_review_subject(rs, commit, workspace_root=str(tmp_path)) is False


def test_validate_manifest_no_inputs_returns_false(tmp_path, monkeypatch):
    """F6a: manifest with no 'inputs' dict (malformed) -> False.

    The manifest committed in X lacks the ``inputs`` field. Step 7 requires a
    dict inputs; a malformed manifest means the input chain cannot be verified.
    """
    # Start from the default fixture, then overwrite the manifest with one
    # lacking inputs and re-commit.
    commit = _init_temp_repo(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    bad_manifest = json.dumps({"schema_version": "manifest-v1"}).encode()
    (tmp_path / "results" / "phase0" / "manifest.json").write_bytes(bad_manifest)
    subprocess.run(
        ["git", "add", "."], check=True, capture_output=True, cwd=str(tmp_path)
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "bad manifest"],
        check=True,
        capture_output=True,
        cwd=str(tmp_path),
    )
    new_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(tmp_path),
    ).stdout.strip()
    hashes = _good_hashes()
    hashes["artifact_manifest_sha256"] = _sha(bad_manifest)
    rs = build_review_subject(subject_commit=new_commit, dirty=False, **hashes)
    assert (
        validate_review_subject(rs, new_commit, workspace_root=str(tmp_path)) is False
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
