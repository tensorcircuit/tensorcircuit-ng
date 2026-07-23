"""Unit tests for the Phase 0 reproducibility manifest (review §11 / plan §14).

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
  python -m pytest results/_phase0/manifest_test.py -v
"""

import os

from results._phase0.manifest import (
    SCHEMA_VERSION,
    REQUIRED_ARTIFACTS,
    INPUT_ARTIFACT_FILES,
    INPUT_ARTIFACT_DIRS,
    OUTPUT_ARTIFACTS,
    C2_CHECKPOINT_KEYS,
    C2_PATH_KEY_ALIASES,
    NUMERICAL_BINDINGS,
    _hash_file,
    _hash_dir,
    _resolve_under_base,
)


def test_schema_constants_complete():
    assert SCHEMA_VERSION == "manifest-v1"
    # every gonogo criterion has a required-artifact entry
    for c in (
        "C1",
        "C2",
        "C3_PLANAR_CORE",
        "C3_PLANAR_FULL_MATRIX",
        "C3_GROUPED",
        "CUTLASS_SM120_4M",
        "REGION_PROTOTYPE",
        "NUMERICAL",
    ):
        assert c in REQUIRED_ARTIFACTS and REQUIRED_ARTIFACTS[c], c
    assert "manifest.json" not in OUTPUT_ARTIFACTS  # no self-hash
    assert OUTPUT_ARTIFACTS == ["gonogo.json", "gonogo.md", "environment.json"]
    assert "c1_optimized_hlo" in INPUT_ARTIFACT_DIRS
    assert "allocation_audit" in C2_PATH_KEY_ALIASES  # alias -> audit path key
    assert C2_PATH_KEY_ALIASES["allocation_audit"] == "audit"


def test_hash_file_sha256_16(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello")
    # sha256("hello")[:16]
    assert _hash_file(str(p)) == "2cf24dba5fb0a30e"
    assert _hash_file(str(tmp_path / "missing.txt")) is None


def test_hash_dir_recursive_sorted(tmp_path):
    base = tmp_path / "phase0"
    d = base / "c1_optimized_hlo"
    d.mkdir(parents=True)
    (d / "n24.hlo").write_bytes(b"x")
    (d / "n22.hlo").write_bytes(b"y")
    out = _hash_dir(str(d))
    assert set(out) == {"c1_optimized_hlo/n24.hlo", "c1_optimized_hlo/n22.hlo"}
    assert all(len(v) == 16 for v in out.values())


def test_resolve_under_base_strips_phase0_prefix():
    assert _resolve_under_base("S", "results/phase0/c1_judgment.json") == os.path.join(
        "S", "c1_judgment.json"
    )
    # already-bare relative path passes through
    assert _resolve_under_base("S", "c1_judgment.json") == os.path.join(
        "S", "c1_judgment.json"
    )


def test_presence_check_all_present_inherits(tmp_path):
    from results._phase0.manifest import _presence_check, REQUIRED_ARTIFACTS

    # create every required file so nothing is missing
    for paths in REQUIRED_ARTIFACTS.values():
        for rel in paths:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
    criteria = {c: "PASS" for c in REQUIRED_ARTIFACTS}
    out = _presence_check(criteria, str(tmp_path))
    assert out == criteria  # nothing downgraded


def test_presence_check_missing_forces_not_run(tmp_path):
    from results._phase0.manifest import _presence_check

    criteria = {"C1": "PASS", "C2": "UNKNOWN", "NUMERICAL": "FAIL"}
    # only c1_judgment.json exists; c1_default_vs_nofusion.csv + c2/numerical missing
    (tmp_path / "c1_judgment.json").write_text("x")
    out = _presence_check(criteria, str(tmp_path))
    assert out["C1"] == "NOT_RUN"  # c1_default_vs_nofusion.csv missing
    assert out["C2"] == "NOT_RUN"  # c2 artifacts missing
    assert out["NUMERICAL"] == "NOT_RUN"


def test_c2_artifact_paths_reads_first_case(tmp_path):
    from results._phase0.manifest import _c2_artifact_paths

    c2j = {
        "n24_d10_default": {
            "artifact_paths": {
                "edge_map": "results/phase0/c1_c2_edge_map.json",
                "audit": "results/phase0/c1_buffer_assignment/n24_d10_default.json",
                "source_hlo": "results/phase0/c1_optimized_hlo/n24_d10_exp_default.hlo",
            }
        }
    }
    paths = _c2_artifact_paths(c2j)
    assert paths["edge_map"].endswith("c1_c2_edge_map.json")
    assert paths["audit"].endswith("n24_d10_default.json")
    assert _c2_artifact_paths({}) == {}


def test_validate_c2_checkpoint_ok_mismatch_unavailable(tmp_path):
    import hashlib
    from results._phase0.manifest import _validate_c2_checkpoint

    # make a source file whose sha256[:16] matches the recorded hash
    content = b"edge-data"
    full = hashlib.sha256(content).hexdigest()
    (tmp_path / "c1_c2_edge_map.json").write_bytes(content)
    c2j = {
        "n24_d10_default": {
            "artifact_paths": {"edge_map": "results/phase0/c1_c2_edge_map.json"}
        }
    }
    ok_ckpt = {"artifact_hashes": {"edge_map": full}}
    assert _validate_c2_checkpoint(str(tmp_path), c2j, ok_ckpt) == "OK"
    bad_ckpt = {"artifact_hashes": {"edge_map": "0" * 64}}
    assert _validate_c2_checkpoint(str(tmp_path), c2j, bad_ckpt) == "MISMATCH"
    assert _validate_c2_checkpoint(str(tmp_path), c2j, {}) == "UNAVAILABLE"


def test_validate_c2_checkpoint_alias_allocation_audit(tmp_path):
    import hashlib
    from results._phase0.manifest import _validate_c2_checkpoint

    content = b"audit-data"
    full = hashlib.sha256(content).hexdigest()
    # file placed where _resolve_under_base expects it (artifact_path is
    # results/phase0/c1_buffer_assignment/n24_d10_default.json -> strips to
    # c1_buffer_assignment/n24_d10_default.json under base)
    sub = tmp_path / "c1_buffer_assignment"
    sub.mkdir()
    (sub / "n24_d10_default.json").write_bytes(content)
    c2j = {
        "n24_d10_default": {
            "artifact_paths": {
                "audit": "results/phase0/c1_buffer_assignment/n24_d10_default.json"
            }
        }
    }
    # checkpoint records under key 'allocation_audit' (alias -> 'audit' path)
    ckpt = {"artifact_hashes": {"allocation_audit": full}}
    assert _validate_c2_checkpoint(str(tmp_path), c2j, ckpt) == "OK"


def test_validate_numerical_binding(tmp_path):
    import hashlib
    from results._phase0.manifest import _validate_numerical_binding

    content = b"edge-data"
    short = hashlib.sha256(content).hexdigest()[:16]
    (tmp_path / "c1_c2_edge_map.json").write_bytes(content)
    ok = {"case_binding": {"edge_map_hash": short}}
    assert _validate_numerical_binding(str(tmp_path), ok) == "OK"
    bad = {"case_binding": {"edge_map_hash": "deadbeef" * 2}}
    assert _validate_numerical_binding(str(tmp_path), bad) == "MISMATCH"
    assert _validate_numerical_binding(str(tmp_path), {}) == "UNAVAILABLE"


def test_apply_checkpoint_validation_downgrades_pass_only(tmp_path):
    from results._phase0.manifest import _apply_checkpoint_validation

    criteria = {"C1": "PASS", "C2": "PASS", "NUMERICAL": "FAIL"}
    # C2 mismatch -> C2 UNKNOWN; NUMERICAL mismatch -> NUMERICAL UNKNOWN too
    out = _apply_checkpoint_validation(criteria, "MISMATCH", "OK")
    assert out["C2"] == "UNKNOWN"
    assert out["C1"] == "PASS"  # untouched
    out2 = _apply_checkpoint_validation(criteria, "OK", "MISMATCH")
    assert out2["NUMERICAL"] == "UNKNOWN"
    # unavailable -> no change (can't validate, don't downgrade)
    out3 = _apply_checkpoint_validation(criteria, "UNAVAILABLE", "UNAVAILABLE")
    assert out3 == criteria


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
