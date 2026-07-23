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
    C2_FIXED_PATH_KEYS,
    NUMERICAL_BINDINGS,
    NUMERICAL_REQUIRED_FILES,
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
    # plan §9 6.1: ALL 7 C2 bindings required (no continue->OK on partial)
    assert "c2_judgment" in C2_CHECKPOINT_KEYS, C2_CHECKPOINT_KEYS
    assert len(C2_CHECKPOINT_KEYS) == 7, C2_CHECKPOINT_KEYS
    assert C2_FIXED_PATH_KEYS["c2_judgment"] == "c2_judgment.json"
    # numerical: 3 hashed bindings + presence-only required files
    assert len(NUMERICAL_BINDINGS) == 3, NUMERICAL_BINDINGS
    assert "numerical_validation.csv" in NUMERICAL_REQUIRED_FILES
    assert "cutlass_sm120_4m.json" in NUMERICAL_REQUIRED_FILES


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

    # Build a fixture satisfying ALL required C2_CHECKPOINT_KEYS (7 keys).
    # Each key maps to a source file with known content + matching full sha256.
    contents = {
        "source_hlo": b"hlo-data",
        "buffer_assignment": b"buf-data",
        "audit": b"audit-data",
        "edge_map": b"edge-data",
        "peak_frontier": b"peak-data",
        "prototype": b"proto-data",
        "c2_judgment": b"judg-data",
    }
    (tmp_path / "source.hlo").write_bytes(contents["source_hlo"])
    (tmp_path / "buffer.txt").write_bytes(contents["buffer_assignment"])
    sub = tmp_path / "c1_buffer_assignment"
    sub.mkdir()
    (sub / "n24.json").write_bytes(contents["audit"])
    (tmp_path / "c1_c2_edge_map.json").write_bytes(contents["edge_map"])
    (tmp_path / "c2_peak_frontier.json").write_bytes(contents["peak_frontier"])
    (tmp_path / "region_prototype.json").write_bytes(contents["prototype"])
    (tmp_path / "c2_judgment.json").write_bytes(contents["c2_judgment"])
    c2j = {
        "n24_d10_default": {
            "artifact_paths": {
                "source_hlo": "results/phase0/source.hlo",
                "buffer_assignment": "results/phase0/buffer.txt",
                "audit": "results/phase0/c1_buffer_assignment/n24.json",
                "edge_map": "results/phase0/c1_c2_edge_map.json",
                "peak_frontier": "results/phase0/c2_peak_frontier.json",
                "prototype": "results/phase0/region_prototype.json",
            }
        }
    }
    ok_ckpt = {
        "artifact_hashes": {
            "source_hlo": hashlib.sha256(contents["source_hlo"]).hexdigest(),
            "buffer_assignment": hashlib.sha256(
                contents["buffer_assignment"]
            ).hexdigest(),
            "allocation_audit": hashlib.sha256(contents["audit"]).hexdigest(),
            "edge_map": hashlib.sha256(contents["edge_map"]).hexdigest(),
            "peak_frontier": hashlib.sha256(contents["peak_frontier"]).hexdigest(),
            "prototype": hashlib.sha256(contents["prototype"]).hexdigest(),
            "c2_judgment": hashlib.sha256(contents["c2_judgment"]).hexdigest(),
        }
    }
    assert _validate_c2_checkpoint(str(tmp_path), c2j, ok_ckpt) == "OK"
    bad_ckpt = {"artifact_hashes": {**ok_ckpt["artifact_hashes"], "edge_map": "0" * 64}}
    assert _validate_c2_checkpoint(str(tmp_path), c2j, bad_ckpt) == "MISMATCH"
    assert _validate_c2_checkpoint(str(tmp_path), c2j, {}) == "UNAVAILABLE"


def test_validate_c2_checkpoint_alias_allocation_audit(tmp_path):
    import hashlib
    from results._phase0.manifest import _validate_c2_checkpoint

    # The allocation_audit checkpoint key aliases to the "audit" artifact_path.
    # Build a fixture satisfying ALL 7 required C2_CHECKPOINT_KEYS so the OK
    # case exercises the alias (allocation_audit -> audit path key).
    contents = {
        "source_hlo": b"hlo",
        "buffer_assignment": b"buf",
        "audit": b"audit-data",
        "edge_map": b"edge",
        "peak_frontier": b"peak",
        "prototype": b"proto",
        "c2_judgment": b"judg",
    }
    (tmp_path / "s.hlo").write_bytes(contents["source_hlo"])
    (tmp_path / "b.txt").write_bytes(contents["buffer_assignment"])
    sub = tmp_path / "c1_buffer_assignment"
    sub.mkdir()
    (sub / "n24_d10_default.json").write_bytes(contents["audit"])
    (tmp_path / "c1_c2_edge_map.json").write_bytes(contents["edge_map"])
    (tmp_path / "c2_peak_frontier.json").write_bytes(contents["peak_frontier"])
    (tmp_path / "region_prototype.json").write_bytes(contents["prototype"])
    (tmp_path / "c2_judgment.json").write_bytes(contents["c2_judgment"])
    c2j = {
        "n24_d10_default": {
            "artifact_paths": {
                "source_hlo": "results/phase0/s.hlo",
                "buffer_assignment": "results/phase0/b.txt",
                "audit": "results/phase0/c1_buffer_assignment/n24_d10_default.json",
                "edge_map": "results/phase0/c1_c2_edge_map.json",
                "peak_frontier": "results/phase0/c2_peak_frontier.json",
                "prototype": "results/phase0/region_prototype.json",
            }
        }
    }
    # checkpoint records under key 'allocation_audit' (alias -> 'audit' path)
    ckpt = {
        "artifact_hashes": {
            "source_hlo": hashlib.sha256(contents["source_hlo"]).hexdigest(),
            "buffer_assignment": hashlib.sha256(
                contents["buffer_assignment"]
            ).hexdigest(),
            "allocation_audit": hashlib.sha256(contents["audit"]).hexdigest(),
            "edge_map": hashlib.sha256(contents["edge_map"]).hexdigest(),
            "peak_frontier": hashlib.sha256(contents["peak_frontier"]).hexdigest(),
            "prototype": hashlib.sha256(contents["prototype"]).hexdigest(),
            "c2_judgment": hashlib.sha256(contents["c2_judgment"]).hexdigest(),
        }
    }
    assert _validate_c2_checkpoint(str(tmp_path), c2j, ckpt) == "OK"


def test_validate_numerical_binding(tmp_path):
    import hashlib
    from results._phase0.manifest import _validate_numerical_binding

    # Build a fixture satisfying ALL required numerical bindings: 3 hashed
    # bindings (edge_map / prototype / contraction_shapes) + 6 presence-only
    # required files (numerical CSV + route source artifacts).
    contents = {
        "edge_map": b"edge-data",
        "prototype": b"proto-data",
        "contraction_shapes": b"shape-data",
    }
    (tmp_path / "c1_c2_edge_map.json").write_bytes(contents["edge_map"])
    (tmp_path / "region_prototype.json").write_bytes(contents["prototype"])
    (tmp_path / "contraction_shapes.csv").write_bytes(contents["contraction_shapes"])
    for f in (
        "numerical_validation.csv",
        "cublaslt_planar_capability.json",
        "cublaslt_full_matrix.csv",
        "cublaslt_grouped_capability.json",
        "cublaslt_grouped.csv",
        "cutlass_sm120_4m.json",
    ):
        (tmp_path / f).write_text("x")
    ok = {
        "case_binding": {
            "edge_map_hash": hashlib.sha256(contents["edge_map"]).hexdigest()[:16],
            "prototype_hash": hashlib.sha256(contents["prototype"]).hexdigest()[:16],
            "contraction_shapes_hash": hashlib.sha256(
                contents["contraction_shapes"]
            ).hexdigest()[:16],
        }
    }
    assert _validate_numerical_binding(str(tmp_path), ok) == "OK"
    bad = {"case_binding": {**ok["case_binding"], "edge_map_hash": "deadbeef" * 2}}
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
    # unavailable -> fail-closed UNKNOWN (binding chain unconfirmable; the prior
    # value may be stale). C1 untouched (no checkpoint binding for C1).
    out3 = _apply_checkpoint_validation(criteria, "UNAVAILABLE", "UNAVAILABLE")
    assert out3["C2"] == "UNKNOWN", out3
    assert out3["NUMERICAL"] == "UNKNOWN", out3
    assert out3["C1"] == "PASS", out3


def test_build_cases_merges_c1_c2(tmp_path):
    from results._phase0.manifest import _build_cases

    c1 = {"n24_d10": {"judgment": {"status": "PASS"}, "n": 24, "depth": 10}}
    c2 = {
        "n24_d10_default": {
            "status": "UNKNOWN",
            "layers": {"C2_CANONICAL": "UNKNOWN"},
            "n": 24,
            "depth": 10,
            "fusion": "default",
        }
    }
    cases = _build_cases(c1, c2, str(tmp_path))
    assert set(cases) == {"n24_d10", "n24_d10_default"}
    assert cases["n24_d10"]["status"] == {"C1": "PASS"}
    assert cases["n24_d10"]["config"]["n"] == 24
    assert cases["n24_d10_default"]["status"]["C2"] == "UNKNOWN"
    assert cases["n24_d10_default"]["config"]["fusion"] == "default"
    assert isinstance(cases["n24_d10"]["artifacts"], list)


def test_collect_inputs_outputs_excludes_manifest(tmp_path):
    from results._phase0.manifest import _collect_inputs_outputs

    (tmp_path / "c1_judgment.json").write_text("x")
    (tmp_path / "gonogo.json").write_text("y")
    (tmp_path / "environment.json").write_text("z")
    (tmp_path / "manifest.json").write_text("self")
    inputs, outputs = _collect_inputs_outputs(str(tmp_path))
    assert "c1_judgment.json" in inputs and len(inputs["c1_judgment.json"]) == 16
    assert outputs["gonogo.json"] and outputs["environment.json"]
    assert "manifest.json" not in outputs and "manifest.json" not in inputs
    # missing input files are simply omitted (not None entries)
    assert "c2_judgment.json" not in inputs


def test_collect_inputs_outputs_hashes_dirs(tmp_path):
    from results._phase0.manifest import _collect_inputs_outputs

    d = tmp_path / "c1_optimized_hlo"
    d.mkdir()
    (d / "n24.hlo").write_bytes(b"x")
    inputs, _ = _collect_inputs_outputs(str(tmp_path))
    assert "c1_optimized_hlo/n24.hlo" in inputs


def test_build_manifest_schema_and_stability(tmp_path):
    import json
    from results._phase0.manifest import build_manifest, SCHEMA_VERSION

    # minimal but complete stage: required artifacts present + a checkpoint match
    (tmp_path / "c1_judgment.json").write_text(
        json.dumps({"n24_d10": {"judgment": {"status": "PASS"}, "n": 24, "depth": 10}})
    )
    (tmp_path / "c1_default_vs_nofusion.csv").write_text("x")
    (tmp_path / "c2_judgment.json").write_text(
        json.dumps(
            {
                "n24_d10_default": {
                    "status": "UNKNOWN",
                    "layers": {"C2_CANONICAL": "UNKNOWN"},
                    "n": 24,
                    "depth": 10,
                    "fusion": "default",
                    "artifact_paths": {
                        "edge_map": "results/phase0/c1_c2_edge_map.json"
                    },
                }
            }
        )
    )
    (tmp_path / "c1_c2_edge_map.json").write_text("e")
    (tmp_path / "c2_checkpoint_manifest.json").write_text(
        json.dumps({"artifact_hashes": {"edge_map": "0" * 64}})
    )  # will MISMATCH (file is 'e')
    (tmp_path / "cublaslt_planar_capability.json").write_text("{}")
    (tmp_path / "cublaslt_full_matrix.csv").write_text("h\n1\n")
    (tmp_path / "cublaslt_grouped_capability.json").write_text("{}")
    (tmp_path / "cutlass_sm120_4m.json").write_text("{}")
    (tmp_path / "region_prototype.json").write_text("{}")
    (tmp_path / "numerical_validation.json").write_text(
        json.dumps({"case_binding": {"edge_map_hash": "0" * 16}})
    )
    (tmp_path / "contraction_shapes.csv").write_text("s")
    (tmp_path / "c2_tileability.csv").write_text("t")
    (tmp_path / "run_context.json").write_text(
        json.dumps(
            {
                "source_commit": "abc123",
                "dirty_worktree": False,
                "dirty_file_count": 0,
                "command_templates": {"gonogo": "python results/_phase0/gonogo.py"},
            }
        )
    )
    (tmp_path / "gonogo.json").write_text(
        json.dumps(
            {
                "schema_version": "gonogo-v2",
                "criteria": {
                    "C1": "PASS",
                    "C2": "UNKNOWN",
                    "C2_REGION_KERNEL": "PASS",
                    "C3_PLANAR_CORE": "PASS",
                    "C3_PLANAR_FULL_MATRIX": "PASS",
                    "C3_GROUPED": "NOT_SUPPORTED",
                    "CUTLASS_SM120_4M": "FEASIBLE_WITH_SM80_FALLBACK",
                    "REGION_PROTOTYPE": "FEASIBLE_WITH_RECOMPUTE",
                    "NUMERICAL": "FAIL",
                },
                "route_verdict": {},
                "phase0_completion": "INCONCLUSIVE",
                "phase1_authorization": "NOT_AUTHORIZED",
            }
        )
    )
    (tmp_path / "gonogo.md").write_text("# md")
    (tmp_path / "environment.json").write_text("{}")

    m = build_manifest(str(tmp_path), generated_at="2026-07-23T00:00:00Z")
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["source_commit"] == "abc123"
    assert m["dirty_worktree"] is False
    assert m["phase0_completion"] == "INCONCLUSIVE"
    assert m["phase1_authorization"] == "NOT_AUTHORIZED"
    # presence + checkpoint validation applied: C2 checkpoint UNAVAILABLE (6 of
    # 7 required bindings missing) -> C2 UNKNOWN (already); NUMERICAL binding
    # UNAVAILABLE (2 of 3 required hashes missing) -> NUMERICAL UNKNOWN (was FAIL)
    assert m["criteria"]["C2"] == "UNKNOWN"
    assert m["criteria"]["NUMERICAL"] == "UNKNOWN"  # unavailable downgraded from FAIL
    assert m["criteria"]["C1"] == "PASS"  # present, no checkpoint
    assert "gonogo.json" in m["outputs"]
    assert "manifest.json" not in m["outputs"]
    assert m["generated_at"] == "2026-07-23T00:00:00Z"
    # stability: same generated_at -> byte-identical JSON
    import json as _j

    m2 = build_manifest(str(tmp_path), generated_at="2026-07-23T00:00:00Z")
    assert _j.dumps(m, sort_keys=True) == _j.dumps(m2, sort_keys=True)
    assert {"n24_d10", "n24_d10_default"} <= set(m["cases"])


def test_main_writes_manifest_v1(tmp_path):
    import json, os, shutil
    from results._phase0 import manifest as M

    src = "results/phase0"
    stage = tmp_path / "phase0"
    stage.mkdir()
    for name in os.listdir(src):
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy(s, stage / name)
    for d in ("c1_optimized_hlo", "c1_buffer_assignment", "c1_xla_dump"):
        sd = os.path.join(src, d)
        if os.path.isdir(sd):
            shutil.copytree(sd, stage / d)
    M.main(stage_dir=str(stage))
    m = json.load(open(stage / "manifest.json"))
    assert m["schema_version"] == "manifest-v1"
    assert m["criteria"]["C1"] == "PASS"
    assert m["phase0_completion"] == "INCONCLUSIVE"
    assert "manifest.json" not in m["outputs"]
    assert m["source_commit"] and m["environment_hash"]


# ---------------------------------------------------------------------------
# Task 0 (SDD plan §3 操作.2): fail-closed RED baseline. The tests below freeze
# the target behavior the manifest gate must adopt after Task 4 wires the
# canonical verdict_schema in. They FAIL on the current implementation by clean
# assertion (not import, not GPU).
# ---------------------------------------------------------------------------


def test_validate_c2_checkpoint_unavailable_when_any_required_binding_missing(
    tmp_path,
):
    """plan §3 操作.2 bullet 5: if ANY required C2 binding key is missing from
    the checkpoint manifest OR the judgment's ``artifact_paths``, the validation
    result must be UNAVAILABLE (the binding chain cannot be confirmed).

    Today ``_validate_c2_checkpoint`` ``continue``s past missing bindings and
    returns ``OK`` as long as >=1 key was checked (manifest.py). That is the
    fail-open surface: 1-of-6 keys present silently passes the whole checkpoint.
    This test freezes the target: every required ``C2_CHECKPOINT_KEYS`` entry
    must be present, else UNAVAILABLE."""
    import hashlib

    from results._phase0.manifest import (
        C2_CHECKPOINT_KEYS,
        _validate_c2_checkpoint,
    )

    # All required keys must be exercised; build a fixture that satisfies ONLY
    # the edge_map key. The other 5 C2_CHECKPOINT_KEYS are absent from both the
    # checkpoint's artifact_hashes and the judgment's artifact_paths.
    content = b"edge-data"
    full = hashlib.sha256(content).hexdigest()
    (tmp_path / "c1_c2_edge_map.json").write_bytes(content)
    c2j = {
        "n24_d10_default": {
            "artifact_paths": {"edge_map": "results/phase0/c1_c2_edge_map.json"}
            # the other 5 path keys (source_hlo/buffer_assignment/audit/
            # peak_frontier/prototype) deliberately absent
        }
    }
    # only 1 of the 6 required C2_CHECKPOINT_KEYS provided
    ckpt = {"artifact_hashes": {"edge_map": full}}

    assert len(C2_CHECKPOINT_KEYS) >= 6, C2_CHECKPOINT_KEYS  # sanity
    result = _validate_c2_checkpoint(str(tmp_path), c2j, ckpt)
    # 1-of-N required bindings present -> the binding chain is UNAVAILABLE, not OK
    assert result == "UNAVAILABLE", result


def test_validate_numerical_binding_unavailable_when_any_required_binding_missing(
    tmp_path,
):
    """plan §3 操作.2 bullet 5 (numerical side): if ANY required numerical case
    binding is missing, validation must be UNAVAILABLE. Today
    ``_validate_numerical_binding`` ``continue``s past missing bindings and
    returns ``OK`` if >=1 was checked."""
    import hashlib

    from results._phase0.manifest import (
        NUMERICAL_BINDINGS,
        _validate_numerical_binding,
    )

    # Provide ONLY the edge_map binding; region_prototype.json +
    # contraction_shapes.csv are absent (so their bindings cannot be validated).
    content = b"edge-data"
    short = hashlib.sha256(content).hexdigest()[:16]
    (tmp_path / "c1_c2_edge_map.json").write_bytes(content)
    numerical_json = {"case_binding": {"edge_map_hash": short}}

    assert len(NUMERICAL_BINDINGS) >= 3, NUMERICAL_BINDINGS  # sanity
    result = _validate_numerical_binding(str(tmp_path), numerical_json)
    # 1-of-3 required bindings present -> UNAVAILABLE
    assert result == "UNAVAILABLE", result


def test_apply_checkpoint_validation_unavailable_downgrades_to_unknown():
    """plan §3 操作.2 bullet 6: UNAVAILABLE must downgrade the dependent
    criterion to UNKNOWN, exactly like MISMATCH. A binding chain that cannot be
    confirmed is fail-closed UNKNOWN (the prior PASS may be stale).

    Today ``_apply_checkpoint_validation`` only downgrades on ``MISMATCH`` and
    explicitly treats ``UNAVAILABLE`` as 'no change' (manifest.py), preserving a
    possibly-stale PASS. This test freezes the target: UNAVAILABLE also forces
    the dependent criterion to UNKNOWN."""
    from results._phase0.manifest import _apply_checkpoint_validation

    criteria = {"C1": "PASS", "C2": "PASS", "NUMERICAL": "PASS"}

    # C2 UNAVAILABLE -> C2 UNKNOWN (not preserved PASS)
    out_c2 = _apply_checkpoint_validation(criteria, "UNAVAILABLE", "OK")
    assert out_c2["C2"] == "UNKNOWN", out_c2
    assert out_c2["C1"] == "PASS"  # untouched

    # NUMERICAL UNAVAILABLE -> NUMERICAL UNKNOWN
    out_num = _apply_checkpoint_validation(criteria, "OK", "UNAVAILABLE")
    assert out_num["NUMERICAL"] == "UNKNOWN", out_num

    # Both UNAVAILABLE -> both UNKNOWN
    out_both = _apply_checkpoint_validation(criteria, "UNAVAILABLE", "UNAVAILABLE")
    assert out_both["C2"] == "UNKNOWN" and out_both["NUMERICAL"] == "UNKNOWN", out_both


def test_build_manifest_recomputes_routes_after_checkpoint_downgrade(tmp_path):
    """plan §3 操作.2 bullet 7: after checkpoint validation downgrades a
    criterion, route_verdict / phase0_completion / phase1_authorization must be
    RECOMPUTED from the validated criteria, not propagated unchanged from
    gonogo.json.

    Scenario: the staged gonogo.json claims C2=PASS, region_fused VIABLE,
    completion COMPLETE, authorization GO_TO_PHASE1. The C2 checkpoint manifest
    records a hash that MISMATCHES the on-disk source file, so the manifest
    downgrades C2 PASS -> UNKNOWN. With C2 UNKNOWN, gonogo's truth table yields
    region_fused UNKNOWN (capability NOT_OK or UNDETERMINED), completion
    INCONCLUSIVE, authorization NOT_AUTHORIZED. Today ``build_manifest`` keeps
    the stale gonogo.json values verbatim, which is the fail-open bug."""
    import json

    from results._phase0.manifest import build_manifest

    # Minimal stage with every required artifact present (so _presence_check
    # does NOT independently force NOT_RUN) and a C2 checkpoint that MISMATCHES.
    (tmp_path / "c1_judgment.json").write_text(
        json.dumps({"n24_d10": {"judgment": {"status": "PASS"}, "n": 24, "depth": 10}})
    )
    (tmp_path / "c1_default_vs_nofusion.csv").write_text("x")
    (tmp_path / "c2_judgment.json").write_text(
        json.dumps(
            {
                "n24_d10_default": {
                    "status": "PASS",  # claimed PASS; checkpoint will contradict
                    "layers": {
                        "C2_CANONICAL": "PASS",
                        "C2_REGION_KERNEL_FEASIBILITY": "PASS",
                    },
                    "n": 24,
                    "depth": 10,
                    "fusion": "default",
                    "artifact_paths": {
                        "edge_map": "results/phase0/c1_c2_edge_map.json"
                    },
                }
            }
        )
    )
    # on-disk edge-map content
    (tmp_path / "c1_c2_edge_map.json").write_text("real-edge-data")
    # checkpoint records only edge_map (1 of 7 required keys); the other 6 are
    # missing -> UNAVAILABLE (not MISMATCH). Both UNAVAILABLE and MISMATCH
    # downgrade C2 to UNKNOWN.
    (tmp_path / "c2_checkpoint_manifest.json").write_text(
        json.dumps({"artifact_hashes": {"edge_map": "0" * 64}})
    )
    (tmp_path / "cublaslt_planar_capability.json").write_text(
        json.dumps({"capability": {"status": "SUPPORTED"}})
    )
    (tmp_path / "cublaslt_full_matrix.csv").write_text(
        "M,N,K,status\n1024,1024,1024,ok\n"
    )
    (tmp_path / "cublaslt_grouped_capability.json").write_text(
        json.dumps({"capability": {"status": "SUPPORTED"}})
    )
    (tmp_path / "cutlass_sm120_4m.json").write_text("{}")
    (tmp_path / "region_prototype.json").write_text("{}")
    (tmp_path / "numerical_validation.json").write_text(
        json.dumps({"case_binding": {"edge_map_hash": "0" * 16}})
    )
    (tmp_path / "contraction_shapes.csv").write_text("s")
    (tmp_path / "c2_tileability.csv").write_text("t")
    (tmp_path / "run_context.json").write_text(
        json.dumps(
            {
                "source_commit": "abc123",
                "dirty_worktree": False,
                "dirty_file_count": 0,
                "command_templates": {"gonogo": "python results/_phase0/gonogo.py"},
            }
        )
    )
    # gonogo.json: claims an OVER-OPTIMISTIC GO (C2 PASS, region VIABLE, COMPLETE)
    # that the manifest's checkpoint validation must overturn.
    (tmp_path / "gonogo.json").write_text(
        json.dumps(
            {
                "schema_version": "gonogo-v2",
                "criteria": {
                    "C1": "PASS",
                    "C2": "PASS",
                    "C2_REGION_KERNEL": "PASS",
                    "C3_PLANAR_CORE": "PASS",
                    "C3_PLANAR_FULL_MATRIX": "PASS",
                    "C3_GROUPED": "PASS",
                    "CUTLASS_SM120_4M": "PASS",
                    "REGION_PROTOTYPE": "PASS",
                    "NUMERICAL": "PASS",
                },
                "route_verdict": {
                    "region_fused": {
                        "status": "VIABLE",
                        "capability": "OK",
                        "numerical": "OK",
                    }
                },
                "phase0_completion": "COMPLETE",
                "phase1_authorization": "GO_TO_PHASE1",
            }
        )
    )
    (tmp_path / "gonogo.md").write_text("# md")
    (tmp_path / "environment.json").write_text("{}")

    m = build_manifest(str(tmp_path), generated_at="2026-07-23T00:00:00Z")
    # The checkpoint UNAVAILABLE (6 of 7 keys missing) must downgrade C2 PASS ->
    # UNKNOWN (bullet 6). Both UNAVAILABLE and MISMATCH force UNKNOWN.
    assert m["criteria"]["C2"] == "UNKNOWN", m["criteria"]
    # Bullet 7: route / completion / authorization recomputed from the validated
    # criteria. C2 UNKNOWN -> completion INCONCLUSIVE and authorization
    # NOT_AUTHORIZED (a GO claim that rested on the stale C2 PASS cannot survive
    # the downgrade). CUTLASS_SM80_FALLBACK_CAPABILITY is also absent from the
    # staged gonogo criteria -> undetermined -> INCONCLUSIVE regardless.
    assert m["phase0_completion"] == "INCONCLUSIVE", m["phase0_completion"]
    assert m["phase1_authorization"] == "NOT_AUTHORIZED", m["phase1_authorization"]
    # Self-consistency invariant: no route may be VIABLE when its dependent
    # criteria are downgraded. C2 was downgraded to UNKNOWN; the gonogo claimed
    # region_fused VIABLE. After recompute, no route should be VIABLE (all
    # depend on at least one undetermined criterion or have UNDETERMINED num).
    assert all(rv["status"] != "VIABLE" for rv in m["route_verdict"].values()), m[
        "route_verdict"
    ]
    # No criterion UNKNOWN + completion COMPLETE contradiction.
    assert not (
        m["criteria"]["C2"] == "UNKNOWN" and m["phase0_completion"] == "COMPLETE"
    )


def test_build_manifest_self_consistent_no_unknown_plus_viable(tmp_path):
    """plan §9 验收: manifest 内部不可能出现 criterion UNKNOWN + dependent route
    VIABLE, nor downgraded-criteria + completion COMPLETE. This test stages a
    gonogo claiming all-PASS + all-VIABLE + COMPLETE + GO_TO_PHASE1, then breaks
    BOTH the C2 and NUMERICAL binding chains. The manifest must recompute to
    INCONCLUSIVE / NOT_AUTHORIZED with no VIABLE route surviving."""
    import json

    from results._phase0.manifest import build_manifest

    (tmp_path / "c1_judgment.json").write_text(
        json.dumps({"n24_d10": {"judgment": {"status": "PASS"}, "n": 24, "depth": 10}})
    )
    (tmp_path / "c1_default_vs_nofusion.csv").write_text("x")
    # c2_judgment with all 6 artifact_paths (the 7th, c2_judgment, is a fixed
    # path -- c2_judgment.json itself).
    (tmp_path / "c2_judgment.json").write_text(
        json.dumps(
            {
                "n24_d10_default": {
                    "status": "PASS",
                    "layers": {
                        "C2_CANONICAL": "PASS",
                        "C2_REGION_KERNEL_FEASIBILITY": "PASS",
                    },
                    "n": 24,
                    "depth": 10,
                    "fusion": "default",
                    "artifact_paths": {
                        "edge_map": "results/phase0/c1_c2_edge_map.json",
                        "peak_frontier": "results/phase0/c2_peak_frontier.json",
                        "prototype": "results/phase0/region_prototype.json",
                        "audit": "results/phase0/c1_buffer_assignment/n24.json",
                        "source_hlo": "results/phase0/source.hlo",
                        "buffer_assignment": "results/phase0/buffer.txt",
                    },
                }
            }
        )
    )
    import hashlib

    for f, c in [
        ("c1_c2_edge_map.json", b"edge"),
        ("c2_peak_frontier.json", b"peak"),
        ("region_prototype.json", b"proto"),
        ("source.hlo", b"hlo"),
        ("buffer.txt", b"buf"),
    ]:
        (tmp_path / f).write_bytes(c)
    sub = tmp_path / "c1_buffer_assignment"
    sub.mkdir()
    (sub / "n24.json").write_bytes(b"audit")
    # checkpoint: all 7 hashes present, but edge_map hash is WRONG -> MISMATCH
    (tmp_path / "c2_checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "artifact_hashes": {
                    "source_hlo": hashlib.sha256(b"hlo").hexdigest(),
                    "buffer_assignment": hashlib.sha256(b"buf").hexdigest(),
                    "allocation_audit": hashlib.sha256(b"audit").hexdigest(),
                    "edge_map": "0" * 64,  # MISMATCH (real content is "edge")
                    "peak_frontier": hashlib.sha256(b"peak").hexdigest(),
                    "prototype": hashlib.sha256(b"proto").hexdigest(),
                    "c2_judgment": hashlib.sha256(
                        (tmp_path / "c2_judgment.json").read_bytes()
                    ).hexdigest(),
                }
            }
        )
    )
    for f in (
        "cublaslt_planar_capability.json",
        "cublaslt_grouped_capability.json",
        "cutlass_sm120_4m.json",
        "cublaslt_grouped.csv",
        "numerical_validation.csv",
    ):
        (tmp_path / f).write_text("x")
    (tmp_path / "cublaslt_full_matrix.csv").write_text("h\n1\n")
    # numerical binding: all 3 hashes present but edge_map_hash MISMATCHES -> MISMATCH
    (tmp_path / "contraction_shapes.csv").write_bytes(b"shapes")
    (tmp_path / "numerical_validation.json").write_text(
        json.dumps(
            {
                "case_binding": {
                    "edge_map_hash": "0" * 16,  # MISMATCH
                    "prototype_hash": hashlib.sha256(b"proto").hexdigest()[:16],
                    "contraction_shapes_hash": hashlib.sha256(b"shapes").hexdigest()[
                        :16
                    ],
                },
                "per_route": [
                    {"route": "planar", "criterion": "PASS"},
                    {"route": "grouped", "criterion": "PASS"},
                    {"route": "region_fused", "criterion": "PASS"},
                    {"route": "cutlass_4m_single", "criterion": "PASS"},
                ],
            }
        )
    )
    (tmp_path / "run_context.json").write_text(
        json.dumps(
            {"source_commit": "abc", "dirty_worktree": False, "dirty_file_count": 0}
        )
    )
    (tmp_path / "gonogo.json").write_text(
        json.dumps(
            {
                "schema_version": "gonogo-v2",
                "criteria": {
                    "C1": "PASS",
                    "C2": "PASS",
                    "C2_REGION_KERNEL": "PASS",
                    "C3_PLANAR_CORE": "PASS",
                    "C3_PLANAR_FULL_MATRIX": "PASS",
                    "C3_GROUPED": "PASS",
                    "CUTLASS_SM120_4M": "PASS",
                    "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
                    "REGION_PROTOTYPE": "PASS",
                    "NUMERICAL": "PASS",
                },
                "route_verdict": {
                    r: {"status": "VIABLE", "capability": "OK", "numerical": "OK"}
                    for r in ("planar", "grouped", "region_fused", "cutlass_4m_single")
                },
                "phase0_completion": "COMPLETE",
                "phase1_authorization": "GO_TO_PHASE1",
            }
        )
    )
    (tmp_path / "gonogo.md").write_text("# md")
    (tmp_path / "environment.json").write_text("{}")

    m = build_manifest(str(tmp_path), generated_at="2026-07-23T00:00:00Z")
    # C2 MISMATCH + NUMERICAL MISMATCH -> both UNKNOWN
    assert m["criteria"]["C2"] == "UNKNOWN", m["criteria"]
    assert m["criteria"]["NUMERICAL"] == "UNKNOWN", m["criteria"]
    # Self-consistency: no UNKNOWN + VIABLE, no downgraded + COMPLETE
    assert m["phase0_completion"] == "INCONCLUSIVE", m["phase0_completion"]
    assert m["phase1_authorization"] == "NOT_AUTHORIZED", m["phase1_authorization"]
    assert all(rv["status"] != "VIABLE" for rv in m["route_verdict"].values()), m[
        "route_verdict"
    ]
    # Numerical binding MISMATCH -> per-route numerical not trusted -> all
    # routes that depend on numerical get UNKNOWN (not VIABLE even though the
    # staged per_route claims all PASS).
    assert all(
        rv["numerical"] == "UNDETERMINED" for rv in m["route_verdict"].values()
    ), m["route_verdict"]


def test_build_manifest_c2_checkpoint_cascade_closes_region_fused_gap(tmp_path):
    """F1: a broken C2 checkpoint binding must cascade to ALL C2-family criteria
    (not just the top-level "C2"), because the C2 checkpoint validates the SHARED
    C2 artifact chain that every C2 sub-criterion rests on. Without the cascade, a
    broken C2 chain + numerical OK could leave region_fused VIABLE while C2=UNKNOWN
    -- a fail-open on the spine.

    This test breaks ONLY the C2 checkpoint binding (MISMATCH) while keeping the
    numerical binding OK with per-route region_fused=PASS. gonogo native criteria
    claim C2_REGION_KERNEL=PASS + REGION_PROTOTYPE=PASS. Before the F1 fix only "C2"
    was downgraded, so region_fused stayed VIABLE (capability OK from
    C2_REGION_KERNEL + REGION_PROTOTYPE, numerical OK from the trusted per-route
    PASS). After the fix C2_REGION_KERNEL also downgrades to UNKNOWN -> region_fused
    capability UNDETERMINED -> region_fused UNKNOWN (not VIABLE). The existing
    self-consistency test breaks BOTH bindings (so its no-VIABLE assertion holds
    for the wrong reason -- the numerical break alone sinks every route); this test
    isolates the C2-only break to prove the cascade is what closes the gap."""
    import hashlib, json

    from results._phase0.manifest import build_manifest

    (tmp_path / "c1_judgment.json").write_text(
        json.dumps({"n24_d10": {"judgment": {"status": "PASS"}, "n": 24, "depth": 10}})
    )
    (tmp_path / "c1_default_vs_nofusion.csv").write_text("x")
    # c2_judgment with all 6 artifact_paths (the 7th, c2_judgment, is the fixed
    # path c2_judgment.json itself).
    (tmp_path / "c2_judgment.json").write_text(
        json.dumps(
            {
                "n24_d10_default": {
                    "status": "PASS",
                    "layers": {
                        "C2_CANONICAL": "PASS",
                        "C2_REGION_KERNEL_FEASIBILITY": "PASS",
                    },
                    "n": 24,
                    "depth": 10,
                    "fusion": "default",
                    "artifact_paths": {
                        "edge_map": "results/phase0/c1_c2_edge_map.json",
                        "peak_frontier": "results/phase0/c2_peak_frontier.json",
                        "prototype": "results/phase0/region_prototype.json",
                        "audit": "results/phase0/c1_buffer_assignment/n24.json",
                        "source_hlo": "results/phase0/source.hlo",
                        "buffer_assignment": "results/phase0/buffer.txt",
                    },
                }
            }
        )
    )
    # on-disk C2 binding source files
    for f, c in [
        ("c1_c2_edge_map.json", b"edge"),
        ("c2_peak_frontier.json", b"peak"),
        ("region_prototype.json", b"proto"),
        ("source.hlo", b"hlo"),
        ("buffer.txt", b"buf"),
    ]:
        (tmp_path / f).write_bytes(c)
    sub = tmp_path / "c1_buffer_assignment"
    sub.mkdir()
    (sub / "n24.json").write_bytes(b"audit")
    # C2 checkpoint: all 7 hashes present but edge_map hash is WRONG -> MISMATCH
    # (only the C2 binding chain is broken; numerical is OK below).
    (tmp_path / "c2_checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "artifact_hashes": {
                    "source_hlo": hashlib.sha256(b"hlo").hexdigest(),
                    "buffer_assignment": hashlib.sha256(b"buf").hexdigest(),
                    "allocation_audit": hashlib.sha256(b"audit").hexdigest(),
                    "edge_map": "0" * 64,  # MISMATCH (real content is "edge")
                    "peak_frontier": hashlib.sha256(b"peak").hexdigest(),
                    "prototype": hashlib.sha256(b"proto").hexdigest(),
                    "c2_judgment": hashlib.sha256(
                        (tmp_path / "c2_judgment.json").read_bytes()
                    ).hexdigest(),
                }
            }
        )
    )
    for f in (
        "cublaslt_planar_capability.json",
        "cublaslt_grouped_capability.json",
        "cutlass_sm120_4m.json",
        "cublaslt_grouped.csv",
        "numerical_validation.csv",
    ):
        (tmp_path / f).write_text("x")
    (tmp_path / "cublaslt_full_matrix.csv").write_text("h\n1\n")
    # numerical binding: all 3 hashes present AND MATCHING -> OK (only C2 broken).
    (tmp_path / "contraction_shapes.csv").write_bytes(b"shapes")
    (tmp_path / "numerical_validation.json").write_text(
        json.dumps(
            {
                "case_binding": {
                    "edge_map_hash": hashlib.sha256(b"edge").hexdigest()[:16],
                    "prototype_hash": hashlib.sha256(b"proto").hexdigest()[:16],
                    "contraction_shapes_hash": hashlib.sha256(b"shapes").hexdigest()[
                        :16
                    ],
                },
                "per_route": [
                    {"route": "region_fused", "criterion": "PASS"},
                ],
            }
        )
    )
    (tmp_path / "run_context.json").write_text(
        json.dumps(
            {"source_commit": "abc", "dirty_worktree": False, "dirty_file_count": 0}
        )
    )
    # gonogo claims C2_REGION_KERNEL_FEASIBILITY=PASS + REGION_PROTOTYPE=PASS +
    # NUMERICAL=PASS and region_fused VIABLE. The C2 checkpoint MISMATCH must
    # cascade to C2_REGION_KERNEL_FEASIBILITY, sinking region_fused.
    (tmp_path / "gonogo.json").write_text(
        json.dumps(
            {
                "schema_version": "gonogo-v2",
                "criteria": {
                    "C1": "PASS",
                    "C2": "PASS",
                    "C2_REGION_KERNEL_FEASIBILITY": "PASS",
                    "C3_PLANAR_CORE": "PASS",
                    "C3_PLANAR_FULL_MATRIX": "PASS",
                    "C3_GROUPED": "PASS",
                    "CUTLASS_SM120_4M": "PASS",
                    "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
                    "REGION_PROTOTYPE": "PASS",
                    "NUMERICAL": "PASS",
                },
                "route_verdict": {
                    "region_fused": {
                        "status": "VIABLE",
                        "capability": "OK",
                        "numerical": "OK",
                    }
                },
                "phase0_completion": "COMPLETE",
                "phase1_authorization": "GO_TO_PHASE1",
            }
        )
    )
    (tmp_path / "gonogo.md").write_text("# md")
    (tmp_path / "environment.json").write_text("{}")

    m = build_manifest(str(tmp_path), generated_at="2026-07-23T00:00:00Z")
    # F1 cascade: C2 checkpoint MISMATCH downgrades the WHOLE C2 family, not just
    # "C2". C2_REGION_KERNEL_FEASIBILITY was PASS in gonogo; it must now be UNKNOWN.
    assert m["criteria"]["C2"] == "UNKNOWN", m["criteria"]
    assert m["criteria"]["C2_REGION_KERNEL_FEASIBILITY"] == "UNKNOWN", m["criteria"]
    # Numerical binding is OK, so NUMERICAL is NOT downgraded (stays PASS) -- the
    # cascade is C2-only, proving the gap is closed by the C2 cascade and not by
    # an incidental numerical break.
    assert m["criteria"]["NUMERICAL"] == "PASS", m["criteria"]
    # The gap (F1): with C2_REGION_KERNEL_FEASIBILITY downgraded, region_fused
    # capability is UNDETERMINED, so region_fused is UNKNOWN -- NOT VIABLE. Before
    # the F1 fix C2_REGION_KERNEL_FEASIBILITY stayed PASS and region_fused
    # (capability OK + numerical OK from the trusted per-route PASS) was VIABLE
    # despite C2=UNKNOWN: the fail-open this test closes.
    assert m["route_verdict"]["region_fused"]["status"] == "UNKNOWN", m["route_verdict"]
    assert m["route_verdict"]["region_fused"]["status"] != "VIABLE", m["route_verdict"]
    # region_fused capability is UNDETERMINED (C2_REGION_KERNEL_FEASIBILITY
    # downgraded); numerical is OK (binding OK + per_route region_fused PASS).
    assert m["route_verdict"]["region_fused"]["capability"] == "UNDETERMINED", m[
        "route_verdict"
    ]
    assert m["route_verdict"]["region_fused"]["numerical"] == "OK", m["route_verdict"]
    # Self-consistency invariant: no UNKNOWN criterion + dependent route VIABLE.
    assert all(rv["status"] != "VIABLE" for rv in m["route_verdict"].values()), m[
        "route_verdict"
    ]


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
