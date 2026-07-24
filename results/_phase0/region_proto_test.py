"""Task 4: real P->T->E two-stage GEMM region prototype (final-remediation Task 4).

Replaces the rejected GEMM->norm artifact (final-review section 3.2). Validates that the
exact layout transform (Task 2) composes into a correct two-stage E = D @ transform(A@B),
and that a fused producer-recompute kernel computes the same E WITHOUT materializing the
full P or T buffers. cupy.RawKernel / nvrtc sm_120. Run:
  pytest results/_phase0/region_proto_test.py -v
"""

import cupy as cp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _free_gpu_pool():
    """Reclaim cupy's memory pool around each test so the full-shape materialized reference
    (P+T+E ~1.7 GB peak) does not OOM when run after other GPU tests in the same process
    (final-review section 12.2: GPU tests are not assumed to coexist in one process)."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device(0).synchronize()
    yield
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device(0).synchronize()


# A small 8-D contract (mirrors the real transform's reshape->transpose->reshape structure)
# for fused-kernel correctness: P[2,16] -> [1,1,1,2,2,2,2,2] -> transpose -> [4,8] = T.
SMALL_STEPS = [
    {
        "op": "bitcast",
        "shape_in": [2, 16],
        "layout_in": [1, 0],
        "shape_out": [1, 1, 1, 2, 2, 2, 2, 2],
        "layout_out": [7, 6, 5, 4, 3, 2, 1, 0],
    },
    {
        "op": "transpose",
        "dimensions": [2, 1, 0, 4, 6, 3, 5, 7],
        "shape_in": [1, 1, 1, 2, 2, 2, 2, 2],
        "layout_in": [7, 6, 5, 4, 3, 2, 1, 0],
        "shape_out": [1, 1, 1, 2, 2, 2, 2, 2],
        "layout_out": [7, 6, 5, 4, 3, 2, 1, 0],
    },
    {
        "op": "bitcast",
        "shape_in": [1, 1, 1, 2, 2, 2, 2, 2],
        "layout_in": [7, 6, 5, 4, 3, 2, 1, 0],
        "shape_out": [4, 8],
        "layout_out": [1, 0],
    },
]
SMALL_SHAPES = {"PM": 2, "PN": 16, "K1": 4, "TM": 4, "TN": 8}


def test_apply_transform_matches_task2_permutation():
    """Vectorized reshape/transpose transform == Task 2's layout-aware permutation (row-major)."""
    from results._phase0.c1_to_c2_map import _linear_permutation
    from results._phase0.region_proto import apply_transform_steps

    steps = SMALL_STEPS
    n = int(np.prod(steps[0]["shape_in"]))
    P = cp.arange(1, n + 1, dtype=cp.float64).reshape(steps[0]["shape_in"])
    T_vec = cp.asnumpy(apply_transform_steps(P, steps)).ravel()
    fwd, _inv = _linear_permutation(steps)
    T_ref = cp.asnumpy(cp.asarray(P).ravel()[fwd])
    assert np.array_equal(T_vec, T_ref), (T_vec.tolist(), T_ref.tolist())


def test_materialized_reference_two_stage_real_shape():
    """Materialized E = D @ transform(A@B) on the real anchor shape: right shape, finite."""
    from results._phase0.region_proto import (
        load_region_contract,
        materialized_reference,
    )

    contract = load_region_contract()
    rng = np.random.default_rng(0)
    A = (
        rng.standard_normal((4096, 1024)) + 1j * rng.standard_normal((4096, 1024))
    ).astype(np.complex64)
    B = (
        rng.standard_normal((1024, 16384)) + 1j * rng.standard_normal((1024, 16384))
    ).astype(np.complex64)
    D = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))).astype(
        np.complex64
    )
    E, P, T = materialized_reference(
        cp.asarray(A), cp.asarray(B), cp.asarray(D), contract["steps"]
    )
    assert E.shape == (64, 1048576), E.shape
    assert T.shape == (64, 1048576) and P.shape == (4096, 16384), (P.shape, T.shape)
    assert bool(cp.all(cp.isfinite(E))), "E has NaN/Inf"
    # E == D @ T exactly re-multiply (independent of the transform path)
    assert bool(cp.allclose(E, cp.asarray(D) @ T, rtol=1e-4, atol=1e-4))


def test_fused_matches_materialized_small_shape():
    """The fused producer-recompute kernel computes E WITHOUT full P/T and matches the
    materialized reference elementwise on the small contract."""
    from results._phase0.region_proto import fused_reference, materialized_reference

    rng = np.random.default_rng(1)
    s = SMALL_SHAPES
    A = (
        rng.standard_normal((s["PM"], s["K1"]))
        + 1j * rng.standard_normal((s["PM"], s["K1"]))
    ).astype(np.complex64)
    B = (
        rng.standard_normal((s["K1"], s["PN"]))
        + 1j * rng.standard_normal((s["K1"], s["PN"]))
    ).astype(np.complex64)
    D = (
        rng.standard_normal((s["TM"], s["TM"]))
        + 1j * rng.standard_normal((s["TM"], s["TM"]))
    ).astype(np.complex64)
    E_mat, _P, _T = materialized_reference(
        cp.asarray(A), cp.asarray(B), cp.asarray(D), SMALL_STEPS
    )
    E_fused = fused_reference(
        cp.asarray(A), cp.asarray(B), cp.asarray(D), SMALL_STEPS, s
    )
    rel_l2 = float(cp.linalg.norm(E_fused - E_mat) / max(1.0, cp.linalg.norm(E_mat)))
    max_rel = float(cp.max(cp.abs(E_fused - E_mat)) / max(1.0, cp.max(cp.abs(E_mat))))
    assert (
        rel_l2 < 1e-5
    ), rel_l2  # c64 fused == materialized (no dtype change -> ~exact)
    assert max_rel < 1e-5, max_rel
    assert E_fused.shape == E_mat.shape


def test_run_verdict_and_no_full_PT(tmp_path):
    from results._phase0.region_proto import run
    from results._phase0.verdict_schema import CRITERION_TOKENS

    out = run(out_dir=str(tmp_path))
    # Task 2a (plan §5 2.1): full-anchor fused run NOT executed -> canonical
    # verdict UNKNOWN (a canonical criterion token), NOT the artifact-native
    # FEASIBLE_WITH_RECOMPUTE detail token that used to live in this field.
    assert out["verdict"] in CRITERION_TOKENS, out["verdict"]
    assert out["verdict"] == "UNKNOWN", out  # full-anchor run pending (Task 2b)
    assert out["fused_full_anchor_run"] is False, out
    assert out["no_full_P_materialized"] is True, out
    assert out["no_full_T_materialized"] is True, out
    assert "relative_l2" in out and out["n_seeds"] >= 1, out
    assert out["relative_l2"] < 1e-4, out  # fused == materialized on the small contract
    # resources reported when nvrtc --res-usage retrieval succeeds; when it does not
    # the fields are None (UNKNOWN, plan §5 2.1 -- the deleted behavior was a 40
    # fallback). On the dev GPU retrieval typically succeeds.
    if out["registers_per_thread"] is not None:
        assert out["registers_per_thread"] > 0, out
        assert out["occupancy_pct"] > 0, out
    # Task 2a: raw allocation delta is reclassified MODEL_ONLY (analytical upper
    # bound), not a runtime peak gain.
    assert out["peak_evidence_class"] == "MODEL_ONLY", out
    assert "analytical_or_allocation_upper_bound_bytes" in out, out
    assert "peak_saved_bytes" not in out, out  # the misleading name is gone
    # Task 2 (plan §5 2.1/2.3): legacy raw-allocation fields renamed to analytical
    # diagnostic names; old names must NOT appear in the producer output.
    assert "analytical_materialized_buffer_floor_bytes" in out, out
    assert "analytical_fused_buffer_floor_bytes" in out, out
    assert "materialized_peak_bytes" not in out, out  # renamed
    assert "fused_peak_bytes" not in out, out  # renamed
    # Task 2 (plan §5 2.1): MEASURED runtime allocator peak schema is predefined
    # as None (GPU Task 2b fills these from the full-anchor fused run). The c2
    # gate reads ONLY these fields for a canonical region peak gain.
    assert out["materialized_runtime_allocator_peak_bytes"] is None, out
    assert out["fused_runtime_allocator_peak_bytes"] is None, out
    assert out["runtime_peak_gain_bytes"] is None, out
    assert out["runtime_peak_measurement_method"] is None, out
    assert out["runtime_peak_scope"] is None, out
    assert out["runtime_peak_sample_count"] is None, out


def test_run_artifacts(tmp_path):
    """Task 12 regression guard: run() writes its four artifacts under out_dir
    (tmp) and must NOT clobber the committed canonical results/phase0 files."""
    import hashlib
    import os

    from results._phase0.region_proto import run

    canonical = "results/phase0/region_prototype.json"
    before = (
        hashlib.sha256(open(canonical, "rb").read()).hexdigest()
        if os.path.exists(canonical)
        else ""
    )
    run(out_dir=str(tmp_path))
    for name in [
        "region_prototype.json",
        "region_prototype_accuracy.csv",
        "region_prototype_memory.csv",
        "region_prototype_bench.csv",
    ]:
        assert (tmp_path / name).exists(), name
    # canonical region_prototype.json must be byte-identical before/after (no clobber)
    after = (
        hashlib.sha256(open(canonical, "rb").read()).hexdigest()
        if os.path.exists(canonical)
        else ""
    )
    assert before == after, "run() clobbered canonical region_prototype.json"


# ---------------------------------------------------------------------------
# Task 0 (SDD plan §3 操作.2): fail-closed RED baseline, producer-side.
# GPU-free: inspects the committed canonical region_prototype.json so the test
# never has to compile or run a GPU kernel. It freezes the producer contract
# expected by Tasks 2a/3a: the verdict field the prototype emits must be a
# CANONICAL criterion token (PASS/FAIL/UNKNOWN/NOT_RUN/NOT_SUPPORTED), and when
# fused_full_anchor_run=False the canonical criterion is UNKNOWN (not the
# artifact-native 'FEASIBLE_WITH_RECOMPUTE' detail token that the current
# producer writes into the verdict field).
# ---------------------------------------------------------------------------


def test_region_prototype_verdict_field_is_canonical_when_full_anchor_not_run():
    """plan §3 操作.2 bullets 1+2 (producer side): the canonical
    region_prototype.json verdict must carry a canonical criterion token. Today
    the producer (region_proto.run) writes ``FEASIBLE_WITH_RECOMPUTE`` into
    ``verdict`` even though ``fused_full_anchor_run=false`` — that detail token
    belongs in detail_status, and a canonical criterion field carrying it is the
    fail-open surface that downstream gates (c2._region_layer) wrongly promote
    to PASS.

    The canonical criterion value when the full-anchor fused run was NOT
    executed is UNKNOWN (the leverage was not measured at the full anchor). This
    test reads the committed artifact and asserts the verdict field is in the
    canonical criterion set; it FAILS today because the field still holds the
    FEASIBLE_WITH_RECOMPUTE detail token."""
    import json
    import os

    from results._phase0.verdict_schema import CRITERION_TOKENS, normalize_criterion

    path = "results/phase0/region_prototype.json"
    assert os.path.exists(path), "canonical region_prototype.json missing"
    with open(path) as fh:
        proto = json.load(fh)

    # the committed canonical artifact records the full-anchor run as NOT done;
    # fail loudly if that precondition ever flips (no silent-skip green).
    assert proto["fused_full_anchor_run"] is False, proto
    # The verdict field must be a canonical criterion token. The canonical
    # value is UNKNOWN (full-anchor leverage unmeasured); the detail token
    # 'FEASIBLE_WITH_RECOMPUTE' must not appear in this canonical field.
    verdict = proto.get("verdict")
    assert verdict in CRITERION_TOKENS, (
        f"region_prototype.verdict={verdict!r} is not a canonical criterion "
        f"token; fused_full_anchor_run=False must yield criterion UNKNOWN "
        f"(normalize_criterion maps {verdict!r} -> {normalize_criterion(verdict)!r})"
    )


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
