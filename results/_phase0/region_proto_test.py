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


def test_run_verdict_and_no_full_PT():
    from results._phase0.region_proto import run

    out = run()
    assert out["verdict"] in {
        "TILE_FUSION_FEASIBLE",
        "FEASIBLE_WITH_RECOMPUTE",
        "NOT_FEASIBLE",
        "BLOCKED",
    }, out["verdict"]
    assert out["no_full_P_materialized"] is True, out
    assert out["no_full_T_materialized"] is True, out
    assert "relative_l2" in out and out["n_seeds"] >= 1, out
    assert out["relative_l2"] < 1e-4, out  # fused == materialized on the small contract
    # resources reported (real kernel compiled for sm_120)
    assert (
        out["registers_per_thread"] is not None and out["registers_per_thread"] > 0
    ), out
    assert out["occupancy_pct"] > 0, out


def test_run_artifacts():
    import os

    from results._phase0.region_proto import run

    run()
    for p in [
        "results/phase0/region_prototype.json",
        "results/phase0/region_prototype_accuracy.csv",
        "results/phase0/region_prototype_memory.csv",
        "results/phase0/region_prototype_bench.csv",
    ]:
        assert os.path.exists(p), p


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
