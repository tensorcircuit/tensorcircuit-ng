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
    """Schema regression guard for run() output after G2 (MEASURED full-anchor run).

    The full-anchor fused run is now executed (G2): fused_full_anchor_run=True,
    peak_evidence_class=MEASURED, and the canonical verdict is derived from real
    measurements (PASS/FAIL/UNKNOWN), not hardcoded UNKNOWN. The small-contract
    correctness, no-full-P/T, and analytical-diagnostic-field guards from Task 2a
    are preserved."""
    from results._phase0.region_proto import run
    from results._phase0.verdict_schema import CRITERION_TOKENS

    out = run(out_dir=str(tmp_path))
    # G2: full-anchor fused run IS executed -> canonical verdict is a real
    # PASS/FAIL/UNKNOWN derived from measured correctness+peak+resources.
    assert out["verdict"] in CRITERION_TOKENS, out["verdict"]
    assert out["verdict"] in ("PASS", "FAIL", "UNKNOWN"), out
    assert out["fused_full_anchor_run"] is True, out
    assert out["no_full_P_materialized"] is True, out
    assert out["no_full_T_materialized"] is True, out
    assert "relative_l2" in out and out["n_seeds"] >= 1, out
    assert out["relative_l2"] < 1e-4, out  # fused == materialized on the small contract
    # G2: resources are now MEASURED (registers via RawKernel.num_regs fallback
    # when nvrtc --res-usage is unavailable, e.g. cupy 14.x / nvrtc 12.8 sm_120).
    assert (
        out["registers_per_thread"] is not None and out["registers_per_thread"] > 0
    ), out
    assert out["occupancy_pct"] is not None and out["occupancy_pct"] > 0, out
    # G2: peak evidence is now MEASURED (runtime allocator high-water mark).
    assert out["peak_evidence_class"] == "MEASURED", out
    assert "analytical_or_allocation_upper_bound_bytes" in out, out
    assert "peak_saved_bytes" not in out, out  # the misleading name is gone
    # Task 2 (plan §5 2.1/2.3): legacy raw-allocation fields renamed to analytical
    # diagnostic names; old names must NOT appear in the producer output.
    assert "analytical_materialized_buffer_floor_bytes" in out, out
    assert "analytical_fused_buffer_floor_bytes" in out, out
    assert "materialized_peak_bytes" not in out, out  # renamed
    assert "fused_peak_bytes" not in out, out  # renamed
    # G2: MEASURED runtime allocator peak schema is now filled from the
    # full-anchor fused run. The c2 gate reads ONLY these fields for
    # region_peak_gain.
    assert out["materialized_runtime_allocator_peak_bytes"] is not None, out
    assert out["materialized_runtime_allocator_peak_bytes"] > 512 * 1024 * 1024, out
    assert out["fused_runtime_allocator_peak_bytes"] is not None, out
    assert (
        out["fused_runtime_allocator_peak_bytes"]
        < out["materialized_runtime_allocator_peak_bytes"]
    ), out
    assert out["runtime_peak_gain_bytes"] is not None, out
    assert out["runtime_peak_measurement_method"] is not None, out
    assert out["runtime_peak_scope"] is not None, out
    assert out["runtime_peak_sample_count"] is not None, out


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


def test_region_prototype_verdict_field_is_canonical():
    """G2: the canonical region_prototype.json verdict must carry a canonical
    criterion token, and the full-anchor fused run must be executed
    (fused_full_anchor_run=True, peak_evidence_class=MEASURED)."""
    import json
    import os

    from results._phase0.verdict_schema import CRITERION_TOKENS, normalize_criterion

    path = "results/phase0/region_prototype.json"
    assert os.path.exists(path), "canonical region_prototype.json missing"
    with open(path) as fh:
        proto = json.load(fh)

    # G2: the committed canonical artifact records the full-anchor run as DONE.
    assert proto["fused_full_anchor_run"] is True, proto
    assert proto["peak_evidence_class"] == "MEASURED", proto
    # The verdict field must be a canonical criterion token (PASS/FAIL/UNKNOWN).
    verdict = proto.get("verdict")
    assert verdict in CRITERION_TOKENS, (
        f"region_prototype.verdict={verdict!r} is not a canonical criterion "
        f"token (normalize_criterion maps {verdict!r} -> "
        f"{normalize_criterion(verdict)!r})"
    )


# ---------------------------------------------------------------------------
# Task G1: full-anchor direct-recompute correctness (GPU phase).
# Runs the existing fused_pte_kernel at FULL anchor dims (PM=4096, PN=16384,
# K1=1024, TM=64, TN=1048576) and compares E_fused against a materialized
# oracle E_mat (which materializes P and T) across 3 seeds. This resolves the
# region_fused criterion's correctness leg from UNKNOWN -> real PASS/FAIL.
# Memory: materialized ~1.7 GB peak (P+T+E+inputs), fused ~672 MiB (A+B+D+E
# only) -- both fit in 12 GB. Per-seed: materialize (P/T transient, freed
# before return), free pool, then fuse with the SAME seed (identical inputs).
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_full_anchor_direct_recompute_correctness():
    """Full-anchor fused (direct recompute) == materialized E, 3 seeds, near-exact."""
    from results._phase0.region_proto import run_full_anchor_correctness

    result = run_full_anchor_correctness(seeds=(0, 1, 2))
    assert result["n_seeds"] == 3
    assert result["worst_relative_l2"] < 1e-4, result
    assert result["worst_max_rel"] < 1e-3, result
    assert result["nan_inf"] is False
    # output shape/dtype/bytes
    assert result["output_shape"] == [64, 1048576]
    assert result["output_dtype"] == "complex64"
    assert result["output_bytes"] == 64 * 1048576 * 8
    # finding 3: make the "fused path does not allocate full P/T" proof
    # executable, not just inspectable. Frozen math contract:
    #   P = c64[4096,16384] = 4096*16384*8 = 536870912 bytes (512 MiB)
    #   T = c64[64,1048576] = 64*1048576*8 = 536870912 bytes (512 MiB)
    # The fused path avoids both; the materialized oracle reports the
    # transient P/T sizes it allocated and freed.
    assert result["P_bytes_avoided"] == 536870912, result
    assert result["T_bytes_avoided"] == 536870912, result


# ---------------------------------------------------------------------------
# Task G2: full-anchor MEASURED resources/peak/latency + verdict (GPU phase).
# Wires the full-anchor measurement into run() to produce the first honest
# MEASURED verdict (PASS/FAIL/UNKNOWN) for the region-fusion criterion,
# replacing the MODEL_ONLY / fused_full_anchor_run=false block.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_full_anchor_measured_verdict():
    """run() produces a MEASURED verdict (not MODEL_ONLY) with real peak/latency/resources."""
    import tempfile

    from results._phase0.region_proto import run

    with tempfile.TemporaryDirectory() as td:
        out = run(n=24, depth=10, out_dir=td)
    assert out["fused_full_anchor_run"] is True
    assert out["peak_evidence_class"] == "MEASURED"
    assert out["runtime_peak_measurement_method"] == "cuda_allocator_highwatermark"
    assert out["materialized_runtime_allocator_peak_bytes"] is not None
    assert (
        out["materialized_runtime_allocator_peak_bytes"] > 512 * 1024 * 1024
    )  # at least E (512MiB)
    assert out["fused_runtime_allocator_peak_bytes"] is not None
    assert (
        out["fused_runtime_allocator_peak_bytes"]
        < out["materialized_runtime_allocator_peak_bytes"]
    )  # leverage
    assert out["registers_per_thread"] is not None and out["registers_per_thread"] > 0
    assert out["occupancy_pct"] is not None
    assert (
        out["kernel_only_latency_ms"] is not None and out["kernel_only_latency_ms"] > 0
    )
    assert out["verdict"] in ("PASS", "FAIL", "UNKNOWN")
    # fused avoided P+T (no full materialization)
    assert out.get("fused_avoided_P_T") is True


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
