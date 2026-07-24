import numpy as np
import pytest


def test_compute_metrics_basic():
    from results._phase0.numerical import compute_metrics

    ref = np.ones((4, 4), dtype=np.complex64)
    out = (1 + 1e-3 * np.ones((4, 4))).astype(np.complex64)
    m = compute_metrics(out, ref)
    assert m["nan_inf"] is False
    assert m["n_elems"] == 16
    assert m["max_abs"] == pytest.approx(1e-3, rel=0.02)  # |out-ref| = 1e-3
    assert m["max_rel"] == pytest.approx(1e-3, rel=0.02)  # denom=max(|ref|,0.5)=1.0
    assert m["relative_l2"] == pytest.approx(1e-3, rel=0.02)  # ||diff||/max(1,||ref||)


def test_compute_metrics_detects_nan():
    from results._phase0.numerical import compute_metrics

    ref = np.ones((2, 2), dtype=np.complex64)
    out = np.array([[1, np.nan], [1, 1]], dtype=np.complex64)
    m = compute_metrics(out, ref)
    assert m["nan_inf"] is True


def test_make_inputs_baseline_stats():
    from results._phase0.numerical import make_inputs

    A, B = make_inputs("baseline", (1024, 1024, 64), seed=0)  # (M,N,K)
    assert A.shape == (1024, 64) and B.shape == (64, 1024)  # A=(M,K), B=(K,N)
    assert A.dtype == np.complex64
    # real & imag ~ N(0,1): mean ~0, std ~1
    assert abs(A.real.mean()) < 0.1 and abs(A.real.std() - 1.0) < 0.1


def test_make_inputs_mixed_scale_dynamic_range():
    from results._phase0.numerical import make_inputs

    A, _ = make_inputs("mixed_scale", (512, 32, 512), seed=1)
    mag = np.abs(A)
    # bimodal: some elements ~1e2, some ~1e-2 -> dynamic range ~1e4
    assert mag.max() > 50 and mag.min() < 0.1
    assert (mag > 50).sum() > 0 and (mag < 0.1).sum() > 0


def test_make_inputs_cancellation_paired_rows():
    from results._phase0.numerical import make_inputs

    _, B = make_inputs("cancellation", (64, 64, 64), seed=2)
    K = 64
    # B[2j+1] == -B[2j] for paired rows (cancellation structure, spec §4.3)
    assert np.allclose(B[1], -B[0])
    assert np.allclose(B[K - 1], -B[K - 2])


def test_make_inputs_deterministic_in_seed():
    from results._phase0.numerical import make_inputs

    a1, _ = make_inputs("baseline", (32, 8, 32), seed=5)
    a2, _ = make_inputs("baseline", (32, 8, 32), seed=5)
    assert np.array_equal(a1, a2)


def test_apply_policy_planar_c16bf_pass_just_under_threshold():
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": 9e-4, "max_abs": 9e-2, "max_rel": 4e-3, "nan_inf": False}
    verdict, _ = apply_policy("planar", "C16BF", m)
    assert verdict == "PASS"


def test_apply_policy_planar_c16bf_fail_on_max_rel():
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": 1e-4, "max_abs": 1e-2, "max_rel": 6e-3, "nan_inf": False}
    verdict, reason = apply_policy("planar", "C16BF", m)
    assert verdict == "FAIL"
    assert "max_rel" in reason


def test_apply_policy_c32f_tighter_than_c16bf():
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": 5e-4, "max_abs": 1e-2, "max_rel": 2e-3, "nan_inf": False}
    # passes C16BF (rel_l2<5e-3) but fails C32F (rel_l2<1e-4)
    assert apply_policy("planar", "C16BF", m)[0] == "PASS"
    assert apply_policy("planar", "C32F", m)[0] == "FAIL"


def test_apply_policy_nan_inf_fails_any_route():
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": 1e-9, "max_abs": 0.0, "max_rel": 0.0, "nan_inf": True}
    for route in ("planar", "grouped", "region_fused", "cutlass_4m_single"):
        assert apply_policy(route, "C16BF", m)[0] == "FAIL", route


def test_apply_policy_missing_metric_returns_none():
    from results._phase0.numerical import apply_policy

    # region_fused/cutlass omit max_abs policy; absent metric -> not FAIL, verdict stays PASS-able
    verdict, _ = apply_policy(
        "region_fused", "c64", {"relative_l2": 1e-5, "max_rel": 1e-4, "nan_inf": False}
    )
    assert verdict == "PASS"


def _row(route, dtype, shape, level, seed, rel_l2, max_abs, max_rel, nan):
    return {
        "route": route,
        "dtype": dtype,
        "shape": shape,
        "level": level,
        "seed": seed,
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nan_inf": nan,
    }


def test_aggregate_pass_when_all_cells_pass():
    from results._phase0.numerical import aggregate

    rows = [
        _row(
            "planar",
            "C16BF",
            (16384, 1024, 1024),
            "baseline",
            0,
            1e-4,
            1e-2,
            1e-3,
            False,
        )
    ]
    expected = {("planar", "C16BF"): 1}
    out = aggregate(rows, expected, case_hashes={}, legit_not_run=[])
    planar = [r for r in out["per_route"] if r["route"] == "planar"][0]
    assert planar["criterion"] == "PASS"
    assert out["overall_numerical_status"] == "PASS"


def test_aggregate_unknown_when_missing_rows():
    from results._phase0.numerical import aggregate

    rows = []  # expected 1 but present 0
    out = aggregate(
        rows, expected_counts={("planar", "C16BF"): 1}, case_hashes={}, legit_not_run=[]
    )
    planar = [r for r in out["per_route"] if r["route"] == "planar"][0]
    assert planar["criterion"] in ("UNKNOWN", "NOT_RUN")


def test_aggregate_fail_on_nan():
    from results._phase0.numerical import aggregate

    rows = [
        _row("planar", "C16BF", (16384, 1024, 1024), "baseline", 0, 0.0, 0.0, 0.0, True)
    ]
    out = aggregate(rows, {("planar", "C16BF"): 1}, {}, [])
    assert out["overall_numerical_status"] == "FAIL"


def test_aggregate_hash_mismatch_forces_unknown():
    from results._phase0.numerical import aggregate

    rows = [
        _row(
            "planar",
            "C16BF",
            (16384, 1024, 1024),
            "baseline",
            0,
            1e-4,
            1e-2,
            1e-3,
            False,
        )
    ]
    out = aggregate(
        rows,
        {("planar", "C16BF"): 1},
        case_hashes={"edge_map_hash": "MISMATCH"},
        legit_not_run=[],
    )
    assert out["overall_numerical_status"] == "INCONCLUSIVE"


import json
import os
import tempfile


def test_write_csv_header_and_rows(tmp_path):
    from results._phase0.numerical import write_csv

    p = tmp_path / "nv.csv"
    write_csv(str(p), [{"route": "planar", "M": 8, "relative_l2": 1e-4}])
    text = p.read_text()
    assert text.startswith(
        "route,M,N,K,out_dtype,dynamic_range_level,seed,relative_l2,max_abs,max_rel,nan_inf,n_elems,policy_pass,reference_dtype,source_hash,source"
    )
    assert "planar" in text


def test_write_json_roundtrip(tmp_path):
    from results._phase0.numerical import write_json

    p = tmp_path / "nv.json"
    payload = {
        "schema_version": "numerical-validation-v1",
        "overall_numerical_status": "PASS",
    }
    write_json(str(p), payload)
    assert json.loads(p.read_text())["overall_numerical_status"] == "PASS"


def test_shape_constants():
    from results._phase0.numerical import SHAPES, REAL_GEMM_SHAPES, LEVELS, SEEDS

    assert len(SHAPES) == 8
    assert (16384, 1024, 1024) in SHAPES
    assert set(REAL_GEMM_SHAPES) == {
        (16384, 1024, 1024),
        (524288, 32, 32),
        (262144, 64, 64),
        (1048576, 16, 16),
    }
    assert LEVELS == ("baseline", "mixed_scale", "cancellation")
    assert SEEDS == (0, 1, 2)


@pytest.mark.gpu
def test_collect_planar_smoke_one_cell(tmp_path):
    from results._phase0.numerical import collect_planar

    row = collect_planar((524288, 32, 32), "C16BF", "baseline", seed=0)
    assert row["route"] == "planar"
    assert row["dtype"] == "C16BF"
    assert row["shape"] == (524288, 32, 32)
    assert row["level"] == "baseline"
    assert "relative_l2" in row and "max_rel" in row and "nan_inf" in row
    assert row["policy_pass"] in (0, 1)
    # C16BF baseline should pass its own policy (bf16 ~4e-3 max_rel on N(0,1))
    assert row["policy_pass"] == 1, row


@pytest.mark.gpu
def test_collect_grouped_smoke_one_cell():
    from results._phase0.numerical import collect_grouped

    row = collect_grouped((524288, 32, 32), "C16BF", "baseline", seed=0, batch=4)
    assert row["route"] == "grouped"
    assert row["dtype"] == "C16BF"
    assert "relative_l2" in row and "nan_inf" in row
    assert row["policy_pass"] == 1, row


@pytest.mark.gpu
def test_collect_region_fused_small_contract():
    from results._phase0.numerical import collect_region_fused
    from results._phase0.region_proto import SMALL_SHAPES

    row = collect_region_fused("baseline", seed=0)
    assert row["route"] == "region_fused"
    assert row["dtype"] == "c64"
    assert row["shape"] == "small_contract"
    assert row["relative_l2"] < 1e-4  # fused == materialized at c64
    assert row["policy_pass"] == 1, row


def test_collect_cutlass_baseline_reads_task8_json():
    """Task 3a: the cutlass artifact measures max_rel but NOT relative_l2. The
    baseline row therefore carries relative_l2=None (never the max_rel proxy) and
    policy_pass=0 (apply_policy flags the cell incomplete on the missing canonical
    metric). The honest max_rel evidence is still recorded."""
    from results._phase0.numerical import collect_cutlass

    row = collect_cutlass("baseline", seed=0)
    assert row["route"] == "cutlass_4m_single"
    assert row["dtype"] == "C16BF"
    # max_rel evidence from Task 8 (~6.5e-5) still passes its own threshold
    assert row["max_rel"] < 5e-3
    # relative_l2 was NOT measured by the artifact -> None, never the max_rel proxy
    assert row["relative_l2"] is None, row
    assert row["relative_l2"] != row["max_rel"], row
    # policy can't conclude PASS without relative_l2 -> cell incomplete
    assert row["policy_pass"] == 0, row


def test_collect_cutlass_adversarial_records_not_run_when_unavailable(monkeypatch):
    from results._phase0 import numerical

    # force the injection probe to report unavailable
    monkeypatch.setattr(numerical, "_cutlass_injection_available", lambda: False)
    row = numerical.collect_cutlass("mixed_scale", seed=0)
    assert row.get("source", "").startswith("not_run")


def test_main_writes_artifacts_with_mocked_collectors(tmp_path, monkeypatch):
    from results._phase0 import numerical

    def fake_row(route, dtype, shape, level, seed):
        return {
            "route": route,
            "dtype": dtype,
            "shape": shape,
            "level": level,
            "seed": seed,
            "reference_dtype": "c64",
            "relative_l2": 1e-5,
            "max_abs": 1e-3,
            "max_rel": 1e-4,
            "nan_inf": False,
            "n_elems": 64,
            "policy_pass": 1,
        }

    monkeypatch.setattr(numerical, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        numerical,
        "collect_planar",
        lambda *a, **k: fake_row("planar", "C16BF", a[0], a[1], a[2]),
    )
    monkeypatch.setattr(
        numerical,
        "collect_grouped",
        lambda *a, **k: fake_row("grouped", "C16BF", a[0], a[1], a[2]),
    )
    monkeypatch.setattr(
        numerical,
        "collect_region_fused",
        lambda *a, **k: fake_row("region_fused", "c64", "small_contract", a[0], a[1]),
    )
    monkeypatch.setattr(
        numerical,
        "collect_cutlass",
        lambda *a, **k: fake_row(
            "cutlass_4m_single", "C16BF", (16384, 1024, 1024), a[0], a[1]
        ),
    )

    payload = numerical.main(run_gpu=False)
    assert (tmp_path / "numerical_validation.csv").exists()
    assert (tmp_path / "numerical_validation.json").exists()
    assert payload["schema_version"] == "numerical-validation-v1"
    routes = {r["route"] for r in payload["per_route"]}
    assert routes == {"planar", "grouped", "region_fused", "cutlass_4m_single"}


# ---------------------------------------------------------------------------
# Task 0 (SDD plan §3 操作.2): fail-closed RED baseline. The tests below freeze
# the target behavior the numerical reader must adopt after Task 3a wires the
# canonical verdict_schema in. They FAIL on the current implementation by clean
# assertion (not import, not GPU).
# ---------------------------------------------------------------------------


def test_aggregate_unknown_when_required_cell_not_run_is_undeclared():
    """plan §3 操作.2 bullet 3: any REQUIRED numerical cell whose source is
    ``not_run:*`` -- WITHOUT being declared in ``legit_not_run`` -- must force
    the route criterion to UNKNOWN.

    The current ``aggregate`` filters ``source=not_run:*`` rows out of
    ``real_cells`` before counting them against ``expected_counts``. That filter
    was added for the §7.2 cutlass-adversarial carve-out but it also masks any
    UNDECLARED not_run cell: as long as the surviving real cells meet
    ``expected_counts`` the route silently returns PASS. This test freezes the
    target: an undeclared ``not_run:*`` cell on a required (route, dtype) yields
    UNKNOWN, not PASS."""
    from results._phase0.numerical import aggregate

    rows = [
        {
            "route": "planar",
            "dtype": "C16BF",
            "shape": (16384, 1024, 1024),
            "level": "baseline",
            "seed": 0,
            "relative_l2": 1e-5,
            "max_abs": 1e-4,
            "max_rel": 1e-5,
            "nan_inf": False,
            "policy_pass": 1,
            # no source -> real measured cell
        },
        {
            "route": "planar",
            "dtype": "C16BF",
            "shape": (16384, 1024, 1024),
            "level": "mixed_scale",
            "seed": 0,
            "relative_l2": None,
            "max_abs": None,
            "max_rel": None,
            "nan_inf": False,
            "policy_pass": 0,
            "source": "not_run:toolchain",  # NOT declared in legit_not_run
        },
    ]
    out = aggregate(
        rows,
        expected_counts={("planar", "C16BF"): 1},  # baseline counted as 'expected'
        case_hashes={},
        legit_not_run=[],  # the not_run cell is NOT declared legit
    )
    planar = [r for r in out["per_route"] if r["route"] == "planar"][0]
    assert planar["criterion"] == "UNKNOWN", planar
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out


def test_collect_cutlass_does_not_substitute_max_rel_for_relative_l2(
    tmp_path, monkeypatch
):
    """plan §3 操作.2 bullet 4: when ``relative_l2`` is missing from the cutlass
    artifact's correctness block, ``collect_cutlass`` must NOT substitute
    ``max_rel`` as a proxy for it. Cross-metric substitution hides the missing
    evidence and lets apply_policy pass a cell that did not actually measure
    relative_l2.

    Today ``collect_cutlass`` baseline-path sets
    ``relative_l2 = c.get('max_rel', 1e9)`` (numerical.py), which is exactly the
    forbidden substitution. The fix emits ``relative_l2=None`` so apply_policy
    flags the cell incomplete. This test uses a synthetic cutlass artifact with
    NO ``relative_l2`` field and asserts the emitted row carries
    ``relative_l2=None`` -- failing today because the row inherits max_rel."""
    import json

    from results._phase0 import numerical

    # Synthetic cutlass_sm120_4m.json: correctness block has max_rel + max_abs
    # but NO relative_l2 field -> collect_cutlass must not invent one.
    (tmp_path / "cutlass_sm120_4m.json").write_text(
        json.dumps(
            {
                "single_4m": {
                    "correctness": {
                        "max_rel": 6.5e-5,
                        "max_abs": 1e-3,
                        "nan_inf": False,
                        # relative_l2 deliberately absent
                    }
                }
            }
        )
    )
    monkeypatch.setattr(numerical, "OUT_DIR", str(tmp_path))

    row = numerical.collect_cutlass("baseline", seed=0)
    # The row must carry relative_l2=None (missing), not the max_rel proxy.
    assert row["relative_l2"] is None, row
    # And it must never equal max_rel (the smoking gun for the substitution).
    assert row["relative_l2"] != row["max_rel"], row


# ---------------------------------------------------------------------------
# Task 3a: required-cell schema generator + JSON accounting (plan §6 3.1 / 3.3).
# These freeze the fail-closed behavior on the canonical schema so the per-route
# criterion is independently recomputable from the CSV + module constants.
# ---------------------------------------------------------------------------


def test_required_cell_keys_covers_all_routes_and_levels():
    """plan §6 3.1: the required-cell schema is the outer product of
    route x dtype x shape x {baseline, mixed_scale, cancellation} x >=3 seeds
    x c64 reference. Region uses the INTENDED full-anchor shape (P=A[4096,1024]
    @B[1024,16384] -> T -> E=D[64,64]@T), NOT_RUN until Task 3b."""
    from results._phase0.numerical import (
        CUTLASS_ANCHOR_SHAPE,
        DTYPES_BY_ROUTE,
        LEVELS,
        REGION_FULL_ANCHOR_SHAPE,
        SEEDS,
        SHAPES,
        required_cell_keys,
    )

    keys = required_cell_keys()
    # planar + grouped: 8 shapes x 2 dtypes x 3 levels x 3 seeds = 72 each (144 total)
    planar = {k for k in keys if k[0] == "planar"}
    grouped = {k for k in keys if k[0] == "grouped"}
    assert len(planar) == len(SHAPES) * len(DTYPES_BY_ROUTE["planar"]) * len(
        LEVELS
    ) * len(SEEDS)
    assert len(grouped) == len(SHAPES) * len(DTYPES_BY_ROUTE["grouped"]) * len(
        LEVELS
    ) * len(SEEDS)
    # region_fused: intended full-anchor shape x c64 x 3 levels x 3 seeds
    region = {k for k in keys if k[0] == "region_fused"}
    assert len(region) == len(LEVELS) * len(SEEDS)
    assert all(k[2] == REGION_FULL_ANCHOR_SHAPE for k in region)
    # cutlass: anchor shape x C16BF x 3 levels x 3 seeds
    cutlass = {k for k in keys if k[0] == "cutlass_4m_single"}
    assert len(cutlass) == len(LEVELS) * len(SEEDS)
    assert all(k[2] == CUTLASS_ANCHOR_SHAPE for k in cutlass)
    # every key carries the c64 reference id
    assert all(k[5] == "c64" for k in keys)


def test_aggregate_region_unknown_when_only_small_contract_measured():
    """plan §6 3.3: region_fused has 9 small-contract diagnostic rows (real
    measured), but those keys do NOT match the required full-anchor shape.
    The 9 intended full-anchor cells are missing -> route UNKNOWN, regardless of
    how good the small-contract correctness is. NOT_RUN cells never let a route
    PASS."""
    from results._phase0.numerical import aggregate, required_cell_keys

    # 9 small_contract diagnostic rows (real, very low error) -- these are NOT
    # the required full-anchor cells.
    rows = [
        {
            "route": "region_fused",
            "dtype": "c64",
            "shape": "small_contract",
            "level": level,
            "seed": seed,
            "reference_dtype": "c64",
            "relative_l2": 1e-7,
            "max_abs": 1e-6,
            "max_rel": 1e-7,
            "nan_inf": False,
        }
        for level in ("baseline", "mixed_scale", "cancellation")
        for seed in (0, 1, 2)
    ]
    out = aggregate(
        rows,
        required_cell_keys(),
        case_hashes={},
        legit_not_run=["region_fused:actual-large-fused:compute-bound (Task 3b)"],
    )
    region = [r for r in out["per_route"] if r["route"] == "region_fused"][0]
    assert region["criterion"] == "UNKNOWN", region
    # 9 expected full-anchor cells, 0 actual (small_contract keys don't match),
    # 9 missing, 9 extra (the diagnostic small_contract rows).
    assert region["expected"] == 9, region
    assert region["actual"] == 0, region
    assert region["missing"] == 9, region
    assert region["extra"] == 9, region
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out


def test_aggregate_cutlass_unknown_when_adversarial_not_run():
    """plan §6 3.3: cutlass baseline cells (3) are measured but the 6 adversarial
    cells are source=not_run:* -> route UNKNOWN. NOT_RUN rows never let a route
    PASS, even when legit (toolchain-injection-unavailable)."""
    from results._phase0.numerical import (
        CUTLASS_ANCHOR_SHAPE,
        aggregate,
        required_cell_keys,
    )

    rows = [
        {
            "route": "cutlass_4m_single",
            "dtype": "C16BF",
            "shape": CUTLASS_ANCHOR_SHAPE,
            "level": "baseline",
            "seed": seed,
            "reference_dtype": "c64",
            "relative_l2": 1e-5,
            "max_abs": 1e-4,
            "max_rel": 1e-5,
            "nan_inf": False,
            "source": "task8_reuse",
        }
        for seed in (0, 1, 2)
    ] + [
        {
            "route": "cutlass_4m_single",
            "dtype": "C16BF",
            "shape": CUTLASS_ANCHOR_SHAPE,
            "level": level,
            "seed": seed,
            "reference_dtype": "c64",
            "relative_l2": None,
            "max_abs": None,
            "max_rel": None,
            "nan_inf": False,
            "source": "not_run:toolchain-injection-unavailable",
        }
        for level in ("mixed_scale", "cancellation")
        for seed in (0, 1, 2)
    ]
    out = aggregate(
        rows,
        required_cell_keys(),
        case_hashes={},
        legit_not_run=["cutlass_4m_single:adversarial:toolchain-injection-unavailable"],
    )
    cutlass = [r for r in out["per_route"] if r["route"] == "cutlass_4m_single"][0]
    assert cutlass["criterion"] == "UNKNOWN", cutlass
    # 9 expected (3 levels x 3 seeds); 3 baseline measured; 6 missing (adversarial).
    assert cutlass["expected"] == 9, cutlass
    assert cutlass["actual"] == 3, cutlass
    assert cutlass["missing"] == 6, cutlass


def test_aggregate_duplicate_key_is_schema_error():
    """plan §6 3.1: a duplicate cell key is a schema error -- the producer must
    not silently dedup. Overall -> INCONCLUSIVE with a fail_closed_reason."""
    from results._phase0.numerical import aggregate

    rows = [
        {
            "route": "planar",
            "dtype": "C16BF",
            "shape": (16384, 1024, 1024),
            "level": "baseline",
            "seed": 0,
            "reference_dtype": "c64",
            "relative_l2": 1e-5,
            "max_abs": 1e-4,
            "max_rel": 1e-5,
            "nan_inf": False,
        },
        {
            "route": "planar",
            "dtype": "C16BF",
            "shape": (16384, 1024, 1024),
            "level": "baseline",
            "seed": 0,
            "reference_dtype": "c64",
            "relative_l2": 2e-5,
            "max_abs": 2e-4,
            "max_rel": 2e-5,
            "nan_inf": False,
        },
    ]
    out = aggregate(
        rows,
        {("planar", "C16BF"): 1},
        case_hashes={},
        legit_not_run=[],
    )
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out
    assert any("duplicate" in r.lower() for r in out["fail_closed_reasons"]), out[
        "fail_closed_reasons"
    ]


# ---------------------------------------------------------------------------
# Task 3a CSV NOT_RUN fix (spec §6 3.3: "CSV 中保留 NOT_RUN row 及 reason").
# The CSV must be self-describing: every required cell with no measured data is
# represented by a NOT_RUN row carrying its reason in the ``source`` column, and
# the aggregate recomputed purely from the CSV returns the same verdicts.
# ---------------------------------------------------------------------------


def test_write_csv_round_trip_preserves_not_run_source(tmp_path):
    """The ``source`` column survives a write->read round trip so the
    ``not_run:<reason>`` / ``diagnostic:small-contract`` labels are not stripped
    (spec §6 3.3). Previously ``source`` lived only ephemerally on in-memory rows
    because ``_CSV_COLUMNS`` lacked the column; write_csv silently dropped it."""
    from results._phase0.numerical import _read_csv_rows, write_csv

    rows = [
        {
            "route": "region_fused",
            "dtype": "c64",
            "shape": (4096, 16384, 1024),
            "level": "baseline",
            "seed": 0,
            "reference_dtype": "c64",
            "relative_l2": None,
            "max_abs": None,
            "max_rel": None,
            "nan_inf": False,
            "n_elems": 0,
            "policy_pass": 0,
            "source": "not_run:compute-bound-actual-large-fused",
        },
        {
            "route": "region_fused",
            "dtype": "c64",
            "shape": "small_contract",
            "level": "baseline",
            "seed": 0,
            "reference_dtype": "c64",
            "relative_l2": 1e-7,
            "max_abs": 1e-6,
            "max_rel": 1e-7,
            "nan_inf": False,
            "n_elems": 32,
            "policy_pass": 1,
            "source": "diagnostic:small-contract",
        },
    ]
    p = tmp_path / "nv.csv"
    write_csv(str(p), rows)
    read_back = _read_csv_rows(str(p))
    by_key = {(r["route"], r["shape"], r["level"], r["seed"]): r for r in read_back}
    not_run = by_key[("region_fused", (4096, 16384, 1024), "baseline", 0)]
    diag = by_key[("region_fused", "small_contract", "baseline", 0)]
    assert not_run["source"] == "not_run:compute-bound-actual-large-fused", not_run
    assert not_run["relative_l2"] is None
    assert diag["source"] == "diagnostic:small-contract", diag
    assert diag["relative_l2"] == pytest.approx(1e-7)


def test_regenerated_csv_contains_region_full_anchor_not_run_rows():
    """The regenerated ``numerical_validation.csv`` (real artifact) MUST now list
    the 9 region_fused intended-full-anchor cells as explicit NOT_RUN rows with a
    ``not_run:<reason>`` source (spec §6 3.3). Previously these 9 required cells
    existed only as JSON ``missing=9`` -- the CSV had zero rows for them."""
    import csv
    import os

    from results._phase0.numerical import REGION_FULL_ANCHOR_SHAPE

    csv_path = os.path.join("results", "phase0", "numerical_validation.csv")
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    region_not_run = [
        r
        for r in rows
        if r["route"] == "region_fused"
        and r["source"].startswith("not_run:")
        and (int(r["M"]), int(r["N"]), int(r["K"])) == REGION_FULL_ANCHOR_SHAPE
    ]
    # 3 levels x 3 seeds = 9 intended full-anchor NOT_RUN cells.
    assert len(region_not_run) == 9, region_not_run
    # every NOT_RUN row carries a non-empty reason after the ``not_run:`` prefix
    for r in region_not_run:
        assert r["source"] != "not_run:", r
        assert r["relative_l2"] == "", r  # empty metrics
    # levels x seeds coverage is complete
    levels_seeds = {(r["dynamic_range_level"], int(r["seed"])) for r in region_not_run}
    assert levels_seeds == {
        (lvl, s)
        for lvl in ("baseline", "mixed_scale", "cancellation")
        for s in (0, 1, 2)
    }


def test_regenerated_csv_contains_cutlass_not_run_rows_with_reason():
    """The regenerated ``numerical_validation.csv`` MUST preserve the cutlass
    adversarial NOT_RUN reason (``not_run:toolchain-injection-unavailable``) in
    the ``source`` column. Previously the reason was stripped because
    ``_CSV_COLUMNS`` lacked ``source``; it lived only ephemerally on the
    in-memory row from collect_cutlass."""
    import csv
    import os

    csv_path = os.path.join("results", "phase0", "numerical_validation.csv")
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cutlass_not_run = [
        r
        for r in rows
        if r["route"] == "cutlass_4m_single" and r["source"].startswith("not_run:")
    ]
    # 2 adversarial levels (mixed_scale, cancellation) x 3 seeds = 6 NOT_RUN cells.
    assert len(cutlass_not_run) == 6, cutlass_not_run
    for r in cutlass_not_run:
        assert "toolchain-injection-unavailable" in r["source"], r
        assert r["relative_l2"] == "", r


def test_csv_is_self_describing_aggregate_matches_json_verdicts():
    """Reading the regenerated CSV and recomputing the aggregate MUST yield the
    same per-route verdicts as the committed JSON -- proving the CSV is now
    self-describing (the NOT_RUN rows carry enough signal for the fail-closed
    aggregate without consulting the JSON's fail_closed_reasons)."""
    import json
    import os

    from results._phase0.numerical import (
        _case_hashes,
        _legit_not_run_reasons,
        _read_csv_rows,
        aggregate,
        required_cell_keys,
    )

    csv_path = os.path.join("results", "phase0", "numerical_validation.csv")
    json_path = os.path.join("results", "phase0", "numerical_validation.json")
    rows = _read_csv_rows(csv_path)
    payload = aggregate(
        rows,
        required_cell_keys(),
        _case_hashes(),
        _legit_not_run_reasons(),
    )
    with open(json_path) as fh:
        committed = json.load(fh)
    verdict_from_csv = {r["route"]: r["criterion"] for r in payload["per_route"]}
    verdict_from_json = {r["route"]: r["criterion"] for r in committed["per_route"]}
    # Verdicts UNCHANGED (planar/grouped FAIL; region_fused/cutlass UNKNOWN).
    assert verdict_from_csv == verdict_from_json, (verdict_from_csv, verdict_from_json)
    assert verdict_from_csv["region_fused"] == "UNKNOWN"
    assert verdict_from_csv["cutlass_4m_single"] == "UNKNOWN"
    assert payload["overall_numerical_status"] == "INCONCLUSIVE"
    # expected/actual/missing/extra counts unchanged (NOT_RUN rows never count as
    # measured): the CSV-derived accounting matches the committed JSON exactly.
    for r_csv, r_json in zip(payload["per_route"], committed["per_route"]):
        assert r_csv["route"] == r_json["route"]
        for field in ("expected", "actual", "missing", "extra"):
            assert r_csv[field] == r_json[field], (r_csv, r_json)


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.4: cancellation input must produce real cancellation.
# ---------------------------------------------------------------------------


def test_cancellation_input_produces_real_cancellation_ratio():
    """Nongpu rereview finding 3.4: the cancellation input must produce a real
    cancellation (``cancellation_norm / baseline_norm < configured_ratio``),
    not just paired B rows. Current construction has ``B[2j+1]=-B[2j]`` but
    ``A[:,2j]`` and ``A[:,2j+1]`` are independent, so the contribution
    ``(A[:,2j]-A[:,2j+1])@B[2j]`` is NOT near zero -> no actual cancellation.

    The test also verifies the reference is non-zero finite (the controlled
    residual that prevents an all-zero reference). Current construction fails
    the ratio assertion (ratio ~1.0, not < 0.1) -> RED."""
    import numpy as np

    from results._phase0.numerical import make_inputs

    shape = (64, 64, 64)  # K=64 (even, required for cancellation)
    seed = 0

    # Baseline: random A, B -> C = A @ B (reference magnitude).
    A_base, B_base = make_inputs("baseline", shape, seed)
    C_base = A_base @ B_base
    baseline_norm = float(np.linalg.norm(C_base))
    assert np.isfinite(baseline_norm) and baseline_norm > 0

    # Cancellation: B[2j+1] = -B[2j], A[:,2j] and A[:,2j+1] independent.
    A_cancel, B_cancel = make_inputs("cancellation", shape, seed)
    C_cancel = A_cancel @ B_cancel
    cancellation_norm = float(np.linalg.norm(C_cancel))
    assert np.isfinite(cancellation_norm)

    # The cancellation ratio must be small (real cancellation). Current
    # construction doesn't achieve cancellation (A columns independent) ->
    # ratio ~ 1.0, NOT < 0.1.
    ratio = cancellation_norm / baseline_norm
    assert ratio < 0.1, (
        f"cancellation_norm/baseline_norm = {ratio:.4f} >= 0.1; "
        f"the cancellation input does not produce real cancellation"
    )


# ---------------------------------------------------------------------------
# Nongpu rereview finding 3.11: numerical shapes must be bound to contraction
# artifact, not hardcoded.
# ---------------------------------------------------------------------------


def test_numerical_shapes_derived_from_contraction_csv_not_hardcoded():
    """Nongpu rereview finding 3.11: ``SHAPES`` must be derived from
    ``contraction_shapes.csv`` (or asserted equal to it), with shape drift
    producing UNKNOWN (not a silent re-hash). Current code hardcodes 8 SHAPES
    as a standalone constant with no CSV binding or drift detection.

    The module must provide a shape loader (``load_current_shapes``) that reads
    the contraction artifact. Currently no such function exists -- SHAPES is
    hardcoded, so a CSV update can silently re-hash while the required shape
    set stays stale."""
    from results._phase0 import numerical
    from results._phase0.numerical import SHAPES

    # The module MUST provide a shape loader bound to contraction_shapes.csv.
    # Currently SHAPES is a standalone hardcoded constant.
    assert hasattr(numerical, "load_current_shapes"), (
        "numerical.SHAPES is hardcoded; must be derived from "
        "contraction_shapes.csv via load_current_shapes() so shape drift "
        "produces UNKNOWN instead of a silent re-hash"
    )
    loaded = numerical.load_current_shapes()
    assert set(SHAPES) == set(
        loaded
    ), f"SHAPES != contraction_shapes.csv shapes: drift must produce UNKNOWN"


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
