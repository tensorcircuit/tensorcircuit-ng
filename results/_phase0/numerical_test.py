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


def test_aggregate_legit_not_run_does_not_sink_overall():
    from results._phase0.numerical import aggregate

    # region_fused actual-large fused is legit NOT_RUN (compute-bound, spec §7.2)
    rows = [
        _row(
            "region_fused",
            "c64",
            "small_contract",
            "baseline",
            0,
            1e-7,
            0.0,
            1e-7,
            False,
        )
    ]
    out = aggregate(
        rows,
        {("region_fused", "c64"): 1},
        {},
        legit_not_run=["region_fused:actual-large-fused:compute-bound"],
    )
    assert out["overall_numerical_status"] == "PASS"
    assert any("compute-bound" in r for r in out["fail_closed_reasons"])


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
        "route,M,N,K,out_dtype,dynamic_range_level,seed,relative_l2,max_abs,max_rel,nan_inf,n_elems,policy_pass,reference_dtype,source_hash"
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
    from results._phase0.numerical import collect_cutlass

    row = collect_cutlass("baseline", seed=0)
    assert row["route"] == "cutlass_4m_single"
    assert row["dtype"] == "C16BF"
    # baseline reuses Task 8 (max_rel ~6.5e-5) -> passes C16BF policy
    assert row["max_rel"] < 5e-3
    assert row["policy_pass"] == 1, row


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


def test_aggregate_cutlass_not_run_adversarial_does_not_sink():
    """Spec §7.2 contract: cutlass criterion == PASS when its baseline cells PASS
    and adversarial rows carry source='not_run:...' — legit NOT_RUN must NOT sink
    overall to INCONCLUSIVE."""
    from results._phase0.numerical import aggregate

    rows = [
        {
            "route": "cutlass_4m_single",
            "dtype": "C16BF",
            "shape": (16384, 1024, 1024),
            "level": "baseline",
            "seed": 0,
            "relative_l2": 1e-5,
            "max_abs": 1e-4,
            "max_rel": 1e-5,
            "nan_inf": False,
            "policy_pass": 1,
            "source": "task8_reuse",
        },
        {
            "route": "cutlass_4m_single",
            "dtype": "C16BF",
            "shape": (16384, 1024, 1024),
            "level": "mixed_scale",
            "seed": 0,
            "relative_l2": None,
            "max_abs": None,
            "max_rel": None,
            "nan_inf": False,
            "policy_pass": 0,
            "source": "not_run:toolchain",
        },
    ]
    out = aggregate(
        rows,
        expected_counts={("cutlass_4m_single", "C16BF"): 1},
        case_hashes={},
        legit_not_run=[],
    )
    cutlass = [r for r in out["per_route"] if r["route"] == "cutlass_4m_single"][0]
    assert (
        cutlass["criterion"] == "PASS"
    ), cutlass  # baseline PASS, adversarial not_run does not sink
