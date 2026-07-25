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

    A, B = make_inputs("cancellation", (64, 64, 64), seed=2)
    K = 64
    # A[:, 2j+1] == A[:, 2j] (paired equal columns, plan §3.1 / spec §3.4)
    assert np.allclose(A[:, 1], A[:, 0])
    assert np.allclose(A[:, K - 1], A[:, K - 2])
    # B[2j+1] ≈ -B[2j] (paired negative + controlled residual so the
    # paired contribution cancels while keeping the reference non-zero).
    # The residual (eps * N(0,1) ~ 1e-3) is small, so B[2j+1] + B[2j] ≈ 0
    # but NOT exactly zero (prevents all-zero reference).
    assert np.allclose(B[1], -B[0], atol=0.1)
    assert np.allclose(B[K - 1], -B[K - 2], atol=0.1)


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


# ---------------------------------------------------------------------------
# F2 (evidence-integrity remediation): apply_policy must reject NaN / inf /
# negative / non-numeric / bool metrics (fail-closed). Previously NaN and
# negative values silently passed because ``NaN >= thresh`` and ``-1 >= thresh``
# are both False -> no FAIL -> PASS (fail-open). A bool metric (True/False) is
# also rejected because bool is not a valid error metric (False would pass every
# threshold, and True would be misreported as a threshold breach rather than an
# invalid metric).
# ---------------------------------------------------------------------------

_INVALID_METRIC_VALUES = [
    float("nan"),
    float("inf"),
    float("-inf"),
    -1.0,
    -1e-9,
    "abc",
    True,
    False,
]


@pytest.mark.parametrize("bad_val", _INVALID_METRIC_VALUES)
def test_apply_policy_relative_l2_invalid_fails(bad_val):
    """F2: an invalid relative_l2 (NaN/inf/negative/non-numeric/bool) -> FAIL."""
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": bad_val, "max_abs": 1e-3, "max_rel": 1e-5, "nan_inf": False}
    verdict, reason = apply_policy("planar", "C16BF", m)
    assert verdict == "FAIL", (bad_val, verdict, reason)
    assert "invalid" in reason, (bad_val, reason)


def test_apply_policy_relative_l2_none_returns_none():
    """F2: relative_l2=None -> None (missing metric, cell incomplete)."""
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": None, "max_abs": 1e-3, "max_rel": 1e-5, "nan_inf": False}
    verdict, reason = apply_policy("planar", "C16BF", m)
    assert verdict is None, (verdict, reason)
    assert "missing" in reason, reason


@pytest.mark.parametrize("bad_val", _INVALID_METRIC_VALUES)
def test_apply_policy_max_rel_invalid_fails(bad_val):
    """F2: an invalid max_rel (NaN/inf/negative/non-numeric/bool) -> FAIL."""
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": 1e-5, "max_abs": 1e-3, "max_rel": bad_val, "nan_inf": False}
    verdict, reason = apply_policy("planar", "C16BF", m)
    assert verdict == "FAIL", (bad_val, verdict, reason)
    assert "invalid" in reason, (bad_val, reason)


def test_apply_policy_max_rel_none_returns_none():
    """F2: max_rel=None -> None (missing metric, cell incomplete)."""
    from results._phase0.numerical import apply_policy

    m = {"relative_l2": 1e-5, "max_abs": 1e-3, "max_rel": None, "nan_inf": False}
    verdict, reason = apply_policy("planar", "C16BF", m)
    assert verdict is None, (verdict, reason)
    assert "missing" in reason, reason


def test_apply_policy_real_measured_values_pass():
    """F2: real measured values (planar C16BF, spec §4.2) -> PASS (not rejected)."""
    from results._phase0.numerical import apply_policy

    m = {
        "relative_l2": 1.66e-3,
        "max_abs": 0.136,
        "max_rel": 3.85e-3,
        "nan_inf": False,
    }
    verdict, reason = apply_policy("planar", "C16BF", m)
    assert verdict == "PASS", (verdict, reason)


def test_apply_policy_bool_metric_fails():
    """F2: a bool metric (True/False) -> FAIL (bool is not a valid metric).

    Without the explicit ``isinstance(val, bool)`` guard, ``False`` would pass
    every threshold (``0 < thresh``) -> PASS (fail-open), and ``True`` would be
    misreported as a threshold breach (``1 >= thresh``) rather than an invalid
    metric.
    """
    from results._phase0.numerical import apply_policy

    for bad in (True, False):
        m = {
            "relative_l2": bad,
            "max_abs": 1e-3,
            "max_rel": 1e-5,
            "nan_inf": False,
        }
        verdict, reason = apply_policy("planar", "C16BF", m)
        assert verdict == "FAIL", (bad, verdict, reason)
        assert "invalid" in reason, (bad, reason)


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


def _valid_case_hashes():
    """Construct a VALID case_hashes dict for aggregate tests: all 9 required
    case-binding keys present with valid 64-char sha256 hex, so
    binding_unavailable=False and the route-local loop runs. (Tests that
    exercise the global-invalid deny-all path construct their own broken
    case_hashes.) Task 4 errata #2: the old ``case_hashes={}`` now triggers
    binding_unavailable (no case keys present) -> deny-all, so tests that need
    to exercise the route-local loop MUST supply valid case_hashes."""
    from results._phase0.numerical import _case_hashes

    valid_hex = "a" * 64
    return {
        "algorithm": "sha256",
        **{k: valid_hex for k in _case_hashes() if k != "algorithm"},
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
    out = aggregate(rows, expected, case_hashes=_valid_case_hashes(), legit_not_run=[])
    planar = [r for r in out["per_route"] if r["route"] == "planar"][0]
    assert planar["criterion"] == "PASS"
    assert out["overall_numerical_status"] == "PASS"


def test_aggregate_unknown_when_missing_rows():
    from results._phase0.numerical import aggregate

    rows = []  # expected 1 but present 0
    out = aggregate(
        rows,
        expected_counts={("planar", "C16BF"): 1},
        case_hashes=_valid_case_hashes(),
        legit_not_run=[],
    )
    planar = [r for r in out["per_route"] if r["route"] == "planar"][0]
    assert planar["criterion"] in ("UNKNOWN", "NOT_RUN")


def test_aggregate_fail_on_nan():
    from results._phase0.numerical import aggregate

    rows = [
        _row("planar", "C16BF", (16384, 1024, 1024), "baseline", 0, 0.0, 0.0, 0.0, True)
    ]
    out = aggregate(rows, {("planar", "C16BF"): 1}, _valid_case_hashes(), [])
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
    # Valid case_hashes for all 9 keys, then set ONE to MISMATCH so
    # binding_mismatch fires (but binding_unavailable does NOT -- the other
    # 8 keys are valid). This isolates the mismatch path from the unavailable
    # path (Task 4 errata: both are global-invalid, but tested separately).
    mismatch_hashes = _valid_case_hashes()
    mismatch_hashes["edge_map_sha256"] = "MISMATCH"
    out = aggregate(
        rows,
        {("planar", "C16BF"): 1},
        case_hashes=mismatch_hashes,
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
        "route,M,N,K,out_dtype,dynamic_range_level,seed,relative_l2,max_abs,max_rel,nan_inf,n_elems,policy_pass,reference_dtype,cell_key_hash,source"
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
    """G5: collect_region_fused now runs the FULL-ANCHOR P->T->E contract
    (PM=4096, PN=16384, K1=1024, TM=64, TN=1048576) via the direct-recompute
    fused kernel (G1) vs the materialized oracle, at the requested dynamic-
    range level. The small-contract diagnostic path is retired (G5 promotes
    region_fused from NOT_RUN to MEASURED at the full anchor)."""
    from results._phase0.numerical import collect_region_fused, REGION_FULL_ANCHOR_SHAPE

    row = collect_region_fused("baseline", seed=0)
    assert row["route"] == "region_fused"
    assert row["dtype"] == "c64"
    assert row["shape"] == REGION_FULL_ANCHOR_SHAPE
    # fused == materialized at c64 (G1 showed rel_l2 ~ 8.5e-7)
    assert row["relative_l2"] < 1e-4
    assert row["source"] == "measured"
    assert row["policy_pass"] == 1, row


@pytest.mark.gpu
def test_region_fused_full_anchor_numerical_measured():
    """G5 Step 1: collect_region_fused(level, seed) returns a MEASURED row at
    the full-anchor shape with real relative_l2 >= 0 and a cell key in
    required_cell_keys(). The row carries the canonical
    input_construction_version token (baseline_v1 / mixed_scale_v1 /
    cancellation_v2) so it matches the 7-tuple required-cell schema."""
    from results._phase0.numerical import (
        collect_region_fused,
        required_cell_keys,
        _cell_key,
    )

    row = collect_region_fused(level="baseline", seed=0)
    assert row["source"] == "measured"
    assert row["input_construction_version"] == "baseline_v1"
    assert row["relative_l2"] is not None and row["relative_l2"] >= 0
    # full-anchor shape (M=4096, N=16384, K=1024)
    assert row["shape"] == (4096, 16384, 1024)
    key = _cell_key(row)
    assert key in required_cell_keys()


@pytest.mark.gpu
def test_collect_cutlass_measured_when_toolchain_available():
    """G5: when the cutlass toolchain is available (CUTLASS_ROOT/CUDA_HOME env
    vars set + cutlass include dir present), collect_cutlass returns a MEASURED
    row with real relative_l2 >= 0 at the cutlass anchor shape. The row carries
    the canonical input_construction_version token and its cell key is in
    required_cell_keys()."""
    import os

    from results._phase0.numerical import (
        collect_cutlass,
        required_cell_keys,
        _cell_key,
        _cutlass_toolchain_available,
    )

    if not _cutlass_toolchain_available():
        pytest.skip("cutlass toolchain not available (CUTLASS_ROOT/CUDA_HOME)")
    row = collect_cutlass(level="baseline", seed=0)
    assert row["source"] == "measured", row
    assert row["input_construction_version"] == "baseline_v1", row
    assert row["relative_l2"] is not None and row["relative_l2"] >= 0, row
    assert row["shape"] == (16384, 1024, 1024), row
    key = _cell_key(row)
    assert key in required_cell_keys(), key


def test_collect_cutlass_baseline_not_run_when_toolchain_unavailable(monkeypatch):
    """G5: when the cutlass toolchain is unavailable (CUTLASS_ROOT/CUDA_HOME env
    vars not set), collect_cutlass returns an honest NOT_RUN row with the real
    failure reason -- never fakes a measurement. The input_construction_version
    token is ALWAYS set so the row's cell key matches required_cell_keys().

    Replaces the old ``test_collect_cutlass_baseline_reads_task8_json`` which
    tested the task8_reuse path (G5 retires that path in favor of real
    measurement when the toolchain is available)."""
    from results._phase0 import numerical

    # Force the toolchain probe to report unavailable (no env vars / include dir).
    monkeypatch.setattr(numerical, "_cutlass_toolchain_available", lambda: False)
    row = numerical.collect_cutlass("baseline", seed=0)
    assert row["route"] == "cutlass_4m_single"
    assert row["dtype"] == "C16BF"
    assert row["source"].startswith("not_run"), row
    assert "toolchain" in row["source"].lower(), row
    # relative_l2 is None (not measured), never faked
    assert row["relative_l2"] is None, row
    # version token is ALWAYS set so the cell key matches required_cell_keys
    assert row["input_construction_version"] == "baseline_v1", row


def test_collect_cutlass_adversarial_records_not_run_when_unavailable(monkeypatch):
    from results._phase0 import numerical

    # force the toolchain probe to report unavailable
    monkeypatch.setattr(numerical, "_cutlass_toolchain_available", lambda: False)
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
        case_hashes=_valid_case_hashes(),
        legit_not_run=[],  # the not_run cell is NOT declared legit
    )
    planar = [r for r in out["per_route"] if r["route"] == "planar"][0]
    assert planar["criterion"] == "UNKNOWN", planar
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out


def test_collect_cutlass_not_run_carries_none_metrics(monkeypatch):
    """G5: when the cutlass toolchain is unavailable, collect_cutlass returns a
    NOT_RUN row with relative_l2=None AND max_rel=None (both absent, not
    substituted). The old task8_reuse path (which carried max_rel from the
    artifact but not relative_l2) is retired by G5 -- when the toolchain is
    unavailable, ALL metrics are None (honest NOT_RUN, no partial reuse).

    Replaces ``test_collect_cutlass_does_not_substitute_max_rel_for_relative_l2``
    which tested the task8_reuse path's max_rel->relative_l2 substitution concern
    (G5 eliminates that path entirely)."""
    from results._phase0 import numerical

    # Force the toolchain probe to report unavailable.
    monkeypatch.setattr(numerical, "_cutlass_toolchain_available", lambda: False)
    row = numerical.collect_cutlass("baseline", seed=0)
    # Both metrics are None (NOT_RUN, no partial reuse from task8 json).
    assert row["relative_l2"] is None, row
    assert row["max_rel"] is None, row
    # And they are not substituting one for the other.
    assert row["relative_l2"] == row["max_rel"]  # both None (honest NOT_RUN)
    assert row["policy_pass"] == 0, row


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
    # every key is a 7-tuple and carries the c64 reference id at position 6
    assert all(len(k) == 7 for k in keys)
    assert all(k[6] == "c64" for k in keys)


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
        case_hashes=_valid_case_hashes(),
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
            "input_construction_version": "baseline_v1",
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
            "input_construction_version": (
                level + "_v1" if level != "cancellation" else "cancellation_v2"
            ),
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
        case_hashes=_valid_case_hashes(),
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
        case_hashes=_valid_case_hashes(),
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


def test_regenerated_csv_contains_region_full_anchor_measured_rows():
    """G5: the regenerated ``numerical_validation.csv`` (real artifact) MUST now
    list the 9 region_fused full-anchor cells as MEASURED rows with real
    ``relative_l2`` (source=``measured``). Previously (pre-G5) these 9 required
    cells were NOT_RUN -- G5 promotes region_fused from NOT_RUN to MEASURED at
    the full anchor via the direct-recompute fused_pte_kernel (G1)."""
    import csv
    import os

    from results._phase0.numerical import REGION_FULL_ANCHOR_SHAPE

    csv_path = os.path.join("results", "phase0", "numerical_validation.csv")
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    region_measured = [
        r
        for r in rows
        if r["route"] == "region_fused"
        and r["source"] == "measured"
        and (int(r["M"]), int(r["N"]), int(r["K"])) == REGION_FULL_ANCHOR_SHAPE
    ]
    # 3 levels x 3 seeds = 9 full-anchor MEASURED cells.
    assert len(region_measured) == 9, region_measured
    for r in region_measured:
        assert r["relative_l2"] != "", r  # real measured metric
        rel_l2 = float(r["relative_l2"])
        assert rel_l2 >= 0, r  # non-negative
        assert rel_l2 < 1e-4, r  # within policy (G1: rel_l2 ~ 8.5e-7)
    # levels x seeds coverage is complete
    levels_seeds = {(r["dynamic_range_level"], int(r["seed"])) for r in region_measured}
    assert levels_seeds == {
        (lvl, s)
        for lvl in ("baseline", "mixed_scale", "cancellation")
        for s in (0, 1, 2)
    }


def test_regenerated_csv_contains_cutlass_measured_or_not_run_rows():
    """G5: the regenerated ``numerical_validation.csv`` lists 9 cutlass_4m_single
    rows (3 levels x 3 seeds) that are EITHER measured (source=``measured`` with
    real relative_l2, when the cutlass toolchain was available during regen) OR
    not_run (source=``not_run:...`` with the real failure reason, when the
    toolchain was unavailable). Replaces the old test that expected exactly 6
    adversarial NOT_RUN rows (G5 measures all 3 levels when the toolchain is
    available)."""
    import csv
    import os

    csv_path = os.path.join("results", "phase0", "numerical_validation.csv")
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cutlass_rows = [r for r in rows if r["route"] == "cutlass_4m_single"]
    # 3 levels x 3 seeds = 9 cutlass rows total (measured or not_run).
    assert len(cutlass_rows) == 9, cutlass_rows
    for r in cutlass_rows:
        if r["source"] == "measured":
            assert r["relative_l2"] != "", r  # real measured metric
        else:
            assert r["source"].startswith("not_run"), r
            assert r["relative_l2"] == "", r  # NOT_RUN -> empty metrics
    # levels x seeds coverage is complete
    levels_seeds = {(r["dynamic_range_level"], int(r["seed"])) for r in cutlass_rows}
    assert levels_seeds == {
        (lvl, s)
        for lvl in ("baseline", "mixed_scale", "cancellation")
        for s in (0, 1, 2)
    }


def test_csv_is_self_describing_aggregate_matches_json_verdicts(tmp_path):
    """Reading a written CSV and recomputing the aggregate MUST yield the
    same per-route verdicts as the JSON payload written alongside it -- proving
    the CSV is self-describing (the NOT_RUN rows carry enough signal for the
    fail-closed aggregate without consulting the JSON's fail_closed_reasons).

    This test uses a synthetic CSV written via ``write_csv`` in ``tmp_path``
    (rather than the committed ``results/phase0/numerical_validation.csv``)
    because the on-disk artifact is regenerated in Task 9's no-GPU reaggregation
    step; until then the committed CSV carries the pre-7-tuple schema and is
    not expected to be self-describing under the new 7-tuple required keys."""
    import json
    import os

    from results._phase0.numerical import (
        CUTLASS_ANCHOR_SHAPE,
        REGION_FULL_ANCHOR_SHAPE,
        _legit_not_run_reasons,
        _read_csv_rows,
        aggregate,
        required_cell_keys,
        write_csv,
        write_json,
    )

    # Synthetic rows: a subset of the required matrix with correct 7-tuple
    # version tokens. region_fused full-anchor + cutlass adversarial are
    # NOT_RUN; planar baseline is measured (PASS). This mirrors the committed
    # artifact's structure but with the new schema.
    rows = [
        {
            "route": "planar",
            "dtype": "C16BF",
            "shape": (16384, 1024, 1024),
            "level": "baseline",
            "input_construction_version": "baseline_v1",
            "seed": seed,
            "reference_dtype": "c64",
            "relative_l2": 1e-5,
            "max_abs": 1e-4,
            "max_rel": 1e-5,
            "nan_inf": False,
            "n_elems": 64,
            "policy_pass": 1,
            "source": "measured",
        }
        for seed in (0, 1, 2)
    ]
    # Add region_fused + cutlass NOT_RUN rows so those routes appear.
    from results._phase0.numerical import _emit_not_run_rows

    rows.extend(_emit_not_run_rows(rows, required_cell_keys()))

    csv_path = str(tmp_path / "numerical_validation.csv")
    write_csv(csv_path, rows)
    payload = aggregate(
        rows, required_cell_keys(), _valid_case_hashes(), _legit_not_run_reasons()
    )
    write_json(str(tmp_path / "numerical_validation.json"), payload)

    # Read the CSV back and recompute -- must match the written JSON.
    read_back = _read_csv_rows(csv_path)
    recomputed = aggregate(
        read_back,
        required_cell_keys(),
        _valid_case_hashes(),
        _legit_not_run_reasons(),
    )
    with open(str(tmp_path / "numerical_validation.json")) as fh:
        committed = json.load(fh)
    verdict_from_csv = {r["route"]: r["criterion"] for r in recomputed["per_route"]}
    verdict_from_json = {r["route"]: r["criterion"] for r in committed["per_route"]}
    assert verdict_from_csv == verdict_from_json, (verdict_from_csv, verdict_from_json)
    assert verdict_from_csv["region_fused"] == "UNKNOWN"
    assert verdict_from_csv["cutlass_4m_single"] == "UNKNOWN"
    assert recomputed["overall_numerical_status"] == "INCONCLUSIVE"
    for r_csv, r_json in zip(recomputed["per_route"], committed["per_route"]):
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


# ---------------------------------------------------------------------------
# Nongpu rereview Task 3 review fix: cancellation metrics must be recorded in
# the numerical output (CSV), not just computable by calling cancellation_metrics
# directly. The 5 fields make the cancellation independently auditable from the
# artifacts.
# ---------------------------------------------------------------------------


def test_cancellation_metrics_recorded_in_numerical_output(tmp_path):
    """Nongpu rereview Task 3 review fix: ``cancellation_metrics()`` was defined
    but never called -- none of the 5 cancellation diagnostic fields appeared in
    any CSV row or JSON payload. The brief requires they be recorded in the
    numerical output so the cancellation is independently auditable from the
    artifacts (the 3.4 RED test passes only because it calls ``make_inputs``
    directly; the recorded-output requirement is separate).

    This test verifies GPU-free that:
    1. ``cancellation_metrics`` returns all 5 required fields with sane values.
    2. The 5 fields are in the CSV schema (``_CSV_COLUMNS``).
    3. ``_enrich_cancellation_metrics`` wires them into a cancellation-level row.
    4. ``write_csv`` -> ``_read_csv_rows`` round-trip preserves the 5 fields.

    RED on current code: ``_enrich_cancellation_metrics`` does not exist, and the
    5 fields are absent from ``_CSV_COLUMNS``.
    """
    from results._phase0.numerical import (
        _CSV_COLUMNS,
        _enrich_cancellation_metrics,
        cancellation_metrics,
        write_csv,
        _read_csv_rows,
    )

    required_fields = (
        "input_construction_version",
        "cancellation_epsilon",
        "reference_norm",
        "baseline_norm",
        "cancellation_ratio",
    )

    # 1. cancellation_metrics returns all 5 required fields with sane values.
    shape = (64, 64, 64)  # K=64 (even, required for cancellation)
    seed = 0
    cm = cancellation_metrics(shape, seed)
    for f in required_fields:
        assert f in cm, f"cancellation_metrics missing field {f}"
    assert cm["input_construction_version"] != ""
    assert np.isfinite(cm["cancellation_epsilon"]) and cm["cancellation_epsilon"] > 0
    assert np.isfinite(cm["reference_norm"]) and cm["reference_norm"] > 0
    assert np.isfinite(cm["baseline_norm"]) and cm["baseline_norm"] > 0
    assert np.isfinite(cm["cancellation_ratio"])
    assert cm["cancellation_ratio"] < 0.1, cm["cancellation_ratio"]

    # 2. The 5 fields are in the CSV schema.
    for f in required_fields:
        assert f in _CSV_COLUMNS, f"_CSV_COLUMNS missing cancellation field {f}"

    # 3. _enrich_cancellation_metrics wires the fields into a cancellation row.
    row = {
        "route": "planar",
        "dtype": "C16BF",
        "shape": shape,
        "level": "cancellation",
        "seed": seed,
        "reference_dtype": "c64",
        "relative_l2": 1e-5,
        "max_abs": 1e-3,
        "max_rel": 1e-4,
        "nan_inf": False,
        "n_elems": 64,
        "policy_pass": 1,
    }
    enriched = _enrich_cancellation_metrics(dict(row))
    for f in required_fields:
        assert f in enriched, f"enriched row missing cancellation field {f}"
    assert enriched["input_construction_version"] == cm["input_construction_version"]
    assert enriched["cancellation_ratio"] == pytest.approx(cm["cancellation_ratio"])

    # Non-cancellation rows are NOT enriched (no false fields).
    baseline_row = dict(row)
    baseline_row["level"] = "baseline"
    baseline_enriched = _enrich_cancellation_metrics(baseline_row)
    for f in required_fields:
        assert (
            f not in baseline_enriched
        ), f"non-cancellation row should not carry cancellation field {f}"

    # 4. write_csv -> _read_csv_rows round-trip preserves the 5 fields.
    p = tmp_path / "nv.csv"
    write_csv(str(p), [enriched])
    read_back = _read_csv_rows(str(p))
    assert len(read_back) == 1
    rr = read_back[0]
    for f in required_fields:
        assert f in rr, f"CSV round-trip lost cancellation field {f}"
    assert rr["input_construction_version"] == cm["input_construction_version"]
    assert rr["cancellation_epsilon"] == pytest.approx(cm["cancellation_epsilon"])
    assert rr["reference_norm"] == pytest.approx(cm["reference_norm"])
    assert rr["baseline_norm"] == pytest.approx(cm["baseline_norm"])
    assert rr["cancellation_ratio"] == pytest.approx(cm["cancellation_ratio"])


# ---------------------------------------------------------------------------
# Task 1 (evidence-integrity remediation): 7-tuple cell key + unified
# ``cancellation_v2`` token + INV-1/INV-2 (finding 3.1). The tests below freeze
# the target behavior: the cell-key schema becomes a 7-tuple that includes
# ``input_construction_version``; the canonical cancellation token is
# ``cancellation_v2`` (producer constant + required key + CSV reader/writer);
# ALL old measured cancellation rows are relabeled ``cancellation_legacy_v1``
# in a no-GPU regen (provenance forgery fix); baseline/mixed_scale producers
# MUST write ``baseline_v1``/``mixed_scale_v1`` tokens so those routes can
# complete. INV-1: non-GPU round ``cancellation_v2`` + ``measured`` row
# count == 0.
# ---------------------------------------------------------------------------

import csv as _csv
import os as _os

from results._phase0 import numerical


def test_emit_not_run_rows_handles_7_tuple(monkeypatch, tmp_path):
    monkeypatch.setattr(numerical, "OUT_DIR", str(tmp_path))
    required = numerical.required_cell_keys()
    assert all(len(k) == 7 for k in required)
    out = numerical._emit_not_run_rows([], required)  # must not raise ValueError
    assert len(out) == len(required)
    for r in out:
        assert r["input_construction_version"] in (
            "cancellation_v2",
            "baseline_v1",
            "mixed_scale_v1",
        )


def test_canonical_token_is_cancellation_v2():
    assert numerical.INPUT_CONSTRUCTION_VERSION == "cancellation_v2"
    m = numerical.cancellation_metrics((16384, 1024, 1024), 0)
    assert m["input_construction_version"] == "cancellation_v2"


def test_legacy_does_not_satisfy_v2_required():
    legacy = {
        "route": "planar",
        "dtype": "C16BF",
        "shape": (16384, 1024, 1024),
        "level": "cancellation",
        "input_construction_version": "cancellation_legacy_v1",
        "seed": 0,
        "reference_dtype": "c64",
    }
    assert numerical._cell_key(legacy) not in numerical.required_cell_keys()


def test_synthetic_gpu_v2_row_satisfies_required():
    gpu = {
        "route": "planar",
        "dtype": "C16BF",
        "shape": (16384, 1024, 1024),
        "level": "cancellation",
        "input_construction_version": "cancellation_v2",
        "seed": 0,
        "reference_dtype": "c64",
        "source": "measured",
        "relative_l2": 1e-4,
        "max_rel": 1e-4,
        "nan_inf": False,
        "policy_pass": True,
    }
    assert numerical._cell_key(gpu) in numerical.required_cell_keys()  # liveness


def test_regen_no_gpu_zero_v2_measured_and_legacy_kept(monkeypatch, tmp_path):
    """errata #3 / INV-1: in a no-GPU regen, ``cancellation_v2`` + ``measured``
    row count MUST be 0 (no GPU v2 run occurred). ALL old measured cancellation
    rows are relabeled ``cancellation_legacy_v1`` regardless of their old token
    (e.g. ``v2_cancellation``). The only ``cancellation_v2`` rows in the output
    are the NOT_RUN required-cell rows emitted by ``_emit_not_run_rows`` (their
    key set must EXACTLY equal the required cancellation-level keys)."""
    monkeypatch.setattr(numerical, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        numerical,
        "collect_cutlass",
        lambda level, seed: {
            "route": "cutlass_4m_single",
            "dtype": "C16BF",
            "shape": numerical.CUTLASS_ANCHOR_SHAPE,
            "level": level,
            "input_construction_version": "cancellation_v2",
            "seed": seed,
            "reference_dtype": "c64",
            "source": "not_run:toolchain-unavailable",
            "relative_l2": None,
        },
    )
    monkeypatch.setattr(numerical, "shapes_in_sync", lambda: True)
    # Capture the real reader before monkeypatching so we can read the written
    # CSV back through the production normalization path (csv.DictReader rows
    # lack dtype/level/shape keys that _cell_key requires).
    _real_read_csv_rows = numerical._read_csv_rows
    monkeypatch.setattr(
        numerical,
        "_read_csv_rows",
        lambda p: [
            {
                "route": "planar",
                "dtype": "C16BF",
                "shape": (16384, 1024, 1024),
                "level": "cancellation",
                "seed": 0,
                "reference_dtype": "c64",
                "source": "measured",
                "relative_l2": 1e-3,
                "input_construction_version": "v2_cancellation",
            }
        ],
    )
    numerical.main(run_gpu=False, regen_no_gpu=True)
    rows = _real_read_csv_rows(_os.path.join(str(tmp_path), "numerical_validation.csv"))
    # INV-1: zero cancellation_v2 + measured rows
    assert (
        len(
            [
                r
                for r in rows
                if r.get("input_construction_version") == "cancellation_v2"
                and r.get("source") == "measured"
            ]
        )
        == 0
    )
    # Old measured cancellation -> legacy
    assert (
        len(
            [
                r
                for r in rows
                if r.get("input_construction_version") == "cancellation_legacy_v1"
            ]
        )
        >= 1
    )
    # Exact v2 NOT_RUN key set: the cancellation_v2 rows must be EXACTLY the
    # required cancellation-level cells (no spurious v2 rows on other levels).
    v2_required = {k for k in numerical.required_cell_keys() if k[3] == "cancellation"}
    v2_notrun = {
        numerical._cell_key(r)
        for r in rows
        if r.get("input_construction_version") == "cancellation_v2"
    }
    assert v2_notrun == v2_required, (
        f"v2_notrun ({len(v2_notrun)} keys) != v2_required ({len(v2_required)} keys); "
        f"extra: {v2_notrun - v2_required}; missing: {v2_required - v2_notrun}"
    )


def test_normative_policy_cell_key_fields_populated():
    """errata #8: ``cell_key_fields`` in ``normative_policy.json`` is the single
    source of truth for cell-key field names and must match the 7-tuple."""
    from results._phase0.gate_contracts import load_normative_policy

    pol = load_normative_policy()
    assert pol["cell_key_fields"] == [
        "route",
        "dtype",
        "shape",
        "level",
        "input_construction_version",
        "seed",
        "reference_dtype",
    ]


def test_baseline_mixed_producers_write_version_tokens():
    """errata #4: baseline/mixed_scale producers MUST write ``baseline_v1`` /
    ``mixed_scale_v1`` tokens so those routes can match ``required_cell_keys``."""
    # collect_cutlass baseline/mixed_scale are GPU-free (artifact read / NOT_RUN).
    from results._phase0.numerical import collect_cutlass

    bl = collect_cutlass("baseline", seed=0)
    assert bl.get("input_construction_version") == "baseline_v1", bl
    ms = collect_cutlass("mixed_scale", seed=0)
    assert ms.get("input_construction_version") == "mixed_scale_v1", ms


# ---------------------------------------------------------------------------
# Task 4 (evidence-integrity remediation, finding 3.4): numerical
# global-invalid explicit flags. The aggregate must compute global-invalid
# (duplicate / shape_drift / binding_mismatch / binding_unavailable) BEFORE
# the per-route loop. If global_invalid, ALL per_route = UNKNOWN and overall =
# INCONCLUSIVE (return early). Previously aggregate computed per-route FIRST,
# so a shape_drift / duplicate / binding error could leave overall=INCONCLUSIVE
# but a route=PASS, and gonogo reads per_route directly -> route VIABLE while
# NUMERICAL=UNKNOWN (fail-open). legit_not_run is informational only (does NOT
# set global_invalid).
# ---------------------------------------------------------------------------


def test_legit_not_run_does_not_clear_per_route():
    """Task 4 errata #3: legit_not_run is informational only -- it must NOT
    change per_route criteria or overall_status. Verified by a with/without
    comparison: the same globally-valid matrix with vs without legit_not_run
    entries yields IDENTICAL per_route criteria and overall_status. (Replaces
    the brief's ``or True`` tautology which asserted nothing.)"""
    from results._phase0.numerical import aggregate, required_cell_keys

    # Construct a globally-valid matrix (no duplicate/drift/mismatch/unavailable)
    # with ALL required cells measured + passing, so per_route would be PASS.
    rows = []
    for k in required_cell_keys():
        route, dtype, shape, level, ver, seed, ref = k
        rows.append(
            {
                "route": route,
                "dtype": dtype,
                "shape": shape,
                "level": level,
                "input_construction_version": ver,
                "seed": seed,
                "reference_dtype": ref,
                "source": "measured",
                "relative_l2": 1e-5,
                "max_rel": 1e-5,
                "nan_inf": False,
                "policy_pass": True,
            }
        )
    hashes = _valid_case_hashes()
    out_with = aggregate(
        rows,
        required_cell_keys(),
        hashes,
        ["some legit not-run reason"],
        shape_drift=False,
    )
    out_without = aggregate(rows, required_cell_keys(), hashes, [], shape_drift=False)
    # per_route criteria IDENTICAL (legit_not_run does NOT clear them)
    pr_with = {r["route"]: r["criterion"] for r in out_with["per_route"]}
    pr_without = {r["route"]: r["criterion"] for r in out_without["per_route"]}
    assert pr_with == pr_without, (pr_with, pr_without)
    # overall_status IDENTICAL
    assert (
        out_with["overall_numerical_status"] == out_without["overall_numerical_status"]
    ), (
        out_with["overall_numerical_status"],
        out_without["overall_numerical_status"],
    )
    # legit_not_run reason IS recorded in fail_closed_reasons (informational)
    assert "some legit not-run reason" in out_with["fail_closed_reasons"]
    # but it does NOT trigger global_invalid (no deny-all -> overall PASS)
    assert out_with["overall_numerical_status"] == "PASS", out_with


def test_duplicate_clears_all_per_route():
    """Task 4 finding 3.4: a duplicate cell key is a schema error -> ALL
    per_route criteria = UNKNOWN (global-invalid deny-all). On the old code,
    the per-route loop ran first, so a duplicate only set overall=INCONCLUSIVE
    while leaving per_route potentially PASS (fail-open: gonogo read per_route
    directly -> route VIABLE while NUMERICAL=UNKNOWN). Uses legacy count mode
    with expected=1 so that WITHOUT the duplicate deny-all, planar would reach
    PASS (1 measured == 1 expected, policy passes)."""
    from results._phase0.numerical import aggregate

    r = {
        "route": "planar",
        "dtype": "C16BF",
        "shape": (16384, 1024, 1024),
        "level": "baseline",
        "input_construction_version": "baseline_v1",
        "seed": 0,
        "reference_dtype": "c64",
        "source": "measured",
        "relative_l2": 1e-4,
        "max_rel": 1e-4,
        "nan_inf": False,
        "policy_pass": True,
    }
    out = aggregate(
        [r, r], {("planar", "C16BF"): 1}, _valid_case_hashes(), [], shape_drift=False
    )
    for pr in out["per_route"]:
        assert pr["criterion"] == "UNKNOWN", pr
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out
    assert any("duplicate" in reason.lower() for reason in out["fail_closed_reasons"])


def test_binding_unavailable_clears_per_route():
    """Task 4 finding 3.4: when case-binding hashes are unavailable (empty
    values), ALL per_route criteria = UNKNOWN (the case binding is broken ->
    cannot validate any route). Uses legacy count mode with expected=1 so that
    WITHOUT the binding-unavailable deny-all, planar would reach PASS."""
    from results._phase0.numerical import aggregate, _case_hashes

    r = {
        "route": "planar",
        "dtype": "C16BF",
        "shape": (16384, 1024, 1024),
        "level": "baseline",
        "input_construction_version": "baseline_v1",
        "seed": 0,
        "reference_dtype": "c64",
        "source": "measured",
        "relative_l2": 1e-4,
        "max_rel": 1e-4,
        "nan_inf": False,
        "policy_pass": True,
    }
    hashes = {k: "" for k in _case_hashes()}  # empty -> UNAVAILABLE
    hashes["algorithm"] = "sha256"
    out = aggregate([r], {("planar", "C16BF"): 1}, hashes, [], shape_drift=False)
    for pr in out["per_route"]:
        assert pr["criterion"] == "UNKNOWN", pr
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out
    assert any("unavailable" in reason.lower() for reason in out["fail_closed_reasons"])


def test_binding_unavailable_empty_dict():
    """Task 4 errata #2: binding_unavailable MUST handle the case where
    case_hashes = {"algorithm": "sha256"} (only the algorithm key, no case
    hashes). The plan's original ``any(v == "" for k,v in case_hashes.items()
    if k != "algorithm")`` would filter out "algorithm", leaving ``{}`` ->
    ``any([])`` = False -> binding_unavailable=False (WRONG -- the binding is
    actually unavailable because there are NO case hashes). Fix:
    binding_unavailable is True if the set of case-hash keys excluding
    "algorithm" is EMPTY (no case bindings present)."""
    from results._phase0.numerical import aggregate

    r = {
        "route": "planar",
        "dtype": "C16BF",
        "shape": (16384, 1024, 1024),
        "level": "baseline",
        "input_construction_version": "baseline_v1",
        "seed": 0,
        "reference_dtype": "c64",
        "source": "measured",
        "relative_l2": 1e-4,
        "max_rel": 1e-4,
        "nan_inf": False,
        "policy_pass": True,
    }
    # Only the algorithm key, no case hashes -> binding_unavailable=True
    out = aggregate(
        [r], {("planar", "C16BF"): 1}, {"algorithm": "sha256"}, [], shape_drift=False
    )
    for pr in out["per_route"]:
        assert pr["criterion"] == "UNKNOWN", pr
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out
    assert any("unavailable" in reason.lower() for reason in out["fail_closed_reasons"])


def test_binding_unavailable_short_hash():
    """Task 4 finding 3.4 fix: a case-binding hash that is valid hex but too
    SHORT (e.g. ``"a"*10`` -- 10 chars, valid hex) MUST trigger
    ``binding_unavailable`` -> ALL per_route = UNKNOWN. ``_case_hashes()``
    returns the full 64-char ``sha256().hexdigest()`` (no truncation), so the
    ``_is_invalid_hash`` threshold is 64 (``_CASE_HASH_MIN_LEN``); the old
    ``< 8`` threshold let an 8-63-char valid-hex string pass -> the route-local
    loop ran -> potential false PASS. Uses legacy count mode with expected=1
    so WITHOUT the deny-all, planar would reach PASS."""
    from results._phase0.numerical import aggregate, _case_hashes

    r = {
        "route": "planar",
        "dtype": "C16BF",
        "shape": (16384, 1024, 1024),
        "level": "baseline",
        "input_construction_version": "baseline_v1",
        "seed": 0,
        "reference_dtype": "c64",
        "source": "measured",
        "relative_l2": 1e-4,
        "max_rel": 1e-4,
        "nan_inf": False,
        "policy_pass": True,
    }
    valid_hex = "a" * 64
    # All required case keys present; ONE set to a 10-char valid-hex string
    # (valid hex, but shorter than the 64-char sha256 case-binding length).
    case_keys = [k for k in _case_hashes() if k != "algorithm"]
    hashes = {"algorithm": "sha256"}
    for k in case_keys:
        hashes[k] = valid_hex
    hashes[case_keys[0]] = "a" * 10  # too-short valid-hex -> UNAVAILABLE
    out = aggregate([r], {("planar", "C16BF"): 1}, hashes, [], shape_drift=False)
    for pr in out["per_route"]:
        assert pr["criterion"] == "UNKNOWN", pr
    assert out["overall_numerical_status"] == "INCONCLUSIVE", out
    assert any("unavailable" in reason.lower() for reason in out["fail_closed_reasons"])


def test_complete_required_matrix_reaches_pass():
    """Task 4 errata #4: a complete required matrix (ALL required cells
    measured + passing policy) must reach PASS via the REAL aggregate function
    (not a mock). The global predicate must be VALID (no duplicate/drift/
    mismatch/unavailable) so it's not a deny-all. The synthetic fixture MAY
    include a cancellation_v2 measured row (the no-GPU prohibition only
    constrains COMMITTED artifacts, not test fixtures)."""
    from results._phase0.numerical import aggregate, required_cell_keys

    rows = []
    for k in required_cell_keys():
        route, dtype, shape, level, ver, seed, ref = k
        rows.append(
            {
                "route": route,
                "dtype": dtype,
                "shape": shape,
                "level": level,
                "input_construction_version": ver,
                "seed": seed,
                "reference_dtype": ref,
                "source": "measured",
                "relative_l2": 1e-5,
                "max_rel": 1e-5,
                "nan_inf": False,
                "policy_pass": True,
            }
        )
    out = aggregate(
        rows, required_cell_keys(), _valid_case_hashes(), [], shape_drift=False
    )
    # Global predicate VALID -> no deny-all reasons
    reasons = " ".join(out["fail_closed_reasons"]).lower()
    assert "duplicate" not in reasons, out["fail_closed_reasons"]
    assert "shape drift" not in reasons, out["fail_closed_reasons"]
    assert "mismatch" not in reasons, out["fail_closed_reasons"]
    assert "unavailable" not in reasons, out["fail_closed_reasons"]
    # ALL required cells measured + pass -> overall PASS via REAL aggregate
    assert out["overall_numerical_status"] == "PASS", out
    for pr in out["per_route"]:
        assert pr["criterion"] == "PASS", pr


# ---------------------------------------------------------------------------
# Task 8 Step 4 concrete: synthetic pipeline VIABLE via recompute_route_verdict
# ---------------------------------------------------------------------------


def test_synthetic_pipeline_route_viable():
    """Construct criteria with all-PASS capability + numerical PASS for
    cutlass_4m_single -> route VIABLE via the REAL
    verdict_schema.recompute_route_verdict. The synthetic fixture builds
    criteria through the real aggregate output pattern (PASS tokens)."""
    from results._phase0.verdict_schema import recompute_route_verdict

    # Only CUTLASS_SM80_FALLBACK_CAPABILITY gates cutlass_4m_single capability.
    criteria = {
        "CUTLASS_SM80_FALLBACK_CAPABILITY": "PASS",
    }
    per_route = {"cutlass_4m_single": "PASS"}
    rv = recompute_route_verdict(criteria, per_route)
    assert rv["cutlass_4m_single"]["status"] == "VIABLE", rv
    assert rv["cutlass_4m_single"]["capability"] == "OK", rv
    assert rv["cutlass_4m_single"]["numerical"] == "OK", rv


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
