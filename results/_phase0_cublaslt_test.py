"""Tests for Phase 0 Plan B Task 2: planar-complex BF16 cublasLt matmul + judge."""

import numpy as np


def test_reference_complex_matmul_matches_numpy():
    from results._phase0_cublaslt import reference_complex_matmul

    m = k = n = 32
    rng = np.random.default_rng(0)
    ar = rng.standard_normal((m, k)).astype(np.float32)
    ai = rng.standard_normal((m, k)).astype(np.float32)
    br = rng.standard_normal((k, n)).astype(np.float32)
    bi = rng.standard_normal((k, n)).astype(np.float32)
    cr, ci = reference_complex_matmul(ar, ai, br, bi)
    A = (ar + 1j * ai).astype(np.complex64)
    B = (br + 1j * bi).astype(np.complex64)
    ref = A @ B
    assert np.allclose(cr, ref.real, atol=1e-3, rtol=1e-3)
    assert np.allclose(ci, ref.imag, atol=1e-3, rtol=1e-3)


def test_judge_capability_supported_when_all_pass():
    from results._phase0_cublaslt import judge_capability

    j = judge_capability(
        max_rel_err=1e-3,
        perf_ratio_vs_c64=1.5,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
    )
    assert j["status"] == "SUPPORTED", j


def test_judge_capability_not_supported_when_slow():
    from results._phase0_cublaslt import judge_capability

    j = judge_capability(
        max_rel_err=1e-3,
        perf_ratio_vs_c64=0.9,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
    )
    assert j["status"] == "NOT_SUPPORTED"
    assert "1.3" in j["reason"] or "speed" in j["reason"].lower()


def test_judge_capability_not_supported_when_no_algo():
    from results._phase0_cublaslt import judge_capability

    j = judge_capability(
        max_rel_err=1e-3,
        perf_ratio_vs_c64=2.0,
        algo_count=0,
        workspace_bytes=0,
        output_bytes=1 << 24,
        has_four_real_temps=False,
    )
    assert j["status"] == "NOT_SUPPORTED"


def test_judge_capability_accuracy_gate_is_max_rel():
    """BF16 output has ~0.4% relative error: a passing rel error (4e-3) must
    NOT be flagged, while a failing rel error (2e-2) must — even though the
    absolute error would look large in BF16-magnitude terms."""
    from results._phase0_cublaslt import judge_capability

    # 0.4% rel error, large abs (BF16-output tail) -> SUPPORTED.
    j_ok = judge_capability(
        max_rel_err=4e-3,
        perf_ratio_vs_c64=1.5,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
        max_abs_err=0.5,
    )
    assert j_ok["status"] == "SUPPORTED", j_ok

    # 2% rel error -> accuracy fail.
    j_bad = judge_capability(
        max_rel_err=2e-2,
        perf_ratio_vs_c64=1.5,
        algo_count=3,
        workspace_bytes=1 << 20,
        output_bytes=1 << 24,
        has_four_real_temps=False,
        max_abs_err=0.5,
    )
    assert j_bad["status"] == "NOT_SUPPORTED"
    assert "max_rel_err" in j_bad["reason"]


def test_load_c1_c2_shapes_filters_by_bytes(tmp_path):
    """load_c1_c2_shapes must keep only rows with bytes >= min_bytes and
    surface M/N/K/bytes/node_id as ints (node_id kept as string)."""
    from results._phase0_cublaslt import load_c1_c2_shapes

    csv = tmp_path / "shapes.csv"
    header = "n,depth,output,node_id,M,N,K,bytes\n"
    rows = [
        "22,10,expectation,0,2,2,2,32\n",  # 32 B  -> below 64 MiB
        "22,10,expectation,1,2048,2048,2048,134217728\n",  # 128 MiB -> kept
        "22,10,expectation,2,16384,1024,1024,134217728\n",  # 128 MiB -> kept
    ]
    csv.write_text(header + "".join(rows))
    out = load_c1_c2_shapes(str(csv), min_bytes=64 << 20)
    assert len(out) == 2
    assert out[0] == {
        "M": 2048,
        "N": 2048,
        "K": 2048,
        "bytes": 134217728,
        "node_id": "1",
    }
    assert out[1]["M"] == 16384 and out[1]["node_id"] == "2"


def test_load_c1_c2_shapes_skips_malformed_rows(tmp_path):
    """Rows with missing/non-int fields must be skipped, not crash."""
    from results._phase0_cublaslt import load_c1_c2_shapes

    csv = tmp_path / "shapes.csv"
    header = "n,depth,node_id,M,N,K,bytes\n"
    rows = [
        "22,10,0,2048,2048,2048,134217728\n",  # good
        "22,10,1,,,,\n",  # empty fields -> ValueError, skipped
        "22,10,2,4,4,4,notanumber\n",  # bytes not int -> skipped
        "22,10,3,8,8,8,256\n",  # below threshold -> skipped
    ]
    csv.write_text(header + "".join(rows))
    out = load_c1_c2_shapes(str(csv), min_bytes=64 << 20)
    assert len(out) == 1
    assert out[0]["M"] == 2048


def test_write_csv_roundtrip(tmp_path):
    from results._phase0_cublaslt import _write_csv
    import csv

    path = tmp_path / "out.csv"
    _write_csv(
        str(path),
        ["M", "N", "status"],
        [[256, 256, "ok"], [2048, 2048, "no-algo"]],
    )
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["M", "N", "status"]
    assert rows[1] == ["256", "256", "ok"]
    assert rows[2] == ["2048", "2048", "no-algo"]


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
