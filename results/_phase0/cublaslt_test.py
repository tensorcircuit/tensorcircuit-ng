"""Tests for Phase 0 Plan B Task 2: planar-complex BF16 cublasLt matmul + judge."""

import numpy as np


def test_reference_complex_matmul_matches_numpy():
    from results._phase0.cublaslt import reference_complex_matmul

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
    from results._phase0.cublaslt import judge_capability

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
    from results._phase0.cublaslt import judge_capability

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
    from results._phase0.cublaslt import judge_capability

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
    from results._phase0.cublaslt import judge_capability

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
    from results._phase0.cublaslt import load_c1_c2_shapes

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
    from results._phase0.cublaslt import load_c1_c2_shapes

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


def test_time_planar_kernelonly_extracts_median_ms():
    """_time_planar_kernelonly delegates to the ext kernel-only timing call and
    returns its median_ms as a float, forwarding iters/warmup. Verified GPU-free
    with a stub ext so the contract is locked without a compiled extension / GPU;
    the live (positive-ms) check is the run_matrix integration run."""
    from results._phase0.cublaslt import _time_planar_kernelonly

    class _StubExt:
        def __init__(self):
            self.last_kwargs = None

        def planar_complex_matmul_bf16_kernelonly_timing(self, *args, **kwargs):
            self.last_kwargs = kwargs
            return {
                "median_ms": 1.25,
                "algo_id": 0,
                "workspace_bytes": 0,
                "iters": kwargs.get("iters", 5),
                "warmup": kwargs.get("warmup", 3),
                "status": "OK",
            }

    ext = _StubExt()
    ms = _time_planar_kernelonly(
        ext, None, None, None, None, 256, 256, 256, iters=7, warmup=2
    )
    assert ms == 1.25
    assert isinstance(ms, float)
    assert ext.last_kwargs == {
        "iters": 7,
        "warmup": 2,
    }


def test_write_csv_roundtrip(tmp_path):
    from results._phase0.cublaslt import _write_csv
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


# ---------------------------------------------------------------------------
# Task 6: C3 planar FULL MATRIX (actual-large policy aggregation, spec §3.6)
# ---------------------------------------------------------------------------

from results._phase0.cublaslt import (  # noqa: E402
    aggregate_capability_full,
    probe_config,
)

# Real-gemm (min dim >= floor) actual-large shapes vs skinny (diagnostic).
_REAL_GEMM = (16384, 1024, 1024)  # the C2 anchor
_SKINNY = (8388608, 2, 2)  # N*K <= 16 -> TC-unfriendly, diagnostic


def _shape_result(mnk, *, algo=3, rel=1e-3, ko=2.0, ws=0, out_bytes=1 << 28):
    m, n, k = mnk
    return {
        "M": m,
        "N": n,
        "K": k,
        "algo_count": algo,
        "max_rel_err": rel,
        "ko_ratio": ko,
        "workspace_bytes": ws,
        "output_bytes": out_bytes,
    }


def test_aggregate_supported_when_all_real_gemm_pass():
    """All real-gemm actual-large shapes pass the 7.5 gate -> SUPPORTED."""
    res = aggregate_capability_full(
        [
            _shape_result(_REAL_GEMM, ko=2.0),
            _shape_result((524288, 32, 32), ko=1.5),
            _shape_result(
                _SKINNY, ko=0.5
            ),  # skinny fails perf -> diagnostic, not gating
        ]
    )
    assert res["status"] == "SUPPORTED", res
    # skinny is recorded as diagnostic, not gating
    assert res["per_shape"][_SKINNY]["is_real_gemm"] is False


def test_aggregate_not_supported_when_only_skinny_pass():
    """Anti-cherry-pick (spec 3.6): only small/skinny shapes passing must NOT trigger
    SUPPORTED, even if every skinny shape is fast."""
    res = aggregate_capability_full(
        [
            _shape_result(_SKINNY, ko=5.0),  # skinny fast
            _shape_result((262144, 64, 4), ko=4.0),  # skinny fast
            _shape_result(_REAL_GEMM, ko=0.8),  # the one real-gemm is slow
        ]
    )
    assert res["status"] == "NOT_SUPPORTED", res
    assert (
        "small" in res["reason"].lower()
        or "skinny" in res["reason"].lower()
        or "real-gemm" in res["reason"].lower()
    )


def test_aggregate_not_supported_when_real_gemm_has_no_algo():
    """A real-gemm shape with algo_count=0 is a real limitation -> NOT_SUPPORTED."""
    res = aggregate_capability_full(
        [
            _shape_result(_REAL_GEMM, algo=0, ko=0.0),
            _shape_result((524288, 32, 32), ko=2.0),
        ]
    )
    assert res["status"] == "NOT_SUPPORTED", res


def test_aggregate_quorum_all_required_by_default():
    """Default quorum=1.0: one real-gemm failing vetoes SUPPORTED."""
    res = aggregate_capability_full(
        [_shape_result(_REAL_GEMM, ko=2.0), _shape_result((524288, 32, 32), ko=0.9)]
    )
    assert res["status"] == "NOT_SUPPORTED", res


def test_aggregate_quorum_configurable():
    """quorum=0.5: 1 of 2 real-gemm passing is enough."""
    res = aggregate_capability_full(
        [_shape_result(_REAL_GEMM, ko=2.0), _shape_result((524288, 32, 32), ko=0.9)],
        quorum=0.5,
    )
    assert res["status"] == "SUPPORTED", res


def test_aggregate_no_real_gemm_is_not_supported():
    """Only skinny shapes evaluated -> NOT_SUPPORTED (no real-gemm evidence)."""
    res = aggregate_capability_full([_shape_result(_SKINNY, ko=5.0)])
    assert res["status"] == "NOT_SUPPORTED", res


def test_full_matrix_csv_schema(tmp_path):
    from results._phase0.cublaslt import write_full_matrix_csv
    import csv

    path = tmp_path / "fm.csv"
    rows = [
        {
            "M": 16384,
            "N": 1024,
            "K": 1024,
            "out_dtype": "bf16",
            "ws_cap": "0",
            "op": "N",
            "aligned": 1,
            "algo_count": 3,
            "first_algo_id": 21,
            "workspace_bytes": 0,
            "status": "ok",
        }
    ]
    write_full_matrix_csv(str(path), rows)
    with open(path) as f:
        out = list(csv.reader(f))
    assert out[0][:7] == ["M", "N", "K", "out_dtype", "ws_cap", "op", "aligned"]
    assert out[0][-1] == "status"
    assert out[1][3] == "bf16" and out[1][-1] == "ok"


def test_probe_config_forwards_params_to_ext():
    """probe_config maps op->transa/transb and forwards out_dtype/ws_limit to the extension.
    GPU-free stub locks the contract; the live (algo_count>0) check is the GPU run."""
    seen = {}

    class _StubExt:
        def probe_planar_capability(self, m, n, k, **kw):
            seen.update(kw)
            return {
                "algo_count": 2,
                "first_algo_id": 9,
                "workspace_bytes": 0,
                "status": "OK",
            }

    ext = _StubExt()
    r_n = probe_config(ext, 1024, 1024, 1024, out_dtype="bf16", ws_cap_bytes=0, op="N")
    assert r_n["algo_count"] == 2
    assert seen["out_dtype"] == "bf16"
    assert seen["ws_limit_bytes"] == 0
    assert seen["transa"] == "N" and seen["transb"] == "N"

    probe_config(ext, 1024, 1024, 1024, out_dtype="fp32", ws_cap_bytes=1 << 20, op="T")
    assert seen["out_dtype"] == "fp32"
    assert seen["ws_limit_bytes"] == 1 << 20
    assert seen["transa"] == "T"


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
