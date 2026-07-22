"""Unit tests for C2 tile-mappability classification (review §6.2).

Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
python -m pytest results/_phase0_c2_test.py -v
"""

from results._phase0.c2 import classify_tileability, judge_c2


def test_large_regular_gemm_is_direct_tileable():
    s = {
        "M": 4096,
        "N": 4096,
        "K": 4096,
        "consumer_count": 1,
        "transpose": False,
        "bytes": 4096 * 4096 * 8,
    }
    c = classify_tileability(s)
    assert c["class"] in ("direct-gemm-tileable", "tileable-with-pack")
    assert c["global_bytes_eliminated"] > 0


def test_multi_consumer_is_not_tileable():
    s = {
        "M": 2048,
        "N": 2048,
        "K": 2048,
        "consumer_count": 4,
        "transpose": False,
        "bytes": 2048**2 * 8,
    }
    assert classify_tileability(s)["class"] == "not-tileable"


def test_judge_c2_pass_with_one_tileable_large_buffer():
    shapes = [
        {
            "M": 4096,
            "N": 4096,
            "K": 4096,
            "consumer_count": 1,
            "transpose": False,
            "bytes": 4096**2 * 8,
        }
    ]
    j = judge_c2(shapes)
    assert j["status"] == "PASS"


def test_judge_c2_unknown_when_all_unknown():
    shapes = [
        {"M": 0, "N": 0, "K": 0, "consumer_count": 1, "transpose": False, "bytes": 0}
    ]
    assert judge_c2(shapes)["status"] == "UNKNOWN"


def test_judge_c2_canonical_fail_not_feasible():
    """Region fusion that cannot reduce the executable peak (raw recomputed reduction <<
    threshold) -> canonical FAIL / prototype NOT_FEASIBLE."""
    from results._phase0.c2 import judge_c2_canonical

    edge = {"terminal_consumer_hlo_value_id": "%custom-call.498"}
    peak = {
        "arena_peak_live_bytes": 1107389712,
        "peak_after_full_PTE_fusion": 1107357584,
    }
    audit = {"allocation_source": "xla_buffer_assignment"}
    j = judge_c2_canonical(edge, peak, audit, case_id="n24_d10")
    assert j["status"] == "FAIL", j
    assert j["prototype_verdict"] == "NOT_FEASIBLE", j
    assert j["basis"] == "hlo_use_def"


def test_judge_c2_canonical_unknown_missing_peak():
    """Fail-closed: missing peak-analysis fields -> UNKNOWN (never default a verdict)."""
    from results._phase0.c2 import judge_c2_canonical

    edge = {"terminal_consumer_hlo_value_id": "%custom-call.498"}
    j = judge_c2_canonical(
        edge, {}, {"allocation_source": "xla_buffer_assignment"}, case_id="x"
    )
    assert j["status"] == "UNKNOWN", j


def test_judge_c2_canonical_unknown_no_edge():
    """Fail-closed: edge does not reach the real terminal consumer -> UNKNOWN."""
    from results._phase0.c2 import judge_c2_canonical

    edge = {"terminal_consumer_hlo_value_id": ""}  # does not reach %custom-call.498
    peak = {"arena_peak_live_bytes": 1000, "peak_after_full_PTE_fusion": 500}
    j = judge_c2_canonical(
        edge, peak, {"allocation_source": "xla_buffer_assignment"}, case_id="x"
    )
    assert j["status"] == "UNKNOWN", j


def test_run_c2_canonical_fail_not_feasible():
    """Integration: the real edge map + peak analysis + allocation audit -> canonical
    FAIL/NOT_FEASIBLE (region fusion of the anchor pair cannot reduce the structural peak).
    """
    from results._phase0.c2 import run_c2_canonical

    j = run_c2_canonical(24, 10, "default")
    assert j["basis"] == "hlo_use_def", j
    assert j["status"] == "FAIL", j
    assert j["prototype_verdict"] == "NOT_FEASIBLE", j
    assert (
        j["peak_reduction_bytes"] is not None
        and j["peak_reduction_bytes"] < 256 * 1024 * 1024
    ), j


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
