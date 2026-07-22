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


def test_judge_c2_canonical_pass():
    from results._phase0.c2 import judge_c2_canonical

    edge = {
        "consumer_count": 1,
        "buffer_bytes": 4096 * 16384 * 8,
        "hlo_value_id": "%custom-call.497",
    }
    proto = {
        "verdict": "TILE_FUSION_FEASIBLE",
        "net_gain_positive": True,
        "correct": True,
        "no_full_c_materialized": True,
        "memory_feasible": True,
        "memory_policy_met": True,
    }
    j = judge_c2_canonical(edge, proto)
    assert j["status"] == "PASS", j
    assert j["basis"] == "hlo_use_def"


def test_judge_c2_canonical_unknown_no_edge():
    from results._phase0.c2 import judge_c2_canonical

    proto = {
        "verdict": "TILE_FUSION_FEASIBLE",
        "net_gain_positive": True,
        "correct": True,
        "no_full_c_materialized": True,
        "memory_feasible": True,
        "memory_policy_met": True,
    }
    j = judge_c2_canonical({"consumer_count": 0, "buffer_bytes": 0}, proto)
    assert j["status"] == "UNKNOWN", j


def test_judge_c2_canonical_fail_not_feasible():
    from results._phase0.c2 import judge_c2_canonical

    edge = {"consumer_count": 1, "buffer_bytes": 4096 * 16384 * 8}
    j = judge_c2_canonical(edge, {"verdict": "NOT_FEASIBLE"})
    assert j["status"] == "FAIL", j


def test_run_c2_canonical_integration_pass():
    """File-based: reads Task 2 edge_map.csv + Task 3 region_prototype.json -> canonical PASS."""
    import json

    from results._phase0.c2 import run_c2_canonical

    j = run_c2_canonical(24, 10, "default")
    assert j["basis"] == "hlo_use_def", j
    assert j["status"] == "PASS", j
    with open("results/phase0/c2_judgment.json") as fh:
        d = json.load(fh)
    assert d["n24_d10"]["basis"] == "hlo_use_def"
    assert d["n24_d10"]["status"] == "PASS"


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
