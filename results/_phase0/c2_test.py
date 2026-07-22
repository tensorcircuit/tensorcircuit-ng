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


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
