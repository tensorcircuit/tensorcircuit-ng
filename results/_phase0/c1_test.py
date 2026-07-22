"""Unit tests for C1 four-condition judgment (review §5.4). Run: pytest results/_phase0_c1_test.py -v"""

from results._phase0.c1 import judge_c1


def test_c1_pass_when_all_conditions_met():
    r = {"runtime_peak_B": 2**24 * 8, "full_state_bytes": 2**24 * 8}  # 1.0x state
    j = judge_c1(
        default_result=r,
        nofusion_result=r,
        repeats_results=[r, r, r],
        materialized_buffer_bytes=2**24 * 8,
        optimized_hlo_has_materialized=True,
    )
    assert j["status"] == "PASS", j


def test_c1_fail_when_buffer_below_half_state():
    r = {"runtime_peak_B": 1000, "full_state_bytes": 2**24 * 8}
    j = judge_c1(
        default_result=r,
        nofusion_result=r,
        repeats_results=[r, r, r],
        materialized_buffer_bytes=1000,
        optimized_hlo_has_materialized=True,
    )
    assert j["status"] == "FAIL"
    assert "0.5" in j["reason"] or "threshold" in j["reason"].lower()


def test_c1_unknown_when_repeats_unstable():
    r = {"runtime_peak_B": 2**24 * 8, "full_state_bytes": 2**24 * 8}
    unstable = [
        {"runtime_peak_B": 2**24 * 8},
        {"runtime_peak_B": 1000},
        {"runtime_peak_B": 2**24 * 8},
    ]
    j = judge_c1(
        default_result=r,
        nofusion_result=r,
        repeats_results=unstable,
        materialized_buffer_bytes=2**24 * 8,
        optimized_hlo_has_materialized=True,
    )
    assert j["status"] == "UNKNOWN"


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
