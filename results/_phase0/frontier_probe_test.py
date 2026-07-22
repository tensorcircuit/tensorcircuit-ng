"""Unit tests for Probe 2 pure logic. Run: pytest results/_phase0_frontier_probe_test.py -v"""

from results._phase0.frontier_probe import (
    build_configs,
    summarize_frontier,
    run_output_kind,
)


def test_build_configs_smoke_has_brickwork_state_jax():
    cfgs = build_configs("smoke")
    kinds = {(c["circuit"], c["output"], c["backend"]) for c in cfgs}
    assert ("brickwork", "state", "jax") in kinds
    # smoke 必须小（< 20 configs），保证快速冒烟
    assert len(cfgs) < 20


def test_build_configs_full_covers_outputs_and_depths():
    cfgs = build_configs("full")
    outputs = {c["output"] for c in cfgs}
    depths = {c["depth"] for c in cfgs}
    assert {"state", "expectation", "norm"} <= outputs
    assert {3, 10, 16, 24} <= depths


def test_summarize_frontier_finds_boundary():
    rows = [
        {
            "config": {"output": "state", "backend": "jax", "n": 18},
            "ok": True,
            "outcome": "run",
            "result": {"peak_B": 2 * 2**18 * 8},
        },
        {
            "config": {"output": "state", "backend": "jax", "n": 26},
            "ok": False,
            "outcome": "oom",
        },
    ]
    summary = summarize_frontier(rows)
    key = ("state", "jax")
    assert key in summary
    assert summary[key]["max_run_n"] == 18
    assert summary[key]["min_fail_n"] == 26


def test_run_output_kind_is_known_token():
    assert run_output_kind("state") in ("state", "expectation", "norm")


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
