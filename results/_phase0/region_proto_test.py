"""Regression test for the region/tile-fusion prototype (Task 3 minimal viable subset).
GPU integration: compiles the cupy.RawKernel kernels and checks the memory-feasibility +
correctness properties. Run: pytest results/_phase0/region_proto_test.py -v
"""


def test_region_memory_feasible_and_correct():
    from results._phase0.region_proto import run

    out = run(correctness_shape=(128, 128, 32))
    # The 512 MiB C buffer is avoidable: fused peak < materialized peak by ~C.
    assert out["memory_feasible"], out
    assert out["fused_peak_bytes"] < out["materialized_peak_bytes"], out
    assert out["delta_bytes"] > 400_000_000, out  # ~ the 512 MiB C buffer
    # The fused (no-C) kernel computes the same result as torch reference.
    assert out["correct"], out
    assert out["rel_diff_fused_vs_ref"] < 1e-3, out
    assert out["verdict"] == "TILE_FUSION_MEMORY_FEASIBLE", out


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
