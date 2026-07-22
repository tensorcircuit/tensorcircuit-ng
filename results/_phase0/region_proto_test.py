"""Regression test for the full region/tile-fusion prototype (rereview §5.3).
GPU integration: compiles the cupy.RawKernel tiled kernel and checks the full §5.3
acceptance -- memory + cost model + resources + correctness. Run:
  pytest results/_phase0/region_proto_test.py -v
"""


def test_region_full_feasible():
    from results._phase0.region_proto import run

    out = run(correctness_shape=(128, 128, 32))
    # §5.3 #1/#6 memory + no full-C materialization
    assert out["memory_feasible"], out
    assert out["no_full_c_materialized"], out
    assert out["delta_bytes"] > 400_000_000, out  # ~ the 512 MiB C buffer
    # §5.3 #2/#4 net byte gain (c64 direct fusion: pack/recompute/conv = 0)
    assert out["net_gain_positive"], out
    assert out["global_bytes_eliminated"] == out["c_buffer_bytes"], out
    # §5.3 #1 correctness: tiled fused == torch ref, no full C
    assert out["correct"], out
    assert out["rel_diff_tiled_vs_ref"] < 1e-3, out
    # §5.3 #3 resources/occupancy reported
    assert out["occupancy_pct"] > 0, out
    assert out["shared_mem_per_block_bytes"] > 0, out
    # §5.3 #5 latency branch (memory-policy OR not-worse)
    assert out["memory_policy_met"], out
    # full verdict
    assert out["verdict"] == "TILE_FUSION_FEASIBLE", out


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
