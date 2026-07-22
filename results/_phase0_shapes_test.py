"""Unit tests for real contraction shape export from the cotengra tree (review §6.1, Task 6).

Run: pytest results/_phase0_shapes_test.py -v
"""

from results._phase0_shapes import export_shapes_from_eq


def test_export_shapes_two_node_einsum():
    # 'ab,bc->ac' : A(4,8) B(8,4) -> C(4,4); the single GEMM contracts the 'b' bond (K=8),
    # leaving the 'a' and 'c' bonds as M=4, N=4 (M*N=16).
    shapes = export_shapes_from_eq("ab,bc->ac", size_dict={"a": 4, "b": 8, "c": 4})
    assert len(shapes) >= 1
    s = shapes[-1]
    assert s["K"] == 8
    assert s["M"] * s["N"] == 16
    assert s["consumer_count"] >= 1


def test_export_shapes_three_node_einsum_has_two_steps():
    # 'ab,bc,cd->ad' : three tensors, two contractions. The walk must emit >= 2 steps and
    # each step must carry the required keys (sanity check on the schema).
    shapes = export_shapes_from_eq(
        "ab,bc,cd", size_dict={"a": 2, "b": 4, "c": 8, "d": 2}
    )
    assert len(shapes) >= 2
    required_keys = {
        "node_id",
        "producer_ids",
        "consumer_ids",
        "modes",
        "extents",
        "M",
        "N",
        "K",
        "batch",
        "transpose",
        "strides",
        "bytes",
        "consumer_count",
        "live_range",
    }
    for s in shapes:
        assert required_keys.issubset(s.keys()), sorted(s.keys())


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
