"""Tests for the C1 anchor -> real contraction consumer edge map (final-remediation Task 2).

The production region is a TWO-STAGE GEMM; the mapper must PIERCE layout-only fusions
(bitcast/transpose/reshape bodies) to reach the true terminal contraction consumer, and
emit the v2 schema (spec `2026-07-22-phase0-final-review-spec.md` section 6): an exact,
invertible index transform plus producer/consumer shape/dtype/layout/bytes and hash binding.
"""

import numpy as np

# --- synthetic fixtures for fusion classification ---

# Layout-only fusion body (parameter + transpose) -> the fusion is a passthrough.
SYNTH_LAYOUT_FUSION = """
%p = c64[4,4] parameter(0)
%cc = (c64[4,4]{1,0}, s8[1]{0}) custom-call(%p, %p), custom_call_target="__cublas$gemm"
%gte = c64[4,4] get-tuple-element(%cc), index=0
%lf = c64[4,4] fusion(%gte), kind=kLoop, calls=%layout_comp
%bc = c64[4,4] bitcast(%lf)
ROOT %sink = c64[4,4] add(%bc, %bc)
%layout_comp (x: c64[4,4]) -> c64[4,4] {
  %x = c64[4,4] parameter(0)
  ROOT %t = c64[4,4] transpose(%x), dimensions={1,0}
}
"""

# Compute fusion body (add) -> the fusion is a terminal compute consumer.
SYNTH_COMPUTE_FUSION = """
%p = c64[4,4] parameter(0)
%cc = (c64[4,4]{1,0}, s8[1]{0}) custom-call(%p, %p), custom_call_target="__cublas$gemm"
%gte = c64[4,4] get-tuple-element(%cc), index=0
%cf = c64[4,4] fusion(%gte), kind=kLoop, calls=%compute_comp
ROOT %sink = c64[4,4] add(%cf, %cf)
%compute_comp (x: c64[4,4]) -> c64[4,4] {
  %x = c64[4,4] parameter(0)
  ROOT %t = c64[4,4] add(%x, %x)
}
"""

# Fusion body with a convert (dtype-changing) op -> NOT pure layout -> terminal / ambiguous.
SYNTH_CONVERT_FUSION = """
%p = c64[4,4] parameter(0)
%cc = (c64[4,4]{1,0}, s8[1]{0}) custom-call(%p, %p), custom_call_target="__cublas$gemm"
%gte = c64[4,4] get-tuple-element(%cc), index=0
%cf = c64[4,4] fusion(%gte), kind=kLoop, calls=%convert_comp
ROOT %sink = c64[4,4] add(%cf, %cf)
%convert_comp (x: c64[4,4]) -> f32[4,4] {
  %x = c64[4,4] parameter(0)
  ROOT %t = f32[4,4] convert(%x)
}
"""

# Two terminal compute consumers of the anchor data -> AMBIGUOUS.
SYNTH_TWO_CONSUMERS = """
%p = c64[4,4] parameter(0)
%cc = (c64[4,4]{1,0}, s8[1]{0}) custom-call(%p, %p), custom_call_target="__cublas$gemm"
%gte = c64[4,4] get-tuple-element(%cc), index=0
ROOT %sink1 = c64[4,4] add(%gte, %gte)
%extra = c64[4,4] add(%gte, %gte)
"""

# A small bitcast->transpose->bitcast transform (32 elements) mirroring the real region.
SMALL_STEPS = [
    {
        "op": "bitcast",
        "shape_in": [4, 8],
        "layout_in": [1, 0],
        "shape_out": [2, 2, 4, 2],
        "layout_out": [3, 2, 1, 0],
    },
    {
        "op": "transpose",
        "dimensions": [2, 0, 3, 1],
        "shape_in": [2, 2, 4, 2],
        "layout_in": [3, 2, 1, 0],
        "shape_out": [4, 2, 2, 2],
        "layout_out": [3, 2, 1, 0],
    },
    {
        "op": "bitcast",
        "shape_in": [4, 2, 2, 2],
        "layout_in": [3, 2, 1, 0],
        "shape_out": [4, 8],
        "layout_out": [1, 0],
    },
]


def test_classify_layout_fusion_is_passthrough():
    from results._phase0.c1_to_c2_map import _build_computation_bodies, _classify_fusion

    bodies = _build_computation_bodies(SYNTH_LAYOUT_FUSION)
    assert _classify_fusion("%layout_comp", bodies) == "layout_passthrough"


def test_classify_compute_fusion_is_terminal():
    from results._phase0.c1_to_c2_map import _build_computation_bodies, _classify_fusion

    bodies = _build_computation_bodies(SYNTH_COMPUTE_FUSION)
    assert _classify_fusion("%compute_comp", bodies) == "compute_consumer"


def test_classify_convert_fusion_is_not_layout():
    """A convert (dtype-changing) op in the body disqualifies pure-layout classification."""
    from results._phase0.c1_to_c2_map import _build_computation_bodies, _classify_fusion

    bodies = _build_computation_bodies(SYNTH_CONVERT_FUSION)
    assert _classify_fusion("%convert_comp", bodies) != "layout_passthrough"


def test_layout_fusion_is_pierced_to_terminal_v2():
    from results._phase0.c1_to_c2_map import build_c1_edge_map

    rec = build_c1_edge_map(SYNTH_LAYOUT_FUSION, "%cc")
    # pierces the layout fusion + bitcast; terminal consumer is the add sink
    hlo_ids = rec["transform"]["hlo_ids"]
    assert "gte" in hlo_ids, hlo_ids
    assert "lf" in hlo_ids, hlo_ids
    assert "bc" in hlo_ids, hlo_ids
    assert rec["consumer"]["hlo_value_id"] == "%sink", rec
    assert rec["consumer_count"] == 1, rec
    assert rec["trace_status"] == "EXACT", rec


def test_compute_fusion_is_terminal_v2():
    from results._phase0.c1_to_c2_map import build_c1_edge_map

    rec = build_c1_edge_map(SYNTH_COMPUTE_FUSION, "%cc")
    # compute fusion stops the trace; it IS the terminal consumer
    assert "gte" in rec["transform"]["hlo_ids"], rec
    assert rec["consumer"]["hlo_value_id"] == "%cf", rec


def test_convert_fusion_is_terminal_and_ambiguous():
    """A convert fusion must not be auto-pierced as pure layout; trace is AMBIGUOUS."""
    from results._phase0.c1_to_c2_map import build_c1_edge_map

    rec = build_c1_edge_map(SYNTH_CONVERT_FUSION, "%cc")
    assert rec["consumer"]["hlo_value_id"] == "%cf", rec
    assert rec["trace_status"] == "AMBIGUOUS", rec


def test_multiple_terminal_consumers_are_ambiguous():
    from results._phase0.c1_to_c2_map import build_c1_edge_map

    rec = build_c1_edge_map(SYNTH_TWO_CONSUMERS, "%cc")
    assert rec["consumer_count"] == 2, rec
    assert rec["trace_status"] == "AMBIGUOUS", rec


def test_transform_roundtrip_is_elementwise_inverse_on_small_shape():
    """forward then inverse must reproduce every element; forward must be a permutation."""
    from results._phase0.c1_to_c2_map import (
        _linear_permutation,
        apply_forward,
        apply_inverse,
    )

    n = int(np.prod(SMALL_STEPS[0]["shape_in"]))
    assert int(np.prod(SMALL_STEPS[-1]["shape_out"])) == n  # element-count preserving

    forward, inverse = _linear_permutation(SMALL_STEPS)
    # forward is a permutation of [0, n)
    assert sorted(int(x) for x in forward) == list(range(n)), forward
    # inverse is the true inverse permutation of forward
    assert np.array_equal(forward[inverse], np.arange(n)), (forward, inverse)

    p_flat = np.arange(n, dtype=np.int64) * 7 + 3  # distinct values
    t_flat = apply_forward(SMALL_STEPS, p_flat)
    assert t_flat.shape == (n,)
    p_back = apply_inverse(SMALL_STEPS, t_flat)
    assert np.array_equal(p_back, p_flat)  # elementwise inverse round-trip


def test_real_transform_steps_match_hlo_literal():
    """The parsed transform steps must equal the literal fused_transpose.2 + external bitcast."""
    rec = _load_real_edge()
    steps = rec["transform"]["steps"]
    assert [s["op"] for s in steps] == ["bitcast", "transpose", "bitcast"], steps
    # step 1: bitcast P[4096,16384]{1,0} -> [2,2,4,256,2,2,2,2048]{7..0}
    assert steps[0]["shape_in"] == [4096, 16384] and steps[0]["layout_in"] == [
        1,
        0,
    ], steps[0]
    assert steps[0]["shape_out"] == [2, 2, 4, 256, 2, 2, 2, 2048], steps[0]
    assert steps[0]["layout_out"] == [7, 6, 5, 4, 3, 2, 1, 0], steps[0]
    # step 2: transpose dimensions={2,1,0,4,6,3,5,7}
    assert steps[1]["dimensions"] == [2, 1, 0, 4, 6, 3, 5, 7], steps[1]
    assert steps[1]["shape_out"] == [4, 2, 2, 2, 2, 256, 2, 2048], steps[1]
    # step 3: bitcast -> [64,1048576]{1,0}
    assert steps[2]["shape_out"] == [64, 1048576] and steps[2]["layout_out"] == [
        1,
        0,
    ], steps[2]


def test_real_edge_v2_schema():
    rec = _load_real_edge()
    assert rec["schema_version"] == "c1-c2-edge-v2", rec
    assert rec["case_id"] == "n24_d10_default", rec
    assert (rec["n"], rec["depth"], rec["fusion"]) == (24, 10, "default"), rec

    assert rec["source_hlo"]["sha256"] and len(rec["source_hlo"]["sha256"]) == 64, rec
    assert (
        rec["allocation_audit"]["sha256"]
        and len(rec["allocation_audit"]["sha256"]) == 64
    ), rec

    p = rec["producer"]
    assert p["hlo_value_id"] == "%custom-call.497", p
    assert p["result_index"] == 0, p
    assert (
        p["dtype"] == "c64" and p["shape"] == [4096, 16384] and p["layout"] == [1, 0]
    ), p
    assert (p["M"], p["N"], p["K"]) == (4096, 16384, 1024), p
    assert p["bytes"] == 536870912, p

    c = rec["consumer"]
    assert c["hlo_value_id"] == "%custom-call.498", c
    assert (
        c["dtype"] == "c64" and c["shape"] == [64, 1048576] and c["layout"] == [1, 0]
    ), c
    assert (c["M"], c["N"], c["K"]) == (64, 1048576, 64), c
    assert c["bytes"] == 536870912, c

    assert rec["consumer_count"] == 1, rec
    assert rec["trace_status"] == "EXACT", rec
    assert rec["transform"]["forward_index_map"], rec["transform"]
    assert rec["transform"]["inverse_index_map"], rec["transform"]
    assert rec["transform"]["output_shape"] == [64, 1048576], rec["transform"]


def test_real_edge_provenance_stale_on_hash_mismatch():
    from results._phase0.c1_to_c2_map import AUDIT_DIR, HLO_DIR, verify_provenance

    rec = _load_real_edge()
    with open(f"{HLO_DIR}/n24_d10_exp_default.hlo") as fh:
        hlo_text = fh.read()
    with open(f"{AUDIT_DIR}/n24_d10_default.json") as fh:
        audit_text = fh.read()
    assert verify_provenance(rec, hlo_text=hlo_text, audit_text=audit_text) == "FRESH"
    assert (
        verify_provenance(rec, hlo_text=hlo_text + "\n//tainted", audit_text=audit_text)
        == "STALE_HLO"
    )
    assert (
        verify_provenance(rec, hlo_text=hlo_text, audit_text=audit_text + "\n//tainted")
        == "STALE_AUDIT"
    )


def _load_real_edge():
    """Map the real n=24 case and return its v2 edge record (regenerates the artifact)."""
    from results._phase0.c1_to_c2_map import map_anchor_for_case

    return map_anchor_for_case(24, 10, "default")


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
