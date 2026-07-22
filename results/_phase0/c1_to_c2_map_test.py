"""Tests for the C1 anchor -> HLO SSA producer/consumer edge map (rereview §5.2)."""

SYNTH_HLO = """
%p.a = c64[4096,1024]{1,0} parameter(0)
%p.b = c64[1024,16384]{1,0} parameter(1)
%custom-call.497 = (c64[4096,16384]{1,0}, s8[33554432]{0}) custom-call(%p.a, %p.b), custom_call_target="__cublas$gemm", api=3
%get-tuple-element.5 = c64[4096,16384]{1,0} get-tuple-element(%custom-call.497), index=0
%bitcast.5337 = c64[2,2,4,256,2,2,2,2048]{7,6,5,4,3,2,1,0} bitcast(%get-tuple-element.5)
ROOT %fused_transpose.2 = c64[4,2,2,2,2,256,2,2048]{7,6,5,4,3,2,1,0} fusion(%bitcast.5337), kind=kLoop, calls=%fused_transpose.2
"""


def test_mnk_from_operands():
    """K is derived from the custom-call operand shapes (A=[M,K], B=[K,N])."""
    from results._phase0.c1_to_c2_map import _mnk_from_custom_call

    M, N, K = _mnk_from_custom_call(SYNTH_HLO, "%custom-call.497")
    assert (M, N, K) == (4096, 16384, 1024)


def test_build_edge_map_finds_anchor_consumer():
    """The anchor's real consumer is recovered by tracing SSA use-def through
    passthrough ops (get-tuple-element, bitcast) to a terminal consumer (fusion)."""
    from results._phase0.c1_to_c2_map import build_c1_edge_map

    edges = build_c1_edge_map(SYNTH_HLO, "%custom-call.497")
    assert len(edges) == 1, edges
    e = edges[0]
    assert e["hlo_value_id"] == "%custom-call.497"
    assert (e["M"], e["N"], e["K"]) == (4096, 16384, 1024)
    assert e["buffer_bytes"] == 4096 * 16384 * 8  # 512 MiB
    assert e["consumer_count"] == 1
    assert any("fused_transpose.2" in c for c in e["consumer_ops"])
    assert "get-tuple-element.5" in e["traced_through"]
    assert "bitcast.5337" in e["traced_through"]


def test_map_anchor_for_case_real_hlo():
    """Integration over the real n=24/d=10/default HLO + Task 1 audit JSON.
    File-based (no GPU): the production anchor %custom-call.497 must map to >=1 consumer.
    """
    from results._phase0.c1_to_c2_map import map_anchor_for_case

    e = map_anchor_for_case(24, 10, "default")
    assert e["hlo_value_id"] == "%custom-call.497", e
    assert (e["M"], e["N"]) == (4096, 16384), e
    assert e["buffer_bytes"] == 4096 * 16384 * 8, e
    assert e["consumer_count"] >= 1, e


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
