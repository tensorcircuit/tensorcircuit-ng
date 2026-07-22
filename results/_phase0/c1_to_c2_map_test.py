"""Tests for the C1 anchor -> real contraction consumer edge map (correction-plan Task B).

The production region is a TWO-STAGE GEMM; the mapper must PIERCE layout-only fusions
(bitcast/transpose/reshape bodies) to reach the true terminal contraction consumer, not
stop at the first fusion.
"""

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


def test_classify_layout_fusion_is_passthrough():
    from results._phase0.c1_to_c2_map import _build_computation_bodies, _classify_fusion

    bodies = _build_computation_bodies(SYNTH_LAYOUT_FUSION)
    assert _classify_fusion("%layout_comp", bodies) == "layout_passthrough"


def test_classify_compute_fusion_is_terminal():
    from results._phase0.c1_to_c2_map import _build_computation_bodies, _classify_fusion

    bodies = _build_computation_bodies(SYNTH_COMPUTE_FUSION)
    assert _classify_fusion("%compute_comp", bodies) == "compute_consumer"


def test_layout_fusion_is_pierced():
    from results._phase0.c1_to_c2_map import build_c1_edge_map

    rec = build_c1_edge_map(SYNTH_LAYOUT_FUSION, "%cc")[0]
    # pierces the layout fusion + bitcast; terminal is the add sink
    assert "gte" in rec["passthrough_hlo_ids"]
    assert "lf" in rec["passthrough_hlo_ids"]
    assert "bc" in rec["passthrough_hlo_ids"]
    assert rec["terminal_consumer_hlo_value_id"] == "%sink"


def test_compute_fusion_is_terminal():
    from results._phase0.c1_to_c2_map import build_c1_edge_map

    rec = build_c1_edge_map(SYNTH_COMPUTE_FUSION, "%cc")[0]
    # compute fusion stops the trace; it IS the terminal consumer
    assert "gte" in rec["passthrough_hlo_ids"]
    assert rec["terminal_consumer_hlo_value_id"] == "%cf"


def test_map_anchor_for_case_real_hlo_two_stage_region():
    """Real n=24 HLO: the anchor's true terminal consumer is the second GEMM .498,
    reached by piercing the layout fusion (loop_transpose_fusion.2 -> fused_transpose.2).
    """
    from results._phase0.c1_to_c2_map import map_anchor_for_case

    rec = map_anchor_for_case(24, 10, "default")
    assert rec["producer_hlo_value_id"] == "%custom-call.497", rec
    assert (rec["producer_M"], rec["producer_N"], rec["producer_K"]) == (
        4096,
        16384,
        1024,
    ), rec
    # the layout fusion + its operands are PIERCED (passthrough), not terminal
    pt = rec["passthrough_hlo_ids"]
    assert "get-tuple-element.246.0" in pt, pt
    assert "loop_transpose_fusion.2" in pt, pt
    assert "bitcast.1317.0" in pt, pt
    # the TRUE terminal consumer is the second GEMM .498, not the layout fusion
    assert rec["terminal_consumer_hlo_value_id"] == "%custom-call.498", rec
    # E = D[64,64] @ T[64,1048576] -> c64[64,1048576] (another 512 MiB output)
    assert (rec["consumer_M"], rec["consumer_N"], rec["consumer_K"]) == (
        64,
        1048576,
        64,
    ), rec
    assert rec["consumer_output_bytes"] == 64 * 1048576 * 8, rec


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
