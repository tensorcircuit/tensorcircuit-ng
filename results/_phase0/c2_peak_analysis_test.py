"""Regression for the aliasing-aware C2 peak analysis + generalized peak frontier
(final-remediation Task 3). Run: pytest results/_phase0/c2_peak_analysis_test.py -v

The legacy ``analyze()`` (correction Task C) stays as a gate-compat shim until Task 5
rewires the gate to the new ``analyze_frontier()`` artifacts. Task 3 adds a de-hardcoded,
physical-union, multi-window single+joint peak frontier that emits three diagnostics and
NO canonical C2 verdict.
"""

import json
import os

# --- legacy: gate-compat shim (unchanged) ---


def test_peak_analysis_PTE_fusion_no_memory_benefit():
    """Region fusion of the anchor pair (P->T->E) cannot reduce the executable peak: the
    ~1.06GB peak is structurally set by the contraction chain of GEMM+transpose pairs, so
    eliminating P and/or T (even both) shifts the peak to another pair, not down.
    """
    from results._phase0.c2_peak_analysis import analyze

    o = analyze()
    assert o["peak_reduction_if_P_eliminated"] < 1024 * 1024, o  # < 1 MiB
    assert (
        o["peak_reduction_if_P_and_T_eliminated"] < 256 * 1024 * 1024
    ), o  # << 512 MiB
    assert o["verdict_hint"].startswith("PTE_FUSION_NO_CLEAR_MEMORY_BENEFIT"), o
    assert o["P_in_peak_live_set"] is False, o


# --- Task 3: physical-union global sweep (no naive sum) ---


def _v(name, alloc, offset, size, b, d, si="0"):
    return {
        "name": name,
        "si": si,
        "alloc_id": alloc,
        "offset": offset,
        "size": size,
        "birth": b,
        "death": d,
        "key": (name, si),
    }


def test_physical_union_does_not_double_count_aliased_ranges():
    """Two co-live values at the SAME offset (alias/in-place) count once, not twice."""
    from results._phase0.c2_peak_analysis import program_peak_physical

    vals = [
        _v("a", 1, 0, 100, 0, 10),
        _v("b", 1, 0, 100, 2, 8),  # aliased to a, co-live during [2,8]
        _v("c", 2, 0, 50, 0, 10),
    ]
    peak, _t, live = program_peak_physical(vals)
    # at [2,8]: alloc1 union([0,100)) = 100, alloc2 = 50 -> 150 (NOT 250)
    assert peak == 150, (peak, live)


def test_physical_union_sums_disjoint_ranges_same_alloc():
    """Two co-live values at DISJOINT offsets in one allocation do sum."""
    from results._phase0.c2_peak_analysis import program_peak_physical

    vals = [
        _v("a", 1, 0, 100, 0, 10),
        _v("b", 1, 100, 100, 0, 10),  # disjoint range
    ]
    peak, _t, _live = program_peak_physical(vals)
    assert peak == 200, peak


def test_peak_excluding_drops_only_named_values():
    from results._phase0.c2_peak_analysis import peak_excluding, program_peak_physical

    vals = [
        _v("a", 1, 0, 100, 0, 10),
        _v("b", 1, 0, 100, 2, 8),
        _v("c", 2, 0, 50, 0, 10),
    ]
    base = program_peak_physical(vals)[0]
    assert base == 150
    # removing only "a" leaves "b" covering [0,100) -> peak unchanged
    assert peak_excluding(vals, {("a", "0")}) == 150
    # removing both aliased values empties alloc1 -> only alloc2's 50 remains
    assert peak_excluding(vals, {("a", "0"), ("b", "0")}) == 50


# --- Task 3: de-hardcoded frontier on the real n=24 case ---


def test_frontier_anchor_window_derived_from_edge_map():
    """The anchor window's producer/consumer come from the v2 edge map (not literals), and
    single-patch elimination is far below the 256 MiB threshold (~structural peak)."""
    from results._phase0.c2_peak_analysis import analyze_frontier

    o = analyze_frontier()
    anchor = o["anchor_window"]
    assert anchor["producer_id"] == "%custom-call.497", anchor
    assert anchor["consumer_id"] == "%custom-call.498", anchor
    assert anchor["single_reduction_bytes"] < 256 * 1024 * 1024, anchor
    # the anchor allocation id is read from the audit, not a literal
    with open("results/phase0/c1_buffer_assignment/n24_d10_default.json") as fh:
        audit = json.load(fh)
    audit_anchor_alloc = next(
        b["allocation_id"] for b in audit["buffers"] if b.get("is_anchor")
    )
    assert anchor["producer_allocation_id"] == audit_anchor_alloc, anchor


def test_frontier_enumerates_multiple_large_windows():
    """The anchor is only ONE of several isomorphic 512 MiB GEMM+transpose windows."""
    from results._phase0.c2_peak_analysis import analyze_frontier

    o = analyze_frontier()
    wins = o["windows"]
    assert len(wins) >= 2, [w["producer_id"] for w in wins]
    assert any(w["producer_id"] == "%custom-call.497" for w in wins), [
        w["producer_id"] for w in wins
    ]


def test_frontier_emits_three_diagnostics_and_no_canonical_verdict():
    from results._phase0.c2_peak_analysis import analyze_frontier

    o = analyze_frontier()
    diag = o["diagnostics"]
    assert "single_anchor_patch_status" in diag, diag
    assert "joint_model_status" in diag, diag
    assert diag["kernel_feasibility_status"] == "UNKNOWN", diag
    blob = json.dumps(o)
    assert "GO_TO_PHASE1" not in blob
    assert "C2_CANONICAL" not in blob  # no canonical verdict from Task 3


def test_frontier_joint_model_reports_reduction_and_min_cover():
    from results._phase0.c2_peak_analysis import analyze_frontier

    o = analyze_frontier()
    jm = o["joint_model"]
    assert "max_joint_reduction_bytes" in jm, jm
    assert "min_cover_by_target" in jm, jm  # per-threshold smallest cover set
    assert "model_assumptions" in o, o
    assert o["model_assumptions"], o["model_assumptions"]


def test_frontier_artifacts_written():
    from results._phase0.c2_peak_analysis import analyze_frontier

    analyze_frontier()
    assert os.path.exists("results/phase0/c2_peak_frontier.json")
    assert os.path.exists("results/phase0/c2_peak_windows.csv")
    with open("results/phase0/c2_peak_windows.csv") as fh:
        header = fh.readline().strip()
    assert "producer_id" in header and "single_reduction_bytes" in header, header


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
