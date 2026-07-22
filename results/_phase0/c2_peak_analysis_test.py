"""Regression for the aliasing-aware C2 peak analysis (correction Task C, §6.5/6.6).
Run: pytest results/_phase0/c2_peak_analysis_test.py -v
"""


def test_peak_analysis_PTE_fusion_no_memory_benefit():
    """Region fusion of the anchor pair (P->T->E) cannot reduce the executable peak: the
    ~1.06GB peak is structurally set by the contraction chain of GEMM+transpose pairs, so
    eliminating P and/or T (even both) shifts the peak to another pair, not down. This
    determines C2 (memory) = NOT_FEASIBLE without building the (now-unwarranted) kernel.
    """
    from results._phase0.c2_peak_analysis import analyze

    o = analyze()
    # eliminating P alone gives ~0; eliminating P+T together is nowhere near 512 MiB
    assert o["peak_reduction_if_P_eliminated"] < 1024 * 1024, o  # < 1 MiB
    assert (
        o["peak_reduction_if_P_and_T_eliminated"] < 256 * 1024 * 1024
    ), o  # << 512 MiB
    assert o["verdict_hint"].startswith("PTE_FUSION_NO_CLEAR_MEMORY_BENEFIT"), o
    # the anchor P is NOT in the peak-live-set (XLA already aliases/schedules around it)
    assert o["P_in_peak_live_set"] is False, o


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
