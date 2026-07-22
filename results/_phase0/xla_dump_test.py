"""File-based regression for the XLA buffer-assignment dump probe (correction Task A2).
Run: pytest results/_phase0/xla_dump_test.py -v
"""


def test_xla_dump_summary_has_jit_f_buffer_assignment():
    """The dump must yield a parseable buffer-assignment for the main expectation module
    (jit_f) -- the in-process serialized_buffer_assignment_proto is empty on GPU, so the
    --xla_dump_to path is the source of real allocation/liveness/aliasing."""
    import json
    import os

    p = "results/phase0/c1_xla_dump/n24_d10_default_summary.json"
    if not os.path.exists(p):
        import pytest

        pytest.skip(
            "xla_dump summary not generated (run python results/_phase0/xla_dump.py)"
        )
    s = json.load(open(p))
    assert s["has_parseable_buffer_assignment"], s
    assert any(
        "jit_f" in f and "buffer-assignment" in f for f in s["buffer_assignment_files"]
    ), s["buffer_assignment_files"]


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
