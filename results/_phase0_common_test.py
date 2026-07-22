"""Unit tests for _phase0_common pure logic. Run: pytest results/_phase0_common_test.py -v
or: python results/_phase0_common_test.py"""

from results._phase0_common import (
    worker_emit,
    parse_last_json,
    classify_stderr,
    fmt_table,
    median_wall_ms,
)
import io, sys, json


def test_worker_emit_one_json_line(capsys):
    worker_emit({"a": 1, "b": "x"})
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert json.loads(out[0]) == {"a": 1, "b": "x"}


def test_parse_last_json_finds_last_object_line():
    stdout = 'noise line\n{"x": 1}\nprogress 50\n{"x": 2}\n'
    assert parse_last_json(stdout) == {"x": 2}


def test_parse_last_json_none_when_absent():
    assert parse_last_json("no json here\n") is None


def test_classify_stderr_oom():
    assert (
        classify_stderr("RuntimeError: CUDA out of memory. Tried to allocate") == "oom"
    )


def test_classify_stderr_int32():
    assert classify_stderr("INVALID_ARGUMENT: int32 overflow") == "crash-int32"


def test_classify_stderr_compile():
    assert classify_stderr("XLA compilation failed: shape mismatch") == "crash-compile"


def test_classify_stderr_unknown():
    assert classify_stderr("some other error") == "crash"


def test_classify_stderr_ok():
    assert classify_stderr("") == "ok"


def test_fmt_table_aligns_columns():
    table = fmt_table(["name", "val"], [["a", 1], ["bbb", 22]])
    lines = table.splitlines()
    assert lines[0].startswith("name")
    assert "bbb" in lines[3]  # header(0) separator(1) row0(2) row1(3)


def test_median_wall_ms_returns_positive():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return calls["n"]

    ms = median_wall_ms(fn, warmup=1, iters=3, sync=None)
    assert ms >= 0.0
    assert calls["n"] == 4  # 1 warmup + 3 iters


def test_orchestrate_respects_worker_crash_outcome(tmp_path):
    """A worker that exits 0 but emits {"outcome":"crash"} must be ok=False (review §4.1)."""
    from results._phase0_common import orchestrate

    script = tmp_path / "w.py"
    script.write_text(
        "import sys,json\n"
        "sys.stdout.write(json.dumps({'outcome':'crash','error':'boom'})+'\\n')\n"
        "sys.stdout.flush()\n"
    )
    rows = orchestrate([{"id": 1}], lambda c: [], str(script), timeout=30)
    assert rows[0]["ok"] is False
    assert rows[0]["outcome"] == "crash"


def test_orchestrate_respects_worker_oom_outcome(tmp_path):
    """Same fix must cover other worker-reported outcomes (e.g. oom), not just 'crash'."""
    from results._phase0_common import orchestrate

    script = tmp_path / "w.py"
    script.write_text(
        "import sys,json\n"
        "sys.stdout.write(json.dumps({'outcome':'oom','error':'cuda oom'})+'\\n')\n"
        "sys.stdout.flush()\n"
    )
    rows = orchestrate([{"id": 1}], lambda c: [], str(script), timeout=30)
    assert rows[0]["ok"] is False
    assert rows[0]["outcome"] == "oom"


def test_orchestrate_run_outcome_still_ok(tmp_path):
    """A genuine {'outcome':'run',...} with exit 0 must still map to ok=True."""
    from results._phase0_common import orchestrate

    script = tmp_path / "w.py"
    script.write_text(
        "import sys,json\n"
        "sys.stdout.write(json.dumps({'outcome':'run','peak_B':123})+'\\n')\n"
        "sys.stdout.flush()\n"
    )
    rows = orchestrate([{"id": 1}], lambda c: [], str(script), timeout=30)
    assert rows[0]["ok"] is True
    assert rows[0]["outcome"] == "run"
    assert rows[0]["result"]["peak_B"] == 123


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
