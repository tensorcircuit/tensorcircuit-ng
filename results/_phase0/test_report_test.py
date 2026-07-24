"""Frozen-schema test for :func:`test_report.run_tests_and_write_report`.

Task 8 Step 4: the report must have schema_version=1, command, exit_code, passed.
The test skips itself in the nested pytest run (recursion guard via
``_PHASE0_TEST_REPORT_NESTED`` env var).
"""

import json
import os

import pytest

from results._phase0.test_report import run_tests_and_write_report


@pytest.mark.skipif(
    os.environ.get("_PHASE0_TEST_REPORT_NESTED") == "1",
    reason="nested pytest run (recursion guard)",
)
def test_run_tests_and_write_report_writes_frozen_schema(tmp_path):
    """Call run_tests_and_write_report to a tmp_path, assert the JSON has
    schema_version=1, command, exit_code, passed (with the correct types)."""
    out = tmp_path / "test_report.json"
    report = run_tests_and_write_report(str(out))

    # Check the returned dict.
    assert report["schema_version"] == 1, report
    assert report["command"] == "python -m pytest results/_phase0/ -m 'not gpu'"
    assert isinstance(report["exit_code"], int), report
    assert isinstance(report["passed"], bool), report

    # Check the written file.
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1, raw
    assert raw["command"] == report["command"]
    assert raw["exit_code"] == report["exit_code"]
    assert raw["passed"] == report["passed"]
