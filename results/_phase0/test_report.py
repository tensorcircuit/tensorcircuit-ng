"""Frozen-schema stdlib pytest wrapper for the self-validating release gate.

Plan Task 8. Runs ``python -m pytest results/_phase0/ -m 'not gpu'`` via
subprocess (cwd = repo root) and writes a frozen-schema report::

    {"schema_version": 1,
     "command": "python -m pytest results/_phase0/ -m 'not gpu'",
     "exit_code": <int>,
     "passed": <bool: exit_code == 0>}

stdlib only (``json`` / ``subprocess`` / ``pathlib``).

Recursion guard: the inner pytest run inherits the ``_PHASE0_TEST_REPORT_NESTED``
environment variable so the wrapper's own test skips itself inside the nested
run (otherwise the test would recurse infinitely).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

#: Frozen schema version.
SCHEMA_VERSION = 1

#: The frozen command string recorded in the report.
COMMAND = "python -m pytest results/_phase0/ -m 'not gpu'"

#: Environment variable set during the nested pytest run so the wrapper's own
#: test can skip itself (prevents infinite recursion).
_NESTED_ENV_VAR = "_PHASE0_TEST_REPORT_NESTED"


def run_tests_and_write_report(out_path):
    """Run the frozen pytest command and write the report to *out_path*.

    Runs ``python -m pytest results/_phase0/ -m 'not gpu'`` via subprocess
    (inheriting the current working directory, which is the repo root when
    invoked from the release gate). The subprocess inherits a copy of the
    environment with ``_PHASE0_TEST_REPORT_NESTED=1`` so the wrapper's own
    test skips itself inside the nested run.

    Writes ``{"schema_version": 1, "command": ..., "exit_code": <int>,
    "passed": <bool>}`` to *out_path* and returns the same dict.
    """
    env = os.environ.copy()
    env[_NESTED_ENV_VAR] = "1"
    result = subprocess.run(
        ["python", "-m", "pytest", "results/_phase0/", "-m", "not gpu"],
        capture_output=True,
        env=env,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "command": COMMAND,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
    }
    Path(out_path).write_text(json.dumps(report), encoding="utf-8")
    return report


__all__ = ["SCHEMA_VERSION", "COMMAND", "run_tests_and_write_report"]
