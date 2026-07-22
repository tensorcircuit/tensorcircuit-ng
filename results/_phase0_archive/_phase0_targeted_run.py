"""Throwaway targeted-frontier driver: reuses Probe 2 infra to run a focused subset at larger n.
Brickwork, n in {22,24,26}, depth in {10,16}, outputs {state,expectation}, jax only.
Purpose: get a real criterion-1 read (where is BIG unavoidable materialization) beyond the n<=22 smoke.
Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_targeted_run.py
"""
from __future__ import annotations
import os

from results._phase0_common import orchestrate, fmt_table
from results import _phase0_frontier_probe as p2


def build_configs():
    cfgs = []
    for n in (22, 24, 26):
        for depth in (10, 16):
            for output in ("state", "expectation"):
                cfgs.append({"circuit": "brickwork", "n": n, "depth": depth,
                             "output": output, "backend": "jax"})
    return cfgs


def main():
    configs = build_configs()
    print(f"# targeted frontier: {len(configs)} configs (brickwork, jax)")
    script = os.path.abspath(p2.__file__)
    rows = orchestrate(configs, p2.build_worker_argv, script, timeout=300)

    table_rows = []
    for r in rows:
        c = r["config"]
        if r["ok"]:
            res = r["result"]
            table_rows.append([c["n"], c["depth"], c["output"], "run",
                               res.get("peak_alloc_B", 0),
                               f"{res.get('peak_ratio_vs_state', 0):.2f}",
                               f"{res.get('ms', 0):.1f}"])
        else:
            table_rows.append([c["n"], c["depth"], c["output"], r["outcome"], "-", "-", "-"])
    print(fmt_table(["n", "depth", "output", "outcome", "peak_alloc_B", "peak/state", "ms"], table_rows))
    print("\n# boundary (output, backend) -> max_run_n / min_fail_n:")
    for (out, be), s in sorted(p2.summarize_frontier(rows).items()):
        print(f"  {out:14s} {be:6s} max_run_n={s['max_run_n']}  min_fail_n={s['min_fail_n']}")
    print("=== phase0_targeted done ===")


if __name__ == "__main__":
    main()
