"""Throwaway targeted fusion A/B driver: Probe 3's decisive lens-2 at larger n.
expectation output (the discriminating scalar terminal), n in {22,24,26}, depth in {10,16}, jax.
Purpose: decisive criterion-1 signal — is intermediate materialization UNAVOIDABLE (fusion A/B peak
ratio ~1 => bf16 window exists) or FUSED-AWAY (ratio >>1 => no window), at user-relevant sizes?
Run: MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_targeted_fusion.py
"""
from __future__ import annotations
import os

from results._phase0_common import orchestrate, fmt_table
from results import _phase0_fusion_window_probe as p3
from results._phase0_fusion_window_probe import classify_materialization


def build_configs():
    cfgs = []
    for n in (22, 24, 26):
        for depth in (10, 16):
            cfgs.append({"n": n, "depth": depth, "output": "expectation", "disable_fusion": 0})
            cfgs.append({"n": n, "depth": depth, "output": "expectation", "disable_fusion": 1})
    return cfgs


def main():
    configs = build_configs()
    print(f"# targeted fusion A/B (expectation): {len(configs)} worker runs (A/B pairs at 6 sizes)")
    script = os.path.abspath(p3.__file__)
    rows = orchestrate(configs, p3.build_worker_argv, script, timeout=300)

    by_key = {}
    failed = []
    for r in rows:
        c = r["config"]
        k = (c["n"], c["depth"], c["output"])
        if not r["ok"]:
            failed.append((k, c["disable_fusion"], r["outcome"]))
            continue
        by_key.setdefault(k, {})[c["disable_fusion"]] = r["result"]

    table_rows = []
    for k in sorted(by_key):
        pair = by_key[k]
        dflt = pair.get(0, {})
        nofus = pair.get(1, {})
        pd = dflt.get("peak_B", 0)
        pn = nofus.get("peak_B", 0)
        cls = classify_materialization(pd, pn)
        ratio = (pn / pd) if pd else 0.0
        table_rows.append([k[0], k[1], pd, pn, f"{ratio:.2f}", cls])
    print(fmt_table(["n", "depth", "peak_default_B", "peak_nofusion_B", "nofus/default", "classification"], table_rows))

    if failed:
        print("\n# failed arms:")
        for k, df, out in failed:
            print(f"  n={k[0]} d={k[1]} {k[2]} disable_fusion={df} -> {out}")
    print("\n# materialized-unavoidable = bf16-window candidate; fused-away = no window.")
    print("=== phase0_targeted_fusion done ===")


if __name__ == "__main__":
    main()
