"""Aliasing-aware peak-live analysis for C2 (correction Task C, §6.5/6.6).

Determines whether tile-fusing the anchor P (so it is never materialized) can reduce the
executable's peak temp, using the XLA buffer-assignment liveness (Task A4). Key subtlety
(Task A4): P (.497) and E (.498) ALIAS the same 512 MiB physical slot (non-overlapping
liveness), so XLA already temporally reuses it. "Eliminate P -> save 512 MiB" is therefore
only true if P is in the peak-live-set at the high-water instant -- if XLA already
schedules around P, fusing P yields ~0 peak benefit and C2 (memory) is NOT_FEASIBLE.

Computes, per allocation, the peak simultaneously-live bytes (event sweep over birth/death)
and the live set at the peak instant, then reports the peak WITH vs WITHOUT the anchor P.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from results._phase0.c1_buffer_audit import (
    _find_buffer_assignment,
    parse_buffer_assignment,
)

OUT_DIR = "results/phase0"


def _peak_sweep(vals):
    """Event sweep over (birth, death] -> (peak_live_bytes, peak_t, live_set_at_peak)."""
    events = []
    for v in vals:
        events.append((v["b"], 1, v))
        events.append((v["d"] + 1, -1, v))
    events.sort(key=lambda e: (e[0], e[1]))
    cur = 0
    peak = 0
    peak_t = 0
    curset = {}
    for t, sign, v in events:
        key = (v["name"], v["si"])
        if sign == 1:
            cur += v["sz"]
            curset[key] = v
        else:
            cur -= v["sz"]
            curset.pop(key, None)
        if cur > peak:
            peak = cur
            peak_t = t
    liveset = [v for v in vals if v["b"] <= peak_t <= v["d"]]
    return peak, peak_t, liveset


def analyze(n=24, depth=10, fusion="default"):
    ba_path = _find_buffer_assignment(n, depth, fusion)
    with open(ba_path) as fh:
        ba_text = fh.read()
    records, _by_key, liveness, _by_physical = parse_buffer_assignment(ba_text)

    alloc_vals = defaultdict(list)
    for r in records:
        bd = liveness.get((r["op_name"], r["shape_index"]))
        if not bd:
            continue
        alloc_vals[r["allocation_id"]].append(
            {
                "b": bd[0],
                "d": bd[1],
                "sz": r["value_size"],
                "name": r["op_name"],
                "si": r["shape_index"],
                "alloc_size": r["allocation_size"],
                "kind": r["allocation_kind"],
            }
        )

    per_alloc = {}
    for aid, vals in alloc_vals.items():
        peak, peak_t, liveset = _peak_sweep(vals)
        per_alloc[aid] = {
            "allocation_size": vals[0]["alloc_size"],
            "kind": vals[0]["kind"],
            "peak_live_bytes": peak,
            "peak_t": peak_t,
            "n_live_at_peak": len(liveset),
            "top_live_at_peak": [
                {"name": v["name"], "si": v["si"], "bytes": v["sz"]}
                for v in sorted(liveset, key=lambda x: -x["sz"])[:8]
            ],
        }

    # focus: allocation 11 (the preallocated-temp arena holding the anchor). Compare peak
    # WITH vs WITHOUT the anchor P (custom-call.497{0}) and vs WITHOUT T (loop_transpose_fusion.2,
    # the layout transform of P that feeds the consumer .498). The peak analysis determines
    # WHICH intermediate a fused region must eliminate to reduce the executable peak.
    arena = 11
    vals = alloc_vals.get(arena, [])
    peak_with, peak_t_with, live_with = _peak_sweep(vals)

    def excl(name, si):
        return [v for v in vals if not (v["name"] == name and v["si"] == si)]

    peak_no_p, _t_p, _live_no_p = _peak_sweep(excl("custom-call.497", "0"))
    peak_no_t, _t_t, live_no_t = _peak_sweep(excl("loop_transpose_fusion.2", ""))
    # the real C2 fusion is P->T->E (eliminate BOTH intermediates P and T, keep only E):
    peak_no_pt, _t_pt, _live_no_pt = _peak_sweep(
        [
            v
            for v in vals
            if not (
                (v["name"] == "custom-call.497" and v["si"] == "0")
                or (v["name"] == "loop_transpose_fusion.2")
            )
        ]
    )
    p_in_peak = any(
        v["name"] == "custom-call.497" and v["si"] == "0" for v in live_with
    )
    t_in_peak = any(v["name"] == "loop_transpose_fusion.2" for v in live_with)

    out = {
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "buffer_assignment_path": ba_path,
        "arena_allocation_id": arena,
        "arena_allocation_size": per_alloc.get(arena, {}).get("allocation_size"),
        "arena_peak_live_bytes": peak_with,
        "arena_peak_t": peak_t_with,
        "arena_top_live_at_peak": per_alloc.get(arena, {}).get("top_live_at_peak"),
        "P_in_peak_live_set": p_in_peak,
        "T_in_peak_live_set": t_in_peak,
        "peak_reduction_if_P_eliminated": peak_with - peak_no_p,
        "peak_reduction_if_T_eliminated": peak_with - peak_no_t,
        "peak_reduction_if_P_and_T_eliminated": peak_with - peak_no_pt,
        "peak_after_full_PTE_fusion": peak_no_pt,
        "verdict_hint": (
            "PTE_FUSION_MEMORY_FEASIBLE: the full P->T->E fusion (eliminate BOTH intermediates "
            "P and T, keep only E) reduces the peak by ~512 MiB; eliminating P alone or T alone "
            "does not (the peak is ~2x512MiB at multiple instants)."
            if (peak_with - peak_no_pt) > 256 * 1024 * 1024
            else "PTE_FUSION_NO_CLEAR_MEMORY_BENEFIT"
        ),
        "all_allocations": {
            str(a): {
                "size": v["allocation_size"],
                "kind": v["kind"],
                "peak_live": v["peak_live_bytes"],
            }
            for a, v in per_alloc.items()
        },
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/c2_peak_analysis.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
