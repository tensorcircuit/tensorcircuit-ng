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

import csv
import itertools
import json
import os
from collections import defaultdict

from results._phase0.c1_buffer_audit import (
    _find_buffer_assignment,
    _sha256_file,
    parse_buffer_assignment,
)
from results._phase0.c1_to_c2_map import _bare, _consumers

OUT_DIR = "results/phase0"
HLO_DIR = f"{OUT_DIR}/c1_optimized_hlo"
AUDIT_DIR = f"{OUT_DIR}/c1_buffer_assignment"
PEAK_FRONTIER_JSON = f"{OUT_DIR}/c2_peak_frontier.json"
PEAK_WINDOWS_CSV = f"{OUT_DIR}/c2_peak_windows.csv"
# A producer GEMM at least this large is a candidate region-fusion window.
WINDOW_MIN_BYTES = 256 * 1024 * 1024
# Joint-elimination reduction targets for the minimum-cover search.
PEAK_REDUCTION_TARGETS = [256 * 1024 * 1024, 512 * 1024 * 1024]


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


# --- final-remediation Task 3: generalized, de-hardcoded peak frontier ---------------------
#
# Physical-union global sweep + enumeration of ALL large GEMM+transpose windows (not just the
# anchor), single-patch and joint elimination frontiers, and three diagnostics. Emits NO
# canonical C2 verdict (that is the gate's job, Task 5). Producers/transforms/consumers are
# identified from the allocation audit + the v2 edge map (Task 2) -- no hardcoded arena id or
# .497/.498 literals.


def union_bytes(vals) -> int:
    """Union of physical byte ranges ``[offset, offset+size)`` for values in ONE allocation."""
    if not vals:
        return 0
    intervals = sorted((v["offset"], v["offset"] + v["size"]) for v in vals)
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return total


def program_peak_physical(values):
    """Global event sweep: at each instant live bytes = sum over allocations of the union of
    that allocation's live physical ranges. Returns ``(peak_bytes, peak_t, live_at_peak)``.

    Aliasing/in-place correct (unlike a naive value-size sum): values sharing an
    ``(allocation_id, offset)`` count once.
    """
    events = []
    for v in values:
        if v["birth"] is None or v["death"] is None:
            continue
        events.append((v["birth"], 1, v))
        events.append((v["death"] + 1, -1, v))
    events.sort(key=lambda e: (e[0], e[1]))
    live = {}  # id(v) -> v
    peak = 0
    peak_t = 0
    peak_live = []
    for t, sign, v in events:
        if sign == 1:
            live[id(v)] = v
        else:
            live.pop(id(v), None)
        by_alloc: dict = {}
        for lv in live.values():
            by_alloc.setdefault(lv["alloc_id"], []).append(lv)
        total = sum(union_bytes(av) for av in by_alloc.values())
        if total > peak:
            peak = total
            peak_t = t
            peak_live = list(live.values())
    return peak, peak_t, peak_live


def peak_excluding(values, excluded_keys) -> int:
    """Program peak when the named values (``(name, si)`` keys) are eliminated."""
    keep = [v for v in values if v["key"] not in excluded_keys]
    return program_peak_physical(keep)[0]


def _values_from_records(records, liveness):
    values = []
    for r in records:
        bd = liveness.get((r["op_name"], r["shape_index"]))
        if not bd:
            continue
        values.append(
            {
                "name": r["op_name"],
                "si": r["shape_index"],
                "alloc_id": r["allocation_id"],
                "offset": r["offset"],
                "size": r["value_size"],
                "birth": bd[0],
                "death": bd[1],
                "key": (r["op_name"], r["shape_index"]),
            }
        )
    return values


def _eliminable_keys(producer_hlo_id, data_result_index, traced, records) -> set:
    """Buffer keys a fused region would avoid materializing: the producer DATA output plus the
    materialized transform intermediates on the pierced chain (intersection with records).
    """
    keys = {(producer_hlo_id.lstrip("%"), str(data_result_index or 0))}
    rec_by_name: dict = {}
    for r in records:
        rec_by_name.setdefault(r["op_name"], []).append(r)
    for tname in traced:
        for r in rec_by_name.get(tname, []):
            keys.add((r["op_name"], r["shape_index"]))
    return keys


def enumerate_windows(hlo_text, audit, records) -> list:
    """Every large (>= WINDOW_MIN_BYTES) GEMM producer whose single consumer is a contraction,
    i.e. an isomorphic producer->transform->consumer region. single_reduction is filled later.
    """
    windows = []
    for idx, b in enumerate(audit["buffers"]):
        if (b.get("data_output_bytes") or 0) < WINDOW_MIN_BYTES:
            continue
        if b.get("allocation_id") is None:
            continue
        consumers, traced, consumer_mnk, had_unpierced = _consumers(
            hlo_text, b["hlo_value_id"]
        )
        if len(consumers) != 1 or consumers[0] not in consumer_mnk:
            continue  # not a two-stage contraction region
        terminal = consumers[0]
        elim = _eliminable_keys(
            b["hlo_value_id"], b.get("data_result_index", 0), traced, records
        )
        windows.append(
            {
                "window_id": f"W{idx:02d}",
                "producer_id": b["hlo_value_id"],
                "producer_allocation_id": b["allocation_id"],
                "producer_offset": b.get("offset"),
                "producer_bytes": b["data_output_bytes"],
                "producer_birth": b.get("birth"),
                "producer_death": b.get("death"),
                "transform_ids": traced,
                "consumer_id": terminal if terminal.startswith("%") else "%" + terminal,
                "consumer_mnk": list(consumer_mnk[terminal]),
                "eliminable_keys": [list(k) for k in sorted(elim)],
                "had_unpierced_fusion": had_unpierced,
            }
        )
    return windows


def analyze_frontier(n=24, depth=10, fusion="default") -> dict:
    """Build the de-hardcoded single+joint peak frontier and write the v2 artifacts.

    Reads the v2 edge map (Task 2) + the allocation audit (Task 1) + the XLA buffer
    assignment. The anchor window is the one whose producer matches the edge map -- no
    literal arena id or .497/.498. Emits ``c2_peak_frontier.json`` + ``c2_peak_windows.csv``
    and three diagnostics (no canonical C2 verdict).
    """
    with open(f"{OUT_DIR}/c1_c2_edge_map.json") as fh:
        edge = json.load(fh)
    audit_path = f"{AUDIT_DIR}/n{n}_d{depth}_{fusion}.json"
    with open(audit_path) as fh:
        audit = json.load(fh)
    ba_path = _find_buffer_assignment(n, depth, fusion)
    with open(ba_path) as fh:
        ba_text = fh.read()
    records, _by_key, liveness, _by_physical = parse_buffer_assignment(ba_text)
    hlo_path = f"{HLO_DIR}/n{n}_d{depth}_exp_{fusion}.hlo"
    with open(hlo_path) as fh:
        hlo_text = fh.read()

    values = _values_from_records(records, liveness)
    base_peak, base_peak_t, _base_live = program_peak_physical(values)

    windows = enumerate_windows(hlo_text, audit, records)
    for w in windows:
        elim = {tuple(k) for k in w["eliminable_keys"]}
        peak_without = peak_excluding(values, elim)
        w["peak_after_single_elimination"] = peak_without
        w["single_reduction_bytes"] = base_peak - peak_without

    edge_producer = edge.get("producer", {}).get("hlo_value_id")
    anchor = next((w for w in windows if w["producer_id"] == edge_producer), None)

    all_elim = set()
    for w in windows:
        all_elim |= {tuple(k) for k in w["eliminable_keys"]}
    max_joint_reduction = base_peak - peak_excluding(values, all_elim)

    min_cover_by_target = {}
    n_win = len(windows)
    for target in PEAK_REDUCTION_TARGETS:
        best = None
        for size in range(1, n_win + 1):
            for combo in itertools.combinations(range(n_win), size):
                elim = set()
                for i in combo:
                    elim |= {tuple(k) for k in windows[i]["eliminable_keys"]}
                red = base_peak - peak_excluding(values, elim)
                if red >= target:
                    best = {
                        "window_ids": [windows[i]["window_id"] for i in combo],
                        "joint_reduction_bytes": red,
                    }
                    break
            if best:
                break
        min_cover_by_target[target] = best

    threshold = 256 * 1024 * 1024
    anchor_red = (anchor or {}).get("single_reduction_bytes")
    diagnostics = {
        "single_anchor_patch_status": (
            "peak_reduction_below_threshold"
            if (anchor_red is None or anchor_red < threshold)
            else "peak_reduction_above_threshold"
        ),
        "single_anchor_reduction_bytes": anchor_red,
        "joint_model_status": (
            "joint_reduction_meets_threshold"
            if max_joint_reduction >= threshold
            else "joint_reduction_below_threshold"
        ),
        "max_joint_reduction_bytes": max_joint_reduction,
        "kernel_feasibility_status": "UNKNOWN",
    }

    out = {
        "schema_version": "c2-peak-frontier-v1",
        "case_id": f"n{n}_d{depth}_{fusion}",
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "source_hlo_sha256": edge.get("source_hlo", {}).get("sha256"),
        "edge_map_sha256": _sha256_file(f"{OUT_DIR}/c1_c2_edge_map.json"),
        "buffer_assignment_path": ba_path,
        "base_peak_bytes": base_peak,
        "base_peak_t": base_peak_t,
        "window_count": len(windows),
        "windows": windows,
        "anchor_window": anchor,
        "joint_model": {
            "max_joint_reduction_bytes": max_joint_reduction,
            "min_cover_by_target": {str(t): v for t, v in min_cover_by_target.items()},
        },
        "diagnostics": diagnostics,
        "model_assumptions": [
            "counterfactual: only the named intermediates are removed; the rest of the "
            "executable schedule is held unchanged",
            "fused-kernel workspace is NOT counted (could raise the peak)",
            "producer tile recompute cost and HBM traffic are NOT counted",
            "physical live bytes use the XLA buffer-assignment (allocation_id, offset, size) "
            "union, so aliased/in-place ranges count once",
        ],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PEAK_FRONTIER_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(PEAK_WINDOWS_CSV, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(
            [
                "window_id",
                "producer_id",
                "consumer_id",
                "producer_allocation_id",
                "producer_bytes",
                "single_reduction_bytes",
                "peak_after_single_elimination",
                "eliminable_keys",
                "transform_ids",
            ]
        )
        for win in windows:
            wr.writerow(
                [
                    win["window_id"],
                    win["producer_id"],
                    win["consumer_id"],
                    win["producer_allocation_id"],
                    win["producer_bytes"],
                    win["single_reduction_bytes"],
                    win["peak_after_single_elimination"],
                    ";".join(f"{k[0]}{{{k[1]}}}" for k in win["eliminable_keys"]),
                    ";".join(win["transform_ids"]),
                ]
            )
    return out


if __name__ == "__main__":
    print(json.dumps(analyze_frontier(), indent=2))
