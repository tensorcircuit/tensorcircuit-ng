"""C2 coverage verdict (rereview §5.3, canonical-completion Task 4).

TWO paths:
- CANONICAL (``basis="hlo_use_def"``): ``judge_c2_canonical`` / ``run_c2_canonical`` -- the
  real producer->terminal-consumer edge from the production HLO use-def (Task B) + the
  aliasing-aware peak analysis (Task C 6.5/6.6) + the allocation audit (Task A). FAIL-CLOSED:
  recomputes the peak reduction from RAW bytes and returns FAIL/NOT_FEASIBLE when region
  fusion of the anchor pair cannot reduce the executable peak, UNKNOWN when artifacts are
  incomplete. The SOLE writer of ``c2_judgment.json`` (+ ``c2_checkpoint_manifest.json``).
- INFORMATIONAL (``basis="cotengra_state_heuristic"``, DEMOTED): ``classify_tileability`` /
  ``judge_c2`` / ``run_c2_integration`` -- the cotengra-state tile-mappability heuristic.
  NON-FAITHFUL (cotengra is a different contractor than production); writes
  ``c2_cotengra_informational.json`` and is NOT consumed by gonogo.

The classes/heuristic below belong to the INFORMATIONAL path. A contraction step is
"tile-mappable" when its output buffer can be kept on-chip (fused into the consuming
GEMM's epilogue or recomputed) instead of round-tripping through HBM -- the prerequisite
for bf16 Tensor Core engagement via region/tile fusion (spec §8.1).

Five classes (review §6.2):
- ``direct-gemm-tileable``: single-consumer + GEMM-shaped + all dims 16-aligned
  -> the consuming GEMM can absorb this buffer as a register/shared-memory tile
  with no pack/recompute cost.
- ``tileable-with-pack``: single-consumer + GEMM-shaped but some dim not 16-aligned
  -> tile-fusable but needs a pack/pad stub (cost ~ M*N*2 bytes).
- ``tileable-with-recompute``: (forward-compat class; the brief's heuristic does not
  emit it, but ``judge_c2`` counts it as tileable).
- ``not-tileable``: multi-consumer (``consumer_count > 1``) -> the buffer has >1 user,
  so fusing into one consumer would not eliminate the global write; no tile fusion.
- ``unknown``: degenerate shape (a zero dim) -> unclassifiable.

Analytic model: for a tileable class, ``global_bytes_eliminated`` = the buffer's
``bytes`` (the HBM write+read that tile fusion removes), ``pack_bytes`` = the
one-time pack cost if misaligned, ``recompute_ratio`` = fraction of K re-floated.
Net byte gain = ``global_bytes_eliminated - pack_bytes - recompute_bytes``.

Two entry points
----------------
- ``classify_tileability(shape)`` -> dict. PURE; unit-tested.
- ``judge_c2(materialized_shapes)`` -> ``{"status", "reason", "rows"}``. PURE;
  unit-tested. PASS iff >=1 shape is in a tileable class with
  ``global_bytes_eliminated > pack_bytes`` (recompute_bytes is 0 for the classes
  the heuristic emits, so this is equivalent to ``> pack_bytes + recompute_bytes``;
  see note in ``judge_c2``).

C1-large threshold (review §6.2 / task handoff): C2=PASS requires the tileable
buffer to be C1-large (``bytes >= 0.5 * full_state_bytes``). ``judge_c2`` itself is
agnostic to ``full_state_bytes`` (it just checks tileable + net gain on whatever
shapes it is handed); the C1-large filter is applied at the integration layer
(``run_c2_integration`` feeds only C1-large state rows to ``judge_c2``). This keeps
the unit-tested contract identical to the task brief while satisfying the spec's
"C2=PASS iff >=1 C1-large tileable buffer" requirement.

Integration: ``run_c2_integration(n=24, depth=10)`` reads
``results/phase0/contraction_shapes.csv`` (Task 6 output), filters to ``state``
rows for the requested ``n`` (NOT ``expectation`` — the expectation tree has only
2 steps and is not representative), classifies every state row, writes one row per
state step to ``results/phase0/c2_tileability.csv``, then filters to C1-large
(``bytes >= 0.5 * 2**n * 8``) and writes the C2 judgment to
``results/phase0/c2_judgment.json``.

Usage
-----
    MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh \
        python results/_phase0_c2.py --n 24 --depth 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any

OUT_DIR = "results/phase0"
SHAPES_CSV_PATH = f"{OUT_DIR}/contraction_shapes.csv"
TILE_CSV_PATH = f"{OUT_DIR}/c2_tileability.csv"
JUDGMENT_JSON_PATH = f"{OUT_DIR}/c2_judgment.json"
# cotengra-state pipeline demoted to INFORMATIONAL (non-faithful: different contractor than
# production -- see plan Global Constraints "contraction contractor"). NOT consumed by gonogo.
COTENGRA_INFO_JSON_PATH = f"{OUT_DIR}/c2_cotengra_informational.json"
EDGE_MAP_CSV_PATH = f"{OUT_DIR}/c1_c2_edge_map.csv"
REGION_PROTOTYPE_JSON_PATH = f"{OUT_DIR}/region_prototype.json"
PEAK_ANALYSIS_JSON_PATH = f"{OUT_DIR}/c2_peak_analysis.json"
AUDIT_DIR = f"{OUT_DIR}/c1_buffer_assignment"
CHECKPOINT_MANIFEST_PATH = f"{OUT_DIR}/c2_checkpoint_manifest.json"
# A region fusion worth its complexity must reduce the executable peak by at least this.
C2_MEMORY_THRESHOLD = 256 * 1024 * 1024

# Tile-fusable classes (review §6.2): a buffer in any of these eliminates its
# global HBM write/read when fused into the consuming GEMM's tile epilogue.
TILEABLE_CLASSES = (
    "direct-gemm-tileable",
    "tileable-with-pack",
    "tileable-with-recompute",
)


def classify_tileability(s: dict[str, Any]) -> dict[str, Any]:
    """Classify one contraction step's tile-mappability.

    ``s`` keys: ``M, N, K, consumer_count, bytes`` (``transpose`` is accepted but
    does not change the heuristic — the brief's classifier keys off alignment and
    consumer_count only). Returns a dict with at least ``class``,
    ``global_bytes_eliminated``, ``pack_bytes``, ``recompute_ratio``; the tileable
    classes additionally carry ``shared_memory_per_CTA`` and ``boundary_conversions``.
    """
    M, N, K = s["M"], s["N"], s["K"]
    cc = s["consumer_count"]
    bytes_ = s["bytes"]
    if M == 0 or N == 0 or K == 0:
        return {
            "class": "unknown",
            "global_bytes_eliminated": 0,
            "pack_bytes": 0,
            "recompute_ratio": 0.0,
        }
    if cc > 1:
        return {
            "class": "not-tileable",
            "global_bytes_eliminated": 0,
            "pack_bytes": 0,
            "recompute_ratio": 1.0,
            "reason": f"{cc} consumers",
        }
    # single consumer: tile-fusable eliminates the global write/read of this buffer
    aligned = M % 16 == 0 and N % 16 == 0 and K % 16 == 0
    cls = "direct-gemm-tileable" if aligned else "tileable-with-pack"
    pack_bytes = 0 if aligned else (M * N * 2)  # rough pack cost if misaligned
    return {
        "class": cls,
        "global_bytes_eliminated": bytes_,
        "pack_bytes": pack_bytes,
        "recompute_ratio": 0.0,
        "shared_memory_per_CTA": min(K, 64) * 32,
        "boundary_conversions": 0 if aligned else 4,
    }


def judge_c2(materialized_shapes: list[dict[str, Any]]) -> dict[str, Any]:
    """C2 verdict over a list of materialized shape dicts.

    PASS iff >=1 shape lands in a tileable class with a positive net byte gain
    (``global_bytes_eliminated > pack_bytes``); the ``recompute_bytes`` term is 0
    for every class the heuristic emits (``recompute_ratio`` is 0.0 for the tileable
    classes and the ``tileable-with-recompute`` class is never produced), so
    ``> pack_bytes`` is equivalent to ``> pack_bytes + recompute_bytes`` here.

    ``unknown`` != YES: an empty input or an all-``unknown`` input yields UNKNOWN
    (no classifiable buffers), not PASS. A non-empty input with no tileable shape
    yields FAIL.
    """
    any_tileable = False
    rows = []
    for s in materialized_shapes:
        c = classify_tileability(s)
        rows.append((s, c))
        if c["class"] in TILEABLE_CLASSES:
            if c["global_bytes_eliminated"] > c["pack_bytes"]:
                any_tileable = True
    if not materialized_shapes or all(c["class"] == "unknown" for _, c in rows):
        return {
            "status": "UNKNOWN",
            "reason": "no classifiable buffers",
            "rows": rows,
        }
    if any_tileable:
        return {
            "status": "PASS",
            "reason": ">=1 large buffer tile-fusable with net byte gain",
            "rows": rows,
        }
    return {
        "status": "FAIL",
        "reason": "materialized buffers not tileable (multi-consumer/irregular)",
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Integration: classify the real n=24/state contraction shapes (Task 6 CSV)
# --------------------------------------------------------------------------


def _load_state_rows(csv_path: str, n: int) -> list[dict[str, Any]]:
    """Read Task 6's contraction_shapes.csv, return ``state`` rows for ``n``.

    Columns (verified): n,depth,output,node_id,modes,extents,M,N,K,batch,transpose,
    strides,bytes,consumer_count,producer_ids,consumer_ids,live_range.
    Numeric fields are cast to int; ``transpose`` is cast to bool (0/1).
    """
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if int(r["n"]) != n:
                continue
            if r["output"] != "state":
                continue
            rows.append(
                {
                    "n": int(r["n"]),
                    "depth": int(r["depth"]),
                    "node_id": int(r["node_id"]),
                    "M": int(r["M"]),
                    "N": int(r["N"]),
                    "K": int(r["K"]),
                    "transpose": bool(int(r["transpose"])),
                    "bytes": int(r["bytes"]),
                    "consumer_count": int(r["consumer_count"]),
                }
            )
    return rows


def _write_tile_csv(rows_with_class: list[tuple[dict, dict]], path: str) -> None:
    """Write one CSV row per (shape, classification). ``rows_with_class`` is the
    ``(shape, class_dict)`` pair list produced by classifying every state row."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        "n",
        "depth",
        "node_id",
        "M",
        "N",
        "K",
        "transpose",
        "bytes",
        "consumer_count",
        "class",
        "global_bytes_eliminated",
        "pack_bytes",
        "recompute_ratio",
        "net_gain",
        "c1_large",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for s, c in rows_with_class:
            net = c["global_bytes_eliminated"] - c["pack_bytes"]
            w.writerow(
                [
                    s["n"],
                    s["depth"],
                    s["node_id"],
                    s["M"],
                    s["N"],
                    s["K"],
                    int(s["transpose"]),
                    s["bytes"],
                    s["consumer_count"],
                    c["class"],
                    c["global_bytes_eliminated"],
                    c["pack_bytes"],
                    f"{c['recompute_ratio']:.4f}",
                    net,
                    "",  # c1_large filled by caller context (per-n threshold)
                ]
            )


def run_c2_integration(n: int = 24, depth: int = 10) -> dict[str, Any]:
    """Classify the real contraction shapes for ``(n, depth)`` and write artifacts.

    Reads ``results/phase0/contraction_shapes.csv`` (Task 6), filters to ``state``
    rows for ``n``, classifies every state row -> ``results/phase0/c2_tileability.csv``,
    then restricts to C1-large buffers (``bytes >= 0.5 * full_state_bytes``) and
    writes the INFORMATIONAL cotengra-state baseline to
    ``results/phase0/c2_cotengra_informational.json`` (``basis="cotengra_state_heuristic"``;
    NON-FAITHFUL -- a different contractor than production, so NOT canonical and NOT consumed
    by gonogo). The canonical C2 verdict is ``run_c2_canonical`` -> ``c2_judgment.json``.

    Returns the judgment payload (also written to disk).
    """
    full_state_bytes = (2**n) * 8
    c1_large_threshold = 0.5 * full_state_bytes

    state_rows = _load_state_rows(SHAPES_CSV_PATH, n)
    all_rows_with_class = [(s, classify_tileability(s)) for s in state_rows]
    _write_tile_csv(all_rows_with_class, TILE_CSV_PATH)

    # Mark c1_large on the CSV (second pass: rewrite the column now that we know the
    # per-n threshold). Kept as a separate step so _write_tile_csv stays threshold-free
    # and unit-testable without a full integration run.
    _backfill_c1_large_column(TILE_CSV_PATH, c1_large_threshold)

    c1_large_shapes = [s for s in state_rows if s["bytes"] >= c1_large_threshold]
    judgment = judge_c2(c1_large_shapes)

    # Enrich the judgment payload with the integration-level summary Task 8 needs.
    c1_large_rows_with_class = [
        (s, c) for s, c in all_rows_with_class if s["bytes"] >= c1_large_threshold
    ]
    class_counts: dict[str, int] = {}
    for _, c in c1_large_rows_with_class:
        class_counts[c["class"]] = class_counts.get(c["class"], 0) + 1
    tileable_c1_large = sum(
        cnt for cls, cnt in class_counts.items() if cls in TILEABLE_CLASSES
    )

    payload = {
        "n": n,
        "depth": depth,
        "basis": "cotengra_state_heuristic",  # INFORMATIONAL: non-faithful (different contractor)
        "full_state_bytes": full_state_bytes,
        "c1_large_threshold_bytes": c1_large_threshold,
        "state_step_count": len(state_rows),
        "c1_large_count": len(c1_large_shapes),
        "c1_large_class_counts": class_counts,
        "c1_large_tileable_count": tileable_c1_large,
        "status": judgment["status"],
        "reason": judgment["reason"],
        "tile_csv_path": TILE_CSV_PATH,
    }
    _update_judgment_json(COTENGRA_INFO_JSON_PATH, f"n{n}_d{depth}", payload)
    return payload


def _backfill_c1_large_column(csv_path: str, threshold: float) -> None:
    """Rewrite the ``c1_large`` column (last col) to ``1``/``0`` based on ``bytes``
    versus the per-n C1-large threshold. The column is written blank by
    ``_write_tile_csv`` because the threshold is only known at integration time."""
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return
    header = rows[0]
    try:
        bytes_idx = header.index("bytes")
        c1_idx = header.index("c1_large")
    except ValueError:
        return
    for r in rows[1:]:
        try:
            r[c1_idx] = "1" if int(r[bytes_idx]) >= threshold else "0"
        except (ValueError, IndexError):
            continue
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerows(rows)


def _update_judgment_json(path: str, key: str, payload: dict[str, Any]) -> None:
    """Read-merge-write a dict keyed by ``key`` into the judgment JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[key] = payload
    with open(path, "w") as fh:
        json.dump(existing, fh, indent=2)


def _load_edge_map_row(n, depth, fusion):
    """Read Task B's c1_c2_edge_map.csv -> the row for (n, depth, fusion), or a fail-closed
    placeholder (drives UNKNOWN) if absent. Uses the post-Task-B schema (producer/terminal).
    """
    case_id = f"n{n}_d{depth}"
    placeholder = {
        "case_id": case_id,
        "producer_hlo_value_id": "",
        "terminal_consumer_hlo_value_id": "",
        "producer_M": 0,
        "producer_N": 0,
        "producer_K": 0,
    }
    if not os.path.exists(EDGE_MAP_CSV_PATH):
        return {**placeholder, "note": "edge_map.csv missing"}
    with open(EDGE_MAP_CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["n"]) == n and int(r["depth"]) == depth and r["fusion"] == fusion:
                return {
                    "case_id": case_id,
                    "producer_hlo_value_id": r.get("producer_hlo_value_id", ""),
                    "producer_M": int(r.get("producer_M", 0)),
                    "producer_N": int(r.get("producer_N", 0)),
                    "producer_K": int(r.get("producer_K", 0)),
                    "terminal_consumer_hlo_value_id": r.get(
                        "terminal_consumer_hlo_value_id", ""
                    ),
                    "consumer_M": int(r.get("consumer_M", 0)),
                    "consumer_N": int(r.get("consumer_N", 0)),
                    "consumer_K": int(r.get("consumer_K", 0)),
                    "consumer_output_bytes": int(r.get("consumer_output_bytes", 0)),
                }
    return {**placeholder, "note": "no edge row for case"}


def _audit_json_path(n, depth, fusion):
    return os.path.join(AUDIT_DIR, f"n{n}_d{depth}_{fusion}.json")


def _sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_checkpoint_manifest(case_id, payload, hashes):
    """Task D5: provenance manifest (source/allocation/edge/prototype/judgment hashes +
    case status). Does NOT replace the final Task 10 manifest; guarantees Task 1-4 reproducibility.
    """
    import time as _time

    manifest = {
        "schema_version": "task1-4-checkpoint-v1",
        "case_id": case_id,
        "generated_at_epoch": int(_time.time()),
        "case_status": payload["status"],
        "prototype_verdict": payload.get("prototype_verdict"),
        "artifact_hashes": hashes,
        "commands": {
            "edge_map": "python results/_phase0/c1_to_c2_map.py (or map_anchor_for_case)",
            "peak_analysis": "python results/_phase0/c2_peak_analysis.py",
            "audit": "audit_buffer_assignment + xla_dump.py",
            "gate": "python results/_phase0/c2.py --n <n> --depth <d>",
        },
    }
    os.makedirs(os.path.dirname(CHECKPOINT_MANIFEST_PATH), exist_ok=True)
    with open(CHECKPOINT_MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def judge_c2_canonical(edge, peak, audit, case_id=""):
    """Fail-closed canonical C2 verdict from the HLO edge (Task B) + aliasing-aware peak
    analysis (Task C 6.5/6.6) + allocation audit (Task A). ``basis="hlo_use_def"``.

    Recomputes the memory benefit from RAW peak bytes (never trusts precomputed booleans).
    FAIL/NOT_FEASIBLE when region fusion of the anchor pair cannot reduce the executable
    peak (the binding peak is structural -- the contraction chain of GEMM+transpose pairs).
    UNKNOWN when artifacts or case-binding are incomplete. PASS would require a real peak
    reduction >= C2_MEMORY_THRESHOLD (not the case on n=24/d=10/default).
    """
    problems = []
    conds = {}
    conds["edge_reaches_real_consumer_498"] = (
        edge.get("terminal_consumer_hlo_value_id") == "%custom-call.498"
    )
    if not conds["edge_reaches_real_consumer_498"]:
        problems.append("edge does not reach terminal consumer %custom-call.498")
    # recompute memory benefit from RAW peak bytes (do NOT trust a precomputed boolean)
    peak_with = peak.get("arena_peak_live_bytes")
    peak_after = peak.get("peak_after_full_PTE_fusion")
    if peak_with is None or peak_after is None:
        problems.append(
            "peak analysis missing arena_peak_live_bytes/peak_after_full_PTE_fusion"
        )
        peak_reduction = None
        conds["peak_reduction_bytes"] = None
        conds["memory_benefit_meets_threshold"] = False
    else:
        peak_reduction = int(peak_with) - int(peak_after)
        conds["peak_reduction_bytes"] = peak_reduction
        conds["memory_benefit_meets_threshold"] = peak_reduction >= C2_MEMORY_THRESHOLD
    conds["allocation_is_real"] = (
        audit.get("allocation_source") == "xla_buffer_assignment"
    )
    if not conds["allocation_is_real"]:
        problems.append("allocation audit not from XLA buffer-assignment")
    base = {"basis": "hlo_use_def", "conditions": conds, "case_id": case_id}
    if problems:
        return {"status": "UNKNOWN", "reason": "; ".join(problems), **base}
    if peak_reduction < C2_MEMORY_THRESHOLD:
        return {
            "status": "FAIL",
            "prototype_verdict": "NOT_FEASIBLE",
            "reason": (
                f"region fusion of the anchor pair (P->T->E) reduces the executable peak by only "
                f"{peak_reduction} B << {C2_MEMORY_THRESHOLD} B threshold; the ~{peak_with} B peak "
                f"is structural (contraction chain of GEMM+transpose pairs), so fusing one pair "
                f"cannot reduce it"
            ),
            **base,
        }
    return {
        "status": "UNKNOWN",
        "prototype_verdict": "PENDING_REAL_PROTOTYPE",
        "reason": "peak reduction meets threshold but the real P->T->E prototype has not been run",
        **base,
    }


def run_c2_canonical(n, depth, fusion="default"):
    """Canonical C2 verdict from the HLO edge map + peak analysis + allocation audit.
    Writes results/phase0/c2_judgment.json (basis=hlo_use_def) + c2_checkpoint_manifest.json.
    The SOLE canonical writer."""
    case_id = f"n{n}_d{depth}"
    edge = _load_edge_map_row(n, depth, fusion)
    peak = _load_json(PEAK_ANALYSIS_JSON_PATH)
    audit = _load_json(_audit_json_path(n, depth, fusion))
    judgment = judge_c2_canonical(edge, peak, audit, case_id=case_id)
    audit_path = _audit_json_path(n, depth, fusion)
    hlo_path = f"{OUT_DIR}/c1_optimized_hlo/n{n}_d{depth}_exp_{fusion}.hlo"
    hashes = {
        "source_hlo": _sha256_file(hlo_path),
        "allocation_audit": _sha256_file(audit_path),
        "edge_map_csv": _sha256_file(EDGE_MAP_CSV_PATH),
        "peak_analysis": _sha256_file(PEAK_ANALYSIS_JSON_PATH),
    }
    payload = {
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "case_id": case_id,
        "basis": judgment["basis"],
        "edge": edge,
        "peak_analysis_verdict_hint": peak.get("verdict_hint"),
        "peak_reduction_bytes": judgment["conditions"].get("peak_reduction_bytes"),
        "memory_threshold_bytes": C2_MEMORY_THRESHOLD,
        "allocation_source": audit.get("allocation_source"),
        "status": judgment["status"],
        "prototype_verdict": judgment.get("prototype_verdict"),
        "reason": judgment["reason"],
        "conditions": judgment["conditions"],
        "artifact_hashes": hashes,
    }
    _update_judgment_json(JUDGMENT_JSON_PATH, case_id, payload)
    _write_checkpoint_manifest(case_id, payload, hashes)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C2 canonical gate (default) or informational cotengra-state baseline."
    )
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--fusion", default="default")
    ap.add_argument(
        "--informational-cotengra",
        action="store_true",
        help="run the (non-faithful) cotengra-state baseline instead of the canonical gate",
    )
    a = ap.parse_args()
    if a.informational_cotengra:
        payload = run_c2_integration(a.n, a.depth)
    else:
        payload = run_c2_canonical(a.n, a.depth, a.fusion)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
