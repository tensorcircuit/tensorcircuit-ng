"""C2 coverage verdict (rereview §5.3, canonical-completion Task 4).

TWO paths:
- CANONICAL (``basis="hlo_use_def"``): ``judge_c2_canonical`` / ``run_c2_canonical`` -- the
  fail-closed C2 v2 gate (final-remediation Task 5, spec §5.4/§8). Consumes the Task 2 edge
  map + Task 3 peak frontier + Task 4 region prototype + Task 1 allocation audit, binds
  case + cross-artifact + on-disk hashes, and SELF-RECOMPUTES accuracy/resource/peak/traffic/
  recompute/workspace/latency from raw fields (self-reported booleans are diagnostic only).
  Emits THREE layers -- ``C2_REGION_KERNEL_FEASIBILITY``,
  ``C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK``, ``C2_JOINT_EXECUTABLE_LEVERAGE`` -- composed into
  ``C2_CANONICAL`` per spec §5.4. Any case/hash/schema mismatch or incomplete evidence ->
  UNKNOWN (a single-pair peak FAIL never alone yields canonical FAIL). The SOLE writer of
  ``c2_judgment.json`` (+ ``c2_checkpoint_manifest.json``).
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
import subprocess
import sys
import time
from typing import Any

OUT_DIR = "results/phase0"
SHAPES_CSV_PATH = f"{OUT_DIR}/contraction_shapes.csv"
TILE_CSV_PATH = f"{OUT_DIR}/c2_tileability.csv"
JUDGMENT_JSON_PATH = f"{OUT_DIR}/c2_judgment.json"
# cotengra-state pipeline demoted to INFORMATIONAL (non-faithful: different contractor than
# production -- see plan Global Constraints "contraction contractor"). NOT consumed by gonogo.
COTENGRA_INFO_JSON_PATH = f"{OUT_DIR}/c2_cotengra_informational.json"
EDGE_MAP_JSON_PATH = f"{OUT_DIR}/c1_c2_edge_map.json"
PEAK_FRONTIER_JSON_PATH = f"{OUT_DIR}/c2_peak_frontier.json"
REGION_PROTOTYPE_JSON_PATH = f"{OUT_DIR}/region_prototype.json"
AUDIT_DIR = f"{OUT_DIR}/c1_buffer_assignment"
CHECKPOINT_MANIFEST_PATH = f"{OUT_DIR}/c2_checkpoint_manifest.json"
# A region fusion worth its complexity must reduce the executable peak by at least this.
C2_MEMORY_THRESHOLD = 256 * 1024 * 1024

# Canonical artifact schema versions the v2 gate binds (spec §6/§8).
EDGE_SCHEMA = "c1-c2-edge-v2"
PEAK_SCHEMA = "c2-peak-frontier-v1"
PROTO_SCHEMA = "region-prototype-v2"
AUDIT_SCHEMA = "c1-buffer-audit-v2"
C2_JUDGMENT_SCHEMA = "c2-judgment-v2"
CHECKPOINT_MANIFEST_SCHEMA = "c2-checkpoint-manifest-v2"
# Self-recompute policies (spec §5.2; mirror the prototype's own contracts).
ACCURACY_REL_L2 = 1e-4
ACCURACY_MAX_REL = 1e-3
RESOURCE_MIN_OCCUPANCY_PCT = 25.0
# A real P->T->E consumer outputs a full E tensor (>= this), not a scalar/reduction.
FULL_E_MIN_BYTES = 1 * 1024 * 1024
_FEASIBLE_VERDICTS = ("FEASIBLE_WITH_RECOMPUTE", "TILE_FUSION_FEASIBLE")
_LAYER_KEYS = (
    "C2_REGION_KERNEL_FEASIBILITY",
    "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK",
    "C2_JOINT_EXECUTABLE_LEVERAGE",
    "C2_CANONICAL",
)

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


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _case_field_mismatch(d, case):
    """True if any case field (``case_id`` / ``n`` / ``depth`` / ``fusion``) that ``d``
    declares disagrees with the judged case. Artifacts that carry only ``case_id`` (e.g. the
    prototype) are bound on that alone."""
    if d.get("case_id") is not None and d.get("case_id") != case.get("case_id"):
        return True
    for k in ("n", "depth", "fusion"):
        if k in d and d[k] != case.get(k):
            return True
    return False


_REDUCTION_MARKERS = ("norm", "reduce", "reduction", "sum(")


def _is_real_pte_prototype(proto, edge):
    """A genuine two-stage ``P=A@B -> T=transform(P) -> E=D@T`` prototype, not the rejected
    GEMM->norm/reduction artifact (final-review §3.2/§7.1). Requires a schema-correct record,
    a full-E GEMM consumer, no full P/T materialization, non-reduction math, and producer/
    consumer MNK matching the edge region being judged."""
    if not isinstance(proto, dict) or not proto:
        return False
    if proto.get("schema_version") != PROTO_SCHEMA:
        return False
    region = proto.get("region") or {}
    prod = region.get("producer")
    cons = region.get("consumer")
    if not (
        isinstance(prod, list) and len(prod) == 3 and all(int(x) > 0 for x in prod)
    ):
        return False
    if not (
        isinstance(cons, list) and len(cons) == 3 and all(int(x) > 0 for x in cons)
    ):
        return False
    if (
        cons[0] * cons[1] * 8 < FULL_E_MIN_BYTES
    ):  # full E tensor, not a scalar/reduction
        return False
    if not (
        proto.get("no_full_P_materialized") and proto.get("no_full_T_materialized")
    ):
        return False
    if any(m in str(proto.get("math", "")).lower() for m in _REDUCTION_MARKERS):
        return False
    ep, ec = edge.get("producer", {}), edge.get("consumer", {})
    if [ep.get("M"), ep.get("N"), ep.get("K")] != [int(x) for x in prod]:
        return False
    if [ec.get("M"), ec.get("N"), ec.get("K")] != [int(x) for x in cons]:
        return False
    return True


def _recompute_conditions(proto, peak):
    """§5.2 self-recompute from RAW fields. Self-reported booleans are diagnostic only.
    ``None`` means the field is absent -> that sub-condition is UNKNOWN (cannot confirm).
    """
    rc: dict[str, Any] = {}
    rel_l2 = proto.get("relative_l2")
    max_rel = proto.get("max_rel")
    if isinstance(rel_l2, (int, float)) and isinstance(max_rel, (int, float)):
        rc["accuracy_pass"] = bool(
            rel_l2 < ACCURACY_REL_L2 and max_rel < ACCURACY_MAX_REL
        )
    else:
        rc["accuracy_pass"] = None
    regs = proto.get("registers_per_thread")
    occ = proto.get("occupancy_pct")
    if isinstance(regs, (int, float)) and isinstance(occ, (int, float)):
        rc["resource_pass"] = bool(regs > 0 and occ >= RESOURCE_MIN_OCCUPANCY_PCT)
    else:
        rc["resource_pass"] = None
    mp = proto.get("materialized_peak_bytes")
    fp = proto.get("fused_peak_bytes")
    rc["region_peak_gain_bytes"] = (
        int(mp) - int(fp)
        if isinstance(mp, (int, float)) and isinstance(fp, (int, float))
        else None
    )
    base = peak.get("base_peak_bytes")
    after = (peak.get("anchor_window") or {}).get("peak_after_single_elimination")
    rc["single_reduction_bytes"] = (
        int(base) - int(after)
        if isinstance(base, (int, float)) and isinstance(after, (int, float))
        else None
    )
    # traffic/workspace are not split out in the prototype -> UNKNOWN. The measured allocator
    # peak already accounts for workspace, so this can never inflate a claimed gain.
    rc["traffic_gain"] = "UNKNOWN"
    rc["workspace_cost"] = "UNKNOWN"
    rcf = proto.get("producer_recompute_factor")
    rcflops = proto.get("producer_recompute_flops")
    rc["recompute_cost"] = (
        {"factor": int(rcf), "flops": int(rcflops)}
        if isinstance(rcf, (int, float)) and isinstance(rcflops, (int, float))
        else None
    )
    # latency policy needs the fused full-anchor run; otherwise UNKNOWN (not measured).
    rc["latency_policy_pass"] = (
        True if proto.get("fused_full_anchor_run") is True else None
    )
    return rc


def _binding_problems(edge, peak, proto, audit, case, file_hashes):
    """Cross-cutting case/hash/schema/contract problems. Any -> the artifacts are
    untrustworthy, so every layer is forced UNKNOWN (fail-closed, spec §8 step 1-2)."""
    probs = []
    for name, d in (("edge", edge), ("peak", peak), ("audit", audit)):
        if not isinstance(d, dict) or not d:
            probs.append(f"{name} artifact missing")
            continue
        if _case_field_mismatch(d, case):
            probs.append(
                f"{name} case fields disagree with judged {case.get('case_id')}"
            )
    if isinstance(proto, dict) and proto and _case_field_mismatch(proto, case):
        probs.append(
            f"prototype case fields disagree with judged {case.get('case_id')}"
        )
    # source-HLO hash triangle: edge / peak / audit must agree
    h_edge = (edge.get("source_hlo") or {}).get("sha256")
    hlo_hashes = {
        h
        for h in (h_edge, peak.get("source_hlo_sha256"), audit.get("source_hlo_sha256"))
        if h
    }
    if len(hlo_hashes) > 1:
        probs.append("source HLO hash mismatch across edge/peak/audit")
    # edge contract: exact trace with a closed inverse mapping
    if edge.get("trace_status") != "EXACT":
        probs.append(f"edge trace_status={edge.get('trace_status')} (not EXACT)")
    t = edge.get("transform") or {}
    if not t.get("inverse_index_map") or not t.get("steps"):
        probs.append("edge transform missing steps/inverse_index_map")
    if audit.get("allocation_source") != "xla_buffer_assignment":
        probs.append(
            f"audit allocation_source={audit.get('allocation_source')} (not real)"
        )
    # on-disk hashes (when provided by the run layer)
    if file_hashes:
        checks = (
            ("source_hlo", h_edge),
            ("allocation_audit", (edge.get("allocation_audit") or {}).get("sha256")),
            ("edge_map", peak.get("edge_map_sha256")),
            ("buffer_assignment", audit.get("buffer_assignment_sha256")),
        )
        for key, recorded in checks:
            on_disk = file_hashes.get(key)
            if on_disk and recorded and on_disk != recorded:
                probs.append(f"on-disk {key} hash != recorded")
        for key in ("peak_frontier", "prototype"):
            if not file_hashes.get(key):
                probs.append(f"on-disk {key} hash missing")
    return probs


def _region_layer(proto, edge, rc):
    """C2_REGION_KERNEL_FEASIBILITY: can the real P->T->E region be computed without
    materializing full P/T (spec §5.1)? Only a real prototype or a definitive blocker
    gives PASS/FAIL; everything else is UNKNOWN."""
    if not _is_real_pte_prototype(proto, edge):
        return (
            "UNKNOWN",
            "no real P->T->E prototype (missing / GEMM->norm / MNK mismatch)",
        )
    verdict = proto.get("verdict")
    if verdict in _FEASIBLE_VERDICTS:
        acc, res = rc["accuracy_pass"], rc["resource_pass"]
        if acc is None or res is None:
            return (
                "UNKNOWN",
                "prototype feasible but accuracy/resource not confirmable",
            )
        if acc and res:
            scope = (
                ""
                if proto.get("fused_full_anchor_run")
                else (
                    " (fused full-anchor latency not measured; feasibility from compile + "
                    "representative-contract correctness)"
                )
            )
            return ("PASS", f"real kernel feasible{scope}")
        return (
            "FAIL",
            "prototype claims feasible but recomputed accuracy/resource fail",
        )
    if verdict == "NOT_FEASIBLE":
        return ("FAIL", "real P->T->E prototype definitively NOT_FEASIBLE")
    return ("UNKNOWN", f"prototype verdict {verdict} is not a definitive kernel result")


def _single_layer(rc):
    """C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK: replacing only the anchor pair, holding the
    rest of the program fixed. A legitimate route-local negative (spec §5.2)."""
    sr = rc["single_reduction_bytes"]
    if sr is None:
        return ("UNKNOWN", "single-anchor peak counterfactual unavailable")
    if sr >= C2_MEMORY_THRESHOLD:
        return ("PASS", f"single-anchor patch reduces peak by {sr} B >= threshold")
    return (
        "FAIL",
        f"single-anchor patch reduces peak by only {sr} B < threshold "
        f"(unchanged-rest-of-program counterfactual; the peak is structural)",
    )


def _joint_layer(peak):
    """C2_JOINT_EXECUTABLE_LEVERAGE (spec §5.3). The frontier joint_model is a COUNTERFACTUAL
    (workspace/recompute uncounted) -> an OPTIMISTIC UPPER BOUND on the reduction:
      * upper bound < threshold  -> genuinely infeasible -> FAIL
      * upper bound >= threshold, no executable joint impl -> UNKNOWN (workspace may eat it)
      * recognized executable joint PASS + model meets threshold -> PASS."""
    max_red = (peak.get("joint_model") or {}).get("max_joint_reduction_bytes")
    diag = peak.get("diagnostics") or {}
    if not isinstance(max_red, (int, float)):
        return ("UNKNOWN", "joint model max reduction unavailable")
    if diag.get("joint_executable_status") == "PASS" and max_red >= C2_MEMORY_THRESHOLD:
        return ("PASS", "executable joint implementation meets threshold")
    if max_red < C2_MEMORY_THRESHOLD:
        return ("FAIL", "joint model upper-bound reduction < threshold (infeasible)")
    return (
        "UNKNOWN",
        "joint model meets threshold but no executable joint implementation",
    )


def _compose_canonical(region, joint):
    """Canonical C2 composition (spec §5.4). A single-pair peak FAIL never becomes a
    canonical FAIL on its own -- only a definitive region-kernel blocker or a proven joint
    verdict can."""
    if region == "FAIL":
        return "FAIL"
    if region == "UNKNOWN":
        return "UNKNOWN"
    if joint == "PASS":
        return "PASS"
    if joint == "FAIL":
        return "FAIL"
    return "UNKNOWN"


def judge_c2_canonical(edge, peak, prototype, audit, *, case=None, file_hashes=None):
    """Fail-closed canonical C2 v2 gate (spec §5.4 / §8 / plan §5).

    Consumes the Task 2 edge map (``c1-c2-edge-v2``) + Task 3 peak frontier
    (``c2-peak-frontier-v1``) + Task 4 region prototype (``region-prototype-v2``) + Task 1
    allocation audit (``c1-buffer-audit-v2``). Processing order (spec §8):
      1. schema/case/hash binding -> any problem forces every layer UNKNOWN;
      2. self-recompute correctness/resource/cost/peak conditions from raw fields (§5.2);
      3. the three independent layers (region kernel / single-patch / joint leverage);
      4. canonical composition (§5.4).
    Returns a dict with ``status`` (== ``layers["C2_CANONICAL"]``), ``layers``,
    ``recomputed``, ``binding``, ``diagnostic_self_reported``, and ``reason``.
    """
    case = case or {}
    file_hashes = file_hashes or {}
    problems = _binding_problems(edge, peak, prototype, audit, case, file_hashes)
    rc = _recompute_conditions(prototype if isinstance(prototype, dict) else {}, peak)
    diag = peak.get("diagnostics") or {}
    diagnostic_self_reported = {
        "prototype_verdict": (
            prototype.get("verdict") if isinstance(prototype, dict) else None
        ),
        "prototype_correct": (
            prototype.get("correct") if isinstance(prototype, dict) else None
        ),
        "prototype_memory_policy_met": (
            prototype.get("memory_policy_met") if isinstance(prototype, dict) else None
        ),
        "fused_full_anchor_run": (
            prototype.get("fused_full_anchor_run")
            if isinstance(prototype, dict)
            else None
        ),
        "frontier_single_anchor_patch_status": diag.get("single_anchor_patch_status"),
        "frontier_joint_model_status": diag.get("joint_model_status"),
    }
    if problems:
        layers = {k: "UNKNOWN" for k in _LAYER_KEYS}
        reason = "fail-closed UNKNOWN: " + "; ".join(problems)
    else:
        r = _region_layer(prototype, edge, rc)
        s = _single_layer(rc)
        jo = _joint_layer(peak)
        layers = {
            "C2_REGION_KERNEL_FEASIBILITY": r[0],
            "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK": s[0],
            "C2_JOINT_EXECUTABLE_LEVERAGE": jo[0],
        }
        layers["C2_CANONICAL"] = _compose_canonical(
            layers["C2_REGION_KERNEL_FEASIBILITY"],
            layers["C2_JOINT_EXECUTABLE_LEVERAGE"],
        )
        reason = (
            f"region={r[0]} ({r[1]}) | single={s[0]} ({s[1]}) | joint={jo[0]} ({jo[1]}) "
            f"-> canonical={layers['C2_CANONICAL']}"
        )
    return {
        "schema_version": C2_JUDGMENT_SCHEMA,
        "basis": "hlo_use_def",
        "case_id": case.get("case_id", ""),
        "status": layers["C2_CANONICAL"],
        "layers": layers,
        "recomputed": rc,
        "binding": {
            "case": case,
            "binding_ok": not problems,
            "problems": problems,
            "file_hashes": file_hashes,
        },
        "diagnostic_self_reported": diagnostic_self_reported,
        "memory_threshold_bytes": C2_MEMORY_THRESHOLD,
        "reason": reason,
    }


def _write_checkpoint_manifest_v2(case_id, payload, file_hashes):
    """Task 5 §5.3 provenance manifest: all input + judgment hashes, the command set, an
    environment fingerprint, per-layer case statuses, and the dirty-worktree flag. Does NOT
    replace the final Task 11 manifest; it guarantees the Task 1-5 evidence chain."""
    from results._phase0.run_context import _versions

    versions = _versions()
    env_hash = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout
        dirty = bool(porcelain.strip())
    except Exception:
        dirty = None
    layers = payload.get("layers", {})
    manifest = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "case_id": case_id,
        "generated_at_epoch": int(time.time()),
        "case_statuses": {
            case_id: {
                "C2_CANONICAL": payload.get("status"),
                "C2_REGION_KERNEL_FEASIBILITY": layers.get(
                    "C2_REGION_KERNEL_FEASIBILITY"
                ),
                "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK": layers.get(
                    "C2_SINGLE_ANCHOR_PATCH_EXECUTABLE_PEAK"
                ),
                "C2_JOINT_EXECUTABLE_LEVERAGE": layers.get(
                    "C2_JOINT_EXECUTABLE_LEVERAGE"
                ),
            }
        },
        "artifact_hashes": {
            "source_hlo": file_hashes.get("source_hlo"),
            "buffer_assignment": file_hashes.get("buffer_assignment"),
            "allocation_audit": file_hashes.get("allocation_audit"),
            "edge_map": file_hashes.get("edge_map"),
            "peak_frontier": file_hashes.get("peak_frontier"),
            "prototype": file_hashes.get("prototype"),
            "c2_judgment": _sha256_file(JUDGMENT_JSON_PATH),
        },
        "environment_hash": env_hash,
        "package_versions": versions,
        "dirty_worktree": dirty,
        "commands": {
            "edge_map": "python results/_phase0/c1_to_c2_map.py",
            "peak_frontier": "python results/_phase0/c2_peak_analysis.py",
            "region_proto": "python results/_phase0/region_proto.py",
            "c2_gate": "python results/_phase0/c2.py --n <n> --depth <depth>",
        },
    }
    os.makedirs(os.path.dirname(CHECKPOINT_MANIFEST_PATH), exist_ok=True)
    with open(CHECKPOINT_MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)


def run_c2_canonical(n, depth, fusion="default"):
    """Canonical C2 v2 verdict from the on-disk Task 1-4 artifacts. Computes the on-disk
    hashes the gate binds against, writes ``c2_judgment.json`` (fresh, single-case) and
    ``c2_checkpoint_manifest.json``. The SOLE canonical writer."""
    case_id = f"n{n}_d{depth}_{fusion}"
    case = {"n": n, "depth": depth, "fusion": fusion, "case_id": case_id}
    edge = _load_json(EDGE_MAP_JSON_PATH)
    peak = _load_json(PEAK_FRONTIER_JSON_PATH)
    proto = _load_json(REGION_PROTOTYPE_JSON_PATH)
    audit_path = _audit_json_path(n, depth, fusion)
    audit = _load_json(audit_path)
    hlo_path = (edge.get("source_hlo") or {}).get("path") or (
        f"{OUT_DIR}/c1_optimized_hlo/n{n}_d{depth}_exp_{fusion}.hlo"
    )
    ba_path = audit.get("buffer_assignment_path") or peak.get("buffer_assignment_path")
    file_hashes = {
        "source_hlo": _sha256_file(hlo_path),
        "allocation_audit": _sha256_file(audit_path),
        "edge_map": _sha256_file(EDGE_MAP_JSON_PATH),
        "peak_frontier": _sha256_file(PEAK_FRONTIER_JSON_PATH),
        "prototype": _sha256_file(REGION_PROTOTYPE_JSON_PATH),
        "buffer_assignment": _sha256_file(ba_path),
    }
    judgment = judge_c2_canonical(
        edge, peak, proto, audit, case=case, file_hashes=file_hashes
    )
    payload = {
        **judgment,
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "edge_producer": (edge.get("producer") or {}).get("hlo_value_id"),
        "edge_consumer": (edge.get("consumer") or {}).get("hlo_value_id"),
        "artifact_paths": {
            "edge_map": EDGE_MAP_JSON_PATH,
            "peak_frontier": PEAK_FRONTIER_JSON_PATH,
            "prototype": REGION_PROTOTYPE_JSON_PATH,
            "audit": audit_path,
            "source_hlo": hlo_path,
            "buffer_assignment": ba_path,
        },
    }
    with open(JUDGMENT_JSON_PATH, "w") as fh:
        json.dump({case_id: payload}, fh, indent=2)
    _write_checkpoint_manifest_v2(case_id, payload, file_hashes)
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
