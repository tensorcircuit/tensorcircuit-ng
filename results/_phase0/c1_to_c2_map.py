"""C1 anchor -> HLO SSA producer/consumer edge map (rereview §5.2).

Recovers the REAL producer->consumer edge for the 512 MiB C1 anchor buffer directly
from the production expectation executable's optimized HLO SSA use-def graph (ground
truth), NOT from the cotengra state tree. See the canonical-completion plan's Global
Constraints ("contraction contractor"): production contracts via the original path
contractor, so the cotengra tree is a different decomposition and cannot be matched by
extent. This module replaces the earlier cotengra-extent-match design.

Pure text parsing over HLO already saved by ``c1.measure_case``; the anchor
``hlo_value_id`` comes from Task 1's ``c1_buffer_audit.audit_buffer_assignment``.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque

OUT_DIR = "results/phase0"
HLO_DIR = f"{OUT_DIR}/c1_optimized_hlo"
AUDIT_DIR = f"{OUT_DIR}/c1_buffer_assignment"
EDGE_CSV_PATH = f"{OUT_DIR}/c1_c2_edge_map.csv"

# Top-level op def: optional ROOT, %ssa-name = <rhs>. SSA names carry `-`/`.`.
_OP_DEF_RE = re.compile(r"^\s*(?:ROOT\s+)?(%[a-zA-Z0-9_.\-]+)\s*=\s*(.+)$")
# The opcode is the identifier immediately preceding the first `(` in the rhs
# (skips the leading TYPE[dims]{layout} / tuple, which use `[`/`{` not `(`).
_OPCODE_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_\-]*)\s*\(")
# Every %ssa-name token (operand references).
_REF_RE = re.compile(r"%[a-zA-Z0-9_.\-]+")
# `%name = (tuple) custom-call` on __cublas$gemm lines.
_CUBLAS_CALL_DEF_RE = re.compile(r"(%[a-zA-Z0-9_.\-]+)\s*=\s*\(([^)]*)\)\s+custom-call")
_TYPED_ELEM_RE = re.compile(r"\b([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]\{[^}]*\}")

# Passthrough ops: forward the buffer without consuming it -> keep tracing.
_PASSTHROUGH = {
    "get-tuple-element",
    "bitcast",
    "transpose",
    "convert",
    "reshape",
    "copy",
    "reduce-precision",
    "broadcast-in-dim",
}


def _bare(name: str) -> str:
    """Strip the leading `%` from an SSA name for output lists."""
    return name[1:] if name.startswith("%") else name


def _iter_op_defs(hlo_text: str):
    """Yield ``(defined_name, rhs)`` for every op-def line (``%name = <rhs>``).

    HLO body-definition lines (``%comp (sig) -> ret {``) lack ``=`` and so do not match
    ``_OP_DEF_RE`` -- they are naturally excluded. Dataflow ops live inside computation
    bodies (brace-depth 1); fusion-body-internal ops also match but reference fusion-local
    params (never the global anchor SSA), so they are harmless to the use-def BFS.
    """
    for line in hlo_text.splitlines():
        m = _OP_DEF_RE.match(line)
        if m:
            yield m.group(1), m.group(2)


def _build_ssa_dims(hlo_text: str) -> dict:
    """Map each op def's SSA name -> its dim list (from the leading typed shape)."""
    dims: dict = {}
    for defined, rhs in _iter_op_defs(hlo_text):
        tm = re.search(r"\b([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]", rhs)
        if tm:
            dims[defined] = [int(x) for x in tm.group(2).split(",")]
    return dims


def _mnk_from_custom_call(hlo_text: str, anchor_id: str):
    """Recover (M, N, K) for the anchor ``__cublas$gemm`` custom-call.

    M,N = the c64 result element dims of the output tuple; K = the operand dim that is
    neither M nor N (A=[M,K] x B=[K,N]).
    """
    ssa_dims = _build_ssa_dims(hlo_text)
    for line in hlo_text.splitlines():
        if "__cublas$gemm" not in line:
            continue
        m = _CUBLAS_CALL_DEF_RE.search(line)
        if not m or m.group(1) != anchor_id:
            continue
        tuple_body = m.group(2)
        out_dims = None
        for elem in _TYPED_ELEM_RE.finditer(tuple_body):
            if elem.group(1) != "c64":
                continue
            d = [int(x) for x in elem.group(2).split(",")]
            if len(d) >= 2:
                out_dims = d
                break
        if out_dims is None:
            raise ValueError(
                f"no c64 result element in {anchor_id} tuple: {tuple_body}"
            )
        M, N = out_dims[0], out_dims[1]
        arg_m = re.search(r"custom-call\(([^)]*)\)", line)
        operands = _REF_RE.findall(arg_m.group(1)) if arg_m else []
        K = 0
        for op in operands:
            od = ssa_dims.get(op)
            if not od or len(od) != 2:
                continue
            cand = [d for d in od if d != M and d != N]
            if len(cand) == 1:
                K = cand[0]
                break
        if K == 0:  # fallback: first operand dim that is not M
            for op in operands:
                od = ssa_dims.get(op)
                if od:
                    for d in od:
                        if d != M:
                            K = d
                            break
                if K:
                    break
        return M, N, K
    raise ValueError(f"anchor custom-call {anchor_id} not found in HLO")


def _build_defs(hlo_text: str) -> list:
    """List of ``(defined_name, opcode, operand_names_set)`` for op defs."""
    defs = []
    for defined, rhs in _iter_op_defs(hlo_text):
        opc_m = _OPCODE_RE.search(rhs)
        opcode = opc_m.group(1) if opc_m else ""
        operands = set(_REF_RE.findall(rhs))
        operands.discard(defined)
        defs.append((defined, opcode, operands))
    return defs


def _consumers(hlo_text: str, anchor_id: str):
    """BFS over SSA use-def from ``anchor_id`` to terminal consumers.

    Returns ``(consumer_ops, traced_through)`` as lists of bare SSA names. Passthrough
    ops (get-tuple-element/bitcast/transpose/...) expand the frontier; any other opcode
    (fusion/custom-call/dot_general/add/...) is a terminal consumer and is recorded.
    """
    defs = _build_defs(hlo_text)
    frontier = deque([anchor_id])
    seen = {anchor_id}
    traced: list[str] = []
    consumers: list[str] = []
    while frontier:
        cur = frontier.popleft()
        for defined, opcode, operands in defs:
            if cur not in operands or defined in seen:
                continue
            seen.add(defined)
            if opcode in _PASSTHROUGH:
                traced.append(_bare(defined))
                frontier.append(defined)
            else:
                consumers.append(_bare(defined))
    return consumers, traced


def build_c1_edge_map(hlo_text: str, anchor_value_id: str) -> list[dict]:
    """The producer->consumer edge record(s) for the anchor buffer, from HLO SSA."""
    M, N, K = _mnk_from_custom_call(hlo_text, anchor_value_id)
    consumers, traced = _consumers(hlo_text, anchor_value_id)
    return [
        {
            "hlo_value_id": anchor_value_id,
            "M": M,
            "N": N,
            "K": K,
            "buffer_bytes": M * N * 8,
            "producer_op": "__cublas$gemm",
            "consumer_ops": consumers,
            "consumer_count": len(consumers),
            "traced_through": traced,
        }
    ]


_EDGE_CSV_COLUMNS = [
    "n",
    "depth",
    "fusion",
    "hlo_value_id",
    "M",
    "N",
    "K",
    "buffer_bytes",
    "producer_op",
    "consumer_ops",
    "traced_through",
    "consumer_count",
    "note",
]


def map_anchor_for_case(n: int, depth: int, fusion: str = "default") -> dict:
    """Read Task 1's audit JSON + the HLO, build the anchor edge map, write the CSV.

    The C2 node identity is the HLO consumer op SSA name (NOT a cotengra node id).
    """
    from results._phase0.c1 import upsert_csv_row

    audit_path = f"{AUDIT_DIR}/n{n}_d{depth}_{fusion}.json"
    with open(audit_path) as fh:
        audit = json.load(fh)
    anchors = [b for b in audit["buffers"] if b.get("is_anchor")]
    hlo_path = f"{HLO_DIR}/n{n}_d{depth}_exp_{fusion}.hlo"
    with open(hlo_path) as fh:
        hlo_text = fh.read()

    if not anchors:
        row = {
            "n": n,
            "depth": depth,
            "fusion": fusion,
            "hlo_value_id": "",
            "M": 0,
            "N": 0,
            "K": 0,
            "buffer_bytes": 0,
            "producer_op": "",
            "consumer_ops": "",
            "traced_through": "",
            "consumer_count": 0,
            "note": "no anchor in audit",
        }
        upsert_csv_row(
            EDGE_CSV_PATH, row, _EDGE_CSV_COLUMNS, key_cols=["n", "depth", "fusion"]
        )
        return row

    e = build_c1_edge_map(hlo_text, anchors[0]["hlo_value_id"])[0]
    row = {
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "hlo_value_id": e["hlo_value_id"],
        "M": e["M"],
        "N": e["N"],
        "K": e["K"],
        "buffer_bytes": e["buffer_bytes"],
        "producer_op": e["producer_op"],
        "consumer_ops": ";".join(e["consumer_ops"]),
        "traced_through": ";".join(e["traced_through"]),
        "consumer_count": e["consumer_count"],
        "note": "",
    }
    upsert_csv_row(
        EDGE_CSV_PATH, row, _EDGE_CSV_COLUMNS, key_cols=["n", "depth", "fusion"]
    )
    return row
