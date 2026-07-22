"""C1 anchor -> HLO SSA producer/consumer edge map (rereview §5.2, correction-plan Task B).

Traces the REAL producer->consumer edge for the 512 MiB C1 anchor buffer through the
production expectation executable's optimized HLO SSA, PIERCING layout-only fusions to
reach the true terminal contraction consumer. The production region is a TWO-STAGE GEMM:

  %custom-call.497  P = A[4096,1024] @ B[1024,16384] -> c64[4096,16384]   (512 MiB anchor)
    -> get-tuple-element.246.0
    -> loop_transpose_fusion.2  (calls %fused_transpose.2: parameter+bitcast+transpose, LAYOUT)
    -> bitcast.1317.0           T = c64[64,1048576]
    -> %custom-call.498         E = D[64,64] @ T -> c64[64,1048576]       (terminal, 512 MiB out)

A fusion is PASSTHROUGH iff its called computation body is layout-only (parameter/
get-tuple-element/bitcast/reshape/transpose/copy/convert); a compute fusion
(dot/reduce/arithmetic/slice/...) or a raw contraction (custom-call/dot_general) is
terminal. Unclassifiable fusions are terminal and flagged -- never auto-PASS.

Pure text parsing over HLO saved by c1.measure_case; the anchor hlo_value_id comes from
c1_buffer_audit.audit_buffer_assignment.
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
EDGE_JSON_PATH = f"{OUT_DIR}/c1_c2_edge_map.json"

_OP_DEF_RE = re.compile(r"^\s*(?:ROOT\s+)?(%[a-zA-Z0-9_.\-]+)\s*=\s*(.+)$")
_OPCODE_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_\-]*)\s*\(")
_REF_RE = re.compile(r"%[a-zA-Z0-9_.\-]+")
_CUBLAS_CALL_DEF_RE = re.compile(r"(%[a-zA-Z0-9_.\-]+)\s*=\s*\(([^)]*)\)\s+custom-call")
_TYPED_ELEM_RE = re.compile(r"\b([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]\{[^}]*\}")
# computation body definition: starts with `%name(`, ends the line with `{`
_COMP_DEF_RE = re.compile(r"^\s*(%[a-zA-Z0-9_.\-]+)\s*\(")
_CALLS_RE = re.compile(r"calls=(%[a-zA-Z0-9_.\-]+)")

# Elementwise/layout ops that forward a buffer without computing on it -> keep tracing.
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
# Opcodes legal inside a LAYOUT-ONLY fusion body (no compute semantics).
_LAYOUT_OPCODES = {
    "parameter",
    "get-tuple-element",
    "bitcast",
    "reshape",
    "transpose",
    "copy",
    "convert",
}
# Opcodes that make a fusion a real COMPUTE consumer (terminal).
_COMPUTE_OPCODES = {
    "dot",
    "dot_general",
    "custom-call",
    "reduce",
    "reduce-product",
    "add",
    "subtract",
    "multiply",
    "divide",
    "slice",
    "dynamic-slice",
    "gather",
    "scatter",
    "fft",
    "convolution",
    "select",
    "maximum",
    "minimum",
    "exponential",
    "log",
    "sqrt",
    "rsqrt",
    "tanh",
    "sine",
    "cosine",
    "abs",
    "negate",
}


def _bare(name: str) -> str:
    return name[1:] if name.startswith("%") else name


def _ssa(name: str) -> str:
    return name if name.startswith("%") else "%" + name


def _iter_op_defs(hlo_text: str):
    """Yield ``(defined_name, rhs)`` for every op-def line (``%name = <rhs>``)."""
    for line in hlo_text.splitlines():
        m = _OP_DEF_RE.match(line)
        if m:
            yield m.group(1), m.group(2)


def _build_computation_bodies(hlo_text: str) -> dict:
    """Map computation-name -> set(body opcodes), for ``%name (...) -> ... { body }`` blocks.

    Body-definition lines end with ``{`` (op-def lines end with ``}`` from metadata), so they
    are distinguished from op-defs. Body opcodes are collected by brace-depth tracking.
    """
    bodies: dict = {}
    cur = None
    depth = 0
    for line in hlo_text.splitlines():
        if depth == 0:
            m = _COMP_DEF_RE.match(line)
            if m and line.rstrip().endswith("{"):
                cur = m.group(1)
                bodies.setdefault(cur, set())
        elif cur is not None:
            mo = _OP_DEF_RE.match(line)
            if mo:
                opc = _OPCODE_RE.search(mo.group(2))
                if opc:
                    bodies[cur].add(opc.group(1))
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            cur = None
            depth = 0
    return bodies


def _classify_fusion(calls_target, bodies) -> str:
    """layout_passthrough | compute_consumer | unknown (unknown is never auto-PASS)."""
    if not calls_target or calls_target not in bodies:
        return "unknown"
    bops = bodies[calls_target]
    if any(o in _COMPUTE_OPCODES for o in bops):
        return "compute_consumer"
    if bops and all(o in _LAYOUT_OPCODES for o in bops):
        return "layout_passthrough"
    return "unknown"


def _build_ssa_dims(hlo_text: str) -> dict:
    dims: dict = {}
    for defined, rhs in _iter_op_defs(hlo_text):
        tm = re.search(r"\b([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]", rhs)
        if tm:
            dims[defined] = [int(x) for x in tm.group(2).split(",")]
    return dims


def _mnk_from_custom_call(hlo_text: str, anchor_id: str):
    """(M, N, K) for a ``__cublas$gemm`` custom-call: M,N from the c64 result element,
    K from the operand dim that is neither M nor N (A=[M,K] x B=[K,N])."""
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
        # Authoritative K from the gemm backend_config dot_dimension_numbers (the
        # shape-value heuristic is ambiguous when the contracting dim equals M or N).
        K = 0
        lhs_c = re.search(r'"lhs_contracting_dimensions":\[([^\]]*)\]', line)
        rhs_c = re.search(r'"rhs_contracting_dimensions":\[([^\]]*)\]', line)
        if operands and lhs_c and rhs_c and len(operands) >= 2:
            li = [
                int(x) for x in lhs_c.group(1).replace('"', "").split(",") if x.strip()
            ]
            ri = [
                int(x) for x in rhs_c.group(1).replace('"', "").split(",") if x.strip()
            ]
            ad = ssa_dims.get(operands[0])
            bd = ssa_dims.get(operands[1])
            if ad and bd and len(li) == 1 and len(ri) == 1:
                K = ad[li[0]]
                if len(ad) == 2:
                    M = ad[1 - li[0]]
                if len(bd) == 2:
                    N = bd[1 - ri[0]]
        if K == 0:
            # heuristic fallback: operand dim that is neither M nor N
            for op in operands:
                od = ssa_dims.get(op)
                if not od or len(od) != 2:
                    continue
                cand = [d for d in od if d != M and d != N]
                if len(cand) == 1:
                    K = cand[0]
                    break
        return M, N, K
    raise ValueError(f"custom-call {anchor_id} not found in HLO")


def _build_defs(hlo_text: str) -> list:
    """List of ``(defined, opcode, operand_names_set, calls_target)`` for op defs."""
    defs = []
    for defined, rhs in _iter_op_defs(hlo_text):
        opc_m = _OPCODE_RE.search(rhs)
        opcode = opc_m.group(1) if opc_m else ""
        operands = set(_REF_RE.findall(rhs))
        operands.discard(defined)
        calls = None
        cm = _CALLS_RE.search(rhs)
        if cm:
            calls = cm.group(1)
        defs.append((defined, opcode, operands, calls))
    return defs


def _consumers(hlo_text: str, anchor_id: str):
    """BFS from anchor_id through passthrough ops AND layout-only fusions to terminal
    contraction consumers. Returns ``(consumer_bare_names, traced_bare_names, consumer_mnk)``.

    ``consumer_mnk`` maps a terminal contraction (custom-call/dot_general) bare name to its
    (M, N, K), so the gate can record the real consumer's shape (e.g. E = D@T -> [64,1048576]).
    """
    defs = _build_defs(hlo_text)
    bodies = _build_computation_bodies(hlo_text)
    frontier = deque([anchor_id])
    seen = {anchor_id}
    traced: list[str] = []
    consumers: list[str] = []
    consumer_mnk: dict = {}
    while frontier:
        cur = frontier.popleft()
        for defined, opcode, operands, calls in defs:
            if cur not in operands or defined in seen:
                continue
            seen.add(defined)
            if opcode in _PASSTHROUGH:
                traced.append(_bare(defined))
                frontier.append(defined)
            elif opcode == "fusion":
                if _classify_fusion(calls, bodies) == "layout_passthrough":
                    traced.append(_bare(defined))
                    frontier.append(defined)
                else:
                    consumers.append(
                        _bare(defined)
                    )  # compute_consumer / unknown -> terminal
            else:
                consumers.append(_bare(defined))
                if opcode in ("custom-call", "dot_general", "dot"):
                    try:
                        consumer_mnk[_bare(defined)] = _mnk_from_custom_call(
                            hlo_text, defined
                        )
                    except ValueError:
                        pass
    return consumers, traced, consumer_mnk


def build_c1_edge_map(hlo_text: str, anchor_value_id: str) -> list[dict]:
    """The producer->terminal-consumer edge record for the anchor, piercing layout fusions."""
    M, N, K = _mnk_from_custom_call(hlo_text, anchor_value_id)
    consumers, traced, consumer_mnk = _consumers(hlo_text, anchor_value_id)
    terminal_bare = consumers[0] if consumers else ""
    cmnk = consumer_mnk.get(terminal_bare)
    return [
        {
            "producer_hlo_value_id": anchor_value_id,
            "producer_M": M,
            "producer_N": N,
            "producer_K": K,
            "producer_output_bytes": M * N * 8,
            "passthrough_hlo_ids": traced,
            "terminal_consumer_hlo_value_id": _ssa(terminal_bare),
            "consumer_count": len(consumers),
            "consumer_M": cmnk[0] if cmnk else 0,
            "consumer_N": cmnk[1] if cmnk else 0,
            "consumer_K": cmnk[2] if cmnk else 0,
            "consumer_output_bytes": (cmnk[0] * cmnk[1] * 8) if cmnk else 0,
        }
    ]


_EDGE_CSV_COLUMNS = [
    "n",
    "depth",
    "fusion",
    "producer_hlo_value_id",
    "producer_M",
    "producer_N",
    "producer_K",
    "producer_output_bytes",
    "passthrough_hlo_ids",
    "terminal_consumer_hlo_value_id",
    "consumer_count",
    "consumer_M",
    "consumer_N",
    "consumer_K",
    "consumer_output_bytes",
    "note",
]


def map_anchor_for_case(n: int, depth: int, fusion: str = "default") -> dict:
    """Read Task 1's audit JSON + the HLO, build the (piercing) edge map, write CSV + JSON.

    The C2 node identity is the HLO terminal consumer op SSA name (NOT a cotengra node id).
    """
    from results._phase0.c1 import upsert_csv_row

    audit_path = f"{AUDIT_DIR}/n{n}_d{depth}_{fusion}.json"
    with open(audit_path) as fh:
        audit = json.load(fh)
    anchors = [b for b in audit["buffers"] if b.get("is_anchor")]
    hlo_path = f"{HLO_DIR}/n{n}_d{depth}_exp_{fusion}.hlo"
    with open(hlo_path) as fh:
        hlo_text = fh.read()

    case_id = f"n{n}_d{depth}"
    if not anchors:
        row = {
            "n": n,
            "depth": depth,
            "fusion": fusion,
            "producer_hlo_value_id": "",
            "producer_M": 0,
            "producer_N": 0,
            "producer_K": 0,
            "producer_output_bytes": 0,
            "passthrough_hlo_ids": "",
            "terminal_consumer_hlo_value_id": "",
            "consumer_count": 0,
            "consumer_M": 0,
            "consumer_N": 0,
            "consumer_K": 0,
            "consumer_output_bytes": 0,
            "note": "no anchor in audit",
        }
        upsert_csv_row(
            EDGE_CSV_PATH, row, _EDGE_CSV_COLUMNS, key_cols=["n", "depth", "fusion"]
        )
        with open(EDGE_JSON_PATH, "w") as fh:
            json.dump({"cases": {}, "last_case": row}, fh, indent=2)
        return row

    rec = build_c1_edge_map(hlo_text, anchors[0]["hlo_value_id"])[0]
    row = {
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "producer_hlo_value_id": rec["producer_hlo_value_id"],
        "producer_M": rec["producer_M"],
        "producer_N": rec["producer_N"],
        "producer_K": rec["producer_K"],
        "producer_output_bytes": rec["producer_output_bytes"],
        "passthrough_hlo_ids": ";".join(rec["passthrough_hlo_ids"]),
        "terminal_consumer_hlo_value_id": rec["terminal_consumer_hlo_value_id"],
        "consumer_count": rec["consumer_count"],
        "consumer_M": rec["consumer_M"],
        "consumer_N": rec["consumer_N"],
        "consumer_K": rec["consumer_K"],
        "consumer_output_bytes": rec["consumer_output_bytes"],
        "note": "",
    }
    upsert_csv_row(
        EDGE_CSV_PATH, row, _EDGE_CSV_COLUMNS, key_cols=["n", "depth", "fusion"]
    )
    with open(EDGE_JSON_PATH, "w") as fh:
        json.dump({"cases": {case_id: rec}}, fh, indent=2)
    return rec
