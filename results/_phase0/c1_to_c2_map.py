"""C1 anchor -> HLO SSA producer/consumer edge map (final-remediation Task 2, v2 schema).

Traces the REAL producer->consumer edge for the C1 anchor buffer through the production
expectation executable's optimized HLO SSA, PIERCING layout-only fusions to reach the true
terminal contraction consumer, and emits the v2 schema (spec
`docs/superpowers/specs/2026-07-22-phase0-final-review-spec.md` section 6): an exact,
invertible index transform plus producer/consumer shape/dtype/layout/bytes and hash binding.

The production region is a TWO-STAGE GEMM:

  %custom-call.497  P = A[4096,1024] @ B[1024,16384] -> c64[4096,16384]   (512 MiB anchor)
    -> get-tuple-element.246.0
    -> loop_transpose_fusion.2  (calls %fused_transpose.2: parameter+bitcast+transpose, LAYOUT)
    -> bitcast.1317.0           T = c64[64,1048576]
    -> %custom-call.498         E = D[64,64] @ T -> c64[64,1048576]       (terminal, 512 MiB out)

A fusion is PASSTHROUGH iff its called computation body is layout-only (parameter/
get-tuple-element/bitcast/reshape/transpose); a compute fusion (dot/reduce/arithmetic/
slice/...) or any convert/copy/unknown opcode is terminal and flagged -- never auto-PASS.
The transform is serialized step-by-step (bitcast dims / transpose dimensions) and turned
into a layout-aware, invertible linear permutation; the canonical gate reads the JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque

import numpy as np

OUT_DIR = "results/phase0"
HLO_DIR = f"{OUT_DIR}/c1_optimized_hlo"
AUDIT_DIR = f"{OUT_DIR}/c1_buffer_assignment"
EDGE_CSV_PATH = f"{OUT_DIR}/c1_c2_edge_map.csv"
EDGE_JSON_PATH = f"{OUT_DIR}/c1_c2_edge_map.json"

_OP_DEF_RE = re.compile(r"^\s*(?:ROOT\s+)?(%[a-zA-Z0-9_.\-]+)\s*=\s*(.+)$")
_OPCODE_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_\-]*)\s*\(")
_REF_RE = re.compile(r"%[a-zA-Z0-9_.\-]+")
_CUBLAS_CALL_DEF_RE = re.compile(r"(%[a-zA-Z0-9_.\-]+)\s*=\s*\(([^)]*)\)\s+custom-call")
# typed element WITH an optional layout: dtype[dims]{layout?}
_TYPED_ELEM_RE = re.compile(r"\b([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]\{[^}]*\}")
_TUPLE_ELEM_RE = re.compile(
    r"([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\](\{([0-9]+(?:,[0-9]+)*)\})?"
)
# computation body definition: starts with `%name(`, ends the line with `{`
_COMP_DEF_RE = re.compile(r"^\s*(%[a-zA-Z0-9_.\-]+)\s*\(")
_CALLS_RE = re.compile(r"calls=(%[a-zA-Z0-9_.\-]+)")

# Elementwise/layout ops that forward a buffer without computing on it -> keep tracing.
_PASSTHROUGH = {
    "get-tuple-element",
    "bitcast",
    "transpose",
    "reshape",
}
# Opcodes legal inside a LAYOUT-ONLY fusion body (no compute/dtype semantics). convert/copy
# are intentionally excluded: a dtype-changing op disqualifies pure-layout classification.
_LAYOUT_OPCODES = {
    "parameter",
    "get-tuple-element",
    "bitcast",
    "reshape",
    "transpose",
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

_DTYPE_BYTES = {
    "c64": 8,
    "c128": 16,
    "f64": 8,
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "s8": 1,
    "s32": 4,
    "pred": 1,
}


def _bare(name: str) -> str:
    return name[1:] if name.startswith("%") else name


def _ssa(name: str) -> str:
    return name if name.startswith("%") else "%" + name


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype, 8)


def _default_layout(ndim: int) -> list:
    return list(range(ndim))


def _shape_layout_str(shape, layout) -> str:
    if layout is not None:
        return f"{list(shape)}{{{','.join(str(x) for x in layout)}}}"
    return f"{list(shape)}"


def _iter_op_defs(hlo_text: str):
    """Yield ``(defined_name, rhs)`` for every op-def line (``%name = <rhs>``)."""
    for line in hlo_text.splitlines():
        m = _OP_DEF_RE.match(line)
        if m:
            yield m.group(1), m.group(2)


def _rhs_of(hlo_text: str, ssa_name: str) -> str:
    for defined, rhs in _iter_op_defs(hlo_text):
        if defined == ssa_name:
            return rhs
    return ""


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
    contraction consumers.

    Returns ``(consumer_bare_names, traced_bare_names, consumer_mnk, had_unpierced_fusion)``.
    ``consumer_mnk`` maps a terminal contraction (custom-call/dot_general) bare name to its
    (M, N, K). ``had_unpierced_fusion`` is True if any terminal fusion was unclassifiable
    (unknown opcode) -- which makes the trace AMBIGUOUS rather than EXACT.
    """
    defs = _build_defs(hlo_text)
    bodies = _build_computation_bodies(hlo_text)
    frontier = deque([anchor_id])
    seen = {anchor_id}
    traced: list[str] = []
    consumers: list[str] = []
    consumer_mnk: dict = {}
    had_unpierced = False
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
                cls = _classify_fusion(calls, bodies)
                if cls == "layout_passthrough":
                    traced.append(_bare(defined))
                    frontier.append(defined)
                else:
                    consumers.append(_bare(defined))
                    if cls == "unknown":
                        had_unpierced = True
            else:
                consumers.append(_bare(defined))
                if opcode in ("custom-call", "dot_general", "dot"):
                    try:
                        consumer_mnk[_bare(defined)] = _mnk_from_custom_call(
                            hlo_text, defined
                        )
                    except ValueError:
                        pass
    return consumers, traced, consumer_mnk, had_unpierced


def _c64_element_of(rhs: str):
    """(result_index, shape, layout) of the c64 element of a tuple-typed op result, or the
    first typed element for a non-tuple op. result_index counts top-level tuple positions.
    """
    elems = list(_TUPLE_ELEM_RE.finditer(rhs))
    for i, m in enumerate(elems):
        if m.group(1) == "c64":
            shape = [int(x) for x in m.group(2).split(",")]
            layout = [int(x) for x in m.group(4).split(",")] if m.group(4) else None
            return i, shape, layout
    if elems:
        m = elems[0]
        shape = [int(x) for x in m.group(2).split(",")]
        layout = [int(x) for x in m.group(4).split(",")] if m.group(4) else None
        return 0, shape, layout
    return 0, [], None


def _parse_op_def(hlo_text: str, ssa_name: str):
    """opcode/shape_out/layout_out/calls/index/dimensions for one op-def line."""
    rhs = _rhs_of(hlo_text, ssa_name)
    if not rhs:
        return None
    opc_m = _OPCODE_RE.search(rhs)
    opcode = opc_m.group(1) if opc_m else ""
    sl = _TUPLE_ELEM_RE.search(rhs)
    shape_out = [int(x) for x in sl.group(2).split(",")] if sl else []
    layout_out = (
        [int(x) for x in sl.group(4).split(",")] if (sl and sl.group(4)) else None
    )
    calls = None
    cm = _CALLS_RE.search(rhs)
    if cm:
        calls = cm.group(1)
    index = None
    im = re.search(r"\bindex=(\d+)", rhs)
    if im:
        index = int(im.group(1))
    dimensions = None
    dm = re.search(r"dimensions=\{([0-9,]+)\}", rhs)
    if dm:
        dimensions = [int(x) for x in dm.group(1).split(",") if x.strip()]
    return {
        "opcode": opcode,
        "shape_out": shape_out,
        "layout_out": layout_out,
        "calls": calls,
        "index": index,
        "dimensions": dimensions,
    }


def _iter_comp_body_op_defs(hlo_text: str, comp_name: str):
    """Yield ``(defined, rhs)`` for op-def lines inside computation comp_name's body, in order."""
    target = _ssa(comp_name)
    cur = None
    depth = 0
    for line in hlo_text.splitlines():
        if depth == 0:
            m = _COMP_DEF_RE.match(line)
            if m and m.group(1) == target and line.rstrip().endswith("{"):
                cur = target
        elif cur is not None:
            mo = _OP_DEF_RE.match(line)
            if mo:
                yield mo.group(1), mo.group(2)
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            if cur is not None:
                return
            cur = None
            depth = 0


def _parse_fusion_body_steps(
    hlo_text: str, comp_name, entry_shape, entry_layout
) -> list:
    """Ordered layout-transform steps from a layout-only fusion's computation body.

    The parameter op fixes the entry shape/layout; each subsequent bitcast/reshape/transpose
    becomes a step. A non-layout op stops the parse (the fusion should not have been pierced).
    """
    steps = []
    cur_shape, cur_layout = entry_shape, entry_layout
    for _defined, rhs in _iter_comp_body_op_defs(hlo_text, comp_name):
        opc_m = _OPCODE_RE.search(rhs)
        opcode = opc_m.group(1) if opc_m else ""
        sl = _TUPLE_ELEM_RE.search(rhs)
        shape_out = [int(x) for x in sl.group(2).split(",")] if sl else list(cur_shape)
        layout_out = (
            [int(x) for x in sl.group(4).split(",")]
            if (sl and sl.group(4))
            else list(cur_layout)
        )
        if opcode == "parameter":
            cur_shape, cur_layout = shape_out, layout_out
            continue
        if opcode in ("bitcast", "reshape"):
            steps.append(
                {
                    "op": opcode,
                    "shape_in": list(cur_shape),
                    "layout_in": list(cur_layout),
                    "shape_out": shape_out,
                    "layout_out": layout_out,
                }
            )
        elif opcode == "transpose":
            dm = re.search(r"dimensions=\{([0-9,]+)\}", rhs)
            dims = [int(x) for x in dm.group(1).split(",")] if dm else []
            steps.append(
                {
                    "op": "transpose",
                    "dimensions": dims,
                    "shape_in": list(cur_shape),
                    "layout_in": list(cur_layout),
                    "shape_out": shape_out,
                    "layout_out": layout_out,
                }
            )
        else:
            break
        cur_shape, cur_layout = shape_out, layout_out
    return steps


def _transform_steps_from_chain(hlo_text: str, traced, producer_shape, producer_layout):
    """Walk the pierced passthrough chain (anchor data -> terminal consumer input) and emit
    ordered transform steps. Returns (steps, hlo_ids, result_index, output_shape, output_layout).
    """
    steps = []
    hlo_ids = []
    result_index = None
    cur_shape, cur_layout = list(producer_shape), list(producer_layout)
    for name in traced:
        op = _parse_op_def(hlo_text, _ssa(name))
        if op is None:
            continue
        hlo_ids.append(name)
        opcode = op["opcode"]
        out_layout = (
            op["layout_out"] if op["layout_out"] is not None else list(cur_layout)
        )
        if opcode == "get-tuple-element":
            result_index = op["index"]
            cur_shape, cur_layout = op["shape_out"], out_layout
        elif opcode == "fusion":
            steps.extend(
                _parse_fusion_body_steps(hlo_text, op["calls"], cur_shape, cur_layout)
            )
            cur_shape, cur_layout = op["shape_out"], out_layout
        elif opcode in ("bitcast", "reshape"):
            steps.append(
                {
                    "op": opcode,
                    "shape_in": list(cur_shape),
                    "layout_in": list(cur_layout),
                    "shape_out": op["shape_out"],
                    "layout_out": out_layout,
                }
            )
            cur_shape, cur_layout = op["shape_out"], out_layout
        elif opcode == "transpose":
            steps.append(
                {
                    "op": "transpose",
                    "dimensions": op["dimensions"],
                    "shape_in": list(cur_shape),
                    "layout_in": list(cur_layout),
                    "shape_out": op["shape_out"],
                    "layout_out": out_layout,
                }
            )
            cur_shape, cur_layout = op["shape_out"], out_layout
    return steps, hlo_ids, result_index, cur_shape, cur_layout


# --- layout-aware, invertible index-transform machinery ---


def _strides(shape, layout):
    """Minor-to-major strides: the most-minor dim (layout[0]) has stride 1."""
    strides = [0] * len(shape)
    acc = 1
    for dim in layout:
        strides[dim] = acc
        acc *= int(shape[dim])
    return strides


def _flatten(idx_tuple, shape, layout):
    strides = _strides(shape, layout)
    return sum(int(i) * s for i, s in zip(idx_tuple, strides))


def _unflatten(linear, shape, layout):
    strides = _strides(shape, layout)
    idx = [0] * len(shape)
    for dim in reversed(layout):  # major-to-minor extraction
        idx[dim] = linear // strides[dim]
        linear %= strides[dim]
    return tuple(idx)


def _invert_steps_to_p_index(t_idx, steps):
    """Given an output (T) multi-index, walk the steps in reverse to the input (P) multi-index."""
    idx = tuple(t_idx)
    for step in reversed(steps):
        op = step["op"]
        if op in ("bitcast", "reshape"):
            linear = _flatten(idx, step["shape_out"], step["layout_out"])
            idx = _unflatten(linear, step["shape_in"], step["layout_in"])
        elif op == "transpose":
            dims = step["dimensions"]
            in_idx = [0] * len(step["shape_in"])
            for k, val in enumerate(idx):
                in_idx[dims[k]] = val
            idx = tuple(in_idx)
        else:
            raise ValueError(f"unsupported transform op {op}")
    return idx


def _linear_permutation(steps):
    """(forward, inverse) int64 permutations of length N over the transform.

    ``forward[k]`` is the P-linear index sourced by T-linear position k, i.e. ``T[k] = P[forward[k]]``.
    ``inverse`` is its inverse permutation so ``forward[inverse] == arange(N)``.
    """
    p_shape = steps[0]["shape_in"]
    p_layout = steps[0]["layout_in"]
    t_shape = steps[-1]["shape_out"]
    t_layout = steps[-1]["layout_out"]
    n = _prod(t_shape)
    forward = np.empty(n, dtype=np.int64)
    for k in range(n):
        t_idx = _unflatten(k, t_shape, t_layout)
        p_idx = _invert_steps_to_p_index(t_idx, steps)
        forward[k] = _flatten(p_idx, p_shape, p_layout)
    inverse = np.empty(n, dtype=np.int64)
    inverse[forward] = np.arange(n, dtype=np.int64)
    return forward, inverse


def apply_forward(steps, p_flat):
    """Apply the transform P -> T over a flat array: T[k] = P[forward[k]]."""
    forward, _inverse = _linear_permutation(steps)
    return np.asarray(p_flat)[forward]


def apply_inverse(steps, t_flat):
    """Apply the inverse transform T -> P over a flat array: P[i] = T[inverse[i]]."""
    _forward, inverse = _linear_permutation(steps)
    return np.asarray(t_flat)[inverse]


def _steps_invertible(steps) -> bool:
    """Structural bijectivity: element counts preserved and every transpose is a true dim permutation."""
    for s in steps:
        if _prod(s["shape_in"]) != _prod(s["shape_out"]):
            return False
        if s["op"] == "transpose":
            dims = s["dimensions"]
            ndim = len(s["shape_in"])
            if len(dims) != ndim or sorted(dims) != list(range(ndim)):
                return False
    return True


def _step_arrow(step, invert=False) -> str:
    op = step["op"]
    label = f"transpose{{dimensions={step['dimensions']}}}" if op == "transpose" else op
    if invert:
        src = _shape_layout_str(step["shape_out"], step["layout_out"])
        dst = _shape_layout_str(step["shape_in"], step["layout_in"])
    else:
        src = _shape_layout_str(step["shape_in"], step["layout_in"])
        dst = _shape_layout_str(step["shape_out"], step["layout_out"])
    return f"{src} --{label}--> {dst}"


def _forward_map_str(steps) -> str:
    return " | ".join(_step_arrow(s) for s in steps) if steps else "identity"


def _inverse_map_str(steps) -> str:
    return (
        " | ".join(_step_arrow(s, invert=True) for s in reversed(steps))
        if steps
        else "identity"
    )


def build_c1_edge_map(hlo_text: str, anchor_value_id: str) -> dict:
    """The v2 producer -> terminal-consumer edge record for the anchor (piercing layout fusions).

    Pure (no file IO): hashes the supplied HLO text for source provenance. ``map_anchor_for_case``
    adds case_id/n/depth/fusion, source_hlo.path and the allocation_audit hash binding.
    """
    anchor_ssa = _ssa(anchor_value_id)
    M, N, K = _mnk_from_custom_call(hlo_text, anchor_ssa)
    consumers, traced, consumer_mnk, had_unpierced = _consumers(hlo_text, anchor_ssa)

    p_rhs = _rhs_of(hlo_text, anchor_ssa)
    p_idx, p_shape, p_layout = _c64_element_of(p_rhs)
    p_layout = p_layout if p_layout is not None else _default_layout(len(p_shape))

    steps, hlo_ids, _result_index, t_shape, t_layout = _transform_steps_from_chain(
        hlo_text, traced, p_shape, p_layout
    )

    terminal = consumers[0] if consumers else ""
    terminal_ssa = _ssa(terminal)
    c_rhs = _rhs_of(hlo_text, terminal_ssa) if terminal else ""
    c_idx, c_shape, c_layout = _c64_element_of(c_rhs) if c_rhs else (None, [], None)
    c_layout = c_layout if c_layout is not None else _default_layout(len(c_shape))
    cmnk = consumer_mnk.get(terminal)

    consumer_count = len(consumers)
    if consumer_count != 1 or had_unpierced or not _steps_invertible(steps):
        trace_status = "AMBIGUOUS"
    else:
        trace_status = "EXACT"

    return {
        "schema_version": "c1-c2-edge-v2",
        "producer": {
            "hlo_value_id": anchor_ssa,
            "result_index": p_idx,
            "dtype": "c64",
            "shape": p_shape,
            "layout": p_layout,
            "M": M,
            "N": N,
            "K": K,
            "bytes": _dtype_bytes("c64") * _prod(p_shape) if p_shape else 0,
        },
        "transform": {
            "hlo_ids": hlo_ids,
            "steps": steps,
            "forward_index_map": _forward_map_str(steps),
            "inverse_index_map": _inverse_map_str(steps),
            "output_shape": t_shape,
            "output_layout": t_layout,
        },
        "consumer": {
            "hlo_value_id": terminal_ssa,
            "result_index": c_idx,
            "dtype": "c64" if c_shape else "",
            "shape": c_shape,
            "layout": c_layout,
            "M": cmnk[0] if cmnk else 0,
            "N": cmnk[1] if cmnk else 0,
            "K": cmnk[2] if cmnk else 0,
            "bytes": _dtype_bytes("c64") * _prod(c_shape) if c_shape else 0,
        },
        "consumer_count": consumer_count,
        "trace_status": trace_status,
        "source_hlo": {
            "path": None,
            "sha256": hashlib.sha256(hlo_text.encode("utf-8")).hexdigest(),
        },
    }


def verify_provenance(edge: dict, hlo_text=None, audit_text=None) -> str:
    """Recompute and compare the source_hlo / allocation_audit hashes bound in the edge record.

    Returns ``"FRESH"`` when every supplied text matches, otherwise ``"STALE_HLO"`` /
    ``"STALE_AUDIT"`` for the first mismatch. Hashes not present in the record are skipped.
    """
    if hlo_text is not None:
        recorded = edge.get("source_hlo", {}).get("sha256")
        if recorded is not None:
            actual = hashlib.sha256(hlo_text.encode("utf-8")).hexdigest()
            if actual != recorded:
                return "STALE_HLO"
    if audit_text is not None:
        recorded = edge.get("allocation_audit", {}).get("sha256")
        if recorded is not None:
            actual = hashlib.sha256(audit_text.encode("utf-8")).hexdigest()
            if actual != recorded:
                return "STALE_AUDIT"
    return "FRESH"


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
    "trace_status",
    "note",
]


def map_anchor_for_case(n: int, depth: int, fusion: str = "default") -> dict:
    """Read Task 1's audit JSON + the HLO, build the v2 (piercing) edge map, write CSV + JSON.

    The C2 node identity is the HLO terminal consumer op SSA name (NOT a cotengra node id).
    The canonical gate reads the JSON; the CSV is a summary view only.
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
            "trace_status": "UNKNOWN",
            "note": "no anchor in audit",
        }
        upsert_csv_row(
            EDGE_CSV_PATH, row, _EDGE_CSV_COLUMNS, key_cols=["n", "depth", "fusion"]
        )
        rec = {
            "schema_version": "c1-c2-edge-v2",
            "case_id": f"{case_id}_{fusion}",
            "n": n,
            "depth": depth,
            "fusion": fusion,
            "producer": {},
            "transform": {},
            "consumer": {},
            "consumer_count": 0,
            "trace_status": "UNKNOWN",
            "source_hlo": {"path": hlo_path, "sha256": None},
            "allocation_audit": {"path": audit_path, "sha256": None},
        }
        with open(EDGE_JSON_PATH, "w") as fh:
            json.dump(rec, fh, indent=2)
        return rec

    rec = build_c1_edge_map(hlo_text, anchors[0]["hlo_value_id"])
    with open(audit_path) as fh:
        audit_text = fh.read()
    rec.update(
        {
            "case_id": f"{case_id}_{fusion}",
            "n": n,
            "depth": depth,
            "fusion": fusion,
        }
    )
    rec["source_hlo"]["path"] = hlo_path
    rec["allocation_audit"] = {
        "path": audit_path,
        "sha256": hashlib.sha256(audit_text.encode("utf-8")).hexdigest(),
    }

    row = {
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "producer_hlo_value_id": rec["producer"]["hlo_value_id"],
        "producer_M": rec["producer"]["M"],
        "producer_N": rec["producer"]["N"],
        "producer_K": rec["producer"]["K"],
        "producer_output_bytes": rec["producer"]["bytes"],
        "passthrough_hlo_ids": ";".join(rec["transform"]["hlo_ids"]),
        "terminal_consumer_hlo_value_id": rec["consumer"]["hlo_value_id"],
        "consumer_count": rec["consumer_count"],
        "consumer_M": rec["consumer"]["M"],
        "consumer_N": rec["consumer"]["N"],
        "consumer_K": rec["consumer"]["K"],
        "consumer_output_bytes": rec["consumer"]["bytes"],
        "trace_status": rec["trace_status"],
        "note": "",
    }
    upsert_csv_row(
        EDGE_CSV_PATH, row, _EDGE_CSV_COLUMNS, key_cols=["n", "depth", "fusion"]
    )
    with open(EDGE_JSON_PATH, "w") as fh:
        json.dump(rec, fh, indent=2)
    return rec
