"""C1 buffer-assignment audit (rereview §4.2/§4.3).

Parses the PRODUCTION expectation executable's optimized HLO for every materialized
contraction buffer — the ``__cublas$gemm`` custom-call output tuples — and assigns a
stable ``hlo_value_id`` (the SSA name, e.g. ``%custom-call.497``), shape, byte size and
an ``is_anchor`` flag for the 512 MiB ``c64[4096,16384]`` buffer. This is the C1
buffer audit + the anchor identity that Task 2 (HLO use-def edge map) consumes.

Pure text parsing over the HLO artifact saved by ``c1.measure_case`` (no GPU/compile).
``allocation_id`` / live-range are best-effort: on jax 0.6.2 GPU
``compiled.memory_analysis().serialized_buffer_assignment_proto`` is empty (len 0), so
``allocation_source``/``live_range_source`` default to ``"hlo_shape_only"`` and Step 3b's
``--xla_dump_to`` enrichment (``dump_buffer_assignment_via_xla``) upgrades them only when
the dump actually yields parseable allocation/liveness data.

Validated against the real n=24/d=10/default HLO: the anchor is
``%custom-call.497 = (c64[4096,16384]{1,0}, s8[33554432]{0}) custom-call(... __cublas$gemm)``
with operands ``c64[4096,1024] x c64[1024,16384]`` (M=4096,K=1024,N=16384), consumed by
``%get-tuple-element.246.0``.
"""

from __future__ import annotations

import json
import os
import re

OUT_DIR = "results/phase0"
HLO_DIR = f"{OUT_DIR}/c1_optimized_hlo"
AUDIT_DIR = f"{OUT_DIR}/c1_buffer_assignment"

_HLO_DTYPE_BYTES = {
    "f32": 4,
    "f64": 8,
    "bf16": 2,
    "f16": 2,
    "c64": 8,
    "c128": 16,
    "s8": 1,
    "s16": 2,
    "s32": 4,
    "s64": 8,
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "pred": 1,
}

# Match `%ssa-name = (tuple-body) custom-call` on lines carrying __cublas$gemm.
# SSA names contain `-`/`.` (e.g. %custom-call.497, %get-tuple-element.5).
_CUBLAS_CALL_DEF_RE = re.compile(r"(%[a-zA-Z0-9_.\-]+)\s*=\s*\(([^)]*)\)\s+custom-call")
# One typed tuple element: TYPE[dims]{layout}
_TYPED_ELEM_RE = re.compile(r"\b([a-z0-9_]+)\[([0-9]+(?:,[0-9]+)*)\]\{[^}]*\}")

ANCHOR_SHAPE = (4096, 16384)
ANCHOR_DTYPE = "c64"


def _elem_bytes(dtype: str, dims_csv: str) -> int:
    """Element-count x bytes-per-element for an HLO shape; 0 for unknown dtypes."""
    bpb = _HLO_DTYPE_BYTES.get(dtype)
    if not bpb:
        return 0
    n = 1
    for d in dims_csv.split(","):
        n *= int(d)
    return n * bpb


def parse_materialized_buffers(hlo_text: str) -> list[dict]:
    """All ``__cublas$gemm`` custom-call RESULT buffers in the HLO.

    The result buffer is the largest typed element of the output tuple (excludes the
    ``s8`` cuBLAS scratch). Returns one dict per custom-call:
    ``{hlo_value_id, dtype, shape, buffer_bytes}``.
    """
    buffers: list[dict] = []
    for line in hlo_text.splitlines():
        if "__cublas$gemm" not in line:
            continue
        m = _CUBLAS_CALL_DEF_RE.search(line)
        if not m:
            continue
        ssa = m.group(1)
        tuple_body = m.group(2)
        best = None  # (dtype, shape_list, bytes)
        for elem in _TYPED_ELEM_RE.finditer(tuple_body):
            dtype, dims = elem.group(1), elem.group(2)
            b = _elem_bytes(dtype, dims)
            if best is None or b > best[2]:
                best = (dtype, [int(x) for x in dims.split(",")], b)
        if best is None:
            continue
        buffers.append(
            {
                "hlo_value_id": ssa,
                "dtype": best[0],
                "shape": best[1],
                "buffer_bytes": best[2],
            }
        )
    return buffers


def audit_buffer_assignment(n: int, depth: int, fusion: str = "default") -> dict:
    """Build the buffer-assignment audit for one C1 case and write it as JSON.

    Reads ``results/phase0/c1_optimized_hlo/n{n}_d{depth}_exp_{fusion}.hlo`` (written by
    ``c1.measure_case``), parses every ``__cublas$gemm`` result buffer, flags the
    512 MiB ``c64[4096,16384]`` anchor, and writes
    ``results/phase0/c1_buffer_assignment/n{n}_d{depth}_{fusion}.json``.
    """
    hlo_path = f"{HLO_DIR}/n{n}_d{depth}_exp_{fusion}.hlo"
    if not os.path.exists(hlo_path):
        raise FileNotFoundError(
            f"HLO artifact not found: {hlo_path}; run c1.measure_case first"
        )
    with open(hlo_path) as fh:
        hlo_text = fh.read()

    raw = parse_materialized_buffers(hlo_text)
    buffers = []
    anchor_count = 0
    for b in raw:
        is_anchor = tuple(b["shape"]) == ANCHOR_SHAPE and b["dtype"] == ANCHOR_DTYPE
        if is_anchor:
            anchor_count += 1
        buffers.append(
            {
                "hlo_value_id": b["hlo_value_id"],
                "dtype": b["dtype"],
                "shape": b["shape"],
                "buffer_bytes": b["buffer_bytes"],
                "is_anchor": is_anchor,
                # Real allocation_id needs XLA --xla_dump_to (Step 3b); the in-process
                # memory_analysis exposes none on GPU (proto len 0).
                "allocation_id": b["hlo_value_id"],
            }
        )

    out = {
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "hlo_path": hlo_path,
        "allocation_source": "hlo_shape_only",
        "live_range_source": "hlo_shape_only",
        "buffer_count": len(buffers),
        "anchor_count": anchor_count,
        "buffers": buffers,
    }
    os.makedirs(AUDIT_DIR, exist_ok=True)
    json_path = f"{AUDIT_DIR}/n{n}_d{depth}_{fusion}.json"
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out
