"""C1 buffer-assignment audit (rereview §4.2/4.3, correction-plan Task A).

Parses the PRODUCTION expectation executable's optimized HLO for every ``__cublas$gemm``
custom-call, splitting the output tuple into DATA output and cuBLAS WORKSPACE by result
index + dtype (NOT by max bytes -- on small GEMMs the s8 workspace can exceed the data
output, e.g. ``(c64[10,2]=160B, s8[192]=192B)``), assigning a stable ``hlo_value_id`` (SSA
name), and flagging the 512 MiB ``c64[4096,16384]`` anchor.

allocation_id / offset / aliases / birth / death come ONLY from a real XLA buffer-assignment
dump (the ``xla_dump`` worker, Task A2). Until that dump yields parseable data they are
recorded as ``unknown``/``None`` -- NEVER fabricated from the HLO SSA name (rereview §4.1:
an HLO value id is not an XLA allocation id).

Pure text parsing over the HLO artifact saved by ``c1.measure_case`` (no GPU/compile).
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
    """All ``__cublas$gemm`` custom-call result buffers, with DATA output and cuBLAS
    WORKSPACE separated by result index + dtype.

    The data output is the non-``s8`` typed element; the workspace is the ``s8`` element.
    Selection is by dtype/index, NOT by max bytes (on small GEMMs the s8 workspace can be
    larger than the data output). Returns one dict per custom-call:
    ``{hlo_value_id, data_result_index, data_dtype, data_shape, data_output_bytes,
    workspace_result_index, workspace_bytes}``.
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
        data = None  # (dtype, dims, bytes)
        ws = None
        data_idx = None
        ws_idx = None
        for idx, elem in enumerate(_TYPED_ELEM_RE.finditer(tuple_body)):
            dtype, dims_csv = elem.group(1), elem.group(2)
            dims = [int(x) for x in dims_csv.split(",")]
            b = _elem_bytes(dtype, dims_csv)
            if dtype == "s8":
                ws = (dtype, dims, b)
                ws_idx = idx
            else:
                data = (dtype, dims, b)
                data_idx = idx
        if data is None:
            continue
        buffers.append(
            {
                "hlo_value_id": ssa,
                "data_result_index": data_idx,
                "data_dtype": data[0],
                "data_shape": data[1],
                "data_output_bytes": data[2],
                "workspace_result_index": ws_idx,
                "workspace_bytes": ws[2] if ws else 0,
            }
        )
    return buffers


def audit_buffer_assignment(n: int, depth: int, fusion: str = "default") -> dict:
    """Build the buffer-assignment audit for one C1 case and write it as JSON.

    Reads ``results/phase0/c1_optimized_hlo/n{n}_d{depth}_exp_{fusion}.hlo`` (written by
    ``c1.measure_case``), parses every ``__cublas$gemm`` tuple (data vs workspace), flags
    the 512 MiB ``c64[4096,16384]`` anchor, and writes
    ``results/phase0/c1_buffer_assignment/n{n}_d{depth}_{fusion}.json``. allocation/liveness
    fields are ``unknown``/``None`` until the XLA dump worker (Task A2) enriches them.
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
        is_anchor = (
            tuple(b["data_shape"]) == ANCHOR_SHAPE and b["data_dtype"] == ANCHOR_DTYPE
        )
        if is_anchor:
            anchor_count += 1
        buffers.append(
            {
                "hlo_value_id": b["hlo_value_id"],
                "data_result_index": b["data_result_index"],
                "data_dtype": b["data_dtype"],
                "data_shape": b["data_shape"],
                "data_output_bytes": b["data_output_bytes"],
                "workspace_result_index": b["workspace_result_index"],
                "workspace_bytes": b["workspace_bytes"],
                "is_anchor": is_anchor,
                # Real allocation/liveness needs the XLA --xla_dump_to worker (Task A2).
                # Until then honestly unknown -- never fake an allocation_id from the SSA name.
                "allocation_id": None,
                "allocation_size": None,
                "offset": None,
                "aliases": None,
                "birth": None,
                "death": None,
            }
        )

    out = {
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "hlo_path": hlo_path,
        "allocation_source": "unknown",
        "live_range_source": "unknown",
        "buffer_count": len(buffers),
        "anchor_count": anchor_count,
        "buffers": buffers,
    }
    os.makedirs(AUDIT_DIR, exist_ok=True)
    json_path = f"{AUDIT_DIR}/n{n}_d{depth}_{fusion}.json"
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out
