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

import glob
import hashlib
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


# --- XLA buffer-assignment dump parser (correction Task A2/A4) ---------------------------
# Format (module_*.jit_f.*-buffer-assignment.txt):
#   allocation N: size S, <kind>:
#    value: <id name{shapeidx} @pos> (size=X,offset=Y): type[dims]{layout}
#   ... and a liveness section at the end:  name{shapeidx}:birth-death
_BA_ALLOC_RE = re.compile(r"^allocation (\d+):\s*size (\d+),\s*([^:]*):")
_BA_VAL_RE = re.compile(
    r"value:\s*<(\d+)\s+(.+?)\s+@(\d+)>\s*\(size=(\d+),offset=(\d+)\)"
)
_BA_LIVE_RE = re.compile(r"^([^\s:]+?\{[^}]*\}):(\d+)-(\d+)\s*$")
_BA_NAMEIDX_RE = re.compile(r"^(.+)\{(\d*)\}$")


def parse_buffer_assignment(ba_text: str):
    """Parse an XLA buffer-assignment dump into allocation/liveness/alias records.

    Returns ``(records, by_key, liveness, by_physical)``:
    - records: one dict per value ``{op_name, shape_index, buffer_id, value_size, offset,
      allocation_id, allocation_size, allocation_kind}``.
    - by_key: ``{(op_name, shape_index_str) -> record}``.
    - liveness: ``{(op_name, shape_index_str) -> (birth, death)}`` (instruction-seq indices).
    - by_physical: ``{(allocation_id, offset) -> [records]}`` -- same physical bytes = true
      buffer reuse (aliasing), e.g. the 512 MiB P (.497) and E (.498) at offset 536956416.
    """
    records: list[dict] = []
    cur = None  # (alloc_id, alloc_size, kind)
    for line in ba_text.splitlines():
        s = line.strip()
        ma = _BA_ALLOC_RE.match(s)
        if ma:
            cur = (int(ma.group(1)), int(ma.group(2)), ma.group(3).strip())
            continue
        mv = _BA_VAL_RE.search(s)
        if mv and cur is not None:
            nameidx = mv.group(2)
            mn = _BA_NAMEIDX_RE.match(nameidx)
            if mn:
                op_name, sidx = mn.group(1), mn.group(2)
            else:
                op_name, sidx = nameidx, ""
            records.append(
                {
                    "op_name": op_name,
                    "shape_index": sidx,
                    "buffer_id": int(mv.group(1)),
                    "value_size": int(mv.group(4)),
                    "offset": int(mv.group(5)),
                    "allocation_id": cur[0],
                    "allocation_size": cur[1],
                    "allocation_kind": cur[2],
                }
            )
    liveness: dict = {}
    for line in ba_text.splitlines():
        ml = _BA_LIVE_RE.match(line.strip())
        if ml:
            mn = _BA_NAMEIDX_RE.match(ml.group(1))
            key = (mn.group(1), mn.group(2)) if mn else (ml.group(1), "")
            liveness[key] = (int(ml.group(2)), int(ml.group(3)))
    by_key = {(r["op_name"], r["shape_index"]): r for r in records}
    by_physical: dict = {}
    for r in records:
        by_physical.setdefault((r["allocation_id"], r["offset"]), []).append(r)
    return records, by_key, liveness, by_physical


def _find_buffer_assignment(n: int, depth: int, fusion: str):
    """Locate the main (jit_f) buffer-assignment dump for a case, or None."""
    pattern = os.path.join(
        OUT_DIR,
        "c1_xla_dump",
        f"n{n}_d{depth}_{fusion}",
        "*jit_f*buffer-assignment.txt",
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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

    # Enrich with REAL XLA allocation/liveness/aliasing from the buffer-assignment dump
    # (Task A2/A4). Falls back to unknown/None if the dump is absent.
    ba_path = _find_buffer_assignment(n, depth, fusion)
    if ba_path:
        with open(ba_path) as fh:
            _, by_key, liveness, by_physical = parse_buffer_assignment(fh.read())
        for b in buffers:
            op_name = b["hlo_value_id"].lstrip("%")
            sidx = str(b["data_result_index"])
            rec = by_key.get((op_name, sidx))
            if not rec:
                continue
            b["allocation_id"] = rec["allocation_id"]
            b["allocation_size"] = rec["allocation_size"]
            b["allocation_kind"] = rec["allocation_kind"]
            b["offset"] = rec["offset"]
            b["value_size"] = rec["value_size"]
            # true aliasing = same physical bytes (allocation_id, offset): the sequential
            # GEMM outputs (.489/.490/.491/.497/.498) reuse one 512 MiB slot temporally.
            mates = by_physical.get((rec["allocation_id"], rec["offset"]), [])
            b["aliases"] = sorted(
                f"{m['op_name']}{{{m['shape_index']}}}"
                for m in mates
                if not (m["op_name"] == op_name and m["shape_index"] == sidx)
            )
            bd = liveness.get((op_name, sidx))
            if bd:
                b["birth"] = bd[0]
                b["death"] = bd[1]
    # allocation_source is xla_buffer_assignment ONLY if the anchor was actually enriched
    # (final-remediation Task 1: a present dump that fails to match the anchor is "unknown",
    # never a silent PASS).
    anchor = next((b for b in buffers if b.get("is_anchor")), None)
    _required = [
        "allocation_id",
        "allocation_size",
        "offset",
        "aliases",
        "birth",
        "death",
    ]
    missing_fields = (
        [f for f in _required if anchor is None or anchor.get(f) in (None, [])]
        if anchor is not None
        else _required
    )
    anchor_enriched = anchor is not None and not missing_fields
    allocation_source = "xla_buffer_assignment" if anchor_enriched else "unknown"
    live_range_source = "xla_buffer_assignment" if anchor_enriched else "unknown"
    audit_status = "COMPLETE" if anchor_enriched else "UNKNOWN"

    out = {
        "schema_version": "c1-buffer-audit-v2",
        "case_id": f"n{n}_d{depth}_{fusion}",
        "n": n,
        "depth": depth,
        "fusion": fusion,
        "hlo_path": hlo_path,
        "source_hlo_sha256": _sha256_file(hlo_path),
        "buffer_assignment_path": ba_path,
        "buffer_assignment_sha256": _sha256_file(ba_path) if ba_path else None,
        "allocation_source": allocation_source,
        "live_range_source": live_range_source,
        "audit_status": audit_status,
        "missing_fields": missing_fields,
        "buffer_count": len(buffers),
        "anchor_count": anchor_count,
        "buffers": buffers,
    }
    os.makedirs(AUDIT_DIR, exist_ok=True)
    json_path = f"{AUDIT_DIR}/n{n}_d{depth}_{fusion}.json"
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out
