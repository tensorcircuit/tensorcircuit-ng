"""Real contraction shape export from the cotengra tree (review §6.1, Plan A Task 6).

Why this module exists
----------------------
Plan A of the BF16 Phase 0 remediation needs to classify which contraction steps are
tile-mappable (eligible for bf16 GEMM engagement) on the **actual** tc-ng contraction
shapes -- not square microbenchmarks. Review §6.1 requires per-step
``M,N,K,batch,transpose,strides,bytes,consumer_count,live_range`` so Task 7 can apply the
``min(M,K,N) >= ~256`` tile-mappability test to real circuits.

Two entry points
----------------
- ``export_shapes_from_eq(eq, size_dict, dtype_bytes=8)`` -- PURE: walks a cotengra tree
  built from an arbitrary einsum, returns one dict per contraction step. Unit-tested.
- ``export_shapes(n, depth, output='expectation')`` -- INTEGRATION: monkey-patches
  ``cons._extract_topology`` (same pattern as ``results/_coarsen_spike.py``) to capture
  the real ``(input_sets, output_set, size_dict)`` of a parameterized tc-ng circuit, then
  delegates to ``export_shapes_from_eq``. Writes ``results/phase0/contraction_shapes.csv``.

Cotengra 0.8.2 tree API (empirically verified)
----------------------------------------------
The brief's ``tree.get_nodes() / get_lhs() / get_rhs()`` DO NOT EXIST in cotengra 0.8.2
(probe output: ``dir(tree)`` lists ``traverse``, ``get_inds``, ``get_size``, ``get_path``
but NOT ``get_nodes``/``get_lhs``/``get_rhs``). We instead use the canonical 0.8.2 walk:

    for parent, left, right in tree.traverse():   # bottom-up, children before parent
        ...

``tree.traverse()`` yields ``(parent, left_child, right_child)`` tuples of INTEGER node
handles (verified: ``type(parent) == int``; ``tree.info`` is keyed by the same ints).
``tree.get_inds(node)`` returns the ordered index string (e.g. ``"ac"``);
``tree.get_size(node)`` returns the tensor element count. This exposes producer/consumer
structure directly, which ``get_path()`` + simulation does not.
This is documented as a deviation from the brief; the test's M/N/K/consumer_count
assertions are preserved.

Layout convention
-----------------
The cotengra tree gives logical modes and extents, NOT physical tensor layouts. We assume
contiguous row-major (the cuBLAS/cublasLt default for the input tensors tc-ng produces):
``strides[-1] = 1``, ``strides[i] = prod(extents[i+1:])``. ``transpose`` is set True when
the contracted ``K`` indices are NOT the trailing dims of an operand (i.e. cuBLAS would
need a transposed access). ``bytes`` = parent output size * ``dtype_bytes`` (tc-ng default
complex64 -> 8 bytes).

``consumer_count`` for a step's output = number of LATER steps that consume it, plus 1 if
it is the root (the external output sink). This guarantees ``consumer_count >= 1`` and is
meaningful: every produced tensor is consumed at least once. ``live_range`` is
``[birth_step, death_step]`` where inputs are born at step -1 and intermediates at the
step that produced them; the root's death is the last step + 1.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Iterable

import cotengra as ctg

OUT_DIR = "results/phase0"
SHAPES_CSV_PATH = f"{OUT_DIR}/contraction_shapes.csv"

# tc-ng default dtype is complex64 -> 8 bytes per element.
DEFAULT_DTYPE_BYTES = 8


def _row_major_strides(extents: Iterable[int]) -> list[int]:
    """Row-major strides (last dim contiguous) for a shape, in elements (not bytes)."""
    ext = list(extents)
    strides = [1] * len(ext)
    for i in range(len(ext) - 2, -1, -1):
        strides[i] = strides[i + 1] * ext[i + 1]
    return strides


def _is_transposed(operand_inds: str, contracted_inds: set[str]) -> bool:
    """An operand is 'transposed' (needs cuBLAS transpose) when its contracted K indices
    are not contiguous at the trailing position. With a single K index this reduces to:
    ``operand_inds[-1] not in contracted_inds``."""
    if not contracted_inds or not operand_inds:
        return False
    # last index should be a contracted (K) index in the natural GEMM layout
    return operand_inds[-1] not in contracted_inds


def _walk_tree(tree, size_dict, dtype_bytes):
    """Walk a cotengra 0.8.2 ``ContractionTree`` via ``tree.traverse()`` (bottom-up) and
    return one dict per contraction step. See module docstring for the API rationale.

    Cotengra 0.8.2 uses INTEGER node IDs in ``traverse()`` (verified: ``type(parent) is
    int``), not the frozensets the older docstring suggests. We key auxiliary dicts
    (``produced_at``, ``child_use_count``) on the raw int node handle. Leaf node IDs come
    from ``tree.get_leaves_ordered()`` (NOT ``tree.inputs``, which holds the tuples of
    index letters)."""
    # Pre-pass: count how many times each node appears as a child across all merges, and
    # record the step index at which each node is PRODUCED (parent of a merge). Leaves
    # (the original inputs, whose node IDs are integers from ``get_leaves_ordered``) are
    # produced before the walk starts (step -1).
    merges = list(tree.traverse())
    produced_at: dict = {}
    child_use_count: dict = {}
    for leaf in tree.get_leaves_ordered():
        produced_at[leaf] = -1
    for step_idx, merge in enumerate(merges):
        parent = merge[0]
        produced_at[parent] = step_idx
        for child in merge[1:]:
            child_use_count[child] = child_use_count.get(child, 0) + 1

    root = merges[-1][0] if merges else None

    shapes = []
    for step_idx, merge in enumerate(merges):
        parent = merge[0]
        children = list(merge[1:])
        # A binary contraction has exactly 2 children; cotengra can yield unary/hyper merges
        # for slicing or single-input contractions. Normalize: if only one child, the
        # "contraction" is a trace/reduce with no second operand -> K = 1, second operand empty.
        left = children[0] if len(children) >= 1 else None
        right = children[1] if len(children) >= 2 else None

        parent_inds = set(tree.get_inds(parent))
        left_inds = set(tree.get_inds(left)) if left is not None else set()
        right_inds = set(tree.get_inds(right)) if right is not None else set()

        # Contracted indices: appear in BOTH operands, absent from the parent (summed).
        if right is not None:
            contracted = (left_inds & right_inds) - parent_inds
        else:
            contracted = left_inds - parent_inds
        # Batch indices: appear in BOTH operands AND the parent (preserved, not summed).
        batch_inds = (
            (left_inds & right_inds) & parent_inds if right is not None else set()
        )
        # Output-only indices (M and N extents).
        left_only = (
            left_inds - right_inds if right is not None else parent_inds - left_inds
        )
        right_only = right_inds - left_inds if right is not None else set()

        K = math.prod(size_dict[c] for c in contracted) if contracted else 1
        M = math.prod(size_dict[x] for x in left_only) if left_only else 1
        N = math.prod(size_dict[x] for x in right_only) if right_only else 1
        batch = math.prod(size_dict[x] for x in batch_inds) if batch_inds else 1

        parent_inds_str = tree.get_inds(parent)
        # The parent's extents follow the order of its index string (cotengra convention).
        extents = [int(size_dict[i]) for i in parent_inds_str]
        strides = _row_major_strides(extents)
        parent_size = tree.get_size(parent)  # element count
        bytes_out = int(parent_size) * dtype_bytes

        transpose = _is_transposed(
            tree.get_inds(left) if left is not None else "",
            {str(c) for c in contracted},
        )

        # consumer_count: # later steps that consume this step's output, plus 1 for the
        # external sink if it is the root. Guarantees >= 1.
        downstream = child_use_count.get(parent, 0)
        is_root = parent == root
        consumer_count = downstream + (1 if is_root else 0)
        # consumer_ids: list of step indices that consume this node (resolved where
        # possible). The root's sink is recorded as step ``len(merges)``.
        consumer_ids = []
        for later_idx, later_merge in enumerate(
            merges[step_idx + 1 :], start=step_idx + 1
        ):
            if parent in later_merge[1:]:
                consumer_ids.append(later_idx)
        if is_root:
            consumer_ids.append(len(merges))

        # producer_ids: steps that produced this step's INPUTS (children). Leaves have
        # producer_id = -1 (original input).
        producer_ids = [produced_at[c] for c in children]

        birth = step_idx
        death = consumer_ids[0] if consumer_ids else len(merges)
        live_range = [birth, death]

        shapes.append(
            {
                "node_id": step_idx,
                "producer_ids": producer_ids,
                "consumer_ids": consumer_ids,
                "modes": parent_inds_str,
                "extents": extents,
                "M": int(M),
                "N": int(N),
                "K": int(K),
                "batch": int(batch),
                "transpose": bool(transpose),
                "strides": strides,
                "bytes": bytes_out,
                "consumer_count": int(consumer_count),
                "live_range": live_range,
            }
        )
    return shapes


def export_shapes_from_eq(eq, size_dict, dtype_bytes: int = DEFAULT_DTYPE_BYTES):
    """PURE: build a cotengra contraction tree for ``eq`` and return one shape dict per
    contraction step. Unit-tested.

    Parameters
    ----------
    eq : str
        Einsum-like string, either ``"ab,bc->ac"`` (with output) or ``"ab,bc"`` (output
        is the union of all inputs -- ``opt_einsum`` "trace-all" convention).
    size_dict : dict[str, int]
        Map from each index letter to its dimension extent.
    dtype_bytes : int
        Bytes per element for the ``bytes`` field (tc-ng default complex64 = 8).

    Returns
    -------
    list[dict]
        One dict per contraction step. See module docstring for the schema.
    """
    if "->" in eq:
        lhs, rhs = eq.split("->")
        output = tuple(rhs)
    else:
        lhs = eq
        output = None
    inputs = [tuple(s) for s in lhs.split(",")]
    if output is None:
        # Default output = union of inputs, preserving first-appearance order.
        seen = []
        for term in inputs:
            for ind in term:
                if ind not in seen:
                    seen.append(ind)
        output = tuple(seen)

    opt = ctg.HyperOptimizer(minimize="size", max_repeats=8, max_time=10)
    tree = opt.search(inputs, output, size_dict)
    return _walk_tree(tree, size_dict, dtype_bytes)


def _capture_tcng_topology(n, depth, output="expectation"):
    """Monkey-patch ``cons._extract_topology`` to capture the real
    ``(input_sets, output_set, size_dict)`` of a parameterized tc-ng circuit, mirroring
    ``results/_coarsen_spike.py.capture_topology``.

    DEVITATION FROM THE BRIEF (documented): the brief and ``_coarsen_spike.py`` use
    ``with bcomplex32():`` (from ``applications/bcomplex32_algebra.py``) to force the
    cotengra ``_algebraic_base_contraction`` path so ``_extract_topology`` fires. We use
    the PUBLIC ``cons.runtime_contraction_algebra(StandardAlgebra())`` context manager
    instead. ``StandardAlgebra`` is documented in ``tensorcircuit/contraction_algebra.py``
    as "identical to native backend behaviour" -- it triggers the same code path WITHOUT
    changing dtype (so the captured topology matches the real default-contraction
    topology) AND without importing the reference application in ``applications/``
    (which ``AGENTS.md`` marks deprecated). The captured topology (input_sets,
    output_set, size_dict) is a function of the tensor-network graph structure, not the
    algebra, so this substitution is exact.
    """
    import tensorcircuit as tc
    import tensorcircuit.cons as cons
    from tensorcircuit.contraction_algebra import StandardAlgebra

    captured = {}
    orig = cons._extract_topology

    def wrapped(nodes):
        topo = orig(nodes)
        captured["topo"] = topo
        return topo

    cons._extract_topology = wrapped
    try:
        from results._phase0.circuits import build_parameterized_circuit

        c = build_parameterized_circuit([0.7] * (depth * n), n, depth)
        with cons.runtime_contraction_algebra(StandardAlgebra()):
            if output == "state":
                _ = c.state()
            else:
                _ = c.expectation((tc.gates.z(), [0]))
    finally:
        cons._extract_topology = orig

    if "topo" not in captured:
        raise RuntimeError(
            "_extract_topology was not invoked; no contraction topology captured"
        )
    raw, input_sets, output_set, size_dict = captured["topo"]
    return input_sets, output_set, size_dict


def export_shapes(
    n, depth, output="expectation", dtype_bytes: int = DEFAULT_DTYPE_BYTES
):
    """INTEGRATION: capture the real tc-ng contraction topology for the parameterized
    C1 circuit (n qubits, ``depth`` brickwork layers) and walk its cotengra tree.

    Returns
    -------
    list[dict]
        One dict per contraction step (same schema as ``export_shapes_from_eq``).
    """
    input_sets, output_set, size_dict = _capture_tcng_topology(n, depth, output=output)
    opt = ctg.HyperOptimizer(minimize="size", max_repeats=8, max_time=10)
    tree = opt.search(input_sets, output_set, size_dict)
    return _walk_tree(tree, size_dict, dtype_bytes)


CSV_COLUMNS = [
    "n",
    "depth",
    "output",
    "node_id",
    "modes",
    "extents",
    "M",
    "N",
    "K",
    "batch",
    "transpose",
    "strides",
    "bytes",
    "consumer_count",
    "producer_ids",
    "consumer_ids",
    "live_range",
]


def write_shapes_csv(rows, path=SHAPES_CSV_PATH):
    """Write ``rows`` (list of dict) to ``path`` as CSV with the schema from review §6.1.
    List-valued fields are serialized as ``";"``-joined so the CSV stays rectangular."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow(
                [
                    r.get("n", ""),
                    r.get("depth", ""),
                    r.get("output", ""),
                    r.get("node_id", ""),
                    r.get("modes", ""),
                    ";".join(str(x) for x in r.get("extents", [])),
                    r.get("M", ""),
                    r.get("N", ""),
                    r.get("K", ""),
                    r.get("batch", ""),
                    int(bool(r.get("transpose", False))),
                    ";".join(str(x) for x in r.get("strides", [])),
                    r.get("bytes", ""),
                    r.get("consumer_count", ""),
                    ";".join(str(x) for x in r.get("producer_ids", [])),
                    ";".join(str(x) for x in r.get("consumer_ids", [])),
                    ";".join(str(x) for x in r.get("live_range", [])),
                ]
            )


def main():
    """CLI: export shapes for the C1 cases (n=22, 24, depth=10) and append to the CSV.

    The CSV at ``results/phase0/contraction_shapes.csv`` is REWRITTEN on each run to keep
    the artifact deterministic (one row per contraction step per case per output mode).

    Both ``expectation`` and ``state`` output modes are exported per case. The C1
    measurement itself uses ``expectation`` (matches the brief's default), but
    ``c.expectation(...)`` returns a tiny 3-tensor topology because tc-ng eagerly
    contracts each gate into the statevector during ``Circuit`` construction. The
    meaningful per-step GEMM tree -- what Task 7 needs to classify tile-mappability --
    is the ``state`` contraction (474 tensors for n=22, depth=10), so both are written
    and disambiguated by the ``output`` CSV column. This is documented in the report."""
    ap_csv = SHAPES_CSV_PATH
    if os.path.exists(ap_csv):
        os.remove(ap_csv)
    cases = [(22, 10), (24, 10)]
    output_modes = ("expectation", "state")
    summary = []
    for n, depth in cases:
        for out_mode in output_modes:
            shapes = export_shapes(n, depth, output=out_mode)
            for s in shapes:
                s["n"] = n
                s["depth"] = depth
                s["output"] = out_mode
            write_shapes_csv(shapes, ap_csv)
            mnk_max = max((s["M"] * s["N"] * s["K"] for s in shapes), default=0)
            cc_dist = {}
            for s in shapes:
                cc_dist[s["consumer_count"]] = cc_dist.get(s["consumer_count"], 0) + 1
            summary.append(
                {
                    "n": n,
                    "depth": depth,
                    "output": out_mode,
                    "steps": len(shapes),
                    "max_MNK": mnk_max,
                    "consumer_count_dist": cc_dist,
                }
            )
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
