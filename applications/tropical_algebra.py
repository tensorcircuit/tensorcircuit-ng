"""Tropical (max-plus) contraction algebra -- a reference APPLICATION of the generic
``ContractionAlgebra`` ABCs shipped in ``tensorcircuit.contraction_algebra``.

The feature is ``tensorcircuit.contraction_algebra`` -- the ``ContractionAlgebra`` and
``Representation`` ABCs -- wired in-source into ``cons._base``, which routes any
non-standard algebra through ``cons._algebraic_base_contraction`` (no monkey-patching).
A custom algebra is activated either via ``tc.set_contractor(algebra=...)`` or by the
``tropical()`` / ``counting_tropical()`` context managers below. This file is one
complete algebra built on top of the ABCs -- an importable reference module, not a
runnable demo (for that, see ``examples/tropical_ising.py``). Kept under
``applications/`` as a reference application; promote it to its own package/location if
it needs active development, independent distribution, or outgrows ``applications/``
conventions.

Implements the three standard tropical-tensor-network outputs (energy /
configuration / degeneracy) from Liu, Wang, Zhang PRL 126, 090506 (2021)
(arXiv:2008.06888). Contract under a tropical algebra via::

    from applications.tropical_algebra import tropical
    with tropical():
        ...  # max-plus contraction

Sections (consolidated from the original package modules):
  1. max-plus primitives + MaxPlusAlgebra              (was tensorcircuit/contraction_algebra/tropical.py)
  2. counting (energy, degeneracy) + split_energy_count (was .../counting.py)
  3. tracking + configuration recovery                  (was .../config.py)
  4. context managers: tropical(track=...) / counting_tropical()
"""

import contextlib
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import tensorcircuit.cons as cons
from tensorcircuit.contraction_algebra import ContractionAlgebra, Representation

Tensor = Any
Backend = Any
_EPS = 1e-9

logger = logging.getLogger(__name__)


# ===== Section 1: max-plus primitives + MaxPlusAlgebra =====


def _pair_layout(
    be: Backend, a: Tensor, b: Tensor, axes: Any
) -> "tuple[Tensor, Tensor, tuple[int, ...]]":
    """Transpose+reshape ``a``, ``b`` into the broadcast pair layout.

    Returns ``(a3, b3, out_shape)`` where ``a3`` is shape ``(m, k, 1)``, ``b3`` is
    ``(1, k, n)``, and ``out_shape = a_free_shape + b_free_shape``. Here ``m`` is the
    product of ``a``'s free axes, ``n`` of ``b``'s free axes, and ``k`` of the
    contracted axes. The native broadcast of ``a3 + b3`` yields the ``(m, k, n)``
    pair-sum tensor required by max-plus (and counting) pairwise contraction.
    """
    ashape = tuple(int(x) for x in be.shape_tuple(a))
    bshape = tuple(int(x) for x in be.shape_tuple(b))
    if isinstance(axes, int):
        a_axes = list(range(len(ashape) - axes, len(ashape)))
        b_axes = list(range(0, axes))
    else:
        a_axes, b_axes = list(axes[0]), list(axes[1])
    a_free = [i for i in range(len(ashape)) if i not in a_axes]
    b_free = [i for i in range(len(bshape)) if i not in b_axes]
    a_t = be.transpose(a, tuple(a_free + a_axes))  # contracted axes to the tail
    b_t = be.transpose(b, tuple(b_axes + b_free))  # contracted axes to the front
    a_fs = [ashape[i] for i in a_free]
    b_fs = [bshape[i] for i in b_free]
    m = int(np.prod(a_fs))
    k = int(np.prod([ashape[i] for i in a_axes]))
    n = int(np.prod(b_fs))
    a3 = be.reshape(a_t, (m, k, 1))
    b3 = be.reshape(b_t, (1, k, n))
    return a3, b3, (tuple(a_fs) + tuple(b_fs))


def _tropical_tensordot(be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
    """max-plus tensordot: Y[i,j] = max_k (A[i,k] + B[k,j])."""
    a3, b3, out_shape = _pair_layout(be, a, b, axes)
    s = a3 + b3  # native broadcast add -> (m, k, n)
    red = be.max(s, axis=1)  # max over contracted axis (tropical addition)
    return be.reshape(red, out_shape)


def _expand_to_layout(
    be: Backend, t: Tensor, idxs: Sequence[str], full: Sequence[str]
) -> Tensor:
    """Reshape/transpose ``t`` (axes == ``idxs``) into the ``full`` index layout,
    inserting size-1 axes for indices not in ``idxs`` (enables broadcasting)."""
    shape = tuple(int(x) for x in be.shape_tuple(t))
    present = {c: shape[i] for i, c in enumerate(idxs)}
    sub = [c for c in full if c in present]  # present indices in full order
    tt = be.transpose(t, tuple(idxs.index(c) for c in sub))
    newshape = tuple(present[c] if c in present else 1 for c in full)
    return be.reshape(tt, newshape)


def _tropical_einsum(be: Backend, eq: str, *operands: Tensor) -> Tensor:
    """max-plus einsum: product -> +, sum -> max. Handles 1- or 2-operand forms."""
    if len(operands) == 1:
        a = operands[0]
        in_str, _sep, out_str = eq.partition("->")
        lhs = in_str.split(",")[0]
        rhs = out_str
        if len(set(lhs)) != len(lhs):
            resolved = "".join(dict.fromkeys(lhs))
            a = be.einsum(lhs + "->" + resolved, a)
            lhs = resolved
        contract = [c for c in lhs if c not in rhs]
        for ax in sorted([lhs.index(c) for c in contract], reverse=True):
            a = be.max(a, axis=ax)
        remaining = [c for c in lhs if c not in contract]
        return be.transpose(a, tuple(remaining.index(c) for c in rhs))
    a, b = operands
    lhs, rhs = eq.split("->")
    ia_s, ib_s = lhs.split(",")
    ia, ib = list(ia_s), list(ib_s)
    if len(set(ia)) != len(ia):
        resolved = "".join(dict.fromkeys(ia))
        a = be.einsum("".join(ia) + "->" + resolved, a)
        ia = list(resolved)
    if len(set(ib)) != len(ib):
        resolved = "".join(dict.fromkeys(ib))
        b = be.einsum("".join(ib) + "->" + resolved, b)
        ib = list(resolved)
    all_idx = list(dict.fromkeys(ia + ib))
    out_idx = list(rhs)
    contract = [c for c in all_idx if c not in out_idx]
    s = _expand_to_layout(be, a, ia, all_idx) + _expand_to_layout(be, b, ib, all_idx)
    for ax in sorted([all_idx.index(c) for c in contract], reverse=True):
        s = be.max(s, axis=ax)
    remaining = [c for c in all_idx if c not in contract]
    return be.transpose(s, tuple(remaining.index(c) for c in out_idx))


class MaxPlusAlgebra(ContractionAlgebra):
    """Tropical (max, +) semiring: addition -> max, multiplication -> +."""

    name = "maxplus"
    # representation defaults to IdentityRepresentation (real tensors, no codec):
    # leaves enter/exit the contraction unchanged; only the kernels differ.

    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        return _tropical_tensordot(be, a, b, axes)

    def einsum(self, be: Backend, eq: str, *operands: Tensor) -> Tensor:
        return _tropical_einsum(be, eq, *operands)


# ===== Section 2: counting (energy, degeneracy) =====


def split_energy_count(stacked: Tensor) -> "tuple[Tensor, Tensor]":
    """Split a stacked ``[..., 2]`` tensor into ``(energy, count)`` numpy arrays.

    The last axis is interpreted as ``[..., 0] = energy, [..., 1] = count``.
    """
    arr = stacked if isinstance(stacked, np.ndarray) else np.asarray(stacked)
    return arr[..., 0], arr[..., 1]


def _stack_last(be: Backend, x: Tensor, n: Tensor) -> Tensor:
    """Stack two same-shape tensors along a new trailing axis (portable).

    All tc-ng concrete backends (numpy, jax, torch, tensorflow, cupy) implement
    ``be.stack`` via ``tensorcircuit.backends.abstract_backend``, so the portable
    path is used directly. ``be.stack`` takes a Python sequence and inserts a new
    axis; ``axis=-1`` puts it last to match the ``[..., 2]`` convention.
    """
    return be.stack([x, n], axis=-1)


def _counting_tensordot(be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
    """(max-plus energy, degeneracy count) pairwise contraction.

    ``a`` and ``b`` are stacked ``[..., 2]`` tensors (``[..., 0]`` = energy,
    ``[..., 1]`` = count). The energy stream follows max-plus; the count stream
    sums ``a_count * b_count`` over the contracted positions that achieve the
    energy max (within ``_EPS``), so each max-tie contributes its multiplicity.
    """
    a3x, b3x, out_shape = _pair_layout(be, a[..., 0], b[..., 0], axes)
    s = a3x + b3x  # (m, k, n) energy pair sums
    y = be.max(s, axis=1)  # (m, n) max energy per output slot
    # abs(s - y) < _EPS, broadcast across the contracted axis k: reshape y to (m, 1, n)
    m_n = tuple(int(d) for d in be.shape_tuple(y))
    y_b = be.reshape(y, (m_n[0], 1, m_n[1]))
    mask = be.abs(s - y_b) < _EPS
    a3n, b3n, _ = _pair_layout(be, a[..., 1], b[..., 1], axes)
    pn = a3n * b3n  # (m, k, n) pairwise count products
    cy = be.reshape(be.sum(pn * mask, axis=1), out_shape)
    y_shaped = be.reshape(y, out_shape)
    return _stack_last(be, y_shaped, cy)


def _insert_axis_of(
    be: Backend, reduced: Tensor, axis: int, full_shape: Sequence[int]
) -> Tensor:
    """Reshape an axis-reduced tensor so its ``axis`` is size 1, restoring the
    pre-reduce rank for broadcasting against the original tensor.

    ``be.max`` lacks ``keepdims``; this reshape re-inserts the size-1 axis so
    ``reduced`` broadcasts against a tensor of shape ``full_shape``.
    """
    new_shape = tuple(
        1 if i == axis else int(full_shape[i]) for i in range(len(full_shape))
    )
    return be.reshape(reduced, new_shape)


def _resolve_repeats(
    be: Backend, pair: Tensor, idxs: Sequence[str]
) -> "tuple[Tensor, Tensor, list[str]]":
    """Resolve intra-operand repeated indices via per-stream diagonal gather.

    Pure per-stream indexing -- the energy and count streams are gathered
    independently (``pair[..., 0]`` and ``pair[..., 1]``) so the trailing
    ``[..., 2]`` stack axis is never treated as an index by ``be.einsum``. The
    max/tie degeneracy logic lives entirely in the later reduction; this helper
    only rewrites each stream's index layout.

    Returns ``(energy, count, resolved_idxs)`` where ``resolved_idxs`` has no
    repeats. If ``idxs`` has no repeats, the streams are sliced out unchanged
    and ``idxs`` is returned as-is (passthrough -- 2-operand behavior is
    identical to the pre-helper code path).
    """
    if len(set(idxs)) != len(idxs):
        resolved = "".join(dict.fromkeys(idxs))
        e = be.einsum("".join(idxs) + "->" + resolved, pair[..., 0])
        n = be.einsum("".join(idxs) + "->" + resolved, pair[..., 1])
        return e, n, list(resolved)
    return pair[..., 0], pair[..., 1], list(idxs)


def _counting_einsum(be: Backend, eq: str, *operands: Tensor) -> Tensor:
    """(energy, degeneracy) einsum over max-plus.

    Handles the 1-operand and 2-operand forms that cotengra can emit.

    Mirrors ``_tropical_einsum``: each operand is broadcast to the full index
    layout via ``_expand_to_layout``; the energy stream sums then takes the max
    over each contracted axis; the count stream sums ``a_count * b_count`` only
    at positions within ``_EPS`` of the running energy max (ties contribute).

    Intra-operand repeated indices (diagonal/trace) are resolved per-stream via
    ``_resolve_repeats`` (pure diagonal gather -- no max/tie logic, which lives
    in the reduction). This is the per-stream analogue of ``_tropical_einsum``'s
    ``be.einsum`` gather, split across the two trailing-axis streams so the
    ``[..., 2]`` stack axis is preserved rather than treated as an index.
    """
    if len(operands) == 1:
        a = operands[0]
        in_str, _sep, out_str = eq.partition("->")
        lhs = in_str.split(",")[0]
        rhs = out_str
        e, n, idxs = _resolve_repeats(be, a, list(lhs))
        contract = [c for c in idxs if c not in rhs]
        for ax in sorted([idxs.index(c) for c in contract], reverse=True):
            shp = tuple(int(d) for d in be.shape_tuple(e))
            y_ax = be.max(e, axis=ax)
            y_b = _insert_axis_of(be, y_ax, ax, shp)
            mask = be.abs(e - y_b) < _EPS
            e = y_ax
            n = be.sum(n * mask, axis=ax)
        remaining = [c for c in idxs if c not in contract]
        out_e = be.transpose(e, tuple(remaining.index(c) for c in rhs))
        out_n = be.transpose(n, tuple(remaining.index(c) for c in rhs))
        return _stack_last(be, out_e, out_n)
    a, b = operands
    lhs, rhs = eq.split("->")
    ia_s, ib_s = lhs.split(",")
    a_e, a_n, ia = _resolve_repeats(be, a, list(ia_s))
    b_e, b_n, ib = _resolve_repeats(be, b, list(ib_s))
    all_idx = list(dict.fromkeys(ia + ib))
    out_idx = list(rhs)
    contract = [c for c in all_idx if c not in out_idx]
    # Energy pair-sum and count product broadcast over the full index layout.
    sx = _expand_to_layout(be, a_e, ia, all_idx) + _expand_to_layout(
        be, b_e, ib, all_idx
    )
    pn = _expand_to_layout(be, a_n, ia, all_idx) * _expand_to_layout(
        be, b_n, ib, all_idx
    )
    # Reduce over contracted axes (highest index first so earlier indices stay valid).
    for ax in sorted([all_idx.index(c) for c in contract], reverse=True):
        sx_shape = tuple(int(d) for d in be.shape_tuple(sx))
        y_ax = be.max(sx, axis=ax)  # axis removed
        y_b = _insert_axis_of(be, y_ax, ax, sx_shape)  # axis size 1 -> broadcasts
        mask = be.abs(sx - y_b) < _EPS
        sx = y_ax
        pn = be.sum(pn * mask, axis=ax)
    remaining = [c for c in all_idx if c not in contract]
    out_x = be.transpose(sx, tuple(remaining.index(c) for c in out_idx))
    out_n = be.transpose(pn, tuple(remaining.index(c) for c in out_idx))
    return _stack_last(be, out_x, out_n)


class CountingRepresentation(Representation):
    """Attach count=1 (the counting-semiring multiplicative identity) to each
    leaf; decode splits the final stacked ``[..., 2]`` tensor into energy
    (primary) + count (aux).

    encode: ``t -> stack([t, ones_like(t, float64)], axis=-1)``. Per-tensor,
    topology-agnostic, run once on the leaves before any pairwise contraction.
    decode: ``tensor[..., 0]`` is the energy (primary, rank == len(output_set));
    ``tensor[..., 1]`` is the degeneracy count (aux, co-indexed with energy).
    """

    name = "counting"

    def encode(self, be: Backend, tensors: List[Tensor]) -> List[Tensor]:
        out = []
        for t in tensors:
            ones = be.ones_like(t, dtype="float64")
            out.append(be.stack([t, ones], axis=-1))
        return out

    def decode(self, be: Backend, tensor: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        return tensor[..., 0], {"count": tensor[..., 1]}


class CountingTropicalAlgebra(ContractionAlgebra):
    """Counting tropical algebra: (energy, degeneracy) over max-plus.

    Carries ``CountingRepresentation``: encode attaches count=1 to each leaf;
    decode splits the final pair into energy (primary) + count (aux, stashed
    via ``cons._stash_aux_outputs`` for ``degeneracy()`` to read). No
    ``on_contraction_start`` hook is needed: the unconditional aux clear in
    ``cons._algebraic_base_contraction`` already wipes stale state per
    contraction (standard or non-standard).
    """

    name = "counting_maxplus"
    representation = CountingRepresentation()

    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        return _counting_tensordot(be, a, b, axes)

    def einsum(self, be: Backend, eq: str, *operands: Tensor) -> Tensor:
        return _counting_einsum(be, eq, *operands)


def degeneracy() -> Optional[Any]:
    """Degeneracy (number of optimal configs) of the most recent counting
    contraction.

    Call inside the ``counting_tropical()`` block (like
    ``recover_configuration``) after a contraction has run. Returns the count
    tensor (same shape as the primary energy) stashed by
    ``CountingRepresentation.decode``; returns ``None`` if no counting decode
    has stashed a count for the current contraction.

    The aux side-channel is cleared at the start of every contraction that
    routes through ``cons._algebraic_base_contraction`` (the unconditional
    ``_stash_aux_outputs({})`` at the top of that function), so a counting
    contraction followed by another algebraic contraction wipes stale state.
    Note that ``cons._base`` only routes to ``_algebraic_base_contraction``
    for non-standard algebras, hyperedge inputs, or ``use_primitives=True``;
    a standard contraction on the original ``_base`` path does NOT clear aux.
    """
    return cons._aux_outputs().get("count")


# ===== Section 3: tracking + configuration recovery =====

# Per-call argmax records, appended in call-order (== tree.traverse() order).
# Reset by on_contraction_start (fired by cons._algebraic_base_contraction) at
# the start of each tracked contraction so it always reflects exactly the most
# recent contraction.
_trace: List[Dict[str, Any]] = []

# Context stashed by the tracking hooks (fired by cons._algebraic_base_contraction):
# the cotengra tree plus the extracted topology (raw leaf arrays + index terms).
_ctx: Dict[str, Any] = {
    "tree": None,
    "input_sets": None,
    "raw_tensors": None,
}


def _reset_trace() -> None:
    """Clear the argmax trace (called by on_contraction_start, fired by cons._algebraic_base_contraction).

    Also drops any stashed tree so a failed/short-circuited contraction cannot
    be confused with the previous one by ``recover_configuration``.
    """
    _trace.clear()
    _ctx["tree"] = None


def _set_tracking_context(
    tree: Any,
    input_sets: Optional[Sequence[Sequence[Any]]] = None,
    raw_tensors: Optional[Sequence[Any]] = None,
) -> None:
    """Stash the cotengra tree (and optionally the leaf topology) for backtracking."""
    _ctx["tree"] = tree
    _ctx["input_sets"] = input_sets
    _ctx["raw_tensors"] = raw_tensors


def get_recorded_topology() -> Tuple[Any, Any, Any]:
    """Return ``(tree, input_sets, raw_tensors)`` stashed by the last tracked
    contraction. The test harness uses this to verify a recovered config's
    energy without needing to know cotengra's internal symbol->spin mapping."""
    return _ctx["tree"], _ctx["input_sets"], _ctx["raw_tensors"]


def _axes_to_lists(ashape: Sequence[int], axes: Any) -> Tuple[List[int], List[int]]:
    """Normalize a tensordot ``axes`` arg to ``(a_axes, b_axes)`` (mirror of
    ``_pair_layout``)."""
    if isinstance(axes, int):
        a_axes = list(range(len(ashape) - axes, len(ashape)))
        b_axes = list(range(0, axes))
    else:
        a_axes, b_axes = list(axes[0]), list(axes[1])
    return a_axes, b_axes


def _tracking_tensordot(be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
    """max-plus tensordot (identical result to ``_tropical_tensordot``) that
    additionally records the argmax over the contracted axis per output pos.

    Record shape: ``argmax`` has the pairwise output shape in the algebra's
    natural order (``a_free + b_free``); values are the *flattened* contracted
    index in row-major over ``a_axes`` (so ``np.unravel_index`` with
    ``contract_dims`` recovers per-label values). ``contract_dims`` follows the
    order of ``a_axes`` (== ``tree.get_tensordot_axes(p)[0]``).

    Argmax uses ``be.argmax`` (portable across numpy/jax/tensorflow); the
    result is materialised to numpy for the backtracking index walk.
    """
    a3, b3, out_shape = _pair_layout(be, a, b, axes)  # (m,k,1),(1,k,n)
    s = a3 + b3  # (m,k,n)
    red = be.max(s, axis=1)  # (m,n) max-plus reduction
    out = be.reshape(red, out_shape)

    ashape = tuple(int(x) for x in be.shape_tuple(a))
    a_axes, _b_axes = _axes_to_lists(ashape, axes)
    contract_dims = tuple(ashape[ax] for ax in a_axes)

    am = be.argmax(s, axis=1)  # (m,n) -> flattened contracted index
    am = be.reshape(am, out_shape)
    _trace.append(
        {
            "kind": "td",
            "argmax": np.asarray(am),
            "contract_dims": contract_dims,
        }
    )
    return out


def _tracking_einsum(be: Backend, eq: str, *operands: Tensor) -> Tensor:
    """max-plus einsum (identical result to ``_tropical_einsum``) that records
    the argmax over the contracted index subspace per output position.

    For 2-operand forms the pair-sum is built over the full index layout (as in
    ``_tropical_einsum``), transposed to ``(out_labels + contract_labels)``,
    the contracted axes flattened into one trailing axis, and ``be.argmax``
    taken over it -- vectorised and portable (numpy/jax/tensorflow). Single-
    operand forms record a no-op entry (no pairwise choice to backtrack).
    """
    if len(operands) == 1:
        _trace.append(
            {
                "kind": "ein1",
                "argmax": None,
                "contract_dims": (),
                "contract_labels": (),
                "eq": eq,
            }
        )
    else:
        a, b = operands
        lhs, rhs = eq.split("->")
        ia_s, ib_s = lhs.split(",")
        ia, ib = list(ia_s), list(ib_s)
        out_labels = list(rhs)
        all_idx = list(dict.fromkeys(ia + ib))
        contract_labels = [c for c in all_idx if c not in out_labels]

        # Full-layout pair sum (same construction as _tropical_einsum).
        s = _expand_to_layout_pair(be, a, b, ia, ib, all_idx)

        if not contract_labels:
            # Pure outer product (hyperedge with no contraction): no argmax to take.
            _trace.append(
                {
                    "kind": "ein2",
                    "argmax": None,
                    "contract_dims": (),
                    "contract_labels": (),
                    "eq": eq,
                }
            )
        else:
            # Reorder to (out_labels + contract_labels) and flatten the contracted axes.
            perm = tuple(all_idx.index(c) for c in out_labels + contract_labels)
            s = be.transpose(s, perm)
            full_shape = tuple(int(d) for d in be.shape_tuple(s))
            n_out = len(out_labels)
            out_shape = full_shape[:n_out]
            contract_dims = full_shape[n_out:]
            flat = int(np.prod(contract_dims))
            s_flat = be.reshape(s, out_shape + (flat,))
            am = be.argmax(
                s_flat, axis=-1
            )  # shape == out_shape, row-major over contract
            _trace.append(
                {
                    "kind": "ein2",
                    "argmax": np.asarray(am),
                    "contract_dims": tuple(contract_dims),
                    "contract_labels": tuple(contract_labels),
                    "eq": eq,
                }
            )
    return _tropical_einsum(be, eq, *operands)


def _expand_to_layout_pair(
    be: Backend,
    a: Tensor,
    b: Tensor,
    ia: Sequence[str],
    ib: Sequence[str],
    all_idx: Sequence[str],
) -> Tensor:
    """Build the full-layout pair-sum ``a + b`` broadcast over ``all_idx``.

    Mirrors the pair-sum construction inside ``_tropical_einsum`` (factors
    ``_expand_to_layout`` for both operands). Kept local so this module does not
    reach into a private helper whose signature may change.
    """

    return _expand_to_layout(be, a, ia, all_idx) + _expand_to_layout(be, b, ib, all_idx)


class MaxPlusTrackingAlgebra(MaxPlusAlgebra):
    """Max-plus algebra that records the per-step argmax for config recovery.

    Produces the same contraction value as ``MaxPlusAlgebra``; the only
    side-effect is appending an argmax record to ``_trace`` per pairwise call.
    The trace is reset by ``on_contraction_start`` (fired by
    ``cons._algebraic_base_contraction``) at the start of each tracked
    contraction, so ``recover_configuration`` always reflects the most recent
    contraction.
    """

    name = "maxplus_tracking"

    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        return _tracking_tensordot(be, a, b, axes)

    def einsum(self, be: Backend, eq: str, *operands: Tensor) -> Tensor:
        return _tracking_einsum(be, eq, *operands)

    def on_contraction_start(self, nodes: Any) -> None:
        _reset_trace()
        try:
            _raw, _inputs, _output, _sizes = cons._extract_topology(nodes)
            _set_tracking_context(tree=None, input_sets=_inputs, raw_tensors=_raw)
        except Exception:
            logger.debug(
                "tracking topology stash failed; recover_configuration may raise "
                "a trace/tree mismatch later",
                exc_info=True,
            )
            _set_tracking_context(tree=None)

    def on_contractor_ready(self, tree: Any) -> None:
        _ctx["tree"] = tree


def _td_backtrack_step(
    tree: Any,
    p: Any,
    l: Any,
    r: Any,
    rec: Dict[str, Any],
    p_pos: Tuple[int, ...],
    assignment: Dict[Any, int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Propagate one tensordot contraction's optimal position to its children.

    Returns ``(l_pos, r_pos)`` (positions in ``get_inds(l)`` / ``get_inds(r)``
    order). Side-effect: records the contracted index-label values into
    ``assignment``.

    ``get_tensordot_perm(p)`` is the transpose cotengra applies AFTER the
    algebra call, mapping the algebra's natural output order (``l_free +
    r_free``) to ``get_inds(p)``. It is therefore INVERTED when indexing the
    recorded argmax (which lives in the algebra's natural order) by the
    canonical (``get_inds(p)``) position: ``td_pos[perm[i]] = p_pos[i]``.
    """
    l_inds = list(tree.get_inds(l))
    r_inds = list(tree.get_inds(r))
    l_axes, r_axes = tree.get_tensordot_axes(p)
    perm = tree.get_tensordot_perm(p)

    # Convert canonical (get_inds(p)) position -> algebra-natural (td) order.
    if perm is None:
        td_pos = list(p_pos)
    else:
        td_pos = [0] * len(p_pos)
        for i, v in enumerate(p_pos):
            td_pos[perm[i]] = v
    td_pos_t = tuple(td_pos)

    argmax = rec["argmax"]
    if argmax is None:
        # No contracted axis (should not happen for a tensordot step); nothing
        # to unravel -- pass the position through unchanged.
        return tuple(p_pos[: len(l_inds)]), tuple(p_pos[len(l_inds) :])

    k = int(argmax[td_pos_t])
    contract_dims = rec["contract_dims"]
    if contract_dims:
        unraveled = np.unravel_index(k, contract_dims)  # per contracted axis
    else:
        unraveled = ()

    # Contracted labels follow l_axes order (== the algebra's a_axes order).
    contract_labels = [l_inds[ax] for ax in l_axes]
    for i, lbl in enumerate(contract_labels):
        assignment[lbl] = int(unraveled[i])

    # Split td_pos into the (l_free, r_free) halves (td-natural order).
    l_free_labels = [c for i, c in enumerate(l_inds) if i not in set(l_axes)]
    r_free_labels = [c for i, c in enumerate(r_inds) if i not in set(r_axes)]
    n_lf = len(l_free_labels)
    l_free_vals = td_pos[:n_lf]
    r_free_vals = td_pos[n_lf:]

    l_pos = [0] * len(l_inds)
    for j, lbl in enumerate(l_free_labels):
        l_pos[l_inds.index(lbl)] = l_free_vals[j]
    for i, ax in enumerate(l_axes):
        l_pos[ax] = int(unraveled[i])

    r_pos = [0] * len(r_inds)
    for j, lbl in enumerate(r_free_labels):
        r_pos[r_inds.index(lbl)] = r_free_vals[j]
    for i, ax in enumerate(r_axes):
        r_pos[ax] = int(unraveled[i])  # r_axes[i] pairs with l_axes[i] (same label)

    return tuple(l_pos), tuple(r_pos)


def _ein_backtrack_step(
    p_pos: Tuple[int, ...],
    rec: Dict[str, Any],
) -> Dict[Any, int]:
    """Recover the contracted-label assignment for one einsum step (given the
    output position). Used when ``tree.get_can_dot(p)`` is False."""
    argmax = rec["argmax"]
    contract_dims = rec["contract_dims"]
    if argmax is None or not contract_dims:
        return {}
    k = int(argmax[p_pos])
    unraveled = np.unravel_index(k, contract_dims)
    return {lbl: int(v) for lbl, v in zip(rec["contract_labels"], unraveled)}


def _validate_trace(tree: Any) -> None:
    """Validate that a stashed contraction tree exists and that ``_trace``
    length matches its number of contractions; raise RuntimeError otherwise.

    Called at the top of ``recover_configuration`` so the trace/tree invariant
    holds before backtracking begins.
    """
    if tree is None:
        raise RuntimeError(
            "recover_configuration: no stashed tree -- contract under "
            "tropical(track=True) first."
        )
    n_nodes = len(list(tree.traverse()))
    if len(_trace) != n_nodes:
        raise RuntimeError(
            f"recover_configuration: trace length ({len(_trace)}) != tree "
            f"contractions ({n_nodes}); call after exactly one tracked "
            "contraction."
        )


def _finalize_from_leaves(
    tree: Any,
    opt_pos: Dict[Any, Tuple[int, ...]],
    assignment: Dict[Any, int],
) -> None:
    """Finalize ``assignment`` from leaf positions, catching labels that appear
    only on a leaf axis (never seen on a contracted/tensordot step).
    Mutates ``assignment`` in place.
    """
    for leaf in tree.gen_leaves():
        pos = opt_pos.get(leaf)
        if pos is None:
            continue
        for ax, lbl in enumerate(list(tree.get_inds(leaf))):
            assignment[lbl] = int(pos[ax])


def recover_configuration() -> Dict[Any, int]:
    """Walk the stashed tree top-down and recover each index label's optimal
    value. Returns ``{index_label: value}``.

    Requires that the contraction was performed under ``MaxPlusTrackingAlgebra``
    (i.e. ``tropical(track=True)``) so that ``_trace`` and ``_ctx['tree']`` are
    populated, and called after exactly one tracked contraction (the trace is
    reset per contraction by ``on_contraction_start``).

    Only scalar roots are supported (a full contraction to an energy, the
    Ising use case). If the contraction has dangling/free output indices
    (``len(tree.output) != 0``) a ``NotImplementedError`` is raised: the
    non-scalar backtracking path has known ordering bugs (``tree.output`` order
    vs result-shape order; output-label values lost in the einsum branch) that
    would produce wrong configs, so it is gated rather than shipping wrong answers.
    Contract to a scalar first.

    Tie-breaking is first-argument-wins (lowest flattened contracted index on
    ties), so the returned configuration is *an* optimum; degenerate optima are
    not enumerated (that is Task B's remit).
    """
    tree = _ctx["tree"]
    _validate_trace(tree)

    trav = list(tree.traverse())  # call-order == algebra call order
    node_rec: Dict[Any, Dict[str, Any]] = {}
    for (p, _l, _r), rec in zip(trav, _trace):
        node_rec[p] = rec

    # Scalar-only: gate the non-scalar (free/output-index) root, whose
    # backtracking wiring has known ordering bugs (tree.output order vs
    # result-shape order; output-label values lost in the einsum branch). The
    # tested scalar path below stays intact; ``get_tensordot_perm`` inversion
    # logic (correct) is still exercised by the scalar canary + synthetic test.
    n_out = len(tree.output)
    if n_out != 0:
        raise NotImplementedError(
            "non-scalar configuration recovery not supported; "
            "contract to a scalar first"
        )
    root_pos: Tuple[int, ...] = ()

    assignment: Dict[Any, int] = {}
    opt_pos: Dict[Any, Tuple[int, ...]] = {}
    opt_pos[tree.root] = root_pos

    for p, l, r in tree.descend():
        rec = node_rec[p]
        p_pos = opt_pos[p]
        if rec["kind"] == "td":
            l_pos, r_pos = _td_backtrack_step(tree, p, l, r, rec, p_pos, assignment)
            opt_pos[l] = l_pos
            opt_pos[r] = r_pos
        else:
            # einsum (hyperedge): recover contracted labels; the child
            # positions in canonical order are derived from the labels.
            cont = _ein_backtrack_step(p_pos, rec)
            assignment.update(cont)
            # Reconstruct child positions from the label->value map so deeper
            # tensordot steps (which read opt_pos) stay consistent. Each child's
            # surviving labels are exactly its get_inds restricted to known vals.
            l_inds = list(tree.get_inds(l))
            r_inds = list(tree.get_inds(r))
            opt_pos[l] = tuple(assignment[c] if c in assignment else 0 for c in l_inds)
            opt_pos[r] = tuple(assignment[c] if c in assignment else 0 for c in r_inds)

    _finalize_from_leaves(tree, opt_pos, assignment)
    return assignment


# ===== Section 4: context managers =====


@contextlib.contextmanager
def tropical(track: bool = False) -> Iterator[None]:
    """Contract under the max-plus (tropical) algebra within the block.

    Swaps ``cons._contraction_algebra`` directly (no monkey-patch activation):
    the in-source ``cons._base`` routes to ``_algebraic_base_contraction`` for
    any non-standard algebra, so this is sufficient.

    ``track=True`` switches to ``MaxPlusTrackingAlgebra`` so
    ``recover_configuration()`` can recover the optimal configuration after the
    contraction. Off by default -> zero behaviour change relative to plain
    max-plus.
    """
    algebra = MaxPlusTrackingAlgebra() if track else MaxPlusAlgebra()
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(algebra)
    try:
        yield
    finally:
        cons.set_contraction_algebra(prev)


@contextlib.contextmanager
def counting_tropical() -> Iterator[None]:
    """Contract under the counting (energy, degeneracy) max-plus algebra within
    the block."""
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(CountingTropicalAlgebra())
    try:
        yield
    finally:
        cons.set_contraction_algebra(prev)
