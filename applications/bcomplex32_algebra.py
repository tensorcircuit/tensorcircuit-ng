"""complex<bfloat16> pair-algebra — a reference APPLICATION of ContractionAlgebra.

Pair repr: complex tensor = stack([re, im], axis=-1) of bf16. Contraction = 4 real
bf16 matmuls (4M). Activated via ``set_contractor(algebra=ComplexPairAlgebra())`` or
the ``bcomplex32()`` CM. encode/decode at the ``_algebraic_base_contraction`` boundary
keep the pair axis off tn.Node (dodges the axis==edge wall).
"""

from typing import Any, Dict, Iterator, List, Tuple
import contextlib

import tensorcircuit.cons as cons
from tensorcircuit.contraction_algebra import ContractionAlgebra, Representation

Tensor = Any
Backend = Any


def _bf16_dtype() -> Any:
    import ml_dtypes

    return ml_dtypes.bfloat16


def _complex_to_pair(be: Backend, t: Tensor) -> Tensor:
    """complex tensor -> stack([re, im], axis=-1) of bf16."""
    bf = _bf16_dtype()
    re = be.cast(be.real(t), bf)
    im = be.cast(be.imag(t), bf)
    return be.stack([re, im], axis=-1)


def _pair_to_complex(be: Backend, pair: Tensor) -> Tensor:
    """pair of bf16 -> complex64 tensor (recombine; no copy risk via cast)."""
    re = be.cast(pair[..., 0], "float32")
    im = be.cast(pair[..., 1], "float32")
    return be.cast(re + 1j * im, "complex64")


def _pair_tensordot(be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
    """Complex tensordot = 4 real bf16 tensordots (4M). Uses be.tensordot (never patched)."""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    cr = be.tensordot(ar, br, axes) - be.tensordot(ai, bi, axes)
    ci = be.tensordot(ar, bi, axes) + be.tensordot(ai, br, axes)
    return be.stack([cr, ci], axis=-1)


def _pair_einsum(be: Backend, eq: str, *operands: Tensor) -> Tensor:
    """Complex einsum = 4 real bf16 einsums (4M for 2 operands, 2 for 1).

    **Two-operand (genuine bf16):** manually decomposed into ``be.tensordot`` +
    ``be.transpose``, because numpy's C ``einsum`` rejects ``ml_dtypes.bfloat16``
    (it is a standalone C routine with a hardcoded dtype allowlist, not a
    ufunc). ``np.tensordot`` accepts bf16 because it dispatches through ufunc
    loops, so the compute is genuine bf16 end-to-end — no float32 upcast.
    The decomposition parses the einsum subscript equation to find contracted
    axes, then uses ``tensordot`` for the contraction and ``transpose`` to
    match the output subscript order.

    **Single-operand (genuine bf16):** decomposed into ``np.diagonal`` +
    ``be.sum`` + ``be.transpose``, all bf16-safe. Handles reductions,
    transposes, diagonals, and traces — the full einsum single-operand
    semantics without any float32 upcast.

    ``_pair_tensordot`` needs no such routing because ``np.tensordot`` already
    preserves bf16 (verified: it accumulates in bf16, not float32).
    cotengra feeds only 1-2-operand equations here (it decomposes hyperedges
    itself), all of which are pairwise contractions that map cleanly to
    tensordot.
    """
    if len(operands) == 1:
        # ── Single-operand: pure bf16 (decompose into sum + transpose + diagonal) ──
        a = operands[0]

        # Implicit mode (no ``->``) is an identity / no-op at the einsum level.
        if "->" not in eq:
            return a

        lhs, out_subs = eq.split("->")

        def _single_op(x: Tensor) -> Tensor:
            """Apply a 1-operand einsum to one bf16 half, staying in bf16."""
            x_subs = list(lhs)  # mutable subscript list we update in place

            # Step 1 — diagonalise every repeated index.
            # ``np.diagonal(x, axis1=p0, axis2=p1)`` removes axis-p1, then
            # appends the diagonal (size = common dim) at the end.
            while True:
                dup = next((c for c in set(x_subs) if x_subs.count(c) > 1), None)
                if dup is None:
                    break
                pos = [i for i, c in enumerate(x_subs) if c == dup]
                x = np.diagonal(x, axis1=pos[0], axis2=pos[-1])
                x_subs = [c for i, c in enumerate(x_subs) if i != pos[-1]] + [dup]

            # Step 2 — sum over indices NOT wanted in the output.
            out_set = set(out_subs)
            sum_indices = [c for c in x_subs if c not in out_set]
            if sum_indices:
                x = be.sum(x, axis=tuple(x_subs.index(c) for c in sum_indices))
                x_subs = [c for c in x_subs if c in out_set]

            # Step 3 — transpose remaining indices into the requested output order.
            if x_subs != list(out_subs):
                perm = tuple(x_subs.index(c) for c in out_subs)
                x = be.transpose(x, perm)

            return x

        return be.stack(
            [_single_op(a[..., 0]), _single_op(a[..., 1])],
            axis=-1,
        )

    # ── 2-operand: bf16-safe tensordot decomposition ──────────────────
    a, b = operands
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]

    # Parse the einsum equation once
    if "->" not in eq:
        raise ValueError(
            f"implicit-mode einsum {eq!r} not supported for bf16; " f"use explicit '->'"
        )
    lhs, out_subs = eq.split("->")
    a_subs, b_subs = lhs.split(",")

    a_set: set[str] = set(a_subs)
    b_set: set[str] = set(b_subs)
    contracted = [c for c in a_subs if c in b_set]

    a_free = [c for c in a_subs if c not in b_set]
    b_free = [c for c in b_subs if c not in a_set]
    out_order = list(out_subs)

    def _contract(x: Tensor, y: Tensor) -> Tensor:
        """Pairwise bf16-safe einsum → tensordot + optional transpose."""
        if contracted:
            a_axes = [a_subs.index(c) for c in contracted]
            b_axes = [b_subs.index(c) for c in contracted]
            result = be.tensordot(x, y, axes=(a_axes, b_axes))
        else:
            result = be.tensordot(x, y, axes=0)

        free_order = a_free + b_free
        if free_order != out_order:
            perm = [free_order.index(c) for c in out_order]
            result = be.transpose(result, perm)
        return result

    cr = _contract(ar, br) - _contract(ai, bi)
    ci = _contract(ar, bi) + _contract(ai, br)
    return be.stack([cr, ci], axis=-1)


class PairBf16Representation(Representation):
    name = "pair_bf16"

    def encode(self, be: Backend, tensors: List[Tensor]) -> List[Tensor]:
        return [_complex_to_pair(be, t) for t in tensors]

    def decode(self, be: Backend, tensor: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        return _pair_to_complex(be, tensor), {}


class ComplexPairAlgebra(ContractionAlgebra):
    name = "bcomplex32_pair"
    representation = PairBf16Representation()
    prefer_einsum = True  # pair operands carry a trailing storage axis

    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        return _pair_tensordot(be, a, b, axes)

    def einsum(self, be: Backend, eq: str, *operands: Tensor) -> Tensor:
        return _pair_einsum(be, eq, *operands)


@contextlib.contextmanager
def bcomplex32() -> Iterator[None]:
    prev = cons.get_contraction_algebra()
    cons.set_contraction_algebra(ComplexPairAlgebra())
    try:
        yield
    finally:
        cons.set_contraction_algebra(prev)
