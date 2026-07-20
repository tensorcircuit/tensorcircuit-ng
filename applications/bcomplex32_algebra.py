"""complex<bfloat16> pair-algebra — a reference APPLICATION of ContractionAlgebra.

Pair repr: complex tensor split into PairTensor(re, im) of bf16. Contraction = 4 real
bf16 matmuls (4M). Activated via ``cons.set_contraction_algebra(ComplexPairAlgebra())`` or
the ``bcomplex32()`` CM. PairTensor keeps the pair axis off tn.Node (dodges the axis==edge wall).
"""

from typing import Any, Dict, Iterator, List, Tuple
import contextlib

import numpy as np

import tensorcircuit.cons as cons
from tensorcircuit.contraction_algebra import (
    ContractionAlgebra,
    PairTensor,
    Representation,
)

Tensor = Any
Backend = Any


def _bf16_dtype() -> Any:
    import ml_dtypes

    return ml_dtypes.bfloat16


def _complex_to_pair(be: Backend, t: Tensor) -> PairTensor:
    """complex tensor -> PairTensor(re, im) of bf16."""
    bf = _bf16_dtype()
    re = be.cast(be.real(t), bf)
    im = be.cast(be.imag(t), bf)
    return PairTensor(re, im)


def _pair_to_complex(be: Backend, pair: PairTensor) -> Tensor:
    """PairTensor of bf16 -> complex tensor (recombine; no copy risk via cast)."""
    re, im = pair.unpack()
    re = be.cast(re, cons.rdtypestr)
    im = be.cast(im, cons.rdtypestr)
    return be.cast(re + 1j * im, cons.dtypestr)


def _pair_tensordot(be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
    """Complex tensordot = 4 real bf16 tensordots (4M). Uses be.tensordot (never patched)."""
    ar, ai = PairTensor.unpack_pair(a)
    br, bi = PairTensor.unpack_pair(b)
    cr = be.tensordot(ar, br, axes) - be.tensordot(ai, bi, axes)
    ci = be.tensordot(ar, bi, axes) + be.tensordot(ai, br, axes)
    return PairTensor.pack_result(be, cr, ci, not isinstance(a, PairTensor))


# Strategy note: bf16 single-operand einsum manually decomposes into diagonal →
# sum → transpose (staying in bf16 end-to-end) because numpy's ``np.einsum``
# rejects bfloat16 dtypes. This is different from the tropical algebra's
# ``_tropical_einsum`` single-operand path, which delegates to ``be.einsum``
# for repeated-index resolution — the tropical backend (float64) has no such
# dtype restriction. Both produce equivalent results under their respective
# semirings, but the implementation strategy is dictated by dtype constraints
# rather than algebraic differences.
def _einsum_single_operand_half(
    be: Backend, x: Tensor, lhs: str, out_subs: str
) -> Tensor:
    """Apply a 1-operand einsum to one bf16 half (decomposed into diagonal + sum
    + transpose), staying in bf16 end-to-end.  Handles reductions, transposes,
    diagonals, and traces without float32 upcast.
    """
    x_subs = list(lhs)  # mutable subscript list we update in place

    # Step 1 — diagonalise every repeated index.
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


def _pair_einsum(be: Backend, eq: str, *operands: Tensor) -> PairTensor:
    """Complex einsum = 4 real bf16 einsums (4M for 2 operands, 2 for 1)."""
    if len(operands) == 1:
        a = operands[0]
        if "->" not in eq:
            return a
        lhs, out_subs = eq.split("->")
        ar, ai = PairTensor.unpack_pair(a)
        return PairTensor(
            _einsum_single_operand_half(be, ar, lhs, out_subs),
            _einsum_single_operand_half(be, ai, lhs, out_subs),
        )

    a, b = operands
    ar, ai = PairTensor.unpack_pair(a)
    br, bi = PairTensor.unpack_pair(b)

    if "->" not in eq:
        raise ValueError(
            f"implicit-mode einsum {eq!r} not supported for bf16; use explicit '->'"
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
    return PairTensor.pack_result(be, cr, ci, not isinstance(a, PairTensor))


class PairBf16Representation(Representation):
    name = "pair_bf16"

    def encode(self, be: Backend, tensors: List[Tensor]) -> List[Tensor]:
        return [_complex_to_pair(be, t) for t in tensors]

    def decode(self, be: Backend, tensor: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        return _pair_to_complex(be, tensor), {}


class ComplexPairAlgebra(ContractionAlgebra):
    name = "bcomplex32_pair"
    representation = PairBf16Representation()

    def get_contractor_kwargs(self) -> Dict[str, Any]:
        return {"prefer_einsum": True}

    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        return _pair_tensordot(be, a, b, axes)

    def einsum(self, be: Backend, eq: str, *operands: Tensor) -> Tensor:
        return _pair_einsum(be, eq, *operands)


@contextlib.contextmanager
def bcomplex32() -> Iterator[None]:
    with cons.runtime_contraction_algebra(ComplexPairAlgebra()):
        yield
