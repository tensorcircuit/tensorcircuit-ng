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
    """Complex einsum = real bf16 einsums (4M for 2 operands, 2 for 1).

    Each pair component is upcast bf16 -> float32 before ``be.einsum``: numpy's
    einsum does not support ``ml_dtypes.bfloat16`` (``TypeError: invalid data
    type for einsum``), and float32 is the universal portable compute dtype for
    the 4M decomposition. The bf16 quantization happened at encode time
    (``_complex_to_pair``), so the values flowing in are already bf16-quantized;
    this cast only widens the compute dtype, it does not undo quantization.
    ``_pair_tensordot`` needs no such cast because ``np.tensordot`` accepts bf16.
    """
    if len(operands) == 1:
        a = operands[0]
        ar = be.cast(a[..., 0], "float32")
        ai = be.cast(a[..., 1], "float32")
        return be.stack([be.einsum(eq, ar), be.einsum(eq, ai)], axis=-1)
    a, b = operands
    ar = be.cast(a[..., 0], "float32")
    ai = be.cast(a[..., 1], "float32")
    br = be.cast(b[..., 0], "float32")
    bi = be.cast(b[..., 1], "float32")
    cr = be.einsum(eq, ar, br) - be.einsum(eq, ai, bi)
    ci = be.einsum(eq, ar, bi) + be.einsum(eq, ai, br)
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
