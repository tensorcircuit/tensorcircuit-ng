"""ContractionAlgebra: a generic interface for swapping contraction primitives +
boundary representation, consulted by ``cons._algebraic_base_contraction``.

Implement ``ContractionAlgebra`` (with a ``Representation``) and activate via
``cons.set_contraction_algebra(...)`` (or ``cons.set_contractor(algebra=...)``,
``cons.runtime_contractor(..., algebra=...)``). The in-source ``cons._base``
routes any non-standard algebra to ``_algebraic_base_contraction``, which runs
encode -> algebra kernels -> decode; no monkey-patching is required.

Reference applications: ``applications/tropical_algebra.py``
(max-plus / counting / tracking) and ``applications/bcomplex32_algebra.py``
(bf16 pair).
"""

from .base import (
    ContractionAlgebra,
    StandardAlgebra,
    Representation,
    IdentityRepresentation,
)

__all__ = [
    "ContractionAlgebra",
    "StandardAlgebra",
    "Representation",
    "IdentityRepresentation",
]
