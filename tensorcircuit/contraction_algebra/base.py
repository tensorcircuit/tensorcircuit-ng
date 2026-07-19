"""ContractionAlgebra: generic interface for swapping contraction primitives +
boundary representation, consulted by ``cons._algebraic_base_contraction``.

Two ABCs:
- ``ContractionAlgebra``: the arithmetic (tensordot/einsum kernels) + observation
  hooks. Carries a ``Representation`` (boundary codec), default identity.
- ``Representation``: boundary encode (leaves -> physical storage) / decode
  (final -> primary tensor + aux). Covers storage-split (complex<->pair) and
  precision cast (f64<->bf16). Default identity.

kernel must be closed over whatever its Representation produces (author's
contract; Representation is bundled in the algebra so users cannot mis-pair).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

Tensor = Any
Backend = Any


class Representation(ABC):
    """Boundary codec between logical elements and physical storage."""

    name: str = "abstract"

    @abstractmethod
    def encode(self, be: Backend, tensors: List[Tensor]) -> List[Tensor]:
        """Transform the leaf raw_tensors (one pass, per-tensor, topology-agnostic)."""

    @abstractmethod
    def decode(self, be: Backend, tensor: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Transform the final contracted tensor into ``(primary, aux)``.
        ``primary`` must have rank == len(output_set) so ``tn.Node`` wraps it
        consistently. ``aux`` carries side outputs (e.g., degeneracy), sharing
        the primary's physical axis order (sliced from the same tensor)."""


class IdentityRepresentation(Representation):
    """No-op codec: standard storage (real/complex scalars), no aux."""

    name = "identity"

    def encode(self, be: Backend, tensors: List[Tensor]) -> List[Tensor]:
        return tensors

    def decode(self, be: Backend, tensor: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        return tensor, {}


class ContractionAlgebra(ABC):
    """How to perform the two atomic ops of a contraction, plus observation hooks.

    cotengra decomposes any contraction into pairwise steps; each is a
    ``tensordot`` (ordinary pair) or ``einsum`` (hyperedge / copy-node). An
    algebra supplies both. The kernel must be closed over the layout its
    ``representation`` produces.
    """

    name: str = "abstract"
    representation: Representation = IdentityRepresentation()

    def get_contractor_kwargs(self) -> dict:
        """Extra kwargs forwarded to cotengra's ``make_contractor``.

        Override to return ``{'prefer_einsum': True}`` when your algebra's
        ``tensordot`` kernel carries non-physical storage axes (e.g. the
        complex<bf16> pair axis) that cotengra's post-tensordot autoray
        transpose would mishandle (ValueError: axes don't match array).
        ``prefer_einsum=True`` forces einsum-only execution, which skips
        the transpose entirely.

        Default ``{}`` keeps the standard tensordot+einsum mix — required
        by tropical config-recovery backtracking, which depends on the
        tensordot intermediate layout.
        """
        return {}

    @abstractmethod
    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        """Pairwise contraction over ``axes`` (np.tensordot convention)."""

    @abstractmethod
    def einsum(self, be: Backend, eq: str, *operands: Tensor) -> Tensor:
        """Pairwise einsum (cotengra may also call with 1 operand)."""

    def on_contraction_start(self, nodes: Any) -> None:
        """Called before each non-standard contraction (default no-op)."""

    def on_contractor_ready(self, tree: Any) -> None:
        """Called when the cotengra tree is built (default no-op)."""


class StandardAlgebra(ContractionAlgebra):
    """The usual (sum, product) ring — identical to native backend behaviour."""

    name = "standard"

    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        return be.tensordot(a, b, axes)

    def einsum(self, be: Backend, eq: str, *operands: Tensor) -> Tensor:
        return be.einsum(eq, *operands)
