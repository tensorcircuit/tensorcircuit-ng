"""
Block-buffer tensor views and elementary symmetry-preserving operations.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Tuple

import jax
import jax.numpy as jnp

from .layout import BlockLayout
from .symmetry import Charge, SectorIndex

Array = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SymTensor:
    """
    A static block layout with dynamic JAX buffers for every legal block.

    :ivar layout: Static legal block layout.
    :ivar buffers: Dynamic values for the legal blocks in layout order.
    """

    layout: BlockLayout
    buffers: Tuple[Array, ...]

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], BlockLayout]:
        """
        Flatten the block tensor into dynamic buffers and static layout data.

        :return: Dynamic buffers and the static block layout.
        :rtype: Tuple[Tuple[Array, ...], BlockLayout]
        """
        return self.buffers, self.layout

    @classmethod
    def tree_unflatten(
        cls, layout: BlockLayout, leaves: Iterable[Array]
    ) -> "SymTensor":
        """
        Reconstruct a block tensor from its static layout and dynamic leaves.

        :param layout: Static block layout.
        :type layout: BlockLayout
        :param leaves: Dynamic block buffers.
        :type leaves: Iterable[Array]
        :return: Reconstructed block tensor.
        :rtype: SymTensor
        """
        leaves = tuple(leaves)
        if len(leaves) != len(layout.block_shapes):
            raise ValueError("SymTensor leaves do not match the static block layout")
        return cls(layout, leaves)


@dataclass(frozen=True)
class FusionPlan:
    """
    Static reversible canonical-basis reordering for fused tensor legs.

    :ivar input_dims: Dimensions of the unfused input axes.
    :ivar output_index: Sector index of the fused axis.
    :ivar product_to_fused: Product-basis to fused-basis permutation.
    :ivar fused_to_product: Fused-basis to product-basis permutation.
    """

    input_dims: Tuple[int, ...]
    output_index: SectorIndex
    product_to_fused: Tuple[int, ...]
    fused_to_product: Tuple[int, ...]


def _basis_charges(index: SectorIndex) -> Tuple[Tuple[int, ...], ...]:
    return tuple(
        charge
        for charge, degeneracy in zip(index.charges, index.degeneracies)
        for _ in range(degeneracy)
    )


def fuse_indices(
    indices: Tuple[SectorIndex, ...], *, flow: int = 1
) -> Tuple[SectorIndex, FusionPlan]:
    """
    Fuse directed Abelian indices and return the reversible basis plan.

    :param indices: Input sector indices to fuse.
    :type indices: Tuple[SectorIndex, ...]
    :param flow: Direction of the fused index, either ``1`` or ``-1``.
    :type flow: int
    :return: The fused sector index and its reversible basis permutation plan.
    :rtype: Tuple[SectorIndex, FusionPlan]
    """
    if not indices:
        raise ValueError("fuse_indices requires at least one index")
    if flow not in (-1, 1):
        raise ValueError("fused index flow must be +1 or -1")
    symmetry = indices[0].symmetry
    if any(index.symmetry != symmetry for index in indices):
        raise ValueError("all fused indices must use the same Abelian symmetry")
    basis_charges = tuple(_basis_charges(index) for index in indices)
    grouped: dict[Charge, list[int]] = {}
    dimensions = tuple(index.dimension for index in indices)
    for coordinates in product(*(range(dimension) for dimension in dimensions)):
        signed_charge = (0,) * symmetry.rank
        for index, charges, coordinate in zip(indices, basis_charges, coordinates):
            charge = charges[coordinate]
            signed_charge = symmetry.add(
                signed_charge,
                charge if index.flow == 1 else symmetry.negate(charge),
            )
        output_charge = signed_charge if flow == 1 else symmetry.negate(signed_charge)
        flat_position = 0
        for dimension, coordinate in zip(dimensions, coordinates):
            flat_position = flat_position * dimension + coordinate
        grouped.setdefault(output_charge, []).append(flat_position)
    charges = tuple(sorted(grouped))
    fused_to_product = tuple(
        position for charge in charges for position in grouped[charge]
    )
    product_to_fused_list = [0] * len(fused_to_product)
    for fused_position, product_position in enumerate(fused_to_product):
        product_to_fused_list[product_position] = fused_position
    output_index = SectorIndex.from_sectors(
        symmetry=symmetry,
        sectors=tuple((charge, len(grouped[charge])) for charge in charges),
        flow=flow,
    )
    return output_index, FusionPlan(
        input_dims=dimensions,
        output_index=output_index,
        product_to_fused=tuple(product_to_fused_list),
        fused_to_product=fused_to_product,
    )


def fuse_array(tensor: Array, axes: Tuple[int, ...], plan: FusionPlan) -> Array:
    """
    Fuse selected axes of a dense canonical-basis tensor using ``plan``.

    :param tensor: Dense tensor in canonical basis order.
    :type tensor: Array
    :param axes: Tensor axes corresponding to ``plan.input_dims``.
    :type axes: Tuple[int, ...]
    :param plan: Static fusion and basis-reordering plan.
    :type plan: FusionPlan
    :return: Tensor with the selected axes replaced by the fused axis.
    :rtype: Array
    """
    rank = tensor.ndim
    if len(axes) != len(plan.input_dims) or len(set(axes)) != len(axes):
        raise ValueError("fusion axes do not match the plan")
    if tuple(tensor.shape[axis] for axis in axes) != plan.input_dims:
        raise ValueError("tensor dimensions do not match the fusion plan")
    remaining = tuple(axis for axis in range(rank) if axis not in axes)
    permuted = jnp.transpose(tensor, axes + remaining)
    product_dimension = 1
    for dimension in plan.input_dims:
        product_dimension *= dimension
    reshaped = jnp.reshape(
        permuted,
        (product_dimension,) + tuple(tensor.shape[axis] for axis in remaining),
    )
    return jnp.take(
        reshaped, jnp.asarray(plan.fused_to_product, dtype=jnp.int32), axis=0
    )


def unfuse_array(tensor: Array, axes: Tuple[int, ...], plan: FusionPlan) -> Array:
    """
    Apply the inverse of :func:`fuse_array`.

    :param tensor: Tensor whose first axis is the fused axis.
    :type tensor: Array
    :param axes: Sorted output positions of the unfused axes.
    :type axes: Tuple[int, ...]
    :param plan: Static fusion and basis-reordering plan.
    :type plan: FusionPlan
    :return: Tensor with the fused axis expanded into the requested axes.
    :rtype: Array
    """
    if tensor.shape[0] != plan.output_index.dimension:
        raise ValueError("fused tensor axis does not match the fusion plan")
    product_order = jnp.take(
        tensor, jnp.asarray(plan.product_to_fused, dtype=jnp.int32), axis=0
    )
    expanded = jnp.reshape(product_order, plan.input_dims + tensor.shape[1:])
    rank = len(plan.input_dims) + tensor.ndim - 1
    if tuple(sorted(axes)) != axes or len(axes) != len(plan.input_dims):
        raise ValueError("unfuse axes must be sorted output positions")
    remaining = tuple(axis for axis in range(rank) if axis not in axes)
    permutation = [0] * rank
    for source, destination in enumerate(axes):
        permutation[destination] = source
    for source, destination in enumerate(remaining, start=len(axes)):
        permutation[destination] = source
    return jnp.transpose(expanded, tuple(permutation))


def site_view(state: Any, site: int) -> SymTensor:
    """
    Return a lightweight block-tensor view of one MPS or MPO site.

    :param state: Symmetric MPS or MPO state.
    :type state: Any
    :param site: Site index to view.
    :type site: int
    :return: A view using the state's static site layout and buffers.
    :rtype: SymTensor
    """
    if state.spec.site_layouts is None:
        raise ValueError("site_view is defined only for symmetric block layouts")
    return SymTensor(state.spec.site_layouts[site], state.buffers[site])


def to_dense(tensor: SymTensor) -> Array:
    """
    Materialize a block tensor in canonical sector-basis order.

    :param tensor: Block tensor to materialize.
    :type tensor: SymTensor
    :return: Dense tensor in canonical sector-basis order.
    :rtype: Array
    """
    shape = tuple(index.dimension for index in tensor.layout.indices)
    result = jnp.zeros(shape, dtype=tensor.buffers[0].dtype)
    offsets = tuple(index.offsets for index in tensor.layout.indices)
    for block, coordinate, block_shape in zip(
        tensor.buffers,
        tensor.layout.block_coordinates,
        tensor.layout.block_shapes,
    ):
        starts = tuple(offsets[axis][sector] for axis, sector in enumerate(coordinate))
        slices = tuple(
            slice(start, start + size) for start, size in zip(starts, block_shape)
        )
        result = result.at[slices].set(block)
    return result


def from_dense(layout: BlockLayout, tensor: Array) -> SymTensor:
    """
    Extract exactly the layout's legal blocks from a canonical dense tensor.

    :param layout: Static layout whose legal blocks should be extracted.
    :type layout: BlockLayout
    :param tensor: Dense tensor matching the layout's full shape.
    :type tensor: Array
    :return: Block tensor containing the layout's legal blocks.
    :rtype: SymTensor
    """
    shape = tuple(index.dimension for index in layout.indices)
    if tensor.shape != shape:
        raise ValueError(f"dense tensor shape {tensor.shape} does not match {shape}")
    offsets = tuple(index.offsets for index in layout.indices)
    buffers = []
    for coordinate, block_shape in zip(layout.block_coordinates, layout.block_shapes):
        starts = tuple(offsets[axis][sector] for axis, sector in enumerate(coordinate))
        slices = tuple(
            slice(start, start + size) for start, size in zip(starts, block_shape)
        )
        buffers.append(tensor[slices])
    return SymTensor(layout, tuple(buffers))


def conjugate(tensor: SymTensor) -> SymTensor:
    """
    Conjugate values and reverse every index flow and total charge.

    :param tensor: Block tensor to conjugate.
    :type tensor: SymTensor
    :return: The conjugated block tensor.
    :rtype: SymTensor
    """
    symmetry = tensor.layout.indices[0].symmetry
    layout = BlockLayout(
        indices=tuple(index.dual() for index in tensor.layout.indices),
        total_charge=symmetry.negate(tensor.layout.total_charge),
        block_coordinates=tensor.layout.block_coordinates,
        block_shapes=tensor.layout.block_shapes,
    )
    return SymTensor(layout, tuple(jnp.conj(block) for block in tensor.buffers))


def transpose(tensor: SymTensor, permutation: Tuple[int, ...]) -> SymTensor:
    """
    Permute axes while retaining the corresponding sector-block coordinates.

    :param tensor: Block tensor to transpose.
    :type tensor: SymTensor
    :param permutation: Permutation of all tensor axes.
    :type permutation: Tuple[int, ...]
    :return: The transposed block tensor.
    :rtype: SymTensor
    """
    rank = len(tensor.layout.indices)
    if tuple(sorted(permutation)) != tuple(range(rank)):
        raise ValueError("permutation must contain each tensor axis exactly once")
    layout = BlockLayout(
        indices=tuple(tensor.layout.indices[axis] for axis in permutation),
        total_charge=tensor.layout.total_charge,
        block_coordinates=tuple(
            tuple(coordinate[axis] for axis in permutation)
            for coordinate in tensor.layout.block_coordinates
        ),
        block_shapes=tuple(
            tuple(shape[axis] for axis in permutation)
            for shape in tensor.layout.block_shapes
        ),
    )
    return SymTensor(
        layout,
        tuple(jnp.transpose(block, permutation) for block in tensor.buffers),
    )


def adjoint(tensor: SymTensor) -> SymTensor:
    """
    Conjugate-transpose a rank-two symmetric matrix tensor.

    :param tensor: Rank-two block tensor to adjoint.
    :type tensor: SymTensor
    :return: The conjugate-transposed block tensor.
    :rtype: SymTensor
    """
    if len(tensor.layout.indices) != 2:
        raise ValueError("adjoint requires a rank-two SymTensor")
    return transpose(conjugate(tensor), (1, 0))


def inner(left: SymTensor, right: SymTensor) -> Array:
    """
    Return a full canonical-basis inner product of two matching layouts.

    :param left: First block tensor.
    :type left: SymTensor
    :param right: Second block tensor with the same layout.
    :type right: SymTensor
    :return: The conjugate inner product ``<left|right>``.
    :rtype: Array
    """
    if left.layout != right.layout:
        raise ValueError("SymTensor inner product requires matching layouts")
    return sum(jnp.vdot(a, b) for a, b in zip(left.buffers, right.buffers))
