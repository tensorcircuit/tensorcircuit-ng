"""
Sector-wise canonicalization and fixed-quota splitting for symmetric MPS.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from ..backends.jax_ops import adaware_qr, adaware_svd
from .layout import MPSSpec
from .symmetric_environment import apply_left_transfer, apply_right_transfer

Array = Any
BufferSites = list[list[Array]]


def _decompose(matrices: Tuple[Array, ...], function: Any) -> Tuple[Any, ...]:
    """
    Batch independent factorizations with equal matrix shapes.
    """
    groups: dict[Any, list[int]] = {}
    for position, matrix in enumerate(matrices):
        groups.setdefault(matrix.shape, []).append(position)
    results: list[Any] = [None] * len(matrices)
    for positions in groups.values():
        if len(positions) == 1:
            results[positions[0]] = function(matrices[positions[0]])
        else:
            factors = jax.vmap(function)(
                jnp.stack(tuple(matrices[position] for position in positions))
            )
            for row, position in enumerate(positions):
                results[position] = tuple(factor[row] for factor in factors)
    return tuple(results)


def _factor_blocks(
    buffers: BufferSites, spec: MPSSpec, site: int, *, left: bool
) -> Tuple[Array, ...]:
    if spec.site_layouts is None:
        raise ValueError("symmetric MPS specification is incomplete")
    layout = spec.site_layouts[site]
    axis = 2 if left else 0
    matrices, groups = [], []
    for sector, dimension in enumerate(layout.indices[axis].degeneracies):
        positions = tuple(
            i
            for i, coordinate in enumerate(layout.block_coordinates)
            if coordinate[axis] == sector
        )
        shapes = tuple(layout.block_shapes[i] for i in positions)
        if left:
            matrix = jnp.concatenate(
                tuple(
                    jnp.reshape(buffers[site][i], (-1, dimension)) for i in positions
                ),
                axis=0,
            )
            widths = tuple(shape[0] * shape[1] for shape in shapes)
        else:
            matrix = jnp.concatenate(
                tuple(
                    jnp.reshape(buffers[site][i], (dimension, -1)) for i in positions
                ),
                axis=1,
            ).T
            widths = tuple(shape[1] * shape[2] for shape in shapes)
        matrices.append(matrix)
        groups.append((positions, shapes, widths))
    factors = _decompose(tuple(matrices), adaware_qr)
    transfers = []
    for (q_matrix, transfer), (positions, shapes, widths) in zip(factors, groups):
        offset = 0
        for position, shape, width in zip(positions, shapes, widths):
            block = q_matrix[offset : offset + width]
            buffers[site][position] = jnp.reshape(block if left else block.T, shape)
            offset += width
        transfers.append(transfer if left else transfer.T)
    return tuple(transfers)


def factor_left_blocks(
    buffers: BufferSites, spec: MPSSpec, site: int
) -> Tuple[Array, ...]:
    """
    Left-orthogonalize a site, batching equal-shape charge sectors.

    :param buffers: Logical block buffers for the MPS sites.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Site to left-orthogonalize.
    :type site: int
    :return: Transfer matrices for the right virtual sectors.
    :rtype: Tuple[Array, ...]
    """
    return _factor_blocks(buffers, spec, site, left=True)


def factor_right_blocks(
    buffers: BufferSites, spec: MPSSpec, site: int
) -> Tuple[Array, ...]:
    """
    Right-orthogonalize a site, batching equal-shape charge sectors.

    :param buffers: Logical block buffers for the MPS sites.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Site to right-orthogonalize.
    :type site: int
    :return: Transfer matrices for the left virtual sectors.
    :rtype: Tuple[Array, ...]
    """
    return _factor_blocks(buffers, spec, site, left=False)


def shift_center_right(buffers: BufferSites, spec: MPSSpec, site: int) -> None:
    """
    Move a symmetric orthogonality center one site to the right.

    :param buffers: Logical block buffers modified in place.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Current center site.
    :type site: int
    """
    transfer = factor_left_blocks(buffers, spec, site)
    buffers[site + 1] = list(
        apply_left_transfer(spec, site + 1, transfer, tuple(buffers[site + 1]))
    )


def shift_center_left(buffers: BufferSites, spec: MPSSpec, site: int) -> None:
    """
    Move a symmetric orthogonality center one site to the left.

    :param buffers: Logical block buffers modified in place.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Current center site.
    :type site: int
    """
    transfer = factor_right_blocks(buffers, spec, site)
    buffers[site - 1] = list(
        apply_right_transfer(spec, site - 1, transfer, tuple(buffers[site - 1]))
    )


def _dense_transfer(spec: MPSSpec, bond: int, blocks: Tuple[Array, ...]) -> Array:
    assert spec.bond_indices is not None
    middle = spec.bond_indices[bond]
    transfer = jnp.zeros((middle.dimension, middle.dimension), dtype=blocks[0].dtype)
    for offset, block in zip(middle.offsets, blocks):
        transfer = transfer.at[
            offset : offset + block.shape[0], offset : offset + block.shape[1]
        ].set(block)
    return transfer


def factor_left(buffers: BufferSites, spec: MPSSpec, site: int) -> Array:
    """
    Left-orthogonalize one site and return a dense block-diagonal transfer.

    :param buffers: Logical block buffers for the MPS sites.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Site to left-orthogonalize.
    :type site: int
    :return: Dense transfer matrix on the right virtual bond.
    :rtype: Array
    """
    return _dense_transfer(spec, site + 1, factor_left_blocks(buffers, spec, site))


def factor_right(buffers: BufferSites, spec: MPSSpec, site: int) -> Array:
    """
    Right-orthogonalize one site and return a dense block-diagonal transfer.

    :param buffers: Logical block buffers for the MPS sites.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Site to right-orthogonalize.
    :type site: int
    :return: Dense transfer matrix on the left virtual bond.
    :rtype: Array
    """
    return _dense_transfer(spec, site, factor_right_blocks(buffers, spec, site))


def apply_two_site_matrix(
    buffers: BufferSites,
    spec: MPSSpec,
    bond: int,
    gate: Array,
    *,
    forward: bool = False,
) -> Tuple[Array, Array]:
    """
    Split a gate in Schmidt-charge blocks with fixed sector quotas.

    ``forward=True`` stores U on the left and S*Vh on the right; otherwise
    store U*S and Vh. The retained orthonormal factor needs no additional QR.

    :param buffers: Logical block buffers for the two MPS sites.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param bond: Bond separating the two MPS sites.
    :type bond: int
    :param gate: Two-site gate in canonical physical-basis order.
    :type gate: Array
    :param forward: Store ``U`` on the left when true, otherwise store ``U*S``.
    :type forward: bool
    :return: Retained weight and discarded weight.
    :rtype: Tuple[Array, Array]
    """
    if (
        spec.site_layouts is None
        or spec.bond_indices is None
        or spec.physical_sector_indices is None
    ):
        raise ValueError("symmetric MPS specification is incomplete")
    left_layout = spec.site_layouts[bond]
    right_layout = spec.site_layouts[bond + 1]
    physical_left = spec.physical_sector_indices[bond]
    physical_right = spec.physical_sector_indices[bond + 1]
    left_offsets = physical_left.offsets
    right_offsets = physical_right.offsets
    middle = spec.bond_indices[bond + 1]
    left_result = [
        jnp.zeros(shape, dtype=gate.dtype) for shape in left_layout.block_shapes
    ]
    right_result = [
        jnp.zeros(shape, dtype=gate.dtype) for shape in right_layout.block_shapes
    ]
    discarded_weight = jnp.asarray(0, dtype=jnp.real(gate).dtype)
    total_weight = jnp.asarray(0, dtype=jnp.real(gate).dtype)
    matrices = []
    splits = []
    for middle_sector, keep in enumerate(middle.degeneracies):
        row_blocks = [
            (position, coordinate, shape)
            for position, (coordinate, shape) in enumerate(
                zip(left_layout.block_coordinates, left_layout.block_shapes)
            )
            if coordinate[2] == middle_sector
        ]
        column_blocks = [
            (position, coordinate, shape)
            for position, (coordinate, shape) in enumerate(
                zip(right_layout.block_coordinates, right_layout.block_shapes)
            )
            if coordinate[0] == middle_sector
        ]
        chunks = []
        for _, row_coordinate, row_shape in row_blocks:
            row_chunks = []
            for _, column_coordinate, column_shape in column_blocks:
                chunk = jnp.zeros(
                    (
                        row_shape[0],
                        row_shape[1],
                        column_shape[1],
                        column_shape[2],
                    ),
                    dtype=gate.dtype,
                )
                for left_position, left_coordinate in enumerate(
                    left_layout.block_coordinates
                ):
                    if left_coordinate[0] != row_coordinate[0]:
                        continue
                    for right_position, right_coordinate in enumerate(
                        right_layout.block_coordinates
                    ):
                        if (
                            right_coordinate[2] != column_coordinate[2]
                            or left_coordinate[2] != right_coordinate[0]
                        ):
                            continue
                        out_left = row_coordinate[1]
                        out_right = column_coordinate[1]
                        in_left = left_coordinate[1]
                        in_right = right_coordinate[1]
                        gate_block = gate[
                            left_offsets[out_left] : left_offsets[out_left]
                            + physical_left.degeneracies[out_left],
                            right_offsets[out_right] : right_offsets[out_right]
                            + physical_right.degeneracies[out_right],
                            left_offsets[in_left] : left_offsets[in_left]
                            + physical_left.degeneracies[in_left],
                            right_offsets[in_right] : right_offsets[in_right]
                            + physical_right.degeneracies[in_right],
                        ]
                        chunk = chunk + jnp.einsum(
                            "lim,mjr,abij->labr",
                            buffers[bond][left_position],
                            buffers[bond + 1][right_position],
                            gate_block,
                        )
                row_chunks.append(jnp.reshape(chunk, (row_shape[0] * row_shape[1], -1)))
            chunks.append(jnp.concatenate(row_chunks, axis=1))
        matrix = jnp.concatenate(chunks, axis=0)
        matrices.append(matrix)
        splits.append((keep, row_blocks, column_blocks))
    for (u, singular_values, vh), (keep, row_blocks, column_blocks) in zip(
        _decompose(tuple(matrices), adaware_svd), splits
    ):
        total_weight = total_weight + jnp.sum(singular_values**2)
        discarded_weight = discarded_weight + jnp.sum(singular_values[keep:] ** 2)
        u = u[:, :keep]
        vh = vh[:keep]
        if forward:
            vh = singular_values[:keep, None] * vh
        else:
            u = u * singular_values[None, :keep]
        offset = 0
        for position, _, shape in row_blocks:
            width = shape[0] * shape[1]
            left_result[position] = jnp.reshape(
                u[offset : offset + width],
                shape,
            )
            offset += width
        offset = 0
        for position, _, shape in column_blocks:
            width = shape[1] * shape[2]
            right_result[position] = jnp.reshape(vh[:, offset : offset + width], shape)
            offset += width
    buffers[bond] = left_result
    buffers[bond + 1] = right_result
    return discarded_weight, total_weight


def apply_onsite_matrix(
    buffers: BufferSites, spec: MPSSpec, site: int, gate: Array
) -> None:
    """
    Apply an already exponentiated charge-conserving onsite gate.

    :param buffers: Logical block buffers modified in place.
    :type buffers: BufferSites
    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Site receiving the gate.
    :type site: int
    :param gate: Exponentiated onsite gate in canonical basis order.
    :type gate: Array
    """
    if spec.site_layouts is None:
        raise ValueError("symmetric MPS specification is incomplete")
    layout = spec.site_layouts[site]
    physical = layout.indices[1]
    offsets = physical.offsets
    updated = []
    for block, coordinate in zip(buffers[site], layout.block_coordinates):
        physical_sector = coordinate[1]
        start = offsets[physical_sector]
        width = physical.degeneracies[physical_sector]
        local_gate = gate[start : start + width, start : start + width]
        updated.append(jnp.einsum("ab,lbr->lar", local_gate, block))
    buffers[site] = updated
