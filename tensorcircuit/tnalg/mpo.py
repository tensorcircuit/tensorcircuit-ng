"""
Functional dense MPO data, Pauli-string builders, and contractions.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .layout import BlockLayout, MPOSpec, validate_mps_mpo_specs
from .mps import (
    BlockBuffers,
    MPSState,
    _dense_tensors as _dense_mps_tensors,
    norm,
    pack_block_buffers,
)
from .operator_strings import (
    canonical_site_operator_tables,
    physical_dims_from_indices,
    validate_coefficient_indices,
    validate_operator_input,
    validate_symmetric_operator_groups,
)
from .symmetric_environment import mpo_expectation, mpo_second_moment
from .symmetry import AbelianSymmetry, SectorIndex
from .tensor import SymTensor, to_dense

Array = Any
NumpyArray = np.ndarray[Any, Any]
Component = tuple[Any, tuple[tuple[int, int, NumpyArray], ...]]
Path = tuple[int, tuple[Any, ...], tuple[Component, ...], int, int]
PathPositions = tuple[tuple[tuple[int, int, int, NumpyArray], ...], ...]


def _operator_components(index: Any, operator: NumpyArray) -> Tuple[Component, ...]:
    """
    Group local blocks by transfer charge, preserving one MPO channel per transfer.
    """
    offsets = index.offsets
    components: dict[Any, list[tuple[int, int, NumpyArray]]] = {}
    for output_sector, output_dimension in enumerate(index.degeneracies):
        for input_sector, input_dimension in enumerate(index.degeneracies):
            block = operator[
                offsets[output_sector] : offsets[output_sector] + output_dimension,
                offsets[input_sector] : offsets[input_sector] + input_dimension,
            ]
            if np.any(block != 0):
                transfer = index.symmetry.subtract(
                    index.charges[output_sector], index.charges[input_sector]
                )
                components.setdefault(transfer, []).append(
                    (output_sector, input_sector, block)
                )
    return tuple(
        (transfer, tuple(blocks)) for transfer, blocks in sorted(components.items())
    )


def _sector_position(index: Any, position: int) -> Tuple[int, int]:
    """
    Convert a canonical flat basis position into sector and inner offset.
    """
    offset = 0
    for sector, dimension in enumerate(index.degeneracies):
        if position < offset + dimension:
            return sector, position - offset
        offset += dimension
    raise ValueError("virtual path position is outside its sector index")


def _component_key(component: Component) -> tuple[Any, ...]:
    """
    Return a host-static identity for one definite-transfer local operator.
    """
    transfer, blocks = component
    return transfer, tuple(
        (output, input, np.asarray(matrix).tobytes())
        for output, input, matrix in blocks
    )


def _append_conserving_paths(
    paths: list[Path],
    coefficient: int,
    choices: Tuple[Tuple[Component, ...], ...],
    start: int,
    end: int,
    symmetry: Any,
) -> None:
    zero = (0,) * symmetry.rank
    for components in product(*choices):
        transfer = zero
        transfers = [zero]
        for component_transfer, _ in components:
            transfer = symmetry.add(transfer, component_transfer)
            transfers.append(transfer)
        if transfer == zero:
            paths.append((coefficient, tuple(transfers), components, start, end))


def _compile_mpo_paths(
    codes: NumpyArray,
    site_tables: Tuple[NumpyArray, ...],
    physical_indices: Tuple[Any, ...],
    coefficient_indices: NumpyArray,
    ncoefficients: int,
    *,
    dense: bool = False,
) -> Tuple[MPOSpec, Callable[[Array], "MPOState"]]:
    """
    Compile shared-tail finite-state paths, including the trivial dense group.
    """
    symmetry = physical_indices[0].symmetry
    nterms, nsites = codes.shape
    paths: list[Path] = []
    zero = (0,) * symmetry.rank
    components_by_site = tuple(
        tuple(_operator_components(index, operator) for operator in table)
        for index, table in zip(physical_indices, site_tables)
    )
    for term in range(nterms):
        support = np.flatnonzero(codes[term])
        start, end = (int(support[0]), int(support[-1])) if len(support) else (0, 0)
        choices = tuple(
            components[code]
            for components, code in zip(components_by_site, codes[term])
        )
        _append_conserving_paths(
            paths, int(coefficient_indices[term]), choices, start, end, symmetry
        )
    tails: dict[Any, int] = {}
    tail_ids = []
    for _, _, components, _, _ in paths:
        suffix = 0
        identifiers = [0] * nsites
        for site in range(nsites - 1, -1, -1):
            key = (_component_key(components[site]), suffix)
            if key not in tails:
                tails[key] = len(tails) + 1
            suffix = tails[key]
            identifiers[site] = suffix
        tail_ids.append(identifiers)
    path_states: list[tuple[int, ...]] = []
    state_charges: list[tuple[Any, ...]] = []
    for bond in range(nsites + 1):
        if bond in (0, nsites):
            path_states.append(tuple(0 for _ in paths))
            state_charges.append((zero,))
            continue
        state_ids: dict[Any, int] = {("left",): 0, ("right",): 1}
        charges: list[Any] = [zero, zero]
        state_positions: list[int] = []
        for path_index, (_, transfers, _, start, end) in enumerate(paths):
            if bond <= start:
                state_positions.append(0)
                continue
            if bond > end:
                state_positions.append(1)
                continue
            state_key = (tail_ids[path_index][bond],)
            if state_key not in state_ids:
                state_ids[state_key] = len(state_ids)
                charges.append(transfers[bond])
            state_positions.append(state_ids[state_key])
        path_states.append(tuple(state_positions))
        state_charges.append(tuple(charges))
    bond_indices: list[Any] = []
    for bond in range(nsites + 1):
        sectors = tuple((charge, 1) for charge in state_charges[bond])
        bond_indices.append(
            type(physical_indices[0]).from_basis(
                symmetry=symmetry,
                basis_charges=tuple(charge for charge, _ in sectors),
                flow=1,
            )
        )
    layouts: list[BlockLayout] = []
    for site in range(nsites):
        left = bond_indices[site]
        physical_out = physical_indices[site]
        physical_in = physical_indices[site].dual()
        right = bond_indices[site + 1].dual()
        coordinates = []
        shapes = []
        for left_sector, left_charge in enumerate(left.charges):
            for output_sector, output_charge in enumerate(physical_out.charges):
                for input_sector, input_charge in enumerate(physical_in.charges):
                    target = symmetry.add(
                        left_charge,
                        symmetry.subtract(output_charge, input_charge),
                    )
                    if target not in right.charges:
                        continue
                    right_sector = right.charges.index(target)
                    coordinates.append(
                        (left_sector, output_sector, input_sector, right_sector)
                    )
                    shapes.append(
                        (
                            left.degeneracies[left_sector],
                            physical_out.degeneracies[output_sector],
                            physical_in.degeneracies[input_sector],
                            right.degeneracies[right_sector],
                        )
                    )
        layouts.append(
            BlockLayout(
                indices=(left, physical_out, physical_in, right),
                total_charge=zero,
                block_coordinates=tuple(coordinates),
                block_shapes=tuple(shapes),
            )
        )
    path_positions: list[PathPositions] = []
    for path_position, (_, transfers, components, _, _) in enumerate(paths):
        site_positions: list[tuple[tuple[int, int, int, NumpyArray], ...]] = []
        for site, (_, blocks) in enumerate(components):
            left_position = (
                (0, 0)
                if site == 0
                else _sector_position(
                    bond_indices[site],
                    bond_indices[site].basis_to_canonical[
                        path_states[site][path_position]
                    ],
                )
            )
            right_position = (
                (0, 0)
                if site == nsites - 1
                else _sector_position(
                    bond_indices[site + 1],
                    bond_indices[site + 1].basis_to_canonical[
                        path_states[site + 1][path_position]
                    ],
                )
            )
            block_positions: list[tuple[int, int, int, NumpyArray]] = []
            for output_sector, input_sector, matrix in blocks:
                coordinate = (
                    left_position[0],
                    output_sector,
                    input_sector,
                    right_position[0],
                )
                if coordinate not in layouts[site].block_coordinates:
                    raise ValueError(
                        f"invalid symmetric MPO path at site {site}: {coordinate}, "
                        f"transfers={transfers}, blocks={layouts[site].block_coordinates}"
                    )
                block_positions.append(
                    (
                        layouts[site].block_coordinates.index(coordinate),
                        left_position[1],
                        right_position[1],
                        matrix,
                    )
                )
            site_positions.append(tuple(block_positions))
        path_positions.append(tuple(site_positions))

    active_positions: list[set[int]] = [set() for _ in range(nsites)]
    for path in path_positions:
        for site, position_data in enumerate(path):
            active_positions[site].update(position for position, *_ in position_data)
    remapped_path_positions: list[
        list[tuple[tuple[int, int, int, NumpyArray], ...]]
    ] = []
    pruned_layouts: list[BlockLayout] = []
    for site, layout in enumerate(layouts):
        active = tuple(sorted(active_positions[site]))
        if not active:
            active = tuple(range(len(layout.block_coordinates)))
        remap = {position: index for index, position in enumerate(active)}
        pruned_layouts.append(
            BlockLayout(
                indices=layout.indices,
                total_charge=layout.total_charge,
                block_coordinates=tuple(
                    layout.block_coordinates[position] for position in active
                ),
                block_shapes=tuple(
                    layout.block_shapes[position] for position in active
                ),
            )
        )
        for path_index, path in enumerate(path_positions):
            if site == 0:
                remapped_path_positions.append([])
            remapped_path_positions[path_index].append(
                tuple(
                    (remap[position], left_offset, right_offset, matrix)
                    for position, left_offset, right_offset, matrix in path[site]
                )
            )
    final_path_positions = tuple(tuple(path) for path in remapped_path_positions)
    final_layouts = tuple(pruned_layouts)
    spec = MPOSpec(
        physical_dims=tuple(index.dimension for index in physical_indices),
        bond_dims=tuple(index.dimension for index in bond_indices),
        symmetry=None if dense else symmetry,
        physical_sector_indices=None if dense else physical_indices,
        bond_indices=None if dense else tuple(bond_indices),
        site_layouts=None if dense else final_layouts,
    )
    edges = []
    populated_edges = set()
    for operator_path, positions in zip(paths, final_path_positions):
        coefficient, _, _, start, _ = operator_path
        for site, position_data in enumerate(positions):
            for position, left_offset, right_offset, matrix in position_data:
                edge = (site, position, left_offset, right_offset, matrix.tobytes())
                if site != start and edge in populated_edges:
                    continue
                populated_edges.add(edge)
                edges.append(
                    (
                        site,
                        position,
                        left_offset,
                        right_offset,
                        matrix,
                        coefficient if site == start else None,
                    )
                )

    def build_mpo(weights: Array) -> MPOState:
        """
        Build an MPO from one dynamic value per coefficient group.

        :param weights: Dynamic coefficient values in group-index order.
        :type weights: Array
        :return: MPO buffers with the static compiled layout.
        :rtype: MPOState
        """
        weights = jnp.asarray(weights)
        if weights.ndim != 1 or weights.shape[0] != ncoefficients:
            raise ValueError(f"weights must have shape ({ncoefficients},)")
        dtype = jnp.result_type(weights, site_tables[0])
        buffers = [
            [jnp.zeros(shape, dtype=dtype) for shape in layout.block_shapes]
            for layout in final_layouts
        ]
        for site, position, left_offset, right_offset, matrix, coefficient in edges:
            scale = weights[coefficient] if coefficient is not None else 1
            buffers[site][position] = (
                buffers[site][position]
                .at[left_offset, :, :, right_offset]
                .add(scale * jnp.asarray(matrix, dtype=dtype))
            )
        if dense:
            return MPOState(spec, tuple(tuple(site) for site in buffers))
        return MPOState(
            spec,
            BlockBuffers(
                spec, pack_block_buffers(spec, tuple(tuple(site) for site in buffers))
            ),
        )

    return spec, build_mpo


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MPOState:
    """
    A fixed-shape MPO whose numerical buffers are JAX PyTree leaves.

    :ivar spec: Static physical, virtual, and symmetry layout.
    :ivar buffers: Dense site buffers or symmetric bucket buffers.
    """

    spec: MPOSpec
    buffers: Any

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], MPOSpec]:
        """
        Flatten MPO buffers and retain the static specification as metadata.

        :return: Dynamic buffer leaves and the static MPO specification.
        :rtype: Tuple[Tuple[Array, ...], MPOSpec]
        """
        if isinstance(self.buffers, BlockBuffers):
            return self.buffers.buckets, self.spec
        leaves = tuple(buffer for site in self.buffers for buffer in site)
        return leaves, self.spec

    @classmethod
    def tree_unflatten(cls, spec: MPOSpec, leaves: Iterable[Array]) -> "MPOState":
        """
        Reconstruct an MPO state from static metadata and dynamic leaves.

        :param spec: Static MPO specification.
        :type spec: MPOSpec
        :param leaves: Flattened dense or block-bucket leaves.
        :type leaves: Iterable[Array]
        :return: Reconstructed MPO state.
        :rtype: MPOState
        """
        leaves = tuple(leaves)
        if spec.symmetry is not None and spec.site_layouts is not None:
            return cls(spec, BlockBuffers(spec, leaves))
        counts = (
            tuple(len(layout.block_shapes) for layout in spec.site_layouts)
            if spec.site_layouts is not None
            else (1,) * len(spec.physical_dims)
        )
        if len(leaves) != sum(counts):
            raise ValueError("MPO PyTree leaves do not match its static block layout")
        offset = 0
        buffers = []
        for count in counts:
            buffers.append(leaves[offset : offset + count])
            offset += count
        return cls(spec, tuple(buffers))


def _dense_tensors(mpo: MPOState) -> Tuple[Array, ...]:
    """
    Return the single dense buffer of every MPO site.
    """
    if mpo.spec.site_layouts is None:
        return tuple(site[0] for site in mpo.buffers)
    return tuple(
        to_dense(SymTensor(layout, buffers))
        for layout, buffers in zip(mpo.spec.site_layouts, mpo.buffers)
    )


def compile_mpo(
    operator_codes: Array,
    physical_indices: Tuple[Any, ...],
    *,
    operator_table: Any = None,
    coefficient_indices: Any = None,
) -> Tuple[MPOSpec, Callable[[Array], MPOState]]:
    """
    Compile fixed tensor Pauli/operator strings into an MPO builder.

    The builder accepts one dynamic value per ``coefficient_indices`` group
    (one per row by default). Shared suffixes define a finite-state MPO;
    channels remain fixed when numerical coefficients vanish.

    :param operator_codes: Integer operator codes with shape
        ``(n_terms, n_sites)``. Code zero is the identity by default.
    :type operator_codes: Array
    :param physical_indices: Local dimensions or explicit sector indices.
    :type physical_indices: Tuple[Any, ...]
    :param operator_table: Optional local operator table. Entry zero must be
        the identity.
    :type operator_table: Any
    :param coefficient_indices: Optional term-to-coefficient group indices.
        When omitted, every operator row receives its own dynamic coefficient.
    :type coefficient_indices: Any
    :return: The static MPO specification and a dynamic MPO builder.
    :rtype: Tuple[MPOSpec, Callable[[Array], MPOState]]
    """
    physical_dims = physical_dims_from_indices(physical_indices)
    codes, table = validate_operator_input(
        operator_codes, physical_dims, operator_table
    )
    nterms = codes.shape[0]
    coefficient_indices, ncoefficients = validate_coefficient_indices(
        coefficient_indices, nterms
    )
    site_tables = canonical_site_operator_tables(table, physical_indices)
    validate_symmetric_operator_groups(
        codes, coefficient_indices, site_tables, physical_indices
    )
    dense = isinstance(physical_indices[0], (int, np.integer))
    if dense:
        physical_indices = tuple(
            SectorIndex.from_sectors(
                symmetry=AbelianSymmetry(()), sectors=(((), dimension),), flow=1
            )
            for dimension in physical_dims
        )
    return _compile_mpo_paths(
        codes,
        site_tables,
        physical_indices,
        coefficient_indices,
        ncoefficients,
        dense=dense,
    )


def as_mpo(
    source: Any,
    *,
    spec: MPOSpec,
    axis_order: Optional[Tuple[str, str, str, str]] = None,
) -> MPOState:
    """
    Import a dense MPO with an explicit source-axis convention.

    Raw tensor sequences and legacy MPO objects require ``axis_order``. The
    target convention is always ``("left", "out", "in", "right")``.

    :param source: MPOState, legacy MPO, or tensor sequence to import.
    :type source: Any
    :param spec: Target dense MPO specification.
    :type spec: MPOSpec
    :param axis_order: Axis labels of raw source tensors.
    :type axis_order: Optional[Tuple[str, str, str, str]]
    :return: The imported MPO state in TNALG axis order.
    :rtype: MPOState
    """
    if isinstance(source, MPOState):
        if source.spec != spec:
            raise ValueError("MPOState specification does not match the requested spec")
        return source
    if spec.site_layouts is not None:
        raise ValueError(
            "importing a dense MPO into a symmetric MPOSpec requires explicit "
            "virtual-sector metadata; use compile_mpo or a future block importer"
        )
    if axis_order is None:
        raise ValueError("axis_order is required when importing a raw MPO")
    expected_axes = ("left", "out", "in", "right")
    if set(axis_order) != set(expected_axes) or len(axis_order) != 4:
        raise ValueError("axis_order must be a permutation of left, out, in, right")
    if hasattr(source, "tensors"):
        source_tensors = tuple(source.tensors)
    else:
        source_tensors = tuple(source)
    if len(source_tensors) != len(spec.physical_dims):
        raise ValueError("source MPO has a different number of sites")
    permutation = tuple(axis_order.index(axis) for axis in expected_axes)
    tensors = []
    for site, tensor in enumerate(source_tensors):
        tensor = jnp.transpose(jnp.asarray(tensor), permutation)
        expected_shape = (
            spec.bond_dims[site],
            spec.physical_dims[site],
            spec.physical_dims[site],
            spec.bond_dims[site + 1],
        )
        if tensor.shape != expected_shape:
            raise ValueError(
                f"source MPO tensor at site {site} has shape {tensor.shape}, "
                f"expected {expected_shape} after axis conversion"
            )
        tensors.append(tensor)
    return MPOState(spec, tuple((tensor,) for tensor in tensors))


def expectation(mps: MPSState, mpo: MPOState, *, normalized: bool = True) -> Array:
    """
    Contract ``<mps|mpo|mps>`` with optional norm-squared normalization.

    :param mps: MPS state used on both sides of the expectation value.
    :type mps: MPSState
    :param mpo: MPO operator to contract.
    :type mpo: MPOState
    :param normalized: Divide by the MPS norm squared when true.
    :type normalized: bool
    :return: The scalar expectation value.
    :rtype: Array
    """
    validate_mps_mpo_specs(mps.spec, mpo.spec)
    if mps.spec.site_layouts is not None and mpo.spec.site_layouts is not None:
        value = mpo_expectation(mps.spec, mpo.spec, mps.buffers, mpo.buffers)
        return value / (norm(mps) ** 2) if normalized else value
    mps_tensors = _dense_mps_tensors(mps)
    environment = jnp.ones(
        (1, 1, 1),
        dtype=jnp.result_type(mps_tensors[0], mpo.buffers[0][0]),
    )
    for tensor, operator in zip(mps_tensors, _dense_tensors(mpo)):
        environment = jnp.einsum(
            "abc,apd,bpqe,cqf->def",
            environment,
            jnp.conj(tensor),
            operator,
            tensor,
        )
    value = environment[0, 0, 0]
    return value / (norm(mps) ** 2) if normalized else value


def variance(mps: MPSState, mpo: MPOState) -> Array:
    """
    Return the real variance of a Hermitian MPO in an MPS state.

    The contraction uses two MPO transfer legs directly, so it keeps the same
    fixed tensor-network representation and does not materialize a dense
    many-body operator.  For a non-Hermitian MPO this quantity is not a
    variance and callers should use an explicit observable instead.

    :param mps: MPS state in which the variance is evaluated.
    :type mps: MPSState
    :param mpo: Hermitian MPO operator.
    :type mpo: MPOState
    :return: The real variance ``<H^2> - <H>^2``.
    :rtype: Array
    """
    validate_mps_mpo_specs(mps.spec, mpo.spec)
    if mps.spec.site_layouts is not None and mpo.spec.site_layouts is not None:
        norm_squared = norm(mps) ** 2
        safe_norm_squared = jnp.where(norm_squared == 0, 1, norm_squared)
        mean = expectation(mps, mpo, normalized=True)
        second_moment = (
            mpo_second_moment(mps.spec, mpo.spec, mps.buffers, mpo.buffers)
            / safe_norm_squared
        )
        return jnp.real(second_moment - mean * mean)
    mps_tensors = _dense_mps_tensors(mps)
    mpo_tensors = _dense_tensors(mpo)
    environment = jnp.ones(
        (1, 1, 1, 1),
        dtype=jnp.result_type(mps_tensors[0], mpo_tensors[0]),
    )
    for tensor, operator in zip(mps_tensors, mpo_tensors):
        environment = jnp.einsum(
            "abcd,ape,bpqf,cqrg,drh->efgh",
            environment,
            jnp.conj(tensor),
            operator,
            operator,
            tensor,
        )
    norm_squared = norm(mps) ** 2
    safe_norm_squared = jnp.where(norm_squared == 0, 1, norm_squared)
    mean = expectation(mps, mpo, normalized=True)
    second_moment = environment[0, 0, 0, 0] / safe_norm_squared
    return jnp.real(second_moment - mean * mean)
