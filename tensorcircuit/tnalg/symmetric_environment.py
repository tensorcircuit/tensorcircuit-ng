"""
Block-buffer environments and local effective operators for Abelian MPS.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, Tuple

import jax
import jax.numpy as jnp

from .layout import MPOSpec, MPSSpec
from .symmetry import SectorIndex
from .block_ops import Metadata, align_blocks, prepare_contraction
from .environment import EnvironmentState

Array = Any
BlockBuffers = Tuple[Array, ...]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LegalEnvironment:
    """
    Only charge-conserving bra-MPO-ket blocks at one MPS cut.

    :ivar mps_spec: Static symmetric MPS layout.
    :ivar mpo_spec: Static symmetric MPO layout.
    :ivar bond: MPS cut represented by the environment.
    :ivar blocks: Legal environment blocks in static coordinate order.
    """

    mps_spec: MPSSpec
    mpo_spec: MPOSpec
    bond: int
    blocks: Tuple[Array, ...]

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], Tuple[MPSSpec, MPOSpec, int]]:
        """
        Flatten legal environment blocks and retain static metadata.

        :return: Dynamic blocks and the MPS/MPO specifications with cut index.
        :rtype: Tuple[Tuple[Array, ...], Tuple[MPSSpec, MPOSpec, int]]
        """
        return self.blocks, (self.mps_spec, self.mpo_spec, self.bond)

    @classmethod
    def tree_unflatten(
        cls, auxiliary: Tuple[MPSSpec, MPOSpec, int], blocks: Tuple[Array, ...]
    ) -> "LegalEnvironment":
        """
        Reconstruct a legal environment from static metadata and blocks.

        :param auxiliary: MPS specification, MPO specification, and cut index.
        :type auxiliary: Tuple[MPSSpec, MPOSpec, int]
        :param blocks: Dynamic legal environment blocks.
        :type blocks: Tuple[Array, ...]
        :return: Reconstructed legal environment.
        :rtype: LegalEnvironment
        """
        mps_spec, mpo_spec, bond = auxiliary
        return cls(mps_spec, mpo_spec, bond, tuple(blocks))

    def _position(self, index: int) -> int:
        if self.mps_spec.bond_indices is None or self.mpo_spec.bond_indices is None:
            raise ValueError("symmetric MPS/MPO specification is incomplete")
        mps_count = len(self.mps_spec.bond_indices[self.bond].charges)
        mpo_count = len(self.mpo_spec.bond_indices[self.bond].charges)
        bra = index // (mpo_count * mps_count)
        remainder = index % (mpo_count * mps_count)
        channel = remainder // mps_count
        ket = remainder % mps_count
        positions = _environment_positions(
            self.mps_spec.bond_indices[self.bond],
            self.mpo_spec.bond_indices[self.bond],
        )
        try:
            return positions[(bra, channel, ket)]
        except KeyError as error:
            raise ValueError("requested an illegal environment block") from error

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self) -> Iterator[Array]:
        return iter(self.blocks)

    def __getitem__(self, index: int) -> Array:
        return self.blocks[self._position(index)]

    def replace(self, index: int, value: Array) -> "LegalEnvironment":
        """
        Return a copy with one logical environment block replaced.

        :param index: Logical environment-block index.
        :type index: int
        :param value: Replacement block value.
        :type value: Array
        :return: Legal environment containing the replacement block.
        :rtype: LegalEnvironment
        """
        position = self._position(index)
        blocks = list(self.blocks)
        blocks[position] = value
        return LegalEnvironment(self.mps_spec, self.mpo_spec, self.bond, tuple(blocks))


def _legal_environment_coordinates(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, bond: int
) -> Tuple[Tuple[int, int, int], ...]:
    if mps_spec.bond_indices is None or mpo_spec.bond_indices is None:
        raise ValueError("symmetric MPS/MPO specification is incomplete")
    return tuple(
        _environment_positions(mps_spec.bond_indices[bond], mpo_spec.bond_indices[bond])
    )


@lru_cache(maxsize=4096)
def _environment_positions(
    mps_index: SectorIndex, mpo_index: SectorIndex
) -> dict[Tuple[int, int, int], int]:
    """
    Join charge tables once, without a quadratic bra/ket search.
    """
    bra_positions = {charge: i for i, charge in enumerate(mps_index.charges)}
    coordinates = []
    for channel, transfer in enumerate(mpo_index.charges):
        for ket, charge in enumerate(mps_index.charges):
            bra = bra_positions.get(mps_index.symmetry.add(charge, transfer))
            if bra is not None:
                coordinates.append((bra, channel, ket))
    return {coordinate: i for i, coordinate in enumerate(sorted(coordinates))}


@dataclass(frozen=True)
class EnvironmentBucketPlan:
    """
    Static exact-shape bucket map for all legal environment blocks.

    :ivar shapes: Padded shapes of environment buckets.
    :ivar positions: Bucket and slot assigned to each legal block.
    :ivar counts: Number of blocks stored in each bucket.
    :ivar true_shapes: Unpadded shape of each legal block.
    :ivar offsets: Flat logical-block offset for every MPS cut.
    """

    shapes: Tuple[Tuple[int, int, int], ...]
    positions: Tuple[Tuple[int, int], ...]
    counts: Tuple[int, ...]
    true_shapes: Tuple[Tuple[int, int, int], ...]
    offsets: Tuple[int, ...]


@lru_cache(maxsize=128)
def environment_bucket_plan(
    mps_spec: MPSSpec, mpo_spec: MPOSpec
) -> EnvironmentBucketPlan:
    """
    Return an exact-shape bucket plan for environments at every cut.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :return: Cached environment bucket plan.
    :rtype: EnvironmentBucketPlan
    """
    if mps_spec.bond_indices is None or mpo_spec.bond_indices is None:
        raise ValueError("symmetric MPS/MPO specification is incomplete")
    true_shapes: list[Tuple[int, int, int]] = []
    offsets: list[int] = []
    for bond in range(mps_spec.nsites + 1):
        offsets.append(len(true_shapes))
        mps_index = mps_spec.bond_indices[bond]
        mpo_index = mpo_spec.bond_indices[bond]
        for bra, channel, ket in _legal_environment_coordinates(
            mps_spec, mpo_spec, bond
        ):
            true_shapes.append(
                (
                    mps_index.degeneracies[bra],
                    mpo_index.degeneracies[channel],
                    mps_index.degeneracies[ket],
                )
            )
    shapes = tuple(sorted(set(true_shapes)))
    shape_positions = {shape: position for position, shape in enumerate(shapes)}
    counts = [0] * len(shapes)
    positions = []
    for shape in true_shapes:
        bucket = shape_positions[shape]
        positions.append((bucket, counts[bucket]))
        counts[bucket] += 1
    return EnvironmentBucketPlan(
        shapes, tuple(positions), tuple(counts), tuple(true_shapes), tuple(offsets)
    )


def pack(buffers: BlockBuffers) -> Array:
    """
    Pack a fixed tuple of legal blocks into one Krylov vector.

    :param buffers: Legal block buffers to flatten and concatenate.
    :type buffers: BlockBuffers
    :return: Flattened Krylov vector.
    :rtype: Array
    """
    return jnp.concatenate(tuple(jnp.reshape(buffer, (-1,)) for buffer in buffers))


def unpack(template: BlockBuffers, vector: Array) -> BlockBuffers:
    """
    Unpack a Krylov vector using the static block shapes in ``template``.

    :param template: Block buffers providing the output shapes.
    :type template: BlockBuffers
    :param vector: Flattened Krylov vector.
    :type vector: Array
    :return: Blocks reshaped according to ``template``.
    :rtype: BlockBuffers
    """
    offset = 0
    result = []
    for buffer in template:
        size = buffer.size
        result.append(jnp.reshape(vector[offset : offset + size], buffer.shape))
        offset += size
    return tuple(result)


def apply_left_transfer(
    spec: MPSSpec, site: int, transfer: BlockBuffers, buffers: BlockBuffers
) -> BlockBuffers:
    """
    Absorb a block-diagonal transfer into the left virtual leg of one site.

    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Site whose left virtual leg is updated.
    :type site: int
    :param transfer: Block-diagonal transfer matrices.
    :type transfer: BlockBuffers
    :param buffers: Site block buffers.
    :type buffers: BlockBuffers
    :return: Updated site block buffers.
    :rtype: BlockBuffers
    """
    if spec.site_layouts is None:
        raise ValueError("symmetric MPS specification is incomplete")
    layout = spec.site_layouts[site]
    return tuple(
        jnp.einsum("ab,bpc->apc", transfer[coordinate[0]], block)
        for block, coordinate in zip(buffers, layout.block_coordinates)
    )


def apply_right_transfer(
    spec: MPSSpec, site: int, transfer: BlockBuffers, buffers: BlockBuffers
) -> BlockBuffers:
    """
    Absorb a block-diagonal transfer into the right virtual leg of one site.

    :param spec: Static symmetric MPS layout.
    :type spec: MPSSpec
    :param site: Site whose right virtual leg is updated.
    :type site: int
    :param transfer: Block-diagonal transfer matrices.
    :type transfer: BlockBuffers
    :param buffers: Site block buffers.
    :type buffers: BlockBuffers
    :return: Updated site block buffers.
    :rtype: BlockBuffers
    """
    if spec.site_layouts is None:
        raise ValueError("symmetric MPS specification is incomplete")
    layout = spec.site_layouts[site]
    return tuple(
        jnp.einsum("lpa,ab->lpb", block, transfer[coordinate[2]])
        for block, coordinate in zip(buffers, layout.block_coordinates)
    )


def _empty_mpo_environment(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, bond: int, dtype: Any
) -> LegalEnvironment:
    """
    Allocate only charge-conserving bra-MPO-ket environment blocks.
    """
    if mps_spec.bond_indices is None or mpo_spec.bond_indices is None:
        raise ValueError("symmetric MPS/MPO specification is incomplete")
    mps_index = mps_spec.bond_indices[bond]
    mpo_index = mpo_spec.bond_indices[bond]
    blocks = tuple(
        jnp.zeros(
            (
                mps_index.degeneracies[bra],
                mpo_index.degeneracies[channel],
                mps_index.degeneracies[ket],
            ),
            dtype=dtype,
        )
        for bra, channel, ket in _legal_environment_coordinates(
            mps_spec, mpo_spec, bond
        )
    )
    return LegalEnvironment(mps_spec, mpo_spec, bond, blocks)


def mpo_left_boundary(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, dtype: Any
) -> LegalEnvironment:
    """
    Return a fully block-sparse left MPS/MPO environment boundary.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param dtype: Dtype of the boundary block.
    :type dtype: Any
    :return: Left-boundary legal environment.
    :rtype: LegalEnvironment
    """
    result = _empty_mpo_environment(mps_spec, mpo_spec, 0, dtype)
    return result.replace(0, jnp.ones((1, 1, 1), dtype=dtype))


def mpo_right_boundary(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, dtype: Any
) -> LegalEnvironment:
    """
    Return a fully block-sparse right MPS/MPO environment boundary.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param dtype: Dtype of the boundary block.
    :type dtype: Any
    :return: Right-boundary legal environment.
    :rtype: LegalEnvironment
    """
    result = _empty_mpo_environment(mps_spec, mpo_spec, mps_spec.nsites, dtype)
    return result.replace(0, jnp.ones((1, 1, 1), dtype=dtype))


def environment_metadata(mps_spec: MPSSpec, mpo_spec: MPOSpec, bond: int) -> Metadata:
    """
    Return legal environment coordinates and their block shapes.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param bond: MPS cut whose metadata is requested.
    :type bond: int
    :return: Legal coordinates and corresponding block shapes.
    :rtype: Metadata
    """
    assert mps_spec.bond_indices is not None and mpo_spec.bond_indices is not None
    coordinates = _legal_environment_coordinates(mps_spec, mpo_spec, bond)
    dims = mps_spec.bond_indices[bond].degeneracies
    channels = mpo_spec.bond_indices[bond].degeneracies
    return coordinates, tuple(
        (dims[a], channels[b], dims[c]) for a, b, c in coordinates
    )


@lru_cache(maxsize=1024)
def _prepare_mpo_contractions(
    mps: Metadata, mpo: Metadata, left: Metadata, right: Metadata
) -> Any:
    def sequence(equations: Tuple[str, ...], operands: Tuple[Metadata, ...]) -> Any:
        result = []
        value = operands[0]
        for equation, operand in zip(equations, operands[1:]):
            plan = prepare_contraction(equation, value, operand)
            result.append(plan)
            value = plan.output
        return tuple(result)

    return (
        sequence(
            ("abc,cqf->abqf", "abqf,bpqe->apef", "apef,apd->def"), (left, mps, mpo, mps)
        ),
        sequence(
            ("def,cqf->decq", "decq,bpqe->dcbp", "dcbp,apd->abc"),
            (right, mps, mpo, mps),
        ),
        sequence(
            ("abc,ceh->abeh", "abeh,bdef->adfh", "adfh,gfh->adg"),
            (left, mps, mpo, right),
        ),
    )


def prepare_mpo_site(mps_spec: MPSSpec, mpo_spec: MPOSpec, site: int) -> Any:
    """
    Prepare charge joins and contraction routes without numerical buffers.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param site: Site whose contraction routes should be prepared.
    :type site: int
    :return: Static metadata and contraction plans for the site.
    :rtype: Any
    """
    assert mps_spec.site_layouts is not None and mpo_spec.site_layouts is not None
    mps, mpo = mps_spec.site_layouts[site], mpo_spec.site_layouts[site]
    metadata = (
        (mps.block_coordinates, mps.block_shapes),
        (mpo.block_coordinates, mpo.block_shapes),
        environment_metadata(mps_spec, mpo_spec, site),
        environment_metadata(mps_spec, mpo_spec, site + 1),
    )
    return metadata, _prepare_mpo_contractions(*metadata)


def mpo_update_left(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    site: int,
    environment: LegalEnvironment,
    mps_buffers: BlockBuffers,
    mpo_buffers: BlockBuffers,
) -> LegalEnvironment:
    """
    Update the left environment with three reusable block contractions.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param site: Site absorbed into the environment.
    :type site: int
    :param environment: Left environment before the site.
    :type environment: LegalEnvironment
    :param mps_buffers: MPS blocks at the site.
    :type mps_buffers: BlockBuffers
    :param mpo_buffers: MPO blocks at the site.
    :type mpo_buffers: BlockBuffers
    :return: Updated left environment at the next cut.
    :rtype: LegalEnvironment
    """
    metadata, plans = prepare_mpo_site(mps_spec, mpo_spec, site)
    first, second, third = plans[0]
    value = first(environment.blocks, mps_buffers)
    value = second(value, mpo_buffers)
    value = third(value, tuple(jnp.conj(block) for block in mps_buffers))
    blocks = align_blocks(
        third.output,
        value,
        metadata[3],
        jnp.result_type(mps_buffers[0], mpo_buffers[0]),
    )
    return LegalEnvironment(mps_spec, mpo_spec, site + 1, blocks)


def mpo_update_right(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    site: int,
    environment: LegalEnvironment,
    mps_buffers: BlockBuffers,
    mpo_buffers: BlockBuffers,
) -> LegalEnvironment:
    """
    Update the right environment with three reusable block contractions.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param site: Site absorbed into the environment.
    :type site: int
    :param environment: Right environment after the site.
    :type environment: LegalEnvironment
    :param mps_buffers: MPS blocks at the site.
    :type mps_buffers: BlockBuffers
    :param mpo_buffers: MPO blocks at the site.
    :type mpo_buffers: BlockBuffers
    :return: Updated right environment at the previous cut.
    :rtype: LegalEnvironment
    """
    metadata, plans = prepare_mpo_site(mps_spec, mpo_spec, site)
    first, second, third = plans[1]
    value = first(environment.blocks, mps_buffers)
    value = second(value, mpo_buffers)
    value = third(value, tuple(jnp.conj(block) for block in mps_buffers))
    blocks = align_blocks(
        third.output,
        value,
        metadata[2],
        jnp.result_type(mps_buffers[0], mpo_buffers[0]),
    )
    return LegalEnvironment(mps_spec, mpo_spec, site, blocks)


def mpo_build_right_environments(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    mps_buffers: Tuple[BlockBuffers, ...],
    mpo_buffers: Tuple[BlockBuffers, ...],
) -> Tuple[LegalEnvironment, ...]:
    """
    Build right environments by contracting only legal MPS/MPO blocks.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param mps_buffers: MPS site block buffers.
    :type mps_buffers: Tuple[BlockBuffers, ...]
    :param mpo_buffers: MPO site block buffers.
    :type mpo_buffers: Tuple[BlockBuffers, ...]
    :return: Right environments indexed by MPS cut.
    :rtype: Tuple[LegalEnvironment, ...]
    """
    dtype = jnp.result_type(mps_buffers[0][0], mpo_buffers[0][0])
    right = [mpo_right_boundary(mps_spec, mpo_spec, dtype)]
    for site in range(mps_spec.nsites - 1, -1, -1):
        right.append(
            mpo_update_right(
                mps_spec,
                mpo_spec,
                site,
                right[-1],
                mps_buffers[site],
                mpo_buffers[site],
            )
        )
    return tuple(reversed(right))


def mpo_build_environments(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    mps_buffers: Tuple[BlockBuffers, ...],
    mpo_buffers: Tuple[BlockBuffers, ...],
) -> EnvironmentState:
    """
    Build all block-sparse environments as a JAX-PyTree carry value.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param mps_buffers: MPS site block buffers.
    :type mps_buffers: Tuple[BlockBuffers, ...]
    :param mpo_buffers: MPO site block buffers.
    :type mpo_buffers: Tuple[BlockBuffers, ...]
    :return: Complete left and right environment state.
    :rtype: EnvironmentState
    """
    dtype = jnp.result_type(mps_buffers[0][0], mpo_buffers[0][0])
    left = [mpo_left_boundary(mps_spec, mpo_spec, dtype)]
    for site in range(mps_spec.nsites):
        left.append(
            mpo_update_left(
                mps_spec,
                mpo_spec,
                site,
                left[-1],
                mps_buffers[site],
                mpo_buffers[site],
            )
        )
    return EnvironmentState(
        tuple(left),
        mpo_build_right_environments(mps_spec, mpo_spec, mps_buffers, mpo_buffers),
    )


def mpo_local_matvec(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    site: int,
    left: LegalEnvironment,
    mpo_buffers: BlockBuffers,
    right: LegalEnvironment,
    vector: BlockBuffers,
) -> BlockBuffers:
    """
    Apply L, W, R successively, sharing intermediates across MPO paths.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param site: Site of the local effective operator.
    :type site: int
    :param left: Left effective environment.
    :type left: LegalEnvironment
    :param mpo_buffers: Local MPO blocks.
    :type mpo_buffers: BlockBuffers
    :param right: Right effective environment.
    :type right: LegalEnvironment
    :param vector: Local MPS blocks to which the operator is applied.
    :type vector: BlockBuffers
    :return: Effective-Hamiltonian action on the local blocks.
    :rtype: BlockBuffers
    """
    metadata, plans = prepare_mpo_site(mps_spec, mpo_spec, site)
    first, second, third = plans[2]
    value = first(left.blocks, vector)
    value = second(value, mpo_buffers)
    value = third(value, right.blocks)
    return align_blocks(
        third.output, value, metadata[0], jnp.result_type(mpo_buffers[0], vector[0])
    )


def mpo_bond_matvec(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    bond: int,
    left: LegalEnvironment,
    right: LegalEnvironment,
    vector: BlockBuffers,
) -> BlockBuffers:
    """
    Apply a zero-site effective Hamiltonian using two block contractions.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param bond: MPS bond carrying the zero-site center.
    :type bond: int
    :param left: Left effective environment.
    :type left: LegalEnvironment
    :param right: Right effective environment.
    :type right: LegalEnvironment
    :param vector: Zero-site center blocks.
    :type vector: BlockBuffers
    :return: Effective-Hamiltonian action on the bond blocks.
    :rtype: BlockBuffers
    """
    assert mps_spec.bond_indices is not None
    dimensions = mps_spec.bond_indices[bond].degeneracies
    target = (
        tuple((i, i) for i in range(len(dimensions))),
        tuple((d, d) for d in dimensions),
    )
    environment = environment_metadata(mps_spec, mpo_spec, bond)
    first = prepare_contraction("abc,ce->abe", environment, target)
    second = prepare_contraction("abe,dbe->ad", first.output, environment)
    return align_blocks(
        second.output,
        second(first(left.blocks, vector), right.blocks),
        target,
        vector[0].dtype,
    )


def mpo_expectation(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    mps_buffers: Tuple[BlockBuffers, ...],
    mpo_buffers: Tuple[BlockBuffers, ...],
) -> Array:
    """
    Contract ``<psi|H|psi>`` entirely in MPS/MPO block buffers.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param mps_buffers: MPS site block buffers.
    :type mps_buffers: Tuple[BlockBuffers, ...]
    :param mpo_buffers: MPO site block buffers.
    :type mpo_buffers: Tuple[BlockBuffers, ...]
    :return: The scalar expectation value.
    :rtype: Array
    """
    environment = mpo_left_boundary(
        mps_spec, mpo_spec, jnp.result_type(mps_buffers[0][0], mpo_buffers[0][0])
    )
    for site in range(mps_spec.nsites):
        environment = mpo_update_left(
            mps_spec,
            mpo_spec,
            site,
            environment,
            mps_buffers[site],
            mpo_buffers[site],
        )
    return environment[0][0, 0, 0]


def mpo_second_moment(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    mps_buffers: Tuple[BlockBuffers, ...],
    mpo_buffers: Tuple[BlockBuffers, ...],
) -> Array:
    """
    Contract H squared through reachable blocks, without forbidden zero blocks.

    :param mps_spec: Static symmetric MPS layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static symmetric MPO layout.
    :type mpo_spec: MPOSpec
    :param mps_buffers: MPS site block buffers.
    :type mps_buffers: Tuple[BlockBuffers, ...]
    :param mpo_buffers: MPO site block buffers.
    :type mpo_buffers: Tuple[BlockBuffers, ...]
    :return: The unnormalized second moment ``<psi|H^2|psi>``.
    :rtype: Array
    """
    assert mps_spec.site_layouts is not None and mpo_spec.site_layouts is not None
    dtype = jnp.result_type(mps_buffers[0][0], mpo_buffers[0][0])
    metadata: Metadata = (((0, 0, 0, 0),), ((1, 1, 1, 1),))
    values = (jnp.ones((1, 1, 1, 1), dtype=dtype),)
    for site in range(mps_spec.nsites):
        mps = mps_spec.site_layouts[site]
        mpo = mpo_spec.site_layouts[site]
        mps_metadata = (mps.block_coordinates, mps.block_shapes)
        mpo_metadata = (mpo.block_coordinates, mpo.block_shapes)
        operations = (
            ("abcd,drh->abcrh", mps_metadata, mps_buffers[site]),
            ("abcrh,cqrg->abqhg", mpo_metadata, mpo_buffers[site]),
            ("abqhg,bpqf->aphgf", mpo_metadata, mpo_buffers[site]),
            (
                "aphgf,ape->efgh",
                mps_metadata,
                tuple(jnp.conj(a) for a in mps_buffers[site]),
            ),
        )
        for equation, operand_metadata, operand in operations:
            plan = prepare_contraction(equation, metadata, operand_metadata)
            values = plan(values, operand)
            metadata = plan.output
            if not values:
                return jnp.zeros((), dtype=dtype)
    return values[0][0, 0, 0, 0]
