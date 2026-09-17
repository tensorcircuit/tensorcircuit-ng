"""
Fixed-shape one-site DMRG for dense and Abelian block-buffer MPS states.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp

from ..krylov import lowest_eigenvector_hermitian
from ..layout import (
    MPOSpec,
    MPSSpec,
    block_mps_spec,
    block_mpo_spec,
    validate_mps_mpo_specs,
)
from ..mpo import MPOState
from ..mps import (
    BlockBuffers,
    MPSState,
    pack_block_buffers,
    unpack_block_buffers,
)
from ..symmetric_environment import (
    LegalEnvironment,
    apply_left_transfer,
    apply_right_transfer,
    mpo_left_boundary,
    mpo_local_matvec,
    mpo_update_left,
    mpo_update_right,
    pack as pack_symmetric_blocks,
    unpack as unpack_symmetric_blocks,
)
from ..symmetric_linalg import factor_left_blocks, factor_right_blocks
from ..sweep import (
    SiteGroup,
    build_right,
    empty_environments,
    environment_access,
    prepare_schedule,
    read_environment,
    site_access,
)

Array = Any


@dataclass(frozen=True)
class DMRGOptions:
    """
    Static one-site DMRG solver settings; evaluate variance separately.

    :ivar krylov_dim: Maximum dimension of each local Krylov subspace.
    :ivar restarts: Number of local Krylov restarts at each site.
    :ivar reorthogonalize: Whether to reorthogonalize the Krylov basis.
    """

    krylov_dim: int = 24
    restarts: int = 1
    reorthogonalize: bool = True

    def __post_init__(self) -> None:
        if self.krylov_dim < 1 or self.restarts < 1:
            raise ValueError("krylov_dim and restarts must be positive")


@dataclass(frozen=True)
class DMRGPlan:
    """
    Host-side static data captured by a one-site DMRG sweep closure.

    :ivar mps_spec: Static MPS physical, bond, and symmetry layout.
    :ivar mpo_spec: Static MPO physical, bond, and symmetry layout.
    :ivar options: Static local eigensolver settings.
    """

    mps_spec: MPSSpec
    mpo_spec: MPOSpec
    options: DMRGOptions


def prepare_dmrg(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, options: DMRGOptions
) -> DMRGPlan:
    """
    Create a fixed-shape one-site DMRG plan.

    :param mps_spec: Static MPS physical, bond, and symmetry layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO physical, bond, and symmetry layout.
    :type mpo_spec: MPOSpec
    :param options: Static local eigensolver settings.
    :type options: DMRGOptions
    :return: The validated host-side DMRG plan.
    :rtype: DMRGPlan
    """
    validate_mps_mpo_specs(mps_spec, mpo_spec)
    return DMRGPlan(mps_spec, mpo_spec, options)


def _lowest_block_mpo_symmetric_blocks(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    site: int,
    left: LegalEnvironment,
    mpo_buffers: Tuple[Array, ...],
    right: LegalEnvironment,
    buffers: Tuple[Array, ...],
    options: DMRGOptions,
) -> Tuple[Tuple[Array, ...], Array]:
    """
    Minimize a local effective Hamiltonian using native MPO block buffers.

    :param mps_spec: Static MPS layout used by the local contraction.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO layout used by the local contraction.
    :type mpo_spec: MPOSpec
    :param site: Index of the local MPS tensor.
    :type site: int
    :param left: Left effective environment at ``site``.
    :type left: LegalEnvironment
    :param mpo_buffers: Local MPO block buffers.
    :type mpo_buffers: Tuple[Array, ...]
    :param right: Right effective environment at ``site``.
    :type right: LegalEnvironment
    :param buffers: Local MPS block buffers used as the initial vector.
    :type buffers: Tuple[Array, ...]
    :param options: Local eigensolver settings.
    :type options: DMRGOptions
    :return: The lowest Ritz vector as block buffers and its Ritz value.
    :rtype: Tuple[Tuple[Array, ...], Array]
    """
    vector = pack_symmetric_blocks(buffers)

    def matvec(value: Array) -> Array:
        return pack_symmetric_blocks(
            mpo_local_matvec(
                mps_spec,
                mpo_spec,
                site,
                left,
                mpo_buffers,
                right,
                unpack_symmetric_blocks(buffers, value),
            )
        )

    candidate = vector
    eigenvalue = jnp.asarray(0, dtype=jnp.real(vector).dtype)
    for _ in range(options.restarts):
        candidate, eigenvalue = lowest_eigenvector_hermitian(
            matvec,
            candidate,
            options.krylov_dim,
            options.reorthogonalize,
            return_report=False,
        )
    return unpack_symmetric_blocks(buffers, candidate), eigenvalue


def _block_mpo_center_energy(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    right: LegalEnvironment,
    mpo_buffers: Tuple[Array, ...],
    buffers: Tuple[Array, ...],
) -> Array:
    """
    Return normalized energy from a right-canonical site-zero center.

    :param mps_spec: Static MPS layout used by the contraction.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO layout used by the contraction.
    :type mpo_spec: MPOSpec
    :param right: Right effective environment after site zero.
    :type right: LegalEnvironment
    :param mpo_buffers: MPO block buffers.
    :type mpo_buffers: Tuple[Array, ...]
    :param buffers: Site-zero MPS block buffers.
    :type buffers: Tuple[Array, ...]
    :return: The normalized real energy.
    :rtype: Array
    """
    effective = mpo_local_matvec(
        mps_spec,
        mpo_spec,
        0,
        mpo_left_boundary(mps_spec, mpo_spec, buffers[0].dtype),
        mpo_buffers,
        right,
        buffers,
    )
    numerator = sum(
        jnp.vdot(buffer, value) for buffer, value in zip(buffers, effective)
    )
    denominator = jnp.real(sum(jnp.vdot(buffer, buffer) for buffer in buffers))
    return jnp.real(numerator) / jnp.where(denominator == 0, 1, denominator)


def make_dmrg_sweep_from_plan(
    plan: DMRGPlan,
) -> Callable[[MPSState, MPOState], Tuple[MPSState, Array]]:
    """
    Build a one-site DMRG sweep from a prepared plan.

    :param plan: Host-side static DMRG plan.
    :type plan: DMRGPlan
    :return: A callable accepting an MPS and an MPO and returning the updated
        MPS with its energy.
    :rtype: Callable[[MPSState, MPOState], Tuple[MPSState, Array]]
    """
    spec, operator = block_mps_spec(plan.mps_spec), block_mpo_spec(plan.mpo_spec)
    dense = plan.mps_spec.symmetry is None
    schedule = prepare_schedule(spec, operator)
    initialization = prepare_schedule(spec, operator, width=1)
    first_access = site_access(spec, (0,))
    first_mpo = site_access(operator, (0,))
    right_access = environment_access(spec, operator, (1,))

    def forward(group: SiteGroup, row: Array, carry: Any, context: Any) -> Any:
        buffers, left = carry
        mpo, right = context
        site = group.site
        local_left = read_environment(group, 0, row, left, spec, operator)
        local_mpo = group.mpo[0].read(mpo, row)
        center, _ = _lowest_block_mpo_symmetric_blocks(
            spec,
            operator,
            site,
            local_left,
            local_mpo,
            read_environment(group, 1, row, right, spec, operator),
            group.mps[0].read(buffers, row),
            plan.options,
        )
        workspace: list[list[Array]] = [[] for _ in range(site + 1)]
        workspace[site] = list(center)
        transfer = factor_left_blocks(workspace, spec, site)
        canonical = tuple(workspace[site])
        next_left = mpo_update_left(
            spec, operator, site, local_left, canonical, local_mpo
        )
        following = apply_left_transfer(
            spec, site + 1, transfer, group.mps[1].read(buffers, row)
        )
        buffers = group.mps[0].write(buffers, row, canonical)
        buffers = group.mps[1].write(buffers, row, following)
        return buffers, group.environments[1].write(left, row, next_left.blocks)

    def backward(group: SiteGroup, row: Array, carry: Any, context: Any) -> Any:
        buffers, right = carry
        mpo, left = context
        site = group.site + 1
        local_right = read_environment(group, 2, row, right, spec, operator)
        local_mpo = group.mpo[1].read(mpo, row)
        center, _ = _lowest_block_mpo_symmetric_blocks(
            spec,
            operator,
            site,
            read_environment(group, 1, row, left, spec, operator),
            local_mpo,
            local_right,
            group.mps[1].read(buffers, row),
            plan.options,
        )
        workspace: list[list[Array]] = [[] for _ in range(site + 1)]
        workspace[site] = list(center)
        transfer = factor_right_blocks(workspace, spec, site)
        canonical = tuple(workspace[site])
        next_right = mpo_update_right(
            spec, operator, site, local_right, canonical, local_mpo
        )
        previous = apply_right_transfer(
            spec, site - 1, transfer, group.mps[0].read(buffers, row)
        )
        buffers = group.mps[1].write(buffers, row, canonical)
        buffers = group.mps[0].write(buffers, row, previous)
        return buffers, group.environments[1].write(right, row, next_right.blocks)

    def sweep(mps: MPSState, mpo: MPOState) -> Tuple[MPSState, Array]:
        if mps.spec != plan.mps_spec or mpo.spec != plan.mpo_spec:
            raise ValueError("DMRG state or MPO specification differs from the plan")
        buffers = (
            pack_block_buffers(spec, mps.buffers) if dense else mps.buffers.buckets
        )
        operators = (
            pack_block_buffers(operator, mpo.buffers) if dense else mpo.buffers.buckets
        )
        dtype = jnp.result_type(buffers[0], operators[0])
        right = build_right(initialization, spec, operator, buffers, operators)
        left = empty_environments(spec, operator, dtype, left=True)
        buffers, left = schedule.run(forward, (buffers, left), (operators, right))
        right = empty_environments(spec, operator, dtype, left=False)
        buffers, right = schedule.run(
            backward, (buffers, right), (operators, left), reverse=True
        )
        local_right = LegalEnvironment(spec, operator, 1, right_access.read(right, 0))
        operator_first = first_mpo.read(operators, 0)
        center, _ = _lowest_block_mpo_symmetric_blocks(
            spec,
            operator,
            0,
            mpo_left_boundary(spec, operator, dtype),
            operator_first,
            local_right,
            first_access.read(buffers, 0),
            plan.options,
        )
        buffers = first_access.write(buffers, 0, center)
        energy = _block_mpo_center_energy(
            spec, operator, local_right, operator_first, center
        )
        state_buffers = (
            unpack_block_buffers(spec, buffers)
            if dense
            else BlockBuffers(spec, buffers)
        )
        return MPSState(plan.mps_spec, state_buffers), energy

    return sweep


def make_dmrg_sweep(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    options: Optional[DMRGOptions] = None,
) -> Callable[[MPSState, MPOState], Tuple[MPSState, Array]]:
    """
    Build a one-site DMRG sweep directly from MPS and MPO specifications.

    :param mps_spec: Static MPS physical, bond, and symmetry layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO physical, bond, and symmetry layout.
    :type mpo_spec: MPOSpec
    :param options: Static local eigensolver settings, or ``None`` for defaults.
    :type options: Optional[DMRGOptions]
    :return: A callable accepting an MPS and an MPO and returning the updated
        MPS with its energy.
    :rtype: Callable[[MPSState, MPOState], Tuple[MPSState, Array]]
    """
    if options is None:
        options = DMRGOptions()
    return make_dmrg_sweep_from_plan(prepare_dmrg(mps_spec, mpo_spec, options))
