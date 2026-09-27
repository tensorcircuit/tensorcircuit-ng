"""
Fixed-shape one-site TDVP for dense and Abelian block-buffer MPS states.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp

from ..krylov import expm_action_hermitian
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
from ..symmetric_linalg import factor_left_blocks, factor_right_blocks
from ..symmetric_environment import (
    LegalEnvironment,
    apply_left_transfer,
    apply_right_transfer,
    mpo_bond_matvec,
    mpo_local_matvec,
    mpo_update_left,
    mpo_update_right,
    pack as pack_symmetric_blocks,
    unpack as unpack_symmetric_blocks,
)

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
class TDVPOptions:
    """
    Static one-site TDVP and local Krylov settings.

    :ivar krylov_dim: Maximum dimension of each local Krylov subspace.
    :ivar reorthogonalize: Whether to reorthogonalize the Krylov basis.
    """

    krylov_dim: int = 16
    reorthogonalize: bool = True

    def __post_init__(self) -> None:
        if self.krylov_dim < 1:
            raise ValueError("krylov_dim must be positive")


@dataclass(frozen=True)
class TDVPPlan:
    """
    Host-side static data captured by a one-site TDVP step closure.

    :ivar mps_spec: Static MPS physical, bond, and symmetry layout.
    :ivar mpo_spec: Static MPO physical, bond, and symmetry layout.
    :ivar options: Static local Krylov solver settings.
    """

    mps_spec: MPSSpec
    mpo_spec: MPOSpec
    options: TDVPOptions


def prepare_tdvp(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, options: TDVPOptions
) -> TDVPPlan:
    """
    Create a fixed-shape one-site TDVP plan.

    :param mps_spec: Static MPS physical, bond, and symmetry layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO physical, bond, and symmetry layout.
    :type mpo_spec: MPOSpec
    :param options: Static local Krylov solver settings.
    :type options: TDVPOptions
    :return: The validated host-side TDVP plan.
    :rtype: TDVPPlan
    """
    validate_mps_mpo_specs(mps_spec, mpo_spec)
    return TDVPPlan(mps_spec, mpo_spec, options)


def _block_mpo_local_exponential(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    site: int,
    left: LegalEnvironment,
    mpo_buffers: Tuple[Array, ...],
    right: LegalEnvironment,
    buffers: Tuple[Array, ...],
    dt: Array,
    options: TDVPOptions,
) -> Tuple[Tuple[Array, ...], dict[str, Array]]:
    """
    Exponentiate a local effective Hamiltonian without dense MPO channels.

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
    :param buffers: Local MPS block buffers to evolve.
    :type buffers: Tuple[Array, ...]
    :param dt: Evolution time assigned to this local exponential.
    :type dt: Array
    :param options: Local Krylov solver settings.
    :type options: TDVPOptions
    :return: Evolved block buffers and the Krylov diagnostic report.
    :rtype: Tuple[Tuple[Array, ...], dict[str, Array]]
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

    evolved, report = expm_action_hermitian(
        matvec, vector, -1j * dt, options.krylov_dim, options.reorthogonalize
    )
    return unpack_symmetric_blocks(buffers, evolved), report


def _block_mpo_bond_exponential(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    bond: int,
    left: LegalEnvironment,
    right: LegalEnvironment,
    buffers: Tuple[Array, ...],
    dt: Array,
    options: TDVPOptions,
) -> Tuple[Tuple[Array, ...], dict[str, Array]]:
    """
    Exponentiate a block-MPO zero-site effective Hamiltonian.

    :param mps_spec: Static MPS layout used by the bond contraction.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO layout used by the bond contraction.
    :type mpo_spec: MPOSpec
    :param bond: Index of the MPS bond carrying the zero-site center.
    :type bond: int
    :param left: Left effective environment adjacent to ``bond``.
    :type left: LegalEnvironment
    :param right: Right effective environment adjacent to ``bond``.
    :type right: LegalEnvironment
    :param buffers: Zero-site center block buffers to evolve.
    :type buffers: Tuple[Array, ...]
    :param dt: Evolution time assigned to this bond exponential.
    :type dt: Array
    :param options: Local Krylov solver settings.
    :type options: TDVPOptions
    :return: Evolved bond buffers and the Krylov diagnostic report.
    :rtype: Tuple[Tuple[Array, ...], dict[str, Array]]
    """
    vector = pack_symmetric_blocks(buffers)

    def matvec(value: Array) -> Array:
        return pack_symmetric_blocks(
            mpo_bond_matvec(
                mps_spec,
                mpo_spec,
                bond,
                left,
                right,
                unpack_symmetric_blocks(buffers, value),
            )
        )

    evolved, report = expm_action_hermitian(
        matvec, vector, 1j * dt, options.krylov_dim, options.reorthogonalize
    )
    return unpack_symmetric_blocks(buffers, evolved), report


def make_tdvp_step_from_plan(
    plan: TDVPPlan,
) -> Callable[[MPSState, MPOState, Array], MPSState]:
    """
    Build a projector-splitting one-site TDVP step from a prepared plan.

    :param plan: Host-side static TDVP plan.
    :type plan: TDVPPlan
    :return: A callable accepting an MPS, an MPO, and a time step.
    :rtype: Callable[[MPSState, MPOState, Array], MPSState]
    """
    spec, operator = block_mps_spec(plan.mps_spec), block_mpo_spec(plan.mpo_spec)
    dense = plan.mps_spec.symmetry is None
    schedule = prepare_schedule(spec, operator)
    initialization = prepare_schedule(spec, operator, width=1)
    last = spec.nsites - 1
    last_access = site_access(spec, (last,))
    last_mpo = site_access(operator, (last,))
    left_access = environment_access(spec, operator, (last,))
    right_access = environment_access(spec, operator, (last + 1,))

    def forward(group: SiteGroup, row: Array, carry: Any, context: Any) -> Any:
        buffers, left = carry
        mpo, right, half_dt = context
        site = group.site
        local_left = read_environment(group, 0, row, left, spec, operator)
        local_right = read_environment(group, 1, row, right, spec, operator)
        local_mpo = group.mpo[0].read(mpo, row)
        center, _ = _block_mpo_local_exponential(
            spec,
            operator,
            site,
            local_left,
            local_mpo,
            local_right,
            group.mps[0].read(buffers, row),
            half_dt,
            plan.options,
        )
        workspace: list[list[Array]] = [[] for _ in range(site + 1)]
        workspace[site] = list(center)
        transfer = factor_left_blocks(workspace, spec, site)
        canonical = tuple(workspace[site])
        next_left = mpo_update_left(
            spec, operator, site, local_left, canonical, local_mpo
        )
        transfer, _ = _block_mpo_bond_exponential(
            spec,
            operator,
            site + 1,
            next_left,
            local_right,
            transfer,
            half_dt,
            plan.options,
        )
        following = apply_left_transfer(
            spec, site + 1, transfer, group.mps[1].read(buffers, row)
        )
        buffers = group.mps[0].write(buffers, row, canonical)
        buffers = group.mps[1].write(buffers, row, following)
        return buffers, group.environments[1].write(left, row, next_left.blocks)

    def backward(group: SiteGroup, row: Array, carry: Any, context: Any) -> Any:
        buffers, right = carry
        mpo, left, half_dt = context
        site = group.site + 1
        local_right = read_environment(group, 2, row, right, spec, operator)
        workspace: list[list[Array]] = [[] for _ in range(site + 1)]
        workspace[site] = list(group.mps[1].read(buffers, row))
        transfer = factor_right_blocks(workspace, spec, site)
        canonical = tuple(workspace[site])
        next_right = mpo_update_right(
            spec,
            operator,
            site,
            local_right,
            canonical,
            group.mpo[1].read(mpo, row),
        )
        transfer, _ = _block_mpo_bond_exponential(
            spec,
            operator,
            site,
            read_environment(group, 1, row, left, spec, operator),
            next_right,
            transfer,
            half_dt,
            plan.options,
        )
        previous = apply_right_transfer(
            spec, site - 1, transfer, group.mps[0].read(buffers, row)
        )
        center, _ = _block_mpo_local_exponential(
            spec,
            operator,
            site - 1,
            read_environment(group, 0, row, left, spec, operator),
            group.mpo[0].read(mpo, row),
            next_right,
            previous,
            half_dt,
            plan.options,
        )
        buffers = group.mps[1].write(buffers, row, canonical)
        buffers = group.mps[0].write(buffers, row, center)
        return buffers, group.environments[1].write(right, row, next_right.blocks)

    def step(state: MPSState, mpo: MPOState, dt: Array) -> MPSState:
        if state.spec != plan.mps_spec or mpo.spec != plan.mpo_spec:
            raise ValueError("TDVP state or MPO specification differs from the plan")
        buffers = (
            pack_block_buffers(spec, state.buffers) if dense else state.buffers.buckets
        )
        operators = (
            pack_block_buffers(operator, mpo.buffers) if dense else mpo.buffers.buckets
        )
        dtype = jnp.result_type(buffers[0], operators[0])
        right = build_right(initialization, spec, operator, buffers, operators)
        left = empty_environments(spec, operator, dtype, left=True)
        buffers, left = schedule.run(
            forward, (buffers, left), (operators, right, dt / 2)
        )
        center, _ = _block_mpo_local_exponential(
            spec,
            operator,
            last,
            LegalEnvironment(spec, operator, last, left_access.read(left, 0)),
            last_mpo.read(operators, 0),
            LegalEnvironment(spec, operator, last + 1, right_access.read(right, 0)),
            last_access.read(buffers, 0),
            dt,
            plan.options,
        )
        buffers = last_access.write(buffers, 0, center)
        right = empty_environments(spec, operator, dtype, left=False)
        buffers, _ = schedule.run(
            backward, (buffers, right), (operators, left, dt / 2), reverse=True
        )
        return MPSState(
            plan.mps_spec,
            (
                unpack_block_buffers(spec, buffers)
                if dense
                else BlockBuffers(spec, buffers)
            ),
        )

    return step


def make_tdvp_step(
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    options: Optional[TDVPOptions] = None,
) -> Callable[[MPSState, MPOState, Array], MPSState]:
    """
    Build a one-site TDVP step directly from MPS and MPO specifications.

    :param mps_spec: Static MPS physical, bond, and symmetry layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO physical, bond, and symmetry layout.
    :type mpo_spec: MPOSpec
    :param options: Static local Krylov solver settings, or ``None`` for defaults.
    :type options: Optional[TDVPOptions]
    :return: A callable accepting an MPS, an MPO, and a time step.
    :rtype: Callable[[MPSState, MPOState, Array], MPSState]
    """
    if options is None:
        options = TDVPOptions()
    return make_tdvp_step_from_plan(prepare_tdvp(mps_spec, mpo_spec, options))
