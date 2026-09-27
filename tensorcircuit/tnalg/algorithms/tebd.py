"""
Fixed-shape dense TEBD with tensor-only runtime inputs and outputs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..layout import GateSpec, MPSSpec, block_mps_spec
from ..mps import (
    BlockBuffers,
    MPSState,
    pack_block_buffers,
    unpack_block_buffers,
)
from ..operator_strings import (
    canonical_site_operator_tables,
    physical_dims_from_indices,
    validate_coefficient_indices,
    validate_operator_input,
    validate_symmetric_operator_groups,
)
from ..symmetric_linalg import (
    apply_onsite_matrix,
    apply_two_site_matrix,
    shift_center_left,
    shift_center_right,
)
from ..sweep import prepare_program, prepare_schedule, run_program, site_access

Array = Any
GeneratorTree = Tuple[Tuple[Array, ...], Tuple[Array, ...]]


@dataclass(frozen=True)
class TEBDOptions:
    """
    Static choices for a fixed-shape TEBD step.

    :ivar order: Suzuki-Trotter order, either one or two.
    :ivar normalize: Whether to normalize the first MPS site after the step.
    """

    order: int = 2
    normalize: bool = False

    def __post_init__(self) -> None:
        if self.order not in (1, 2):
            raise ValueError("TEBD order must be 1 or 2")


@dataclass(frozen=True)
class TEBDPlan:
    """
    Host-side static schedule captured by a TEBD step closure.

    :ivar mps_spec: Static MPS physical, bond, and symmetry layout.
    :ivar gate_spec: Static physical dimensions of the compiled generators.
    :ivar options: Static TEBD splitting and normalization settings.
    """

    mps_spec: MPSSpec
    gate_spec: GateSpec
    options: TEBDOptions


def compile_tebd_gates(
    operator_codes: Array,
    physical_indices: Tuple[Any, ...],
    *,
    operator_table: Any = None,
    coefficient_indices: Any = None,
) -> Tuple[GateSpec, Callable[[Array], GeneratorTree]]:
    """
    Compile concrete Pauli/operator strings into a dynamic generator builder.

    ``operator_codes`` uses TensorCircuit's Pauli convention by default:
    ``0=I, 1=X, 2=Y, 3=Z``. It is inspected outside JIT to fix the nearest-
    neighbor schedule; ``weights`` remains the only dynamic builder input.

    :param operator_codes: Integer operator codes with shape
        ``(n_terms, n_sites)``.
    :type operator_codes: Array
    :param physical_indices: Local dimensions or explicit sector indices.
    :type physical_indices: Tuple[Any, ...]
    :param operator_table: Optional local operator table whose entry zero is the
        identity.
    :type operator_table: Any
    :param coefficient_indices: Optional term-to-coefficient group indices.
    :type coefficient_indices: Any
    :return: The static gate specification and a dynamic generator builder.
    :rtype: Tuple[GateSpec, Callable[[Array], GeneratorTree]]
    """
    physical_dims = physical_dims_from_indices(physical_indices)
    codes, table = validate_operator_input(
        operator_codes, physical_dims, operator_table
    )
    nterms, nsites = codes.shape
    coefficient_indices, ncoefficients = validate_coefficient_indices(
        coefficient_indices, nterms
    )
    site_tables = canonical_site_operator_tables(table, physical_indices)
    validate_symmetric_operator_groups(
        codes, coefficient_indices, site_tables, physical_indices
    )
    supports = []
    for row in codes:
        support = tuple(int(site) for site in np.flatnonzero(row))
        if len(support) > 2 or (len(support) == 2 and support[1] != support[0] + 1):
            raise ValueError("TEBD supports only onsite and nearest-neighbor strings")
        supports.append(support)
    table_arrays = tuple(jnp.asarray(table) for table in site_tables)
    code_array = jnp.asarray(codes, dtype=jnp.int32)
    coefficient_array = jnp.asarray(coefficient_indices, dtype=jnp.int32)
    onsite_terms = tuple(
        tuple(term for term, support in enumerate(supports) if support == (site,))
        for site in range(nsites)
    )
    bond_terms = tuple(
        tuple(
            term for term, support in enumerate(supports) if support == (site, site + 1)
        )
        for site in range(nsites - 1)
    )
    identity_terms = tuple(term for term, support in enumerate(supports) if not support)

    def build_gates(weights: Array) -> GeneratorTree:
        """
        Build onsite and nearest-neighbor generators from dynamic weights.

        :param weights: Dynamic coefficient values in group-index order.
        :type weights: Array
        :return: Onsite and bond generator tuples.
        :rtype: GeneratorTree
        """
        weights = jnp.asarray(weights)
        if weights.ndim != 1 or weights.shape[0] != ncoefficients:
            raise ValueError(f"weights must have shape ({ncoefficients},)")
        dtype = jnp.result_type(weights, table_arrays[0])
        local_operators = jnp.stack(
            tuple(
                jnp.take(table_array, code_array[:, site], axis=0)
                for site, table_array in enumerate(table_arrays)
            ),
            axis=1,
        )
        term_weights = jnp.take(weights, coefficient_array, axis=0)
        onsite = []
        for site, term_indices in enumerate(onsite_terms):
            generator = jnp.zeros(
                (physical_dims[site], physical_dims[site]), dtype=dtype
            )
            for term in term_indices:
                generator = generator + term_weights[term] * local_operators[term, site]
            if site == 0:
                for term in identity_terms:
                    generator = (
                        generator + term_weights[term] * local_operators[term, site]
                    )
            onsite.append(generator)
        bonds = []
        for site, term_indices in enumerate(bond_terms):
            dimension = physical_dims[site]
            generator = jnp.zeros(
                (dimension, dimension, dimension, dimension), dtype=dtype
            )
            for term in term_indices:
                product = jnp.einsum(
                    "ab,cd->acbd",
                    local_operators[term, site],
                    local_operators[term, site + 1],
                )
                generator = generator + term_weights[term] * product
            bonds.append(generator)
        return tuple(onsite), tuple(bonds)

    return GateSpec(physical_dims), build_gates


def prepare_tebd(
    mps_spec: MPSSpec, gate_spec: GateSpec, options: TEBDOptions
) -> TEBDPlan:
    """
    Create a host-side fixed TEBD schedule.

    :param mps_spec: Static MPS physical, bond, and symmetry layout.
    :type mps_spec: MPSSpec
    :param gate_spec: Static physical dimensions of the compiled generators.
    :type gate_spec: GateSpec
    :param options: Static TEBD splitting and normalization settings.
    :type options: TEBDOptions
    :return: The validated host-side TEBD plan.
    :rtype: TEBDPlan
    """
    if mps_spec.physical_dims != gate_spec.physical_dims:
        raise ValueError(
            "MPS and gate specifications have different physical dimensions"
        )
    return TEBDPlan(mps_spec, gate_spec, options)


def make_tebd_step_from_plan(
    plan: TEBDPlan,
) -> Callable[[MPSState, GeneratorTree, Array], MPSState]:
    """
    Build a complete fixed-shape TEBD step with shared spatial kernels.

    Singular values follow the sweep direction; the final parity layer runs
    right to left to return the canonical center to zero. Generators and time
    steps remain dynamic. Apply JIT to this complete step or an enclosing scan.

    :param plan: Host-side static TEBD plan.
    :type plan: TEBDPlan
    :return: A callable accepting an MPS, generators, and a time step.
    :rtype: Callable[[MPSState, GeneratorTree, Array], MPSState]
    """
    spec = block_mps_spec(plan.mps_spec)
    dense = plan.mps_spec.symmetry is None
    n = spec.nsites
    singles = prepare_schedule(spec, width=1)
    pairs = prepare_schedule(spec)
    onsite_shapes: dict[Any, list[int]] = {}
    bond_shapes: dict[Any, list[int]] = {}
    for site, dimension in enumerate(spec.physical_dims):
        onsite_shapes.setdefault((dimension,), []).append(site)
    for bond in range(n - 1):
        bond_shapes.setdefault(spec.physical_dims[bond : bond + 2], []).append(bond)
    onsite_groups = tuple(onsite_shapes.items())
    bond_groups = tuple(bond_shapes.items())
    onsite_locations = {
        site: (bucket, row)
        for bucket, (_, sites) in enumerate(onsite_groups)
        for row, site in enumerate(sites)
    }
    bond_locations = {
        site: (bucket, row)
        for bucket, (_, sites) in enumerate(bond_groups)
        for row, site in enumerate(sites)
    }
    branches: list[Callable[..., Any]] = []

    for group in singles.groups:
        bucket = onsite_locations[group.site][0]
        slots = np.asarray(
            [onsite_locations[site][1] for site in group.sites], dtype=np.int32
        )

        def onsite(
            operand: Any, group: Any = group, bucket: int = bucket, slots: Any = slots
        ) -> Any:
            buffers, onsite_gates, _, row, _ = operand
            work: list[list[Array]] = [[] for _ in range(group.site + 1)]
            work[group.site] = list(group.mps[0].read(buffers, row))
            apply_onsite_matrix(
                work, spec, group.site, onsite_gates[bucket][jnp.asarray(slots)[row]]
            )
            return group.mps[0].write(buffers, row, tuple(work[group.site]))

        branches.append(onsite)

    for kind in ("gate_right", "gate_left", "right", "left"):
        for group in pairs.groups:
            access = site_access(spec, group.sites, width=2)
            left_count = group.mps[0].count
            bucket = bond_locations[group.site][0]
            slots = np.asarray(
                [bond_locations[site][1] for site in group.sites], dtype=np.int32
            )

            def pair(
                operand: Any,
                group: Any = group,
                kind: str = kind,
                bucket: int = bucket,
                slots: Any = slots,
                access: Any = access,
                left_count: int = left_count,
            ) -> Any:
                buffers, _, bond_gates, row, coefficient = operand
                site = group.site
                work: list[list[Array]] = [[] for _ in range(site + 2)]
                blocks = access.read(buffers, row)
                work[site] = list(blocks[:left_count])
                work[site + 1] = list(blocks[left_count:])
                if kind in ("gate_right", "gate_left"):
                    apply_two_site_matrix(
                        work,
                        spec,
                        site,
                        bond_gates[bucket][coefficient, jnp.asarray(slots)[row]],
                        forward=kind == "gate_right",
                    )
                elif kind == "right":
                    shift_center_right(work, spec, site)
                else:
                    shift_center_left(work, spec, site + 1)
                return access.write(buffers, row, tuple(work[site] + work[site + 1]))

            branches.append(pair)

    ops = []
    center = 0
    pair_offset = len(singles.groups)
    pair_count = len(pairs.groups)

    def append_pair(kind: int, bond: int, coefficient: int = 0) -> None:
        code, row = pairs.indices[bond]
        ops.append((pair_offset + kind * pair_count + code, row, coefficient))

    def move(target: int) -> None:
        nonlocal center
        while center < target:
            append_pair(2, center)
            center += 1
        while center > target:
            append_pair(3, center - 1)
            center -= 1

    def onsites() -> None:
        for code, row in singles.indices:
            ops.append((code, row, 0))

    def bonds(parity: int, coefficient: int, *, reverse: bool = False) -> None:
        nonlocal center
        indices = range(parity, n - 1, 2)
        for bond in reversed(indices) if reverse else indices:
            move(bond + int(reverse))
            append_pair(int(reverse), bond, coefficient)
            center = bond if reverse else bond + 1

    onsites()
    if plan.options.order == 1:
        bonds(0, 1)
        bonds(1, 1, reverse=True)
    else:
        bonds(0, 0)
        bonds(1, 1)
        bonds(0, 0, reverse=True)
        onsites()
    move(0)
    indices = np.asarray(ops, dtype=np.int32)
    program = prepare_program(indices)
    first_access = site_access(spec, (0,))

    def exponentiate(
        generators: Any, groups: Any, dt: Array, coefficient: float
    ) -> Tuple[Array, ...]:
        result = []
        for dimensions, sites in groups:
            size = int(np.prod(dimensions))
            matrices = jnp.stack(
                tuple(jnp.reshape(generators[site], (size, size)) for site in sites)
            )
            gates = jax.vmap(jax.scipy.linalg.expm)(-1j * coefficient * dt * matrices)
            result.append(jnp.reshape(gates, (len(sites),) + dimensions * 2))
        return tuple(result)

    def step(state: MPSState, generators: GeneratorTree, dt: Array) -> MPSState:
        if state.spec != plan.mps_spec:
            raise ValueError("MPS specification differs from the TEBD plan")
        onsite_generators, bond_generators = generators
        if len(onsite_generators) != n or len(bond_generators) != n - 1:
            raise ValueError("generator PyTree does not match the TEBD plan")
        onsite_gates = exponentiate(
            onsite_generators,
            onsite_groups,
            dt,
            1.0 if plan.options.order == 1 else 0.5,
        )
        full_gates = exponentiate(bond_generators, bond_groups, dt, 1.0)
        half_gates = (
            exponentiate(bond_generators, bond_groups, dt, 0.5)
            if plan.options.order == 2
            else full_gates
        )
        bond_gates = tuple(
            jnp.stack((half, full)) for half, full in zip(half_gates, full_gates)
        )

        kernels = tuple(
            lambda buffers, record, branch=branch: branch(
                (buffers, onsite_gates, bond_gates, record[0], record[1])
            )
            for branch in branches
        )
        buffers = (
            pack_block_buffers(spec, state.buffers) if dense else state.buffers.buckets
        )
        buffers = run_program(program, kernels, buffers)
        if plan.options.normalize:
            first = first_access.read(buffers, 0)
            magnitude = jnp.sqrt(
                jnp.real(sum(jnp.vdot(block, block) for block in first))
            )
            first = tuple(
                block / jnp.where(magnitude > 0, magnitude, 1) for block in first
            )
            buffers = first_access.write(buffers, 0, first)
        return MPSState(
            plan.mps_spec,
            (
                unpack_block_buffers(spec, buffers)
                if dense
                else BlockBuffers(spec, buffers)
            ),
        )

    return step


def make_tebd_step(
    mps_spec: MPSSpec,
    gate_spec: GateSpec,
    options: Optional[TEBDOptions] = None,
) -> Callable[[MPSState, GeneratorTree, Array], MPSState]:
    """
    Build a TEBD step directly from MPS and gate specifications.

    :param mps_spec: Static MPS physical, bond, and symmetry layout.
    :type mps_spec: MPSSpec
    :param gate_spec: Static physical dimensions of the compiled generators.
    :type gate_spec: GateSpec
    :param options: Static TEBD settings, or ``None`` for defaults.
    :type options: Optional[TEBDOptions]
    :return: A callable accepting an MPS, generators, and a time step.
    :rtype: Callable[[MPSState, GeneratorTree, Array], MPSState]
    """
    if options is None:
        options = TEBDOptions()
    return make_tebd_step_from_plan(prepare_tebd(mps_spec, gate_spec, options))
