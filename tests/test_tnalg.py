"""Core correctness regressions for ``tensorcircuit.tnalg``."""

import importlib

import numpy as np
import pytest
import scipy.linalg

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.scipy.linalg import expm

import tensorcircuit as tc
import tensorcircuit.tnalg.interop as interop
import tensorcircuit.tnalg.mpo as mpo_module
import tensorcircuit.tnalg.mps as mps_module
import tensorcircuit.tnalg.tensor as tensor_module
from tensorcircuit.tnalg import block_ops
from tensorcircuit.tnalg.algorithms import tebd as tebd_module
from tensorcircuit.tnalg.environment import (
    boundary_environment,
    build_environments,
    build_right_environments,
    local_matvec,
    update_left,
)
from tensorcircuit.tnalg.krylov import (
    _projected_exponential,
    expm_action_hermitian,
    lowest_eigenvector_hermitian,
)
from tensorcircuit.tnalg.layout import block_bucket_plan, block_mpo_spec, block_mps_spec
from tensorcircuit.tnalg.mpo import _dense_tensors as dense_mpo_tensors
from tensorcircuit.tnalg.mps import (
    _blocks_to_dense,
    _dense_tensors as dense_mps_tensors,
    _with_buffers,
    pack_block_buffers,
    unpack_block_buffers,
)
from tensorcircuit.tnalg.sweep import (
    BlockAccess,
    prepare_program,
    prepare_schedule,
    run_program,
    site_access,
)
from tensorcircuit.tnalg.symmetric_environment import (
    environment_bucket_plan,
    mpo_build_environments,
    mpo_build_right_environments,
    mpo_left_boundary,
    mpo_local_matvec,
)
from tensorcircuit.tnalg.symmetric_linalg import factor_left, factor_right

# Independent small-system dense references.  They do not use TNALG builders,
# environments, or solvers.
PAULI_I = np.asarray(((1, 0), (0, 1)), dtype=np.complex128)
PAULI_X = np.asarray(((0, 1), (1, 0)), dtype=np.complex128)
PAULI_Y = np.asarray(((0, -1j), (1j, 0)), dtype=np.complex128)
PAULI_Z = np.asarray(((1, 0), (0, -1)), dtype=np.complex128)


def kron_sites(operators):
    result = np.asarray(((1,),), dtype=np.complex128)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def xyz_hamiltonian(nsites, xx=1.0, yy=0.7, zz=0.5, x_field=0.31, z_field=0.17):
    hamiltonian = np.zeros((2**nsites, 2**nsites), dtype=np.complex128)
    for site in range(nsites - 1):
        for coefficient, operator in ((xx, PAULI_X), (yy, PAULI_Y), (zz, PAULI_Z)):
            local = [PAULI_I] * nsites
            local[site] = operator
            local[site + 1] = operator
            hamiltonian += coefficient * kron_sites(local)
    for site in range(nsites):
        for coefficient, operator in ((x_field, PAULI_X), (z_field, PAULI_Z)):
            local = [PAULI_I] * nsites
            local[site] = operator
            hamiltonian += coefficient * kron_sites(local)
    return hamiltonian


def u1_xxz_hamiltonian(nsites, delta=0.7, field=0.23):
    hamiltonian = np.zeros((2**nsites, 2**nsites), dtype=np.complex128)
    for site in range(nsites - 1):
        for operator, coefficient in ((PAULI_X, 1.0), (PAULI_Y, 1.0), (PAULI_Z, delta)):
            local = [PAULI_I] * nsites
            local[site] = operator
            local[site + 1] = operator
            hamiltonian += coefficient * kron_sites(local)
    for site in range(nsites):
        local = [PAULI_I] * nsites
        local[site] = PAULI_Z
        hamiltonian += field * (-1) ** site * kron_sites(local)
    return hamiltonian


def excitation_sector_indices(nsites, charge):
    return np.asarray(
        [index for index in range(2**nsites) if index.bit_count() == charge],
        dtype=np.intp,
    )


def mps_statevector(tensors):
    iterator = iter(tensors)
    result = next(iterator)
    for tensor in iterator:
        result = np.tensordot(result, tensor, axes=(-1, 0))
    return np.reshape(np.squeeze(result, axis=(0, -1)), (-1,))


def center_embedding(tensors, site):
    shape = tensors[site].shape
    columns = []
    for flat_index in range(int(np.prod(shape))):
        basis_tensor = np.zeros(shape, dtype=np.complex128)
        basis_tensor.reshape(-1)[flat_index] = 1
        candidate = list(tensors)
        candidate[site] = basis_tensor
        columns.append(mps_statevector(candidate))
    return np.stack(columns, axis=1)


def test_abelian_group_basis_and_u1_mps_layout(jaxb):
    """U(1) sector quotas produce the canonical MPS block layout."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((1,), (0,), (1,)), flow=1
    )
    assert physical.charges == ((0,), (1,))
    assert physical.degeneracies == (1, 2)
    assert physical.offsets == (0, 1)
    assert physical.basis_to_canonical == (1, 0, 2)
    assert physical.canonical_to_basis == (1, 0, 2)
    assert physical.dual().flow == -1

    spin_half = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(spin_half,) * 4,
        total_charge=(2,),
        bond_sectors=(
            {(0,): 1},
            {(0,): 1, (1,): 1},
            {(0,): 1, (1,): 2, (2,): 1},
            {(1,): 1, (2,): 1},
            {(2,): 1},
        ),
    )
    assert spec.physical_dims == (2, 2, 2, 2)
    assert spec.bond_dims == (1, 2, 4, 2, 1)
    assert spec.total_charge == (2,)
    assert spec.site_layouts is not None
    assert spec.site_layouts[1].block_coordinates == (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 2),
    )


def test_high_level_u1_and_z2_mps_specs(jaxb):
    """Spin symmetry constructors allocate valid fixed-quota layouts."""

    u1 = tc.tnalg.MPSSpec.u1(4, total_charge=2, chi=8, charge_sectors=3)
    assert u1.symmetry == tc.tnalg.AbelianSymmetry((0,))
    assert u1.total_charge == (2,)
    assert u1.bond_dims == (1, 2, 4, 2, 1)
    assert u1.bond_sectors is not None
    assert u1.bond_sectors[2] == (((0,), 1), ((1,), 2), ((2,), 1))

    z2 = tc.tnalg.MPSSpec.z2(6, chi=5, parity=1)
    assert z2.symmetry == tc.tnalg.AbelianSymmetry((2,))
    assert z2.total_charge == (1,)
    assert z2.bond_dims == (1, 2, 4, 5, 4, 2, 1)
    assert z2.bond_sectors is not None
    assert z2.bond_sectors[3] == (((0,), 3), ((1,), 2))

    with pytest.raises(ValueError, match="between one and chi"):
        tc.tnalg.MPSSpec.u1(4, total_charge=2, chi=2, charge_sectors=3)
    with pytest.raises(ValueError, match="zero or one"):
        tc.tnalg.MPSSpec.z2(4, chi=4, parity=2)


def test_automatic_u1_profile_is_center_enhanced(jaxb):
    """The automatic U(1) layout uses a centered nonuniform quota profile."""

    spec = tc.tnalg.MPSSpec.u1(32, total_charge=16, chi=32)
    assert spec.bond_sectors is not None
    assert spec.bond_sectors[16] == (
        ((6,), 2),
        ((7,), 8),
        ((8,), 12),
        ((9,), 8),
        ((10,), 2),
    )
    assert max(spec.bond_dims) == 32


def test_u1_accepts_explicit_bond_sector_dimensions(jaxb):
    """The U(1) convenience constructor accepts full per-bond quotas."""

    spec = tc.tnalg.MPSSpec.u1(
        4,
        total_charge=2,
        chi=8,
        bond_sectors=(
            {0: 1},
            {0: 1, 1: 1},
            {0: 1, 1: 2, 2: 1},
            {1: 1, 2: 1},
            {2: 1},
        ),
    )
    assert spec.bond_sectors is not None
    assert spec.bond_sectors[2] == (((0,), 1), ((1,), 2), ((2,), 1))
    with pytest.raises(ValueError, match="mutually exclusive"):
        tc.tnalg.MPSSpec.u1(
            4,
            total_charge=2,
            chi=8,
            charge_sectors=3,
            bond_sectors=({0: 1},) * 5,
        )


def test_cyclic_charge_and_invalid_sector_capacity(jaxb):
    """Z3 arithmetic normalizes charges and rejects unreachable quotas."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((3,))
    assert symmetry.add((2,), (2,)) == (1,)
    assert symmetry.negate((1,)) == (2,)
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,), (2,)), flow=1
    )
    with pytest.raises(ValueError, match="exceeds left capacity"):
        tnalg.MPSSpec.from_sectors(
            physical_indices=(physical, physical),
            total_charge=(0,),
            bond_sectors=({(0,): 1}, {(0,): 2}, {(0,): 1}),
        )


def test_symmetric_product_state_uses_block_buffer_pytree(jaxb):
    """A charge-valid product state packs legal blocks into tensor buckets."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,) * 4,
        total_charge=(2,),
        bond_sectors=(
            {(0,): 1},
            {(0,): 1, (1,): 1},
            {(0,): 1, (1,): 2, (2,): 1},
            {(1,): 1, (2,): 1},
            {(2,): 1},
        ),
    )
    state, _ = tnalg.product_state((0, 1, 0, 1), spec=spec, dtype=jnp.complex64)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    plan = block_bucket_plan(spec.site_layouts)
    assert len(leaves) == len(plan.shapes)
    assert len(leaves) < sum(len(layout.block_shapes) for layout in spec.site_layouts)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np_state = dense_mps_tensors(restored)
    vector = np_state[0]
    for tensor in np_state[1:]:
        vector = jnp.einsum("...a,apb->...pb", vector, tensor)
    vector = jnp.reshape(vector, (-1,))
    np.testing.assert_allclose(np.asarray(tnalg.norm(restored)), 1.0, atol=2e-5)
    np.testing.assert_allclose(np.asarray(vector[5]), 1.0, atol=2e-5)
    view = tnalg.site_view(restored, 1)
    np.testing.assert_allclose(
        np.asarray(tnalg.to_dense(view)), np.asarray(np_state[1]), atol=2e-5
    )
    transposed = tnalg.transpose(view, (2, 1, 0))
    np.testing.assert_allclose(
        np.asarray(tnalg.to_dense(transposed)),
        np.asarray(jnp.transpose(np_state[1], (2, 1, 0))),
        atol=2e-5,
    )


def test_symmetric_pauli_generator_requires_coefficient_tying(jaxb):
    """XX and YY must share a coefficient before they define a U(1) generator."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    with pytest.raises(ValueError, match="breaks"):
        tnalg.compile_tebd_gates(codes, (physical, physical))
    _, build_gates = tnalg.compile_tebd_gates(
        codes,
        (physical, physical),
        coefficient_indices=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    onsite, bonds = build_gates(jnp.asarray((1.0,), dtype=jnp.float32))
    assert len(onsite) == 2
    assert len(bonds) == 1


def test_symmetric_long_range_mpo_validates_tied_operator_strings(jaxb):
    """Long-range MPO terms obey the same static conservation contract."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    breaking = jnp.asarray(((1, 0, 1),), dtype=jnp.int32)
    with pytest.raises(ValueError, match="breaks"):
        tnalg.compile_mpo(breaking, (physical,) * 3)
    conserving = jnp.asarray(((1, 0, 1), (2, 0, 2)), dtype=jnp.int32)
    spec, build_mpo = tnalg.compile_mpo(
        conserving,
        (physical,) * 3,
        coefficient_indices=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    mpo = build_mpo(jnp.asarray((1.0,), dtype=jnp.float32))
    assert mpo.spec == spec


def test_symmetric_mpo_shares_identity_channels_without_path_explosion(jaxb):
    """The direct Pauli-string compiler shares identity channels along a chain."""

    tnalg = tc.tnalg
    length = 12
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    rows, groups = [], []
    for site in range(length - 1):
        for code in (1, 2):
            row = [0] * length
            row[site] = row[site + 1] = code
            rows.append(row)
            groups.append(site)
    spec, _ = tnalg.compile_mpo(
        jnp.asarray(rows, dtype=jnp.int32),
        (physical,) * length,
        coefficient_indices=jnp.asarray(groups, dtype=jnp.int32),
    )
    assert spec.bond_indices is not None
    assert max(index.dimension for index in spec.bond_indices) == 6


def test_symmetric_mpo_uses_bucket_pytree_storage(jaxb):
    """A block-sparse MPO has one PyTree leaf per padded shape bucket."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    codes = jnp.asarray(
        ((1, 1, 0, 0), (2, 2, 0, 0), (0, 0, 1, 1), (0, 0, 2, 2)),
        dtype=jnp.int32,
    )
    spec, build_mpo = tnalg.compile_mpo(
        codes,
        (physical,) * 4,
        coefficient_indices=jnp.asarray((0, 0, 1, 1), dtype=jnp.int32),
    )
    mpo = build_mpo(jnp.asarray((1.0, 0.7), dtype=jnp.float32))
    plan = block_bucket_plan(spec.site_layouts)
    assert len(jax.tree.leaves(mpo)) == len(plan.shapes)
    assert len(jax.tree.leaves(mpo)) < sum(
        len(layout.block_shapes) for layout in spec.site_layouts
    )


def test_symmetric_mpo_preserves_a_constant_identity_term(jaxb):
    """The finite-state compiler represents a scalar identity without a support set."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
    )
    mps, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    _, build = tnalg.compile_mpo(
        jnp.zeros((1, 2), dtype=jnp.int32), spec.physical_indices
    )
    mpo = build(jnp.asarray((0.37,), dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(tnalg.expectation(mps, mpo)), 0.37, atol=2e-5)


def test_symmetric_tebd_preserves_u1_and_matches_xy_exchange(jaxb):
    """Sector-wise TEBD evolves a U(1) charge-one two-site XY state."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
    )
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    gate_spec, build_gates = tnalg.compile_tebd_gates(
        codes,
        spec.physical_indices,
        coefficient_indices=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    state, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    evolution = state
    step = tnalg.make_tebd_step(spec, gate_spec, tnalg.TEBDOptions(order=2))
    time_step = jnp.asarray(0.05, dtype=jnp.float32)
    evolved = jax.jit(step)(
        evolution,
        build_gates(jnp.asarray((1.0,), dtype=jnp.float32)),
        time_step,
    )
    tensors = dense_mps_tensors(evolved)
    vector = tensors[0]
    for tensor in tensors[1:]:
        vector = jnp.einsum("...a,apb->...pb", vector, tensor)
    vector = jnp.reshape(vector, (-1,))
    expected = jnp.asarray(
        (0.0, jnp.cos(2 * time_step), -1j * jnp.sin(2 * time_step), 0.0),
        dtype=jnp.complex64,
    )
    np.testing.assert_allclose(np.asarray(vector), np.asarray(expected), atol=2e-4)


def test_symmetric_tdvp_preserves_u1_and_matches_xy_exchange(jaxb):
    """Sector-wise TDVP uses the same U(1) state and Pauli-string contract."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
    )
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(
        codes,
        spec.physical_indices,
        coefficient_indices=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    state, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    evolution = state
    step = tnalg.make_tdvp_step(spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=4))
    time_step = jnp.asarray(0.05, dtype=jnp.float32)
    evolved = jax.jit(step)(
        evolution,
        build_mpo(jnp.asarray((1.0,), dtype=jnp.float32)),
        time_step,
    )
    tensors = dense_mps_tensors(evolved)
    vector = tensors[0]
    for tensor in tensors[1:]:
        vector = jnp.einsum("...a,apb->...pb", vector, tensor)
    vector = jnp.reshape(vector, (-1,))
    expected = jnp.asarray(
        (0.0, jnp.cos(2 * time_step), -1j * jnp.sin(2 * time_step), 0.0),
        dtype=jnp.complex64,
    )
    np.testing.assert_allclose(np.asarray(vector), np.asarray(expected), atol=2e-4)

    _, build_observable = tnalg.compile_mpo(
        jnp.asarray(((3, 0),), dtype=jnp.int32), spec.physical_indices
    )
    observable = build_observable(jnp.asarray((1.0,), dtype=jnp.float32))

    def loss(weight):
        result = step(
            evolution, build_mpo(jnp.asarray((weight,), dtype=jnp.float32)), time_step
        )
        return jnp.real(tnalg.expectation(result, observable))

    _, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.7, dtype=jnp.float32))
    finite_difference = (
        loss(jnp.asarray(0.701, dtype=jnp.float32))
        - loss(jnp.asarray(0.699, dtype=jnp.float32))
    ) / 0.002
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(finite_difference), atol=2e-3
    )


def test_symmetric_dmrg_finds_charge_one_xy_ground_state(jaxb):
    """Symmetric one-site DMRG remains in its target U(1) sector."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
    )
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(
        codes,
        spec.physical_indices,
        coefficient_indices=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    initial, _ = tnalg.random_mps(jax.random.key(2), spec=spec, dtype=jnp.complex64)
    sweep = tnalg.make_dmrg_sweep(spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=4))
    final, energy = jax.jit(sweep)(
        initial, build_mpo(jnp.asarray((1.0,), dtype=jnp.float32))
    )
    np.testing.assert_allclose(np.asarray(energy), -2.0, atol=2e-4)
    np.testing.assert_allclose(
        np.asarray(energy),
        np.asarray(tnalg.expectation(final, build_mpo(jnp.asarray((1.0,))))),
        atol=2e-4,
    )
    np.testing.assert_allclose(np.asarray(tnalg.norm(final)), 1.0, atol=2e-4)


def test_checkpoint_roundtrip_preserves_symmetric_mps_state(jaxb, tmp_path):
    """Checkpoint metadata reconstructs the sector layout without pickled plans."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
    )
    mps, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    state = mps
    path = tmp_path / "state.npz"
    tnalg.save_checkpoint(path, state)
    restored = tnalg.load_checkpoint(path)
    for expected, actual in zip(dense_mps_tensors(state), dense_mps_tensors(restored)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_z2_and_z3_symmetric_tebd_models(jaxb):
    """Cyclic charge arithmetic works for parity and three-state clock exchange."""

    tnalg = tc.tnalg

    z2 = tnalg.AbelianSymmetry((2,))
    qubit = tnalg.SectorIndex.from_basis(
        symmetry=z2, basis_charges=((0,), (1,)), flow=1
    )
    z2_spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(qubit, qubit),
        total_charge=(0,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(0,): 1}),
    )
    table_z2 = jnp.asarray(
        (((1, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 0), (0, -1))),
        dtype=jnp.complex64,
    )
    codes_z2 = jnp.asarray(((1, 1),), dtype=jnp.int32)
    gate_spec, build_gates = tnalg.compile_tebd_gates(
        codes_z2, z2_spec.physical_indices, operator_table=table_z2
    )
    mps, _ = tnalg.product_state((0, 0), spec=z2_spec, dtype=jnp.complex64)
    state = mps
    step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(z2_spec, gate_spec, tnalg.TEBDOptions(order=2))
    )
    evolved = jax.jit(step)(
        state,
        build_gates(jnp.asarray((1.0,), dtype=jnp.float32)),
        jnp.asarray(0.1, dtype=jnp.float32),
    )
    tensors = dense_mps_tensors(evolved)
    vector = jnp.reshape(jnp.einsum("lpa,aqr->lpqr", tensors[0], tensors[1]), (-1,))
    np.testing.assert_allclose(np.asarray(vector[1:3]), np.zeros(2), atol=2e-5)

    z3 = tnalg.AbelianSymmetry((3,))
    qutrit = tnalg.SectorIndex.from_basis(
        symmetry=z3, basis_charges=((0,), (1,), (2,)), flow=1
    )
    z3_spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(qutrit, qutrit),
        total_charge=(0,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1, (2,): 1}, {(0,): 1}),
    )
    shift = jnp.asarray(((0, 0, 1), (1, 0, 0), (0, 1, 0)), dtype=jnp.complex64)
    table_z3 = jnp.stack((jnp.eye(3, dtype=jnp.complex64), shift, jnp.conj(shift.T)))
    codes_z3 = jnp.asarray(((1, 2), (2, 1)), dtype=jnp.int32)
    gate_spec, build_gates = tnalg.compile_tebd_gates(
        codes_z3,
        z3_spec.physical_indices,
        operator_table=table_z3,
        coefficient_indices=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    mps, _ = tnalg.product_state((0, 0), spec=z3_spec, dtype=jnp.complex64)
    state = mps
    step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(z3_spec, gate_spec, tnalg.TEBDOptions(order=2))
    )
    evolved = jax.jit(step)(
        state,
        build_gates(jnp.asarray((1.0,), dtype=jnp.float32)),
        jnp.asarray(0.1, dtype=jnp.float32),
    )
    np.testing.assert_allclose(np.asarray(tnalg.norm(evolved)), 1.0, atol=2e-5)


def test_z3_symmetric_tebd_matches_dense_reference(jaxb, highp):
    """The Z3 block-sparse exchange follows its independent dense evolution."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((3,))
    qutrit = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,), (2,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(qutrit, qutrit),
        total_charge=(0,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1, (2,): 1}, {(0,): 1}),
    )
    shift = np.asarray(((0, 0, 1), (1, 0, 0), (0, 1, 0)), dtype=np.complex128)
    table = np.stack((np.eye(3), shift, shift.conj().T))
    codes = np.asarray(((1, 2), (2, 1)), dtype=np.int32)
    gate_spec, build_gates = tnalg.compile_tebd_gates(
        codes,
        spec.physical_indices,
        operator_table=table,
        coefficient_indices=np.asarray((0, 0), dtype=np.int32),
    )
    initial, _ = tnalg.product_state((0, 0), spec=spec, dtype=jnp.complex128)
    time_step = 0.13
    step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(spec, gate_spec, tnalg.TEBDOptions(order=2))
    )
    evolved = jax.jit(step)(
        initial,
        build_gates(jnp.asarray((1.0,), dtype=jnp.float64)),
        jnp.asarray(time_step, dtype=jnp.float64),
    )
    actual = mps_statevector(dense_mps_tensors(evolved))
    hamiltonian = np.kron(shift, shift.conj().T) + np.kron(shift.conj().T, shift)
    initial_vector = np.zeros(9, dtype=np.complex128)
    initial_vector[0] = 1.0
    expected = scipy.linalg.expm(-1j * time_step * hamiltonian) @ initial_vector
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(tnalg.norm(evolved)), 1.0, atol=2e-10)


def test_z2_tdvp_matches_exact_tfi(jaxb):
    """Parity-preserving TDVP has the same TFI evolution as exact evolution."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((2,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(0,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(0,): 1}),
    )
    codes = jnp.asarray(((1, 1), (3, 0), (0, 3)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    weights = jnp.asarray((-1.0, -0.8, -0.8), dtype=jnp.float32)
    mps, _ = tnalg.product_state((0, 0), spec=spec, dtype=jnp.complex64)
    state = mps
    time_step = jnp.asarray(0.05, dtype=jnp.float32)
    plan = tnalg.prepare_tdvp(spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=4))
    monolithic = jax.jit(tnalg.make_tdvp_step_from_plan(plan))(
        state, build_mpo(weights), time_step
    )

    def statevector(value):
        tensors = dense_mps_tensors(value)
        return jnp.reshape(jnp.einsum("lpa,aqr->lpqr", tensors[0], tensors[1]), (-1,))

    tdvp_vector = np.asarray(statevector(monolithic))
    hamiltonian = -np.kron(PAULI_X, PAULI_X) - 0.8 * (
        np.kron(PAULI_Z, PAULI_I) + np.kron(PAULI_I, PAULI_Z)
    )
    exact = scipy.linalg.expm(-1j * float(time_step) * hamiltonian) @ np.asarray(
        (1.0, 0.0, 0.0, 0.0), dtype=np.complex64
    )
    np.testing.assert_allclose(tdvp_vector, exact, atol=3e-5)

    _, build_observable = tnalg.compile_mpo(
        jnp.asarray(((3, 0),), dtype=jnp.int32), spec.physical_indices
    )
    observable = build_observable(jnp.asarray((1.0,), dtype=jnp.float32))
    tdvp_step = tnalg.make_tdvp_step_from_plan(plan)

    def loss(coupling):
        dynamic_weights = weights.at[0].set(coupling)
        result = tdvp_step(state, build_mpo(dynamic_weights), time_step)
        return jnp.real(tnalg.expectation(result, observable))

    _, gradient = jax.jit(jax.value_and_grad(loss))(
        jnp.asarray(-1.0, dtype=jnp.float32)
    )
    finite_difference = (
        loss(jnp.asarray(-0.999, dtype=jnp.float32))
        - loss(jnp.asarray(-1.001, dtype=jnp.float32))
    ) / 0.002
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(finite_difference), atol=2e-3
    )


def test_z2_dmrg_reaches_full_space_tfi_ground_state(jaxb):
    """A full parity-preserving MPS reaches the TFI ground state under DMRG."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((2,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,) * 4,
        total_charge=(0,),
        bond_sectors=(
            {(0,): 1},
            {(0,): 1, (1,): 1},
            {(0,): 2, (1,): 2},
            {(0,): 1, (1,): 1},
            {(0,): 1},
        ),
    )
    codes = []
    for site in range(3):
        row = [0] * 4
        row[site] = row[site + 1] = 1
        codes.append(row)
    for site in range(4):
        row = [0] * 4
        row[site] = 3
        codes.append(row)
    mpo_spec, build_mpo = tnalg.compile_mpo(
        jnp.asarray(codes, dtype=jnp.int32), spec.physical_indices
    )
    mpo = build_mpo(jnp.asarray((-1.0,) * 3 + (-0.8,) * 4, dtype=jnp.float32))
    initial, _ = tnalg.random_mps(jax.random.key(8), spec=spec, dtype=jnp.complex64)
    sweep = tnalg.make_dmrg_sweep_from_plan(
        tnalg.prepare_dmrg(spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=16))
    )

    def solve(mps):
        return jax.lax.scan(lambda carry, _: sweep(carry, mpo), mps, xs=None, length=2)

    final, energies = jax.jit(solve)(initial)
    hamiltonian = np.zeros((16, 16), dtype=np.complex128)
    for site in range(3):
        operators = [PAULI_I] * 4
        operators[site] = operators[site + 1] = PAULI_X
        hamiltonian -= kron_sites(operators)
    for site in range(4):
        operators = [PAULI_I] * 4
        operators[site] = PAULI_Z
        hamiltonian -= 0.8 * kron_sites(operators)
    exact_energy = np.linalg.eigvalsh(hamiltonian)[0]
    np.testing.assert_allclose(np.asarray(energies[-1]), exact_energy, atol=2e-4)
    np.testing.assert_allclose(np.asarray(tnalg.variance(final, mpo)), 0.0, atol=2e-4)


def test_direct_product_u1_sector_layout_and_state(jaxb):
    """A two-component charge index keeps both commuting charges in the layout."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0, 0))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry,
        basis_charges=((0, 0), (1, 0), (0, 1), (1, 1)),
        flow=1,
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1, 1),
        bond_sectors=(
            {(0, 0): 1},
            {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1},
            {(1, 1): 1},
        ),
    )
    state, _ = tnalg.product_state((3, 0), spec=spec, dtype=jnp.complex64)
    assert spec.bond_dims == (1, 4, 1)
    np.testing.assert_allclose(np.asarray(tnalg.norm(state)), 1.0, atol=2e-5)


def test_charge_fusion_plan_roundtrips_canonical_dense_axes(jaxb):
    """Fusion records a reversible canonical basis reordering by total charge."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    fused, plan = tnalg.fuse_indices((physical, physical), flow=1)
    assert fused.charges == ((0,), (1,), (2,))
    assert fused.degeneracies == (1, 2, 1)
    tensor = jnp.arange(12, dtype=jnp.float32).reshape((2, 2, 3))
    fused_tensor = tnalg.fuse_array(tensor, (0, 1), plan)
    restored = tnalg.unfuse_array(fused_tensor, (0, 1), plan)
    np.testing.assert_allclose(np.asarray(restored), np.asarray(tensor))


def test_static_plans_and_environment_are_jax_pytree_safe(jaxb):
    """Plans stay host-static while environment buffers remain dynamic leaves."""

    tnalg = tc.tnalg
    mps_spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    _, build_mpo = tnalg.compile_mpo(
        jnp.asarray(((3, 0),), dtype=jnp.int32), mps_spec.physical_indices
    )
    mps, _ = tnalg.product_state((0, 1), spec=mps_spec, dtype=jnp.complex64)
    environment = build_environments(
        dense_mps_tensors(mps),
        dense_mpo_tensors(build_mpo(jnp.asarray((1.0,), dtype=jnp.float32))),
    )
    leaves, tree = jax.tree_util.tree_flatten(environment)
    assert len(leaves) == 6
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert isinstance(restored, tnalg.EnvironmentState)
    plan = tnalg.prepare_tebd(
        mps_spec,
        tnalg.compile_tebd_gates(
            jnp.asarray(((3, 0),), dtype=jnp.int32), mps_spec.physical_indices
        )[0],
        tnalg.TEBDOptions(order=2),
    )
    assert plan.mps_spec.bond_dims == (1, 2, 1)


def test_symmetric_legacy_export_requires_explicit_densification(jaxb):
    """Converting a block MPS to legacy objects cannot silently densify it."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
    )
    state, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    with pytest.raises(ValueError, match="allow_dense"):
        tnalg.to_mpscircuit(state)
    legacy = tnalg.to_mpscircuit(state, allow_dense=True, max_elements=100)
    assert legacy.get_center_position() == 0


def test_symmetric_qr_gauge_moves_preserve_the_full_state(jaxb):
    """Sector-wise QR/RQ transfers are exact gauge changes before optimization."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,) * 4,
        total_charge=(2,),
        bond_sectors=(
            {(0,): 1},
            {(0,): 1, (1,): 1},
            {(0,): 1, (1,): 2, (2,): 1},
            {(1,): 1, (2,): 1},
            {(2,): 1},
        ),
    )
    state, _ = tnalg.random_mps(jax.random.key(9), spec=spec, dtype=jnp.complex64)

    def vector(value):
        tensors = [
            _blocks_to_dense(spec, site, buffers)
            for site, buffers in enumerate(value.buffers)
        ]
        result = tensors[0]
        for tensor in tensors[1:]:
            result = jnp.einsum("...a,apb->...pb", result, tensor)
        return jnp.reshape(result, (-1,))

    reference = vector(state)
    buffers = [list(site) for site in state.buffers]
    transfer = factor_left(buffers, spec, 1)
    next_tensor = jnp.einsum(
        "ab,bpc->apc", transfer, _blocks_to_dense(spec, 2, tuple(buffers[2]))
    )
    buffers[2] = list(tnalg.from_dense(spec.site_layouts[2], next_tensor).buffers)
    left_shifted = _with_buffers(spec, tuple(tuple(site) for site in buffers))
    np.testing.assert_allclose(
        np.asarray(vector(left_shifted)), np.asarray(reference), atol=3e-5
    )

    buffers = [list(site) for site in state.buffers]
    transfer = factor_right(buffers, spec, 2)
    previous_tensor = jnp.einsum(
        "apb,bc->apc", _blocks_to_dense(spec, 1, tuple(buffers[1])), transfer
    )
    buffers[1] = list(tnalg.from_dense(spec.site_layouts[1], previous_tensor).buffers)
    right_shifted = _with_buffers(spec, tuple(tuple(site) for site in buffers))
    np.testing.assert_allclose(
        np.asarray(vector(right_shifted)), np.asarray(reference), atol=3e-5
    )


def test_tebd_scan_and_vmap_share_one_static_spec(jaxb):
    """A fixed plan supports time scan and independent state batching."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    gate_spec, build_gates = tnalg.compile_tebd_gates(codes, spec.physical_indices)
    step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(spec, gate_spec, tnalg.TEBDOptions(order=2))
    )
    generators = build_gates(jnp.asarray((1.0, 1.0), dtype=jnp.float32))
    time_step = jnp.asarray(0.05, dtype=jnp.float32)
    initial, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    state = initial

    jax.jit(
        lambda value: jax.lax.scan(
            lambda carry, _: (step(carry, generators, time_step), None),
            value,
            xs=None,
            length=2,
        )
    )(state)
    other, _ = tnalg.product_state((1, 0), spec=spec, dtype=jnp.complex64)
    batch = jax.tree_util.tree_map(
        lambda first, second: jnp.stack((first, second)),
        state,
        other,
    )
    evolved = jax.jit(jax.vmap(lambda value: step(value, generators, time_step)))(batch)
    assert evolved.buffers[0][0].shape[0] == 2


def test_fixed_lanczos_handles_breakdown_without_fake_ritz_ground_state(jaxb):
    """An exact positive-energy eigenvector remains the chosen Ritz vector."""

    hamiltonian = jnp.asarray(
        ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 5.0)),
        dtype=jnp.complex64,
    )
    vector = jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.complex64)
    candidate, eigenvalue, report = jax.jit(
        lambda value: lowest_eigenvector_hermitian(
            lambda direction: hamiltonian @ direction, value, max_dim=3
        )
    )(vector)
    np.testing.assert_allclose(np.asarray(eigenvalue), 2.0, atol=2e-5)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(vector))
    np.testing.assert_allclose(np.asarray(report["residual"]), 0.0, atol=2e-5)
    evolved, _ = jax.jit(
        lambda value: expm_action_hermitian(
            lambda direction: hamiltonian @ direction,
            value,
            jnp.asarray(-0.2j, dtype=jnp.complex64),
            max_dim=3,
        )
    )(jnp.asarray((1.0, -2j, 0.5), dtype=jnp.complex64))
    reference = jsp_linalg.expm(-0.2j * hamiltonian) @ jnp.asarray(
        (1.0, -2j, 0.5), dtype=jnp.complex64
    )
    np.testing.assert_allclose(np.asarray(evolved), np.asarray(reference), atol=3e-5)


def test_fixed_lanczos_zero_vector_is_finite_and_shape_stable(jaxb):
    """A zero input does not create NaNs or change static Krylov output shape."""

    result, _ = jax.jit(
        lambda vector: expm_action_hermitian(
            lambda value: 2.0 * value,
            vector,
            jnp.asarray(-0.1j, dtype=jnp.complex64),
            max_dim=4,
        )
    )(jnp.zeros((4,), dtype=jnp.complex64))
    np.testing.assert_array_equal(np.asarray(result), np.zeros(4, dtype=np.complex64))


def test_symmetric_block_environment_and_local_matvec_match_dense_reference(jaxb):
    """Block environments act on MPS buffers without dense-site reconstruction."""

    tnalg = tc.tnalg
    symmetry = tnalg.AbelianSymmetry((0,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical, physical),
        total_charge=(1,),
        bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
    )
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(
        codes,
        spec.physical_indices,
        coefficient_indices=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    mpo = build_mpo(jnp.asarray((1.0,), dtype=jnp.float32))
    mps, _ = tnalg.random_mps(jax.random.key(10), spec=spec, dtype=jnp.complex64)
    mpo_tensors = dense_mpo_tensors(mpo)
    assert mpo.spec.site_layouts is not None
    block_right = mpo_build_right_environments(spec, mpo_spec, mps.buffers, mpo.buffers)
    environment_state = mpo_build_environments(spec, mpo_spec, mps.buffers, mpo.buffers)
    leaves, tree = jax.tree_util.tree_flatten(environment_state)
    assert len(leaves) == sum(
        len(environment)
        for environment in environment_state.left + environment_state.right
    )
    assert isinstance(
        jax.tree_util.tree_unflatten(tree, leaves), tnalg.EnvironmentState
    )
    for actual, expected_environment in zip(environment_state.right, block_right):
        for actual_block, expected_block in zip(actual, expected_environment):
            np.testing.assert_allclose(
                np.asarray(actual_block), np.asarray(expected_block)
            )
    block_output = mpo_local_matvec(
        spec,
        mpo_spec,
        0,
        mpo_left_boundary(spec, mpo_spec, jnp.complex64),
        mpo.buffers[0],
        block_right[1],
        mps.buffers[0],
    )
    dense_tensors = tuple(
        _blocks_to_dense(spec, site, buffers)
        for site, buffers in enumerate(mps.buffers)
    )
    dense_right = build_right_environments(dense_tensors, mpo_tensors)
    dense_output = local_matvec(
        boundary_environment(jnp.complex64),
        mpo_tensors[0],
        dense_right[1],
        dense_tensors[0],
    )
    expected = tnalg.from_dense(spec.site_layouts[0], dense_output).buffers
    for actual, reference in zip(block_output, expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), atol=3e-5)
    state_vector = jnp.reshape(
        jnp.einsum("apb,bqc->apqc", dense_tensors[0], dense_tensors[1]), (-1,)
    )
    xx_plus_yy = jnp.kron(
        jnp.asarray(((0, 1), (1, 0))), jnp.asarray(((0, 1), (1, 0)))
    ) + jnp.kron(jnp.asarray(((0, -1j), (1j, 0))), jnp.asarray(((0, -1j), (1j, 0))))
    state_norm = jnp.vdot(state_vector, state_vector)
    dense_mean = jnp.vdot(state_vector, xx_plus_yy @ state_vector) / state_norm
    dense_variance = jnp.real(
        jnp.vdot(state_vector, xx_plus_yy @ xx_plus_yy @ state_vector) / state_norm
        - dense_mean * dense_mean
    )
    np.testing.assert_allclose(
        np.asarray(tnalg.expectation(mps, mpo)), np.asarray(dense_mean), atol=3e-5
    )
    np.testing.assert_allclose(
        np.asarray(tnalg.variance(mps, mpo)), np.asarray(dense_variance), atol=3e-5
    )


def _statevector(mps, jnp):
    tensors = tuple(site[0] for site in mps.buffers)
    state = tensors[0]
    for tensor in tensors[1:]:
        state = jnp.einsum("...a,apb->...pb", state, tensor)
    return jnp.reshape(state, (-1,))


def _mpo_matrix(mpo, jnp):
    tensors = tuple(site[0] for site in mpo.buffers)
    tensor = tensors[0][0]
    for site in tensors[1:]:
        tensor = jnp.tensordot(tensor, site, axes=(-1, 0))
    tensor = tensor[..., 0]
    nsites = len(tensors)
    axes = tuple(range(0, 2 * nsites, 2)) + tuple(range(1, 2 * nsites, 2))
    dimension = int(np.prod(tuple(site.shape[1] for site in tensors)))
    return jnp.reshape(jnp.transpose(tensor, axes), (dimension, dimension))


def test_tebd_step_is_jittable_and_evolves_xy(jaxb):
    """A complete fixed-shape TEBD step returns the next MPS state."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    initial, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    operator_codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    gate_spec, build_gates = tnalg.compile_tebd_gates(
        operator_codes, spec.physical_indices
    )
    step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(spec, gate_spec, tnalg.TEBDOptions(order=2))
    )
    initial_state = initial
    time_step = jnp.asarray(0.125, dtype=jnp.float32)
    generators = jax.jit(build_gates)(jnp.asarray((1.0, 1.0), dtype=jnp.float32))
    evolved = jax.jit(step)(initial_state, generators, time_step)

    vector = _statevector(evolved, jnp)
    expected = jnp.asarray(
        (0.0, jnp.cos(2 * time_step), -1j * jnp.sin(2 * time_step), 0.0),
        dtype=jnp.complex64,
    )
    np.testing.assert_allclose(np.asarray(vector), np.asarray(expected), atol=2e-5)
    np.testing.assert_allclose(np.asarray(tnalg.norm(evolved)), 1.0, atol=2e-5)

    leaves, treedef = jax.tree_util.tree_flatten(evolved)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(
        np.asarray(_statevector(restored, jnp)), np.asarray(vector)
    )


def test_mpo_builder_reuses_tensor_pauli_strings(jaxb):
    """MPO construction shares the pure tensor Pauli-string input with TEBD."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    operator_codes = jnp.asarray(((1, 1), (2, 2), (0, 3)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(operator_codes, spec.physical_indices)
    weights = jnp.asarray((1.0, 1.0, 0.3), dtype=jnp.float32)
    mpo = jax.jit(build_mpo)(weights)

    pauli_x = jnp.asarray(((0, 1), (1, 0)), dtype=jnp.complex64)
    pauli_y = jnp.asarray(((0, -1j), (1j, 0)), dtype=jnp.complex64)
    pauli_z = jnp.asarray(((1, 0), (0, -1)), dtype=jnp.complex64)
    identity = jnp.eye(2, dtype=jnp.complex64)
    expected = (
        jnp.kron(pauli_x, pauli_x)
        + jnp.kron(pauli_y, pauli_y)
        + 0.3 * jnp.kron(identity, pauli_z)
    )
    np.testing.assert_allclose(
        np.asarray(_mpo_matrix(mpo, jnp)), np.asarray(expected), atol=2e-5
    )

    mps, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    np.testing.assert_allclose(np.asarray(tnalg.expectation(mps, mpo)), -0.3, atol=2e-5)
    assert mpo.spec == mpo_spec


def test_expectation_supports_unnormalized_mps(jaxb):
    """Unnormalized expectation values retain the physical norm factor."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2,), bond_dims=(1, 1))
    state, _ = tnalg.product_state((0,), spec=spec, dtype=jnp.complex64)
    scaled = _with_buffers(spec, ((2.0 * state.buffers[0][0],),))
    _, build_mpo = tnalg.compile_mpo(
        np.asarray(((3,),), dtype=np.int32), spec.physical_indices
    )
    mpo = build_mpo(jnp.asarray((1.0,), dtype=jnp.float32))
    vector = np.asarray(_statevector(scaled, jnp))
    expected = np.vdot(vector, PAULI_Z @ vector)
    np.testing.assert_allclose(
        np.asarray(tnalg.expectation(scaled, mpo, normalized=False)), expected
    )
    np.testing.assert_allclose(np.asarray(tnalg.expectation(scaled, mpo)), 1.0)


def test_dense_mps_overlap_matches_independent_dense_inner_product(jaxb):
    """Dense MPS overlap agrees with the independent state-vector inner product."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2, 2), bond_dims=(1, 2, 2, 1))
    left, _ = tnalg.random_mps(jax.random.key(12), spec=spec, dtype=jnp.complex64)
    right, _ = tnalg.random_mps(jax.random.key(13), spec=spec, dtype=jnp.complex64)
    left_vector = np.asarray(_statevector(left, jnp))
    right_vector = np.asarray(_statevector(right, jnp))
    expected = np.vdot(left_vector, right_vector)
    np.testing.assert_allclose(
        np.asarray(tnalg.overlap(left, right)), expected, atol=2e-5
    )
    np.testing.assert_allclose(
        np.asarray(tnalg.overlap(right, left)), np.conj(expected), atol=2e-5
    )


def test_mps_preparation_random_state_and_legacy_roundtrip(jaxb):
    """State preparation embeds, compresses, and round-trips without mutation."""

    tnalg = tc.tnalg
    source_spec = tnalg.MPSSpec.dense(physical_dims=(2, 2, 2), bond_dims=(1, 2, 2, 1))
    source, _ = tnalg.random_mps(
        jax.random.key(0), spec=source_spec, dtype=jnp.complex64
    )
    legacy = tnalg.to_mpscircuit(source)
    before = tuple(jnp.array(tensor) for tensor in legacy.get_tensors())
    restored, report = tnalg.as_mps(legacy, spec=source_spec)
    np.testing.assert_allclose(
        np.asarray(_statevector(restored, jnp)),
        np.asarray(_statevector(source, jnp)),
        atol=2e-5,
    )
    for expected, actual in zip(before, legacy.get_tensors()):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5)
    np.testing.assert_allclose(np.asarray(report["input_norm"]), 1.0, atol=2e-5)

    compressed_spec = tnalg.MPSSpec.dense(
        physical_dims=(2, 2, 2), bond_dims=(1, 1, 1, 1)
    )
    compressed, compressed_report = tnalg.as_mps(
        source, spec=compressed_spec, truncate=True
    )
    assert float(tnalg.norm(compressed)) > 0.0
    assert float(tnalg.norm(compressed)) <= 1.0
    assert np.all(np.asarray(compressed_report["discarded_weight_abs"]) >= 0.0)


def test_mpo_import_requires_and_obeys_axis_order(jaxb):
    """Raw MPO import never guesses a potentially ambiguous tensor axis order."""

    tnalg = tc.tnalg
    mps_spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    codes = jnp.asarray(((3, 0), (0, 3)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(codes, mps_spec.physical_indices)
    mpo = build_mpo(jnp.asarray((0.2, -0.4), dtype=jnp.float32))
    legacy_order = tuple(jnp.transpose(site[0], (0, 3, 1, 2)) for site in mpo.buffers)

    with pytest.raises(ValueError, match="axis_order"):
        tnalg.as_mpo(legacy_order, spec=mpo_spec)
    imported = tnalg.as_mpo(
        legacy_order,
        spec=mpo_spec,
        axis_order=("left", "right", "out", "in"),
    )
    for expected, actual in zip(mpo.buffers, imported.buffers):
        np.testing.assert_allclose(np.asarray(actual[0]), np.asarray(expected[0]))
    legacy = tnalg.to_tn_mpo(mpo)
    roundtrip = tnalg.as_mpo(
        legacy,
        spec=mpo_spec,
        axis_order=("left", "right", "out", "in"),
    )
    for expected, actual in zip(mpo.buffers, roundtrip.buffers):
        np.testing.assert_allclose(np.asarray(actual[0]), np.asarray(expected[0]))


def test_tdvp_step_is_jittable_for_full_two_site_space(jaxb):
    """One-site TDVP reproduces the analytic two-site XY evolution."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    mpo = build_mpo(jnp.asarray((1.0, 1.0), dtype=jnp.float32))
    mps, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    state = mps
    step = tnalg.make_tdvp_step_from_plan(
        tnalg.prepare_tdvp(spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=4))
    )
    time_step = jnp.asarray(0.05, dtype=jnp.float32)
    evolved = jax.jit(step)(state, mpo, time_step)
    expected = jnp.asarray(
        (0.0, jnp.cos(2 * time_step), -1j * jnp.sin(2 * time_step), 0.0),
        dtype=jnp.complex64,
    )
    np.testing.assert_allclose(
        np.asarray(_statevector(evolved, jnp)), np.asarray(expected), atol=2e-4
    )


def test_dmrg_sweep_finds_two_site_xy_ground_state(jaxb):
    """One-site DMRG reaches the full-space XY ground-state energy."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    codes = jnp.asarray(((1, 1), (2, 2)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    mpo = build_mpo(jnp.asarray((1.0, 1.0), dtype=jnp.float32))
    initial, _ = tnalg.random_mps(jax.random.key(1), spec=spec, dtype=jnp.complex64)
    sweep = tnalg.make_dmrg_sweep_from_plan(
        tnalg.prepare_dmrg(
            spec,
            mpo_spec,
            tnalg.DMRGOptions(krylov_dim=4),
        )
    )
    final, energy = jax.jit(sweep)(initial, mpo)
    np.testing.assert_allclose(np.asarray(energy), -2.0, atol=2e-4)
    np.testing.assert_allclose(
        np.asarray(tnalg.expectation(final, mpo)), -2.0, atol=2e-4
    )
    np.testing.assert_allclose(np.asarray(tnalg.variance(final, mpo)), 0.0, atol=2e-5)


def test_tdvp_parameter_gradient_matches_finite_difference(jaxb):
    """The MPO builder and one-site TDVP retain the Hamiltonian gradient path."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2,), bond_dims=(1, 1))
    model_spec, build_model = tnalg.compile_mpo(
        jnp.asarray(((1,),), dtype=jnp.int32), spec.physical_indices
    )
    _, build_observable = tnalg.compile_mpo(
        jnp.asarray(((3,),), dtype=jnp.int32), spec.physical_indices
    )
    initial, _ = tnalg.product_state((0,), spec=spec, dtype=jnp.complex64)
    state = initial
    step = tnalg.make_tdvp_step_from_plan(
        tnalg.prepare_tdvp(spec, model_spec, tnalg.TDVPOptions(krylov_dim=2))
    )
    observable = build_observable(jnp.asarray((1.0,), dtype=jnp.float32))
    time_step = jnp.asarray(0.2, dtype=jnp.float32)

    def loss(weight):
        evolved = step(state, build_model(jnp.asarray((weight,))), time_step)
        return jnp.real(tnalg.expectation(evolved, observable))

    value, gradient = jax.jit(jax.value_and_grad(loss))(
        jnp.asarray(0.7, dtype=jnp.float32)
    )
    finite_difference = (
        loss(jnp.asarray(0.701, dtype=jnp.float32))
        - loss(jnp.asarray(0.699, dtype=jnp.float32))
    ) / 0.002
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(finite_difference), atol=2e-3
    )
    np.testing.assert_allclose(np.asarray(value), np.cos(0.28), atol=2e-5)


def test_tebd_has_expected_trotter_step_convergence(jaxb):
    """A noncommuting nearest-neighbor model has the expected Trotter orders."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2, 2), bond_dims=(1, 2, 2, 1))
    codes = jnp.asarray(((1, 1, 0), (0, 3, 3)), dtype=jnp.int32)
    _mpo_spec, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    gate_spec, build_gates = tnalg.compile_tebd_gates(codes, spec.physical_indices)
    weights = jnp.asarray((0.7, -0.4), dtype=jnp.float32)
    hamiltonian = _mpo_matrix(build_mpo(weights), jnp)
    initial, _ = tnalg.product_state((0, 0, 1), spec=spec, dtype=jnp.complex64)
    initial_vector = _statevector(initial, jnp)
    total_time = jnp.asarray(0.2, dtype=jnp.float32)
    exact = jsp_linalg.expm(-1j * total_time * hamiltonian) @ initial_vector
    generators = build_gates(weights)

    def evolve(order, number_steps):
        step = tnalg.make_tebd_step_from_plan(
            tnalg.prepare_tebd(spec, gate_spec, tnalg.TEBDOptions(order=order))
        )
        dt = total_time / number_steps
        state = initial

        def body(carry, _):
            return step(carry, generators, dt), None

        final, _ = jax.lax.scan(body, state, xs=None, length=number_steps)
        return _statevector(final, jnp)

    first_order = float(
        jnp.log2(
            jnp.linalg.norm(evolve(1, 4) - exact)
            / jnp.linalg.norm(evolve(1, 8) - exact)
        )
    )
    second_order = float(
        jnp.log2(
            jnp.linalg.norm(evolve(2, 4) - exact)
            / jnp.linalg.norm(evolve(2, 8) - exact)
        )
    )
    assert 0.7 < first_order < 1.3
    assert 1.6 < second_order < 2.4


def test_tdvp_midpoint_time_dependent_mpo_builder(jaxb):
    """Caller-built midpoint MPOs give the expected time-dependent evolution."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2,), bond_dims=(1, 1))
    model_spec, build_mpo = tnalg.compile_mpo(
        jnp.asarray(((1,),), dtype=jnp.int32), spec.physical_indices
    )
    _, build_z = tnalg.compile_mpo(
        jnp.asarray(((3,),), dtype=jnp.int32), spec.physical_indices
    )
    initial, _ = tnalg.product_state((0,), spec=spec, dtype=jnp.complex64)
    state = initial
    step = tnalg.make_tdvp_step_from_plan(
        tnalg.prepare_tdvp(spec, model_spec, tnalg.TDVPOptions(krylov_dim=2))
    )
    observable = build_z(jnp.asarray((1.0,), dtype=jnp.float32))
    number_steps = 20
    total_time = jnp.asarray(0.4, dtype=jnp.float32)
    dt = total_time / number_steps

    def body(mps, time):
        midpoint = time + dt / 2
        return step(mps, build_mpo(jnp.asarray((jnp.sin(midpoint),))), dt), None

    times = jnp.arange(number_steps, dtype=jnp.float32) * dt
    final, _ = jax.jit(lambda value: jax.lax.scan(body, value, xs=times))(state)
    angle = 1.0 - jnp.cos(total_time)
    expected_z = jnp.cos(2 * angle)
    np.testing.assert_allclose(
        np.asarray(tnalg.expectation(final, observable)),
        np.asarray(expected_z),
        atol=5e-5,
    )


def test_spin_one_xxz_tdvp_matches_full_space_evolution(jaxb):
    """A non-qubit operator table follows the same TDVP tensor contract."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(3, 3), bond_dims=(1, 3, 1))
    root_two = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))
    plus = jnp.asarray(
        ((0, root_two, 0), (0, 0, root_two), (0, 0, 0)), dtype=jnp.complex64
    )
    minus = jnp.conj(plus.T)
    sx = (plus + minus) / 2
    sy = (plus - minus) / (2j)
    sz = jnp.diag(jnp.asarray((1.0, 0.0, -1.0), dtype=jnp.complex64))
    table = jnp.stack((jnp.eye(3, dtype=jnp.complex64), sx, sy, sz))
    codes = jnp.asarray(((1, 1), (2, 2), (3, 3), (3, 0), (0, 3)), dtype=jnp.int32)
    mpo_spec, build_mpo = tnalg.compile_mpo(
        codes, spec.physical_indices, operator_table=table
    )
    weights = jnp.asarray((0.6, 0.6, 0.3, -0.2, 0.15), dtype=jnp.float32)
    mpo = build_mpo(weights)
    initial, _ = tnalg.product_state((0, 1), spec=spec, dtype=jnp.complex64)
    state = initial
    time_step = jnp.asarray(0.07, dtype=jnp.float32)
    step = tnalg.make_tdvp_step_from_plan(
        tnalg.prepare_tdvp(spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=9))
    )
    evolved = jax.jit(step)(state, mpo, time_step)
    expected = jsp_linalg.expm(-1j * time_step * _mpo_matrix(mpo, jnp)) @ _statevector(
        initial, jnp
    )
    np.testing.assert_allclose(
        np.asarray(_statevector(evolved, jnp)), np.asarray(expected), atol=3e-4
    )


def _mpo_matrix(mpo, jnp):
    tensors = tuple(site[0] for site in mpo.buffers)
    tensor = tensors[0][0]
    for site in tensors[1:]:
        tensor = jnp.tensordot(tensor, site, axes=(-1, 0))
    tensor = tensor[..., 0]
    nsites = len(tensors)
    axes = tuple(range(0, 2 * nsites, 2)) + tuple(range(1, 2 * nsites, 2))
    dimension = int(np.prod(tuple(site.shape[1] for site in tensors)))
    return jnp.reshape(jnp.transpose(tensor, axes), (dimension, dimension))


def _xyz_codes_and_weights(nsites, jnp):
    codes = []
    weights = []
    for site in range(nsites - 1):
        for code, weight in ((1, 1.0), (2, 0.7), (3, 0.5)):
            row = [0] * nsites
            row[site] = code
            row[site + 1] = code
            codes.append(row)
            weights.append(weight)
    for site in range(nsites):
        for code, weight in ((1, 0.31), (3, 0.17)):
            row = [0] * nsites
            row[site] = code
            codes.append(row)
            weights.append(weight)
    return jnp.asarray(codes, dtype=jnp.int32), jnp.asarray(weights, dtype=jnp.float32)


def test_xyz_mpo_environment_and_variance_match_independent_dense_reference(jaxb):
    """MPO and local effective matvec agree with an independently built XYZ H."""

    tnalg = tc.tnalg
    nsites = 4
    spec = tnalg.MPSSpec.dense(physical_dims=(2,) * nsites, bond_dims=(1, 2, 4, 2, 1))
    codes, weights = _xyz_codes_and_weights(nsites, jnp)
    _, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    mpo = build_mpo(weights)
    exact_hamiltonian = xyz_hamiltonian(nsites)
    np.testing.assert_allclose(
        np.asarray(_mpo_matrix(mpo, jnp)), exact_hamiltonian, atol=2e-5
    )

    mps, _ = tnalg.random_mps(jax.random.key(0), spec=spec, dtype=jnp.complex64)
    mps_tensors = dense_mps_tensors(mps)
    mpo_tensors = dense_mpo_tensors(mpo)
    site = 1
    left = boundary_environment(jnp.result_type(mps_tensors[0], mpo_tensors[0]))
    for position in range(site):
        left = update_left(left, mps_tensors[position], mpo_tensors[position])
    right = build_right_environments(mps_tensors, mpo_tensors)[site + 1]
    center_shape = mps_tensors[site].shape
    vector = jax.random.normal(jax.random.key(1), center_shape, dtype=jnp.float32)
    vector = vector + 1j * jax.random.normal(
        jax.random.key(2), center_shape, dtype=jnp.float32
    )
    embedding = center_embedding(
        tuple(np.asarray(tensor) for tensor in mps_tensors), site
    )
    reference = (
        embedding.conj().T
        @ exact_hamiltonian
        @ embedding
        @ np.asarray(vector).reshape(-1)
    )
    np.testing.assert_allclose(
        np.asarray(local_matvec(left, mpo_tensors[site], right, vector)).reshape(-1),
        reference,
        atol=4e-5,
    )

    statevector = mps_statevector(tuple(np.asarray(tensor) for tensor in mps_tensors))
    statevector = statevector / np.linalg.norm(statevector)
    mean = np.vdot(statevector, exact_hamiltonian @ statevector)
    reference_variance = np.real(
        np.vdot(statevector, exact_hamiltonian @ exact_hamiltonian @ statevector)
        - mean * mean
    )
    np.testing.assert_allclose(
        np.asarray(tnalg.variance(mps, mpo)), reference_variance, atol=6e-5
    )


def test_high_precision_mpo_and_environment_reference(jaxb, highp):
    """The fixed tensor contractions retain complex128 reference accuracy."""

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense(physical_dims=(2, 2), bond_dims=(1, 2, 1))
    codes, _ = _xyz_codes_and_weights(2, jnp)
    weights = jnp.asarray((1.0, 0.7, 0.5, 0.31, 0.17, 0.31, 0.17), dtype=jnp.float64)
    _, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    mpo = build_mpo(weights)
    exact_hamiltonian = xyz_hamiltonian(2)
    np.testing.assert_allclose(
        np.asarray(_mpo_matrix(mpo, jnp)), exact_hamiltonian, atol=1e-10, rtol=1e-9
    )
    mps, _ = tnalg.random_mps(jax.random.key(3), spec=spec, dtype=jnp.complex128)
    mps_tensors = dense_mps_tensors(mps)
    mpo_tensors = dense_mpo_tensors(mpo)
    left = boundary_environment(jnp.result_type(mps_tensors[0], mpo_tensors[0]))
    right = build_right_environments(mps_tensors, mpo_tensors)[1]
    vector = jnp.asarray(
        (((1.0 + 0.2j, -0.3j), (0.4, -0.1 + 0.7j)),), dtype=jnp.complex128
    )
    embedding = center_embedding(tuple(np.asarray(tensor) for tensor in mps_tensors), 0)
    reference = (
        embedding.conj().T
        @ exact_hamiltonian
        @ embedding
        @ np.asarray(vector).reshape(-1)
    )
    np.testing.assert_allclose(
        np.asarray(local_matvec(left, mpo_tensors[0], right, vector)).reshape(-1),
        reference,
        atol=1e-10,
        rtol=1e-9,
    )


def test_n4_xyz_tebd_and_tdvp_follow_independent_full_space_evolution(jaxb):
    """Both dense algorithms evolve a complete N=4 XYZ MPS correctly."""

    tnalg = tc.tnalg
    nsites = 4
    spec = tnalg.MPSSpec.dense(physical_dims=(2,) * nsites, bond_dims=(1, 2, 4, 2, 1))
    codes, weights = _xyz_codes_and_weights(nsites, jnp)
    mpo_spec, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    gate_spec, build_gates = tnalg.compile_tebd_gates(codes, spec.physical_indices)
    mpo = build_mpo(weights)
    generators = build_gates(weights)
    initial, _ = tnalg.random_mps(jax.random.key(7), spec=spec, dtype=jnp.complex64)
    initial_state = initial
    initial_vector = mps_statevector(
        tuple(np.asarray(site[0]) for site in initial.buffers)
    )
    total_time = 0.04
    exact = (
        scipy.linalg.expm(-1j * total_time * xyz_hamiltonian(nsites)) @ initial_vector
    )

    tebd_step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(spec, gate_spec, tnalg.TEBDOptions(order=2))
    )
    time_step = jnp.asarray(total_time / 4, dtype=jnp.float32)
    tebd_final, _ = jax.jit(
        lambda state: jax.lax.scan(
            lambda carry, _: (tebd_step(carry, generators, time_step), None),
            state,
            xs=None,
            length=4,
        )
    )(initial_state)
    tebd_vector = mps_statevector(
        tuple(np.asarray(site[0]) for site in tebd_final.buffers)
    )
    tebd_vector *= np.exp(-1j * np.angle(np.vdot(exact, tebd_vector)))
    np.testing.assert_allclose(tebd_vector, exact, atol=3e-4)

    tdvp_step = tnalg.make_tdvp_step_from_plan(
        tnalg.prepare_tdvp(spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=16))
    )
    tdvp_final = jax.jit(tdvp_step)(
        initial_state, mpo, jnp.asarray(total_time, dtype=jnp.float32)
    )
    tdvp_vector = mps_statevector(
        tuple(np.asarray(site[0]) for site in tdvp_final.buffers)
    )
    tdvp_vector *= np.exp(-1j * np.angle(np.vdot(exact, tdvp_vector)))
    np.testing.assert_allclose(tdvp_vector, exact, atol=5e-4)


def test_n4_xyz_dmrg_matches_independent_ground_energy_and_variance(jaxb):
    """Fixed-work one-site DMRG reaches the complete N=4 dense ground state."""

    tnalg = tc.tnalg
    nsites = 4
    spec = tnalg.MPSSpec.dense(physical_dims=(2,) * nsites, bond_dims=(1, 2, 4, 2, 1))
    codes, weights = _xyz_codes_and_weights(nsites, jnp)
    mpo_spec, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    mpo = build_mpo(weights)
    exact_energy = np.linalg.eigvalsh(xyz_hamiltonian(nsites))[0]
    sweep = tnalg.make_dmrg_sweep_from_plan(
        tnalg.prepare_dmrg(
            spec,
            mpo_spec,
            tnalg.DMRGOptions(krylov_dim=16),
        )
    )

    def optimize(initial):
        return jax.lax.scan(
            lambda state, _: sweep(state, mpo), initial, xs=None, length=8
        )

    for key in (jax.random.key(0), jax.random.key(1)):
        initial, _ = tnalg.random_mps(key, spec=spec, dtype=jnp.complex64)
        final, energies = jax.jit(optimize)(initial)
        np.testing.assert_allclose(np.asarray(energies[-1]), exact_energy, atol=5e-4)
        np.testing.assert_allclose(np.asarray(tnalg.norm(final)), 1.0, atol=2e-4)


def test_grouped_tdvp_matches_dense_with_inhomogeneous_sites(jaxb, highp):

    tnalg = tc.tnalg
    n = 8
    symmetry = tnalg.AbelianSymmetry((2,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,))
    )
    bonds = (
        ({(0,): 1}, {(0,): 1, (1,): 1})
        + ({(0,): 2, (1,): 2},) * (n - 3)
        + ({(0,): 1, (1,): 1}, {(0,): 1})
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,) * n, total_charge=(0,), bond_sectors=bonds
    )
    dense_spec = tnalg.MPSSpec.dense((2,) * n, spec.bond_dims)
    initial, _ = tnalg.random_mps(jax.random.key(4), spec=spec, dtype=jnp.complex128)
    dense_initial, _ = tnalg.as_mps(dense_mps_tensors(initial), spec=dense_spec)
    rows = []
    for site in range(n - 1):
        row = [0] * n
        row[site : site + 2] = [1, 1]
        rows.append(row)
    for site in range(n):
        row = [0] * n
        row[site] = 3
        rows.append(row)
    codes = jnp.asarray(rows, dtype=jnp.int32)
    weights = jnp.linspace(-0.9, 0.7, len(rows))
    values = []
    for current_spec, current in ((spec, initial), (dense_spec, dense_initial)):
        mpo_spec, build = tnalg.compile_mpo(codes, current_spec.physical_indices)
        step = tnalg.make_tdvp_step_from_plan(
            tnalg.prepare_tdvp(current_spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=8))
        )
        state = current
        result = jax.jit(step)(state, build(weights), jnp.asarray(0.013))
        values.append(
            mps_statevector(tuple(np.asarray(a) for a in dense_mps_tensors(result)))
        )
    phase = np.vdot(values[0], values[1])
    np.testing.assert_allclose(values[0] * phase / abs(phase), values[1], atol=1e-9)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_symmetric_tebd_order_and_normalization(jaxb, order, normalize):

    tnalg = tc.tnalg
    spec = _u1_spec(4, 2, tnalg)
    initial, _ = tnalg.random_mps(jax.random.key(5), spec=spec, dtype=jnp.complex64)
    blocks = list(initial.buffers)
    blocks[0] = tuple(2 * block for block in blocks[0])
    initial = _with_buffers(spec, tuple(blocks))
    dense_spec = tnalg.MPSSpec.dense((2,) * 4, spec.bond_dims)
    dense_initial = tnalg.MPSState(
        dense_spec, tuple((a,) for a in dense_mps_tensors(initial))
    )
    codes = jnp.asarray(
        ((1, 1, 0, 0), (2, 2, 0, 0), (0, 1, 1, 0), (0, 2, 2, 0), (3, 0, 0, 0)),
        dtype=jnp.int32,
    )
    groups = jnp.asarray((0, 0, 1, 1, 2), dtype=jnp.int32)
    values = []
    for current_spec, current in ((spec, initial), (dense_spec, dense_initial)):
        gate_spec, build = tnalg.compile_tebd_gates(
            codes, current_spec.physical_indices, coefficient_indices=groups
        )
        step = tnalg.make_tebd_step_from_plan(
            tnalg.prepare_tebd(
                current_spec,
                gate_spec,
                tnalg.TEBDOptions(order=order, normalize=normalize),
            )
        )
        state = current
        result = jax.jit(step)(
            state, build(jnp.asarray((0.7, 1.1, 0.3))), jnp.asarray(0.07)
        )
        values.append(
            mps_statevector(tuple(np.asarray(a) for a in dense_mps_tensors(result)))
        )
    np.testing.assert_allclose(values[0], values[1], atol=3e-5)
    np.testing.assert_allclose(
        np.linalg.norm(values[0]), 1 if normalize else 2, atol=3e-5
    )


@pytest.mark.parametrize("symmetric", [False, True])
def test_joint_site_access_matches_separate_updates_and_gradients(jaxb, symmetric):
    tnalg = tc.tnalg
    spec = _u1_spec(4, 2, tnalg) if symmetric else tnalg.MPSSpec.dense((2,) * 8, chi=3)
    state, _ = tnalg.random_mps(jax.random.key(11), spec=spec, dtype=jnp.complex64)
    block_spec = block_mps_spec(spec)
    buffers = pack_block_buffers(block_spec, tuple(state.buffers))
    for group in prepare_schedule(block_spec).groups:
        joint = site_access(block_spec, group.sites, width=2)
        left, right = group.mps
        for row in range(len(group.sites)):
            expected = left.read(buffers, row) + right.read(buffers, row)
            for block, reference in zip(joint.read(buffers, row), expected):
                np.testing.assert_array_equal(block, reference)

            def update(scale, combined):
                blocks = tuple(scale * block + 0.1j for block in expected)
                if combined:
                    return joint.write(buffers, row, blocks)
                partial = left.write(buffers, row, blocks[: left.count])
                return right.write(partial, row, blocks[left.count :])

            def loss(scale, combined):
                return sum(jnp.real(jnp.vdot(b, b)) for b in update(scale, combined))

            for value, reference in zip(update(0.8, True), update(0.8, False)):
                np.testing.assert_array_equal(value, reference)
            np.testing.assert_allclose(
                jax.grad(loss)(0.8, True), jax.grad(loss)(0.8, False), rtol=2e-6
            )


def test_noncontiguous_bucket_access_roundtrips_selected_slots(jaxb):
    """Non-contiguous bucket reads and writes update only their selected slots."""

    access = BlockAccess(
        2,
        (
            (
                0,
                (2,),
                (0, 1),
                np.asarray(((0, 2), (1, 3)), dtype=np.int32),
                False,
            ),
        ),
    )
    buckets = (jnp.arange(8, dtype=jnp.float32).reshape(4, 2),)
    selected = access.read(buckets, 1)
    np.testing.assert_array_equal(
        np.asarray(selected[0]), np.asarray((2.0, 3.0), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(selected[1]), np.asarray((6.0, 7.0), dtype=np.float32)
    )

    replacement = (
        jnp.asarray((10.0, 11.0), dtype=jnp.float32),
        jnp.asarray((12.0, 13.0), dtype=jnp.float32),
    )
    updated = access.write(buckets, 1, replacement)
    expected = np.asarray(((0.0, 1.0), (10.0, 11.0), (4.0, 5.0), (12.0, 13.0)))
    np.testing.assert_array_equal(np.asarray(updated[0]), expected)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("nsites", [1, 2, 5, 32, 256])
def test_tebd_directional_schedule_bounds_qr_and_returns_center_zero(
    jaxb, monkeypatch, order, nsites
):
    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense((2,) * nsites, chi=3)
    records = []

    def capture(indices):
        records.extend(indices)
        return prepare_program(indices)

    monkeypatch.setattr(tebd_module, "prepare_program", capture)
    tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(
            spec, tnalg.GateSpec((2,) * nsites), tnalg.TEBDOptions(order)
        )
    )
    block_spec = block_mps_spec(spec)
    singles = prepare_schedule(block_spec, width=1)
    pairs = prepare_schedule(block_spec)
    center, qr_count, gate_count = 0, 0, 0
    for code, row, _ in records:
        if code < len(singles.groups):
            continue
        kind, group_index = divmod(code - len(singles.groups), len(pairs.groups))
        site = pairs.groups[group_index].sites[row]
        rightward = kind in (0, 2)
        assert center == site + int(not rightward)
        center = site + int(rightward)
        qr_count += kind >= 2
        gate_count += kind < 2
    assert center == 0
    assert gate_count == nsites - 1 + (nsites // 2 if order == 2 else 0)
    assert qr_count <= (nsites - 1 if order == 1 else max(0, 3 * nsites - 5))


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_tebd_truncation_and_gradients_match_full_state_schmidt_reference(
    jaxb, highp, order, symmetric
):
    tnalg = tc.tnalg
    n = 4 if symmetric else 5
    spec = _u1_spec(n, 2, tnalg) if symmetric else tnalg.MPSSpec.dense((2,) * n, chi=3)

    def statevector(mps):
        tensors = dense_mps_tensors(mps)
        result = tensors[0]
        for tensor in tensors[1:]:
            result = jnp.tensordot(result, tensor, axes=(-1, 0))
        return result.reshape(-1)

    initial, _ = tnalg.random_mps(jax.random.key(17), spec=spec, dtype=jnp.complex128)
    initial_vector = np.asarray(statevector(initial))
    state = initial
    step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(spec, tnalg.GateSpec((2,) * n), tnalg.TEBDOptions(order))
    )
    dt = 0.17
    xx, yy, zz = (np.kron(p, p) for p in (PAULI_X, PAULI_Y, PAULI_Z))

    def generators(weights):
        onsite = tuple(
            weights[0] * (1 + 0.1 * site) * PAULI_Z
            + weights[1] * ((-1) ** site * PAULI_Z if symmetric else PAULI_X)
            for site in range(n)
        )
        bonds = tuple(
            (
                weights[2] * (1 + 0.03 * site) * xx
                + (weights[2] * (1 + 0.03 * site) if symmetric else 0.27) * yy
                + 0.19 * zz
            ).reshape(2, 2, 2, 2)
            for site in range(n - 1)
        )
        return onsite, bonds

    def apply(vector, matrix, sites):
        width = len(sites)
        result = np.tensordot(
            matrix.reshape((2,) * (2 * width)),
            vector.reshape((2,) * n),
            axes=(tuple(range(width, 2 * width)), sites),
        )
        axes = sites + tuple(site for site in range(n) if site not in sites)
        return np.transpose(result, np.argsort(axes)).reshape(-1)

    def reference(weights):
        onsite, bonds = generators(weights)
        onsite_gates = [
            scipy.linalg.expm(-1j * dt * (0.5 if order == 2 else 1) * h) for h in onsite
        ]
        vector = initial_vector
        for site, gate in enumerate(onsite_gates):
            vector = apply(vector, gate, (site,))
        layers = (
            ((range(0, n - 1, 2), 1), (reversed(range(1, n - 1, 2)), 1))
            if order == 1
            else (
                (range(0, n - 1, 2), 0.5),
                (range(1, n - 1, 2), 1),
                (reversed(range(0, n - 1, 2)), 0.5),
            )
        )
        for sites, coefficient in layers:
            for site in sites:
                gate = scipy.linalg.expm(
                    -1j * dt * coefficient * bonds[site].reshape(4, 4)
                )
                vector = apply(vector, gate, (site, site + 1))
                u, s, vh = np.linalg.svd(
                    vector.reshape(2 ** (site + 1), -1), full_matrices=False
                )
                keep = spec.bond_dims[site + 1]
                vector = ((u[:, :keep] * s[:keep]) @ vh[:keep]).reshape(-1)
        if order == 2:
            for site, gate in enumerate(onsite_gates):
                vector = apply(vector, gate, (site,))
        return vector

    weights = jnp.asarray((0.31, 0.23, 0.47))
    result = jax.jit(step)(state, generators(weights), jnp.asarray(dt))
    np.testing.assert_allclose(
        statevector(result), reference(np.asarray(weights)), atol=1e-11
    )
    assert result.spec == spec
    for tensor in dense_mps_tensors(result)[1:]:
        matrix = np.asarray(tensor).reshape(tensor.shape[0], -1)
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-11
        )
    observable = np.linspace(-0.7, 1.3, 2**n)

    def loss(w):
        vector = statevector(step(state, generators(w), jnp.asarray(dt)))
        return jnp.real(jnp.vdot(vector, observable * vector))

    gradient = jax.jit(jax.grad(loss))(weights)
    numerical = []
    for direction in np.eye(3) * 1e-5:
        plus, minus = reference(np.asarray(weights) + direction), reference(
            np.asarray(weights) - direction
        )
        numerical.append(
            np.real(
                np.vdot(plus, observable * plus) - np.vdot(minus, observable * minus)
            )
            / 2e-5
        )
    np.testing.assert_allclose(gradient, numerical, atol=2e-6, rtol=2e-5)


def test_dense_tebd_joint_updates_do_not_copy_the_whole_bulk_bucket(jaxb):
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU HLO copy regression")
    tnalg = tc.tnalg
    n = 32
    spec = tnalg.MPSSpec.dense((2,) * n, chi=4)
    initial, _ = tnalg.random_mps(jax.random.key(19), spec=spec, dtype=jnp.complex64)
    codes, groups, weights = _u1_xxz_codes_and_weights(n, jnp)
    gates, build = tnalg.compile_tebd_gates(
        codes, spec.physical_indices, coefficient_indices=groups
    )
    step = tnalg.make_tebd_step_from_plan(
        tnalg.prepare_tebd(spec, gates, tnalg.TEBDOptions())
    )
    executable = (
        jax.jit(step)
        .lower(
            initial,
            build(weights),
            jnp.asarray(0.02),
        )
        .compile()
    )
    plan = block_bucket_plan(block_mps_spec(spec).site_layouts)
    bulk = max(zip(plan.counts, plan.shapes), key=lambda item: item[0])
    shape = "c64[" + ",".join(map(str, (bulk[0],) + bulk[1])) + "]"
    copies = [
        line
        for line in executable.as_text().splitlines()
        if " copy(" in line and shape in line
    ]
    assert not copies, copies


def test_projected_exponential_derivative_at_degeneracy(jaxb, highp):

    matrix = jnp.diag(jnp.asarray((0.0, 0.0, 2.0, 2.0)))
    direction = jnp.asarray(
        (
            (1.0, 2.0, 0.0, 1.0),
            (2.0, 0.0, -1.0, 3.0),
            (0.0, -1.0, 2.0, 4.0),
            (1.0, 3.0, 4.0, -2.0),
        )
    )
    coefficient = jnp.asarray(-0.17j)
    actual = jax.jvp(
        _projected_exponential, (matrix, coefficient), (direction, jnp.asarray(0.23j))
    )
    expected = jax.jvp(
        lambda a, c: jax.scipy.linalg.expm(c * a)[:, 0],
        (matrix, coefficient),
        (direction, jnp.asarray(0.23j)),
    )
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(value), np.asarray(reference), atol=1e-10)


def test_long_random_mps_is_finite_and_symmetric_norm_stays_blockwise(
    jaxb, monkeypatch
):

    tnalg = tc.tnalg
    n = 128
    symmetry = tnalg.AbelianSymmetry((2,))
    physical = tnalg.SectorIndex.from_basis(
        symmetry=symmetry, basis_charges=((0,), (1,))
    )
    bonds = ({(0,): 1},) + ({(0,): 1, (1,): 1},) * (n - 1) + ({(0,): 1},)
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,) * n, total_charge=(0,), bond_sectors=bonds
    )
    state, _ = tnalg.random_mps(jax.random.key(8), spec=spec, dtype=jnp.complex64)

    def reject_dense(*args, **kwargs):
        raise AssertionError("symmetric norm must not densify")

    monkeypatch.setattr(mps_module, "_dense_tensors", reject_dense)
    np.testing.assert_allclose(np.asarray(tnalg.norm(state)), 1.0, atol=2e-5)
    np.testing.assert_allclose(np.asarray(tnalg.overlap(state, state)), 1.0, atol=2e-5)


def test_one_dimensional_krylov_exponential(jaxb):

    vector = jnp.asarray((1j,), dtype=jnp.complex64)
    result, report = jax.jit(
        lambda x: expm_action_hermitian(lambda v: 2 * v, x, -0.1j, 1)
    )(vector)
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(vector * jnp.exp(-0.2j)), atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(report["residual"]), 0.0)


def test_krylov_without_reorthogonalization_matches_dense_reference(jaxb, highp):
    """The no-reorthogonalization Krylov path remains correct on a full spectrum."""

    matrix = jnp.diag(jnp.asarray((0.4, 1.1, 2.3, 3.7), dtype=jnp.float64)).astype(
        jnp.complex128
    )
    vector = jnp.asarray((1.0, 0.3j, -0.2, 0.7j), dtype=jnp.complex128)
    coefficient = jnp.asarray(-0.17j, dtype=jnp.complex128)
    evolve = jax.jit(
        lambda value: expm_action_hermitian(
            lambda direction: matrix @ direction,
            value,
            coefficient,
            max_dim=4,
            reorthogonalize=False,
        )
    )
    actual, report = evolve(vector)
    expected = scipy.linalg.expm(np.asarray(coefficient * matrix)) @ np.asarray(vector)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-11, rtol=1e-11)
    assert bool(report["finite"])

    solve = jax.jit(
        lambda value: lowest_eigenvector_hermitian(
            lambda direction: matrix @ direction,
            value,
            max_dim=4,
            reorthogonalize=False,
        )
    )
    candidate, eigenvalue, eigen_report = solve(vector)
    np.testing.assert_allclose(np.asarray(eigenvalue), 0.4, atol=1e-11)
    np.testing.assert_allclose(np.asarray(eigen_report["residual"]), 0.0, atol=1e-11)
    np.testing.assert_allclose(np.abs(np.asarray(candidate[0])), 1.0, atol=1e-11)
    np.testing.assert_allclose(np.asarray(candidate[1:]), 0.0, atol=1e-11)


@pytest.mark.parametrize("evolution", [False, True])
def test_fixed_krylov_traces_one_matvec_body(jaxb, evolution):

    calls = []

    def matvec(vector):
        calls.append(None)
        return jnp.arange(1, 5) * vector

    def solve(vector):
        if evolution:
            return expm_action_hermitian(matvec, vector, -0.1j, 4)
        return lowest_eigenvector_hermitian(matvec, vector, 4, return_report=False)

    jax.make_jaxpr(solve)(jnp.ones((4,), dtype=jnp.complex64))
    assert len(calls) == 1


@pytest.mark.parametrize("pattern", [(0,), (0, 1), (0, 1, 2, 1)])
def test_periodic_sweep_program_preserves_order_and_gradients(jaxb, pattern):

    codes = (2,) + pattern * 5 + (1, 0)
    indices = np.column_stack((codes, np.arange(len(codes)))).astype(np.int32)
    program = prepare_program(indices)
    kernels = tuple(
        lambda value, record, code=code: value * (1 + 0.01 * code) + record[0]
        for code in range(3)
    )

    def reference(value):
        for code, row in indices:
            value = kernels[code](value, (row,))
        return value

    compiled = jax.jit(
        jax.value_and_grad(lambda value: run_program(program, kernels, value))
    )
    actual = compiled(jnp.asarray(0.7))
    expected = jax.value_and_grad(reference)(jnp.asarray(0.7))
    for value, target in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(value), np.asarray(target), rtol=2e-6)


def _u1_spec(nsites, total_charge, tnalg):
    if (nsites, total_charge) != (4, 2):
        raise ValueError("the fixed complete U(1) test layout is defined for N=4, Q=2")
    return tnalg.MPSSpec.u1(nsites, total_charge=total_charge, chi=8, charge_sectors=3)


def _u1_xxz_codes_and_weights(nsites, jnp):
    rows = []
    coefficient_indices = []
    weights = []
    coefficient = 0
    for site in range(nsites - 1):
        for code in (1, 2):
            row = [0] * nsites
            row[site] = code
            row[site + 1] = code
            rows.append(row)
            coefficient_indices.append(coefficient)
        weights.append(1.0)
        coefficient += 1
        row = [0] * nsites
        row[site] = 3
        row[site + 1] = 3
        rows.append(row)
        coefficient_indices.append(coefficient)
        weights.append(0.7)
        coefficient += 1
    for site in range(nsites):
        row = [0] * nsites
        row[site] = 3
        rows.append(row)
        coefficient_indices.append(coefficient)
        weights.append(0.23 * (-1) ** site)
        coefficient += 1
    return (
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(coefficient_indices, dtype=jnp.int32),
        jnp.asarray(weights, dtype=jnp.float32),
    )


def _symmetric_statevector(mps):

    return mps_statevector(
        tuple(np.asarray(tensor) for tensor in dense_mps_tensors(mps))
    )


def _phase_aligned(candidate, reference):
    return candidate * np.exp(-1j * np.angle(np.vdot(reference, candidate)))


def test_dense_xxz_mpo_has_five_channels_at_256_sites(jaxb):
    """Nearest-neighbor strings share fixed channels, independent of chain length."""

    codes, groups, weights = _u1_xxz_codes_and_weights(256, jnp)
    spec, build = tc.tnalg.compile_mpo(codes, (2,) * 256, coefficient_indices=groups)
    assert spec.symmetry is None
    assert spec.bond_dims == (1,) + (5,) * 255 + (1,)
    mpo = build(weights)
    assert sum(site[0].size for site in mpo.buffers) == 25440


@pytest.mark.parametrize("symmetric", [False, True])
def test_mpo_preserves_small_static_operator_entries(jaxb, highp, symmetric):

    tnalg = tc.tnalg
    physical = (
        tnalg.SectorIndex.from_basis(
            symmetry=tnalg.AbelianSymmetry((0,)), basis_charges=((0,), (1,))
        )
        if symmetric
        else 2
    )
    table = np.stack([PAULI_I, 1e-14 * PAULI_Z])
    _, build = tnalg.compile_mpo(np.asarray([[1]]), (physical,), operator_table=table)
    mpo = build(jnp.asarray([1e14]))
    legacy = tnalg.to_tn_mpo(mpo, allow_dense=True)
    np.testing.assert_allclose(
        np.asarray(legacy.tensors[0]).reshape(2, 2), PAULI_Z, atol=1e-12
    )


def test_single_block_contraction_uses_true_dimensions_without_padding(
    jaxb, highp, monkeypatch
):

    left = jnp.arange(15, dtype=jnp.float64).reshape(3, 5)
    right = jnp.arange(35, dtype=jnp.float64).reshape(5, 7)
    plan = block_ops.prepare_contraction(
        "ab,bc->ac", (((0, 0),), ((3, 5),)), (((0, 0),), ((5, 7),))
    )

    def forbidden(*args):
        raise AssertionError("a single contraction must not pad or batch")

    monkeypatch.setattr(jnp, "pad", forbidden)
    function = jax.jit(lambda a, b: plan((a,), (b,))[0])
    np.testing.assert_allclose(function(left, right), left @ right, atol=1e-12)
    actual = jax.grad(lambda a: jnp.sum(function(a, right)))(left)
    expected = jnp.broadcast_to(jnp.sum(right, axis=1), left.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dense_bucket_storage_preserves_exact_chi_and_mpo_dimensions(jaxb, monkeypatch):

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense((2,) * 6, chi=3)
    state, _ = tnalg.product_state((0, 1, 0, 1, 0, 1), spec=spec, dtype=jnp.complex64)
    codes, groups, weights = _u1_xxz_codes_and_weights(6, jnp)
    mpo_spec, build = tnalg.compile_mpo(
        codes, spec.physical_indices, coefficient_indices=groups
    )
    mpo = build(weights)
    mps_blocks, mpo_blocks = block_mps_spec(spec), block_mpo_spec(mpo_spec)
    plans = [block_bucket_plan(s.site_layouts) for s in (mps_blocks, mpo_blocks)]
    plans.append(environment_bucket_plan(mps_blocks, mpo_blocks))
    for plan in plans:
        assert all(
            plan.shapes[bucket] == shape
            for (bucket, _), shape in zip(plan.positions, plan.true_shapes)
        )

    def forbidden(*args):
        raise AssertionError("dense bucket packing must not pad")

    monkeypatch.setattr(jnp, "pad", forbidden)
    for s, buffers in ((mps_blocks, state.buffers), (mpo_blocks, mpo.buffers)):
        packed = pack_block_buffers(s, buffers)
        assert sum(a.size for a in packed) == sum(
            a.size for site in buffers for a in site
        )
        restored = unpack_block_buffers(s, packed)
        for actual, expected in zip(restored, buffers):
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
    assert state.spec.bond_dims == (1, 2, 3, 3, 3, 2, 1)


def test_symmetric_numeric_paths_do_not_densify_implicitly(jaxb, highp, monkeypatch):

    tnalg = tc.tnalg
    spec = _u1_spec(4, 2, tnalg)
    state, _ = tnalg.random_mps(jax.random.key(12), spec=spec, dtype=jnp.complex128)
    vector = _symmetric_statevector(state)
    codes, groups, weights = _u1_xxz_codes_and_weights(4, jnp)
    mpo_spec, build = tnalg.compile_mpo(
        codes, spec.physical_indices, coefficient_indices=groups
    )
    mpo = build(weights.astype(jnp.float64))
    h = u1_xxz_hamiltonian(4)
    mean = np.vdot(vector, h @ vector)
    expected_variance = np.vdot(h @ vector, h @ vector) - mean**2

    def forbidden(*args):
        raise AssertionError("a symmetric numerical path must not densify")

    monkeypatch.setattr(mps_module, "_dense_tensors", forbidden)
    monkeypatch.setattr(mpo_module, "_dense_tensors", forbidden)
    monkeypatch.setattr(mpo_module, "_dense_mps_tensors", forbidden)
    monkeypatch.setattr(mpo_module, "to_dense", forbidden)
    monkeypatch.setattr(tensor_module, "to_dense", forbidden)
    np.testing.assert_allclose(tnalg.norm(state), 1.0, atol=1e-12)
    np.testing.assert_allclose(tnalg.expectation(state, mpo), mean, atol=1e-7)
    np.testing.assert_allclose(
        jax.jit(tnalg.variance)(state, mpo), expected_variance, atol=1e-6
    )
    for algorithm in ("tdvp", "dmrg"):
        if algorithm == "tdvp":
            step = tnalg.make_tdvp_step_from_plan(
                tnalg.prepare_tdvp(spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=8))
            )
            result = jax.jit(step)(state, mpo, 0.01)
        else:
            sweep = tnalg.make_dmrg_sweep_from_plan(
                tnalg.prepare_dmrg(spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=8))
            )
            result, _ = jax.jit(sweep)(state, mpo)
        np.testing.assert_allclose(tnalg.norm(result), 1.0, atol=1e-10)
    dense = tnalg.MPSSpec.dense((2,) * 4, chi=4)
    with pytest.raises(ValueError, match="implicitly densify"):
        tnalg.as_mps(state, spec=dense)
    dense_mpo, dense_build = tnalg.compile_mpo(
        codes, dense.physical_indices, coefficient_indices=groups
    )
    assert dense_mpo.symmetry is None
    for observable in (tnalg.expectation, tnalg.variance):
        with pytest.raises(ValueError, match="symmetr"):
            observable(state, dense_build(weights))


def test_symmetric_bucket_storage_does_not_round_sector_quotas(jaxb):

    tnalg = tc.tnalg
    physical = tnalg.SectorIndex.from_basis(
        symmetry=tnalg.AbelianSymmetry((0,)), basis_charges=((0,), (1,))
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,) * 6,
        total_charge=(3,),
        bond_sectors=(
            {(0,): 1},
            {(0,): 1, (1,): 1},
            {(0,): 1, (1,): 2, (2,): 1},
            {(0,): 1, (1,): 3, (2,): 3, (3,): 1},
            {(1,): 1, (2,): 2, (3,): 1},
            {(2,): 1, (3,): 1},
            {(3,): 1},
        ),
    )
    codes, groups, weights = _u1_xxz_codes_and_weights(6, jnp)
    mpo_spec, build = tnalg.compile_mpo(
        codes, spec.physical_indices, coefficient_indices=groups
    )
    state, _ = tnalg.random_mps(jax.random.key(33), spec=spec, dtype=jnp.complex64)
    mpo = build(weights)
    for value in (state, mpo):
        true_elements = sum(
            np.prod(shape)
            for layout in value.spec.site_layouts
            for shape in layout.block_shapes
        )
        assert sum(a.size for a in jax.tree_util.tree_leaves(value)) == true_elements
    plans = [block_bucket_plan(s.site_layouts) for s in (spec, mpo_spec)]
    plans.append(environment_bucket_plan(spec, mpo_spec))
    for plan in plans:
        assert all(
            plan.shapes[bucket] == shape
            for (bucket, _), shape in zip(plan.positions, plan.true_shapes)
        )


def test_dense_automaton_preserves_tied_duplicate_zero_coefficients_and_gradients(
    jaxb, highp
):
    """Shared tails must neither duplicate paths nor remove zero-valued channels."""

    codes = np.asarray(
        [[1, 1, 0], [1, 1, 0], [2, 2, 0], [0, 1, 1], [3, 0, 3], [0, 0, 0], [0, 0, 3]]
    )
    groups = np.asarray([0, 1, 0, 2, 3, 4, 2])
    _, build = tc.tnalg.compile_mpo(codes, (2,) * 3, coefficient_indices=groups)
    table = (PAULI_I, PAULI_X, PAULI_Y, PAULI_Z)
    matrices = jnp.asarray(
        np.stack([kron_sites([table[c] for c in row]) for row in codes])
    )
    weights = jnp.asarray([0.0, 0.7, -0.2, 1.3, 0.4])
    function = jax.jit(lambda w: _mpo_matrix(build(w), jnp))
    reference = lambda w: jnp.einsum("t,tij->ij", w[groups], matrices)
    for values in (weights, weights + 0.3):
        np.testing.assert_allclose(function(values), reference(values), atol=1e-12)
    tangent = jnp.asarray([1.0, -0.2, 0.4, -0.1, 0.8])
    np.testing.assert_allclose(
        jax.jvp(function, (weights,), (tangent,))[1], reference(tangent), atol=1e-12
    )
    loss = lambda w: jnp.real(jnp.vdot(function(w), matrices[0]))
    expected_loss = lambda w: jnp.real(jnp.vdot(reference(w), matrices[0]))
    np.testing.assert_allclose(
        jax.grad(loss)(weights), jax.grad(expected_loss)(weights), atol=1e-12
    )


def test_long_symmetric_strings_validate_without_many_body_dense_tensors(jaxb):

    tnalg = tc.tnalg
    physical = tnalg.SectorIndex.from_basis(
        symmetry=tnalg.AbelianSymmetry((0,)), basis_charges=((0,), (1,))
    )
    codes = np.full((1, 256), 3, dtype=np.int32)
    spec, build = tnalg.compile_mpo(codes, (physical,) * 256)
    assert max(spec.bond_dims) <= 3
    assert all(
        np.all(np.isfinite(b)) for site in build(jnp.ones(1)).buffers for b in site
    )
    codes = np.zeros((2, 256), dtype=np.int32)
    codes[0, [0, -1]] = 1
    codes[1, [0, -1]] = 2
    tnalg.compile_mpo(
        codes, (physical,) * 256, coefficient_indices=np.zeros(2, np.int32)
    )
    with pytest.raises(ValueError, match="breaks"):
        tnalg.compile_mpo(codes[:1], (physical,) * 256)
    table = np.stack([PAULI_I, PAULI_X, -PAULI_X, np.zeros((2, 2))])
    # A zero operator and exact cancellation are conserving, even on long support.
    tnalg.compile_mpo(
        np.full((1, 40), 3, np.int32), (physical,) * 40, operator_table=table
    )
    tnalg.compile_mpo(
        np.asarray([[1], [2]]),
        (physical,),
        operator_table=table,
        coefficient_indices=np.zeros(2, np.int32),
    )


@pytest.mark.parametrize("compiler", ["compile_mpo", "compile_tebd_gates"])
def test_operator_compilers_reject_invalid_physical_indices(jaxb, compiler):

    physical = tc.tnalg.SectorIndex.from_basis(
        symmetry=tc.tnalg.AbelianSymmetry((0,)), basis_charges=((0,), (1,))
    )
    compile_ = getattr(tc.tnalg, compiler)
    for indices in ((object(), object()), (2, physical), (2.0, 2.0)):
        with pytest.raises(TypeError, match="physical_indices"):
            compile_(np.zeros((1, 2), np.int32), indices)
    for indices in ((), (0, 2), (physical, physical.dual())):
        with pytest.raises(ValueError):
            compile_(np.zeros((1, len(indices)), np.int32), indices)


@pytest.mark.parametrize("algorithm", ["tdvp", "dmrg", "tebd"])
def test_dense_sweeps_trace_shared_kernels_not_one_per_site(
    jaxb, monkeypatch, algorithm
):
    """Whole-sweep JIT traces a bounded number of local bodies for repeated bulk layouts."""

    tnalg = tc.tnalg
    module = importlib.import_module("tensorcircuit.tnalg.algorithms." + algorithm)
    name = {
        "tdvp": "_block_mpo_local_exponential",
        "dmrg": "_lowest_block_mpo_symmetric_blocks",
        "tebd": "apply_two_site_matrix",
    }[algorithm]
    original = getattr(module, name)
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, observed)
    counts = []
    for n in (32, 256):
        spec = tnalg.MPSSpec.dense((2,) * n, chi=4)
        mps, _ = tnalg.product_state((0, 1) * (n // 2), spec=spec, dtype=jnp.complex64)
        codes, groups, weights = _u1_xxz_codes_and_weights(n, jnp)
        calls.clear()
        if algorithm == "tebd":
            gates, build = tnalg.compile_tebd_gates(
                codes, spec.physical_indices, coefficient_indices=groups
            )
            function = tnalg.make_tebd_step_from_plan(
                tnalg.prepare_tebd(spec, gates, tnalg.TEBDOptions())
            )
            inputs = (
                mps,
                build(weights),
                jnp.asarray(0.02),
            )
        else:
            mpo, build = tnalg.compile_mpo(
                codes, spec.physical_indices, coefficient_indices=groups
            )
            if algorithm == "tdvp":
                function = tnalg.make_tdvp_step_from_plan(
                    tnalg.prepare_tdvp(spec, mpo, tnalg.TDVPOptions(krylov_dim=4))
                )
                inputs = (
                    mps,
                    build(weights),
                    jnp.asarray(0.02),
                )
            else:
                function = tnalg.make_dmrg_sweep_from_plan(
                    tnalg.prepare_dmrg(spec, mpo, tnalg.DMRGOptions(krylov_dim=4))
                )
                inputs = (mps, build(weights))
        jax.make_jaxpr(function)(*inputs)
        counts.append(len(calls))
    assert counts[0] > 0
    assert counts[0] == counts[1]


def test_compression_restores_center_zero_and_matches_tt_svd(jaxb, highp):

    tnalg = tc.tnalg
    large = tnalg.MPSSpec.dense((2,) * 6, chi=8)
    target = tnalg.MPSSpec.dense((2,) * 6, chi=2)
    source, _ = tnalg.random_mps(jax.random.key(31), spec=large, dtype=jnp.complex128)
    vector = np.asarray(_statevector(source, jnp))
    remainder = vector
    left = 1
    reference_tensors, discarded = [], []
    for cut in range(1, 6):
        u, s, vh = np.linalg.svd(remainder.reshape(left * 2, -1), full_matrices=False)
        keep = target.bond_dims[cut]
        reference_tensors.append(u[:, :keep].reshape(left, 2, keep))
        discarded.append(np.sum(s[keep:] ** 2))
        remainder = s[:keep, None] * vh[:keep]
        left = keep
    reference_tensors.append(remainder.reshape(left, 2, 1))
    result, report = tnalg.as_mps(source, spec=target, truncate=True)
    actual = np.asarray(_statevector(result, jnp))
    np.testing.assert_allclose(actual, mps_statevector(reference_tensors), atol=1e-11)
    np.testing.assert_allclose(report["discarded_weight_abs"], discarded, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(vector - actual) ** 2, sum(discarded), atol=1e-12
    )
    for site in result.buffers[1:]:
        matrix = np.asarray(site[0]).reshape(site[0].shape[0], -1)
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-12
        )
    np.testing.assert_allclose(
        report["output_norm"], np.linalg.norm(actual), atol=1e-12
    )
    # Canonical import must not change the subsequent projected evolution.
    canonical, _ = tnalg.as_mps(tuple(site[0] for site in result.buffers), spec=target)
    codes, weights = _xyz_codes_and_weights(6, jnp)
    mpo_spec, build = tnalg.compile_mpo(codes, target.physical_indices)
    step = jax.jit(
        tnalg.make_tdvp_step_from_plan(
            tnalg.prepare_tdvp(target, mpo_spec, tnalg.TDVPOptions(krylov_dim=8))
        )
    )
    outputs = [
        step(
            m,
            build(weights.astype(jnp.float64)),
            jnp.asarray(0.03),
        )
        for m in (result, canonical)
    ]
    vectors = [np.asarray(_statevector(m, jnp)) for m in outputs]
    np.testing.assert_allclose(
        _phase_aligned(vectors[0], vectors[1]), vectors[1], atol=1e-10
    )


def test_legacy_exports_restore_user_basis_and_check_budget_first(
    jaxb, highp, monkeypatch
):

    tnalg = tc.tnalg
    physical = tnalg.SectorIndex.from_basis(
        symmetry=tnalg.AbelianSymmetry((0,)), basis_charges=((1,), (0,))
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,) * 2,
        total_charge=(2,),
        bond_sectors=({(0,): 1}, {(1,): 1}, {(2,): 1}),
    )
    state, _ = tnalg.product_state((0, 0), spec=spec, dtype=jnp.complex128)
    legacy = tnalg.to_tn_mps(state, allow_dense=True)
    np.testing.assert_allclose(
        mps_statevector(legacy.tensors), [1, 0, 0, 0], atol=1e-12
    )
    circuit = tnalg.to_mpscircuit(state, allow_dense=True)
    np.testing.assert_allclose(
        np.asarray(circuit.state()).reshape(-1), [1, 0, 0, 0], atol=1e-12
    )
    _, build = tnalg.compile_mpo(
        np.asarray([[3, 0], [1, 1], [2, 2]]),
        spec.physical_indices,
        coefficient_indices=np.asarray([0, 1, 1]),
    )
    mpo = build(jnp.asarray([0.3, 0.7]))
    with pytest.raises(ValueError, match="allow_dense"):
        tnalg.to_tn_mpo(mpo)
    legacy_mpo = tnalg.to_tn_mpo(mpo, allow_dense=True)
    dense_spec = tnalg.MPOSpec(mpo.spec.physical_dims, mpo.spec.bond_dims)
    restored = tnalg.as_mpo(
        legacy_mpo, spec=dense_spec, axis_order=("left", "right", "out", "in")
    )
    expected = 0.3 * np.kron(PAULI_Z, PAULI_I) + 0.7 * (
        np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y)
    )
    np.testing.assert_allclose(_mpo_matrix(restored, jnp), expected, atol=1e-12)

    def forbidden(*args):
        raise AssertionError("densification preceded the memory guard")

    monkeypatch.setattr(interop, "_dense_tensors", forbidden)
    monkeypatch.setattr(interop, "_dense_mpo_tensors", forbidden)
    for export in (tnalg.to_tn_mps, tnalg.to_mpscircuit):
        with pytest.raises(ValueError, match="max_elements"):
            export(state, allow_dense=True, max_elements=1)
    with pytest.raises(ValueError, match="max_elements"):
        tnalg.to_tn_mpo(mpo, allow_dense=True, max_elements=1)


def test_symtensor_inner_stays_block_sparse_and_differentiable(
    jaxb, highp, monkeypatch
):

    tnalg = tc.tnalg
    state, _ = tnalg.random_mps(
        jax.random.key(22), spec=_u1_spec(4, 2, tnalg), dtype=jnp.complex128
    )
    tensor = tnalg.site_view(state, 1)
    expected = jnp.vdot(tnalg.to_dense(tensor), tnalg.to_dense(tensor))

    def forbidden(*args):
        raise AssertionError("inner must not reconstruct dense tensors")

    monkeypatch.setattr(tensor_module, "to_dense", forbidden)
    loss = lambda scale: jnp.real(
        tnalg.inner(
            tensor,
            tnalg.SymTensor(tensor.layout, tuple(scale * a for a in tensor.buffers)),
        )
    )
    value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.7))
    np.testing.assert_allclose(value, 0.7 * expected, atol=1e-12)
    np.testing.assert_allclose(gradient, expected, atol=1e-12)


def test_legacy_export_handles_non_involutive_qutrit_permutation(jaxb, highp):

    tnalg = tc.tnalg
    physical = tnalg.SectorIndex.from_basis(
        symmetry=tnalg.AbelianSymmetry((0,)), basis_charges=((2,), (0,), (1,))
    )
    spec = tnalg.MPSSpec.from_sectors(
        physical_indices=(physical,),
        total_charge=(2,),
        bond_sectors=({(0,): 1}, {(2,): 1}),
    )
    state, _ = tnalg.product_state((0,), spec=spec, dtype=jnp.complex128)
    np.testing.assert_allclose(
        mps_statevector(tnalg.to_tn_mps(state, allow_dense=True).tensors),
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )
    operator = np.diag([2.0, 3.0, 5.0])
    _, build = tnalg.compile_mpo(
        np.asarray([[1]]), (physical,), operator_table=np.stack([np.eye(3), operator])
    )
    exported = tnalg.to_tn_mpo(build(jnp.ones(1)), allow_dense=True)
    np.testing.assert_allclose(
        np.asarray(exported.tensors[0]).reshape(3, 3), operator, atol=1e-12
    )


def test_krylov_breakdown_derivative_converges_with_budget(jaxb, highp):

    def derivative(budget):
        def action(coupling):
            h = jnp.asarray([[0.0, coupling], [coupling, 100.0]])
            return expm_action_hermitian(
                lambda x: h @ x,
                jnp.asarray([1.0, 0.0], dtype=jnp.complex128),
                -0.5j,
                budget,
            )[0]

        return jax.jit(lambda: jax.jvp(action, (0.0,), (1.0,))[1])()

    reference = np.asarray([0.0, (np.exp(-50j) - 1) / 100])
    coarse, fine = derivative(4), derivative(32)
    assert np.linalg.norm(np.asarray(fine) - reference) < np.linalg.norm(
        np.asarray(coarse) - reference
    )
    np.testing.assert_allclose(fine, reference, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("algorithm", ["tdvp", "dmrg"])
def test_mps_mpo_plans_reject_mixed_symmetry_and_physical_basis(jaxb, algorithm):

    tnalg = tc.tnalg
    spec = _u1_spec(4, 2, tnalg)
    prepare = getattr(tnalg, "prepare_" + algorithm)
    options = getattr(tnalg, algorithm.upper() + "Options")()
    dense_mpo, _ = tnalg.compile_mpo(np.asarray([[3, 0, 0, 0]]), (2,) * 4)
    with pytest.raises(ValueError, match="symmetr"):
        prepare(spec, dense_mpo, options)
    physical = tnalg.SectorIndex.from_basis(
        symmetry=spec.symmetry, basis_charges=((1,), (0,))
    )
    reversed_mpo, _ = tnalg.compile_mpo(np.asarray([[3, 0, 0, 0]]), (physical,) * 4)
    with pytest.raises(ValueError, match="basis"):
        prepare(spec, reversed_mpo, options)


def test_tdvp_gradient_at_zero_coupling_matches_linear_response(jaxb, highp):

    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense((2,), chi=1)
    initial, _ = tnalg.product_state((0,), spec=spec, dtype=jnp.complex128)
    mpo, build = tnalg.compile_mpo(np.asarray([[1]]), spec.physical_indices)
    step = tnalg.make_tdvp_step_from_plan(
        tnalg.prepare_tdvp(spec, mpo, tnalg.TDVPOptions(krylov_dim=4))
    )

    def observable(coupling):
        state = initial
        result = step(state, build(coupling[None]), jnp.asarray(0.1))
        vector = _statevector(result, jnp)
        return jnp.real(jnp.vdot(vector, jnp.asarray(PAULI_Y) @ vector))

    for coupling in (0.0, 0.2):
        value, gradient = jax.jit(jax.value_and_grad(observable))(jnp.asarray(coupling))
        np.testing.assert_allclose(value, -np.sin(0.2 * coupling), atol=1e-12)
        np.testing.assert_allclose(gradient, -0.2 * np.cos(0.2 * coupling), atol=1e-12)


@pytest.mark.parametrize("case", ["zero_h", "eigenvector", "degenerate", "zero_vector"])
def test_krylov_breakdown_jvp_and_vjp_match_dense_exponential(jaxb, highp, case):
    """Differentiate H, initial vector, and time, including complementary eigenspaces."""

    spectra = {
        "zero_h": [0.0, 0.0, 0.0, 0.0],
        "eigenvector": [2.0, -3.0, 7.0, 11.0],
        "degenerate": [2.0, 2.0, -1.0, -1.0],
        "zero_vector": [2.0, -3.0, 7.0, 11.0],
    }
    matrix = jnp.diag(jnp.asarray(spectra[case])).astype(jnp.complex128)
    vector = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)
    if case == "zero_vector":
        vector = jnp.zeros_like(vector)
    elif case == "degenerate":
        vector = jnp.asarray([1.0, 0.3j, 0.2, -0.4j])
    rng = np.random.default_rng(21)
    tangent = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    tangent = jnp.asarray((tangent + tangent.conj().T) / 2)
    dv = jnp.asarray(rng.normal(size=4) + 1j * rng.normal(size=4))
    primals = (matrix, vector, jnp.asarray(-0.3j))
    tangents = (tangent, dv, jnp.asarray(0.1j))
    function = lambda h, v, c: expm_action_hermitian(lambda x: h @ x, v, c, max_dim=4)[
        0
    ]
    reference = lambda h, v, c: expm(c * h) @ v
    actual = jax.jit(lambda *p: jax.jvp(function, p, tangents))(*primals)
    expected = jax.jvp(reference, primals, tangents)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10)
    # Probe a real scalar path so reverse mode checks the same Hermitian tangent space.
    loss = lambda x: jnp.real(
        jnp.vdot(dv, function(*(p + x * t for p, t in zip(primals, tangents))))
    )
    exact_loss = lambda x: jnp.real(
        jnp.vdot(dv, reference(*(p + x * t for p, t in zip(primals, tangents))))
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(0.0), jax.grad(exact_loss)(0.0), atol=2e-10, rtol=2e-10
    )


def test_charge_window_and_sweep_groups_scale_to_256(jaxb):
    """The U(1) example keeps its fixed layout and schedule at long length."""

    counts = []
    for n in (32, 256):
        charge = n // 2
        spec = tc.tnalg.MPSSpec.u1(n, total_charge=charge, chi=64, charge_sectors=8)
        assert max(spec.bond_dims) <= 64
        rows, groups = [], []
        for site in range(n - 1):
            for code in (1, 2):
                row = [0] * n
                row[site] = row[site + 1] = code
                rows.append(row)
                groups.append(2 * site)
            row = [0] * n
            row[site] = row[site + 1] = 3
            rows.append(row)
            groups.append(2 * site + 1)
        codes, groups = jnp.asarray(rows, jnp.int32), jnp.asarray(groups, jnp.int32)
        mpo_spec, _ = tc.tnalg.compile_mpo(
            codes, spec.physical_indices, coefficient_indices=groups
        )
        counts.append(len(prepare_schedule(spec, mpo_spec).groups))
    assert counts[0] == counts[1]


@pytest.mark.parametrize("algorithm", ["tebd", "tdvp", "dmrg"])
def test_dense_algorithms_match_tenpy_without_symmetry(jaxb, highp, algorithm):
    """Optional TeNPy reference for the three dense algorithms."""

    pytest.importorskip("tenpy")
    from tenpy.algorithms import dmrg, tebd
    from tenpy.algorithms.tdvp import SingleSiteTDVPEngine
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS

    tnalg = tc.tnalg
    n = 4
    spec = tnalg.MPSSpec.dense((2,) * n, chi=4)
    rows, weights = [], []
    for site in range(n - 1):
        for code, weight in ((1, 0.25), (2, 0.25), (3, 0.175)):
            row = [0] * n
            row[site] = row[site + 1] = code
            rows.append(row)
            weights.append(weight)
    codes, weights = np.asarray(rows), jnp.asarray(weights)
    mpo_spec, build = tnalg.compile_mpo(codes, spec.physical_indices)
    mpo = build(weights)
    initial, _ = tnalg.random_mps(jax.random.key(17), spec=spec, dtype=jnp.complex128)
    model = SpinChain(
        {
            "L": n,
            "S": 0.5,
            "bc_MPS": "finite",
            "conserve": None,
            "Jx": 1.0,
            "Jy": 1.0,
            "Jz": 0.7,
        }
    )

    def export(state):
        return MPS.from_Bflat(
            model.lat.mps_sites(),
            [np.asarray(t).transpose(1, 0, 2) for t in tnalg.to_tn_mps(state).tensors],
            bc="finite",
            form=None,
        )

    reference = export(initial)
    np.testing.assert_array_equal(reference.chi, spec.bond_dims[1:-1])
    options = {
        "dt": 0.02,
        "N_steps": 1,
        "order": 2,
        "trunc_params": {
            "chi_max": 4,
            "svd_min": np.finfo(float).tiny,
            "trunc_cut": None,
        },
    }
    evolution = initial
    if algorithm == "tebd":
        gate_spec, build_gates = tnalg.compile_tebd_gates(codes, spec.physical_indices)
        step = tnalg.make_tebd_step_from_plan(
            tnalg.prepare_tebd(spec, gate_spec, tnalg.TEBDOptions())
        )
        result = jax.jit(step)(evolution, build_gates(weights), 0.02)
        tebd.TEBDEngine(reference, model, options).run()
    elif algorithm == "tdvp":
        step = tnalg.make_tdvp_step_from_plan(
            tnalg.prepare_tdvp(spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=16))
        )
        result = jax.jit(step)(evolution, mpo, 0.02)
        SingleSiteTDVPEngine(
            reference,
            model,
            {
                "dt": 0.02,
                "N_steps": 1,
                "lanczos_params": {
                    "N_min": 2,
                    "N_max": 16,
                    "P_tol": 1e-14,
                    "reortho": True,
                },
            },
        ).run()
    else:
        step = tnalg.make_dmrg_sweep_from_plan(
            tnalg.prepare_dmrg(spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=16))
        )
        result, _ = jax.jit(step)(initial, mpo)
        options = {
            "diag_method": "lanczos",
            "lanczos_params": {
                "N_min": 2,
                "N_max": 16,
                "P_tol": 1e-14,
                "E_tol": 1e-12,
                "reortho": True,
            },
            "trunc_params": options["trunc_params"],
        }
        engine = dmrg.SingleSiteDMRGEngine(reference, model, options)
        assert engine.mixer is None
        engine.sweep()
    actual = export(result)
    np.testing.assert_allclose(
        model.H_MPO.expectation_value(actual),
        model.H_MPO.expectation_value(reference),
        atol=1e-9,
        rtol=0,
    )
    fidelity = abs(actual.overlap(reference)) ** 2 / abs(
        actual.overlap(actual) * reference.overlap(reference)
    )
    np.testing.assert_allclose(fidelity, 1.0, atol=1e-10, rtol=0)
