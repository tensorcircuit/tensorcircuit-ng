"""Independent references for converged TNALG eigensolver derivatives."""

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

import tensorcircuit as tc
from tensorcircuit.tnalg.krylov import lowest_eigenvector_hermitian


@pytest.mark.parametrize("imaginary", [False, True])
@pytest.mark.parametrize("reorthogonalize", [False, True])
@pytest.mark.parametrize("field", [0.0, 0.2])
def test_converged_ritz_response(jaxb, highp, imaginary, reorthogonalize, field):
    """An isolated ground state responds even when its Krylov basis is exhausted."""
    coupling = 1j if imaginary else 1.0
    direction = jnp.asarray(
        [[0, coupling, 0, 0], [np.conj(coupling), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=jnp.complex128,
    )
    diagonal = jnp.diag(jnp.asarray([1, -1, 3, 3], dtype=jnp.complex128))
    initial = jnp.asarray([0, 1, 0, 0], dtype=jnp.complex128)

    def solve(parameter):
        return lowest_eigenvector_hermitian(
            lambda value: (diagonal + parameter * direction) @ value,
            initial,
            max_dim=4,
            reorthogonalize=reorthogonalize,
        )

    def expectation(parameter):
        vector, _, _ = solve(parameter)
        return jnp.real(jnp.vdot(vector, direction @ vector))

    def energy(parameter):
        return solve(parameter)[1]

    _, value, report = jax.jit(solve)(field)
    assert bool(report["breakdown"])
    assert bool(report["finite"])
    np.testing.assert_allclose(value, -np.sqrt(1 + field**2), atol=1e-12)
    np.testing.assert_allclose(report["residual"], 0, atol=1e-12)
    reference = -((1 + field**2) ** -1.5)
    np.testing.assert_allclose(jax.grad(expectation)(field), reference, atol=1e-10)
    np.testing.assert_allclose(
        jax.jit(jax.grad(expectation))(field), reference, atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jvp(expectation, (field,), (1.0,))[1], reference, atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(jax.grad(energy)))(field), reference, atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(energy))(field), reference, atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(jax.grad(expectation)))(field),
        3 * field / (1 + field**2) ** 2.5,
        atol=1e-10,
    )


def test_converged_ritz_preserves_seed_phase_derivative(jaxb, highp):
    """Changing a complex seed's phase must change the eigenvector's phase."""
    matrix = jnp.diag(jnp.asarray([-1, 1, 2], dtype=jnp.complex128))
    initial = jnp.asarray([1, 0, 0], dtype=jnp.complex128)

    def solve(phase):
        vector, _ = lowest_eigenvector_hermitian(
            lambda value: matrix @ value,
            2 * jnp.exp(1j * phase) * initial,
            3,
            return_report=False,
        )
        return jnp.imag(vector[0])

    np.testing.assert_allclose(jax.jit(jax.grad(solve))(0.3), np.cos(0.3), atol=1e-12)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(solve))(0.3), -np.sin(0.3), atol=1e-12
    )


def test_unconverged_ritz_keeps_algorithmic_gradient(jaxb, highp):
    """A truncated Ritz solve still differentiates its actual finite algorithm."""
    matrix = jnp.diag(jnp.asarray([-2, -0.5, 1, 3], dtype=jnp.float64))
    initial = jnp.asarray([1, 0.3, 0.7, -0.2], dtype=jnp.float64)

    def solve(parameter):
        seed = initial.at[1].add(parameter)
        return lowest_eigenvector_hermitian(lambda value: matrix @ value, seed, 2)

    assert not bool(solve(0.2)[2]["breakdown"])
    energy = lambda parameter: solve(parameter)[1]
    step = 1e-5
    reference = (energy(0.2 + step) - energy(0.2 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(0.2), reference, atol=1e-8)


@pytest.mark.parametrize("field", [0.0, 0.2])
def test_dmrg_single_site_ground_response(jaxb, highp, field):
    """The public sweep preserves the response of H = Z + g X."""
    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.dense((2,), chi=1)
    initial, _ = tnalg.product_state((1,), spec=spec, dtype=jnp.complex128)
    mpo_spec, build = tnalg.compile_mpo(np.asarray([[3], [1]]), spec.physical_indices)
    _, build_x = tnalg.compile_mpo(np.asarray([[1]]), spec.physical_indices)
    observable = build_x(jnp.asarray([1.0]))
    sweep = tnalg.make_dmrg_sweep(spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=2))

    def solve(parameter):
        state, energy = sweep(initial, build(jnp.asarray([1.0, parameter])))
        return jnp.real(tnalg.expectation(state, observable)), energy

    reference = -((1 + field**2) ** -1.5)
    actual = jax.jit(jax.grad(lambda value: solve(value)[0]))(field)
    curvature = jax.jit(jax.grad(jax.grad(lambda value: solve(value)[1])))(field)
    np.testing.assert_allclose(actual, reference, atol=1e-10)
    np.testing.assert_allclose(curvature, reference, atol=1e-10)


@pytest.mark.parametrize("symmetric", [False, True])
def test_dmrg_repeated_sweep_response(jaxb, highp, symmetric):
    """Converged complete sweeps agree with independent eigenstate perturbation."""
    tnalg = tc.tnalg
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.diag([1.0, -1.0])
    identity = np.eye(2)
    if symmetric:
        symmetry = tnalg.AbelianSymmetry((0,))
        physical = tnalg.SectorIndex.from_basis(
            symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
        )
        spec = tnalg.MPSSpec.from_sectors(
            physical_indices=(physical, physical),
            total_charge=(1,),
            bond_sectors=({(0,): 1}, {(0,): 1, (1,): 1}, {(1,): 1}),
        )
        codes = np.asarray([[3, 0], [0, 3], [1, 1], [2, 2]])
        coefficients = lambda g: jnp.asarray([1 + g, 0.53 - g, 0.27])
        observable_codes = np.asarray([[3, 0], [0, 3]])
        observable_weights = jnp.asarray([1.0, -1.0])
        matrix = (
            np.kron(z, identity)
            + 0.53 * np.kron(identity, z)
            + 0.27 * (np.kron(x, x) + np.kron(y, y))
        )
        direction = np.kron(z, identity) - np.kron(identity, z)
        indices = [1, 2]
        matrix, direction = (
            matrix[np.ix_(indices, indices)],
            direction[np.ix_(indices, indices)],
        )
    else:
        spec = tnalg.MPSSpec.dense((2, 2), chi=2)
        codes = np.asarray([[3, 0], [0, 3], [1, 1], [1, 0]])
        coefficients = lambda g: jnp.asarray([1.0, 0.53, 0.27, g])
        observable_codes = np.asarray([[1, 0]])
        observable_weights = jnp.asarray([1.0])
        matrix = (
            np.kron(z, identity) + 0.53 * np.kron(identity, z) + 0.27 * np.kron(x, x)
        )
        direction = np.kron(x, identity)
    initial, _ = tnalg.random_mps(jax.random.key(43), spec=spec, dtype=jnp.complex128)
    mpo_spec, build = tnalg.compile_mpo(
        codes,
        spec.physical_indices,
        coefficient_indices=np.asarray([0, 1, 2, 2]) if symmetric else np.arange(4),
    )
    _, build_observable = tnalg.compile_mpo(observable_codes, spec.physical_indices)
    observable = build_observable(observable_weights)
    sweep = tnalg.make_dmrg_sweep(spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=4))

    def solve(parameter):
        mpo = build(coefficients(parameter))
        state, _ = sweep(initial, mpo)
        state, energy = sweep(state, mpo)
        return jnp.real(tnalg.expectation(state, observable)), energy

    field = 0.31
    energies, vectors = np.linalg.eigh(matrix + field * direction)
    elements = vectors.conj().T @ direction @ vectors[:, 0]
    response = 2 * np.sum(np.abs(elements[1:]) ** 2 / (energies[0] - energies[1:]))
    vector_response = vectors[:, 1:] @ (elements[1:] / (energies[0] - energies[1:]))
    curvature = 6 * np.real(
        np.vdot(
            vector_response,
            (direction - elements[0].real * np.eye(len(energies))) @ vector_response,
        )
    )
    np.testing.assert_allclose(
        jax.jit(solve)(field), [elements[0].real, energies[0]], atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda g: solve(g)[0]))(field), response, atol=1e-9
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(jax.grad(lambda g: solve(g)[1])))(field), response, atol=1e-9
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(jax.grad(lambda g: solve(g)[0])))(field), curvature, atol=1e-9
    )


@pytest.mark.parametrize("complex_matrix", [False, True])
def test_converged_ritz_full_response_space(jaxb, highp, complex_matrix):
    """The response explores directions absent from the one-vector primal basis."""
    rng = np.random.default_rng(17)
    seed = rng.normal(size=(6, 6))
    perturbation = rng.normal(size=(6, 6))
    if complex_matrix:
        seed = seed + 1j * rng.normal(size=(6, 6))
        perturbation = perturbation + 1j * rng.normal(size=(6, 6))
    vectors, _ = np.linalg.qr(seed)
    energies = np.asarray([-3, -1, 0.5, 2, 2, 4])
    matrix = (vectors * energies) @ vectors.conj().T
    direction = (perturbation + perturbation.conj().T) / 2
    initial = jnp.asarray(vectors[:, 0])
    elements = vectors.conj().T @ direction @ vectors[:, 0]
    vector_response = vectors[:, 1:] @ (elements[1:] / (energies[0] - energies[1:]))
    energy_curvature = 2 * np.sum(
        np.abs(elements[1:]) ** 2 / (energies[0] - energies[1:])
    )

    def solve(parameter):
        return lowest_eigenvector_hermitian(
            lambda value: (jnp.asarray(matrix) + parameter * jnp.asarray(direction))
            @ value,
            initial,
            max_dim=6,
            return_report=False,
        )

    result, tangent = jax.jit(lambda g: jax.jvp(solve, (g,), (1.0,)))(0.0)
    phase = np.vdot(vectors[:, 0], np.asarray(result[0]))
    np.testing.assert_allclose(tangent[0], phase * vector_response, atol=1e-10)
    np.testing.assert_allclose(tangent[1], elements[0].real, atol=1e-10)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda g: solve(g)[1]))(0.0), energy_curvature, atol=1e-10
    )


def test_ritz_response_single_precision_batch(jaxb):
    """Default precision and batched derivatives include a breakdown point."""
    matrix = jnp.diag(jnp.asarray([1, -1, 3, 4], dtype=jnp.complex64))
    direction = jnp.asarray(
        [[0, 1, 0, 0], [1, 0, 0.3, 0.5], [0, 0.3, 0, 0], [0, 0.5, 0, 0]],
        dtype=jnp.complex64,
    )
    initial = jnp.asarray([0, 1, 0, 0], dtype=jnp.complex64)

    def expectation(field):
        vector, _ = lowest_eigenvector_hermitian(
            lambda value: (matrix + field * direction) @ value,
            initial,
            4,
            return_report=False,
        )
        return jnp.real(jnp.vdot(vector, direction @ vector))

    fields = jnp.asarray([0.0, 0.2], dtype=jnp.float32)
    reference = []
    breakdowns = []
    for field in np.asarray(fields):
        dense = np.asarray(matrix, dtype=np.complex128) + field * np.asarray(direction)
        energies, vectors = np.linalg.eigh(dense)
        elements = vectors.conj().T @ np.asarray(direction) @ vectors[:, 0]
        reference.append(
            2 * np.sum(np.abs(elements[1:]) ** 2 / (energies[0] - energies[1:]))
        )
        report = lowest_eigenvector_hermitian(
            lambda value: (matrix + field * direction) @ value, initial, 4
        )[2]
        breakdowns.append(bool(report["breakdown"]))
    assert breakdowns == [True, False]
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(expectation)))(fields), reference, atol=2e-5
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda values: jnp.sum(jax.vmap(expectation)(values))))(
            fields
        ),
        reference,
        atol=2e-5,
    )
