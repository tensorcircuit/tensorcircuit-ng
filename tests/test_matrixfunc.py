"""Tests for backend-generic matrix-function primitives."""

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
from scipy.sparse import csr_matrix

import tensorcircuit as tc


def _reference_action(matrix, vector, coefficient):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (
        eigenvectors
        @ np.diag(np.exp(coefficient * eigenvalues))
        @ eigenvectors.conj().T
        @ vector
    )


def test_matrixfunc_config_validation():
    with pytest.raises(ValueError):
        tc.matrixfunc.KrylovConfig(0)
    with pytest.raises(ValueError):
        tc.matrixfunc.ChebyshevConfig(2, (1.0, 1.0))
    with pytest.raises(ValueError):
        tc.matrixfunc.ChebyshevConfig(2, (-1.0, 1.0), scaling_steps=0)
    with pytest.raises(TypeError):
        tc.matrixfunc.ChebyshevConfig(2, (-1.0, 1.0), scaling_steps=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.ChebyshevConfig(2, (-1.0, 1.0), tail_decay_tolerance=0.0)
    with pytest.raises(ValueError):
        tc.matrixfunc.kernel_weights(
            tc.matrixfunc.ChebyshevConfig(1, (-1.0, 1.0), kernel="jackson")
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.TaylorConfig(1, 0)


def test_matrixfunc_dense_sparse_callable_parity(npb):
    matrix = np.array([[0.3, 0.2 - 0.1j], [0.2 + 0.1j, -0.4]], dtype=np.complex64)
    vector = np.array([1.0, 0.2 + 0.3j], dtype=np.complex64)
    vector /= np.linalg.norm(vector)
    coefficient = -0.7j
    config = tc.matrixfunc.KrylovConfig(4)
    expected = _reference_action(matrix, vector, coefficient)

    operators = (
        matrix,
        csr_matrix(matrix),
        lambda value: matrix @ value,
    )
    for operator in operators:
        result = tc.matrixfunc.exponential_action(
            operator,
            vector,
            coefficient,
            config,
            dimension=2,
        )
        np.testing.assert_allclose(result, expected, atol=2e-5)


def test_matrixfunc_chebyshev_and_taylor_match(npb):
    matrix = np.array([[0.3, 0.2 - 0.1j], [0.2 + 0.1j, -0.4]], dtype=np.complex64)
    vector = np.array([1.0, 0.2 + 0.3j], dtype=np.complex64)
    vector /= np.linalg.norm(vector)
    expected = _reference_action(matrix, vector, -0.7j)
    chebyshev = tc.matrixfunc.ChebyshevConfig(32, (-1.0, 1.0))
    taylor = tc.matrixfunc.TaylorConfig(24, 2)
    np.testing.assert_allclose(
        tc.matrixfunc.exponential_action(matrix, vector, -0.7j, chebyshev),
        expected,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        tc.matrixfunc.exponential_action(matrix, vector, -0.7j, taylor),
        expected,
        atol=2e-5,
    )


def test_matrixfunc_projection_reuse_and_trace(npb):
    matrix = np.diag(np.array([-0.5, 0.2, 0.9], dtype=np.complex64))
    vector = np.array([1.0, 0.2j, 0.3], dtype=np.complex64)
    projection = tc.matrixfunc.lanczos_project(
        matrix, vector, tc.matrixfunc.KrylovConfig(5)
    )
    values = tc.matrixfunc.lanczos_resolvent(projection, 0.4 + 0.2j)
    expected = np.vdot(
        vector, np.linalg.solve((0.4 + 0.2j) * np.eye(3) - matrix, vector)
    )
    np.testing.assert_allclose(values, expected, atol=2e-5)

    probes = np.eye(3, dtype=np.complex64) * np.sqrt(3.0)
    measure = tc.matrixfunc.slq_measure(matrix, probes, tc.matrixfunc.KrylovConfig(5))
    estimate = tc.matrixfunc.evaluate_slq_trace(
        measure, lambda nodes: np.exp(-0.3 * nodes)
    )
    np.testing.assert_allclose(
        estimate,
        np.trace(
            np.linalg.matrix_power(np.eye(3), 1)
            @ np.diag(np.exp(-0.3 * np.diag(matrix)))
        ),
        atol=2e-5,
    )


def test_lanczos_lowest_eigenpair_dense_sparse_mvp(npb):
    matrix = np.array(
        [[1.0, 0.2 - 0.1j, 0.0], [0.2 + 0.1j, -0.4, 0.3], [0.0, 0.3, 0.8]],
        dtype=np.complex64,
    )
    vector = np.array([1.0, 0.2j, 0.4], dtype=np.complex64)
    expected_energy, expected_vectors = np.linalg.eigh(matrix)
    config = tc.matrixfunc.KrylovConfig(4)

    for operator in (matrix, csr_matrix(matrix), lambda value: matrix @ value):
        energy, state = tc.matrixfunc.lanczos_lowest_eigenpair(
            operator, vector, config, dimension=3
        )
        np.testing.assert_allclose(energy, expected_energy[0], atol=2e-5)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=2e-5)
        np.testing.assert_allclose(matrix @ state - energy * state, 0.0, atol=2e-5)
        np.testing.assert_allclose(
            np.abs(np.vdot(expected_vectors[:, 0], state)), 1.0, atol=2e-5
        )


def test_lanczos_lowest_eigenpair_breakdown_uses_seed_subspace(npb):
    matrix = np.diag(np.array([2.0, -3.0, 4.0], dtype=np.complex64))
    vector = np.array([2.0, 0.0, 0.0], dtype=np.complex64)
    energy, state = tc.matrixfunc.lanczos_lowest_eigenpair(
        matrix, vector, tc.matrixfunc.KrylovConfig(4)
    )
    np.testing.assert_allclose(energy, 2.0, atol=2e-5)
    np.testing.assert_allclose(state, [1.0, 0.0, 0.0], atol=2e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("torchb")])
def test_lanczos_lowest_eigenpair_backend_mvp(backend):
    matrix = tc.backend.convert_to_tensor(
        np.array([[0.5, 0.2], [0.2, -0.4]], dtype=np.complex64)
    )
    vector = tc.backend.convert_to_tensor(np.array([1.0, 0.3j], dtype=np.complex64))
    operator = lambda value: tc.backend.matvec(matrix, value)
    energy, state = tc.matrixfunc.lanczos_lowest_eigenpair(
        operator, vector, tc.matrixfunc.KrylovConfig(3)
    )
    expected_energy = np.linalg.eigvalsh(np.asarray(tc.backend.numpy(matrix)))[0]
    np.testing.assert_allclose(energy, expected_energy, atol=2e-5)
    np.testing.assert_allclose(
        operator(state) - tc.backend.cast(energy, tc.dtypestr) * state,
        0.0,
        atol=2e-5,
    )


def test_lanczos_lowest_eigenpair_jax_jit_mvp(jaxb):
    matrix = tc.backend.convert_to_tensor(
        np.array([[0.5, 0.2], [0.2, -0.4]], dtype=np.complex64)
    )
    vector = tc.backend.convert_to_tensor(np.array([1.0, 0.3j], dtype=np.complex64))
    config = tc.matrixfunc.KrylovConfig(3)
    energy, state = tc.backend.jit(
        lambda seed: tc.matrixfunc.lanczos_lowest_eigenpair(
            lambda value: tc.backend.matvec(matrix, value), seed, config
        )
    )(vector)
    expected_energy = np.linalg.eigvalsh(np.asarray(tc.backend.numpy(matrix)))[0]
    np.testing.assert_allclose(energy, expected_energy, atol=2e-5)
    np.testing.assert_allclose(
        tc.backend.matvec(matrix, state) - energy * state, 0.0, atol=2e-5
    )


def test_prepared_matrixfunc_queries_inside_jax_jit(jaxb):
    energies = np.array([-0.4, 0.6], dtype=np.float32)
    matrix = tc.backend.convert_to_tensor(np.diag(energies).astype(np.complex64))
    vector = tc.backend.convert_to_tensor(np.array([1.0, 0.5j], dtype=np.complex64))
    probes = tc.backend.convert_to_tensor(np.sqrt(2.0) * np.eye(2, dtype=np.complex64))
    krylov = tc.matrixfunc.KrylovConfig(2)
    chebyshev = tc.matrixfunc.ChebyshevConfig(64, (-1.0, 1.0))

    def evaluate(beta):
        projection = tc.matrixfunc.lanczos_project(matrix, vector, krylov)
        quadratic = tc.matrixfunc.lanczos_quadrature(
            projection, lambda nodes: tc.backend.exp(-beta * nodes)
        )
        measure = tc.matrixfunc.slq_measure(matrix, probes, krylov)
        trace = tc.matrixfunc.evaluate_slq_trace(
            measure, lambda nodes: tc.backend.exp(-beta * nodes)
        )
        moments = tc.matrixfunc.chebyshev_moments(matrix, vector, chebyshev)
        coefficients = tc.matrixfunc.fermi_dirac_coefficients(beta, 0.0, chebyshev)
        occupancy = tc.matrixfunc.evaluate_chebyshev_moments(moments, coefficients)
        return quadratic, trace, occupancy

    beta = 0.4
    quadratic, trace, occupancy = tc.backend.jit(evaluate)(
        tc.backend.convert_to_tensor(beta, dtype=tc.rdtypestr)
    )
    boltzmann = np.exp(-beta * energies)
    np.testing.assert_allclose(quadratic, boltzmann @ [1.0, 0.25], atol=2e-5)
    np.testing.assert_allclose(trace, np.sum(boltzmann), atol=2e-5)
    np.testing.assert_allclose(
        occupancy, [1.0, 0.25] @ (1.0 / (1.0 + np.exp(beta * energies))), atol=2e-5
    )


def test_estimate_spectral_bounds_inside_jax_jit(jaxb):
    matrix = tc.backend.convert_to_tensor(
        np.diag(np.array([-0.5, 0.9], dtype=np.complex64))
    )
    vector = tc.backend.convert_to_tensor(np.array([1.0, 1.0], dtype=np.complex64))
    config = tc.matrixfunc.KrylovConfig(2)

    def bounds(seed):
        return tc.matrixfunc.estimate_spectral_bounds(
            lambda value: tc.backend.matvec(matrix, value), seed, config, padding=0.1
        )

    eager = bounds(vector)
    compiled = tc.backend.jit(bounds)(vector)
    assert isinstance(eager[0], float) and isinstance(eager[1], float)
    np.testing.assert_allclose(compiled, eager, atol=2e-5)


def test_matrixfunc_jax_jit_and_gradient(jaxb):
    matrix = tc.backend.convert_to_tensor(
        np.array([[0.3, 0.2], [0.2, -0.4]], dtype=np.complex64)
    )
    vector = tc.backend.convert_to_tensor(np.array([1.0, 0.2j], dtype=np.complex64))
    config = tc.matrixfunc.KrylovConfig(3)

    def loss(time):
        state = tc.matrixfunc.exponential_action(matrix, vector, -1.0j * time, config)
        return tc.backend.real(
            tc.backend.sum(tc.backend.abs(state) * tc.backend.abs(state))
        )

    value, gradient = tc.backend.value_and_grad(loss)(tc.backend.convert_to_tensor(0.3))
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    state = tc.backend.jit(
        lambda time: tc.matrixfunc.exponential_action(
            matrix, vector, -1.0j * time, config
        )
    )(tc.backend.convert_to_tensor(0.3))
    assert state.shape == (2,)


def test_matrixfunc_jax_exact_breakdown_gradient(jaxb):
    vector = tc.backend.convert_to_tensor(np.array([1.0, 0.0], dtype=np.complex64))
    config = tc.matrixfunc.KrylovConfig(3)

    def loss(theta):
        zero = tc.backend.convert_to_tensor(0.0, dtype=tc.dtypestr)
        two = tc.backend.convert_to_tensor(2.0, dtype=tc.dtypestr)
        matrix = tc.backend.stack(
            [
                tc.backend.stack([theta, zero]),
                tc.backend.stack([zero, two]),
            ]
        )
        state = tc.matrixfunc.exponential_action(matrix, vector, -0.3, config)
        return tc.backend.real(state[0])

    theta = tc.backend.convert_to_tensor(1.0, dtype=tc.rdtypestr)
    value, gradient = tc.backend.value_and_grad(loss)(theta)
    expected = np.exp(-0.3)
    np.testing.assert_allclose(value, expected, atol=2e-6)
    np.testing.assert_allclose(gradient, -0.3 * expected, atol=2e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
def test_matrixfunc_backend_dtype_paths(backend):
    matrix = tc.backend.convert_to_tensor(
        np.array([[0.3, 0.2], [0.2, -0.4]], dtype=np.complex64)
    )
    vector = tc.backend.convert_to_tensor(np.array([1.0, 0.2j], dtype=np.complex64))
    expected = _reference_action(
        np.asarray(tc.backend.numpy(matrix)),
        np.asarray(tc.backend.numpy(vector)),
        -0.2j,
    )
    for method in (
        tc.matrixfunc.KrylovConfig(3),
        tc.matrixfunc.ChebyshevConfig(16, (-1.0, 1.0)),
        tc.matrixfunc.TaylorConfig(16, 2),
    ):
        result = tc.matrixfunc.exponential_action(matrix, vector, -0.2j, method)
        np.testing.assert_allclose(
            np.asarray(tc.backend.numpy(result)), expected, atol=2e-5
        )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb")])
def test_nonchebyshev_matrixfunc_backend_paths(backend):
    matrix_np = np.diag(np.array([0.3, -0.4], dtype=np.float32)).astype(np.complex64)
    vector_np = np.array([1.0, 0.2 + 0.3j], dtype=np.complex64)
    matrix = tc.backend.convert_to_tensor(matrix_np)
    vector = tc.backend.convert_to_tensor(vector_np)
    coefficient = -0.7j
    expected = _reference_action(matrix_np, vector_np, coefficient)

    krylov = tc.matrixfunc.exponential_action(
        matrix, vector, coefficient, tc.matrixfunc.KrylovConfig(3)
    )
    taylor = tc.matrixfunc.exponential_action(
        matrix, vector, coefficient, tc.matrixfunc.TaylorConfig(24, 2)
    )
    np.testing.assert_allclose(tc.backend.numpy(krylov), expected, atol=3e-5)
    np.testing.assert_allclose(tc.backend.numpy(taylor), expected, atol=3e-5)

    query = 0.6 + 0.2j
    resolvent = tc.matrixfunc.lanczos_resolvent(
        tc.matrixfunc.lanczos_project(matrix, vector, tc.matrixfunc.KrylovConfig(3)),
        query,
    )
    expected_resolvent = np.vdot(
        vector_np, np.linalg.solve(query * np.eye(2) - matrix_np, vector_np)
    )
    np.testing.assert_allclose(
        tc.backend.numpy(resolvent), expected_resolvent, atol=3e-5
    )

    probes = tc.backend.convert_to_tensor(np.sqrt(2.0) * np.eye(2, dtype=np.complex64))
    measure = tc.matrixfunc.slq_measure(
        matrix,
        probes,
        tc.matrixfunc.KrylovConfig(3),
        probe_batch_size=1,
    )
    trace = tc.matrixfunc.evaluate_slq_trace(
        measure, lambda nodes: tc.backend.exp(-0.2 * nodes)
    )
    np.testing.assert_allclose(
        tc.backend.numpy(trace), np.sum(np.exp(-0.2 * np.diag(matrix_np))), atol=3e-5
    )

    times = tc.backend.convert_to_tensor([0.0, 0.2, 0.4], dtype=tc.rdtypestr)
    evolved = tc.backend.jit(
        lambda values: tc.timeevol.krylov_evol(
            matrix, vector, values, subspace_dimension=2
        )
    )(times)
    times_np = np.asarray(tc.backend.numpy(times))
    expected_evolved = np.stack(
        [_reference_action(matrix_np, vector_np, -1.0j * time) for time in times_np]
    )
    np.testing.assert_allclose(tc.backend.numpy(evolved), expected_evolved, atol=3e-5)


def test_lanczos_breakdown_masks_padded_ritz_slots(npb):
    hamiltonian = 100.0 * np.eye(3, dtype=np.complex64)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.complex64)
    projection = tc.matrixfunc.lanczos_project(
        hamiltonian, vector, tc.matrixfunc.KrylovConfig(4)
    )
    np.testing.assert_array_equal(
        np.asarray(projection.recurrence.active), [True, False, False, False]
    )
    np.testing.assert_allclose(
        tc.matrixfunc.lanczos_resolvent(projection, 0.0), -0.01, atol=2e-6
    )
    result = tc.matrixfunc.lanczos_apply(projection, lambda nodes: 1.0 / nodes)
    np.testing.assert_allclose(result, vector / 100.0, atol=2e-6)
    bounds = tc.matrixfunc.estimate_spectral_bounds(
        hamiltonian, vector, tc.matrixfunc.KrylovConfig(4)
    )
    assert bounds[0] > 0.0 and bounds[1] < 200.0


def test_lanczos_breakdown_never_evaluates_inactive_overflow(npb):
    hamiltonian = -100.0 * np.eye(3, dtype=np.complex64)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.complex64)
    projection = tc.matrixfunc.lanczos_project(
        hamiltonian, vector, tc.matrixfunc.KrylovConfig(4)
    )
    result = tc.matrixfunc.lanczos_apply(
        projection, lambda nodes: tc.backend.exp(nodes)
    )
    np.testing.assert_allclose(result, np.exp(-100.0) * vector, atol=1e-44)


def test_lanczos_default_near_breakdown_is_stable(npb, highp):
    hamiltonian = np.array(
        [[100.0, 1.0e-12, 0.0], [1.0e-12, 100.2, 0.0], [0.0, 0.0, -3.0]],
        dtype=np.complex128,
    )
    vector = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    projection = tc.matrixfunc.lanczos_project(
        hamiltonian, vector, tc.matrixfunc.KrylovConfig(4)
    )
    np.testing.assert_array_equal(
        np.asarray(projection.recurrence.active), [True, False, False, False]
    )
    result = tc.matrixfunc.exponential_action(
        hamiltonian, vector, -0.2j, tc.matrixfunc.KrylovConfig(4)
    )
    np.testing.assert_allclose(result, np.exp(-20.0j) * vector, atol=2e-12)


def test_lanczos_near_breakdown_action_and_resolvent_share_projection(npb):
    hamiltonian = np.array([[1.0, 1.0e-8], [1.0e-8, 1.0]], dtype=np.complex64)
    vector = np.array([1.0, 0.0], dtype=np.complex64)
    projection = tc.matrixfunc.lanczos_project(
        hamiltonian, vector, tc.matrixfunc.KrylovConfig(3)
    )
    np.testing.assert_array_equal(
        np.asarray(projection.recurrence.active), [True, False, False]
    )
    np.testing.assert_allclose(np.asarray(projection.basis)[:, 1:], 0.0, atol=0.0)

    z = 0.3 + 0.2j
    resolvent = tc.matrixfunc.lanczos_resolvent(projection, z)
    action = tc.matrixfunc.lanczos_apply(projection, lambda nodes: 1.0 / (z - nodes))
    np.testing.assert_allclose(resolvent, np.vdot(vector, action), atol=3e-6)
    np.testing.assert_allclose(resolvent, 1.0 / (z - 1.0), atol=3e-6)


def test_lanczos_breakdown_does_not_complete_larger_space(npb):
    hamiltonian = np.diag(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)).astype(
        np.complex64
    )
    vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    projection = tc.matrixfunc.lanczos_project(
        hamiltonian, vector, tc.matrixfunc.KrylovConfig(3)
    )
    np.testing.assert_array_equal(
        np.asarray(projection.recurrence.active), [True, False, False]
    )
    np.testing.assert_allclose(np.asarray(projection.basis)[:, 1:], 0.0, atol=0.0)
    z = 0.3 + 0.2j
    np.testing.assert_allclose(
        tc.matrixfunc.lanczos_resolvent(projection, z), 1.0 / (z - 1.0), atol=3e-6
    )


def test_lanczos_zero_vector(npb):
    hamiltonian = np.diag(np.array([-0.3, 0.7], dtype=np.complex64))
    vector = np.array([1.0, 0.2j], dtype=np.complex64)
    scan_projection = tc.matrixfunc.lanczos_project(
        hamiltonian, vector, tc.matrixfunc.KrylovConfig(3)
    )
    np.testing.assert_array_equal(
        np.asarray(scan_projection.recurrence.active), [True, True, False]
    )
    with pytest.raises(ValueError):
        tc.matrixfunc.lanczos_project(
            hamiltonian,
            np.zeros(2, dtype=np.complex64),
            tc.matrixfunc.KrylovConfig(3),
        )


def _dense_function_action(matrix, vector, function):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (
        eigenvectors @ np.diag(function(eigenvalues)) @ eigenvectors.conj().T @ vector
    )


def test_lanczos_query_shapes_and_trace_statistics(npb):
    matrix = np.diag(np.array([-0.6, 0.1, 0.8], dtype=np.float32)).astype(np.complex64)
    vector = np.array([1.0, 0.2j, -0.3], dtype=np.complex64)
    projection = tc.matrixfunc.lanczos_project(
        matrix, vector, tc.matrixfunc.KrylovConfig(4)
    )
    queries = np.array([0.1, 0.4], dtype=np.float32)
    applied = tc.matrixfunc.lanczos_apply(
        projection, lambda nodes: np.exp(-queries[:, None] * nodes[None, :])
    )
    expected = np.stack(
        [
            _dense_function_action(matrix, vector, lambda x: np.exp(-q * x))
            for q in queries
        ]
    )
    np.testing.assert_allclose(applied, expected, atol=3e-5)

    quadratic = tc.matrixfunc.lanczos_quadrature(
        projection, lambda nodes: np.exp(-queries[:, None] * nodes[None, :])
    )
    expected_quadratic = np.array(
        [
            np.vdot(
                vector, _dense_function_action(matrix, vector, lambda x: np.exp(-q * x))
            )
            for q in queries
        ]
    )
    np.testing.assert_allclose(quadratic, expected_quadratic, atol=3e-5)

    probes = np.sqrt(3.0) * np.eye(3, dtype=np.complex64)
    measure = tc.matrixfunc.slq_measure(
        matrix, probes, tc.matrixfunc.KrylovConfig(4), probe_batch_size=2
    )
    values, error = tc.matrixfunc.evaluate_slq_trace(
        measure,
        lambda nodes: np.exp(-queries[:, None, None] * nodes[None, :, :]),
        with_std=True,
    )
    samples = 3.0 * np.exp(-queries[:, None] * np.diag(matrix)[None, :])
    np.testing.assert_allclose(values, np.mean(samples, axis=-1), atol=3e-5)
    np.testing.assert_allclose(
        error,
        np.std(samples, axis=-1, ddof=1) / np.sqrt(samples.shape[-1]),
        atol=3e-5,
    )


def test_chebyshev_moments_coefficients_and_actions(npb):
    matrix = np.diag(np.array([-0.4, 0.3], dtype=np.float32)).astype(np.complex64)
    right = np.array([1.0, 0.4j], dtype=np.complex64)
    left = np.array([0.2 - 0.1j, 1.0], dtype=np.complex64)
    config = tc.matrixfunc.ChebyshevConfig(64, (-1.0, 1.0))
    moments = tc.matrixfunc.chebyshev_moments(matrix, right, config, left_vector=left)
    expected_moments = np.array(
        [
            np.vdot(
                left,
                _dense_function_action(
                    matrix, right, lambda x, n=n: np.cos(n * np.arccos(x))
                ),
            )
            for n in range(config.order)
        ]
    )
    np.testing.assert_allclose(moments.moments, expected_moments, atol=3e-5)

    coefficients = tc.matrixfunc.chebyshev_coefficients(
        lambda energy: energy * energy, config
    )
    expected_coefficients = np.zeros(config.order, dtype=np.complex64)
    expected_coefficients[[0, 2]] = [0.5, 0.5]
    np.testing.assert_allclose(coefficients, expected_coefficients, atol=3e-5)
    action = tc.matrixfunc.chebyshev_action(matrix, right, coefficients, config)
    np.testing.assert_allclose(action, matrix @ matrix @ right, atol=3e-5)

    exponential = tc.matrixfunc.exponential_coefficients(-0.3j, config)
    expected_action = _dense_function_action(matrix, right, lambda x: np.exp(-0.3j * x))
    np.testing.assert_allclose(
        tc.matrixfunc.chebyshev_action(matrix, right, exponential, config),
        expected_action,
        atol=3e-5,
    )
    resolvent = tc.matrixfunc.resolvent_coefficients(0.7 + 0.2j, config)
    expected_resolvent = np.vdot(
        left,
        np.linalg.solve((0.7 + 0.2j) * np.eye(2) - matrix, right),
    )
    np.testing.assert_allclose(
        tc.matrixfunc.evaluate_chebyshev_moments(moments, resolvent),
        expected_resolvent,
        atol=3e-5,
    )


def test_chebyshev_trace_kernels_and_query_broadcasting(npb):
    matrix = np.diag(np.array([-0.5, 0.25, 0.7], dtype=np.float32)).astype(np.complex64)
    probes = np.sqrt(3.0) * np.eye(3, dtype=np.complex64)
    config = tc.matrixfunc.ChebyshevConfig(32, (-1.0, 1.0))
    moments = tc.matrixfunc.stochastic_chebyshev_moments(
        matrix, probes, config, probe_batch_size=2
    )
    beta = np.array([0.2, 0.6], dtype=np.float32)
    coefficients = tc.matrixfunc.exponential_coefficients(-beta, config)
    values, error = tc.matrixfunc.evaluate_chebyshev_trace(
        moments, coefficients, with_std=True
    )
    expected_samples = 3.0 * np.exp(-beta[:, None] * np.diag(matrix)[None, :])
    np.testing.assert_allclose(values, np.mean(expected_samples, axis=-1), atol=4e-5)
    np.testing.assert_allclose(
        error,
        np.std(expected_samples, axis=-1, ddof=1) / np.sqrt(probes.shape[0]),
        atol=4e-5,
    )

    fermi = tc.matrixfunc.fermi_dirac_coefficients(
        np.array([0.5, 1.0], dtype=np.float32), 0.1, config
    )
    expected_fermi = np.stack(
        [1.0 / (1.0 + np.exp(b * (np.diag(matrix) - 0.1))) for b in (0.5, 1.0)]
    )
    np.testing.assert_allclose(
        tc.matrixfunc.evaluate_chebyshev_trace(moments, fermi),
        np.mean(3.0 * expected_fermi, axis=-1),
        atol=4e-5,
    )

    for kernel in ("dirichlet", "jackson", "lorentz"):
        kernel_config = tc.matrixfunc.ChebyshevConfig(16, (-1.0, 1.0), kernel=kernel)
        weights = np.asarray(tc.matrixfunc.kernel_weights(kernel_config))
        assert weights.shape == (16,)
        np.testing.assert_allclose(weights[0], 1.0, atol=2e-6)
        np.testing.assert_allclose(
            tc.matrixfunc.delta_coefficients(
                np.array([-0.2, 0.4]), kernel_config
            ).shape,
            (2, 16),
        )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_kernel_weights_backends_highp(backend, highp):
    order, lorentz_lambda = 16, 3.0
    n = np.arange(order)
    angle = np.pi / order
    expected = {
        "dirichlet": np.ones(order),
        "jackson": ((order - n) * np.cos(angle * n) + np.sin(angle * n) / np.tan(angle))
        / order,
        "lorentz": np.sinh(lorentz_lambda * (1.0 - n / order))
        / np.sinh(lorentz_lambda),
    }
    for kernel, reference in expected.items():
        config = tc.matrixfunc.ChebyshevConfig(
            order, (-1.0, 1.0), kernel=kernel, lorentz_lambda=lorentz_lambda
        )
        weights = tc.matrixfunc.kernel_weights(config)
        assert tc.backend.dtype(weights) == tc.backend.dtype(
            tc.backend.ones([1], dtype=tc.rdtypestr)
        )
        np.testing.assert_allclose(tc.backend.numpy(weights), reference, atol=1e-12)


def test_resolvent_coefficients_reject_nondecaying_tail(npb):
    config = tc.matrixfunc.ChebyshevConfig(32, (-1.5, 1.5))
    with pytest.raises(ValueError, match="tail exceeds"):
        tc.matrixfunc.resolvent_coefficients(1.0e-6j, config)
    with pytest.raises(ValueError, match=r"indices \[1\].*have z="):
        tc.matrixfunc.resolvent_coefficients(np.array([1.0j, 1.0e-6j]), config)
    with pytest.raises(ValueError, match=r"real interval.*indices \[1\]"):
        tc.matrixfunc.resolvent_coefficients(np.array([2.0j, 0.0]), config)

    strict_config = tc.matrixfunc.ChebyshevConfig(
        32, (-1.5, 1.5), tail_decay_tolerance=1.0e-12
    )
    with pytest.raises(ValueError, match="tail_decay_tolerance=1e-12"):
        tc.matrixfunc.resolvent_coefficients(1.0j, strict_config)

    matrix = np.diag(np.array([-0.8, 0.2, 1.0], dtype=np.float32)).astype(np.complex64)
    vector = np.array([1.0, -0.2j, 0.5], dtype=np.complex64)
    z = 1.0j
    coefficients = tc.matrixfunc.resolvent_coefficients(z, config)
    result = tc.matrixfunc.chebyshev_action(matrix, vector, coefficients, config)
    np.testing.assert_allclose(
        result, np.linalg.solve(z * np.eye(3) - matrix, vector), atol=3e-5
    )

    with pytest.raises(ValueError, match="purely real or purely imaginary"):
        tc.matrixfunc.exponential_coefficients(-200.0 + 1.0e-30j, config)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
def test_scaled_chebyshev_large_imaginary_time(backend, highp):
    energies = np.array([0.01, 0.5, 2.0], dtype=np.float64)
    matrix = np.diag(energies).astype(np.complex128)
    vector = np.ones(3, dtype=np.complex128)
    config = tc.matrixfunc.ChebyshevConfig(64, (-0.5, 8.0), scaling_steps=16)
    result = tc.matrixfunc.exponential_action(matrix, vector, -200.0, config)
    np.testing.assert_allclose(result, np.exp(-200.0 * energies), atol=2e-12)
    coefficients = np.array([-20.0, -40.0], dtype=np.float64)
    batched = tc.matrixfunc.exponential_action(matrix, vector, coefficients, config)
    np.testing.assert_allclose(
        batched,
        np.exp(coefficients[:, None] * energies[None, :]),
        atol=2e-12,
    )


def test_taylor_queries_shifts_and_validation(npb):
    matrix = np.diag(np.array([-0.3, 0.8], dtype=np.float32)).astype(np.complex64)
    vector = np.array([1.0, -0.2j], dtype=np.complex64)
    coefficient = np.array([-0.2j, -0.4j], dtype=np.complex64)
    config = tc.matrixfunc.TaylorConfig(20, 2)
    result = tc.matrixfunc.exponential_action(matrix, vector, coefficient, config)
    expected = np.stack(
        [
            _dense_function_action(matrix, vector, lambda x, c=c: np.exp(c * x))
            for c in coefficient
        ]
    )
    np.testing.assert_allclose(result, expected, atol=3e-5)
    zero_degree = tc.matrixfunc.exponential_action(
        matrix, vector, 0.7, tc.matrixfunc.TaylorConfig(0, 3)
    )
    np.testing.assert_allclose(zero_degree, vector, atol=0.0)
    shifted = tc.matrixfunc.exponential_action(
        matrix,
        vector,
        -0.4j,
        tc.matrixfunc.KrylovConfig(3),
        energy_shift=-0.2,
    )
    np.testing.assert_allclose(
        shifted,
        _dense_function_action(matrix, vector, lambda x: np.exp(-0.4j * x)),
        atol=3e-5,
    )

    with pytest.raises(ValueError):
        tc.matrixfunc.chebyshev_action(
            matrix, vector, np.ones(3), tc.matrixfunc.ChebyshevConfig(4, (-1, 1))
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.exponential_action(
            matrix, vector, 1.0, tc.matrixfunc.KrylovConfig(2), energy_shift=[0.0, 1.0]
        )
    with pytest.raises(TypeError):
        tc.matrixfunc.lanczos_project(matrix, vector, 3)  # type: ignore[arg-type]


def test_matrixfunc_operator_and_query_validation(npb):
    matrix = np.eye(2, dtype=np.complex64)
    vector = np.ones(2, dtype=np.complex64)
    config = tc.matrixfunc.KrylovConfig(2)
    with pytest.raises(ValueError):
        tc.matrixfunc.lanczos_project(np.ones((2, 3)), vector, config)
    with pytest.raises(ValueError):
        tc.matrixfunc.lanczos_project(matrix, vector, config, dimension=3)
    with pytest.raises(ValueError):
        tc.matrixfunc.lanczos_project(lambda x: x, np.ones((2, 1)), config)
    with pytest.raises(ValueError):
        tc.matrixfunc.lanczos_apply(
            tc.matrixfunc.lanczos_project(matrix, vector, config),
            lambda nodes: np.ones((2, 2, 2)),
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.resolvent_coefficients(
            np.ones((2, 2)), tc.matrixfunc.ChebyshevConfig(4, (-1, 1))
        )
    with pytest.raises(ValueError, match="real interval"):
        tc.matrixfunc.resolvent_coefficients(
            0.0, tc.matrixfunc.ChebyshevConfig(4, (-1, 1))
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.delta_coefficients(
            np.array([-1.0, 0.0]), tc.matrixfunc.ChebyshevConfig(4, (-1, 1))
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.fermi_dirac_coefficients(
            np.array([1.0, 2.0]),
            np.array([0.0]),
            tc.matrixfunc.ChebyshevConfig(8, (-1, 1)),
        )


def test_matrixfunc_jax_physical_jit_vmap_and_ad(jaxb, highp):
    matrix = tc.backend.convert_to_tensor(
        np.diag(np.array([-0.7, 0.2, 0.9], dtype=np.float64)).astype(np.complex128)
    )
    observable = tc.backend.convert_to_tensor(
        np.diag(np.array([1.0, 2.0, -1.0], dtype=np.float64)).astype(np.complex128)
    )
    probes = tc.backend.convert_to_tensor(np.sqrt(3.0) * np.eye(3, dtype=np.complex128))
    config = tc.matrixfunc.KrylovConfig(4)

    def expectation(beta):
        return tc.spectral.thermal_expectation(
            matrix, observable, beta, probes=probes, method=config
        )

    beta = 0.4
    weights = np.exp(-beta * np.diag(matrix))
    energies = np.diag(matrix).real
    observable_values = np.diag(observable).real
    expected = np.sum(observable_values * weights) / np.sum(weights)
    expected_gradient = -(
        np.sum(observable_values * energies * weights) / np.sum(weights)
        - expected * np.sum(energies * weights) / np.sum(weights)
    )
    np.testing.assert_allclose(
        tc.backend.jit(expectation)(tc.backend.convert_to_tensor(beta)),
        expected,
        atol=2e-10,
    )
    gradient = tc.backend.jit(
        tc.backend.grad(lambda value: tc.backend.real(expectation(value)))
    )(tc.backend.convert_to_tensor(beta))
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-9)
    vectorized = tc.backend.vmap(expectation)(
        tc.backend.convert_to_tensor(np.array([0.0, beta], dtype=np.float64))
    )
    expected_vectorized = np.array(
        [
            np.mean(observable_values),
            expected,
        ]
    )
    np.testing.assert_allclose(vectorized, expected_vectorized, atol=2e-9)
    fermi_config = tc.matrixfunc.ChebyshevConfig(16, (-10.0, 10.0))
    fermi_gradient = tc.backend.grad(
        lambda b: tc.backend.real(
            tc.backend.sum(tc.matrixfunc.fermi_dirac_coefficients(b, 0.0, fermi_config))
        )
    )(100.0)
    assert np.isfinite(np.asarray(fermi_gradient))

    chebyshev = tc.matrixfunc.ChebyshevConfig(64, (-0.5, 8.0), scaling_steps=16)
    imaginary_time = tc.backend.jit(
        lambda value: tc.matrixfunc.exponential_action(
            matrix, probes[0], value, chebyshev
        )
    )
    evolved = imaginary_time(tc.backend.convert_to_tensor(-200.0))
    np.testing.assert_allclose(
        evolved,
        np.exp(-200.0 * np.diag(matrix)) * np.asarray(probes[0]),
        atol=2e-12,
    )

    resolvent_config = tc.matrixfunc.ChebyshevConfig(32, (-1.5, 1.5))
    jitted_resolvent = tc.backend.jit(
        lambda broadening: tc.matrixfunc.resolvent_coefficients(
            0.0 + 1.0j * broadening, resolvent_config
        )
    )
    assert np.all(np.isnan(np.asarray(jitted_resolvent(1.0e-6))))
    assert np.all(np.isfinite(np.asarray(jitted_resolvent(1.0))))

    jitted_exponential = tc.backend.jit(
        lambda imaginary: tc.matrixfunc.exponential_coefficients(
            -1.0 + 1.0j * imaginary, resolvent_config
        )
    )
    assert np.all(np.isnan(np.asarray(jitted_exponential(1.0e-300))))


def test_matrixfunc_small_shapes_and_validation_paths(npb):
    matrix = np.diag(np.array([-0.2, 0.6], dtype=np.float32)).astype(np.complex64)
    vector = np.array([1.0, 0.5j], dtype=np.complex64)
    with pytest.raises(TypeError):
        tc.matrixfunc.KrylovConfig(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.KrylovConfig(2, reorthogonalization="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.KrylovConfig(2, breakdown_tol=-1.0)
    with pytest.raises(TypeError):
        tc.matrixfunc.ChebyshevConfig(2.0, (-1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.ChebyshevConfig(0, (-1.0, 1.0))
    with pytest.raises(ValueError):
        tc.matrixfunc.ChebyshevConfig(4, (-1.0, 1.0), kernel="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.ChebyshevConfig(4, (-1.0, 1.0), lorentz_lambda=0.0)
    with pytest.raises(TypeError):
        tc.matrixfunc.TaylorConfig(1.0, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tc.matrixfunc.TaylorConfig(1, 2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.TaylorConfig(-1, 2)
    with pytest.raises(ValueError):
        tc.matrixfunc.TaylorConfig(1, 0)
    with pytest.raises(ValueError):
        tc.matrixfunc.ChebyshevConfig(4, (-1.0,))

    one_dim = tc.matrixfunc.lanczos_project(
        matrix, vector, tc.matrixfunc.KrylovConfig(1, reorthogonalization="none")
    )
    np.testing.assert_allclose(
        tc.matrixfunc.lanczos_apply(one_dim, lambda nodes: np.exp(nodes)),
        np.exp(np.vdot(vector, matrix @ vector) / np.vdot(vector, vector)) * vector,
        atol=3e-5,
    )
    projection = tc.matrixfunc.lanczos_project(
        matrix, vector, tc.matrixfunc.KrylovConfig(2, breakdown_tol=1.0e-5)
    )
    basis = np.asarray(projection.basis)
    recurrence = projection.recurrence
    projected = np.diag(np.asarray(recurrence.diagonal))
    projected += np.diag(np.asarray(recurrence.off_diagonal), k=1)
    projected += np.diag(np.asarray(recurrence.off_diagonal).conj(), k=-1)
    np.testing.assert_allclose(basis @ projected @ basis.conj().T, matrix, atol=3e-5)
    with pytest.raises(TypeError):
        tc.matrixfunc.lanczos_apply(1, np.exp)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tc.matrixfunc.lanczos_quadrature(1, np.exp)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tc.matrixfunc.lanczos_resolvent(1, 0.2)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        tc.matrixfunc.lanczos_resolvent(
            tc.matrixfunc.LanczosMeasure(
                nodes=np.array([[-0.2, 0.6]], dtype=np.float32),
                weights=np.array([[1.0, 0.25]], dtype=np.float32),
                active=np.array([[True, True]]),
                dimension=np.array(2),
            ),
            np.array([0.4 + 0.2j, 0.7 + 0.3j]),
        ),
        np.array(
            [
                1.0 / (0.4 + 0.2j + 0.2) + 0.25 / (0.4 + 0.2j - 0.6),
                1.0 / (0.7 + 0.3j + 0.2) + 0.25 / (0.7 + 0.3j - 0.6),
            ]
        )[:, None],
        atol=3e-5,
    )

    with pytest.raises(ValueError):
        tc.matrixfunc.slq_measure(matrix, vector, tc.matrixfunc.KrylovConfig(2))
    with pytest.raises(ValueError):
        tc.matrixfunc.slq_measure(
            matrix, np.empty((0, 2), dtype=np.complex64), tc.matrixfunc.KrylovConfig(2)
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.slq_measure(
            matrix,
            np.ones((2, 2), dtype=np.complex64),
            tc.matrixfunc.KrylovConfig(2),
            probe_batch_size=0,
        )
    measure = tc.matrixfunc.slq_measure(
        matrix,
        np.sqrt(2.0) * np.eye(2, dtype=np.complex64),
        tc.matrixfunc.KrylovConfig(2),
        probe_batch_size=2,
    )
    one_measure = tc.matrixfunc.LanczosMeasure(
        measure.nodes[:1],
        measure.weights[:1],
        measure.active[:1],
        measure.dimension,
    )
    one_value = tc.matrixfunc.evaluate_slq_trace(one_measure, np.exp)
    assert np.isfinite(np.asarray(one_value))
    with pytest.raises(ValueError, match="standard error"):
        tc.matrixfunc.evaluate_slq_trace(one_measure, np.exp, with_std=True)
    with pytest.raises(ValueError):
        tc.matrixfunc.evaluate_slq_trace(measure, lambda nodes: np.ones((2, 2, 2, 2)))

    order_one = tc.matrixfunc.ChebyshevConfig(1, (-1.0, 1.0))
    np.testing.assert_allclose(
        tc.matrixfunc.chebyshev_action(matrix, vector, np.array([2.0]), order_one),
        2.0 * vector,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tc.matrixfunc.chebyshev_moments(matrix, vector, order_one).moments,
        np.array([np.vdot(vector, vector)]),
        atol=3e-5,
    )
    with pytest.raises(ValueError):
        tc.matrixfunc.kernel_weights(order_one)
    with pytest.raises(TypeError):
        tc.matrixfunc.evaluate_chebyshev_moments(1, np.ones(2))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.chebyshev_coefficients(
            lambda x: np.ones_like(x),
            tc.matrixfunc.ChebyshevConfig(4, (-1, 1)),
            quadrature_order=2,
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.chebyshev_coefficients(
            lambda x: np.ones(2), tc.matrixfunc.ChebyshevConfig(4, (-1, 1))
        )
    with pytest.raises(TypeError):
        tc.matrixfunc.chebyshev_coefficients(lambda x: x, 1)  # type: ignore[arg-type]

    config = tc.matrixfunc.ChebyshevConfig(16, (-1.0, 1.0))
    scalar_fermi = tc.matrixfunc.fermi_dirac_coefficients(0.7, -0.1, config)
    vector_mu = tc.matrixfunc.fermi_dirac_coefficients(
        0.7, np.array([-0.1, 0.2]), config
    )
    vector_beta = tc.matrixfunc.fermi_dirac_coefficients(
        np.array([0.7, 1.2]), -0.1, config
    )
    paired_beta = np.array([0.7, 1.2], dtype=np.float32)
    paired_mu = np.array([-0.1, 0.2], dtype=np.float32)
    paired_fermi = tc.matrixfunc.fermi_dirac_coefficients(
        paired_beta, paired_mu, config
    )
    np.testing.assert_allclose(vector_mu[0], scalar_fermi, atol=3e-5)
    np.testing.assert_allclose(vector_beta[0], scalar_fermi, atol=3e-5)
    np.testing.assert_allclose(
        paired_fermi,
        np.stack(
            [
                tc.matrixfunc.fermi_dirac_coefficients(b, mu, config)
                for b, mu in zip(paired_beta, paired_mu)
            ]
        ),
        atol=3e-5,
    )
    np.testing.assert_allclose(
        tc.matrixfunc.delta_coefficients(0.0, config).shape, (16,)
    )
    with pytest.raises(TypeError):
        tc.matrixfunc.kernel_weights(1)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        tc.matrixfunc.exponential_action(matrix, vector, 1.0, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tc.matrixfunc.estimate_spectral_bounds(
            matrix, vector, tc.matrixfunc.KrylovConfig(2), padding=-1.0
        )


def test_matrixfunc_scalar_prepared_and_shifted_query_paths(npb):
    matrix = np.diag(np.array([-0.2, 0.6], dtype=np.float32)).astype(np.complex64)
    vector = np.array([1.0, 0.5j], dtype=np.complex64)
    probes = np.sqrt(2.0) * np.eye(2, dtype=np.complex64)

    projection = tc.matrixfunc.lanczos_project(
        matrix, vector, tc.matrixfunc.KrylovConfig(2)
    )
    expected_quadrature = np.vdot(
        vector, _dense_function_action(matrix, vector, np.exp)
    )
    np.testing.assert_allclose(
        tc.matrixfunc.lanczos_quadrature(projection, np.exp),
        expected_quadrature,
        atol=3e-5,
    )

    order_two = tc.matrixfunc.ChebyshevConfig(2, (-1.0, 1.0))
    expected_scaled_action = matrix @ vector
    np.testing.assert_allclose(
        tc.matrixfunc.chebyshev_action(
            matrix, vector, np.array([0.0, 1.0], dtype=np.complex64), order_two
        ),
        expected_scaled_action,
        atol=3e-5,
    )
    order_two_moments = tc.matrixfunc.chebyshev_moments(matrix, vector, order_two)
    np.testing.assert_allclose(
        order_two_moments.moments,
        np.array([np.vdot(vector, vector), np.vdot(vector, matrix @ vector)]),
        atol=3e-5,
    )
    order_one = tc.matrixfunc.ChebyshevConfig(1, (-1.0, 1.0))
    rank_one_moments = tc.matrixfunc.chebyshev_moments(matrix, vector, order_one)
    with pytest.raises(ValueError, match="trace estimators"):
        tc.matrixfunc.evaluate_chebyshev_trace(
            rank_one_moments,
            np.array([2.0], dtype=np.complex64),
            with_std=True,
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.evaluate_chebyshev_moments(
            rank_one_moments, np.ones(1), kernel_weights=np.ones(2)
        )

    order_one_trace = tc.matrixfunc.stochastic_chebyshev_moments(
        matrix, probes, order_one
    )
    np.testing.assert_allclose(
        order_one_trace.moments,
        np.full((2, 1), 2.0),
        atol=3e-5,
    )
    order_two_trace = tc.matrixfunc.stochastic_chebyshev_moments(
        matrix, probes, order_two
    )
    np.testing.assert_allclose(
        order_two_trace.moments,
        np.array([[2.0, -0.4], [2.0, 1.2]], dtype=np.complex64),
        atol=3e-5,
    )

    coefficient = np.array([-0.2j, -0.4j], dtype=np.complex64)
    shifted = tc.matrixfunc.exponential_action(
        matrix,
        vector,
        coefficient,
        tc.matrixfunc.KrylovConfig(2),
        energy_shift=0.15,
    )
    expected_shifted = np.stack(
        [
            _dense_function_action(matrix, vector, lambda x, c=c: np.exp(c * x))
            for c in coefficient
        ]
    )
    np.testing.assert_allclose(shifted, expected_shifted, atol=3e-5)

    np.testing.assert_allclose(
        tc.matrixfunc.estimate_spectral_bounds(
            matrix, vector, tc.matrixfunc.KrylovConfig(2), padding=0.1
        ),
        (-0.28, 0.68),
        atol=3e-5,
    )

    with pytest.raises(TypeError):
        tc.matrixfunc.slq_measure(matrix, probes, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tc.matrixfunc.evaluate_slq_trace(1, np.exp)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tc.matrixfunc.chebyshev_action(
            matrix, vector, np.ones(1), 1  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        tc.matrixfunc.chebyshev_moments(matrix, vector, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tc.matrixfunc.stochastic_chebyshev_moments(
            matrix, probes, 1  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.stochastic_chebyshev_moments(matrix, probes[0], order_one)
    with pytest.raises(ValueError):
        tc.matrixfunc.stochastic_chebyshev_moments(
            matrix, np.empty((0, 2), dtype=np.complex64), order_one
        )
    with pytest.raises(ValueError):
        tc.matrixfunc.stochastic_chebyshev_moments(
            matrix, probes, order_one, probe_batch_size=0
        )
    measure = tc.matrixfunc.slq_measure(matrix, probes, tc.matrixfunc.KrylovConfig(2))
    with pytest.raises(ValueError):
        tc.matrixfunc.evaluate_slq_trace(measure, lambda nodes: np.ones((2, 3)))
    with pytest.raises(ValueError):
        tc.matrixfunc.evaluate_slq_trace(measure, lambda nodes: np.ones((3, 2, 3)))
