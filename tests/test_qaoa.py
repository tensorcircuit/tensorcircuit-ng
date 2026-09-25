import sys
import os

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
from scipy.linalg import expm

import tensorcircuit as tc

from tensorcircuit.applications.dqas import set_op_pool
from tensorcircuit.applications.graphdata import get_graph
from tensorcircuit.applications.layers import Hlayer, rxlayer, zzlayer
from tensorcircuit.applications.vags import evaluate_vag
from tensorcircuit.templates.ansatz import QAOA_ansatz_for_Ising
from tensorcircuit.circuit import Circuit


def test_vag(tfb):
    set_op_pool([Hlayer, rxlayer, zzlayer])
    params = np.array([0.0, 0.3, 0.5, 0.7, -0.8])
    preset = [0, 2, 1, 2, 1]
    graph = get_graph("10A")
    expene, ene, eneg, p = evaluate_vag(
        params, preset, graph, lbd=0, overlap_threhold=11
    )
    print(expene, eneg, p)
    np.testing.assert_allclose(ene.numpy(), -7.01, rtol=1e-2)
    # finite-difference reference for the energy gradient (eneg)
    eps = 1e-3
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        dp = np.zeros_like(params)
        dp[i] = eps
        _, ene_plus, _, _ = evaluate_vag(
            params + dp, preset, graph, lbd=0, overlap_threhold=11
        )
        _, ene_minus, _, _ = evaluate_vag(
            params - dp, preset, graph, lbd=0, overlap_threhold=11
        )
        fd_grad[i] = (ene_plus.numpy() - ene_minus.numpy()) / (2 * eps)
    np.testing.assert_allclose(eneg.numpy(), fd_grad, atol=1e-2)


cases = [
    ("X", True),
    ("X", False),
    ("XY", True),
    ("XY", False),
    ("ZZ", True),
    ("ZZ", False),
]


@pytest.fixture
def example_inputs():
    params = [0.1, 0.2, 0.3, 0.4]
    nlayers = 2
    pauli_terms = [[0, 1, 0], [1, 0, 1]]
    weights = [0.5, -0.5]
    return params, nlayers, pauli_terms, weights


@pytest.mark.parametrize("mixer, full_coupling", cases)
def test_QAOA_ansatz_for_Ising(example_inputs, full_coupling, mixer):
    params, nlayers, pauli_terms, weights = example_inputs
    circuit = QAOA_ansatz_for_Ising(
        params, nlayers, pauli_terms, weights, full_coupling, mixer
    )
    n = len(pauli_terms[0])
    assert isinstance(circuit, Circuit)
    assert circuit._nqubits == n

    if mixer == "X":
        assert circuit.gate_count() == n + nlayers * (len(pauli_terms) + n)
    elif mixer == "XY":
        if full_coupling is False:
            assert circuit.gate_count() == n + nlayers * (len(pauli_terms) + 2 * n)
        else:
            assert circuit.gate_count() == n + nlayers * (
                len(pauli_terms) + sum(range(n + 1))
            )
    else:
        if full_coupling is False:
            assert circuit.gate_count() == n + nlayers * (len(pauli_terms) + n)
        else:
            assert circuit.gate_count() == n + nlayers * (
                len(pauli_terms) + sum(range(n + 1)) / 2
            )


@pytest.mark.parametrize("mixer, full_coupling", [("AB", True), ("XY", 1), ("TC", 5)])
def test_QAOA_ansatz_errors(example_inputs, full_coupling, mixer):
    params, nlayers, pauli_terms, weights = example_inputs
    with pytest.raises(ValueError):
        QAOA_ansatz_for_Ising(
            params, nlayers, pauli_terms, weights, full_coupling, mixer
        )


def _ising_qaoa_reference(params, pauli_terms, weights):
    n = len(pauli_terms[0])
    basis = np.indices([2] * n).reshape(n, -1).T
    spins = 1 - 2 * basis
    diagonal = sum(
        weight * np.prod(spins[:, np.flatnonzero(term)], axis=1)
        for term, weight in zip(pauli_terms, weights)
    )
    mixer = np.zeros((2**n, 2**n))
    for qubit in range(n):
        operator = np.ones((1, 1))
        for site in range(n):
            operator = np.kron(
                operator, np.array([[0, 1], [1, 0]]) if site == qubit else np.eye(2)
            )
        mixer += operator
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    for gamma, beta in np.asarray(params).reshape(-1, 2):
        state = np.exp(-1j * gamma * diagonal) * state
        state = expm(-0.5j * beta * mixer) @ state
    return state, diagonal


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb"), lf("torchb")])
@pytest.mark.parametrize("double_precision", [False, True])
@pytest.mark.parametrize("nlayers", [1, 2])
@pytest.mark.parametrize(
    "pauli_terms, weights",
    [
        ([[1, 0, 0], [0, 0, 1]], [0.7, -0.4]),
        ([[1, 1, 0], [1, 0, 1]], [-0.6, 0.9]),
        ([[1, 0, 0], [0, 0, 1], [1, 0, 1]], [0.7, -0.4, 1.2]),
    ],
    ids=["Z", "ZZ", "mixed"],
)
def test_ising_qaoa_exact_evolution(
    backend, double_precision, nlayers, pauli_terms, weights, request
):
    if double_precision:
        request.getfixturevalue("highp")
    params = np.array([0.37, -0.21, -0.43, 0.19][: 2 * nlayers], dtype=tc.rdtypestr)
    expected, _ = _ising_qaoa_reference(params, pauli_terms, weights)
    tensor_params = tc.backend.convert_to_tensor(params)

    def state(parameters):
        return QAOA_ansatz_for_Ising(parameters, nlayers, pauli_terms, weights).state()

    tolerance = 1e-10 if double_precision else 5e-6
    np.testing.assert_allclose(
        tc.backend.numpy(state(tensor_params)), expected, atol=tolerance, rtol=0
    )
    if tc.backend.name in ("jax", "tensorflow"):
        np.testing.assert_allclose(
            tc.backend.numpy(tc.backend.jit(state)(tensor_params)),
            expected,
            atol=tolerance,
            rtol=0,
        )


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb"), lf("torchb")])
@pytest.mark.parametrize("nlayers", [1, 2])
@pytest.mark.parametrize("gamma", [0.0, 0.37])
def test_ising_qaoa_gradients(backend, nlayers, gamma, highp):
    pauli_terms = [[1, 0, 0], [0, 0, 1], [1, 0, 1]]
    params = np.array([gamma, 0.43, -0.21, 0.17][: 2 * nlayers])
    weights = np.array([0.7, -0.4, 1.2])

    def reference_energy(parameters, coefficients):
        state, diagonal = _ising_qaoa_reference(parameters, pauli_terms, coefficients)
        return np.dot(diagonal, np.abs(state) ** 2)

    def energy(parameters, coefficients):
        circuit = QAOA_ansatz_for_Ising(parameters, nlayers, pauli_terms, coefficients)
        return sum(
            coefficients[k]
            * tc.backend.real(circuit.expectation_ps(z=np.flatnonzero(term).tolist()))
            for k, term in enumerate(pauli_terms)
        )

    expected_gradients = []
    step = 1e-5
    for argnum, values in enumerate((params, weights)):
        gradient = np.empty_like(values)
        for index in range(len(values)):
            plus = [params.copy(), weights.copy()]
            minus = [params.copy(), weights.copy()]
            plus[argnum][index] += step
            minus[argnum][index] -= step
            gradient[index] = (reference_energy(*plus) - reference_energy(*minus)) / (
                2 * step
            )
        expected_gradients.append(gradient)

    value_and_grad = tc.backend.value_and_grad(energy, argnums=(0, 1))
    functions = [value_and_grad]
    if tc.backend.name in ("jax", "tensorflow"):
        functions.append(tc.backend.jit(value_and_grad))
    for function in functions:
        value, gradients = function(
            tc.backend.convert_to_tensor(params), tc.backend.convert_to_tensor(weights)
        )
        np.testing.assert_allclose(
            tc.backend.numpy(value), reference_energy(params, weights), atol=1e-10
        )
        for actual, expected in zip(gradients, expected_gradients):
            np.testing.assert_allclose(
                tc.backend.numpy(actual), expected, atol=2e-8, rtol=1e-6
            )


def test_ising_qaoa_qubo_cost(npb):
    matrix = np.array([[1.0, 0.4, -0.2], [0.4, -0.7, 0.3], [-0.2, 0.3, 0.5]])
    pauli_terms, weights, offset = tc.templates.conversions.QUBO_to_Ising(matrix)
    gamma = 0.31
    circuit = QAOA_ansatz_for_Ising([gamma, 0.0], 1, pauli_terms, weights)
    basis = np.indices([2] * 3).reshape(3, -1).T
    energies = np.einsum("bi,ij,bj->b", basis, matrix, basis)
    expected = np.exp(-1j * gamma * (energies - offset)) / np.sqrt(8)
    np.testing.assert_allclose(circuit.state(), expected, atol=1e-6, rtol=0)
