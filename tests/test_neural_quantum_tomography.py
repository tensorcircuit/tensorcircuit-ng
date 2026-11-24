"""Independent numerical checks for the Torlai W-state reproduction kernels."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tensorcircuit as tc

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "reproduce_papers"
    / "2018_neural_quantum_tomography"
    / "main.py"
)
SPEC = importlib.util.spec_from_file_location("neural_quantum_tomography", SOURCE)
assert SPEC is not None and SPEC.loader is not None
tomography = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tomography)


@pytest.mark.parametrize("n", [4, 8])
def test_w_preparation(npb, highp, n):
    phases = np.random.default_rng(10).uniform(-np.pi, np.pi, n)
    for angles in (None, phases):
        actual = tomography.w_circuit(n, angles).state()
        expected = tomography.analytic_w(n, angles)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_rbm_hidden_sum(jaxb, highp):
    n = 3
    visible = tomography.configurations(n)
    hidden = tomography.configurations(n)
    parameters = np.random.default_rng(20).normal(0, 0.3, n * (n + 2))
    bias, fields = parameters[:n], parameters[n : 2 * n]
    weights = parameters[2 * n :].reshape(n, n)
    energy = visible @ bias[:, None] + hidden @ fields
    energy = energy + visible @ weights.T @ hidden.T
    marginal = np.exp(energy).sum(axis=1)
    expected = marginal / marginal.sum()
    actual = tc.backend.exp(
        tomography.normalized_log_probability(
            tc.backend.convert_to_tensor(parameters),
            tc.backend.convert_to_tensor(visible),
        )
    )
    np.testing.assert_allclose(tc.backend.numpy(actual), expected, atol=1e-12)


def test_pair_rotations(jaxb, highp):
    n = 4
    rng = np.random.default_rng(30)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    state /= np.linalg.norm(state)
    for pair in range(n - 1):
        for axis in ("X", "Y"):
            actual = tomography.pair_rotation(
                tc.backend.convert_to_tensor(state), n, pair, axis
            )
            circuit = tc.Circuit(n, inputs=tc.backend.convert_to_tensor(state))
            circuit.h(pair)
            if axis == "Y":
                circuit.rz(pair + 1, theta=-np.pi / 2)
            circuit.h(pair + 1)
            expected = circuit.state()
            if axis == "Y":
                expected = expected * np.exp(-1j * np.pi / 4)
            np.testing.assert_allclose(
                tc.backend.numpy(actual), tc.backend.numpy(expected), atol=1e-10
            )


@pytest.mark.parametrize("objective", ["amplitude", "joint", "phase_warmup"])
def test_likelihood_gradients(jaxb, highp, objective):
    model = tomography.RBMTomography(3)
    rng = np.random.default_rng(40)
    parameter_count = model.parameter_count * (2 if objective == "joint" else 1)
    parameters = rng.normal(0, 0.3, parameter_count)
    frequencies = (
        rng.dirichlet(np.ones(8), size=5)
        if objective != "amplitude"
        else rng.dirichlet(np.ones(8))
    )
    if objective == "phase_warmup":
        amplitude = tc.backend.convert_to_tensor(
            rng.normal(0, 0.3, model.parameter_count)
        )

        def loss_function(parameters, frequency):
            return model.phase_warmup_loss(parameters, amplitude, frequency)

        value_grad = tc.backend.jit(tc.backend.value_and_grad(loss_function))
    elif objective == "joint":
        loss_function = model.joint_loss
        value_grad = model.joint_value_grad
    else:
        loss_function = model.amplitude_loss
        value_grad = model.amplitude_value_grad
    frequency = tc.backend.convert_to_tensor(frequencies)
    _, gradient = value_grad(tc.backend.convert_to_tensor(parameters), frequency)
    loss = tc.backend.jit(loss_function)
    numerical = []
    for index in range(parameter_count):
        shift = np.zeros(parameter_count)
        shift[index] = 1e-5
        plus = loss(tc.backend.convert_to_tensor(parameters + shift), frequency)
        minus = loss(tc.backend.convert_to_tensor(parameters - shift), frequency)
        numerical.append(float(tc.backend.numpy(plus - minus)) / 2e-5)
    np.testing.assert_allclose(
        tc.backend.numpy(gradient), numerical, rtol=2e-4, atol=2e-6
    )


def test_phase_identifiability_and_y_sign(jaxb, highp):
    n = 4
    phases = np.array([0, 0.6, -1.1, 2.2])
    positive = tomography.analytic_w(n)
    phased = tomography.w_circuit(n, phases).state()
    np.testing.assert_allclose(
        np.abs(tc.backend.numpy(phased)) ** 2, np.abs(positive) ** 2, atol=1e-10
    )
    circuit = tc.Circuit(n, inputs=phased)
    np.testing.assert_allclose(
        tc.backend.numpy(circuit.expectation_ps(x=[0, 1])),
        2 * np.cos(0.6) / n,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        tc.backend.numpy(circuit.expectation_ps(x=[0], y=[1])),
        2 * np.sin(0.6) / n,
        atol=1e-10,
    )
    orthogonal = tomography.analytic_w(n, np.array([0, np.pi, 0, np.pi]))
    np.testing.assert_allclose(
        np.abs(orthogonal) ** 2, np.abs(positive) ** 2, atol=1e-10
    )
    np.testing.assert_allclose(np.vdot(positive, orthogonal), 0, atol=1e-10)


@pytest.mark.parametrize("phased", [False, True])
def test_measurement_shots(jaxb, highp, phased):
    n = 4
    angles = np.array([0, 0.6, -1.1, 2.2]) if phased else None
    state = tomography.w_circuit(n, angles).state()
    streams = tomography.measurement_streams(state, n, 10000, np.random.default_rng(50))
    counts = tomography.measurement_counts(streams, n)
    np.testing.assert_array_equal(counts.sum(axis=1), np.full(2 * n - 1, 10000))
    exact = tomography.basis_probabilities(state, n)
    np.testing.assert_allclose(counts / 10000, tc.backend.numpy(exact), atol=0.03)
