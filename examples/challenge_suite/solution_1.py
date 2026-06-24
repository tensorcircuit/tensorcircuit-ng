"""
Challenge Suite Problem 1: DMRG-MPS input with variational circuit refinement.

The DMRG state is injected into a regular TensorCircuit Circuit. The solution
returns NumPy values only; external validation lives in evaluate_1.py.
"""

import numpy as np
import optax

import tensorcircuit as tc
from tensorcircuit.templates.measurements import parameterized_measurements

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")


def parameter_count(config):
    count = 0
    for layer in range(config["n_layers"]):
        count += 3 * config["n_qubits"]
        count += 3 * len(range(layer % 2, config["n_qubits"] - 1, 2))
    return count


def initial_parameters(config):
    rng = np.random.default_rng(1234)
    params = rng.normal(scale=1e-4, size=parameter_count(config)).astype(np.float32)
    return K.convert_to_tensor(params)


def tfim_measurement_data(config):
    patterns = []
    weights = []

    for i in range(config["n_qubits"] - 1):
        pattern = [0] * config["n_qubits"]
        pattern[i] = 3
        pattern[i + 1] = 3
        patterns.append(pattern)
        weights.append(-1.0)

    for i in range(config["n_qubits"]):
        pattern = [0] * config["n_qubits"]
        pattern[i] = 1
        patterns.append(pattern)
        weights.append(-config["field"])

    return (
        K.convert_to_tensor(np.array(patterns, dtype=np.int32)),
        K.convert_to_tensor(np.array(weights, dtype=np.float32)),
    )


def apply_variational_layers(circuit, params, config):
    offset = 0
    for layer in range(config["n_layers"]):
        for i in range(config["n_qubits"]):
            circuit.rz(i, theta=params[offset])
            circuit.ry(i, theta=params[offset + 1])
            circuit.rz(i, theta=params[offset + 2])
            offset += 3

        for i in range(layer % 2, config["n_qubits"] - 1, 2):
            circuit.rxx(i, i + 1, theta=params[offset])
            circuit.ryy(i, i + 1, theta=params[offset + 1])
            circuit.rzz(i, i + 1, theta=params[offset + 2])
            offset += 3


def circuit_energy(params, mps_input, config, patterns, weights):
    circuit = tc.Circuit(config["n_qubits"], mps_inputs=mps_input)
    apply_variational_layers(circuit, params, config)

    def measure(pattern):
        return parameterized_measurements(circuit, pattern, onehot=True, reuse=False)

    expectations = K.vmap(measure, vectorized_argnums=0)(patterns)
    return K.sum(expectations * weights)


def run_solution(config):
    mps_input = tc.quantum.quimb2qop(config["dmrg_state"])
    params = initial_parameters(config)
    patterns, weights = tfim_measurement_data(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    energy_fn = lambda p, m: circuit_energy(p, m, config, patterns, weights)
    value_and_grad = K.jit(K.value_and_grad(energy_fn), static_argnums=(1,))

    energy, grads = value_and_grad(params, mps_input)
    grad_norm = K.norm(grads)

    energy_history = []
    grad_norm_history = []
    for step in range(config["max_steps"]):
        energy_history.append(energy)
        grad_norm_history.append(grad_norm)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step + 1 < config["max_steps"]:
            energy, grads = value_and_grad(params, mps_input)
            grad_norm = K.norm(grads)

    return {
        "energy_history": K.numpy(K.stack(energy_history)),
        "grad_norm_history": K.numpy(K.stack(grad_norm_history)),
    }
