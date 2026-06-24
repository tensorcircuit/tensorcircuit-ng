"""
Challenge Suite Problem 5: custom non-unitary gate cooling.

The cooling filters are implemented as ordinary RX/RZZ gates with imaginary
angles. The solution returns only NumPy values consumed by evaluate_5.py.
"""

import jax
import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def initial_parameters(config):
    value = config["initial_filter_strength"]
    initial = np.full((config["n_layers"] // 2, 2), value, dtype=np.float32)
    return {
        "a": K.convert_to_tensor(initial),
        "b": K.convert_to_tensor(initial),
    }


def initial_state(config):
    circuit = tc.Circuit(config["n_qubits"])
    for i in range(config["n_qubits"]):
        circuit.h(i)
    return circuit.state()


def apply_filter_layer(state, a, b, bonds, config):
    circuit = tc.Circuit(config["n_qubits"], inputs=state)
    for i in range(config["n_qubits"]):
        circuit.rx(i, theta=2.0j * a)
    for i in bonds:
        circuit.rzz(i, i + 1, theta=2.0j * b)
    state = circuit.state()
    return state / K.norm(state)


def cooling_trajectory(params, input_state, config):
    even = list(range(0, config["n_qubits"] - 1, 2))
    odd = list(range(1, config["n_qubits"] - 1, 2))

    def block_step(state, p):
        state = apply_filter_layer(state, p["a"][0], p["b"][0], even, config)
        state = apply_filter_layer(state, p["a"][1], p["b"][1], odd, config)
        return state, None

    final_state, _ = jax.lax.scan(block_step, input_state, params)
    return final_state


def build_tfim_mvp(config):
    structures = []
    weights = []

    for i in range(config["n_qubits"] - 1):
        term = [0] * config["n_qubits"]
        term[i] = 3
        term[i + 1] = 3
        structures.append(term)
        weights.append(-1.0)

    for i in range(config["n_qubits"]):
        term = [0] * config["n_qubits"]
        term[i] = 1
        structures.append(term)
        weights.append(-config["transverse_field"])

    return tc.quantum.PauliStringSum2MVP(structures, weights)


def tfim_energy(state, hamiltonian_mvp):
    h_state = hamiltonian_mvp(state)
    return K.real(K.tensordot(K.conj(state), h_state, 1))


def energy_density(params, input_state, hamiltonian_mvp, config):
    final_state = cooling_trajectory(params, input_state, config)
    return tfim_energy(final_state, hamiltonian_mvp) / config["n_qubits"]


def run_solution(config):
    params = initial_parameters(config)
    input_state = initial_state(config)
    hamiltonian_mvp = build_tfim_mvp(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return energy_density(p, input_state, hamiltonian_mvp, config)

    value_and_grad = K.jit(K.value_and_grad(loss_fn))

    energy_density_history = []
    for _ in range(config["max_steps"]):
        energy, grads = value_and_grad(params)
        energy_density_history.append(energy)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        "initial_energy_density": K.numpy(energy_density_history[0]),
        "final_energy_density": K.numpy(energy_density_history[-1]),
        "final_a": K.numpy(params["a"]),
        "final_b": K.numpy(params["b"]),
        "energy_density_history": K.numpy(K.stack(energy_density_history)),
    }
