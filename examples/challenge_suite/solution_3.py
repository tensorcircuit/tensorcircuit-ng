"""
Challenge Suite Problem 3: probability-aware post-selected cooling.

The selected measurement branch is represented with Circuit.post_select. The
solution returns only NumPy values consumed by evaluate_3.py.
"""

import jax
import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def block_count(config):
    return config["n_steps"] // 2


def even_bonds(config):
    return list(range(0, config["n_qubits"] - 1, 2))


def odd_bonds(config):
    return list(range(1, config["n_qubits"] - 1, 2))


def measured_qubits(config):
    return list(range(0, config["n_qubits"], 2))


def initial_parameters(config):
    rng = np.random.default_rng(2027)
    n_repeats = block_count(config)
    n_qubits = config["n_qubits"]
    n_even = len(even_bonds(config))
    n_odd = len(odd_bonds(config))

    def normal(shape):
        return K.convert_to_tensor(
            rng.normal(scale=0.02, size=shape).astype(np.float32)
        )

    return {
        "even": {
            "rx": normal((n_repeats, n_qubits)),
            "xx": normal((n_repeats, n_even)),
            "zz": normal((n_repeats, n_even)),
        },
        "odd": {
            "rx": normal((n_repeats, n_qubits)),
            "xx": normal((n_repeats, n_odd)),
            "zz": normal((n_repeats, n_odd)),
        },
    }


def initial_state(config):
    circuit = tc.Circuit(config["n_qubits"])
    for i in range(config["n_qubits"]):
        circuit.h(i)
    return circuit.state()


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


def apply_unitary_step(state, step_params, bonds, config):
    circuit = tc.Circuit(config["n_qubits"], inputs=state)
    for bond_index, i in enumerate(bonds):
        circuit.rxx(i, i + 1, theta=step_params["xx"][bond_index])
        circuit.rzz(i, i + 1, theta=step_params["zz"][bond_index])
    for i in range(config["n_qubits"]):
        circuit.rx(i, theta=step_params["rx"][i])
    return circuit.state()


def inner_product(left, right):
    return K.tensordot(K.conj(left), right, 1)


def apply_postselected_measurements(state, config):
    log_probabilities = []

    for q in measured_qubits(config):
        circuit = tc.Circuit(config["n_qubits"], inputs=state)
        circuit.post_select(q, keep=0)
        state = circuit.state()
        probability = K.real(inner_product(state, state))
        state = state / K.sqrt(probability + 1e-12)
        log_probabilities.append(K.log(probability + 1e-12))

    return state, K.stack(log_probabilities)


def cooling_trajectory(params, input_state, config):
    even = even_bonds(config)
    odd = odd_bonds(config)

    def block_step(state, p):
        state = apply_unitary_step(state, p["even"], even, config)
        state, even_logps = apply_postselected_measurements(state, config)
        state = apply_unitary_step(state, p["odd"], odd, config)
        state, odd_logps = apply_postselected_measurements(state, config)
        return state, K.concat([even_logps, odd_logps])

    final_state, logps = jax.lax.scan(block_step, input_state, params)
    return final_state, K.reshape(logps, [-1])


def tfim_energy(state, hamiltonian_mvp):
    h_state = hamiltonian_mvp(state)
    return K.real(inner_product(state, h_state))


def observables(params, input_state, hamiltonian_mvp, config):
    final_state, log_probabilities = cooling_trajectory(params, input_state, config)
    energy_density = tfim_energy(final_state, hamiltonian_mvp) / config["n_qubits"]
    mean_log_probability = K.mean(log_probabilities)
    success_probability = K.exp(K.sum(log_probabilities))
    loss = energy_density - config["log_probability_weight"] * mean_log_probability
    return loss, (energy_density, success_probability, mean_log_probability)


def run_solution(config):
    params = initial_parameters(config)
    input_state = initial_state(config)
    hamiltonian_mvp = build_tfim_mvp(config)
    optimizer = K.optimizer(optax.adam(config["learning_rate"]))

    def loss_fn(p):
        return observables(p, input_state, hamiltonian_mvp, config)

    value_and_grad = K.jit(K.value_and_grad(loss_fn, has_aux=True))

    energy_density_history = []
    success_probability_history = []
    mean_log_probability_history = []
    loss_history = []

    for _ in range(config["max_steps"]):
        (loss, aux), grads = value_and_grad(params)
        energy_density, success_probability, mean_log_probability = aux
        energy_density_history.append(energy_density)
        success_probability_history.append(success_probability)
        mean_log_probability_history.append(mean_log_probability)
        loss_history.append(loss)
        params = optimizer.update(grads, params)

    (final_loss, final_aux), _ = value_and_grad(params)
    final_energy_density, final_success, final_mean_log_probability = final_aux

    return {
        "initial_energy_density": K.numpy(energy_density_history[0]),
        "final_energy_density": K.numpy(final_energy_density),
        "final_success_probability": K.numpy(final_success),
        "final_mean_log_probability": K.numpy(final_mean_log_probability),
        "initial_loss": K.numpy(loss_history[0]),
        "final_loss": K.numpy(final_loss),
        "energy_density_history": K.numpy(K.stack(energy_density_history)),
        "success_probability_history": K.numpy(K.stack(success_probability_history)),
        "mean_log_probability_history": K.numpy(K.stack(mean_log_probability_history)),
        "loss_history": K.numpy(K.stack(loss_history)),
    }
