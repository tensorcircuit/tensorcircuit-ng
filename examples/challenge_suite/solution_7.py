"""
Challenge Suite Problem 7: 50-qubit two-excitation VQE.

The variational state is evolved with U1Circuit in the compressed two-particle
sector. The solution returns only NumPy values consumed by evaluate_7.py.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex128")


def hamiltonian_terms(config):
    n_qubits = config["n_qubits"]
    sites = np.arange(n_qubits)
    chemical = 0.25 * np.cos(2.0 * np.pi * sites / n_qubits) + 0.10 * ((-1.0) ** sites)
    interaction = config["interaction"]

    ps_list = [{}]
    coefficients = [0.5 * np.sum(chemical) + 0.25 * interaction * (n_qubits - 1)]

    z_coefficients = -0.5 * chemical
    z_coefficients[0] -= 0.25 * interaction
    z_coefficients[-1] -= 0.25 * interaction
    z_coefficients[1:-1] -= 0.5 * interaction
    for i, coefficient in enumerate(z_coefficients):
        ps_list.append({"z": [i]})
        coefficients.append(coefficient)

    for i in range(n_qubits - 1):
        ps_list.extend(
            [
                {"x": [i, i + 1]},
                {"y": [i, i + 1]},
                {"z": [i, i + 1]},
            ]
        )
        coefficients.extend([-0.5, -0.5, 0.25 * interaction])

    return ps_list, K.convert_to_tensor(np.asarray(coefficients))


def initial_parameters(config):
    rng = np.random.default_rng(2031)
    n_blocks = config["n_layers"] // 2
    n_even = len(range(0, config["n_qubits"] - 1, 2))
    n_odd = len(range(1, config["n_qubits"] - 1, 2))
    scale = config["initial_parameter_scale"]

    def normal(shape):
        return K.convert_to_tensor(rng.normal(scale=scale, size=shape))

    return {
        "even": {
            "mix": normal((n_blocks, n_even)),
            "left": normal((n_blocks, n_even)),
            "right": normal((n_blocks, n_even)),
        },
        "odd": {
            "mix": normal((n_blocks, n_odd)),
            "left": normal((n_blocks, n_odd)),
            "right": normal((n_blocks, n_odd)),
        },
    }


def apply_bond_layer(circuit, layer_params, bonds):
    for bond_index, i in enumerate(bonds):
        circuit.iswap(i, i + 1, theta=layer_params["mix"][bond_index])
        circuit.rz(i, theta=layer_params["left"][bond_index])
        circuit.rz(i + 1, theta=layer_params["right"][bond_index])


def build_state(params, config):
    even = list(range(0, config["n_qubits"] - 1, 2))
    odd = list(range(1, config["n_qubits"] - 1, 2))
    input_state = tc.U1Circuit(
        config["n_qubits"],
        k=config["n_particles"],
        filled=config["initial_occupied"],
    ).state()

    def block_step(state, block_params):
        circuit = tc.U1Circuit(
            config["n_qubits"], k=config["n_particles"], inputs=state
        )
        apply_bond_layer(circuit, block_params["even"], even)
        apply_bond_layer(circuit, block_params["odd"], odd)
        return circuit.state()

    return K.scan(block_step, params, input_state)


def energy(params, config, ps_list, coefficients):
    state = build_state(params, config)
    circuit = tc.U1Circuit(config["n_qubits"], k=config["n_particles"], inputs=state)
    return circuit.expectation_pss(ps_list, coefficients)


def run_solution(config):
    ps_list, coefficients = hamiltonian_terms(config)
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return energy(p, config, ps_list, coefficients)

    def train_step(p, state):
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, value

    train_step = K.jit(train_step)

    energy_history = []
    for _ in range(config["max_steps"]):
        params, opt_state, value = train_step(params, opt_state)
        energy_history.append(value)

    return {
        "energy_history": K.numpy(K.stack(energy_history)),
    }
