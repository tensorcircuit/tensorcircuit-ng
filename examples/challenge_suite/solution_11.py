"""
Challenge Suite Problem 11: mixed-mode differentiable quantum sensor training.

The solution computes field-response columns by forward-mode differentiation and
optimizes the trainable probe circuit by reverse-mode differentiation.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")


def sensor_positions(config):
    return K.convert_to_tensor(
        np.linspace(-1.0, 1.0, config["n_qubits"], dtype=np.float32)
    )


def target_response_matrix(config, x_sites):
    return K.stack([K.ones_like(x_sites), x_sites], axis=1)


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    n_qubits = config["n_qubits"]
    n_pairs = config["n_layers"] // 2
    n_even = len(range(0, n_qubits - 1, 2))
    n_odd = len(range(1, n_qubits - 1, 2))
    scale = config["initial_parameter_scale"]

    def normal(shape):
        return K.convert_to_tensor(
            rng.normal(scale=scale, size=shape).astype(np.float32)
        )

    return {
        "even": {
            "rot": normal((n_pairs, n_qubits, 2)),
            "rzz": normal((n_pairs, n_even)),
        },
        "odd": {
            "rot": normal((n_pairs, n_qubits, 2)),
            "rzz": normal((n_pairs, n_odd)),
        },
    }


def probe_state(params, config):
    n_qubits = config["n_qubits"]
    even_bonds = list(range(0, n_qubits - 1, 2))
    odd_bonds = list(range(1, n_qubits - 1, 2))

    def apply_layer(state, rotations, rzz_params, bonds):
        circuit = tc.Circuit(n_qubits, inputs=state)
        for i in range(n_qubits):
            circuit.rz(i, theta=rotations[i, 0])
            circuit.ry(i, theta=rotations[i, 1])
        for j, i in enumerate(bonds):
            circuit.rzz(i, i + 1, theta=rzz_params[j])
        return circuit.state()

    def apply_pair(state, pair_params):
        state = apply_layer(
            state,
            pair_params["even"]["rot"],
            pair_params["even"]["rzz"],
            even_bonds,
        )
        return apply_layer(
            state,
            pair_params["odd"]["rot"],
            pair_params["odd"]["rzz"],
            odd_bonds,
        )

    initial_state = tc.Circuit(n_qubits).state()
    return K.scan(apply_pair, params, initial_state)


def pytree_norm(tree):
    leaves, _ = K.tree_flatten(tree)
    squared_norms = [K.norm(leaf) ** 2 for leaf in leaves]
    return K.sum(K.stack(squared_norms)) ** 0.5


def x_readouts_after_sensing(state, fields, config, x_sites):
    n_qubits = config["n_qubits"]
    psi = K.reshape(state, [2] * n_qubits)
    readouts = []
    for i in range(n_qubits):
        perm = [i] + [j for j in range(n_qubits) if j != i]
        pair = K.reshape(K.transpose(psi, perm), [2, -1])
        coherence = K.sum(K.conj(pair[0]) * pair[1])
        alpha = fields[0] + fields[1] * x_sites[i]
        phase = K.exp(2.0j * K.cast(alpha, "complex64"))
        readouts.append(2.0 * K.real(phase * coherence))
    return K.stack(readouts)


def response_and_readout(params, config, x_sites):
    state = probe_state(params, config)
    zero_fields = K.zeros([config["n_field_params"]])

    def field_readouts(fields):
        return x_readouts_after_sensing(state, fields, config, x_sites)

    response = K.real(K.jacfwd(field_readouts)(zero_fields))
    zero_readouts = field_readouts(zero_fields)
    return response, zero_readouts


def loss_and_metrics(params, config, x_sites, target):
    response, zero_readouts = response_and_readout(params, config, x_sites)
    response_mse = K.mean((response - target) ** 2)
    readout_penalty = K.mean(zero_readouts**2)
    loss = response_mse + config["readout_penalty_weight"] * readout_penalty
    return K.real(loss), (
        K.real(response_mse),
        K.real(readout_penalty),
        response,
        zero_readouts,
    )


def run_solution(config):
    x_sites = sensor_positions(config)
    target = target_response_matrix(config, x_sites)
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def objective(p):
        return loss_and_metrics(p, config, x_sites, target)

    value_and_grad = K.jit(K.value_and_grad(objective, has_aux=True))

    loss_history = []
    response_mse_history = []
    readout_penalty_history = []
    for _ in range(config["max_steps"]):
        (loss, aux), grads = value_and_grad(params)
        response_mse, readout_penalty, response, zero_readouts = aux
        loss_history.append(loss)
        response_mse_history.append(response_mse)
        readout_penalty_history.append(readout_penalty)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        "loss_history": K.numpy(K.stack(loss_history)),
        "response_mse_history": K.numpy(K.stack(response_mse_history)),
        "readout_penalty_history": K.numpy(K.stack(readout_penalty_history)),
        "final_response_matrix": K.numpy(response),
        "final_zero_field_readouts": K.numpy(zero_readouts),
        "final_grad_norm": K.numpy(pytree_norm(grads)),
    }
