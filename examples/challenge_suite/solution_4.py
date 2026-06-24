"""
Challenge Suite Problem 4: trainable Kraus noise calibration.

The asymmetric bit-flip channel is implemented as user-defined Kraus tensor
algebra. The solution returns only NumPy values consumed by evaluate_4.py.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
PROBE_COUNT = 4


def raw_from_probability(value):
    return K.convert_to_tensor(np.asarray(np.log(value / (1.0 - value)), np.float32))


def probabilities(raw_params):
    return K.sigmoid(raw_params[0]), K.sigmoid(raw_params[1])


def initial_parameters(config):
    return K.stack(
        [
            raw_from_probability(config["initial_p01"]),
            raw_from_probability(config["initial_p10"]),
        ]
    )


def observable_signs(config):
    basis = np.arange(2 ** config["n_qubits"], dtype=np.uint32)
    signs = []
    for q in range(config["n_qubits"]):
        bits = (basis >> (config["n_qubits"] - 1 - q)) & 1
        signs.append(1.0 - 2.0 * bits.astype(np.float32))
    signs.append(np.prod(np.stack(signs), axis=0))
    return K.convert_to_tensor(np.stack(signs))


def asymmetric_bitflip_kraus(p01, p10):
    zero = K.cast(K.convert_to_tensor(0.0), "complex64")
    k0 = K.stack(
        [
            K.stack([K.cast(K.sqrt(1.0 - p01), "complex64"), zero]),
            K.stack([zero, K.cast(K.sqrt(1.0 - p10), "complex64")]),
        ]
    )
    k1 = K.stack(
        [
            K.stack([zero, K.cast(K.sqrt(p10), "complex64")]),
            K.stack([zero, zero]),
        ]
    )
    k2 = K.stack(
        [
            K.stack([zero, zero]),
            K.stack([K.cast(K.sqrt(p01), "complex64"), zero]),
        ]
    )
    return [k0, k1, k2]


def apply_channel(circuit, kraus, qubits):
    for q in qubits:
        circuit.apply_general_kraus(kraus, q)


def apply_noisy_entangler_layer(circuit, kraus, config):
    for i in range(0, config["n_qubits"] - 1, 2):
        circuit.rxx(i, i + 1, theta=config["entangler_angle"])
        apply_channel(circuit, kraus, (i, i + 1))


def prepare_initial_state(circuit, probe_index, config):
    if probe_index == 1:
        for i in range(config["n_qubits"]):
            circuit.x(i)
    elif probe_index == 2:
        for i in range(1, config["n_qubits"], 2):
            circuit.x(i)
    elif probe_index == 3:
        for i in range(config["n_qubits"]):
            circuit.h(i)


def probe_observables(probe_index, p01, p10, config, signs):
    circuit = tc.DMCircuit(config["n_qubits"])
    kraus = asymmetric_bitflip_kraus(p01, p10)
    prepare_initial_state(circuit, probe_index, config)
    apply_noisy_entangler_layer(circuit, kraus, config)
    probabilities = circuit.probability()
    return K.real(K.tensordot(signs, probabilities, 1))


def observable_table(p01, p10, config, signs):
    return K.stack(
        [
            probe_observables(probe_index, p01, p10, config, signs)
            for probe_index in range(PROBE_COUNT)
        ]
    )


def loss_and_observables(raw_params, target_expectations, config, signs):
    p01, p10 = probabilities(raw_params)
    fitted_expectations = observable_table(p01, p10, config, signs)
    loss = K.mean((fitted_expectations - target_expectations) ** 2)
    return loss, (p01, p10, fitted_expectations)


def run_solution(config):
    signs = observable_signs(config)
    true_target = observable_table(
        K.convert_to_tensor(config["true_p01"]),
        K.convert_to_tensor(config["true_p10"]),
        config,
        signs,
    )
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return loss_and_observables(p, true_target, config, signs)

    value_and_grad = K.jit(K.value_and_grad(loss_fn, has_aux=True))

    loss_history = []
    initial_p01, initial_p10 = probabilities(params)
    for _ in range(config["max_steps"]):
        (loss, aux), grads = value_and_grad(params)
        final_p01, final_p10, fitted_expectations = aux
        loss_history.append(loss)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        "initial_p01": K.numpy(initial_p01),
        "initial_p10": K.numpy(initial_p10),
        "final_p01": K.numpy(final_p01),
        "final_p10": K.numpy(final_p10),
        "initial_loss": K.numpy(loss_history[0]),
        "final_loss": K.numpy(loss_history[-1]),
        "loss_history": K.numpy(K.stack(loss_history)),
        "target_expectations": K.numpy(true_target),
        "fitted_expectations": K.numpy(fitted_expectations),
    }
