"""
Challenge Suite Problem 9: 200-qubit local light-cone optimization.

The solution optimizes the central local observable <Z100> and reuses one
JIT-compiled value-and-gradient function across optimizer steps and restarts.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def run_solution(config):
    n_qubits = config["n_qubits"]
    parameter_count = 0
    for layer in range(config["n_layers"]):
        parameter_count += 2 * n_qubits
        parameter_count += len(range(layer % 2, n_qubits - 1, 2))

    def loss_fn(params):
        circuit = tc.Circuit(n_qubits)
        for i in range(n_qubits):
            circuit.h(i)

        offset = 0
        for layer in range(config["n_layers"]):
            for i in range(n_qubits):
                circuit.ry(i, theta=params[offset])
                circuit.rz(i, theta=params[offset + 1])
                offset += 2

            for i in range(layer % 2, n_qubits - 1, 2):
                circuit.rzz(i, i + 1, theta=params[offset])
                offset += 1

        return -K.real(
            circuit.expectation_ps(
                z=[config["observable_index"]],
                enable_lightcone=True,
            )
        )

    value_and_grad = K.jit(K.value_and_grad(loss_fn))

    histories = []
    final_grad_norms = []

    for restart_index in range(config["n_restarts"]):
        rng = np.random.default_rng(config["seed"] + restart_index)
        params = rng.normal(
            scale=config["initial_parameter_scale"],
            size=(parameter_count,),
        ).astype(np.float32)
        params = K.convert_to_tensor(params)
        optimizer = optax.adam(config["learning_rate"])
        opt_state = optimizer.init(params)
        history = []

        for _ in range(config["max_steps"]):
            value, grads = value_and_grad(params)
            history.append(value)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        histories.append(
            K.numpy(K.stack(history)).astype(np.float64),
        )
        final_grad_norms.append(K.numpy(K.norm(grads)))

    histories = -np.asarray(histories, dtype=np.float64)
    final_grad_norms = np.asarray(final_grad_norms, dtype=np.float64)

    return {
        "observable_history": histories,
        "final_grad_norms": final_grad_norms,
    }
