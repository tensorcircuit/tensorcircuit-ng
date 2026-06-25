"""
Challenge Suite Problem 9: random local light-cone optimization.

The evaluator passes an explicit framework-neutral gate tape and Pauli-term
list. This solution constructs the full 512-qubit circuit from that tape and
uses TensorCircuit's automatic light-cone contraction for the two local terms.
"""

import numpy as np

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def pauli_indices(term):
    xs, ys, zs = [], [], []
    for pauli, qubit in term:
        if pauli == "x":
            xs.append(qubit)
        elif pauli == "y":
            ys.append(qubit)
        elif pauli == "z":
            zs.append(qubit)
        else:
            raise ValueError(f"Unknown Pauli axis: {pauli}")
    return xs, ys, zs


def initial_parameters(config):
    params = []
    for restart_index in range(config["n_restarts"]):
        rng = np.random.default_rng(config["seed"] + 100000 + restart_index)
        params.append(
            rng.normal(
                scale=config["initial_parameter_scale"],
                size=(config["parameter_count"],),
            ).astype(np.float32)
        )
    return K.convert_to_tensor(np.asarray(params, dtype=np.float32))


def run_solution(config):
    gate_tape = tuple(config["gate_tape"])
    pauli_terms = tuple(config["pauli_terms"])
    pauli_data = tuple((coeff, pauli_indices(term)) for coeff, term in pauli_terms)

    def loss_fn(params):
        circuit = tc.Circuit(config["n_qubits"])
        for qubit in range(config["n_qubits"]):
            circuit.h(qubit)

        for gate in gate_tape:
            if len(gate) == 3:
                getattr(circuit, gate[0])(gate[1], theta=params[gate[2]])
            else:
                getattr(circuit, gate[0])(gate[1], gate[2], theta=params[gate[3]])

        total = 0.0
        for coeff, (xs, ys, zs) in pauli_data:
            total += coeff * K.real(
                circuit.expectation_ps(
                    x=xs,
                    y=ys,
                    z=zs,
                    enable_lightcone=True,
                )
            )
        return -total

    value_and_grad = K.vmap(K.value_and_grad(loss_fn), vectorized_argnums=0)

    def train_step(params, moment1, moment2, step_count):
        values, grads = value_and_grad(params)
        step_count = step_count + 1.0
        beta1 = 0.9
        beta2 = 0.999
        moment1 = beta1 * moment1 + (1.0 - beta1) * grads
        moment2 = beta2 * moment2 + (1.0 - beta2) * grads * grads
        moment1_hat = moment1 / (1.0 - beta1**step_count)
        moment2_hat = moment2 / (1.0 - beta2**step_count)
        params = params - config["learning_rate"] * moment1_hat / (
            K.sqrt(moment2_hat) + 1.0e-8
        )
        return params, moment1, moment2, step_count, -values

    train_step = K.jit(train_step)

    params = initial_parameters(config)
    moment1 = K.zeros_like(params)
    moment2 = K.zeros_like(params)
    step_count = K.convert_to_tensor(np.array(0.0, dtype=np.float32))
    history = []

    for _ in range(config["max_steps"]):
        params, moment1, moment2, step_count, observable = train_step(
            params, moment1, moment2, step_count
        )
        history.append(observable)

    return {
        "observable_history": K.numpy(K.stack(history, axis=1)).astype(np.float64),
    }
