"""
Challenge Suite Problem 8: 7x7 grid RZZ sampling.

The solution builds the full TensorCircuit tensor network and samples from it
without constructing the 2^49 statevector.
"""

import numpy as np

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco-32-32")


def grid_sublayers(config):
    n_side = config["grid_side"]
    layers = []

    for parity in (0, 1):
        layers.append(
            [
                (row * n_side + col, row * n_side + col + 1)
                for row in range(n_side)
                for col in range(parity, n_side - 1, 2)
            ]
        )

    for parity in (0, 1):
        layers.append(
            [
                (row * n_side + col, (row + 1) * n_side + col)
                for row in range(parity, n_side - 1, 2)
                for col in range(n_side)
            ]
        )

    return layers


def edge_angle(index, left, right, config):
    return (
        config["theta_offset"]
        + config["theta_sin_scale"]
        * np.sin(config["theta_sin_frequency"] * (index + 1))
        + config["theta_cos_scale"]
        * np.cos(config["theta_cos_frequency"] * (left + 2 * right + 1))
    )


def build_circuit(config):
    circuit = tc.Circuit(config["n_qubits"])
    for i in range(config["n_qubits"]):
        circuit.h(i)

    edge_index = 0
    for layer in grid_sublayers(config):
        for left, right in layer:
            circuit.rzz(left, right, theta=edge_angle(edge_index, left, right, config))
            edge_index += 1

    for i in range(config["n_qubits"]):
        circuit.h(i)
    return circuit


def run_solution(config):
    circuit = build_circuit(config)
    rng = np.random.default_rng(2033)
    status = K.convert_to_tensor(
        rng.random((config["n_samples"], config["n_qubits"]), dtype=np.float32)
    )
    samples = circuit.sample(
        batch=config["n_samples"],
        allow_state=False,
        format="sample_bin",
        status=status,
    )
    return {
        "samples": K.numpy(samples),
        "largest_index_dimension": np.asarray(2, dtype=np.int32),
    }
