"""
Challenge Suite Problem 10: VQE with an 18-qubit CMZ hyperedge.

The ansatz uses TensorCircuit-NG's built-in cmz gate on non-adjacent selected
qubits. The Hamiltonian is evaluated as an MPO expectation so the optimization
step can be JIT-compiled into a highly efficient fixed graph.
"""

import numpy as np
import optax
import quimb.tensor as qtn

import tensorcircuit as tc
from tensorcircuit.quantum import quimb2qop
from tensorcircuit.templates.measurements import mpo_expectation

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    shape = (config["n_layers"], config["n_qubits"], 3)
    params = rng.normal(
        scale=config["initial_parameter_scale"],
        size=shape,
    ).astype(np.float32)
    return K.convert_to_tensor(params)


def build_tfim_mpo(config):
    n_qubits = int(config["n_qubits"])
    zz = float(config["zz_strength"])
    xs = float(config["x_strength"])
    mpo_q = qtn.MPO_ham_ising(n_qubits, j=-4.0 * zz, bx=2.0 * xs, cyclic=False)
    return quimb2qop(mpo_q)


def apply_rotation_block(circuit, block, n_qubits):
    for q in range(n_qubits):
        circuit.rx(q, theta=block[q, 0])
        circuit.rz(q, theta=block[q, 1])
        circuit.ry(q, theta=block[q, 2])


def energy_density(params, mpo, config):
    circuit = tc.Circuit(config["n_qubits"])
    for q in config["initial_ones"]:
        circuit.x(q)
    for layer in range(config["n_layers"]):
        apply_rotation_block(circuit, params[layer], config["n_qubits"])
        circuit.cmz(*config["selected_qubits"])
    return mpo_expectation(circuit, mpo) / config["n_qubits"]


def run_solution(config):
    params = initial_parameters(config)
    mpo = build_tfim_mpo(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return energy_density(p, mpo, config)

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
        "final_parameters": K.numpy(params),
    }
