"""
Challenge Suite Problem 10: VQE with an 18-qubit CMZ hyperedge.

The ansatz uses TensorCircuit-NG's built-in cmz gate on non-adjacent selected
qubits. The Hamiltonian is an ordinary transverse-field Ising model evaluated
with TensorCircuit's PauliStringSum2MVP.
"""

import numpy as np
import optax

import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2MVP

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


def build_tfim_mvp(config):
    structures = []
    weights = []
    n_qubits = config["n_qubits"]

    for q in range(n_qubits - 1):
        term = [0] * n_qubits
        term[q] = 3
        term[q + 1] = 3
        structures.append(term)
        weights.append(-config["zz_strength"])

    for q in range(n_qubits):
        term = [0] * n_qubits
        term[q] = 1
        structures.append(term)
        weights.append(-config["x_strength"])

    return PauliStringSum2MVP(structures, weights)


def apply_rotation_block(circuit, block, n_qubits):
    for q in range(n_qubits):
        circuit.rx(q, theta=block[q, 0])
        circuit.rz(q, theta=block[q, 1])
        circuit.ry(q, theta=block[q, 2])


def build_state(params, input_state, config):
    def layer_step(state, layer_params):
        circuit = tc.Circuit(config["n_qubits"], inputs=state)
        apply_rotation_block(circuit, layer_params, config["n_qubits"])
        circuit.cmz(*config["selected_qubits"])
        return circuit.state()

    return K.scan(layer_step, params, input_state)


def state_energy(state, hamiltonian_mvp):
    h_state = hamiltonian_mvp(state)
    return K.real(K.tensordot(K.conj(state), h_state, 1))


def energy_density(params, input_state, hamiltonian_mvp, config):
    state = build_state(params, input_state, config)
    return state_energy(state, hamiltonian_mvp) / config["n_qubits"]


def run_solution(config):
    params = initial_parameters(config)
    circuit = tc.Circuit(config["n_qubits"])
    for q in config["initial_ones"]:
        circuit.x(q)
    input_state = circuit.state()
    hamiltonian_mvp = build_tfim_mvp(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return energy_density(p, input_state, hamiltonian_mvp, config)

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
