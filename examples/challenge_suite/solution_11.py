"""
Challenge Suite Problem 11: spin-1 Haldane-chain VQE.

The TensorCircuit-NG baseline uses QuditCircuit and direct qudit unitary APIs for
all variational gates. Repeated layers are staged through scan to reduce JIT
tracing overhead.
"""

import numpy as np
import optax
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")

DIM = 3
SQRT2 = np.sqrt(2.0).astype(np.float32)

SX = K.convert_to_tensor(
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex64)
    / SQRT2
)
SY = K.convert_to_tensor(
    np.array(
        [[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]],
        dtype=np.complex64,
    )
    / SQRT2
)
SZ = K.convert_to_tensor(np.diag([1.0, 0.0, -1.0]).astype(np.complex64))
SZ2 = K.convert_to_tensor(np.diag([1.0, 0.0, 1.0]).astype(np.complex64))
STRING_MIDDLE = K.convert_to_tensor(np.diag([-1.0, 1.0, -1.0]).astype(np.complex64))

DOT_BOND = K.kron(SX, SX) + K.kron(SY, SY) + K.kron(SZ, SZ)
DOT_BOND_SQUARED = DOT_BOND @ DOT_BOND
ZZ_BOND = K.kron(SZ, SZ)


def n_even_bonds(config):
    return config["n_sites"] // 2


def n_odd_bonds(config):
    return (config["n_sites"] - 1) // 2


def string_pairs(config):
    n_sites = config["n_sites"]
    return tuple((i, n_sites - 1 - i) for i in range(3))


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    scale = config["initial_parameter_scale"]
    n_layers = config["n_layers"]
    n_sites = config["n_sites"]
    return {
        "single_rz1": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_sites)).astype(np.float32)
        ),
        "single_ry": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_sites)).astype(np.float32)
        ),
        "single_rz2": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_sites)).astype(np.float32)
        ),
        "even_theta": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_even_bonds(config))).astype(
                np.float32
            )
        ),
        "even_phi": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_even_bonds(config))).astype(
                np.float32
            )
        ),
        "odd_theta": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_odd_bonds(config))).astype(
                np.float32
            )
        ),
        "odd_phi": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_odd_bonds(config))).astype(
                np.float32
            )
        ),
    }


def initial_state(config):
    neel = np.zeros(DIM ** config["n_sites"], dtype=np.complex64)
    digits = [0 if i % 2 == 0 else 2 for i in range(config["n_sites"])]
    index = 0
    for digit in digits:
        index = index * DIM + digit
    neel[index] = 1.0
    return K.convert_to_tensor(neel)


def string_orders_from_state(state, config):
    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
    values = []
    for i, j in string_pairs(config):
        operators = [(tc.gates.Gate(SZ), [i])]
        for site in range(i + 1, j):
            operators.append((tc.gates.Gate(STRING_MIDDLE), [site]))
        operators.append((tc.gates.Gate(SZ), [j]))
        values.append(K.real(circuit.expectation(*operators)))
    return K.numpy(K.stack(values))


def spin1_rz(theta):
    zero = K.cast(0.0, "complex64")
    return K.cast(
        K.stack(
            [
                K.stack([K.exp(-1.0j * theta), zero, zero]),
                K.stack([zero, K.cast(1.0, "complex64"), zero]),
                K.stack([zero, zero, K.exp(1.0j * theta)]),
            ]
        ),
        "complex64",
    )


def spin1_ry(theta):
    c = K.cos(theta)
    s = K.sin(theta)
    return K.cast(
        K.stack(
            [
                K.stack([(1.0 + c) / 2.0, -s / SQRT2, (1.0 - c) / 2.0]),
                K.stack([s / SQRT2, c, -s / SQRT2]),
                K.stack([(1.0 - c) / 2.0, s / SQRT2, (1.0 + c) / 2.0]),
            ]
        ),
        "complex64",
    )


def spin1_entangler(theta, phi, beta):
    generator = theta * DOT_BOND + (phi - theta) * ZZ_BOND + beta * DOT_BOND_SQUARED
    return K.expm(-1.0j * generator)


def apply_layer(state, layer_params, config):
    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)

    for site in range(config["n_sites"]):
        circuit.unitary(
            site, unitary=tc.gates.Gate(spin1_rz(layer_params["single_rz1"][site]))
        )
        circuit.unitary(
            site, unitary=tc.gates.Gate(spin1_ry(layer_params["single_ry"][site]))
        )
        circuit.unitary(
            site, unitary=tc.gates.Gate(spin1_rz(layer_params["single_rz2"][site]))
        )

    even_index = 0
    for left in range(0, config["n_sites"] - 1, 2):
        gate = spin1_entangler(
            layer_params["even_theta"][even_index],
            layer_params["even_phi"][even_index],
            config["beta"],
        )
        circuit.unitary(
            left,
            left + 1,
            unitary=tc.gates.Gate(K.reshape(gate, (DIM, DIM, DIM, DIM))),
            name="spin1_even",
        )
        even_index += 1

    odd_index = 0
    for left in range(1, config["n_sites"] - 1, 2):
        gate = spin1_entangler(
            layer_params["odd_theta"][odd_index],
            layer_params["odd_phi"][odd_index],
            config["beta"],
        )
        circuit.unitary(
            left,
            left + 1,
            unitary=tc.gates.Gate(K.reshape(gate, (DIM, DIM, DIM, DIM))),
            name="spin1_odd",
        )
        odd_index += 1

    return circuit.state()


def build_state(params, config):
    return K.scan(lambda s, p: apply_layer(s, p, config), params, initial_state(config))


def bond_hamiltonian(config):
    return DOT_BOND + config["beta"] * DOT_BOND_SQUARED


def energy_density_from_state(state, config):
    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
    bond_op = tc.gates.Gate(K.reshape(bond_hamiltonian(config), (DIM, DIM, DIM, DIM)))
    onsite_op = tc.gates.Gate(SZ2)

    energy = K.cast(0.0, "complex64")
    for left in range(config["n_sites"] - 1):
        energy += circuit.expectation((bond_op, [left, left + 1]))
    for site in range(config["n_sites"]):
        energy += config["single_ion_anisotropy"] * circuit.expectation(
            (onsite_op, [site])
        )
    return K.real(energy) / config["n_sites"]


def energy_density(params, config):
    return energy_density_from_state(build_state(params, config), config)


def run_solution(config):
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return energy_density(p, config)

    def train_step(p, state):
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, value

    train_step = K.jit(train_step)

    history = []
    for _ in range(config["max_steps"]):
        params, opt_state, value = train_step(params, opt_state)
        history.append(value)

    final_state = build_state(params, config)
    final_energy_density = energy_density_from_state(final_state, config)
    final_string_orders = string_orders_from_state(final_state, config)

    return {
        "energy_density_history": K.numpy(K.stack(history)),
        "final_energy_density": K.numpy(final_energy_density),
        "final_string_orders": final_string_orders,
    }
