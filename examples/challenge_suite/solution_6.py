"""
Challenge Suite Problem 6: digital-analog hybrid VQE with trainable analog blocks.

Each block evolves under a trainable sparse Hamiltonian via
tc.timeevol.ode_evol_global raw mode, then applies trainable local digital
rotations. The block loop uses jax.lax.scan for JIT-friendly staging. The
solution returns only NumPy values consumed by evaluate_6.py.
"""

import jax
import numpy as np
import optax

import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2MVP

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def build_hamiltonians(config):
    n = config["n_qubits"]
    xy_ls, xy_w = [], []
    for i in range(n - 1):
        for p in (1, 2):
            s = [0] * n
            s[i] = p
            s[i + 1] = p
            xy_ls.append(s)
            xy_w.append(1.0)
    Hxy_mvp = PauliStringSum2MVP(xy_ls, xy_w)

    field_ls, field_w = [], []
    for i in range(n):
        s = [0] * n
        s[i] = 3
        field_ls.append(s)
        field_w.append((-1.0) ** i)
    Hfield_mvp = PauliStringSum2MVP(field_ls, field_w)

    target_ls, target_w = [], []
    for i in range(n - 1):
        for p, c in ((1, 0.7), (2, 0.7), (3, 1.1)):
            s = [0] * n
            s[i] = p
            s[i + 1] = p
            target_ls.append(s)
            target_w.append(c)
    for i in range(n):
        s = [0] * n
        s[i] = 3
        target_ls.append(s)
        target_w.append(0.25 * ((-1.0) ** i))
    Htarget_mvp = PauliStringSum2MVP(target_ls, target_w)

    return Hxy_mvp, Hfield_mvp, Htarget_mvp


def initial_state(config):
    circuit = tc.Circuit(config["n_qubits"])
    for i in range(1, config["n_qubits"], 2):
        circuit.x(i)
    return circuit.state()


def initial_parameters(config):
    rng = np.random.default_rng(2026)
    n = config["n_qubits"]
    nb = config["n_blocks"]
    return {
        "s": K.convert_to_tensor(np.zeros(nb, dtype=np.float32)),
        "j": K.convert_to_tensor(0.1 * np.ones(nb, dtype=np.float32)),
        "d": K.convert_to_tensor(0.1 * np.ones(nb, dtype=np.float32)),
        "rot": K.convert_to_tensor(
            rng.normal(scale=0.1, size=(nb, n, 3)).astype(np.float32)
        ),
    }


def forward(params, psi0, Hxy_mvp, Hfield_mvp, Htarget_mvp, config):
    t_min = config["t_min"]
    t_max = config["t_max"]
    rtol = config["ode_rtol"]
    atol = config["ode_atol"]

    def block_step(psi, block_params):
        s_l, j_l, d_l, rot_l = block_params
        t = t_min + (t_max - t_min) * K.sigmoid(s_l)
        Jc = K.cast(K.tanh(j_l), tc.dtypestr)
        Dc = K.cast(K.tanh(d_l), tc.dtypestr)

        def vf(y, tt):
            return -1.0j * (Jc * Hxy_mvp(y) + Dc * Hfield_mvp(y))

        times = K.stack([t * 0.0, t])
        psi = tc.timeevol.ode_evol_global(
            vf,
            psi,
            times,
            mode="raw",
            ode_backend="diffrax",
            rtol=rtol,
            atol=atol,
            max_steps=config["ode_max_steps"],
        )[-1]

        circuit = tc.Circuit(config["n_qubits"], inputs=psi)
        for i in range(config["n_qubits"]):
            circuit.rz(i, theta=rot_l[i, 0])
            circuit.ry(i, theta=rot_l[i, 1])
            circuit.rz(i, theta=rot_l[i, 2])
        psi = circuit.state()
        return psi, None

    block_xs = (params["s"], params["j"], params["d"], params["rot"])
    final_psi, _ = jax.lax.scan(block_step, psi0, block_xs)

    h_psi = Htarget_mvp(final_psi)
    energy_density = (
        K.real(K.tensordot(K.conj(final_psi), h_psi, 1)) / config["n_qubits"]
    )
    return energy_density


def run_solution(config):
    Hxy_mvp, Hfield_mvp, Htarget_mvp = build_hamiltonians(config)
    psi0 = initial_state(config)
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return forward(p, psi0, Hxy_mvp, Hfield_mvp, Htarget_mvp, config)

    value_and_grad = K.jit(K.value_and_grad(loss_fn))

    energy_density_history = []
    for _ in range(config["max_steps"]):
        energy_density, grads = value_and_grad(params)
        energy_density_history.append(energy_density)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    t_min = config["t_min"]
    t_max = config["t_max"]
    final_times = K.numpy(t_min + (t_max - t_min) * K.sigmoid(params["s"])).astype(
        np.float32
    )
    final_couplings = K.numpy(K.tanh(params["j"])).astype(np.float32)
    final_detunings = K.numpy(K.tanh(params["d"])).astype(np.float32)

    return {
        "final_analog_times": final_times,
        "final_analog_couplings": final_couplings,
        "final_analog_detunings": final_detunings,
        "energy_density_history": K.numpy(K.stack(energy_density_history)),
    }
