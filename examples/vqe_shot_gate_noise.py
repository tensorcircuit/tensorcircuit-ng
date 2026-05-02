"""
VQE with measurement shot noise and quantum gate noise,
gradients are evaluated using parameter shift.
This script requires a great amount of circuit evaluation,
and match the pattern on real hardware experiments.
"""

import os
import jax
import optax
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import tensorcircuit as tc
from tensorcircuit.experimental import parameter_shift_grad_v2
from tensorcircuit.quantum import PauliStringSum2COO

# Set JAX as the backend
K = tc.set_backend("jax")

# Configuration
n = 10
n_layers = 3
p_noise = 0.003  # Error probability per gate
n_shots = 256  # Number of trajectories for expectation estimation
learning_rate = 0.1
n_steps = 100

# Total random variables needed per trajectory:
# Each layer: n (depolarizing)
# Measurement: 1 (for bitstring choice)
# We do 2 independent noisy circuit paths (one for Z basis, one for X basis)
n_random = (n_layers * n + 1) * 2


def get_exact_energy():
    """Exact ground state energy for OBC TFIM via scipy.sparse."""

    ls, weights = [], []
    for i in range(n - 1):
        # ZZ terms (Open Boundary Conditions)
        s_zz = [0] * n
        s_zz[i], s_zz[i + 1] = 3, 3
        ls.append(s_zz)
        weights.append(-1.0)
    for i in range(n):
        # X terms
        s_x = [0] * n
        s_x[i] = 1
        ls.append(s_x)
        weights.append(-1.0)
    h_sparse = PauliStringSum2COO(ls, weights, numpy=True)
    eigvals, _ = eigsh(h_sparse, k=1, which="SA")
    return float(eigvals[0])


def get_energy_trajectory(params, status):
    """
    Computes OBC TFIM energy for one MC trajectory using external status.
    Returns a scalar energy value.
    """

    def run_circuit(p, s, basis="z"):
        c_init = tc.Circuit(n)
        s_initial = c_init.state()

        def layer_scan(state, p_layer_status_layer):
            p_l, s_l = p_layer_status_layer
            c_in = tc.Circuit(n, inputs=state)
            # 1. Single Qubit Rotations (Rx + Ry)
            for i in range(n):
                c_in.rx(i, theta=p_l[0, i])
                c_in.ry(i, theta=p_l[1, i])
            # 2. Parameterized Entanglement Layer (RZZ - OBC)
            for i in range(n - 1):
                c_in.rzz(i, i + 1, theta=p_l[2, i])
            # 3. Depolarizing Noise Layer
            for i in range(n):
                c_in.depolarizing2(i, px=p_noise, py=p_noise, pz=p_noise, status=s_l[i])
            return c_in.state(), None

        s_layers = K.reshape(s[:-1], [n_layers, n])
        final_state, _ = jax.lax.scan(layer_scan, s_initial, (p, s_layers))

        c_final = tc.Circuit(n, inputs=final_state)
        if basis == "x":
            for i in range(n):
                c_final.h(i)

        res = c_final.sample(
            batch=1, allow_state=True, status=s[-1:], format="sample_int"
        )
        return res[0]

    n_single = n_layers * n + 1
    sample_z_int = run_circuit(params, status[:n_single], basis="z")
    sample_x_int = run_circuit(params, status[n_single:], basis="x")

    bits_z = tc.quantum.sample_int2bin(K.reshape(sample_z_int, [1]), n)[0]
    bits_x = tc.quantum.sample_int2bin(K.reshape(sample_x_int, [1]), n)[0]

    spins_z = 1 - 2 * bits_z
    spins_x = 1 - 2 * bits_x

    # Calculate ZZ terms (OBC)
    e_zz = K.sum(spins_z[:-1] * spins_z[1:])
    e_x = K.sum(spins_x)

    return K.real(-(e_zz + e_x))


@K.jit
def mean_energy(params, status_batch):
    """Sum of energies over trajectories."""
    energies = K.vmap(get_energy_trajectory, vectorized_argnums=1)(params, status_batch)
    return K.mean(energies)


# Gradient function for the sum of trajectories
grad_f = parameter_shift_grad_v2(mean_energy, argnums=0, random_argnums=1)


def main():
    print(f"--- Noisy OBC VQE (TFIM) | L={n} | p={p_noise} | shots={n_shots} ---")
    exact_energy = get_exact_energy()
    print(f"Exact Ground State Energy (OBC): {exact_energy:.6f}")

    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, [n_layers, 3, n]) * 0.1
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    history = []

    for i in range(n_steps):
        key, subkey = jax.random.split(key)
        status_batch = jax.random.uniform(subkey, [n_shots, n_random])

        loss = mean_energy(params, status_batch)
        grads = grad_f(params, status_batch)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        mean_loss = loss
        history.append(float(mean_loss))

        if i % 10 == 0:
            print(
                f"Step {i:3d} | Mean Energy: {mean_loss:8.4f} | Error: {abs(mean_loss - exact_energy):.4f}"
            )

    print(
        f"Final Mean Energy: {history[-1]:.4f} | Error: {abs(history[-1] - exact_energy):.4f}"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Stochastic OBC VQE")
    plt.axhline(y=exact_energy, color="r", linestyle="--", label="Exact")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title(f"OBC TFIM VQE (L={n}, p={p_noise}, shots={n_shots})")
    plt.legend()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, "vqe_tfim_stochastic.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
