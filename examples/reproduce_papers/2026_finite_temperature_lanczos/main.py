"""Reproduction of "QUANTUM FINITE TEMPERATURE LANCZOS METHOD"
Link: https://arxiv.org/abs/2603.25394
Description:
This script reproduces Figure 1(a) from the paper using TensorCircuit-NG.
It implements the Quantum Finite Temperature Lanczos Method (QFTLM) with high-performance
JAX optimizations (vmap, jit, lax.scan) and exploits the Toeplitz structure of Krylov matrices.
"""

from functools import partial
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

# Set backend to JAX for high performance
tc.set_backend("jax")
tc.set_dtype("complex128")


def get_hamiltonian(L):
    """Define the TFIM Hamiltonian: H = sum X_i - sum Z_i Z_{i+1}."""
    ls = []
    weights = []
    # Field terms: sum X_i
    for i in range(L):
        l = [0] * L
        l[i] = 1  # X
        ls.append(l)
        weights.append(1.0)
    # Interaction terms: -sum Z_i Z_{i+1}
    for i in range(L - 1):
        l = [0] * L
        l[i] = 3  # Z
        l[i + 1] = 3  # Z
        ls.append(l)
        weights.append(-1.0)

    # Generate matrix form for ED
    h_matrix = tc.quantum.PauliStringSum2Dense(ls, weights)
    return h_matrix


def exact_diagonalization(eigvals, betas):
    """Compute exact thermal expectation value of energy with numerical stability."""
    min_eig = jnp.min(eigvals)

    def single_beta_ed(beta):
        shifted_eigvals = eigvals - min_eig
        exp_vals = jnp.exp(-beta * shifted_eigvals)
        partition_function = jnp.sum(exp_vals)
        energy = jnp.sum(eigvals * exp_vals) / partition_function
        return energy

    return jax.vmap(single_beta_ed)(betas)


def get_typical_state(L, key):
    """Prepare a Quantum Hutchinson state using JAX random PRNG."""
    dim = 2**L
    phases = jax.random.uniform(key, (dim,), minval=0, maxval=2 * np.pi)
    state = jnp.exp(1j * phases) / jnp.sqrt(dim)
    return state


def qftlm_single_state(key, U, h_matrix, betas, D, eta, L, global_min_e):
    """Core QFTLM logic for a single random state, optimized with lax.scan."""
    psi_0 = get_typical_state(L, key)

    # Exploiting Toeplitz structure:
    # S_nm = <psi_0 | U^{m-n} | psi_0>
    # H_nm = <psi_0 | H U^{m-n} | psi_0>
    # We only need to compute overlaps for k = 0 to D-1

    def scan_fun(state, _):
        curr_psi = state
        f_k = jnp.vdot(psi_0, curr_psi)
        g_k = jnp.vdot(psi_0, h_matrix @ curr_psi)
        next_psi = U @ curr_psi
        return next_psi, (f_k, g_k)

    _, (f_ks, g_ks) = jax.lax.scan(scan_fun, psi_0, None, length=D)

    def build_toeplitz(vals):
        idx = jnp.arange(D)
        diff = idx[None, :] - idx[:, None]  # m - n
        res = jnp.where(diff >= 0, vals[jnp.abs(diff)], jnp.conj(vals[jnp.abs(diff)]))
        return res

    S = build_toeplitz(f_ks)
    H_sub = build_toeplitz(g_ks)
    S_reg = S + eta * jnp.eye(D)

    # Solve generalized eigenvalue problem natively in JAX via Cholesky
    # S_reg = L @ L.H
    L_mat = jnp.linalg.cholesky(S_reg)
    L_inv = jnp.linalg.inv(L_mat)
    H_tilde = L_inv @ H_sub @ jnp.conj(L_inv.T)
    e, y = jnp.linalg.eigh(H_tilde)
    c = jnp.conj(L_inv.T) @ y

    # Overlaps w_j = <psi_0 | phi_j> = sum_n c_nj <psi_0 | v_n> = sum_n c_nj f_n
    overlaps = c.T @ f_ks
    sq_overlaps = jnp.abs(overlaps) ** 2

    # Vectorized computation of thermal averages for all betas
    def compute_avgs(beta):
        boltzmann = jnp.exp(-beta * (e - global_min_e))
        z_k = jnp.sum(boltzmann * sq_overlaps)
        e_k = jnp.sum(e * boltzmann * sq_overlaps)
        return z_k, e_k

    return jax.vmap(compute_avgs)(betas)


@partial(jax.jit, static_argnums=(4, 5, 7))
def qftlm_main_logic(key, U, h_matrix, betas, D, K, eta, L, global_min_e):
    """Vmap over K random states for the Hutchinson trace estimator."""
    keys = jax.random.split(key, K)
    zs, es = jax.vmap(
        lambda k: qftlm_single_state(k, U, h_matrix, betas, D, eta, L, global_min_e)
    )(keys)

    total_z = jnp.sum(zs, axis=0)
    total_e = jnp.sum(es, axis=0)
    return jnp.real(total_e / total_z)


def qftlm_reproduction():
    L = 10  # Scaled to L=10
    K = 40  # Number of Hutchinson states
    D = 20  # Krylov Dimension
    eta = 1e-6  # Tikhonov shift

    h_matrix = get_hamiltonian(L)

    # Calculate Lanczos step size Delta t
    eigvals_h = jnp.linalg.eigvalsh(h_matrix)
    norm_h = jnp.max(jnp.abs(eigvals_h))
    global_min_e = jnp.min(eigvals_h)
    dt = 0.5 * np.pi / norm_h

    # Time evolution operator U = exp(-i dt H)
    U = jax.scipy.linalg.expm(-1j * dt * h_matrix)

    # Temperatures to sample
    Ts = jnp.logspace(-2, 2, 20)
    betas = 1.0 / Ts

    print(f"Starting Optimized QFTLM with L={L}, K={K}, D={D}...")

    key = jax.random.PRNGKey(42)
    energies_qftlm = qftlm_main_logic(
        key, U, h_matrix, betas, D, K, eta, L, global_min_e
    )

    # Exact values for comparison (vectorized)
    energies_exact = exact_diagonalization(eigvals_h, betas)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.semilogx(Ts, energies_exact, "k-", label="Exact (ED)")
    plt.semilogx(Ts, energies_qftlm, "ro", mfc="none", label=f"QFTLM (D={D}, K={K})")
    plt.xlabel(r"Temperature $T$")
    plt.ylabel(r"Energy $\langle E \rangle$")
    plt.title(f"Optimized QFTLM Reproduction (L={L} TFIM)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # Ensure outputs directory exists
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "result.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    qftlm_reproduction()
