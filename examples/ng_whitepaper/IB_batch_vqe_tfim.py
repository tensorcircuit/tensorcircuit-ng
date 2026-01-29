"""
Batch VQE on Transverse Field Ising Model (TFIM) with TensorCircuit

Demonstrates TensorCircuit's "Tensor-Native" paradigms:
- Tensor Network: Quantum gates as local tensor contractions
- AD: Reverse-mode autodiff for efficient gradient computation
- JIT: XLA compilation for hardware acceleration
- vmap: Batch parallel optimization with multiple random initializations

The script optimizes multiple random parameter sets in parallel and generates
a visualization of all optimization trajectories.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import optax
import matplotlib.pyplot as plt

import tensorcircuit as tc

# Use JAX backend for best vmap + JIT support
K = tc.set_backend("jax")
# Use cotengra contractor for tensor network contraction with default settings
tc.set_contractor("cotengra")
# Use double precision for the simulation
tc.set_dtype("complex128")

# ============================================================================
# Configuration
# ============================================================================

n = 10  # Number of qubits
nlayers = 6  # Ansatz depth
g = 1.0  # Transverse field strength
batch_size = 16  # Number of parallel random initializations
n_steps = 500  # Optimization steps
learning_rate = 0.02

# ============================================================================
# TFIM Energy Function (Matching Whitepaper Code)
# ============================================================================


def tfim_energy(param, g=1.0, n=10):
    """
    Compute TFIM energy expectation value.

    The ansatz is a Hardware-Efficient Ansatz with:
      - Layer 1: Single-qubit RX rotations
      - Layer 2: Nearest-neighbor RZZ entangling gates

    The TFIM Hamiltonian is:
      H = -∑_{i} Z_i Z_{i+1} - g ∑_{i} X_i  (OBC)

    Args:
        param: Shape [nlayers, n] - variational parameters
        g: Transverse field strength
        n: Number of qubits

    Returns:
        Real-valued energy expectation
    """
    c = tc.Circuit(n)
    c.h(range(n))
    # --- Ansatz Construction ---
    for layer in range(param.shape[0] // 2):
        # Layer 1: Single qubit rotations
        for i in range(n):
            c.rx(i, theta=param[2 * layer, i])
        # Layer 2: Entangling ZZ gates
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=param[2 * layer + 1, i])

    # --- Expectation Calculation ---
    e = 0.0
    # Interaction term: -<Z_i Z_{i+1}>
    for i in range(n - 1):
        e -= c.expectation_ps(z=[i, i + 1])
    # Transverse field term: -g * <X_i>
    for i in range(n):
        e -= g * c.expectation_ps(x=[i])

    return K.real(e)


# ============================================================================
# Exact Ground State Energy (via Sparse Diagonalization)
# ============================================================================


def compute_exact_ground_energy(n, g):
    """
    Compute exact TFIM ground state energy via sparse diagonalization.

    TFIM Hamiltonian: H = -∑ Z_i Z_{i+1} - g ∑ X_i
    """
    # Build Pauli string representation
    ps = []
    weights = []

    # ZZ interaction terms: -Z_i Z_{i+1}
    for i in range(n - 1):
        l = [0] * n
        l[i] = 3  # Z
        l[i + 1] = 3  # Z
        ps.append(l)
        weights.append(-1.0)

    # Transverse field terms: -g X_i
    for i in range(n):
        l = [0] * n
        l[i] = 1  # X
        ps.append(l)
        weights.append(-g)

    # Convert to sparse matrix and diagonalize
    hm = tc.quantum.PauliStringSum2COO(ps, weights, numpy=True)

    return eigsh(hm, k=1, which="SA", return_eigenvectors=False)[0]


# ============================================================================
# Batch VQE with vmap
# ============================================================================

# Transform 1: Add gradient capability (Value, Gradient)
vqe_grad = K.value_and_grad(tfim_energy)

# Transform 2: Apply vmap for batch parallelism
# Vectorize over the first argument (param)
vqe_batch = K.vmap(vqe_grad, vectorized_argnums=0)

# Transform 3: Compile for speed (JIT)
vqe_step = K.jit(vqe_batch, static_argnums=(2,))

# ============================================================================
# Training Loop
# ============================================================================


def run_batch_vqe():
    """Run batch VQE optimization and return trajectories."""
    print("=" * 60)
    print("Batch VQE on TFIM with TensorCircuit")
    print("=" * 60)
    print(f"Qubits: {n}, Layers: {nlayers}, Batch Size: {batch_size}")
    print(f"Transverse field g: {g}")
    print()

    # Compute exact ground state energy
    exact_energy = compute_exact_ground_energy(n, g)
    print(f"Exact Ground State Energy: {exact_energy:.6f}")
    print()

    # Initialize optimizer
    optimizer = K.optimizer(optax.adam(learning_rate))

    # Random initialization: [batch_size, 2*nlayers, n]
    K.set_random_state(42)
    params = K.implicit_randn(shape=[batch_size, 2 * nlayers, n], stddev=0.1)

    # Store energy trajectories
    trajectories = []

    print("Starting optimization...")
    print("-" * 40)

    for step in range(n_steps):
        # Batched forward + backward pass
        energies, grads = vqe_step(params)

        # Update parameters
        params = optimizer.update(grads, params)

        # Store energies
        energies_np = np.array(energies)
        trajectories.append(energies_np)

        # Print progress
        if step % 20 == 0 or step == n_steps - 1:
            mean_e = np.mean(energies_np)
            min_e = np.min(energies_np)
            print(
                f"Step {step:3d}: Mean Energy = {mean_e:.6f}, "
                f"Min Energy = {min_e:.6f}"
            )

    print("-" * 40)
    print()

    # Final results
    final_energies = trajectories[-1]
    print("Final Energies (all batches):")
    for i, e in enumerate(final_energies):
        delta = e - exact_energy
        print(f"  Batch {i}: {e:.6f}  (Δ = {delta:+.6f})")

    print()
    print(f"Best Final Energy: {np.min(final_energies):.6f}")
    print(f"Exact Ground State: {exact_energy:.6f}")
    print(f"Best Error: {np.min(final_energies) - exact_energy:.6f}")

    return np.array(trajectories), exact_energy


# ============================================================================
# Visualization
# ============================================================================


def plot_trajectories(trajectories, exact_energy, save_path="batch_vqe_trajectory.pdf"):
    """
    Plot optimization trajectories and save as PDF.

    Args:
        trajectories: Shape [n_steps, batch_size] - energy history
        exact_energy: Exact ground state energy for reference
        save_path: Output PDF path
    """
    _, ax = plt.subplots(figsize=(10, 6))

    n_steps, batch_size = trajectories.shape
    steps = np.arange(n_steps)

    # Color palette
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, batch_size))

    # Plot individual trajectories
    for i in range(batch_size):
        ax.plot(
            steps,
            trajectories[:, i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Batch {i}" if i < 4 else None,
        )

    # Plot exact ground state
    ax.axhline(
        exact_energy,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Exact GS = {exact_energy:.4f}",
    )

    # Plot mean trajectory
    mean_traj = np.mean(trajectories, axis=1)
    ax.plot(
        steps,
        mean_traj,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Mean",
    )

    # Styling
    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Energy", fontsize=12)
    ax.set_title(
        f"Batch VQE Optimization Trajectories (TFIM, n={n}, g={g})",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=150, bbox_inches="tight")
    print(f"\nOptimization trajectories saved to: {save_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run batch VQE
    trajectories, exact_energy = run_batch_vqe()

    # Generate visualization
    plot_trajectories(trajectories, exact_energy)
