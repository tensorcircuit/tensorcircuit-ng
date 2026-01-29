r"""
This script demonstrates the simulation of non-equilibrium and equilibrium dynamics
using Fermionic Gaussian States (FGS) in TensorCircuit.

It covers:
1.  **Quench Dynamics**: Real-time evolution of a Neel state under a hopping Hamiltonian.
2.  **Entanglement & Correlations**: Linear entropy growth and light-cone spreading
    (using OTOC-style time-separated correlators).
3.  **Imaginary Time Evolution**: Cooling a state to its ground state using imaginary time steps.

---
**FGS Conventions in TensorCircuit**:
- **Basis**: The $2L$ basis follows the Nambu-like ordering:
    $\Psi = (c_1, \dots, c_L, c_1^\dagger, \dots, c_L^\dagger)^T$.
- **Hamiltonian Matrix**: A quadratic Hamiltonian $\hat{H}$ is represented by a $2L \times 2L$ matrix $h$.
- **Correlation Matrix**: `sim.get_cmatrix()` returns $C_{pq} = \langle \Psi_p \Psi_q^\dagger \rangle$.
  - Top-left $L \times L$ block: $\langle c_i c_j^\dagger \rangle = \delta_{ij} - \langle c_j^\dagger c_i \rangle$.
  - Bottom-right $L \times L$ block: $\langle c_i^\dagger c_j \rangle$.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import tensorcircuit as tc

# Set backend to JAX for JIT compilation and speed
K = tc.set_backend("jax")
tc.set_dtype("complex128")


def get_hopping_hamiltonian(L, t):
    r"""
    Constructs the Hamiltonian matrix for a simple hopping model.
    H = -t sum (c_i^\dag c_{i+1} + h.c.)
    Returns the 2L x 2L matrix h.
    """
    h = jnp.zeros((2 * L, 2 * L), dtype=jnp.complex128)
    for i in range(L - 1):
        h += tc.FGSSimulator.hopping(-t, i, i + 1, L)
    return h


def run_dynamics():
    L = 60
    t_hopping = 1.0
    dt = 0.1
    steps = 100

    print(f"\n=== Part 1: Real-Time Quench Dynamics (L={L}) ===")
    print("Initial State: Neel State |101010...>")
    print(f"Evolving under hopping Hamiltonian (t_hopping={t_hopping})")

    # 1. Initialize State: Neel State (Occupied at even sites)
    filled_sites = [2 * i for i in range(L // 2)]
    sim = tc.FGSSimulator(L, filled=filled_sites)

    # 2. Preparation
    h_mat = get_hopping_hamiltonian(L, t_hopping)
    step_op = h_mat * dt * 2

    times = []
    entropies = []
    densities = []
    corrs_sep = []  # To store <c_i^\dagger(t) c_{L/2}(0)>

    subsystem_b = list(range(L // 2, L))
    center = L // 2

    print(f"Simulating {steps} steps...")
    t0 = time.time()

    for step in range(steps + 1):
        times.append(step * dt)

        # 1. Entanglement Entropy (Half-chain)
        ee = K.real(sim.entropy(subsystems_to_trace_out=subsystem_b))
        entropies.append(ee)

        # 2. Density Profile
        cm = sim.get_cmatrix()
        # Bottom-right diagonal is <c_i^\dagger c_i>
        ni = K.real(jnp.diag(cm)[L:])
        densities.append(ni)

        # 3. Time-separated Correlation (OTOC style)
        # <\Psi_p(t) \Psi_q^\dagger(0)>
        cm_otoc = sim.get_cmatrix(now_i=True, now_j=False)
        # We look at <c_i^\dagger(t) c_{center}(0)>
        row = cm_otoc[L:, center + L]
        corrs_sep.append(jnp.abs(row) ** 2)

        if step < steps:
            sim.evol_hamiltonian(step_op)

    print(f"Dynamics finished in {time.time() - t0:.4f}s")

    # --- Visualization ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(times, entropies, lw=2, color="navy")
    plt.title("Entanglement Entropy Growth")
    plt.xlabel("Time $t$")
    plt.ylabel("$S_{L/2}(t)$")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.imshow(
        np.array(densities),
        aspect="auto",
        origin="lower",
        extent=[0, L, 0, steps * dt],
        cmap="RdBu_r",
    )
    plt.colorbar(label="$n_i(t)$")
    plt.title("Neel State Melting")
    plt.xlabel("Site $i$")
    plt.ylabel("Time $t$")

    plt.subplot(1, 3, 3)
    plt.imshow(
        np.array(corrs_sep),
        aspect="auto",
        origin="lower",
        extent=[0, L, 0, steps * dt],
        cmap="magma",
    )
    plt.colorbar(label=r"$|\langle c_i^\dagger(t) c_{L/2}(0) \rangle|^2$")
    plt.title("Light-Cone (Time-separated Corr)")
    plt.xlabel("Site $i$")
    plt.ylabel("Time $t$")

    plt.tight_layout()
    plt.savefig("fgs_real_time.pdf")
    print("Saved 'fgs_real_time.pdf'")


def run_imaginary_time():
    L = 60
    t_hopping = 1.0
    dtau = 0.2
    steps = 50

    print(f"\n=== Part 2: Imaginary-Time Evolution (Cooling L={L}) ===")
    print("Starting from Neel State, cooling under hopping Hamiltonian...")

    filled_sites = [2 * i for i in range(L // 2)]
    sim = tc.FGSSimulator(L, filled=filled_sites)
    h_mat = get_hopping_hamiltonian(L, t_hopping)

    step_op = h_mat * dtau * 2

    energies = []
    entropies = []

    subsystem_b = list(range(L // 2, L))

    for step in range(steps + 1):
        cm = sim.get_cmatrix()
        c_occcupied = cm[L:, L:]
        energy = 0
        for i in range(L - 1):
            val = -t_hopping * (c_occcupied[i, i + 1] + c_occcupied[i + 1, i])
            energy += val

        energies.append(K.real(energy))
        entropies.append(K.real(sim.entropy(subsystems_to_trace_out=subsystem_b)))

        if step < steps:
            sim.evol_ihamiltonian(step_op)

    # Compare with Exact Ground State
    h_l = np.zeros((L, L))
    for i in range(L - 1):
        h_l[i, i + 1] = h_l[i + 1, i] = -t_hopping
    eigvals = np.linalg.eigvalsh(h_l)
    energy_exact = np.sum(eigvals[eigvals < 0])

    print(f"Final Energy: {energies[-1]:.6f} (Exact GS: {energy_exact:.6f})")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(energies, "go-", label="FGS Imaginary Time")
    plt.axhline(energy_exact, color="r", ls="--", label="Exact Ground State")
    plt.title("Energy Convergence")
    plt.xlabel("Step")
    plt.ylabel("Energy $E$")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(entropies, "mo-")
    plt.title("Entanglement Entropy Convergence")
    plt.xlabel("Step")
    plt.ylabel("$S_{L/2}$")

    plt.tight_layout()
    plt.savefig("fgs_imaginary_time.pdf")
    print("Saved 'fgs_imaginary_time.pdf'")


if __name__ == "__main__":
    run_dynamics()
    run_imaginary_time()
