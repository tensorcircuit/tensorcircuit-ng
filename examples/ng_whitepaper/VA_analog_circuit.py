"""
Digital-Analog Variational Quantum Eigensolver (DA-VQE) Demo

This script demonstrates the `AnalogCircuit` interface in TensorCircuit-NG.
It implements a hybrid digital-analog variational ansatz to find the ground state
of a 1D Transverse Field Ising Model (TFIM) at the critical point.

Key features demonstrated:
1.  **AnalogCircuit**: Mixing digital gates with analog time evolution blocks.
2.  **Hybrid Ansatz**:
    - Digital: Parameterized single-qubit rotations (Rx, Ry).
    - Analog (Global): Hamiltonian evolution of driving fields (X).
3.  **JAX + Optax**: End-to-end automatic differentiation and optimization.
4.  **Parameter Optimization**: Simulatneous optimization of gate angles and evolution times.
"""

import time
import matplotlib.pyplot as plt
import optax
import tensorcircuit as tc

# 1. Setup Backend
# AnalogCircuit currently supports JAX backend
tc.set_backend("jax")
tc.set_dtype("complex128")
K = tc.backend

from tensorcircuit.quantum import PauliStringSum2COO


def get_tfim_hamiltonian(n, J, g):
    r"""
    Construct the Transverse Field Ising Model (TFIM) Hamiltonian.
    H = -J \sum Z_i Z_{i+1} - g \sum X_i
    """
    ls = []
    weights = []

    # -J \sum Z_i Z_{i+1}
    for i in range(n):
        s = [0] * n
        s[i] = 3  # Z
        s[(i + 1) % n] = 3  # Z (PBC)
        ls.append(s)
        weights.append(-J)

    # -g \sum X_i
    for i in range(n):
        s = [0] * n
        s[i] = 1  # X
        ls.append(s)
        weights.append(-g)

    H_sparse = PauliStringSum2COO(ls, weights, numpy=False)
    H_dense = K.to_dense(H_sparse)
    return H_dense


def run_digital_analog_vqe():
    print("\n--- Digital-Analog VQE (DA-VQE) Demo ---\n")

    # --- Configuration ---
    nqubits = 6
    depth = 6  # Number of layers
    J = 1.0
    g = 1.0  # Critical point
    print(f"System: {nqubits} qubits, Depth: {depth}")
    print(f"Target Model: TFIM (J={J}, g={g})")

    # Get target ground state energy
    H_target = get_tfim_hamiltonian(nqubits, J, g)
    evals = K.eigh(H_target)[0]
    expected_energy = evals[0]
    print(f"Target Ground State Energy: {expected_energy:.6f}")

    # --- Define Hamiltonian Functions for Analog Blocks ---
    # These functions take time 't' and return the Hamiltonian matrix H(t).

    # Pre-compute matrices to avoid re-creation in loop
    # --- Hamiltonian Functions ---
    # Interaction: Z_i Z_{i+1}
    # helper gives -J ZZ - g X.
    # To get ZZ, we set J=-1, g=0.
    H_int_dense = get_tfim_hamiltonian(nqubits, J=-1.0, g=0.0)

    # Drive: \sum X_i
    # To get X, we set J=0, g=-1.
    H_drive_dense = get_tfim_hamiltonian(nqubits, J=0.0, g=-1.0)

    def h_interaction(t):
        return H_int_dense

    def h_drive(t):
        return H_drive_dense

    # --- Ansatz Construction ---
    def ansatz(params):
        """
        params structure:
        - gate_params: shape [depth, nqubits, 2] (Rx, Ry angles)
        - time_params: shape [depth, 2] (t_interaction, t_drive)
        """
        gate_params, time_params = params

        c = tc.AnalogCircuit(nqubits)
        c.set_solver_options(ode_backend="diffrax")

        for d in range(depth):
            # 1. Digital Layer: Rx, Ry
            for i in range(nqubits):
                c.rx(i, theta=gate_params[d, i, 0])
                c.ry(i, theta=gate_params[d, i, 1])

            # 2. Analog Interaction Block (Global ZZ evolution)
            # Duration is optimized
            t_int = K.abs(time_params[d, 0])  # Ensure positive time
            c.add_analog_block(h_interaction, time=t_int)

            # 3. Analog Drive Block (Global X evolution)
            # Demonstrating mixing
            t_drive = K.abs(time_params[d, 1])
            c.add_analog_block(h_drive, time=t_drive)

        return c

    # --- Loss Function ---
    def loss_fn(params):
        c = ansatz(params)

        return K.real(tc.templates.measurements.operator_expectation(c, H_target))

    # --- Optimization Loop ---

    # Initialize parameters
    # Gates: Random small angles
    # Times: Small positive times
    key = K.get_random_state(42)
    key1, key2 = K.random_split(key)

    init_gate_params = K.stateful_randn(key1, [depth, nqubits, 2], stddev=0.1)
    init_time_params = K.abs(K.stateful_randn(key2, [depth, 2], stddev=0.1)) + 0.1

    params = (init_gate_params, init_time_params)

    optimizer = optax.adam(0.02)
    opt_state = optimizer.init(params)

    loss_val_grad = K.jit(K.value_and_grad(loss_fn))

    history = []

    print("Starting optimization...")
    start_time = time.time()

    steps = 500
    for i in range(steps):
        loss, grads = loss_val_grad(params)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Ensure times stay positive (optional, but physically meaningful)
        # Note: In ansatz we use K.abs, so parameters can be negative, but let's clamp for cleanliness
        # params = (params[0], K.abs(params[1]))

        history.append(loss)

        if i % 10 == 0:
            print(f"Step {i:03d}: Energy = {loss:.6f}")

    total_time = time.time() - start_time
    print(f"\nOptimization completed in {total_time:.2f}s")
    print(f"Final Energy: {history[-1]:.6f}")
    print(f"Error: {history[-1] - expected_energy:.2e}")
    print("\nOptimized Time Intervals:")
    for d in range(depth):
        t_int = K.abs(params[1][d, 0])
        t_drive = K.abs(params[1][d, 1])
        print(f"Layer {d}: Interaction Time = {t_int:.4f}, Drive Time = {t_drive:.4f}")

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    plt.plot(history, label="DA-VQE Energy", linewidth=2)
    plt.axhline(expected_energy, color="r", linestyle="--", label="Target Ground State")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Energy")
    plt.title(f"Digital-Analog VQE on {nqubits}-qubit TFIM")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = "analog_vqe_convergence.pdf"
    plt.savefig(save_path)
    print(f"Convergence plot saved to '{save_path}'")


if __name__ == "__main__":
    run_digital_analog_vqe()
