"""
Batch VQE for Transverse Field Clock Model (Qudit System)
=========================================================

This script demonstrates the unique capabilities of TensorCircuit-NG in simulating and optimizing
qudit quantum systems (d=3, 4). It showcases:

1.  **Transverse Field Clock Model**: A generalization of the Ising model to d-dimensions.
2.  **Batch VQE**: Using `vmap` to optimize multiple random ansatz initializations in parallel.
3.  **Exact Verification**: Comparing VQE results with exact diagonalization of the sparse Hamiltonian.

Model:
    H = - J \\sum_i (Z_i Z_{i+1}^\\dagger + h.c.) - h \\sum_i (X_i + X_i^\\dagger)

    where Z is the Clock operator and X is the Shift operator.
"""

import time
import numpy as np
import scipy.sparse as sp
import optax
import tensorcircuit as tc

# Use JAX backend for high-performance JIT, VMAP and AD
tc.set_backend("jax")
tc.set_dtype("complex128")


def get_clock_hamiltonian_sparse(n, d, J, h_field):
    """
    Constructs the Transverse Field Clock Model Hamiltonian as a sparse matrix.
    H = - J * sum (Z_i Z_{i+1}^dag + h.c.) - h_field * sum (X_i + X_i^dag)
    """

    import tensorcircuit.quditgates as qg

    X = qg.x_matrix_func(d)
    Z = qg.z_matrix_func(d)

    # Check X
    # print(X) # [[0 0 1] [1 0 0] [0 1 0]]

    X_dag = X.conj().T
    Z_dag = Z.conj().T

    # Interaction term: -J (Z_i Z_{i+1}^dag + h.c.)
    H_int = sp.coo_matrix((d**n, d**n), dtype=np.complex128)

    for i in range(n - 1):  # Open Boundary Conditions
        # Z_i Z_{i+1}^dag
        ops = [sp.eye(d)] * n
        ops[i] = sp.csr_matrix(Z)
        ops[i + 1] = sp.csr_matrix(Z_dag)

        term = ops[0]
        for op in ops[1:]:
            term = sp.kron(term, op)

        H_int += term + term.conj().T

    # Field term: -h (X_i + X_i^dag)
    H_field = sp.coo_matrix((d**n, d**n), dtype=np.complex128)
    for i in range(n):
        ops = [sp.eye(d)] * n
        ops[i] = sp.csr_matrix(X + X_dag)

        term = ops[0]
        for op in ops[1:]:
            term = sp.kron(term, op)

        H_field += term

    H = -J * H_int - h_field * H_field
    return H


def run_simulation(n, d, n_layers, batch_size, J, h_field):
    print(f"\n=== Running Simulation for System: {n} qudits (d={d}) ===")
    print(f"VQE Batch Size: {batch_size}")

    # --- 1. Exact Diagonalization ---
    print("\n[1] Calculating Exact Ground State Energy...")
    H_sparse = get_clock_hamiltonian_sparse(n, d, J, h_field)

    # Use scipy.sparse.linalg.eigsh for lowest eigenvalue
    evals, _ = sp.linalg.eigsh(H_sparse, k=1, which="SA")
    exact_energy = evals[0]
    print(f"Exact Ground State Energy: {exact_energy:.6f}")

    # --- 2. VQE Setup ---

    # Convert Hamiltonian to TensorCircuit Sparse Tensor
    H_sparse = H_sparse.tocoo()
    indices = np.vstack((H_sparse.row, H_sparse.col)).T
    values = H_sparse.data

    indices_tc = tc.backend.convert_to_tensor(indices)
    values_tc = tc.backend.convert_to_tensor(values, dtype="complex128")
    shape_tc = H_sparse.shape

    H_tc_sparse = tc.backend.coo_sparse_matrix(indices_tc, values_tc, shape_tc)

    def ansatz_circuit(params):
        # params shape: (n_layers, n, n_params_per_qudit)
        # n_params_per_qudit = d + d*(d-1)/2
        c = tc.QuditCircuit(n, dim=d)

        # Iterate over layers
        for l in range(n_layers):
            # Single qudit rotations
            for i in range(n):
                p_idx = 0

                # RZ on each level
                for k in range(d):
                    c.rz(i, theta=params[l, i, p_idx], j=k)
                    p_idx += 1

                # RY on each bilevel pair (j, k)
                for j in range(d):
                    for k in range(j + 1, d):
                        c.ry(i, theta=params[l, i, p_idx], j=j, k=k)
                        p_idx += 1

            # Entangling layer (CSUM gates)
            # Nearest neighbor chain
            for i in range(n - 1):
                c.csum(i, i + 1)

        return c

    def get_energy(params):
        c = ansatz_circuit(params)
        return tc.templates.measurements.operator_expectation(c, H_tc_sparse)

    # --- 3. Batch VQE with VMAP ---
    print("\n[2] Running Batch VQE (JIT + VMAP)...")

    # Parameter shape
    # Rz on each level (d) + Ry on each pair (d*(d-1)/2)
    n_params_per_qudit = d + (d * (d - 1)) // 2
    param_shape = (n_layers, n, n_params_per_qudit)

    # Initialize batch of parameters
    # shape: (batch_size, n_layers, n, 2)
    params_batch = tc.backend.implicit_randn(shape=(batch_size,) + param_shape)

    # VQE Optimization with Optax
    lr = 0.02
    optimizer = optax.adam(lr)

    # Define single-step update function
    def step_fn(param, state):
        # Calculate gradients for a single instance
        energy, grads = tc.backend.value_and_grad(get_energy)(param)
        # Update optimizer state
        updates, state = optimizer.update(grads, state, param)
        # Apply updates
        param = optax.apply_updates(param, updates)
        return param, state, energy

    # Vectorize and JIT the step function
    # this maps step_fn over the leading batch dimension (0) for both params and state
    batch_step = tc.backend.jit(tc.backend.vmap(step_fn, vectorized_argnums=(0, 1)))

    # Initialize optimizer state for the batch
    opt_state = tc.backend.vmap(optimizer.init)(params_batch)

    print("Optimization Loop (Optax Adam)...")
    losses_history = []

    t0 = time.time()
    for i in range(500):
        params_batch, opt_state, energies = batch_step(params_batch, opt_state)

        losses_history.append(tc.backend.numpy(energies))

        if i % 100 == 0:
            min_e = np.min(energies)
            print(
                f"Step {i:4d}: Min Energy = {min_e:.6f} | Mean Energy = {np.mean(energies):.6f}"
            )

    t1 = time.time()
    print(f"Optimization finished in {t1-t0:.4f} seconds.")

    # --- 4. Results ---
    final_energies = losses_history[-1]
    best_idx = np.argmin(final_energies)
    best_energy = final_energies[best_idx]

    print("\n[3] Results Analysis")
    print(f"Best VQE Energy: {best_energy:.6f}")
    print(f"Exact Energy:    {exact_energy:.6f}")
    print(f"Error:           {abs(best_energy - exact_energy):.2e}")

    if abs(best_energy - exact_energy) < 1e-2:
        print("SUCCESS: VQE converged to ground state!")
    else:
        print("WARNING: VQE close but did not fully converge.")


def main():
    print("TensorCircuit-NG Qudit Batch VQE Demo")
    print("-------------------------------------")

    # Hamiltonian Parameters
    J = 1.0
    h_field = 1.0

    # Simulation 1: Qutrits (d=3)
    run_simulation(n=4, d=3, n_layers=6, batch_size=8, J=J, h_field=h_field)

    # Simulation 2: Ququarts (d=4)
    run_simulation(n=4, d=4, n_layers=6, batch_size=8, J=J, h_field=h_field)


if __name__ == "__main__":
    main()
