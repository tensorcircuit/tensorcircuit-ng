"""
Qiskit Translation VQE Demo (JAX + Optax)
=========================================

This script demonstrates the key power of TensorCircuit:
Translating a parameterized Qiskit circuit (Ansatz) into a TensorCircuit function,
and then JIT-compiling the entire VQE optimization loop.
"""

import jax
import optax
from qiskit.circuit.library import RealAmplitudes
import scipy.sparse as sp
import tensorcircuit as tc

# Use JAX backend
K = tc.set_backend("jax")


def main():
    print("TensorCircuit Qiskit Translation VQE Demo")
    print("-------------------------------------------------------")

    # 1. Define the Problem (Hamiltonian)
    n = 8
    g = 1.0
    h = 0.5
    print(f"System size: {n} qubits")
    print(f"Hamiltonian: TFIM with J=1.0, g={g}, h={h}")

    def get_tfim_hamiltonian_sparse(n, g, h):
        # H = - \sum Z_i Z_{i+1} - g \sum X_i - h \sum Z_i

        # tc.quantum.PauliStringSum2COO expects list of integer lists
        # 0: I, 1: X, 2: Y, 3: Z

        ls = []
        weights = []

        # ZZ terms
        for i in range(n - 1):
            term = [0] * n
            term[i] = 3  # Z
            term[i + 1] = 3  # Z
            ls.append(term)
            weights.append(-1.0)

        # X terms
        for i in range(n):
            term = [0] * n
            term[i] = 1  # X
            ls.append(term)
            weights.append(-g)

        # Z terms
        for i in range(n):
            term = [0] * n
            term[i] = 3  # Z
            ls.append(term)
            weights.append(-h)

        # Construct sparse matrix
        ham = tc.quantum.PauliStringSum2COO(ls, weights)
        return ham

    print("Constructing sparse Hamiltonian with PauliStringSum2COO...")
    h_scipy_sparse = get_tfim_hamiltonian_sparse(n, g, h)

    # Check sparsity
    print(f"Hamiltonian shape: {h_scipy_sparse.shape}")

    # Calculate Exact Ground State Energy
    print("Calculating exact ground state energy...")
    # eigsh for smallest real eigenvalue (which='SA')
    # Use scipy sparse matrix for eigsh
    w, _ = sp.linalg.eigsh(K.numpy(h_scipy_sparse), k=1, which="SA")
    exact_energy = w[0]
    print(f"Exact Ground State Energy: {exact_energy:.6f}")

    # 2. Define Qiskit Ansatz
    print("Constructing Qiskit Ansatz (RealAmplitudes)...")
    ansatz = RealAmplitudes(num_qubits=n, reps=3, entanglement="linear")
    num_params = ansatz.num_parameters
    print(f"Number of parameters: {num_params}")

    # 3. Define VQE Loss Function

    def vqe_loss(params):
        # Translate Qiskit circuit to TensorCircuit
        c = tc.translation.qiskit2tc(ansatz, n=n, binding_params=params)

        # Calculate Energy using operator_expectation
        e = tc.templates.measurements.operator_expectation(c, h_scipy_sparse)
        return tc.backend.real(e)

    # 4. Optimization Loop with Optax

    vqe_loss_vg = tc.backend.jit(tc.backend.value_and_grad(vqe_loss))

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, shape=(num_params,)) * 0.1

    # Optimizer setup
    lr = 2e-2
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    print("\nStarting Optimization Loop...")

    max_steps = 300
    history = []

    # JIT the update step
    @jax.jit
    def update_step(params, opt_state):
        loss, grads = vqe_loss_vg(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for i in range(max_steps):
        params, opt_state, loss = update_step(params, opt_state)
        loss_val = float(loss)
        history.append(loss_val)

        if i % 20 == 0:
            print(f"Step {i}: Energy = {loss_val:.6f}")

    print(f"Final Energy: {history[-1]:.6f}")
    print(f"Exact Energy: {exact_energy:.6f}")
    print(f"Difference: {abs(history[-1] - exact_energy):.6f}")


if __name__ == "__main__":
    main()
