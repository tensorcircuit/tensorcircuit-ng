"""
VQE optimization for 20-qubit TFIM using Matrix Product States (MPS)
"""

import time
from scipy.sparse.linalg import eigsh
import optax
import tensorcircuit as tc

# Use JAX backend for automatic differentiation and JIT compilation
K = tc.set_backend("jax")
tc.set_dtype("complex64")

# ============================================================================
# Configuration
# ============================================================================

n = 18  # Number of qubits
g = 1.0  # Transverse field strength
nlayers = 5  # Depth of the circuit
maxiter = 300  # Optimization steps (increased for better convergence)
lr = 0.01  # Learning rate

# ============================================================================
# Exact Ground State Energy
# ============================================================================


def get_exact_energy(n, g):
    """
    Compute the exact ground state energy of TFIM using sparse matrix diagonalization.
    H = - ∑ Z_i Z_{i+1} - g ∑ X_i
    """
    print(f"Computing exact ground state energy for n={n}...")
    J = -1.0
    h_x = -g
    # Periodic Boundary Conditions (PBC) for comparison if needed, but usually OBC for MPS
    # Let's stick to Open Boundary Conditions (OBC) to match MPS implementation simpler

    # Construct Hamiltonian in sparse format
    # Terms: Z_i Z_{i+1}
    terms = []
    weights = []

    for i in range(n - 1):
        t = [0] * n
        t[i] = 3  # Z
        t[i + 1] = 3  # Z
        terms.append(t)
        weights.append(J)

    # Terms: X_i
    for i in range(n):
        t = [0] * n
        t[i] = 1  # X
        terms.append(t)
        weights.append(h_x)

    hamiltonian = tc.quantum.PauliStringSum2COO(terms, weights, numpy=True)
    energy, _ = eigsh(hamiltonian, k=1, which="SA")
    return energy[0]


# ============================================================================
# MPS VQE Model
# ============================================================================


def tfim_energy_mps(params):
    """
    Compute the energy expectation value using MPS simulator.
    """
    # MPSCircuit helps manage bond dimensions and truncation automatically/manually
    c = tc.MPSCircuit(n, split={"max_singular_values": 16})

    # Apply Ansatz
    # Using a hardware-efficient ansatz suitable for MPS (linear connectivity)

    # Initial layer of H
    for i in range(n):
        c.h(i)

    # Variational layers
    param_idx = 0
    for _ in range(nlayers):
        # Odd layers: Rzz entangling gates
        for i in range(0, n - 1, 2):
            c.rzz(i, i + 1, theta=params[param_idx])
            param_idx += 1
        for i in range(1, n - 1, 2):
            c.rzz(i, i + 1, theta=params[param_idx])
            param_idx += 1

        # Even layers: Rx and Ry rotations
        for i in range(n):
            c.rx(i, theta=params[param_idx])
            param_idx += 1
        for i in range(n):
            c.ry(i, theta=params[param_idx])
            param_idx += 1

    # Calculate Energy Expectation
    # H = - ∑ Z_i Z_{i+1} - g ∑ X_i

    e_total = 0.0

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # MPSCircuit.expectation takes a tuple of (gate, index) or list of them
        # Note: expectation returns a tensor, we need to sum them
        val = c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [i + 1]))
        e_total -= val

    # X_i terms
    for i in range(n):
        val = c.expectation((tc.gates.x(), [i]))
        e_total -= g * val

    return K.real(e_total)


# ============================================================================
# Main Optimization Loop
# ============================================================================


def main():
    print(f"TensorCircuit MPS VQE Demo")
    print(f"Configuration: n={n}, g={g}, layers={nlayers}")

    # 1. Exact Energy
    t0 = time.time()
    e_exact = get_exact_energy(n, g)
    print(f"Exact Energy: {e_exact:.6f} (Time: {time.time()-t0:.2f}s)")

    # 2. Setup VQE
    # Layer structure:
    # Rzz (evens): n//2 parameters
    # Rzz (odds): (n-1)//2 parameters
    # Rx: n parameters
    # Ry: n parameters
    # Total per layer: n - 1 + 2n = 3n - 1
    # Total params = nlayers * (3n - 1)

    params_per_layer = (n // 2) + ((n - 1) // 2) + 2 * n
    n_params = nlayers * params_per_layer
    print(f"Number of parameters: {n_params}")

    # Initialize parameters
    params = K.implicit_randn([n_params], stddev=0.1)

    # Optimizer
    opt = K.optimizer(optax.adam(lr))

    # Value and Grad
    veg_func = K.value_and_grad(tfim_energy_mps)
    # JIT the step function
    jit_veg = K.jit(veg_func)

    print("\nStarting VQE Optimization...")
    print(f"{'Step':<10} {'Energy':<15} {'Error':<15} {'Time':<10}")
    print("-" * 50)

    # t_start = time.time()
    for i in range(maxiter):
        t_step = time.time()
        energy, grads = jit_veg(params)
        params = opt.update(grads, params)
        t_step_end = time.time()
        step_time = t_step_end - t_step

        error = energy - e_exact

        # Print first few steps to show JIT compilation difference, then every 10 steps
        if i % 10 == 0 or i == maxiter - 1 or i < 3:
            print(f"{i:<10d} {energy:<15.6f} {error:<15.6f} {step_time:<10.4f}")

    print("-" * 50)
    print(f"Final Energy: {energy:.6f}")
    print(f"Exact Energy: {e_exact:.6f}")
    print(f"Final Error:  {error:.6f}")


if __name__ == "__main__":
    main()
