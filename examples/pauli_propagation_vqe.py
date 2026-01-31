"""
TFIM VQE with Pauli Propagation
================================

This example demonstrates how to use the Pauli Propagation module to find the
ground state energy of a Transverse Field Ising Model (TFIM) Hamiltonian.

Pauli Propagation tracks the evolution of Pauli strings in the Heisenberg picture,
which is particularly efficient for circuits with limited qubit-interaction locality.
"""

import time
import numpy as np
import jax.numpy as jnp
import optax
import tensorcircuit as tc

# Use JAX backend for automatic differentiation and efficient scanning
tc.set_backend("jax")


def example_tfim_vqe():
    # 1. Configuration
    N = 12  # Number of qubits
    layers = 4  # Number of circuit layers
    k = 3  # Local Pauli tracking limit (Pauli string length <= k)
    g = 1.0  # Transverse field strength

    print(f"TFIM VQE with Pauli Propagation: N={N}, layers={layers}, k={k}, g={g}")

    # 2. Define Hamiltonian: H = \sum Z_i Z_{i+1} + g \sum X_i
    # We use tc-internal format for efficiency in the engine:
    # 0 -> I, 1 -> X, 2 -> Y, 3 -> Z
    num_terms = (N - 1) + N
    ham_structures = np.zeros((num_terms, N), dtype=int)
    ham_weights = np.zeros(num_terms, dtype=np.float32)

    idx = 0
    # Nearest-neighbor ZZ terms
    for i in range(N - 1):
        ham_structures[idx, i] = 3  # Z
        ham_structures[idx, i + 1] = 3  # Z
        ham_weights[idx] = 1.0
        idx += 1

    # Transverse field X terms
    for i in range(N):
        ham_structures[idx, i] = 1  # X
        ham_weights[idx] = g
        idx += 1

    # 3. Parameter Initialization
    # Each layer has (N-1) Rzz parameters and N Rx parameters
    params_per_layer = (N - 1) + N
    total_params = layers * params_per_layer
    # Using small random initialization
    params = np.random.normal(scale=0.1, size=total_params).astype(np.float32)
    params = jnp.array(params)

    # 4. Define Loss Function
    pp_engine = tc.pauliprop.PauliPropagationEngine(N, k)

    def loss_fn(p_flat):
        # Reshape parameters for efficient scanning over layers
        p_layers = jnp.reshape(p_flat, (layers, params_per_layer))

        # Define how one layer of the circuit is constructed
        def layer_fn(c, p_l):
            p_rzz = p_l[: N - 1]
            p_rx = p_l[N - 1 :]

            # 1. Apply Rx gates
            for i in range(N):
                c.rx(i, theta=p_rx[i])

            # 2. Apply Rzz gates
            # Heisenberg ordering is handled automatically by pp_engine
            # Even-indexed pairs
            for i in range(0, N - 1, 2):
                c.rzz(i, i + 1, theta=p_rzz[i // 2])
            # Odd-indexed pairs
            num_even = len(range(0, N - 1, 2))
            for i in range(1, N - 1, 2):
                c.rzz(i, i + 1, theta=p_rzz[num_even + (i - 1) // 2])

        # Compute expectation by propagating the Hamiltonian back through layers
        energy = pp_engine.compute_expectation_scan(
            ham_structures, ham_weights, layer_fn, p_layers
        )
        return jnp.real(energy)

    # 5. Optimization Loop
    optimizer = optax.adam(learning_rate=0.02)
    opt_state = optimizer.init(params)

    # JIT compile the value and gradient function for maximum performance
    loss_val_grad_fn = tc.backend.jit(tc.backend.value_and_grad(loss_fn))

    print("\nStarting Optimization Loop...")
    t_start = time.time()
    for i in range(201):
        # Compute value and gradients
        val, grads = loss_val_grad_fn(params)

        # Apply updates
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i % 10 == 0:
            # Sync JAX if measuring time inside the loop is needed
            # val.block_until_ready()
            print(f"Step {i:3d}: Energy = {val:.6f}")

    duration = time.time() - t_start
    print(f"\nOptimization finished in {duration:.2f}s")
    print(f"Final VQE Energy: {val:.6f}")


if __name__ == "__main__":
    example_tfim_vqe()
