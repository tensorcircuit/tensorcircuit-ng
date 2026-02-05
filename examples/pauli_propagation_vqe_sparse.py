r"""
VQE for 2D Heisenberg Model with Sparse Pauli Propagation
==========================================================

This example demonstrates the use of the `SparsePauliPropagationEngine` (PPE)
for variational quantum eigensolver (VQE) tasks on 2D lattices.

Key features demonstrated:
1.  **Sparse vs Dense PPE**: Comparing accuracy and performance between the standard
    dense engine and the memory-efficient sparse engine.
2.  **Scalability**: Running a 100-qubit circuit simulation that would be
    prohibitively expensive for standard state-vector or dense PPE methods.
3.  **Automatic Differentiation**: Using JAX to compute gradients through the
    Heisenberg-picture propagation.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.pauliprop import PauliPropagationEngine, SparsePauliPropagationEngine

# Configure TensorCircuit to use JAX backend and high precision
tc.set_backend("jax")
tc.set_dtype("complex128")


def get_heisenberg_2d_hamiltonian(Lx, Ly, J=(1.0, 1.0, 1.0), pbc=True):
    """
    Generate the 2D Heisenberg Hamiltonian on a grid.

    :param Lx: Lattice width
    :param Ly: Lattice height
    :param J: Tuple of (Jx, Jy, Jz) interaction strengths
    :param pbc: Whether to use periodic boundary conditions
    :return: (structures, weights) in TensorCircuit PPE format
    """
    num_qubits = Lx * Ly
    lattice = tc.templates.graphs.Grid2DCoord(Lx, Ly).lattice_graph(pbc=pbc)
    edges = list(lattice.edges)
    num_edges = len(edges)

    num_terms = 3 * num_edges
    structures = np.zeros((num_terms, num_qubits), dtype=int)
    weights = np.zeros(num_terms, dtype=np.complex128)

    idx = 0
    for u, v in edges:
        for op in range(1, 4):  # 1:X, 2:Y, 3:Z
            structures[idx, u] = op
            structures[idx, v] = op
            weights[idx] = J[op - 1]
            idx += 1

    return structures, weights, edges


def run_comparison_benchmark(Lx=4, Ly=2, k=3, layers=3, buffer_size=1000):
    """
    Compare Dense and Sparse engines on a small lattice.
    """
    num_qubits = Lx * Ly
    print(f"\n--- Benchmark: {Lx}x{Ly} Lattice ({num_qubits} Qubits) ---")
    print(f"Parameters: k={k}, layers={layers}, buffer_size={buffer_size}")

    structures, weights, edges = get_heisenberg_2d_hamiltonian(Lx, Ly)
    weights_t = tc.backend.convert_to_tensor(weights)
    # edges_t = jnp.array(edges)
    num_edges = len(edges)

    # Initialize random parameters
    key = jax.random.PRNGKey(42)
    params = jax.random.uniform(key, (layers, num_edges, 3), minval=-0.05, maxval=0.05)

    def loss_factory(engine):
        def loss_fn(p_layers):
            state = engine.get_initial_state(structures, weights_t)

            # Apply layers back-to-front (Heisenberg picture)
            for p_l in p_layers[::-1]:
                for i in range(num_edges - 1, -1, -1):
                    u, v = edges[i]
                    ps = p_l[i]
                    # Apply RXX, RYY, RZZ gates
                    state = engine.apply_gate(state, "rzz", [u, v], params=ps[2])
                    state = engine.apply_gate(state, "ryy", [u, v], params=ps[1])
                    state = engine.apply_gate(state, "rxx", [u, v], params=ps[0])

            # Add some local fields
            for i in range(num_qubits):
                state = engine.apply_gate(state, "rx", [i], params=0.1)

            return jnp.real(engine.expectation(state))

        return loss_fn

    # 1. Dense Engine
    dense_engine = PauliPropagationEngine(num_qubits, k)
    dense_loss_fn = jax.jit(tc.backend.value_and_grad(loss_factory(dense_engine)))

    t0 = time.time()
    v_dense, g_dense = dense_loss_fn(params)
    v_dense.block_until_ready()
    t_dense = time.time() - t0
    print(f"Dense Engine  | Loss: {v_dense:.8f} | Time: {t_dense:.4f}s (JIT + Exec)")

    # 2. Sparse Engine
    sparse_engine = SparsePauliPropagationEngine(num_qubits, k, buffer_size=buffer_size)
    sparse_loss_fn = jax.jit(tc.backend.value_and_grad(loss_factory(sparse_engine)))

    t0 = time.time()
    v_sparse, g_sparse = sparse_loss_fn(params)
    v_sparse.block_until_ready()
    t_sparse = time.time() - t0
    print(f"Sparse Engine | Loss: {v_sparse:.8f} | Time: {t_sparse:.4f}s (JIT + Exec)")

    # 3. Accuracy Check
    diff = jnp.abs(v_dense - v_sparse)
    g_diff = jnp.linalg.norm(
        jax.tree_util.tree_flatten(g_dense)[0][0]
        - jax.tree_util.tree_flatten(g_sparse)[0][0]
    )
    print(f"Accuracy      | Value Diff: {diff:.2e} | Grad Norm Diff: {g_diff:.2e}")


def run_large_scale_demo():
    """
    Demonstrate 100-qubit scalability using the Sparse engine.
    """
    Lx, Ly = 10, 10
    num_qubits = Lx * Ly
    k = 3
    buffer_size = 5000

    print(f"\n--- 100-Qubit Scalability Demo (Sparse PPE) ---")
    print(f"Lattice: {Lx}x{Ly} Grid | k={k} | Buffer Size={buffer_size}")

    structures, weights, edges = get_heisenberg_2d_hamiltonian(Lx, Ly, pbc=False)
    weights_t = tc.backend.convert_to_tensor(weights)
    # edges_t = jnp.array(edges)
    num_edges = len(edges)

    engine = SparsePauliPropagationEngine(num_qubits, k, buffer_size=buffer_size)

    @jax.jit
    def apply_all_rxx(s):
        for u, v in edges:
            # Simulate a small time evolution or variational layer
            s = engine.apply_gate(s, "rxx", [u, v], params=0.05)
        return s

    state = engine.get_initial_state(structures, weights_t)

    print(f"Propagating {num_edges} RXX gates across 100 qubits...")
    t0 = time.time()
    state = apply_all_rxx(state)
    # PPE expectation is cheap once the state is propagated
    final_exp = engine.expectation(state)
    # Trigger execution
    final_exp.block_until_ready()
    duration = time.time() - t0

    print(f"Finished in {duration:.4f}s")
    print(f"Final Expectation Value: {final_exp:.6f}")


if __name__ == "__main__":
    # Small scale comparison to verify correctness
    run_comparison_benchmark(Lx=4, Ly=2, k=3, layers=2, buffer_size=500)

    # Large scale demo to show performance
    run_large_scale_demo()
