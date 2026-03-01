"""
Physics-relevant demonstration of hyperedge support in TensorCircuit.
Computing the partition function of a 2D classical Ising model using CopyNodes.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import tensornetwork as tn
import tensorcircuit as tc

# Set backend to JAX for JIT and AD support
tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")


def ising_partition_function(L, beta, J=1.0):
    """
    Compute the partition function of a 2D Ising model on an L x L grid.
    Uses CopyNodes to represent spins and 2-index tensors for Boltzmann factors.

    The partition function is Z = sum_{s} exp(beta * J * sum_{<i,j>} s_i * s_j).
    Each site i has a spin s_i in {1, -1}.
    Each bond <i,j> contributes a factor exp(beta * J * s_i * s_j).
    """
    # Boltzmann factor matrix M_{si, sj} = exp(beta * J * si * sj)
    # spins are {1, -1}, mapped to indices {0, 1}
    # si*sj = 1 if si==sj (indices 00 or 11), -1 if si!=sj (indices 01 or 10)
    # Using jnp.exp to allow AD through beta
    M = jnp.array(
        [
            [jnp.exp(beta * J), jnp.exp(-beta * J)],
            [jnp.exp(-beta * J), jnp.exp(beta * J)],
        ]
    )

    nodes = []
    # Grid of CopyNodes (Delta tensors) representing the spins
    grid = [[None for _ in range(L)] for _ in range(L)]

    for i in range(L):
        for j in range(L):
            # Determine degree of CopyNode based on neighbors (open BC)
            degree = 0
            if i > 0:
                degree += 1
            if i < L - 1:
                degree += 1
            if j > 0:
                degree += 1
            if j < L - 1:
                degree += 1

            # A CopyNode(degree, 2) enforces that all connected legs have the same value (spin state)
            cn = tn.CopyNode(degree, 2, name=f"site_{i}_{j}")
            grid[i][j] = cn

    # Track which axis of each CopyNode is used as we connect bonds
    axis_ptr = [[0 for _ in range(L)] for _ in range(L)]

    # Add bond tensors and connect to the CopyNodes
    for i in range(L):
        for j in range(L):
            # Horizontal bond to the right
            if j < L - 1:
                bond_h = tn.Node(M, name=f"bond_h_{i}_{j}")
                nodes.append(bond_h)
                grid[i][j][axis_ptr[i][j]] ^ bond_h[0]
                grid[i][j + 1][axis_ptr[i][j + 1]] ^ bond_h[1]
                axis_ptr[i][j] += 1
                axis_ptr[i][j + 1] += 1

            # Vertical bond downwards
            if i < L - 1:
                bond_v = tn.Node(M, name=f"bond_v_{i}_{j}")
                nodes.append(bond_v)
                grid[i][j][axis_ptr[i][j]] ^ bond_v[0]
                grid[i + 1][j][axis_ptr[i + 1][j]] ^ bond_v[1]
                axis_ptr[i][j] += 1
                axis_ptr[i + 1][j] += 1

    # Multi-node contraction with cotengra (which handles hyperedges efficiently)
    # The algebraic path is triggered automatically because CopyNodes are present.
    all_nodes = nodes + [grid[i][j] for i in range(L) for j in range(L)]

    # Ensure cotengra is used for high-performance contraction
    z_node = tc.contractor(all_nodes)

    return z_node.tensor


def ising_partition_function_peps(L, beta, J=1.0):
    """
    Compute the partition function using standard explicit PEPS (without CopyNodes).
    Serves as a verification baseline.
    """
    M = jnp.array(
        [
            [jnp.exp(beta * J), jnp.exp(-beta * J)],
            [jnp.exp(-beta * J), jnp.exp(beta * J)],
        ]
    )

    nodes = []
    grid = [[None for _ in range(L)] for _ in range(L)]

    for i in range(L):
        for j in range(L):
            degree = 0
            if i > 0:
                degree += 1
            if i < L - 1:
                degree += 1
            if j > 0:
                degree += 1
            if j < L - 1:
                degree += 1

            T = jnp.zeros((2,) * degree, dtype=M.dtype)
            T = T.at[(0,) * degree].set(1.0)
            T = T.at[(1,) * degree].set(1.0)
            grid[i][j] = tn.Node(T, name=f"site_peps_{i}_{j}")

    axis_ptr = [[0 for _ in range(L)] for _ in range(L)]

    for i in range(L):
        for j in range(L):
            if j < L - 1:
                bond_h = tn.Node(M, name=f"bond_peps_h_{i}_{j}")
                nodes.append(bond_h)
                grid[i][j][axis_ptr[i][j]] ^ bond_h[0]
                grid[i][j + 1][axis_ptr[i][j + 1]] ^ bond_h[1]
                axis_ptr[i][j] += 1
                axis_ptr[i][j + 1] += 1

            if i < L - 1:
                bond_v = tn.Node(M, name=f"bond_peps_v_{i}_{j}")
                nodes.append(bond_v)
                grid[i][j][axis_ptr[i][j]] ^ bond_v[0]
                grid[i + 1][j][axis_ptr[i + 1][j]] ^ bond_v[1]
                axis_ptr[i][j] += 1
                axis_ptr[i + 1][j] += 1

    all_nodes = nodes + [grid[i][j] for i in range(L) for j in range(L)]
    z_node = tc.contractor(all_nodes)
    return z_node.tensor


def main():
    L = 15
    J = 1.0
    betas = [0.1, 0.4, 0.44, 0.5, 0.8]  # Multiple betas including near critical
    print(f"--- 2D Ising Model Partition Function ({L}x{L} lattice) ---")
    print(f"Backend: {tc.backend.name}, Parameters: J={J}, Betas={betas}")

    print("\n--- Verifying Correctness & AD ---")
    # 1. Verification and AD for a single beta
    beta_test = betas[1]
    z = ising_partition_function(L, beta_test, J)
    z_peps = ising_partition_function_peps(L, beta_test, J)
    np.testing.assert_allclose(
        z, z_peps, rtol=1e-5, err_msg="Mismatch between CopyNode and PEPS construction!"
    )

    def log_z(beta_val):
        val = ising_partition_function(L, beta_val, J)
        return jnp.log(tc.backend.real(val))

    energy_fn = jax.grad(log_z)
    energy = energy_fn(beta_test)
    print(f"Expectation of Internal Energy <E> at beta={beta_test} = {-energy:.6f}")

    print("\n--- Benchmarking JIT and Execution Time ---")
    # 2. Performance comparison across betas (JIT Staging vs Running)
    ising_jit = jax.jit(ising_partition_function, static_argnums=(0,))
    ising_peps_jit = jax.jit(ising_partition_function_peps, static_argnums=(0,))

    # 2a. Hyperedge (CopyNode) Method
    print("\n[Hyperedge / CopyNode Method]")
    for i, beta in enumerate(betas):
        start = time.time()
        _ = ising_jit(L, beta, J).block_until_ready()
        elapsed = time.time() - start
        if i == 0:
            print(f"  beta={beta:.2f} | Time: {elapsed:.4f}s (JIT staging + execution)")
        else:
            print(f"  beta={beta:.2f} | Time: {elapsed:.4f}s (JIT cached execution)")

    # 2b. Explicit PEPS Method
    print("\n[Explicit PEPS Method]")
    for i, beta in enumerate(betas):
        start = time.time()
        _ = ising_peps_jit(L, beta, J).block_until_ready()
        elapsed = time.time() - start
        if i == 0:
            print(f"  beta={beta:.2f} | Time: {elapsed:.4f}s (JIT staging + execution)")
        else:
            print(f"  beta={beta:.2f} | Time: {elapsed:.4f}s (JIT cached execution)")


if __name__ == "__main__":
    main()
