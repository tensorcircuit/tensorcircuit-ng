"""
Physics-relevant demonstration of hyperedge support in TensorCircuit.
Computing the partition function of a 2D classical Ising model using CopyNodes.
"""

import time
import jax
import jax.numpy as jnp
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


def main():
    L = 8
    J = 1.0
    beta = 0.4  # Near critical point beta_c approx 0.44 for 2D Ising
    print(f"--- 2D Ising Model Partition Function ({L}x{L} lattice) ---")
    print(f"Backend: {tc.backend.name}, Parameters: J={J}, beta={beta}")

    # 1. Direct computation
    z = ising_partition_function(L, beta, J)
    print(f"Z({beta}) = {z:.6f}")

    # 2. JIT-compiled version
    # JIT significantly accelerates repeated calls with the same topology
    print("\nDemonstrating JIT acceleration...")
    ising_jit = jax.jit(ising_partition_function, static_argnums=(0,))

    start = time.time()
    _ = ising_jit(L, beta, J)
    print(f"First run (with JIT warmup): {time.time() - start:.4f}s")

    start = time.time()
    _ = ising_jit(L, beta, J)
    print(f"Second run (JIT cached):     {time.time() - start:.4f}s")

    # 3. Automatic Differentiation (AD)
    # Internal Energy U = - d(ln Z) / d(beta)
    print("\nComputing Internal Energy via Automatic Differentiation...")

    def log_z(beta_val):
        # We take the real part as the partition function is real
        val = ising_partition_function(L, beta_val, J)
        return jnp.log(tc.backend.real(val))

    energy_fn = jax.grad(log_z)
    energy = energy_fn(beta)

    # U = -d(ln Z)/d(beta) in our convention
    print(f"Expectation of Internal Energy <E> = {-energy:.6f}")

    # 4. Scaling demonstration
    L_larger = 12
    print(f"\nScaling check: L={L_larger} ({L_larger*L_larger} spins)...")
    start = time.time()
    z_large = ising_jit(L_larger, beta, J)
    print(
        f"Result for L={L_larger}: {z_large:.2e} (computed in {time.time()-start:.4f}s)"
    )


if __name__ == "__main__":
    main()
