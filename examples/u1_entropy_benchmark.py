"""
Benchmark for U(1) conserving circuit RDM and entropy efficiency.
"""

import time
import numpy as np
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex128")


def benchmark_u1_efficiency(n: int, k: int, cut: int):
    print(f"Benchmarking U1Circuit: n={n}, k={k}, cut={cut}")

    # 1. Initialize U1Circuit
    # Start with all k qubits filled at the beginning
    filled = list(range(k))
    c = tc.U1Circuit(n, k=k, filled=filled)

    # Apply multiple layers of U(1) conserving gates to create deep entanglement
    for _ in range(3):
        for i in range(n - 1):
            c.iswap(i, i + 1, theta=0.5)
        for i in range(0, n - 1, 2):
            c.rzz(i, i + 1, theta=0.3)
        for i in range(1, n - 1, 2):
            c.rzz(i, i + 1, theta=0.4)

    subsystem = list(range(cut))

    # 2. Benchmark Entanglement Entropy
    start = time.time()
    # We wrap in jit for fair benchmarking if using jax/tf
    # subsystem_to_keep must be static for JAX due to Python logic in U1Circuit
    ee_func = K.jit(c.entanglement_entropy, static_argnums=0)
    # Warmup
    _ = ee_func(tuple(subsystem))

    start = time.time()
    entropy = ee_func(tuple(subsystem))
    if hasattr(entropy, "block_until_ready"):
        entropy.block_until_ready()
    end = time.time()
    print(f"Entanglement Entropy: {entropy:.6f}")
    print(f"Time (JITed): {end - start:.6f} s")

    # 3. Benchmark Reduced Density Matrix (Blocks)
    rdm_func = K.jit(
        lambda s: c.reduced_density_matrix(s, return_blocks=True), static_argnums=0
    )
    # Warmup
    _ = rdm_func(tuple(subsystem))

    start = time.time()
    blocks = rdm_func(tuple(subsystem))
    if isinstance(blocks, list):
        for b in blocks:
            if hasattr(b, "block_until_ready"):
                b.block_until_ready()
    elif hasattr(blocks, "block_until_ready"):
        blocks.block_until_ready()
    end = time.time()
    print(f"Number of RDM blocks: {len(blocks)}")
    for i, b in enumerate(blocks):
        print(f"  Block {i} shape: {b.shape}")
    print(f"Time (JITed): {end - start:.6f} s")

    # 4. Compare with Dense (only if n is small)
    if n <= 20:
        print("\nComparing with dense simulation...")
        dense_state = c.to_dense()
        start = time.time()
        # tc.quantum.reduced_density_matrix expects traced out indices
        traceout = [i for i in range(n) if i not in subsystem]
        rdm_dense = tc.quantum.reduced_density_matrix(dense_state, traceout)
        entropy_dense = tc.quantum.entropy(rdm_dense)
        end = time.time()
        print(f"Dense Entropy: {entropy_dense:.6f}")
        print(f"Dense Time: {end - start:.6f} s")
        np.testing.assert_allclose(entropy, entropy_dense, atol=1e-5)
        print("Verification successful!")


if __name__ == "__main__":
    # Case 1: Small system for verification
    benchmark_u1_efficiency(n=12, k=6, cut=6)

    print("-" * 40)

    # Case 2: Large system (The 60-qubit challenge)
    # n=60, k=4 is dim=487,635, fast in U1 but impossible in dense
    benchmark_u1_efficiency(n=60, k=4, cut=30)
