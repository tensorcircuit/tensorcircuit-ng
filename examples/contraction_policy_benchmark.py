"""
Benchmark JAX JIT staging and execution time for VQE for two different contraction policies.
Comparing use_primitives=True (Algebraic Path) vs use_primitives=False (Legacy Path).
"""

import time
import jax
import jax.numpy as jnp
import tensorcircuit as tc

# Set backend to JAX
tc.set_backend("jax")


def run_vqe_benchmark(n, nlayers, use_primitives, preprocessing):
    # Set the contractor configuration
    tc.set_contractor(
        "cotengra", use_primitives=use_primitives, preprocessing=preprocessing
    )

    def energy_fn(params):
        c = tc.Circuit(n)
        idx = 0
        for _ in range(nlayers):
            # Single qubit rotations
            for i in range(n):
                c.rx(i, theta=params[idx])
                idx += 1
                c.ry(i, theta=params[idx])
                idx += 1
            # Entangling Rzz gates
            for i in range(n - 1):
                c.rzz(i, i + 1, theta=params[idx])
                idx += 1

        # Compute expectation value of a TFIM Hamiltonian
        e = 0.0
        for i in range(n - 1):
            e += c.expectation_ps(z=[i, i + 1])
        for i in range(n):
            e += c.expectation_ps(x=[i])
        return jnp.real(e)

    # Use value_and_grad for a more realistic VQE benchmark
    val_grad_fn = jax.jit(jax.value_and_grad(energy_fn))

    # Prepare random parameters
    num_params = nlayers * (2 * n + (n - 1))
    params = jax.random.normal(jax.random.PRNGKey(42), (num_params,))

    # 1. Staging Time (Compilation + First Execution)
    start = time.time()
    v, g = val_grad_fn(params)
    v.block_until_ready()
    # Gradient nodes are usually ready when the value is ready if fused,
    # but we can block on jnp.sum(g) to be sure
    jnp.sum(g).block_until_ready()
    staging_time = time.time() - start

    # 2. Execution Time (Subsequent Runs)
    iters = 10
    start = time.time()
    for _ in range(iters):
        v, g = val_grad_fn(params)
        v.block_until_ready()
        jnp.sum(g).block_until_ready()
    exec_time = (time.time() - start) / iters

    return staging_time, exec_time


def main():
    n = 12
    nlayers = 5
    print(f"--- VQE Benchmark: {n} Qubits, {nlayers} Layers ---")
    print(f"Backend: {tc.backend.name}")

    results = []
    for use_primitives in [False, True]:
        for preprocessing in [False, True]:
            print(
                f"\nTesting: use_primitives={use_primitives}, preprocessing={preprocessing}"
            )
            s, e = run_vqe_benchmark(n, nlayers, use_primitives, preprocessing)
            print(f"  Staging Time:   {s:.4f}s")
            print(f"  Execution Time: {e:.6f}s")
            results.append((use_primitives, preprocessing, s, e))

    print("\n" + "=" * 62)
    print(
        f"{'Primitives':<12} | {'Preprocess':<12} | {'Staging (s)':<12} | {'Exec (s)':<12}"
    )
    print("-" * 62)
    for up, pre, s, e in results:
        print(f"{str(up):<12} | {str(pre):<12} | {s:<12.4f} | {e:<12.6f}")
    print("=" * 62)


if __name__ == "__main__":
    main()

# to me, the diff is not significant
