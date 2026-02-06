"""
VQE using mvp method to evaluate Hamiltonian expectation
"""

import time
import jax
import jax.numpy as jnp
import tensorcircuit as tc

# Configuration
tc.set_backend("jax")
tc.set_dtype("complex128")

n = 18  # Qubit number
nlayers = 4

# TFIM Hamiltonian parameters
j, h = 1.0, -1.0


def ansatz(param, n, nlayers):
    c = tc.Circuit(n)
    for i in range(nlayers):
        for j in range(n):
            c.rx(j, theta=param[i, j, 0])
            c.rz(j, theta=param[i, j, 1])
        for j in range(n - 1):
            c.cnot(j, j + 1)
    return c


# 1. Prepare Hamiltonian structures and weights
structures = []
weights = []

# Transverse field: sum h * Z_i
for i in range(n):
    s = [0] * n
    s[i] = 3  # Z
    structures.append(s)
    weights.append(h)

# Ising interaction: sum j * X_i X_{i+1}
for i in range(n - 1):
    s = [0] * n
    s[i] = 1  # X
    s[i + 1] = 1  # X
    structures.append(s)
    weights.append(j)

# Prepare Sparse Matrix for operator_expectation
hamiltonian_sparse = tc.quantum.PauliStringSum2COO(structures, weights)

# Generate MVP function
mvp_func = tc.quantum.PauliStringSum2MVP(structures, weights)

# 2. Define Loss Functions


def loss_sparse(param):
    c = ansatz(param, n, nlayers)
    return tc.templates.measurements.operator_expectation(c, hamiltonian_sparse)


def loss_mvp(param):
    c = ansatz(param, n, nlayers)
    psi = c.state()
    h_psi = mvp_func(psi)
    # <psi|H|psi>
    return jnp.real(jnp.vdot(psi, h_psi))


# 3. Benchmarking function
def benchmark(loss_fn, param, name):
    # JIT the value and grad
    vag_fn = jax.jit(jax.value_and_grad(loss_fn))

    print(f"\nBenchmarking {name}...")

    # Warmup / Compilation
    t0 = time.time()
    v, g = vag_fn(param)
    jax.block_until_ready(v)
    t_compile = time.time() - t0
    print(f"Compile time: {t_compile:.4f} s")

    # Running time
    t0 = time.time()
    iterations = 50
    for _ in range(iterations):
        v, g = vag_fn(param)
        jax.block_until_ready(v)
    t_run = (time.time() - t0) / iterations
    print(f"Running time (avg of {iterations} iterations): {t_run*1000:.4f} ms")

    return v, g, t_run


if __name__ == "__main__":
    param = jax.random.normal(jax.random.PRNGKey(42), shape=(nlayers, n, 2))

    # Compare correctness and performance
    res_sparse, grad_sparse, t_sparse = benchmark(
        loss_sparse, param, "Sparse Matrix (COO)"
    )
    res_mvp, grad_mvp, t_mvp = benchmark(loss_mvp, param, "Matrix-Free MVP")

    print("\n--- Results Comparison ---")
    print(f"Energy (Sparse): {res_sparse:.8f}")
    print(f"Energy (MVP):    {res_mvp:.8f}")

    diff_val = jnp.abs(res_sparse - res_mvp)
    diff_grad = jnp.linalg.norm(grad_sparse - grad_mvp)

    print(f"Energy Diff:     {diff_val:.2e}")
    print(f"Gradient Norm Diff: {diff_grad:.2e}")

    speedup = t_sparse / t_mvp
    print(f"Speedup: {speedup:.2f}x")

    if diff_val < 1e-5 and diff_grad < 1e-4:
        print("\nSUCCESS: Results match between methods.")
    else:
        print("\nFAILURE: Significant discrepancy detected.")
