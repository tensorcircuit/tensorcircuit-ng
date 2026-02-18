"""
Advanced AD and JIT time evolution benchmark script.

This script benchmarks the correctness, stability, and efficiency of combining
AD+JIT with krylov_evol and chebyshev_evol, comparing against exact diagonalization.
"""

import time
import argparse
import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp

# Set backend to JAX
tc.set_backend("jax")
tc.set_dtype("complex128")

def get_hamiltonian_terms(n):
    """
    Creates base Hamiltonian terms for a 1D Heisenberg chain.
    """
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # We construct terms separately so that we can linearly combine them with JAX tracers
    # sparse=True returns BCOO matrix in JAX
    hxx = tc.quantum.heisenberg_hamiltonian(g, hxx=1.0, hyy=0, hzz=0, hx=0, hy=0, hz=0, sparse=True)
    hyy = tc.quantum.heisenberg_hamiltonian(g, hxx=0, hyy=1.0, hzz=0, hx=0, hy=0, hz=0, sparse=True)
    hzz = tc.quantum.heisenberg_hamiltonian(g, hxx=0, hyy=0, hzz=1.0, hx=0, hy=0, hz=0, sparse=True)
    hz = tc.quantum.heisenberg_hamiltonian(g, hxx=0, hyy=0, hzz=0, hx=0, hy=0, hz=1.0, sparse=True)

    return hxx, hyy, hzz, hz

def measure_magnetization(state):
    """
    Measures the average magnetization Z of the state.
    """
    n = int(np.log2(state.shape[0]))
    c = tc.Circuit(n, inputs=state)
    mag = 0.0
    for i in range(n):
        mag += c.expectation_ps(z=[i])
    return jnp.real(mag) / n

def benchmark_evolution(n=6, t=1.0, steps=10, methods=["ed", "krylov", "chebyshev"]):
    """
    Benchmarks time evolution methods.
    """
    print(f"Benchmarking Time Evolution (n={n}, t={t})")
    print("-" * 60)

    # Initial state: product state |+>|+>...|+>
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    psi0 = c.state()

    # Pre-calculate Hamiltonian terms
    hxx_term, hyy_term, hzz_term, hz_term = get_hamiltonian_terms(n)

    # Parameters to differentiate
    j_coupling = 1.0
    h_field = 0.5

    # Function to construct H from params
    def construct_h(j_c, h_f):
        # We assume j_c couples XX, YY, ZZ equally
        # BCOO matrices addition
        h = j_c * (hxx_term + hyy_term + hzz_term) + h_f * hz_term
        return h

    # Pre-calculate spectral bounds using initial parameters (outside JIT)
    h_init = construct_h(j_coupling, h_field)
    e_max, e_min = tc.timeevol.estimate_spectral_bounds(h_init)

    # Estimate k and M using float bounds
    # Add buffer to ensure stability if params change slightly during optimization (if we were optimizing)
    # For benchmarking gradients at fixed point, these bounds are fine.
    # If parameters change significantly, bounds should be re-estimated, but that requires non-JIT call.
    # In practice, for AD, we might use loose bounds or recompute periodically.
    e_max_val = float(e_max) + 5.0
    e_min_val = float(e_min) - 5.0

    k_cheb = tc.timeevol.estimate_k(t, (e_max_val, e_min_val))
    M_cheb = tc.timeevol.estimate_M(t, (e_max_val, e_min_val), k_cheb)
    print(f"Chebyshev params: k={k_cheb}, M={M_cheb}")

    # 1. Exact Diagonalization (Real Time)
    @jax.jit
    @jax.value_and_grad
    def loss_ed(params):
        j_c, h_f = params
        h = construct_h(j_c, h_f)
        h_dense = tc.backend.to_dense(h)
        # ed_evol defaults to imaginary time. Pass 1j*t for real time.
        psi_t = tc.timeevol.ed_evol(h_dense, psi0, [1j * t])[-1]
        return measure_magnetization(psi_t)

    # 2. Krylov (Scan implementation)
    k_dim = min(2**n, 30)

    @jax.jit
    @jax.value_and_grad
    def loss_krylov(params):
        j_c, h_f = params
        h = construct_h(j_c, h_f)
        # krylov_evol with scan_impl=True
        psi_t = tc.timeevol.krylov_evol(h, psi0, [t], subspace_dimension=k_dim, scan_impl=True)[-1]
        return measure_magnetization(psi_t)

    # 3. Chebyshev with precomputed bounds
    @jax.jit
    @jax.value_and_grad
    def loss_chebyshev(params):
        j_c, h_f = params
        h = construct_h(j_c, h_f)

        # Use precomputed bounds (captured as constants)
        psi_t = tc.timeevol.chebyshev_evol(
            h, psi0, t, spectral_bounds=(e_max_val, e_min_val), k=k_cheb, M=M_cheb
        )
        # Normalize
        psi_t = psi_t / jnp.linalg.norm(psi_t)
        return measure_magnetization(psi_t)

    params = jnp.array([j_coupling, h_field])

    results = {}

    # Run benchmarks
    for method in methods:
        print(f"\nTesting {method}...")
        try:
            if method == "ed":
                loss_fn = loss_ed
            elif method == "krylov":
                loss_fn = loss_krylov
            elif method == "chebyshev":
                loss_fn = loss_chebyshev
            else:
                continue

            # Warmup
            start = time.time()
            val, grad = loss_fn(params)
            val.block_until_ready()
            end = time.time()
            print(f"  Warmup time: {end - start:.4f}s")
            print(f"  Value: {val:.6f}")
            print(f"  Grad: {grad}")

            # Run multiple times
            times = []
            for _ in range(10):
                start = time.time()
                val, grad = loss_fn(params)
                val.block_until_ready()
                end = time.time()
                times.append(end - start)

            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"  Avg time: {avg_time:.4f}s (+/- {std_time:.4f})")

            results[method] = {"val": val, "grad": grad, "time": avg_time}

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Compare
    if "ed" in results:
        base_val = results["ed"]["val"]
        base_grad = results["ed"]["grad"]

        for method in results:
            if method == "ed": continue

            val = results[method]["val"]
            grad = results[method]["grad"]

            val_diff = abs(val - base_val)
            grad_diff = jnp.linalg.norm(grad - base_grad)

            print(f"\nComparison {method} vs ED:")
            print(f"  Value Diff: {val_diff:.2e}")
            print(f"  Grad Diff: {grad_diff:.2e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Number of qubits")
    parser.add_argument("--t", type=float, default=1.0, help="Total time")
    args = parser.parse_args()

    benchmark_evolution(n=args.n, t=args.t)

if __name__ == "__main__":
    main()
