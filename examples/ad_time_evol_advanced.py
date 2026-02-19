"""
Advanced AD and JIT time evolution benchmark script.

This script benchmarks the correctness, stability, and efficiency of combining
AD+JIT with krylov_evol and chebyshev_evol, comparing against exact diagonalization.
"""

import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc

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
    hxx = tc.quantum.heisenberg_hamiltonian(
        g, hxx=1.0, hyy=0, hzz=0, hx=0, hy=0, hz=0, sparse=True
    )
    hyy = tc.quantum.heisenberg_hamiltonian(
        g, hxx=0, hyy=1.0, hzz=0, hx=0, hy=0, hz=0, sparse=True
    )
    hzz = tc.quantum.heisenberg_hamiltonian(
        g, hxx=0, hyy=0, hzz=1.0, hx=0, hy=0, hz=0, sparse=True
    )
    hz = tc.quantum.heisenberg_hamiltonian(
        g, hxx=0, hyy=0, hzz=0, hx=0, hy=0, hz=1.0, sparse=True
    )
    hx = tc.quantum.heisenberg_hamiltonian(
        g, hxx=0, hyy=0, hzz=0, hx=1.0, hy=0, hz=0, sparse=True
    )

    return hxx, hyy, hzz, hz, hx


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


def benchmark_evolution(n=10, t_list=None, methods=None):
    """
    Benchmarks time evolution methods.
    """
    if t_list is None:
        t_list = [1.0, 5.0, 20.0]
    if methods is None:
        methods = ["ed", "krylov", "chebyshev"]
    print(f"Benchmarking Time Evolution (n={n})")
    print("-" * 60)

    print("Initial state: |+> product state")
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    psi0 = c.state()

    # Pre-calculate Hamiltonian terms
    # We will use H = J(XX+YY+ZZ) + hZ + hX (to break conservation)
    hxx_term, hyy_term, hzz_term, hz_term, hx_term = get_hamiltonian_terms(n)

    # Parameters to differentiate
    j_coupling = 1.0
    h_field_z = 0.5
    h_field_x = 0.3  # Non-zero to break symmetry

    # Function to construct H from params
    def construct_h(j_c, h_z, h_x):
        h = j_c * (hxx_term + hyy_term + hzz_term) + h_z * hz_term + h_x * hx_term
        return h

    # Pre-calculate spectral bounds using initial parameters (outside JIT)
    h_init = construct_h(j_coupling, h_field_z, h_field_x)
    e_max, e_min = tc.timeevol.estimate_spectral_bounds(h_init)

    # Buffer for bounds
    e_max_val = float(e_max) + 5.0
    e_min_val = float(e_min) - 5.0
    print(f"Spectral bounds: [{e_min_val}, {e_max_val}]")

    # Define Loss Functions
    @jax.jit
    @jax.value_and_grad
    def loss_ed(params, t):
        j_c, h_z, h_x = params
        h = construct_h(j_c, h_z, h_x)
        h_dense = tc.backend.to_dense(h)
        psi_t = tc.timeevol.ed_evol(h_dense, psi0, [1j * t])[-1]
        return measure_magnetization(psi_t)

    k_dim = min(2**n, 50)  # Increased Krylov dimension

    @jax.jit
    @jax.value_and_grad
    def loss_krylov(params, t):
        j_c, h_z, h_x = params
        h = construct_h(j_c, h_z, h_x)
        psi_t = tc.timeevol.krylov_evol(
            h, psi0, [t], subspace_dimension=k_dim, scan_impl=True
        )[-1]
        return measure_magnetization(psi_t)

    # Chebyshev needs static k and M, so we need a factory or partial
    def make_chebyshev_loss(k, M):
        @jax.jit
        @jax.value_and_grad
        def loss_chebyshev(
            params, t
        ):  # t is passed but assumed static-like for k/M validness
            j_c, h_z, h_x = params
            h = construct_h(j_c, h_z, h_x)
            psi_t = tc.timeevol.chebyshev_evol(
                h, psi0, t, spectral_bounds=(e_max_val, e_min_val), k=k, M=M
            )
            psi_t = psi_t / jnp.linalg.norm(psi_t)
            return measure_magnetization(psi_t)

        return loss_chebyshev

    params = jnp.array([j_coupling, h_field_z, h_field_x])

    for t in t_list:
        print(f"\nTime t={t}")
        print("=" * 20)

        # Estimate Chebyshev params for this t
        k_cheb = tc.timeevol.estimate_k(t, (e_max_val, e_min_val))
        M_cheb = tc.timeevol.estimate_M(t, (e_max_val, e_min_val), k_cheb)
        print(f"Chebyshev params: k={k_cheb}, M={M_cheb}")

        loss_chebyshev = make_chebyshev_loss(k_cheb, M_cheb)

        results = {}

        for method in methods:
            print(f"  Testing {method}...")
            try:
                if method == "ed":
                    loss_fn = loss_ed
                elif method == "krylov":
                    loss_fn = loss_krylov
                elif method == "chebyshev":
                    loss_fn = loss_chebyshev
                else:
                    continue

                # JIT Compilation (Staging) Time
                start_compile = time.time()
                # Trigger compilation
                val, grad = loss_fn(params, t)
                val.block_until_ready()
                end_compile = time.time()
                compile_time = end_compile - start_compile

                # Execution Time
                times = []
                for _ in range(10):
                    start = time.time()
                    val, grad = loss_fn(params, t)
                    val.block_until_ready()
                    end = time.time()
                    times.append(end - start)

                avg_time = np.mean(times)
                std_time = np.std(times)

                print(f"    Compile time: {compile_time:.4f}s")
                print(f"    Run time:     {avg_time:.4f}s (+/- {std_time:.4f})")
                print(f"    Value:        {val:.6f}")
                print(f"    Grad norm:    {jnp.linalg.norm(grad):.6f}")

                results[method] = {"val": val, "grad": grad}

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback

                traceback.print_exc()

        # Comparisons
        if "ed" in results:
            base_val = results["ed"]["val"]
            base_grad = results["ed"]["grad"]

            for method in results:
                if method == "ed":
                    continue

                val = results[method]["val"]
                grad = results[method]["grad"]

                val_diff = abs(val - base_val)
                grad_diff = jnp.linalg.norm(grad - base_grad)

                print(f"  Comparison {method} vs ED:")
                print(f"    Value Diff: {val_diff:.2e}")
                print(f"    Grad Diff:  {grad_diff:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of qubits")
    args = parser.parse_args()

    benchmark_evolution(n=args.n)


if __name__ == "__main__":
    main()
