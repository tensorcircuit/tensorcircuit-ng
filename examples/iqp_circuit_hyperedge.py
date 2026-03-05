import time
import argparse
import functools
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_contractor("cotengra")


def simulate_iqp(diag, n, depth, use_hyperedge=True):
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)

    for _ in range(depth):
        if use_hyperedge:
            c.diagonal(*range(n), diag=diag)
        else:
            diag_matrix = tc.backend.diagflat(diag)
            c.any(*range(n), unitary=diag_matrix)
        for i in range(n):
            c.h(i)

    # Calculate amplitude of |0...0>
    amp = c.amplitude("0" * n)
    return amp


# JIT compile the simulation functions
@functools.partial(tc.backend.jit, static_argnums=(1, 2))
def simulate_iqp_hyperedge(diag, n, depth):
    return simulate_iqp(diag, n, depth, use_hyperedge=True)


@functools.partial(tc.backend.jit, static_argnums=(1, 2))
def simulate_iqp_dense(diag, n, depth):
    return simulate_iqp(diag, n, depth, use_hyperedge=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=3, help="Depth of IQP circuit")
    args = parser.parse_args()

    # We use a random diagonal tensor
    np.random.seed(42)
    diag_np = np.exp(1.0j * np.random.uniform(0, 2 * np.pi, size=(2**args.n,)))
    diag = tc.backend.convert_to_tensor(diag_np)

    # --- Run with Hyperedge ---
    t0 = time.time()
    amp_hyper = simulate_iqp_hyperedge(diag, args.n, args.depth).block_until_ready()
    t1 = time.time()
    print(f"Hyperedge JIT Staging + Initial Run time: {t1 - t0:.4f}s")

    t0 = time.time()
    amp_hyper = simulate_iqp_hyperedge(diag, args.n, args.depth).block_until_ready()
    t1 = time.time()
    print(f"Hyperedge Execution time:                 {t1 - t0:.4f}s")
    print(f"Amplitude: {amp_hyper}\n")

    # --- Run Dense ---
    t0 = time.time()
    amp_dense = simulate_iqp_dense(diag, args.n, args.depth).block_until_ready()
    t1 = time.time()
    print(f"Dense JIT Staging + Initial Run time:     {t1 - t0:.4f}s")

    t0 = time.time()
    amp_dense = simulate_iqp_dense(diag, args.n, args.depth).block_until_ready()
    t1 = time.time()
    print(f"Dense Execution time:                     {t1 - t0:.4f}s")
    print(f"Amplitude: {amp_dense}\n")

    np.testing.assert_allclose(amp_hyper, amp_dense, atol=1e-5)
    print("Results match!")
