import time

import numpy as np
import tensorcircuit as tc

# Set backend to JAX for performance and JIT support
tc.set_backend("jax")
# Use cotengra for optimized tensor network contraction
tc.set_contractor("cotengra-120-512")


def grover_iteration(c, n, method="cmz"):
    """
    Perform one Grover iteration.
    Oracle marks the |1...1> state.
    """

    def apply_oracle():
        if method == "cmz":
            # Uses exact MPS decomposition with bond dimension chi=2
            c.cmz(*range(n))
        elif method == "multicontrol":
            # Uses exact MPO decomposition
            c.multicontrol(*range(n), ctrl=[1] * (n - 1), unitary=tc.gates._z_matrix)
        elif method == "dense":
            diag = np.ones(2**n, dtype=np.complex128)
            diag[-1] = -1.0
            diag_matrix = np.diag(diag)
            c.any(*range(n), unitary=tc.backend.convert_to_tensor(diag_matrix))
        elif method == "hyperedge":
            diag = np.ones(2**n, dtype=np.complex128)
            diag[-1] = -1.0
            c.diagonal(*range(n), diag=tc.backend.convert_to_tensor(diag))

    # 1. Apply Oracle
    apply_oracle()

    # 2. Diffuser: 2|s><s| - I = H^n (2|0><0| - I) H^n
    # Note: 2|0><0| - I = -(I - 2|0><0|)
    # I - 2|0><0| = X^n (I - 2|1><1|) X^n
    for i in range(n):
        c.h(i)
    for i in range(n):
        c.x(i)

    apply_oracle()

    for i in range(n):
        c.x(i)
    for i in range(n):
        c.h(i)


def run_grover(n, n_iterations, method="cmz"):
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    for _ in range(n_iterations):
        grover_iteration(c, n, method=method)

    # We measure the amplitude of the marked state |1...1>
    return c.amplitude("1" * n)


# JIT-wrapped version for benchmarking
run_grover_jit = tc.backend.jit(run_grover, static_argnums=(0, 1, 2))

if __name__ == "__main__":
    n = 10
    n_iterations = int(np.pi / 4 * np.sqrt(2**n))
    print(f"--- n = {n} ---")
    print(f"Running Grover's Search for n={n} qubits with {n_iterations} iterations\n")

    methods = ["cmz", "multicontrol", "dense", "hyperedge"]
    results = {}

    for name in methods:
        print(f"--- Method: {name} ---")
        # Warm up / JIT staging
        t0 = time.time()
        res = run_grover_jit(n, n_iterations, name)
        # res.block_until_ready() if using jax.numpy, but amplitude returns a tensor
        # tc.backend.numpy forces completion
        val = tc.backend.numpy(res)
        t1 = time.time()
        print(f"JIT + First Run: {t1 - t0:.4f}s")

        # Second run
        t0 = time.time()
        res = run_grover_jit(n, n_iterations, name)
        val = tc.backend.numpy(res)
        t1 = time.time()
        print(f"Pure Execution:  {t1 - t0:.4f}s")

        prob = np.abs(val) ** 2
        print(f"Success Probability: {prob:.6f}")
        results[name] = val

    # Correctness verification
    ref = results["cmz"]
    for name, val in results.items():
        if name == "cmz":
            continue
        # We check relative closeness, ignoring global phase differences if any
        # (though they should match here as we used the same decomposition logic)
        np.testing.assert_allclose(np.abs(val), np.abs(ref), atol=1e-4)
        # Check actual values (they should be identical or off by a clean sign)
        if not np.allclose(val, ref, atol=1e-4):
            if np.allclose(val, -ref, atol=1e-4):
                print(
                    f"Method {name} and cmz match up to a minus sign (expected due to diffuser convention)."
                )
            else:
                print(
                    f"Warning: Method {name} and cmz have larger discrepancy than expected."
                )
        else:
            print(f"Method {name} and cmz match perfectly.")
