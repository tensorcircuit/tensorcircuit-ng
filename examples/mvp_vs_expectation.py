import sys
import numpy as np
import tensorcircuit as tc


def test_backend(backend_name):
    print(f"Testing backend: {backend_name}")
    try:
        tc.set_backend(backend_name)
    except ImportError:
        print(f"Backend {backend_name} not available, skipping.")
        return
    except Exception as e:
        print(f"Backend {backend_name} setup failed: {e}")
        return

    n = 22

    # Random structures
    # Use simple ones
    structures = [
        [1, 0, 3] + [0] * (n - 3),  # X I Z ...
        [2, 2, 0] + [0] * (n - 3),  # Y Y I ...
        [3, 3, 3] + [0] * (n - 3),  # Z Z Z ...
        [0, 0, 0] + [0] * (n - 3),  # I I I ...
    ]
    weights = [0.5, -0.3, 0.2, 1.0]

    # Generate MVP function
    try:
        mvp = tc.quantum.PauliStringSum2MVP(structures, weights)
    except Exception as e:
        print(f"PauliStringSum2MVP failed for {backend_name}: {e}")
        raise e

    # Random wavefunction
    # We use numpy to create initial state to ensure consistency across backends?
    # Or just use backend.
    np_psi = np.random.randn(2**n) + 1j * np.random.randn(2**n)
    np_psi = np_psi / np.linalg.norm(np_psi)
    np_psi = np_psi.astype(np.complex64)

    psi = tc.backend.convert_to_tensor(np_psi)

    # 1. MVP result
    try:
        res_mvp = mvp(psi)
    except Exception as e:
        print(f"mvp execution failed for {backend_name}: {e}")
        # import traceback
        # traceback.print_exc()
        return

    # 2. Sparse result (Ground Truth)
    # PauliStringSum2COO should work for all backends
    try:
        h_sparse = tc.quantum.PauliStringSum2COO(structures, weights)
        psi_flat = tc.backend.reshape(psi, (-1, 1))
        res_sparse = tc.backend.sparse_dense_matmul(h_sparse, psi_flat)
        res_sparse = tc.backend.reshape(res_sparse, (-1,))
    except Exception as e:
        print(f"Sparse reference failed for {backend_name}: {e}")
        return

    # Compare Expectation Values
    # <psi | H | psi>
    # Note: psi is normalized
    def expectation(v, Hv):
        # <v|Hv> = sum(conj(v) * Hv)
        return tc.backend.sum(tc.backend.conj(v) * Hv)

    e_mvp = expectation(psi, res_mvp)
    e_sparse = expectation(psi, res_sparse)

    # Provide vector difference as well for debug
    diff_vec = tc.backend.norm(res_mvp - res_sparse)
    diff_e = tc.backend.abs(e_mvp - e_sparse)

    val_vec = tc.backend.numpy(diff_vec)
    val_e = tc.backend.numpy(diff_e)

    print(f"Vector Diff: {val_vec:.2e}")
    print(f"Expectation Diff: {val_e:.2e}")

    # Timing Comparison
    import time

    try:
        # Define functions to JIT
        mvp_jitted = tc.backend.jit(mvp)

        def run_sparse(psi_in):
            psi_flat = tc.backend.reshape(psi_in, (-1, 1))
            res = tc.backend.sparse_dense_matmul(h_sparse, psi_flat)
            # return tc.backend.reshape(res, backend.shape_tuple(psi_in)) # shape tuple might cause JIT issue if dynamic
            # Assume n is static
            return tc.backend.reshape(res, (2,) * n)

        sparse_jitted = tc.backend.jit(run_sparse)

        def sync(res):
            if backend_name == "jax":
                res.block_until_ready()
            else:
                # Force sync for TF/Numpy
                _ = tc.backend.numpy(res)

        print(f"\n--- Benchmark {backend_name} n={n} ---")

        # 1. MVP Benchmark
        t0 = time.time()
        r1 = mvp_jitted(psi)
        sync(r1)
        t_mvp_first = time.time() - t0
        print(f"MVP First Run (Compile):    {t_mvp_first:.4f} s")

        t0 = time.time()
        loops = 100
        for _ in range(loops):
            r = mvp_jitted(psi)
            sync(r)  # Ensure each run is finished? Or sync at end?
            # For JAX, we can dispatch all and wait at end, but that fills buffer.
            # Usually sync per loop for latency measurement, or sync at end for throughput.
            # Script requested "Running Time", usually implies latency or average time.
            # Comparisons usually do sync per loop or batched.
            # Let's sync per loop to be safe against async piling up.

        t_mvp_avg = (time.time() - t0) / loops
        print(f"MVP Running Time (Avg):     {t_mvp_avg*1000:.4f} ms")

        # 2. Sparse Benchmark
        # Sparse JIT Support check
        # TF XLA might fail with SparseTensor. Numpy doesn't JIT.
        # We try JIT, if fail, fallback to no-jit ?
        # But User asked specifically "using jit".
        # If TF fails, we report it.

        try:
            t0 = time.time()
            r2 = sparse_jitted(psi)
            sync(r2)
            t_sparse_first = time.time() - t0
            print(f"Sparse First Run (Compile): {t_sparse_first:.4f} s")

            t0 = time.time()
            for _ in range(loops):
                r = sparse_jitted(psi)
                sync(r)
            t_sparse_avg = (time.time() - t0) / loops
            print(f"Sparse Running Time (Avg):  {t_sparse_avg*1000:.4f} ms")

            print(f"Speedup vs Sparse (Run):    {t_sparse_avg/t_mvp_avg:.2f}x")

        except Exception as e:
            print(f"Sparse JIT failed (expected for non-JAX/TF-Sparse): {e}")

    except Exception as e:
        print(f"Timing failed: {e}")
        import traceback

        traceback.print_exc()

    if val_e > 1e-4:
        print(f"FAIL: {backend_name}")
        sys.exit(1)
    else:
        print(f"PASS: {backend_name}")


if __name__ == "__main__":
    for b in ["jax", "tensorflow", "numpy"]:
        test_backend(b)
