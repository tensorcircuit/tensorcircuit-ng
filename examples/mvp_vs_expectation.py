import time
import numpy as np
import tensorcircuit as tc


def test_backend(backend_name):
    print(f"Testing backend: {backend_name}")
    tc.set_backend(backend_name)

    n = 22

    # 1D Heisenberg Hamiltonian (OBC)
    structures = []
    weights = []
    for i in range(n - 1):
        for op in [1, 2, 3]:  # X, Y, Z
            s = [0] * n
            s[i] = op
            s[i + 1] = op
            structures.append(s)
            weights.append(1.0)

    # Generate Hamiltonian operators
    mvp = tc.quantum.PauliStringSum2MVP(structures, weights)
    h_sparse = tc.quantum.PauliStringSum2COO(structures, weights)

    # Random wavefunction
    np_psi = np.random.randn(2**n) + 1j * np.random.randn(2**n)
    np_psi = np_psi / np.linalg.norm(np_psi)
    np_psi = np_psi.astype(np.complex64)

    psi = tc.backend.convert_to_tensor(np_psi)

    def exp_mvp(v):
        Hv = mvp(v)
        return tc.backend.real(tc.backend.sum(tc.backend.conj(v) * Hv))

    def exp_sparse(v):
        c = tc.Circuit(n, inputs=v)
        return tc.templates.measurements.operator_expectation(c, h_sparse)

    exp_mvp_jitted = tc.backend.jit(exp_mvp)
    exp_sparse_jitted = tc.backend.jit(exp_sparse)

    def sync(res):
        if backend_name == "jax":
            res.block_until_ready()
        else:
            _ = tc.backend.numpy(res)

    print(f"\n--- Benchmark {backend_name} n={n} ---")

    # 1. MVP Expectation Benchmark
    t0 = time.time()
    e_mvp = exp_mvp_jitted(psi)
    sync(e_mvp)
    t_mvp_first = time.time() - t0
    print(f"MVP Expectation First Run (Compile):    {t_mvp_first:.4f} s")

    loops = 10
    t0 = time.time()
    for _ in range(loops):
        r = exp_mvp_jitted(psi)
        sync(r)

    t_mvp_avg = (time.time() - t0) / loops
    print(f"MVP Expectation Running Time (Avg):     {t_mvp_avg*1000:.4f} ms")

    # 2. Sparse Expectation Benchmark
    t0 = time.time()
    e_sparse = exp_sparse_jitted(psi)
    sync(e_sparse)
    t_sparse_first = time.time() - t0
    print(f"Sparse Expectation First Run (Compile): {t_sparse_first:.4f} s")

    t0 = time.time()
    for _ in range(loops):
        r = exp_sparse_jitted(psi)
        sync(r)
    t_sparse_avg = (time.time() - t0) / loops
    print(f"Sparse Expectation Running Time (Avg):  {t_sparse_avg*1000:.4f} ms")

    print(f"Speedup vs Sparse (Run):    {t_sparse_avg/t_mvp_avg:.2f}x")

    val_e_mvp = tc.backend.numpy(e_mvp)
    val_e_sparse = tc.backend.numpy(e_sparse)
    diff_e = np.abs(val_e_mvp - val_e_sparse)
    print(f"Expectation Diff: {diff_e:.2e}")

    np.testing.assert_allclose(val_e_mvp, val_e_sparse, atol=1e-4)
    print(f"PASS: {backend_name}")


if __name__ == "__main__":
    for b in ["jax", "tensorflow", "numpy"]:
        test_backend(b)
