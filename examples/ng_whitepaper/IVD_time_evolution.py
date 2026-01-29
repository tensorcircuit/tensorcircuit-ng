"""
Time Evolution Methods for Spin Models
======================================

This script demonstrates various time evolution methods available in TensorCircuit
for simulating the dynamics of spin models. We showcase:

1. Exact Diagonalization (ED) for small systems (benchmark).
2. Krylov Subspace Method (Lanczos) for intermediate systems.
3. Chebyshev Expansion Method for real-time evolution.
4. ODE-based Evolution for time-dependent Hamiltonians.

The physical system is a 1D Heisenberg chain with an oscillating magnetic field.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from tensorcircuit.timeevol import (
    hamiltonian_evol,
    krylov_evol,
    chebyshev_evol,
    ode_evol_global,
    estimate_spectral_bounds,
    estimate_k,
    estimate_M,
)

# Use JAX backend for best performance with these methods
K = tc.set_backend("jax")
tc.set_dtype("complex128")


def main():
    # -----------------------------------------------------------------------
    # 1. System Setup
    # -----------------------------------------------------------------------
    print("Setting up system...")
    n = 10  # Number of qubits/spins
    J = 1.0  # Exchange interaction
    h_z = 0.5  # Static random field scale

    # Create static Heisenberg Hamiltonian
    # H = sum_{i} J (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) + sum_i h_i Z_i
    np.random.seed(42)
    random_fields = np.random.uniform(-h_z, h_z, n)

    ls = []
    weights = []

    # Interaction terms
    for i in range(n - 1):  # Open boundary condition
        # X X
        s = [0] * n
        s[i] = 1
        s[i + 1] = 1
        ls.append(s)
        weights.append(J)

        # Y Y
        s = [0] * n
        s[i] = 2
        s[i + 1] = 2
        ls.append(s)
        weights.append(J)

        # Z Z
        s = [0] * n
        s[i] = 3
        s[i + 1] = 3
        ls.append(s)
        weights.append(J)

    # Field terms
    for i in range(n):
        s = [0] * n
        s[i] = 3  # Z
        ls.append(s)
        weights.append(random_fields[i])

    h_static_sparse = tc.quantum.PauliStringSum2COO(ls, weights)

    # Dense version for ED
    h_static_dense = tc.backend.to_dense(h_static_sparse)

    # Initial State: Neel state |010101...>
    print(f"System size: {n} qubits")
    c = tc.Circuit(n)
    for i in range(1, n, 2):
        c.x(i)
    psi0 = c.state()
    # psi0 = tc.backend.cast(psi0, "complex128") # Ensure high precision

    # Time Evolution parameters
    t_max = 5.0
    steps = 50
    times = np.linspace(0, t_max, steps)
    times_tc = tc.backend.convert_to_tensor(times)
    # times_tc = tc.backend.cast(times_tc, "complex128")

    # Observable: Magnetization of site 0 <Z_0>
    @K.jit
    def measure_z0(state):
        c_m = tc.Circuit(n, inputs=state)
        return tc.backend.real(c_m.expectation_ps(z=[0]))

    # -----------------------------------------------------------------------
    # 2. Time-Independent Evolution Comparison
    # -----------------------------------------------------------------------
    print("\n--- Time-Independent Evolution ---")

    # Method A: Exact Diagonalization (Hamiltonian Evol)
    # Note: hamiltonian_evol computes exp(-tH). For real time exp(-iHt), pass 1j*t
    print("Running Exact Diagonalization (ED)...")
    start = time.time()
    # 1j * times for real time evolution
    ed_obs = hamiltonian_evol(h_static_dense, psi0, 1j * times_tc, callback=measure_z0)
    ed_time = time.time() - start
    print(f"ED finished in {ed_time:.4f}s")

    # Method B: Krylov Subspace (Lanczos)
    print("Running Krylov/Lanczos...")
    start = time.time()
    # krylov_evol computes exp(-iHt), so just pass times
    krylov_obs = krylov_evol(
        h_static_sparse,
        psi0,
        times_tc,
        subspace_dimension=60,
        scan_impl=True,
        callback=measure_z0,
    )
    krylov_time = time.time() - start
    print(f"Krylov finished in {krylov_time:.4f}s")

    # Method C: Chebyshev Expansion
    print("Running Chebyshev...")
    start = time.time()
    # Need spectral bounds
    e_max, e_min = estimate_spectral_bounds(h_static_sparse)
    # Estimate k and M
    k = estimate_k(t_max, (e_max, e_min))
    M = estimate_M(t_max, (e_max, e_min), k)
    print(f"Estimated bounds: [{e_min:.2f}, {e_max:.2f}], k={k}, M={M}")

    # JIT one step
    @tc.backend.jit
    def run_chebyshev_step(t):
        s = chebyshev_evol(
            h_static_sparse,
            psi0,
            t,
            (float(e_max) + 0.1, float(e_min) - 0.1),  # slight padding
            k,
            M,
        )
        return measure_z0(s)

    # Explicit python loop as requested
    chebyshev_obs = []
    for t in times:
        # t is float here from numpy array
        res = run_chebyshev_step(t)
        chebyshev_obs.append(res)

    chebyshev_time = time.time() - start
    print(f"Chebyshev finished in {chebyshev_time:.4f}s")

    # -----------------------------------------------------------------------
    # 3. Time-Dependent Evolution
    # -----------------------------------------------------------------------
    print("\n--- Time-Dependent Evolution (ODE) ---")

    # Define time-dependent Hamiltonian: Static H + Driving field on qubit 0
    # H(t) = H_static + A * cos(omega * t) * X_0
    A = 1.0
    omega = 2.0

    x0_op = tc.quantum.PauliStringSum2COO([[1] + [0] * (n - 1)], [1.0])  # X on site 0

    def h_time_dep(t, a, w):
        # t is a tensor
        # return sparse matrix
        factor = a * tc.backend.cos(w * t)
        return h_static_sparse + factor * x0_op

    print("Running ODE Global Evolution (Driving)...")
    start = time.time()
    # ode_evol_global takes real times
    ode_obs = ode_evol_global(
        h_time_dep,
        psi0,
        tc.backend.real(times_tc),
        measure_z0,  # callback
        A,  # *args element 1
        omega,  # *args element 2
        ode_backend="jaxode",
        rtol=1e-6,
        atol=1e-6,
    )
    ode_time = time.time() - start
    print(f"ODE (Driven) finished in {ode_time:.4f}s")

    # Compare with Static ODE
    print("Running ODE Global Evolution (Static)...")

    def h_static_func(t):
        return h_static_sparse

    start = time.time()
    ode_static_obs = ode_evol_global(
        h_static_func,
        psi0,
        tc.backend.real(times_tc),
        measure_z0,
        ode_backend="jaxode",
        rtol=1e-6,
        atol=1e-6,
    )
    ode_static_time = time.time() - start
    print(f"ODE (Static) finished in {ode_static_time:.4f}s")

    # -----------------------------------------------------------------------
    # 4. Visualization
    # -----------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    # Plot 1: Comparison of Time Independent Methods
    plt.subplot(1, 2, 1)
    plt.plot(times, ed_obs, "k-", lw=3, alpha=0.5, label="Exact (ED)")
    plt.plot(times, krylov_obs, "r--", label="Krylov")
    plt.plot(times, chebyshev_obs, "b:", label="Chebyshev")
    plt.plot(times, ode_static_obs, "y-.", label="ODE (Static)")
    plt.title("Time-Independent Evolution\n$<Z_0(t)>$")
    plt.xlabel("Time")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Time Dependent Dynamics
    plt.subplot(1, 2, 2)
    plt.plot(times, ed_obs, "k--", alpha=0.5, label="Static H (ED)")
    plt.plot(times, ode_obs, "g-", label="Driven H (ODE)")
    plt.title("Time-Dependent Dynamics\nDriving $X_0$")
    plt.xlabel("Time")
    plt.ylabel("Magnetization $<Z_0(t)>$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("time_evolution_methods.pdf")
    print("\nResults plotted to time_evolution_methods.pdf")

    # Verification of accuracy
    # Compare Krylov/Chebyshev/ODE-Static with ED
    mse_krylov = np.mean((np.array(krylov_obs) - np.array(ed_obs)) ** 2)
    mse_chebyshev = np.mean((np.array(chebyshev_obs) - np.array(ed_obs)) ** 2)
    mse_ode_static = np.mean((np.array(ode_static_obs) - np.array(ed_obs)) ** 2)

    print(f"\nAccuracy Check (MSE vs ED):")
    print(f"Krylov MSE: {mse_krylov:.2e}")
    print(f"Chebyshev MSE: {mse_chebyshev:.2e}")
    print(f"ODE Static MSE: {mse_ode_static:.2e}")

    # Assert basic correctness
    assert mse_krylov < 5e-4, "Krylov deviation too large"
    assert mse_chebyshev < 1e-4, "Chebyshev deviation too large"
    assert mse_ode_static < 1e-4, "ODE Static deviation too large"


if __name__ == "__main__":
    main()
