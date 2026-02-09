"""
VQE benchmark comparing U1Circuit vs regular Circuit.

This example solves a 1D Heisenberg XXZ model which conserves total Sz (particle number).
U1Circuit exploits this symmetry for more efficient simulation.

The Hamiltonian is:
    H = sum_i [ Jxy (X_i X_{i+1} + Y_i Y_{i+1}) + Jz Z_i Z_{i+1} ]

In the half-filling sector (k = n/2), U1Circuit operates on C(n,k) dimensional space
instead of 2^n, providing exponential speedup for small k or k close to n.
"""

import time
import numpy as np
import tensorcircuit as tc
from tensorcircuit.u1circuit import U1Circuit

tc.set_backend("jax")
tc.set_dtype("complex64")

# Model parameters
n = 16  # number of qubits
k = 4  # half-filling (particle number)
nlayers = 4  # number of ansatz layers
Jxy = 1.0  # XX + YY coupling
Jz = 1.0  # ZZ coupling (Jz=Jxy is isotropic Heisenberg)


def heisenberg_energy(params, u1=False):
    """VQE energy calculation supporting both regular Circuit and U1Circuit."""
    if u1:
        # Initial state: half-filled |01000100...>
        filled = list(range(0, n, 4))
        c = U1Circuit(n, k=k, filled=filled)
    else:
        c = tc.Circuit(n)
        # Initial state: half-filled product state |01000100...>
        for i in range(0, n, 4):
            c.x(i)

    # Ansatz: layers of RZZ + RZ + iSWAP (U1-conserving gates)
    for layer in range(nlayers):
        # RZZ gates
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=params[layer, i, 0])
        # RZ gates
        for i in range(n):
            c.rz(i, theta=params[layer, i, 1])
        # SWAP-like entangling (iswap)
        for i in range(layer % 2, n - 1, 2):
            c.iswap(i, i + 1, theta=params[layer, i, 2])

    # Compute Heisenberg energy using expectation_ps
    energy = 0.0
    for i in range(n - 1):
        # XX + YY term
        energy += Jxy * c.expectation_ps(x=[i, i + 1])
        energy += Jxy * c.expectation_ps(y=[i, i + 1])
        # ZZ term
        energy += Jz * c.expectation_ps(z=[i, i + 1])

    return tc.backend.real(energy)


def benchmark(name, vqe_fn, params, nruns=10):
    """Benchmark a VQE function including JIT compilation."""
    vg_fn = tc.backend.jit(tc.backend.value_and_grad(vqe_fn))

    # Warmup and JIT compile
    print(f"\n{name}")
    print("-" * 50)

    time0 = time.time()
    energy, grad = vg_fn(params)
    # Force computation to complete (JAX is lazy)
    energy_val = float(tc.backend.numpy(energy))
    time_first = time.time() - time0
    print(f"First call (includes JIT): {time_first:.3f}s")
    print(f"Initial energy: {energy_val:.6f}")

    # Timed runs
    time0 = time.time()
    for _ in range(nruns):
        energy, grad = vg_fn(params)
        _ = float(tc.backend.numpy(energy))  # Force computation
    time_runs = time.time() - time0

    print(f"Average time ({nruns} runs): {time_runs/nruns:.4f}s")

    return energy_val, tc.backend.numpy(grad), time_first, time_runs / nruns


def run_optimization(name, vqe_fn, params, nsteps=100, lr=0.05):
    """Run VQE optimization."""
    vg_fn = tc.backend.jit(tc.backend.value_and_grad(vqe_fn))

    print(f"\n{name} - Optimization")
    print("-" * 50)

    params = tc.backend.convert_to_tensor(params)
    energies = []

    time0 = time.time()
    for step in range(nsteps):
        energy, grad = vg_fn(params)
        params = params - lr * grad

        if step % 20 == 0 or step == nsteps - 1:
            e = float(tc.backend.numpy(energy))
            energies.append(e)
            print(f"Step {step:3d}: energy = {e:.6f}")

    total_time = time.time() - time0
    print(f"Total optimization time: {total_time:.2f}s")
    print(f"Final energy: {energies[-1]:.6f}")

    return energies[-1], total_time


if __name__ == "__main__":
    print("=" * 60)
    print("U1Circuit vs Circuit VQE Benchmark")
    print("=" * 60)
    print(f"Qubits: {n}, Particles: {k}, Layers: {nlayers}")
    print(f"Circuit dim: 2^{n} = {2**n}")
    print(f"U1Circuit dim: C({n},{k}) = {U1Circuit(n, k=k, filled=[0])._dim}")

    # Initialize parameters
    np.random.seed(42)
    params_np = np.random.normal(0, 0.1, size=[nlayers, n, 3]).astype(np.float32)
    params = tc.backend.convert_to_tensor(params_np)

    # Benchmark both implementations
    nruns = 10

    e1, g1, t1_first, t1_avg = benchmark(
        "Regular Circuit", lambda p: heisenberg_energy(p, u1=False), params, nruns
    )
    e2, g2, t2_first, t2_avg = benchmark(
        "U1Circuit", lambda p: heisenberg_energy(p, u1=True), params, nruns
    )

    # Verify correctness
    print("\n" + "=" * 60)
    print("Correctness Check")
    print("=" * 60)
    energy_diff = abs(e1 - e2)
    grad_diff = np.max(np.abs(g1 - g2))
    print(f"Energy difference: {energy_diff:.2e}")
    print(f"Max gradient difference: {grad_diff:.2e}")

    if energy_diff < 1e-4 and grad_diff < 1e-4:
        print("PASSED: Results match within tolerance")
    else:
        print("WARNING: Results differ significantly")

    # Summary
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"{'Method':<20} {'First (s)':<12} {'Avg (s)':<12} {'Speedup':<10}")
    print("-" * 54)
    print(f"{'Circuit':<20} {t1_first:<12.3f} {t1_avg:<12.4f} {'1.00x':<10}")
    speedup = t1_avg / t2_avg if t2_avg > 0 else float("inf")
    print(f"{'U1Circuit':<20} {t2_first:<12.3f} {t2_avg:<12.4f} {speedup:.2f}x")

    # Optional: run optimization
    print("\n" + "=" * 60)
    print("VQE Optimization Comparison")
    print("=" * 60)

    params_opt = np.random.normal(0, 0.1, size=[nlayers, n, 3]).astype(np.float32)

    e_circuit, t_circuit = run_optimization(
        "Circuit",
        lambda p: heisenberg_energy(p, u1=False),
        params_opt.copy(),
        nsteps=100,
    )
    e_u1, t_u1 = run_optimization(
        "U1Circuit",
        lambda p: heisenberg_energy(p, u1=True),
        params_opt.copy(),
        nsteps=100,
    )

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Circuit final energy: {e_circuit:.6f} (time: {t_circuit:.2f}s)")
    print(f"U1Circuit final energy: {e_u1:.6f} (time: {t_u1:.2f}s)")
    print(f"Optimization speedup: {t_circuit/t_u1:.2f}x")
