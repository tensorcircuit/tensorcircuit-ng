"""
Benchmark comparing StabilizerCircuit (Stim) and StabilizerTCircuit (ZX+JAX)
on a surface code circuit with detectors.
"""

import time
import numpy as np
import stim

from tensorcircuit.zx.stabilizertcircuit import StabilizerTCircuit


def run_benchmark(distance=3, rounds=3, batch=1000):
    print(
        f"--- Surface Code Benchmark (d={distance}, rounds={rounds}, batch={batch}) ---"
    )

    # 1. Generate Surface Code with Stim
    stim_sc = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.001,
        after_reset_flip_probability=0.001,
        before_measure_flip_probability=0.001,
    )
    num_qubits = stim_sc.num_qubits
    num_detectors = stim_sc.num_detectors
    print(f"Circuit stats: Qubits: {num_qubits}, Detectors: {num_detectors}")

    # 2. Convert to TC
    stc = StabilizerTCircuit.from_stim_circuit(stim_sc)

    # --- StabilizerCircuit (Stim Backend) ---
    print("\n[StabilizerCircuit / Stim Backend]")
    start = time.time()
    sampler_stim = stim_sc.compile_detector_sampler()
    samples_stim = sampler_stim.sample(batch)
    end = time.time()
    print(f"Stim Sampling Time: {end - start:.4f}s")

    p_stim = np.mean(samples_stim, axis=0)

    # --- StabilizerTCircuit (ZX+JAX Backend) ---
    print("\n[StabilizerTCircuit / ZX+JAX Backend]")

    # Compilation time (includes ZX reduction)
    start_comp = time.time()
    stc._compile(sample_detectors=True)
    end_comp = time.time()
    print(f"ZX Compilation & Reduction Time: {end_comp - start_comp:.4f}s")

    # Execution (Run 1: includes JIT)
    start_exec1 = time.time()
    samples_jax = stc.sample_detectors(shots=batch, seed=42, batch_size=2000)
    end_exec1 = time.time()
    print(f"Execution Time (Run 1, including JIT): {end_exec1 - start_exec1:.4f}s")

    # Execution (Run 2: pure execution)
    start_exec2 = time.time()
    stc.sample_detectors(shots=batch, seed=43, batch_size=2000)
    end_exec2 = time.time()
    print(f"Execution Time (Run 2, pure execution): {end_exec2 - start_exec2:.4f}s")

    p_jax = np.mean(samples_jax, axis=0)

    # --- Correctness Check ---
    print("\n--- Correctness Check (Per-Detector Probability of 1) ---")
    print(f"{'Det Index':<10} | {'Stim Prob':<15} | {'ZX+JAX Prob':<15} | {'Diff':<10}")
    print("-" * 60)

    max_diff = 0
    for i in range(num_detectors):
        diff = abs(p_stim[i] - p_jax[i])
        max_diff = max(max_diff, diff)
        print(f"{i:<10} | {p_stim[i]:<15.6f} | {p_jax[i]:<15.6f} | {diff:<10.6f}")

    print("-" * 60)
    print(f"Max statistical difference: {max_diff:.6f}")

    threshold = 5 * np.sqrt(0.2 * 0.8 / batch)
    if max_diff < threshold:
        print(f"SUCCESS: Results match within statistical threshold ({threshold:.4f})")
    else:
        print(f"WARNING: Results differ by more than threshold ({threshold:.4f})")


if __name__ == "__main__":
    run_benchmark(distance=5, rounds=5, batch=50000)
