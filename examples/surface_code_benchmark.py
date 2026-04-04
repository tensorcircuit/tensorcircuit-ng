"""
Benchmark comparing StabilizerCircuit (Stim) and StabilizerTCircuit (ZX+JAX)
on a surface code circuit with detectors.
"""

import time
import numpy as np
import stim
import jax

# Enable x64 for ExactScalarArray's int64 coefficients
jax.config.update("jax_enable_x64", True)

import tensorcircuit as tc
from tensorcircuit.stabilizertcircuit import StabilizerTCircuit


def stim_to_tc(stim_circuit: stim.Circuit) -> StabilizerTCircuit:
    """
    Parser to convert a stim circuit to StabilizerTCircuit directly.
    """
    nqubits = stim_circuit.num_qubits
    c = StabilizerTCircuit(nqubits)

    rec_count = 0

    for instr in stim_circuit:
        name = instr.name
        targets = instr.targets_copy()

        if name == "H":
            for t in targets:
                c.h(t.value)
        elif name == "X":
            for t in targets:
                c.x(t.value)
        elif name == "Y":
            for t in targets:
                c.y(t.value)
        elif name == "Z":
            for t in targets:
                c.z(t.value)
        elif name == "S":
            for t in targets:
                c.s(t.value)
        elif name == "S_DAG":
            for t in targets:
                c.sd(t.value)
        elif name == "CNOT" or name == "CX":
            for i in range(0, len(targets), 2):
                c.cnot(targets[i].value, targets[i + 1].value)
        elif name == "CZ":
            for i in range(0, len(targets), 2):
                c.cz(targets[i].value, targets[i + 1].value)
        elif name == "SWAP":
            for i in range(0, len(targets), 2):
                c.swap(targets[i].value, targets[i + 1].value)
        elif name == "M":
            for t in targets:
                c.measure_instruction(t.value)
                rec_count += 1
        elif name == "MR":
            for t in targets:
                c.mr_instruction(t.value)
                rec_count += 1
        elif name == "R":
            for t in targets:
                c.reset_instruction(t.value)
        elif name == "DETECTOR":
            recs = []
            for t in targets:
                if t.is_measurement_record_target:
                    abs_idx = rec_count + t.value
                    recs.append(abs_idx)
            c.detector_instruction(recs)
        elif name == "OBSERVABLE_INCLUDE":
            recs = []
            obs_idx = int(instr.gate_args_copy()[0])
            for t in targets:
                if t.is_measurement_record_target:
                    abs_idx = rec_count + t.value
                    recs.append(abs_idx)
            c.observable_instruction(recs, observable_index=obs_idx)
        elif name == "DEPOLARIZE1":
            p = instr.gate_args_copy()[0]
            for t in targets:
                c.depolarizing(t.value, p=p)
        elif name == "DEPOLARIZE2":
            p = instr.gate_args_copy()[0]
            for i in range(0, len(targets), 2):
                c.depolarizing2(targets[i].value, targets[i + 1].value, p=p)
        elif name == "X_ERROR":
            p = instr.gate_args_copy()[0]
            for t in targets:
                c.x_error(t.value, p=p)
        elif name == "Y_ERROR":
            p = instr.gate_args_copy()[0]
            for t in targets:
                c.y_error(t.value, p=p)
        elif name == "Z_ERROR":
            p = instr.gate_args_copy()[0]
            for t in targets:
                c.z_error(t.value, p=p)
        elif name == "TICK":
            c.tick_instruction()
        elif name == "QUBIT_COORDS":
            coords = instr.gate_args_copy()
            for t in targets:
                c.qubit_coords_instruction(t.value, list(coords))
        elif name == "SHIFT_COORDS":
            pass
    return c


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
    tc_c = stim_to_tc(stim_sc.flattened())

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

    stc = stim_to_tc(stim_sc.flattened())

    # Compilation time (includes ZX reduction)
    start_comp = time.time()
    stc._compile(sample_detectors=True)
    end_comp = time.time()
    print(f"ZX Compilation & Reduction Time: {end_comp - start_comp:.4f}s")

    # Execution (Run 1: includes JIT)
    start_exec1 = time.time()
    samples_jax = stc.sample_detectors(shots=batch, seed=42)
    end_exec1 = time.time()
    print(f"Execution Time (Run 1, including JIT): {end_exec1 - start_exec1:.4f}s")

    # Execution (Run 2: pure execution)
    start_exec2 = time.time()
    stc.sample_detectors(shots=batch, seed=43)
    end_exec2 = time.time()
    print(
        f"Execution Time (Run 2, pure execution, batch=1000): {end_exec2 - start_exec2:.4f}s"
    )

    # Execution (Run 3: larger batch_size)
    start_exec3 = time.time()
    stc.sample_detectors(shots=batch, seed=44, batch_size=1000)
    end_exec3 = time.time()
    print(
        f"Execution Time (Run 3, larger batch, batch=5000): {end_exec3 - start_exec3:.4f}s"
    )

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
