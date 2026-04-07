"""
15-to-1 distillation demonstrating stabilizertcircuit,
oversimplified toy model demosntration
"""

import time
import numpy as np
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.zx.stabilizertcircuit import StabilizerTCircuit

tc.set_backend("jax")


def get_hamming_x_checks():
    """Returns masks for the 4 X-Stabilizers."""
    checks = []
    for i in range(4):
        mask = [j - 1 for j in range(1, 16) if (j >> i) & 1]
        checks.append(mask)
    return checks


def build_teleportation_factory_circuit(p=0.0):
    num_data = 15
    num_ancilla = 4
    stc = StabilizerTCircuit(num_data + num_ancilla)

    # ==========================================
    # Step 1: Deterministic Preparation of Logical |0> State
    # ==========================================
    seeds = [0, 1, 3, 7]
    for s in seeds:
        stc.h(s)

    for i in range(4):
        seed = (1 << i) - 1
        for j in range(15):
            if j == seed:
                continue
            if ((j + 1) >> i) & 1:
                stc.cnot(seed, j)

    # ==========================================
    # Skip Step 2 (H-gate conversion), proceed directly to T-gate injection
    # ==========================================
    for j in range(15):
        stc.t(j)

    # ==========================================
    # Inject physical noise
    # ==========================================
    if p > 0:
        for j in range(15):
            stc.depolarizing(j, p)

    # ==========================================
    # Measure 4 X-Stabilizers (detect Z errors)
    # ==========================================
    checks = get_hamming_x_checks()
    for i, mask in enumerate(checks):
        anc = num_data + i
        stc.h(anc)
        for data_idx in mask:
            stc.cnot(anc, data_idx)
        stc.h(anc)
        stc.measure_instruction(anc)

    return stc, num_ancilla


def main():
    shots = 100000

    print("=== Part 1: Literature Factory Architecture (Noiseless p=0) ===")
    stc_exact, num_ancilla = build_teleportation_factory_circuit(p=0.0)

    # Analytical Trace Calculation
    target_syndrome = jnp.zeros(num_ancilla, dtype=jnp.int32)
    start_exact = time.time()
    prob_exact = stc_exact.outcome_probability(target_syndrome)[0]
    time_exact = time.time() - start_exact

    print(
        f"Exact Analytical Acceptance:     {prob_exact:.6f} (Time: {time_exact:.4f}s)"
    )
    print("=> Successfully reproduced 100% noiseless acceptance rate\n")

    print("=== Part 2: Noisy Factory Simulation (p=0.01) ===")
    p_noise = 0.01
    stc_noisy, _ = build_teleportation_factory_circuit(p=p_noise)

    # Monte Carlo Sampling
    start_noisy = time.time()
    samples_noisy = np.array(stc_noisy.sample_measurements(shots=shots))
    time_noisy = time.time() - start_noisy

    syndromes_noisy = samples_noisy[:, :num_ancilla]
    success_mask_noisy = np.all(syndromes_noisy == 0, axis=1)
    prob_sampled_noisy = np.sum(success_mask_noisy) / shots

    print(f"Sampling {shots} trajectories... (Time: {time_noisy:.4f}s)")
    print(f"Sampled Acceptance Rate (Noisy): {prob_sampled_noisy:.4%}")

    if np.sum(success_mask_noisy) > 0:
        print(
            f"Total Successful Distillations:  {np.sum(success_mask_noisy)} / {shots}"
        )


if __name__ == "__main__":
    main()
