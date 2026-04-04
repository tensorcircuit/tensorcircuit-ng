"""
Magic state distillation example using StabilizerTCircuit (ZX-calculus).
Aligned with tsim implementation.
"""

import numpy as np
from tensorcircuit.zx.stabilizertcircuit import StabilizerTCircuit


def post_select(samples):
    """
    Post-select on the [1, 0, 1, 1] distillation syndrome for the last 4 qubits.
    """
    distilled_output = samples[:, 0]
    distillation_syndromes = samples[:, 1:]
    # Syndrome [1, 0, 1, 1] means qubits 1, 3, 4 are 1 and qubit 2 is 0
    sel = np.all(distillation_syndromes == np.array([1, 0, 1, 1]), axis=1)
    return distilled_output[sel]


def main():
    # 1. Setup parameters
    p = 0.05
    theta = -np.arccos(np.sqrt(1 / 3)) / np.pi  # Distillation angle in units of pi
    theta_rad = theta * np.pi  # Distillation angle in radians for TC methods
    shots = 50_000

    print(f"Distillation angle: {theta:.5f} pi")
    print(f"Initial infidelity: {p:.2f}")

    # 2. Build the distillation circuit
    # The circuit prepares 5 magic states, applies distillation gates, and measures.
    c = StabilizerTCircuit(5, seed=0)

    # State preparation (noisy)
    for i in range(5):
        c.r(i)
        c.rx(i, theta=theta_rad)
        c.td(i)
        c.depolarizing(i, p)

    # Distillation circuit (Clifford gates)
    c.sqrt_x(0, 1, 4)
    c.cz(0, 1)
    c.cz(2, 3)
    c.sqrt_y(0, 3)
    c.cz(0, 2)
    c.cz(3, 4)

    c.tick_instruction()
    c.sqrt_x_dag(0)
    c.cz(0, 4)
    c.cz(1, 3)

    c.tick_instruction()
    c.sqrt_x_dag(0, 1, 2, 3, 4)

    # Undo preparation on qubit 0 to measure fidelity
    c.t(0)
    c.rx(0, theta=-theta_rad)

    # Measurement
    for i in range(5):
        c.measure_instruction(i)

    # 3. Sample and Post-select
    print(f"Sampling {shots} shots...")
    samples = c.sample_measurements(shots=shots, batch_size=10_000)
    samples = np.array(samples)

    post_selected_samples = post_select(samples)

    infidelity = np.count_nonzero(post_selected_samples) / len(post_selected_samples)
    success_rate = len(post_selected_samples) / shots

    print(f"Distilled Infidelity: {infidelity:.5f}")
    print(f"Post-selection rate: {success_rate * 100:.2f}%")

    # Expected values from tsim notebook:
    # Infidelity: ~0.007 (Fidelity ~99.3%)
    # Success rate: ~14%


if __name__ == "__main__":
    main()
