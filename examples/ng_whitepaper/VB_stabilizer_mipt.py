"""
Measurement Induced Phase Transition (MIPT) in Random Clifford Circuits
=====================================================================
 This script demonstrates the usage of StabilizerCircuit in TensorCircuit
 to simulate measurement induced phase transitions.
 We observe the transition from volume law to area law entanglement scaling
 by plotting the steady-state late time entanglement entropy for different system sizes.
 The crossing point determines the critical measurement probability.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorcircuit as tc


def run_single_circuit(L, p, depth):
    """
    Run a single instance of random Clifford circuit with measurements.

    Args:
        L: System size
        p: Measurement probability
        depth: Circuit depth

    Returns:
        float: Average entanglement entropy over the late time winow.
    """
    c = tc.StabilizerCircuit(L)

    # We record entropy in the second half of the evolution to average
    entropies = []

    for t in range(depth):
        # Apply random 2-qubit Clifford gates
        # Even layers
        for i in range(0, L, 2):
            c.random_gate(i, (i + 1) % L)

        for i in range(L):
            if np.random.random() < p:
                c.cond_measure(i)

        # Odd layers
        for i in range(1, L, 2):
            c.random_gate(i, (i + 1) % L)

        # Measurements
        for i in range(L):
            if np.random.random() < p:
                c.cond_measure(i)

        # Calculate Entropy
        # We only calculate it in the late time window to save time
        if t >= depth * 0.8:
            # Cut at L/2
            cut = list(range(L // 2))
            sent = c.entanglement_entropy(cut)
            entropies.append(sent)

    return np.mean(entropies)


def main():
    Ls = [50, 70, 90]
    ps = np.linspace(0.0, 0.4, 21)

    depth_factor = 2  # depth = L * depth_factor
    n_trials = 20  # Number of trajectories to average

    results = {}

    print("Simulating MIPT with StabilizerCircuit...")
    for L in Ls:
        results[L] = []
        depth = L * depth_factor
        print(f"Simulating L = {L}")
        for p in tqdm(ps):
            avg_s = 0
            for i in range(n_trials):
                avg_s += run_single_circuit(L, p, depth)
            avg_s /= n_trials
            results[L].append(avg_s)
            # print(f"  Finished p={p:.3f}, S={avg_s:.4f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    colors = ["#FF9999", "#66B2FF", "#99FF99"]
    markers = ["o", "s", "^"]

    for i, L in enumerate(Ls):
        plt.plot(
            ps,
            results[L],
            marker=markers[i],
            label=f"L={L}",
            color=colors[i],
            markeredgecolor="black",
            alpha=0.9,
            lw=2,
        )

    plt.xlabel(r"Measurement Probability $p$", fontsize=14)
    plt.ylabel(r"Half-chain Entanglement Entropy $S_{L/2}$", fontsize=14)
    plt.title("Measurement Induced Phase Transition", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Heuristic for crossing location
    # Ideally around 0.16

    output_path = "mipt_crossing.pdf"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
