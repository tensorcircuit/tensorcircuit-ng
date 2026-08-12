"""Reproduction of "A Quantum Approximate Optimization Algorithm"
Link: https://arxiv.org/abs/1411.4028
Description:
This script reproduces the ring-of-disagrees result from Section IV using TensorCircuit-NG.
It optimizes QAOA angles and compares the approximation ratio with the analytic curve.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NQUBITS = 12
MAX_DEPTH = 4
OPTIMIZATION_STEPS = 140
RESTARTS = 3
LEARNING_RATE = 0.04


def ring_edges(nqubits):
    """Return the edges of an even ring graph."""

    return [(site, (site + 1) % nqubits) for site in range(nqubits)]


def make_cost_function(depth):
    """Create the QAOA MaxCut expectation for a fixed circuit depth."""

    edges = ring_edges(NQUBITS)

    def cost(parameters):
        circuit = tc.Circuit(NQUBITS)
        for site in range(NQUBITS):
            circuit.h(site)
        for layer in range(depth):
            gamma = parameters[layer, 0]
            beta = parameters[layer, 1]
            for left, right in edges:
                circuit.rzz(left, right, theta=-gamma)
            for site in range(NQUBITS):
                circuit.rx(site, theta=2.0 * beta)

        values = []
        for left, right in edges:
            correlation = tc.backend.real(circuit.expectation_ps(z=[left, right]))
            values.append(0.5 * (1.0 - correlation))
        return tc.backend.sum(tc.backend.stack(values))

    return cost


def optimize_depth(depth, rng):
    """Optimize QAOA angles and return the best expected cut ratio."""

    cost = make_cost_function(depth)
    value_and_grad = tc.backend.jit(tc.backend.value_and_grad(cost))
    best_value = -np.inf
    for _ in range(RESTARTS):
        initial = rng.uniform(
            low=-np.pi,
            high=np.pi,
            size=(depth, 2),
        ).astype(np.float32)
        parameters = tc.backend.convert_to_tensor(initial)
        optimizer = optax.adam(LEARNING_RATE)
        opt_state = optimizer.init(parameters)

        @tc.backend.jit
        def update(parameters, opt_state):
            value, gradient = value_and_grad(parameters)
            updates, opt_state = optimizer.update(-gradient, opt_state)
            parameters = optax.apply_updates(parameters, updates)
            return parameters, opt_state, value

        for _ in range(OPTIMIZATION_STEPS):
            parameters, opt_state, _ = update(parameters, opt_state)
        value = float(tc.backend.numpy(cost(parameters)))
        if value > best_value:
            best_value = value
    return best_value / NQUBITS


def plot_results(ratios):
    """Plot numerical QAOA ratios against the analytic ring result."""

    depths = np.arange(1, MAX_DEPTH + 1)
    analytic = (2.0 * depths + 1.0) / (2.0 * depths + 2.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(depths, ratios, "o-", linewidth=2.0, label="TensorCircuit QAOA")
    ax.plot(
        depths,
        analytic,
        "k--",
        linewidth=1.5,
        label=r"analytic $(2p+1)/(2p+2)$",
    )
    ax.set_xticks(depths)
    ax.set_xlabel("QAOA depth $p$")
    ax.set_ylabel("Expected cut / optimal cut")
    ax.set_ylim(0.65, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "result.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    """Run the scaled QAOA ring experiment."""

    rng = np.random.default_rng(14114028)
    ratios = []
    for depth in range(1, MAX_DEPTH + 1):
        ratio = optimize_depth(depth, rng)
        ratios.append(ratio)
        print(f"p={depth}: approximation ratio={ratio:.4f}")
    plot_results(np.asarray(ratios))


if __name__ == "__main__":
    main()
