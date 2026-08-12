"""Reproduction of "Measurement-Induced Phase Transitions in the Dynamics of Entanglement"
Link: https://arxiv.org/abs/1808.05953
Description:
This script reproduces Figure 13(a) from the paper using TensorCircuit-NG.
It measures bipartite entanglement in monitored random circuits.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_SIZES = (6, 8, 10, 12)
MEASUREMENT_RATES = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.6])
REALIZATIONS = 12
TRAJECTORIES = 32
SEED = 180805953


def make_haar_unitary(rng):
    """Draw one reproducible Haar-random two-qubit unitary."""

    matrix = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
    matrix, diagonal = np.linalg.qr(matrix)
    phases = np.array(np.diag(diagonal), copy=True)
    phases /= np.abs(phases)
    return (matrix * phases.conj()).astype(np.complex64)


def make_measurement_kraus(rate):
    """Return the three Kraus operators used by the standard MIPT example."""

    projector_zero = tc.backend.convert_to_tensor(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
    )
    projector_one = tc.backend.convert_to_tensor(
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex64)
    )
    identity = tc.backend.eye(2)
    rate_tensor = tc.backend.convert_to_tensor(rate)
    return [
        tc.backend.sqrt(rate_tensor) * projector_zero,
        tc.backend.sqrt(rate_tensor) * projector_one,
        tc.backend.sqrt(1.0 - rate_tensor) * identity,
    ]


def make_trajectory_runner(nqubits, rounds, rate):
    """Create the standard TensorCircuit Kraus-trajectory evaluator."""

    kraus = make_measurement_kraus(rate)

    def circuit_output(random_matrix, status):
        random_matrix = tc.backend.reshape(random_matrix, [rounds, nqubits, 4, 4])
        status = tc.backend.reshape(status, [rounds, nqubits])
        inputs = None
        for round_index in range(rounds):
            start = round_index % 2
            circuit = (
                tc.Circuit(nqubits)
                if inputs is None
                else tc.Circuit(nqubits, inputs=inputs)
            )
            for site in range(start, nqubits - 1, 2):
                circuit.unitary(
                    site,
                    site + 1,
                    unitary=random_matrix[round_index, site],
                )
            inputs = circuit.state()
            circuit = tc.Circuit(nqubits, inputs=inputs)
            for site in range(start, nqubits - 1, 2):
                for measured_site in (site, site + 1):
                    circuit.general_kraus(
                        kraus,
                        measured_site,
                        status=status[round_index, measured_site],
                    )
                    inputs = circuit.state()
                    circuit = tc.Circuit(nqubits, inputs=inputs)
            inputs = circuit.state()
            inputs /= tc.backend.norm(inputs)

        reduced = tc.quantum.reduced_density_matrix(
            inputs, cut=list(range(nqubits // 2))
        )
        return tc.quantum.entropy(reduced) / np.log(2.0)

    return tc.backend.jit(tc.backend.vmap(circuit_output, vectorized_argnums=(0, 1)))


def entropy_for_rate(nqubits, rate, rng):
    """Average entropy over random circuits and Kraus measurement trajectories."""

    rounds = 2 * nqubits
    batch_size = REALIZATIONS * TRAJECTORIES
    random_matrix = np.zeros((batch_size, rounds, nqubits, 4, 4), dtype=np.complex64)
    for realization in range(REALIZATIONS):
        base = np.zeros((rounds, nqubits, 4, 4), dtype=np.complex64)
        for round_index in range(rounds):
            for site in range(round_index % 2, nqubits - 1, 2):
                base[round_index, site] = make_haar_unitary(rng)
        start = realization * TRAJECTORIES
        random_matrix[start : start + TRAJECTORIES] = base

    status = rng.random((batch_size, rounds, nqubits)).astype(np.float32)
    runner = make_trajectory_runner(nqubits, rounds, rate)
    values = runner(
        tc.backend.convert_to_tensor(random_matrix),
        tc.backend.convert_to_tensor(status),
    )
    return float(np.mean(np.asarray(tc.backend.numpy(values))))


def simulate_entropy():
    """Generate the finite-size entropy curves used in the reproduction."""

    rng = np.random.default_rng(SEED)
    curves = np.zeros((len(SYSTEM_SIZES), len(MEASUREMENT_RATES)))
    for size_index, nqubits in enumerate(SYSTEM_SIZES):
        for rate_index, rate in enumerate(MEASUREMENT_RATES):
            curves[size_index, rate_index] = entropy_for_rate(nqubits, rate, rng)
        print(f"L={nqubits}: {np.round(curves[size_index], 3)}")
    return curves


def plot_results(curves):
    """Plot entropy versus measurement rate for the scaled system sizes."""

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for nqubits, curve in zip(SYSTEM_SIZES, curves):
        ax.plot(
            MEASUREMENT_RATES,
            curve,
            "o-",
            linewidth=1.8,
            markersize=4,
            label=rf"$L={nqubits}$",
        )
    ax.axvline(0.26, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.27, ax.get_ylim()[1] * 0.92, r"$p_c\approx0.26$", fontsize=9)
    ax.set_xlabel("Measurement rate $p$")
    ax.set_ylabel("Bipartite von Neumann entropy $S_1$ (bits)")
    ax.set_title("Monitored random circuits")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "result.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    """Run the scaled measurement-induced transition experiment."""

    plot_results(simulate_entropy())


if __name__ == "__main__":
    main()
