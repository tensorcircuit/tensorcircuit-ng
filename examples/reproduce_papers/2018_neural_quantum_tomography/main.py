"""Torlai et al., Figure 1: unrestricted RBM tomography of a 20-qubit W state.

Run with JAX and Optax. Simulated projective measurements come from a bond-two
TensorCircuit MPS. Training uses Monte Carlo amplitude updates followed by a
local natural gradient for the phase. Exact normalization is used for evaluation.
"""

import argparse
import json
from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from measurements import (
    make_sampler,
    sample_bases,
    sample_basis,
    unpack_bits,
    w_tensors,
)
from rbm import (
    AMPLITUDE_BATCH,
    AMPLITUDE_RATE,
    AMPLITUDE_STEPS,
    FISHER_DIAGONAL,
    PHASE_RATE,
    PHASE_STEPS,
    fit_phase,
    log_weight,
    make_amplitude_trainer,
    phase_data,
)
import tensorcircuit as tc

N = 20
SEEDS = (0, 1, 2)
SHOTS = (50, 1000, 20000)
PHASE_SHOTS = 6400


def singleton_probabilities(parameters):
    """Normalize in blocks after training, then evaluate W-component probabilities."""
    K = tc.backend
    block_size = 8192

    @K.jit
    def block_partition(params, bits):
        return jax.scipy.special.logsumexp(
            log_weight(params, K.cast(bits, tc.rdtypestr))
        )

    block_totals = []
    for start in range(0, 2**N, block_size):
        bits = unpack_bits(np.arange(start, start + block_size, dtype=np.uint32), N)
        block_totals.append(block_partition(parameters, K.convert_to_tensor(bits)))
    log_partition = logsumexp(np.asarray(block_totals, dtype=np.float64))
    logits = K.numpy(log_weight(parameters, K.eye(N, dtype=tc.rdtypestr)))
    return np.exp(logits.astype(np.float64) - log_partition)


def overlap(probability, phase_difference):
    """Return the magnitude of the inner product, as in Figure 1."""
    return float(abs(np.mean(np.sqrt(N * probability) * np.exp(1j * phase_difference))))


def relative_phase(phase):
    """Reference all singleton phases to site one, using the principal angle."""
    return np.angle(np.exp(1j * (phase - phase[0])))


def run_seed(seed, sampler, train_amplitude):
    """Simulate the registered shot budgets and fit both experiments."""
    K = tc.backend
    positive_rng, phase_rng, target_rng = [
        np.random.default_rng(stream)
        for stream in np.random.SeedSequence(seed).spawn(3)
    ]
    positive = sample_basis(sampler, w_tensors(np.zeros(N)), SHOTS[-1], positive_rng)
    target = target_rng.uniform(-np.pi, np.pi, N)
    phased = sample_bases(sampler, target, PHASE_SHOTS, phase_rng)
    init_key, phase_init_key, *fit_keys = jax.random.split(jax.random.key(seed), 6)
    initial = 0.1 * jax.random.normal(init_key, (N * (N + 2),), dtype=np.float32)
    initial = initial.at[: 2 * N].set(0)
    phase_init = 0.1 * jax.random.normal(
        phase_init_key, initial.shape, dtype=np.float32
    )

    data_sets = [positive[:shots] for shots in SHOTS] + [phased[0]]
    amplitudes = [
        train_amplitude(
            K.cast(K.convert_to_tensor(unpack_bits(data, N)), tc.rdtypestr),
            initial,
            key,
        )
        for data, key in zip(data_sets, fit_keys)
    ]
    terms, coefficients, counts, angles = phase_data(phased, N)
    phase = fit_phase(amplitudes[-1], phase_init, terms, coefficients, counts, angles)
    probabilities = np.stack([singleton_probabilities(p) for p in amplitudes])
    learned = relative_phase(
        0.5 * K.numpy(log_weight(phase, K.eye(N, dtype=tc.rdtypestr)))
    )
    target = relative_phase(target)
    phase_probability = probabilities[-1]
    record = {
        "seed": seed,
        "positive": [
            {
                "shots": shots,
                "overlap": overlap(p, 0),
                "single_excitation_mass": float(p.sum()),
                "populations": p.tolist(),
            }
            for shots, p in zip(SHOTS, probabilities)
        ],
        "complex": {
            "overlap": overlap(phase_probability, learned - target),
            "z_only_overlap": overlap(phase_probability, -target),
            "single_excitation_mass": float(phase_probability.sum()),
            "phase_rms_radians": float(
                np.sqrt(np.mean(np.angle(np.exp(1j * (learned - target))) ** 2))
            ),
            "target_relative_phase": target.tolist(),
            "learned_relative_phase": learned.tolist(),
            "populations": phase_probability.tolist(),
        },
    }
    raw = {
        "positive_z": positive,
        "phased_bases": phased,
        "amplitude_parameters": np.stack([K.numpy(p) for p in amplitudes]),
        "phase_parameters": K.numpy(phase),
    }
    return record, raw


def plot_results(records, output):
    """Plot the effect of shot count and the information from rotated bases."""
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout="constrained")
    values = np.array([[r["overlap"] for r in row["positive"]] for row in records])
    axes[0].scatter(
        np.tile(SHOTS, len(SEEDS)), values.ravel(), color="#5091bb", alpha=0.6
    )
    axes[0].plot(SHOTS, values.mean(axis=0), "o-", color="#14567c", label="Mean")
    axes[0].set(
        xscale="log",
        xticks=SHOTS,
        xticklabels=["50", "1,000", "20,000"],
        xlabel="Z training shots",
        ylabel="Overlap",
        title="Positive W",
    )
    axes[0].legend()
    sites = np.arange(1, N + 1)
    for index, shots in enumerate(SHOTS):
        populations = [row["positive"][index]["populations"] for row in records]
        axes[1].plot(
            sites, np.mean(populations, axis=0), ".-", label=f"{shots:,} shots"
        )
    axes[1].axhline(1 / N, color="0.3", linestyle=":", label="Exact")
    axes[1].set(
        xlabel="Excited site",
        ylabel="Unconditional probability",
        title="Single-excitation populations",
        ylim=(0, None),
    )
    axes[1].legend(fontsize=8)
    for column, key, label in [
        (0, "z_only_overlap", "Z only"),
        (1, "overlap", "Z + XX + XY"),
    ]:
        positions = column + np.linspace(-0.06, 0.06, len(SEEDS))
        axes[2].scatter(
            positions, [row["complex"][key] for row in records], label=label
        )
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["Z only", "Z + XX + XY"],
        xlim=(-0.3, 1.3),
        ylim=(0, 1.04),
        ylabel="Overlap",
        title="Random-phase W",
    )
    fig.suptitle("W-state tomography at 20 qubits · three seeds")
    fig.savefig(output / "result.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(SEEDS), figsize=(12, 3.7), layout="constrained", sharey=True
    )
    for ax, row in zip(axes, records):
        values = row["complex"]
        ax.plot(sites, values["target_relative_phase"], "o", color="0.2", label="Exact")
        ax.plot(
            sites,
            values["learned_relative_phase"],
            "x",
            color="#c84e36",
            label="Learned",
        )
        ax.set(
            xlabel="Excited site",
            ylim=(-3.4, 3.4),
            title=f"Seed {row['seed']} · overlap {values['overlap']:.3f}",
        )
    axes[0].set_ylabel("Phase relative to site 1 (rad)")
    axes[0].legend()
    fig.suptitle("Random-phase W · 6,400 training shots per basis")
    fig.savefig(output / "phases.png", dpi=160)
    plt.close(fig)


def main():
    """Run the registered experiment and save its figure data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs"
    )
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    tc.set_backend("jax")
    sampler = make_sampler()
    train_amplitude = make_amplitude_trainer()
    records, samples = zip(
        *(run_seed(seed, sampler, train_amplitude) for seed in SEEDS)
    )
    metrics = {
        "protocol": {
            "qubits": N,
            "seeds": SEEDS,
            "positive_shots": SHOTS,
            "phase_shots_per_basis": PHASE_SHOTS,
            "phase_bases": 2 * N - 1,
            "amplitude_steps": AMPLITUDE_STEPS,
            "amplitude_batch": AMPLITUDE_BATCH,
            "amplitude_rate": AMPLITUDE_RATE,
            "phase_steps": PHASE_STEPS,
            "phase_trust_radius": PHASE_RATE,
            "fisher_diagonal": FISHER_DIAGONAL,
        },
        "records": records,
    }
    np.savez_compressed(
        output / "raw_data.npz",
        **{key: np.stack([row[key] for row in samples]) for key in samples[0]},
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(records, output)


if __name__ == "__main__":
    main()
