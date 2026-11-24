"""Recompute the archived W-state metrics without training."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np


def state(parameters, bits, phased=False):
    """Evaluate a full-space binary RBM in big-endian bit order."""
    n = bits.shape[1]
    width = n * (n + 2)
    if parameters.shape != (width * (2 if phased else 1),):
        raise ValueError("Unexpected parameter shape")

    def log_weight(p):
        bias, hidden = p[:n], p[n : 2 * n]
        fields = bits @ p[2 * n :].reshape(n, n).T + hidden
        return bits @ bias + np.logaddexp(0.0, fields).sum(axis=1)

    weights = log_weight(parameters[:width])
    magnitudes = np.exp((weights - np.logaddexp.reduce(weights)) / 2)
    if not phased:
        return magnitudes
    return magnitudes * np.exp(0.5j * log_weight(parameters[width:]))


def compare(records, raw):
    """Check grid coverage, O/F, leakage, and the matched Z-only controls."""
    sizes = (8, 10, 12)
    seeds = (0, 1, 2)
    minima = []
    for phased in (False, True):
        if phased:
            rows = records["complex"]
            expected = set(product(sizes, seeds))
            keys = [(r["n"], r["seed"]) for r in rows]
        else:
            rows = records["positive"]
            expected = set(product(sizes, seeds, (20, 100, 500, 2000, 10000)))
            keys = [(r["n"], r["seed"], r["shots"]) for r in rows]
        if set(keys) != expected or len(keys) != len(expected):
            raise ValueError("Incomplete or duplicated experiment grid")
        for row in rows:
            n, seed = row["n"], row["seed"]
            shifts = np.arange(n - 1, -1, -1)
            support = 1 << shifts
            bits = ((np.arange(2**n)[:, None] >> shifts) & 1).astype(float)
            tag = f"n{n}_seed{seed}"
            target = np.zeros(2**n, dtype=complex)
            if phased:
                target[support] = np.exp(
                    1j * raw[f"complex_target_phases_{tag}"]
                ) / np.sqrt(n)
                fitted = state(raw[f"complex_parameters_{tag}"], bits, phased=True)
                control = state(raw[f"complex_z_parameters_{tag}"], bits)
                np.testing.assert_allclose(
                    abs(np.vdot(target, control)) ** 2,
                    row["z_only_fidelity"],
                    rtol=0,
                    atol=1e-10,
                    err_msg=f"{tag}: Z-only fidelity",
                )
            else:
                target[support] = 1 / np.sqrt(n)
                fitted = state(
                    raw[f"positive_parameters_{tag}_shots{row['shots']}"], bits
                )
            overlap = abs(np.vdot(target, fitted))
            leakage = 1 - np.sum(np.abs(fitted[support]) ** 2)
            np.testing.assert_allclose(
                [overlap, overlap**2, leakage, np.vdot(fitted, fitted).real],
                [row["overlap"], row["fidelity"], row["leakage"], 1],
                rtol=0,
                atol=1e-10,
                err_msg=f"{tag}: O, F, leakage, normalization",
            )
            if phased:
                fitted_phases = np.angle(fitted[support] / fitted[support[0]])
                target_phases = np.angle(target[support] / target[support[0]])
                np.testing.assert_allclose(
                    np.exp(1j * fitted_phases),
                    np.exp(1j * np.asarray(row["relative_phases"])),
                    rtol=0,
                    atol=1e-10,
                    err_msg=f"{tag}: learned relative phases",
                )
                np.testing.assert_allclose(
                    np.exp(1j * target_phases),
                    np.exp(1j * np.asarray(row["target_relative_phases"])),
                    rtol=0,
                    atol=1e-10,
                    err_msg=f"{tag}: target relative phases",
                )
                aligned = fitted[support] * target[support].conj()
                phase_error = np.angle(
                    aligned * np.exp(-1j * np.angle(np.vdot(target, fitted)))
                )
                np.testing.assert_allclose(
                    np.sqrt(np.mean(phase_error**2)),
                    row["phase_rmse"],
                    rtol=0,
                    atol=1e-10,
                    err_msg=f"{tag}: phase RMSE",
                )
                minima.append(overlap**2)
    return min(minima)


def main():
    """Run the read-only check on the reference or a fresh run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory containing metrics.json and raw_data.npz",
    )
    folder = parser.parse_args().directory
    records = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    with np.load(folder / "raw_data.npz", allow_pickle=False) as raw:
        minimum = compare(records, raw)
    print(
        f"Verified 54 fits and 9 matched Z-only controls; min random-phase F={minimum:.6f}"
    )


if __name__ == "__main__":
    main()
