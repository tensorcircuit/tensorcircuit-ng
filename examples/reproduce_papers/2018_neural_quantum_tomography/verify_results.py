"""Verify measurement encoding, small-system training math and saved Figure 1 results."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import optax
from scipy.special import logsumexp

import measurements
import rbm
import tensorcircuit as tc


def reference_log_weight(parameters, bits):
    """Independent double-precision RBM marginal."""
    n = bits.shape[-1]
    parameters = np.asarray(parameters, dtype=np.float64)
    bias, hidden, weights = np.split(parameters, (n, 2 * n))
    fields = bits @ weights.reshape(n, n).T + hidden
    return bits @ bias + np.logaddexp(0.0, fields).sum(axis=-1)


def reference_rotation(n, site, use_y):
    """Dense small-system reference in the computational basis."""
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    second = np.array([[1, -1j], [1, 1j]]) / np.sqrt(2) if use_y else h
    return np.kron(
        np.kron(np.eye(2**site), np.kron(h, second)), np.eye(2 ** (n - site - 2))
    )


def verify_sampler():
    """Exercise the public sampler and its on-disk encoding at the target size."""
    K = tc.backend
    n = 20
    sampler = measurements.make_sampler()
    expected = np.eye(n, dtype=np.uint32)
    uniforms = 0.999 * expected + 0.001 * (1 - expected)
    tensors = measurements.w_tensors(np.zeros(n))
    circuit = tc.MPSCircuit(n, tensors=tensors, center_position=0)
    assert circuit.get_bond_dimensions() == [1] + [2] * (n - 1) + [1]
    bits = K.numpy(sampler(tensors, K.convert_to_tensor(uniforms.astype(np.float32))))
    assert np.array_equal(bits, expected)
    encoded = measurements.pack_bits(bits)
    assert encoded.dtype == np.uint32
    assert np.array_equal(encoded, 1 << np.arange(n - 1, -1, -1))

    rng = np.random.default_rng(8125)
    z = measurements.sample_basis(sampler, tensors, 20_000, rng)
    with TemporaryDirectory() as directory:
        archive = Path(directory) / "raw_data.npz"
        np.savez_compressed(archive, positive_z=z, singletons=encoded)
        with np.load(archive, allow_pickle=False) as saved:
            assert saved["singletons"].dtype == saved["positive_z"].dtype == np.uint32
            assert np.array_equal(
                measurements.unpack_bits(saved["singletons"], n), expected
            )
            assert np.array_equal(saved["positive_z"], z)
            singletons, counts = np.unique(saved["positive_z"], return_counts=True)
    assert np.array_equal(singletons, np.sort(encoded))
    np.testing.assert_allclose(counts / z.size, 1 / n, atol=0.008, rtol=0)

    phases = np.array([0.0, 0.9, -0.8])
    n = len(phases)
    shots = 32768
    samples = measurements.sample_bases(sampler, phases, shots, rng)
    psi = np.zeros(2**n, dtype=complex)
    psi[1 << np.arange(n - 1, -1, -1)] = np.exp(1j * phases) / np.sqrt(n)
    errors, moments = [], []
    for edge in range(n - 1):
        for axis in (0, 1):
            rotation = reference_rotation(n, edge, axis)
            probability = abs(rotation @ psi) ** 2
            row = samples[1 + 2 * edge + axis]
            empirical = np.bincount(row, minlength=2**n) / shots
            np.testing.assert_allclose(empirical, probability, atol=0.008, rtol=0)
            assert np.all(probability[row] > 1e-12)
            errors.append(float(np.max(abs(empirical - probability))))
            bits = measurements.unpack_bits(row, n).astype(int)
            focal = bits[:, edge : edge + 2]
            pair_only = bits.sum(axis=1) == focal.sum(axis=1)
            moment = float(np.mean((-1) ** focal[pair_only].sum(axis=1)))
            delta = phases[edge + 1] - phases[edge]
            reference = (np.cos(delta), np.sin(delta))[axis]
            np.testing.assert_allclose(moment, reference, atol=0.025, rtol=0)
            moments.append(moment)
    return {
        "singletons_round_tripped": 20,
        "z_shots": z.size,
        "z_counts_msb_first": counts[::-1].tolist(),
        "max_rotated_probability_error": max(errors),
        "xx_xy_parities": moments,
    }


def verify_phase_math():
    """Compare JAX scores and one update to full rotations and finite differences."""
    K = tc.backend
    rng = np.random.default_rng(518)
    n = 3
    size = n * (n + 2)
    amplitude, initial = rng.normal(0, 0.4, (2, size)).astype(np.float32)
    row = np.r_[np.arange(2**n, dtype=np.uint32), 0, 0, 1, 5].astype(np.uint32)
    streams = np.tile(row, (2 * n - 1, 1))
    terms, rotations, counts, _ = rbm.phase_data(streams, n)
    logits = rbm.log_weight(K.convert_to_tensor(amplitude), terms)
    components = rotations * K.exp((logits - K.max(logits, axis=1)[:, None]) / 2)
    phase = K.convert_to_tensor(initial)
    actual_scores = K.numpy(rbm.phase_scores(phase, terms, components))

    indices = np.arange(2**n)
    bits = (indices[:, None] >> np.arange(n - 1, -1, -1)) & 1
    operators = [
        reference_rotation(n, site, axis) for site in range(n - 1) for axis in (0, 1)
    ]
    scores, weights = [], []
    a = reference_log_weight(amplitude, bits)

    def loss(parameters, operator):
        wavefunction = np.exp((a + 1j * reference_log_weight(parameters, bits)) / 2)
        return -np.log(abs(operator @ wavefunction) ** 2)

    shift = 1e-5 * np.eye(size)
    for operator in operators:
        differences = [
            (loss(initial + step, operator) - loss(initial - step, operator)) / 2e-5
            for step in shift
        ]
        scores.append(np.stack(differences, axis=1))
        weights.extend(np.bincount(row, minlength=2**n))
    scores = np.concatenate(scores)
    weights = np.asarray(weights) / len(operators) / len(row)
    np.testing.assert_allclose(actual_scores, scores, atol=3e-4, rtol=3e-4)
    gradient = weights @ scores
    fisher = scores.T @ (weights[:, None] * scores)
    direction = np.linalg.solve(fisher + rbm.FISHER_DIAGONAL * np.eye(size), gradient)
    scale = rbm.PHASE_RATE / max(
        rbm.PHASE_RATE, np.sqrt(direction @ fisher @ direction)
    )
    expected = initial - scale * direction
    optimizer = optax.sgd(1.0)
    updated, _ = rbm.phase_step(
        phase, optimizer.init(phase), terms, components, counts, optimizer
    )
    native = K.numpy(updated)
    np.testing.assert_allclose(native, expected, atol=3e-4, rtol=3e-4)
    return {
        "phase_gradient_max_abs_error": float(np.max(abs(actual_scores - scores))),
        "phase_update_max_abs_error": float(np.max(abs(native - expected))),
    }


def preflight():
    """Run sampling and gradient checks before starting the expensive fits."""
    tc.set_backend("jax")
    result = verify_sampler()
    result.update(verify_phase_math())
    print(json.dumps(result, indent=2))


def verify_saved(output):
    """Recompute the published probabilities, overlaps and phase errors from the archive."""
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    protocol = metrics["protocol"]
    n = protocol["qubits"]
    assert n == 20
    assert protocol["seeds"] == [0, 1, 2]
    assert protocol["positive_shots"] == [50, 1000, 20000]
    assert protocol["phase_bases"] == 39 and protocol["phase_shots_per_basis"] == 6400
    count = len(protocol["seeds"])
    size = n * (n + 2)
    powers = 1 << np.arange(n - 1, -1, -1)
    singleton = np.eye(n)
    with np.load(output / "raw_data.npz", allow_pickle=False) as raw:
        positive, phased = raw["positive_z"], raw["phased_bases"]
        amplitude, phase = raw["amplitude_parameters"], raw["phase_parameters"]
    assert positive.shape == (count, 20000) and phased.shape == (count, 39, 6400)
    assert positive.dtype == phased.dtype == np.uint32
    assert amplitude.shape == (count, 4, size) and phase.shape == (count, size)
    assert np.isfinite(amplitude).all() and np.isfinite(phase).all()
    assert np.array_equal(np.unique(positive), np.sort(powers))
    assert np.array_equal(np.unique(phased[:, 0]), np.sort(powers))
    for site in range(n - 1):
        mask = ((1 << n) - 1) ^ (int(powers[site]) | int(powers[site + 1]))
        outcomes = phased[:, 1 + 2 * site : 3 + 2 * site]
        assert (outcomes < 2**n).all()
        assert np.isin(outcomes & mask, np.r_[0, powers]).all()

    partitions = np.full((count, 4), -np.inf)
    block = 32768
    for start in range(0, 2**n, block):
        states = np.arange(start, start + block)
        bits = ((states[:, None] & powers) != 0).astype(float)
        for index in np.ndindex(partitions.shape):
            total = logsumexp(reference_log_weight(amplitude[index], bits))
            partitions[index] = np.logaddexp(partitions[index], total)

    def compare(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-5)

    records = metrics["records"]
    assert [record["seed"] for record in records] == protocol["seeds"]
    for run, record in enumerate(records):
        assert [row["shots"] for row in record["positive"]] == protocol[
            "positive_shots"
        ]
        for fit, row in enumerate([*record["positive"], record["complex"]]):
            probability = np.exp(
                reference_log_weight(amplitude[run, fit], singleton)
                - partitions[run, fit]
            )
            compare(row["populations"], probability)
            compare(row["single_excitation_mass"], probability.sum())
            coefficients = np.sqrt(probability / n)
            if fit < 3:
                compare(row["overlap"], coefficients.sum())
                continue
            seed = record["seed"]
            target_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
            target = target_rng.uniform(-np.pi, np.pi, n)
            target = np.angle(np.exp(1j * (target - target[0])))
            learned = reference_log_weight(phase[run], singleton) / 2
            learned = np.angle(np.exp(1j * (learned - learned[0])))
            error = np.angle(np.exp(1j * (learned - target)))
            compare(row["target_relative_phase"], target)
            compare(row["learned_relative_phase"], learned)
            compare(row["phase_rms_radians"], np.sqrt(np.mean(error**2)))
            compare(row["overlap"], abs(coefficients @ np.exp(1j * error)))
            compare(row["z_only_overlap"], abs(coefficients @ np.exp(-1j * target)))
    print(
        f"Verified encoding, populations, overlaps and relative phases for {count} seeds."
    )


def main():
    """Select independent published-output verification or numerical preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs"
    )
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        verify_saved(args.output)


if __name__ == "__main__":
    main()
