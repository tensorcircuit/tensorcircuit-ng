"""Check saved figure data against matrix-exponential solutions of both models."""

import numpy as np
from main import DEPHASING, HOPPING, OUTPUT, SITES, evolve, initial_populations
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply
from scipy.special import xlogy

import tensorcircuit as tc


def references(times):
    identity = np.eye(SITES)
    upper = np.diag(np.ones(SITES - 1), 1)
    h = HOPPING * (upper + upper.T)
    # Row-major vectorization. Construct each onsite jump independently.
    generator = -1j * (np.kron(h, identity) - np.kron(identity, h.T))
    for site in identity:
        projector = np.outer(site, site)
        generator += DEPHASING * (
            np.kron(projector, projector)
            - (np.kron(projector, identity) + np.kron(identity, projector)) / 2
        )
    initial = initial_populations()
    vectors = np.stack([np.diag(p).ravel() for p in initial], axis=1)
    propagated = expm_multiply(
        generator, vectors, start=times[0], stop=times[-1], num=len(times)
    )
    states = propagated.transpose(2, 0, 1).reshape(2, len(times), SITES, SITES)

    bonds = np.diff(identity, axis=0)
    rates, modes = np.linalg.eigh(-2 * HOPPING**2 / DEPHASING * (bonds.T @ bonds))
    walk = ((initial @ modes)[:, None, :] * np.exp(np.outer(times, rates))) @ modes.T
    return states, walk


def crossing(times, distance):
    difference = PchipInterpolator(times, distance[0] - distance[1])
    return brentq(difference, times[0], times[-1])


def decay_rates(times, occupations):
    deviation = np.abs(occupations - 1 / SITES)
    return -np.polyfit(times, np.log(deviation.T), 1)[0]


def main():
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    with np.load(OUTPUT / "curves.npz") as data:
        times = data["times"]
        np.testing.assert_allclose(times, np.linspace(0, 70, len(times)))
        states, walk = references(times)
        padded = evolve(times, initial_populations())
        leakage = max(
            np.abs(padded[:, :, SITES:, :]).max(), np.abs(padded[:, :, :, SITES:]).max()
        )
        print(f"Largest entry outside the nine-site block = {leakage:.3g}")
        np.testing.assert_allclose(leakage, 0, atol=2e-11, rtol=0)
        physical = padded[:, :, :SITES, :SITES]
        np.testing.assert_allclose(physical, states, atol=2e-11, rtol=0)
        print(
            f"Full density-matrix max absolute error = {np.max(np.abs(physical - states)):.3g}"
        )
        exact = data["entropy_deficit"]
        effective = data["effective_entropy_deficit"]
        populations = np.diagonal(states, axis1=-2, axis2=-1).real
        for key, reference in (
            ("populations", populations),
            ("effective_populations", walk),
        ):
            error = np.max(np.abs(data[key] - reference))
            print(f"{key} max absolute error = {error:.3g}")
            np.testing.assert_allclose(data[key], reference, atol=2e-11, rtol=0)
        for result, spectrum in (
            (exact, np.linalg.eigvalsh(states)),
            (effective, walk),
        ):
            spectrum = np.maximum(spectrum, 0)
            reference = np.log(SITES) + xlogy(spectrum, spectrum).sum(axis=-1)
            np.testing.assert_allclose(result, reference, atol=2e-11, rtol=0)
            np.testing.assert_allclose(result[:, 0], np.log([9, 9 / 8]), atol=1e-13)
        print(
            f"Entropy crossing Jt: Lindblad {crossing(times, exact):.5f}, "
            f"effective {crossing(times, effective):.5f}"
        )
        # A reflection-odd slow mode vanishes at the center of an odd chain.
        fit = (times >= 50) & (times <= 68)
        rate = 2 * HOPPING**2 / DEPHASING
        slow, even = 2 * rate * (1 - np.cos(np.pi * np.array([1, 2]) / SITES))
        for site in (1, 3, 5):
            reference = decay_rates(times[fit], populations[:, fit, site - 1])
            for key, expected in (
                ("populations", reference),
                ("effective_populations", [even, even if site == 5 else slow]),
            ):
                decay = decay_rates(times[fit], data[key][:, fit, site - 1])
                np.testing.assert_allclose(decay, expected, rtol=0.01)
                print(
                    f"{key}, site {site}: decay rates / J (localized, spread) = {decay}"
                )


if __name__ == "__main__":
    main()
