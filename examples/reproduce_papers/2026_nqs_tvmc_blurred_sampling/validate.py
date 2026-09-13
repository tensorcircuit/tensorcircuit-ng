"""
Independent small-system and nodal checks for the Figure 5 reproduction.

All exact sums here are validation references, never the MC evolution path.
"""

import json
from pathlib import Path

import numpy as np
from main import exact_operators, exact_trajectory, initial_parameters
from physics import (
    SpinProblem,
    inner_product_moments,
    reweighted_moments,
    schmitt_velocity,
    standard_moments,
)
import tensorcircuit as tc

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def check_problem(nsites):
    """Check local H, exact initial state, TC circuits, and analytic derivatives."""
    k = tc.backend
    problem = SpinProblem(nsites)
    theta = initial_parameters(problem)
    basis, h, observable, hfull, indices, gauge = exact_operators(problem)
    a, d, local_hpsi, _ = problem.local_batch(theta, basis, 0.5)
    np.testing.assert_allclose(k.numpy(local_hpsi), k.numpy(h @ a), atol=3e-14)
    psi = a / k.norm(a)
    # Embed the Marshall-basis wave function into the physical spin basis.
    embedding = k.transpose(k.onehot(indices, 2**nsites))
    full = embedding @ (gauge * psi)
    circuit = tc.Circuit(nsites, inputs=full)
    np.testing.assert_allclose(k.numpy(circuit.state()), k.numpy(full), atol=3e-14)
    if nsites == 1:
        measured = circuit.expectation_ps(z=[0])
    else:
        measured = (
            sum(circuit.expectation_ps(**{axis: [0, 1]}) for axis in ["x", "y", "z"])
            / 4
        )
    expected = k.sum(k.conj(psi) * (observable @ psi))
    np.testing.assert_allclose(k.numpy(measured), k.numpy(expected), atol=3e-14)
    # A dense exact gate is legitimate here: it is used only as an independent reference.
    t = 0.37
    circuit.any(*range(nsites), unitary=k.expm(-1j * t * hfull))
    _, states = exact_trajectory(problem, theta, np.array([t]))
    evolved = k.numpy(circuit.state())
    np.testing.assert_allclose(
        evolved,
        k.numpy(embedding @ (gauge * k.convert_to_tensor(states[0]))),
        atol=3e-14,
    )
    ad = k.vmap(
        lambda x: k.grad(lambda p: k.real(problem.amplitude(p, x)))(theta)
        + 1j * k.grad(lambda p: k.imag(problem.amplitude(p, x)))(theta)
    )(basis)
    np.testing.assert_allclose(k.numpy(ad), k.numpy(d), atol=3e-14)
    report = {"local_H_max_error": float(np.max(np.abs(k.numpy(local_hpsi - h @ a))))}
    # Grouping repeated samples must leave all MC moments and the SNR solve
    # unchanged, including when some allowed configurations were not sampled.
    repeated = k.concat([k.tile(basis[:1], [7, 1]), k.tile(basis[1:2], [3, 1])])
    compressed, counts = problem.compress_samples(repeated)
    for q in [0.0, 0.5]:
        full_values = problem.local_batch(theta, repeated, q)
        compact_values = problem.local_batch(theta, compressed, q)
        if q == 0:
            full_moments = standard_moments(*full_values[:3])
            compact_moments = standard_moments(*compact_values[:3], counts)
        else:
            full_moments = reweighted_moments(*full_values)
            compact_moments = reweighted_moments(*compact_values, counts)
        for actual, compact in zip(full_moments[:2], compact_moments[:2]):
            np.testing.assert_allclose(k.numpy(actual), k.numpy(compact), atol=3e-14)
        full_velocity = schmitt_velocity(*full_moments[:3])
        compact_velocity = schmitt_velocity(*compact_moments[:3], counts=counts)
        np.testing.assert_allclose(
            k.numpy(full_velocity), k.numpy(compact_velocity), atol=3e-13
        )
    if nsites == 4:
        ground_problem = SpinProblem(4, delta_j=0.0)
        _, h0, _, _, _, _ = exact_operators(ground_problem)
        energies, vectors = k.eigh(h0)
        energy = k.real(k.sum(k.conj(psi) * (h0 @ psi)))
        fidelity = k.abs(k.sum(k.conj(vectors[:, 0]) * psi)) ** 2
        np.testing.assert_allclose(k.numpy(energies[0]), -2.0, atol=3e-14)
        with np.load(Path(__file__).resolve().parent / "initial_state.npz") as data:
            order = [
                np.flatnonzero(np.all(data["basis"] == row, axis=1))[0]
                for row in k.numpy(basis)
            ]
            reference = data["amplitudes"][order]
            np.testing.assert_allclose(
                k.numpy(psi), reference / np.linalg.norm(reference), atol=3e-14
            )
        report.update(
            ground_energy_error=float(k.numpy(energy - energies[0])),
            ground_infidelity=float(k.numpy(1 - fidelity)),
        )
    else:
        times = np.linspace(0, 2, 101)
        exact, _ = exact_trajectory(problem, theta, times)
        np.testing.assert_allclose(exact, -np.sin(2 * times), atol=3e-14)
    return report


def check_nodal_estimator(nsites):
    """Use an actual zero amplitude for one spin and an RBM cosh node."""
    k = tc.backend
    problem = SpinProblem(nsites)
    basis, h, _, _, _, _ = exact_operators(problem)
    if nsites == 1:
        theta = k.convert_to_tensor([1.0, 0.0, 0.0, 0.0])
    else:
        weights = k.convert_to_tensor([0.2, -0.1, 0.35, -0.2])
        bias = -k.sum(weights * basis[0])
        theta = k.concat(
            [
                weights,
                k.reshape(bias, [1]),
                k.convert_to_tensor([0.0, 0.0, 0.0, 0.0, np.pi / 2]),
            ]
        )
    a, d, hp, density = problem.local_batch(theta, basis, 0.5)
    np.testing.assert_allclose(k.numpy(hp), k.numpy(h @ a), atol=3e-14)
    ad = k.vmap(
        lambda x: k.grad(lambda p: k.real(problem.amplitude(p, x)))(theta)
        + 1j * k.grad(lambda p: k.imag(problem.amplitude(p, x)))(theta)
    )(basis)
    np.testing.assert_allclose(k.numpy(d), k.numpy(ad), atol=3e-14)
    exact_s, exact_f = inner_product_moments(a, d, hp)
    adjacency = k.cast(k.abs(h) > 0, "float64") * (
        1 - k.eye(h.shape[0], dtype="float64")
    )
    degree = k.sum(adjacency, axis=0)
    kernel = (
        0.5 * k.eye(h.shape[0], dtype="float64") + 0.5 * adjacency / degree[None, :]
    )
    np.testing.assert_allclose(k.numpy(k.sum(kernel, axis=0)), 1.0, atol=3e-14)
    np.testing.assert_allclose(
        k.numpy(density), k.numpy(kernel @ k.abs(a) ** 2), atol=3e-14
    )
    r = density / k.sum(k.abs(a) ** 2)
    factors = k.sqrt(r * a.shape[0])
    estimated_s, estimated_f, _, _ = reweighted_moments(
        a * factors, d * factors[:, None], hp * factors, density
    )
    np.testing.assert_allclose(
        k.numpy(estimated_s), k.numpy(exact_s), atol=3e-13, rtol=3e-13
    )
    np.testing.assert_allclose(
        k.numpy(estimated_f), k.numpy(exact_f), atol=3e-13, rtol=3e-13
    )
    report = {
        "smallest_amplitude": float(np.min(np.abs(k.numpy(a)))),
        "blur_S_max_error": float(np.max(np.abs(k.numpy(estimated_s - exact_s)))),
        "blur_F_max_error": float(np.max(np.abs(k.numpy(estimated_f - exact_f)))),
        "smallest_blur_probability": float(np.min(k.numpy(r))),
    }
    if nsites == 1:
        # Sampling only |0> has exactly zero covariance but the true derivative in
        # the missing |1> direction is finite. No masked log-score reference.
        repeated = k.convert_to_tensor([[1]] * 128)
        ap = problem.amplitudes(theta, repeated)
        dp = problem.derivatives(theta, repeated)
        hpp = problem.local_batch(theta, repeated, 0.0)[2]
        ss, ff, _, _ = standard_moments(ap, dp, hpp)
        np.testing.assert_array_equal(k.numpy(ss), np.zeros((4, 4)))
        np.testing.assert_array_equal(k.numpy(ff), np.zeros(4))
        np.testing.assert_allclose(k.numpy(exact_s[1, 1]), 1.0, atol=3e-14)
        np.testing.assert_allclose(k.numpy(exact_f[1]), 1j, atol=3e-14)
        report.update(
            standard_missing_S=float(k.numpy(exact_s[1, 1]).real),
            standard_missing_abs_F=float(k.numpy(k.abs(exact_f[1]))),
        )
    # Independent chains and replicas quantify MCMC uncertainty of S and F.
    estimates_s, estimates_f, corrected_s, corrected_f = [], [], [], []
    nrep, samples = 32, 4096
    estimate = k.jit(lambda c, rng: problem.estimate(theta, c, rng, 0.5))
    burn = k.jit(lambda c, rng: problem.metropolis(theta, c, rng, 128))
    for seed in range(700, 700 + nrep):
        chains, key = burn(
            problem.initial_chains(samples), k.set_random_state(seed, get_only=True)
        )
        ss, ff, _, weights, counts, _, _ = estimate(chains, key)
        # Eq. (11), with normalized weights p/r (norm is known only here).
        factor = (
            samples
            * float(k.numpy(k.sum(counts * weights) / samples)) ** 2
            / (samples - 1)
        )
        estimates_s.append(k.numpy(ss))
        estimates_f.append(k.numpy(ff))
        corrected_s.append(factor * k.numpy(ss))
        corrected_f.append(factor * k.numpy(ff))
    for name, estimates, reference in [
        ("S", estimates_s, exact_s),
        ("F", estimates_f, exact_f),
        ("Eq11_S", corrected_s, exact_s),
        ("Eq11_F", corrected_f, exact_f),
    ]:
        values, reference = np.array(estimates), k.numpy(reference)
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(nrep)
        residual = np.abs(mean - reference)
        # A simultaneous 5-SEM check over at most 100 components, with floating
        # point tolerance for exactly vanishing variances.
        if np.any(residual > 5 * sem + 1e-12):
            raise AssertionError(f"Nodal {name} differs by more than five replica SEM")
        report[f"MC_{name}_max_error"] = float(residual.max())
        report[f"MC_{name}_max_sem"] = float(sem.max())
    report["replicas"] = nrep
    report["samples_per_replica"] = samples
    return report


def main():
    """Validate Hamiltonians, derivatives, TC circuits, and nodal MC estimates."""
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    report = {
        "physics": {str(n): check_problem(n) for n in [1, 4]},
        "nodes": {str(n): check_nodal_estimator(n) for n in [1, 4]},
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
