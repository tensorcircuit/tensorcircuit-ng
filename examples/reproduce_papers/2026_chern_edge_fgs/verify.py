"""Independent sector, Gaussian, derivative, sampling and integration checks."""

import argparse
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm
import tencirpauli as tcp
import tensorcircuit as tc

if __package__:
    from .main import (
        FermionPEPS,
        Hofstadter,
        MonteCarlo,
        born_draw,
        enlarge_bond,
        gaussian_reference,
        determinant,
        measure,
        sr_cg_solve,
        sr_moments_solve,
    )
else:
    from main import (
        FermionPEPS,
        Hofstadter,
        MonteCarlo,
        born_draw,
        enlarge_bond,
        gaussian_reference,
        determinant,
        measure,
        sr_cg_solve,
        sr_moments_solve,
    )


def gauge_vectors(peps, theta):
    """Analytic virtual GL(D/2) x GL(D/2), global and fixed-number gauge tangents."""
    K = tc.backend
    output, source, columns, signs = [], [], [], []
    lookups = [dict(zip(c, range(len(c)))) for c in peps.coordinates]
    column = 0
    for i, j, ai, aj in peps.bonds:
        for a in range(peps.bond_dim):
            for b in range(peps.bond_dim):
                if (a - b) % 2:
                    continue
                for site, axis, old, new, sign in ((i, ai, a, b, 1), (j, aj, b, a, -1)):
                    for k, coordinate in enumerate(peps.coordinates[site]):
                        if coordinate[axis] == new:
                            changed = list(coordinate)
                            changed[axis] = old
                            output.append(peps.offsets[site] + k)
                            source.append(
                                peps.offsets[site] + lookups[site][tuple(changed)]
                            )
                            columns.append(column)
                            signs.append(sign)
                column += 1
    vectors = K.scatter(
        K.zeros((peps.nparams, column), dtype="complex128"),
        K.convert_to_tensor(list(zip(output, columns))),
        theta[K.convert_to_tensor(source)] * K.convert_to_tensor(signs),
    )
    physical = K.convert_to_tensor([c[0] for site in peps.coordinates for c in site])
    return K.concat([vectors, theta[:, None], (theta * physical)[:, None]], axis=1)


def gauge_projector(peps, theta):
    """Independent explicit projector, never constructed by production SR."""
    K = tc.backend
    u, singular, _, _ = K.svd(gauge_vectors(peps, theta))
    basis = u * K.cast(K.real(singular) > K.real(singular[0]) * 1e-10, "complex128")
    return K.eye(peps.nparams, dtype="complex128") - basis @ K.adjoint(basis)


def sr_solve(scores, energies, weights, projector=None, regulator=1e-8):
    """Dense sampled reference, also used to validate streamed sufficient statistics."""
    K = tc.backend
    mean_score = K.sum(weights[:, None] * scores, axis=0)
    mean_energy = K.sum(weights * energies)
    centered = scores - mean_score
    metric = K.adjoint(centered) @ (weights[:, None] * centered)
    force = K.adjoint(centered) @ (weights * (energies - mean_energy))
    if projector is None:
        velocity, residual = sr_moments_solve(metric, force, regulator)
    else:
        projected = projector @ metric @ projector
        projected = (projected + K.adjoint(projected)) / 2
        rhs = projector @ force
        identity = K.eye(metric.shape[0], dtype="complex128")
        system = projected + regulator * projector + identity - projector
        velocity = projector @ K.solve(system, rhs, assume_a="pos")
        residual = K.norm(metric @ velocity - force) ** 2 / K.norm(force) ** 2
    return velocity, mean_energy, residual


def sector_reference(problem, potential=0.0):
    """Use TenCirPauli's native restricted operator without rebuilding CAR signs."""
    space = tcp.OperatorSpace(fermions=problem.nsites)
    number = tcp.AdditiveCharge(
        space, name="N", fermions={i: 1 for i in range(problem.nsites)}
    )
    sector = tcp.ChargeSector(((number, problem.particles),))
    operator = problem.operator.add(problem.pin_operator.scale(potential))
    return sector.basis_states(), operator.restrict_charge(sector).dense()


def swap_network_reference(peps, theta, occupation):
    """Explicit virtual-bond sum with individual physical/virtual swap factors."""
    tensors = [np.asarray(tc.backend.numpy(t)) for t in peps.tensors(theta)]
    amplitude = 0j
    for bonds in itertools.product(range(peps.bond_dim), repeat=len(peps.bonds)):
        indices = [[0, 0, 0, 0] for _ in tensors]
        for value, (i, j, ai, aj) in zip(bonds, peps.bonds):
            indices[i][ai - 1] = value
            indices[j][aj - 1] = value
        value = 1 + 0j
        for i, tensor in enumerate(tensors):
            value *= tensor[(occupation[i],) + tuple(indices[i])]
        # Draw each physical leg toward the upper-left boundary of the grid.
        for i, n in enumerate(occupation):
            y, x = divmod(i, peps.columns)
            for k in range(x):
                crossed = y * peps.columns + k
                value *= (-1) ** (int(n) * indices[crossed][3])
        amplitude += value
    return amplitude


def check_kernels(rows=2, columns=3, bond_dim=2):
    K = tc.backend
    peps = FermionPEPS(rows, columns, bond_dim)
    problem = Hofstadter(peps, 2 * peps.nsites // 3)
    theta = peps.random_parameters(K.get_random_state(71))
    basis, hamiltonian = sector_reference(problem)
    basis = K.cast(K.convert_to_tensor(basis), "int64")
    amplitude, scores = K.jit(peps.batch_scores)(theta, basis)
    psi, jac = [
        np.asarray(K.numpy(x)) for x in (amplitude, amplitude[:, None] * scores)
    ]
    local = np.asarray(K.numpy(K.jit(problem.local_batch)(theta, basis)))
    np.testing.assert_allclose(local * psi, hamiltonian @ psi, rtol=1e-10, atol=1e-12)
    if rows * columns <= 6 and bond_dim == 2:
        explicit = np.array(
            [swap_network_reference(peps, theta, np.asarray(s)) for s in basis]
        )
        np.testing.assert_allclose(psi, explicit, rtol=1e-11, atol=1e-13)
    # Direct circuit-state MVP checks both the native restricted basis order and
    # the mapped Pauli convention used by local-energy evaluation.
    full = np.zeros(2**peps.nsites, dtype=np.complex128)
    powers = 2 ** np.arange(peps.nsites - 1, -1, -1)
    labels = np.asarray(basis) @ powers
    full[labels] = psi
    circuit = tc.Circuit(peps.nsites, inputs=K.convert_to_tensor(full))
    mvp = tcp.backend_mvp(problem.pauli.backend_mvp_plan(), backend=K)
    acted = np.asarray(K.numpy(mvp(circuit.state())))
    np.testing.assert_allclose(acted[labels], hamiltonian @ psi, rtol=1e-11, atol=1e-13)
    direction = peps.random_parameters(K.get_random_state(97))
    eps = 1e-6
    amplitudes = K.jit(peps.batch_amplitude)
    finite = np.asarray(
        K.numpy(
            (
                amplitudes(theta + eps * direction, basis)
                - amplitudes(theta - eps * direction, basis)
            )
            / (2 * eps)
        )
    )
    np.testing.assert_allclose(
        finite, jac @ np.asarray(direction), rtol=1e-7, atol=1e-10
    )
    vectors = np.asarray(K.numpy(gauge_vectors(peps, theta)))
    centered = np.asarray(scores) - np.mean(np.asarray(scores), axis=0)
    gauge_error = np.linalg.norm(centered @ vectors) / np.linalg.norm(centered)
    np.testing.assert_allclose(centered @ vectors, 0.0, atol=2e-10)
    projector = K.jit(lambda p: gauge_projector(peps, p))(theta)
    np.testing.assert_allclose(projector @ vectors, 0.0, atol=2e-10)
    weights = K.abs(amplitude) ** 2 / K.sum(K.abs(amplitude) ** 2)
    velocity, _, residual = sr_solve(
        scores, K.convert_to_tensor(local), weights, projector
    )
    ridge_velocity, _, _ = sr_solve(scores, K.convert_to_tensor(local), weights)
    np.testing.assert_allclose(
        jac @ np.asarray(ridge_velocity),
        jac @ np.asarray(velocity),
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        vectors.conj().T @ np.asarray(ridge_velocity), 0.0, atol=2e-7
    )
    cg_velocity, _, _, linear_residual, _ = K.jit(sr_cg_solve)(
        scores, K.convert_to_tensor(local), weights
    )
    np.testing.assert_allclose(
        jac @ np.asarray(cg_velocity), jac @ np.asarray(velocity), rtol=3e-6, atol=3e-8
    )
    np.testing.assert_allclose(
        vectors.conj().T @ np.asarray(cg_velocity), 0.0, atol=2e-7
    )
    np.testing.assert_allclose(linear_residual, 0.0, atol=1e-7)
    np.testing.assert_allclose(vectors.conj().T @ np.asarray(velocity), 0.0, atol=2e-8)
    return {
        "shape": [rows, columns],
        "D": bond_dim,
        "parameters": peps.nparams,
        "local_energy_max_error": float(
            np.max(np.abs(local * psi - hamiltonian @ psi))
        ),
        "gauge_null_error": float(gauge_error),
        "sr_residual": float(residual),
    }


def check_cg():
    """Weighted complex ridge equations, rank deficiency, and zero-force behavior."""
    K = tc.backend
    rng = np.random.default_rng(102)
    errors = []
    for samples, parameters in ((9, 24), (41, 12)):
        scores = rng.normal(size=(samples, parameters)) + 1j * rng.normal(
            size=(samples, parameters)
        )
        scores[:, -1] = scores[:, 0]
        energies = rng.normal(size=samples) + 1j * rng.normal(size=samples)
        weights = rng.uniform(size=samples)
        weights[0] = 0.0
        weights /= weights.sum()
        centered = scores - weights @ scores
        force = centered.conj().T @ (weights * (energies - weights @ energies))
        metric = centered.conj().T @ (weights[:, None] * centered)
        reference = np.linalg.solve(metric + 1e-3 * np.eye(parameters), force)
        values = tuple(K.convert_to_tensor(a) for a in (scores, energies, weights))
        solve = K.jit(lambda o, e, w: sr_cg_solve(o, e, w, 1e-3, 1e-10))
        velocity, _, _, residual, iterations = solve(*values)
        np.testing.assert_allclose(velocity, reference, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

        def loss(scale):
            direction = sr_cg_solve(
                values[0] * scale, values[1], values[2], 1e-3, 1e-10
            )[0]
            return K.real(K.sum(K.conj(direction) * direction))

        derivative = K.jit(K.grad(loss))(K.cast(1.0, "float64"))
        epsilon = 1e-5
        norms = []
        for scale in (1 + epsilon, 1 - epsilon):
            direction = np.linalg.solve(
                scale**2 * metric + 1e-3 * np.eye(parameters), scale * force
            )
            norms.append(np.vdot(direction, direction).real)
        np.testing.assert_allclose(
            derivative, (norms[0] - norms[1]) / (2 * epsilon), rtol=2e-6, atol=1e-8
        )
        failed = sr_cg_solve(*values, regulator=1e-3, tolerance=1e-10, maxiter=1)
        assert float(failed[3]) > 1e-5
        zero = solve(values[0], K.zeros((samples,), dtype="complex128"), values[2])
        np.testing.assert_allclose(zero[0], 0.0, atol=0)
        np.testing.assert_allclose(zero[3], 0.0, atol=0)
        errors.append(
            {
                "N": samples,
                "P": parameters,
                "iterations": int(iterations),
                "relative_residual": float(residual),
            }
        )
    return errors


def check_cg_estimate():
    """CG dispatch and RK stages agree with gauge-projected SR on exact scores."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2, exact_optimizer="greedy")
    problem = Hofstadter(peps, 4)
    cg = MonteCarlo(problem, chains=32, solver="cg", regulator=1e-3)
    dense = MonteCarlo(problem, chains=32, solver="dense", regulator=1e-3)
    theta = peps.random_parameters(K.get_random_state(104))
    states = cg.initial_chains(K.get_random_state(105))
    key = K.get_random_state(106)
    estimate = K.jit(cg.estimate)(theta, states, key)
    reference = K.jit(dense.estimate)(theta, states, key)
    for value, expected in zip(estimate, reference):
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-7)
    step = K.jit(lambda p, s, k: cg.rk4_step(p, s, k, 0.001))(theta, states, key)
    expected = K.jit(lambda p, s, k: dense.rk4_step(p, s, k, 0.001))(theta, states, key)
    for value, reference in zip(step, expected):
        np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-7)
    failed = MonteCarlo(problem, chains=32, solver="cg", regulator=1e-3, cg_maxiter=1)
    rejected = K.jit(failed.estimate)(theta, states, key)
    assert np.isnan(np.asarray(rejected[0])).all()
    return {"estimate_and_rk4_agree": True, "nonconverged_direction_rejected": True}


def check_batched_moments():
    """Contraction chunking, including the padded final chunk, preserves moments."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    sampler = MonteCarlo(problem, contraction_batch_size=4)
    basis, _ = sector_reference(problem)
    states = K.cast(K.convert_to_tensor(basis), "int64")
    theta = peps.random_parameters(K.get_random_state(83))
    scores, energies = K.jit(sampler.sampled_moments)(theta, states, -1.0)
    np.testing.assert_allclose(
        scores, K.jit(peps.batch_scores)(theta, states)[1], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        energies,
        K.jit(problem.local_batch)(theta, states, -1.0),
        rtol=1e-10,
        atol=1e-10,
    )
    amplitudes = K.jit(peps.batch_amplitude)(theta, states)
    probabilities = K.abs(amplitudes) ** 2
    for nonzero in (len(basis), 5):
        weights = K.where(K.arange(len(basis)) < nonzero, probabilities, 0.0)
        weights = weights / K.sum(weights)
        metric, force, energy, values = K.jit(sampler.sr_statistics)(
            theta, states, weights, -1.0
        )
        centered = scores - K.sum(weights[:, None] * scores, axis=0)
        expected_energy = K.sum(weights * energies)
        np.testing.assert_allclose(
            metric,
            K.adjoint(centered) @ (weights[:, None] * centered),
            rtol=1e-11,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            force,
            K.adjoint(centered) @ (weights * (energies - expected_energy)),
            rtol=1e-11,
            atol=1e-10,
        )
        np.testing.assert_allclose(energy, expected_energy, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(values, energies, rtol=1e-11, atol=1e-11)
    return {"configurations": len(basis), "contraction_batch_size": 4}


def check_sampling():
    """Born frequencies and moments are checked against independent exact sums."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    theta = peps.random_parameters(K.get_random_state(41))
    sampler = MonteCarlo(problem, chains=256, sweeps=4)
    chains = sampler.initial_chains(K.get_random_state(43))
    chains, key, _ = K.jit(lambda t, s, k: sampler.advance(t, s, k, 512))(
        theta, chains, K.get_random_state(47)
    )

    def draw(carry, _):
        states, rng = carry
        states, rng, _ = sampler.advance(theta, states, rng, 4 * len(peps.bonds))
        return (states, rng), states

    _, samples = K.jit(lambda s, k: K.jaxy_scan(draw, (s, k), K.arange(128)))(
        chains, key
    )
    samples = samples.reshape(-1, peps.nsites)
    samples = np.asarray(K.numpy(samples))
    np.testing.assert_array_equal(samples.sum(axis=1), 4)
    basis, _ = sector_reference(problem)
    amplitudes = np.asarray(
        K.numpy(peps.batch_amplitude(theta, K.convert_to_tensor(basis)))
    )
    probabilities = np.abs(amplitudes) ** 2
    probabilities /= probabilities.sum()
    density = probabilities @ basis
    block_means = samples.reshape(128, 256, 6).mean(axis=0)
    stderr = block_means.std(axis=0, ddof=1) / np.sqrt(256)
    np.testing.assert_allclose(samples.mean(axis=0), density, atol=0.025)
    np.testing.assert_array_less(
        np.abs(samples.mean(axis=0) - density), 6 * stderr + 0.002
    )
    return {
        "samples": int(len(samples)),
        "density_max_error": float(np.max(np.abs(samples.mean(axis=0) - density))),
    }


class ExactMoments(MonteCarlo):
    """Deterministic validation of the production RK4 integrator on a tiny lattice."""

    def __init__(self, problem):
        super().__init__(problem, chains=2, regulator=1e-10)
        self.basis, self.hamiltonian = sector_reference(problem)
        self.basis = tc.backend.cast(tc.backend.convert_to_tensor(self.basis), "int64")

    def estimate(self, theta, chains, key, potential=0.0):
        K = tc.backend
        amplitude, scores = self.peps.batch_scores(theta, self.basis)
        energies = self.problem.local_batch(theta, self.basis, potential)
        weights = K.abs(amplitude) ** 2 / K.sum(K.abs(amplitude) ** 2)
        velocity, energy, residual = sr_solve(
            scores, energies, weights, gauge_projector(self.peps, theta), self.regulator
        )
        diagnostic = K.zeros((4,), dtype="float64")
        diagnostic = K.scatter(
            diagnostic,
            K.convert_to_tensor([[0], [2]]),
            K.stack([K.real(energy), residual]),
        )
        return velocity, diagnostic, chains, key


def check_preparation():
    """Check parity-preserving growth and the sign of the Euler VMC update."""
    K = tc.backend
    peps, larger = FermionPEPS(2, 3, 2), FermionPEPS(2, 3, 4)
    problem = Hofstadter(peps, 4)
    engine = ExactMoments(problem)
    theta = peps.random_parameters(K.get_random_state(141))
    grown = enlarge_bond(theta, peps, larger, noise=0.0)
    before = np.asarray(K.jit(peps.batch_amplitude)(theta, engine.basis))
    after = np.asarray(K.jit(larger.batch_amplitude)(grown, engine.basis))
    np.testing.assert_allclose(after, before, rtol=1e-11, atol=1e-14)
    chains = engine.initial_chains(K.get_random_state(143))
    key = K.get_random_state(149)
    final, _, _, _ = K.jit(
        lambda p, s, k: engine.trajectory(p, s, k, 0.001, 1, -1.0, True)
    )(theta, chains, key)
    psi = np.asarray(K.jit(peps.batch_amplitude)(final, engine.basis))
    _, h = sector_reference(problem, -1.0)
    initial_energy = np.vdot(before, h @ before).real / np.vdot(before, before).real
    final_energy = np.vdot(psi, h @ psi).real / np.vdot(psi, psi).real
    np.testing.assert_array_less(final_energy, initial_energy)
    return {
        "embedding_max_error": float(np.max(abs(after - before))),
        "preparation_energy_change": float(final_energy - initial_energy),
    }


def check_integrator():
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    engine = ExactMoments(problem)
    theta = peps.random_parameters(K.get_random_state(121))
    psi = np.array(K.numpy(peps.batch_amplitude(theta, engine.basis)), copy=True)
    psi /= np.linalg.norm(psi)
    exact = expm(-0.04j * engine.hamiltonian) @ psi
    errors = []
    for steps in (4, 8):
        trajectory = K.jit(
            lambda p, s, k: engine.trajectory(p, s, k, 0.04 / steps, steps)
        )
        final, _, _, _ = trajectory(
            theta,
            engine.initial_chains(K.get_random_state(131)),
            K.get_random_state(137),
        )
        state = np.array(K.numpy(peps.batch_amplitude(final, engine.basis)), copy=True)
        state /= np.linalg.norm(state)
        state *= np.exp(-1j * np.angle(np.vdot(exact, state)))
        errors.append(float(np.linalg.norm(state - exact)))
    np.testing.assert_allclose(errors, 0.0, atol=2e-5)
    np.testing.assert_array_less(errors[1], errors[0] + 1e-8)
    return {"dt": [0.01, 0.005], "state_norm_errors": errors}


def born_draw_fgs(alpha, randoms):
    K = tc.backend
    n = randoms.shape[0]

    def measure(carry, i):
        current, occupation = carry
        state = tc.FGSSimulator(n, alpha=current)
        value = state.cond_measure(i, randoms[i])
        occupation = K.scatter(occupation, K.reshape(i, (1, 1)), K.reshape(value, (1,)))
        return state.alpha, occupation

    _, occupation = K.scan(
        measure, K.arange(n), (alpha, K.zeros((n,), dtype="float64"))
    )
    return K.cast(occupation, "int32")


def check_small():
    """Check Gaussian phases and the overlap estimator against a native sector sum."""
    K = tc.backend
    peps = FermionPEPS(2, 3, 2)
    problem = Hofstadter(peps, 4)
    theta = peps.random_parameters(K.get_random_state(73))
    basis, h = sector_reference(problem, -1.0)
    alpha, orbitals = gaussian_reference(problem, 0.0)
    uniforms = K.convert_to_tensor(np.random.default_rng(13).random((4096, 6)))
    for physical_time in (0.0, 0.7):
        alpha, occupied = gaussian_reference(problem, physical_time)
        correlation = occupied @ K.adjoint(occupied)
        actual = K.jit(K.vmap(lambda u: born_draw(correlation, u)))(uniforms)
        expected = K.jit(K.vmap(lambda u: born_draw_fgs(alpha, u)))(uniforms)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    reference = np.asarray(
        K.vmap(lambda s: determinant(orbitals, s, 4))(K.convert_to_tensor(basis))
    )
    energy = float(K.sum(K.eigh(problem.one_body(-1.0))[0][:4]))
    np.testing.assert_allclose(np.vdot(reference, reference), 1.0, atol=1e-12)
    np.testing.assert_allclose(h @ reference, energy * reference, atol=1e-11)
    psi = np.asarray(K.jit(peps.batch_amplitude)(theta, K.convert_to_tensor(basis)))
    exact = 1 - abs(np.vdot(reference, psi)) ** 2 / np.vdot(psi, psi).real
    report, arrays = measure(problem, theta, 0.0, samples=16384)
    np.testing.assert_allclose(
        report["estimated_infidelity"], exact, atol=6 * report["jackknife_stderr"]
    )
    exact_density = np.abs(psi) ** 2 @ basis / np.vdot(psi, psi).real
    np.testing.assert_array_less(
        np.abs(arrays["density"] - exact_density), 6 * arrays["density_stderr"] + 0.002
    )
    return {
        **report,
        "exact_infidelity": float(exact),
        "fgs_matching_configurations": 8192,
    }


def check_paths():
    """Different paths agree with a literal virtual-bond sum; hot calls never replan."""
    K = tc.backend
    first = FermionPEPS(2, 3, 2, exact_optimizer="greedy")
    second = FermionPEPS(2, 3, 2)
    theta = first.random_parameters(K.get_random_state(903))
    occupation = K.convert_to_tensor([1, 1, 0, 1, 1, 0])
    with patch.object(
        tc, "get_contractor", side_effect=AssertionError("Repeated path search")
    ):
        a = K.jit(first.scores)(theta, occupation)
        b = K.jit(second.scores)(theta, occupation)
    for actual, expected in zip(a, b):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        a[0], swap_network_reference(first, theta, occupation), rtol=1e-11, atol=1e-12
    )
    return {"amplitudes_and_scores_agree": True, "path_reused": True}


def check_summary(output):
    """Validate the generated numerical summary independently of the plot."""
    with np.load(output / "summary.npz") as saved:
        density, exact = saved["density"], saved["exact_density"]
        for name in (
            "density",
            "exact_density",
            "density_stderr",
            "infidelity",
            "infidelity_stderr",
            "effective_samples",
        ):
            assert np.isfinite(saved[name]).all(), name
        np.testing.assert_allclose(density.sum(axis=1), 24.0, atol=1e-10)
        np.testing.assert_allclose(exact.sum(axis=1), 24.0, atol=1e-10)
        np.testing.assert_array_less(-1e-12, density)
        np.testing.assert_array_less(density, 1.0 + 1e-12)
        np.testing.assert_array_less(-1e-12, saved["density_stderr"])
        difference = density - exact
        np.testing.assert_array_equal(
            np.argmax(density - saved["background"], axis=1),
            np.argmax(exact - saved["background"], axis=1),
        )
        np.testing.assert_array_less(np.sqrt(np.mean(difference**2, axis=1)), 0.001)
        np.testing.assert_array_less(np.max(np.abs(difference), axis=1), 0.003)
        return {
            "times": saved["times"].tolist(),
            "density_rmse": np.sqrt(np.mean(difference**2, axis=1)).tolist(),
            "density_max_error": np.max(np.abs(difference), axis=1).tolist(),
            "infidelity": saved["infidelity"].tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    checks = [
        check_kernels,
        check_cg,
        check_cg_estimate,
        check_paths,
        check_batched_moments,
        check_sampling,
        check_preparation,
        check_integrator,
        check_small,
    ]
    for check in checks:
        print(check.__name__, json.dumps(check(), indent=2), flush=True)
    if args.output is not None:
        print("summary", json.dumps(check_summary(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
