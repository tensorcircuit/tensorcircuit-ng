"""
Independent small-system and nodal checks for the Figure 5 reproduction.

All exact sums here are validation references, never the MC evolution path.
"""

from pathlib import Path

import numpy as np
from main import (
    exact_operators,
    exact_trajectory,
    initial_parameters,
    make_heun,
    make_rhs,
    make_rk45,
    make_trajectory,
)
from physics import (
    SpinProblem,
    inner_product_moments,
    reweighted_moments,
    schmitt_velocity,
    standard_moments,
)

import tensorcircuit as tc


def check_problem(nsites):
    """Check local H, exact initial state, TC circuits, and analytic derivatives."""
    K = tc.backend
    problem = SpinProblem(nsites)
    theta = initial_parameters(problem)
    basis, h, observable, hfull, indices, gauge = exact_operators(problem)
    a, d, local_hpsi, _ = problem.local_batch(theta, basis, 0.5)
    np.testing.assert_allclose(K.numpy(local_hpsi), K.numpy(h @ a), atol=3e-14)
    psi = a / K.norm(a)
    # Embed the Marshall-basis wave function into the physical spin basis.
    embedding = K.transpose(K.onehot(indices, 2**nsites))
    full = embedding @ (gauge * psi)
    circuit = tc.Circuit(nsites, inputs=full)
    np.testing.assert_allclose(K.numpy(circuit.state()), K.numpy(full), atol=3e-14)
    if nsites == 1:
        measured = circuit.expectation_ps(z=[0])
    else:
        measured = (
            sum(circuit.expectation_ps(**{axis: [0, 1]}) for axis in ["x", "y", "z"])
            / 4
        )
    expected = K.sum(K.conj(psi) * (observable @ psi))
    np.testing.assert_allclose(K.numpy(measured), K.numpy(expected), atol=3e-14)
    # A dense exact gate is legitimate here: it is used only as an independent reference.
    t = 0.37
    circuit.any(*range(nsites), unitary=K.expm(-1j * t * hfull))
    _, states = exact_trajectory(problem, theta, np.array([t]))
    evolved = K.numpy(circuit.state())
    np.testing.assert_allclose(
        evolved,
        K.numpy(embedding @ (gauge * K.convert_to_tensor(states[0]))),
        atol=3e-14,
    )
    ad = K.vmap(
        lambda x: K.grad(lambda p: K.real(problem.amplitude(p, x)))(theta)
        + 1j * K.grad(lambda p: K.imag(problem.amplitude(p, x)))(theta)
    )(basis)
    np.testing.assert_allclose(K.numpy(ad), K.numpy(d), atol=3e-14)
    # Grouping repeated samples must leave all MC moments and the SNR solve
    # unchanged, including when some allowed configurations were not sampled.
    repeated = K.concat([K.tile(basis[:1], [7, 1]), K.tile(basis[1:2], [3, 1])])
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
            np.testing.assert_allclose(K.numpy(actual), K.numpy(compact), atol=3e-14)
        full_velocity = schmitt_velocity(*full_moments[:3])
        compact_velocity = schmitt_velocity(*compact_moments[:3], counts=counts)
        np.testing.assert_allclose(
            K.numpy(full_velocity), K.numpy(compact_velocity), atol=3e-13
        )
    if nsites == 4:
        ground_problem = SpinProblem(4, delta_j=0.0)
        _, h0, _, _, _, _ = exact_operators(ground_problem)
        energies, vectors = K.eigh(h0)
        energy = K.real(K.sum(K.conj(psi) * (h0 @ psi)))
        fidelity = K.abs(K.sum(K.conj(vectors[:, 0]) * psi)) ** 2
        np.testing.assert_allclose(K.numpy(energies[0]), -2.0, atol=3e-14)
        with np.load(Path(__file__).resolve().parent / "initial_state.npz") as data:
            order = [
                np.flatnonzero(np.all(data["basis"] == row, axis=1))[0]
                for row in K.numpy(basis)
            ]
            reference = data["amplitudes"][order]
            np.testing.assert_allclose(
                K.numpy(psi), reference / np.linalg.norm(reference), atol=3e-14
            )
        np.testing.assert_allclose(K.numpy(energy), -2.0, rtol=0, atol=5e-3)
        np.testing.assert_allclose(K.numpy(fidelity), 1.0, rtol=0, atol=2e-3)
    else:
        times = np.linspace(0, 2, 101)
        exact, _ = exact_trajectory(problem, theta, times)
        np.testing.assert_allclose(exact, -np.sin(2 * times), atol=3e-14)


def check_nodal_estimator(nsites):
    """Use an actual zero amplitude for one spin and an RBM cosh node."""
    K = tc.backend
    problem = SpinProblem(nsites)
    basis, h, _, _, _, _ = exact_operators(problem)
    if nsites == 1:
        theta = K.convert_to_tensor([1.0, 0.0, 0.0, 0.0])
    else:
        weights = K.convert_to_tensor([0.2, -0.1, 0.35, -0.2])
        bias = -K.sum(weights * basis[0])
        theta = K.concat(
            [
                weights,
                K.reshape(bias, [1]),
                K.convert_to_tensor([0.0, 0.0, 0.0, 0.0, np.pi / 2]),
            ]
        )
    a, d, hp, density = problem.local_batch(theta, basis, 0.5)
    np.testing.assert_allclose(K.numpy(hp), K.numpy(h @ a), atol=3e-14)
    ad = K.vmap(
        lambda x: K.grad(lambda p: K.real(problem.amplitude(p, x)))(theta)
        + 1j * K.grad(lambda p: K.imag(problem.amplitude(p, x)))(theta)
    )(basis)
    np.testing.assert_allclose(K.numpy(d), K.numpy(ad), atol=3e-14)
    exact_s, exact_f = inner_product_moments(a, d, hp)
    adjacency = K.cast(K.abs(h) > 0, "float64") * (
        1 - K.eye(h.shape[0], dtype="float64")
    )
    degree = K.sum(adjacency, axis=0)
    kernel = (
        0.5 * K.eye(h.shape[0], dtype="float64") + 0.5 * adjacency / degree[None, :]
    )
    np.testing.assert_allclose(K.numpy(K.sum(kernel, axis=0)), 1.0, atol=3e-14)
    np.testing.assert_allclose(
        K.numpy(density), K.numpy(kernel @ K.abs(a) ** 2), atol=3e-14
    )
    r = density / K.sum(K.abs(a) ** 2)
    factors = K.sqrt(r * a.shape[0])
    estimated_s, estimated_f, _, _ = reweighted_moments(
        a * factors, d * factors[:, None], hp * factors, density
    )
    np.testing.assert_allclose(
        K.numpy(estimated_s), K.numpy(exact_s), atol=3e-13, rtol=3e-13
    )
    np.testing.assert_allclose(
        K.numpy(estimated_f), K.numpy(exact_f), atol=3e-13, rtol=3e-13
    )
    np.testing.assert_allclose(np.min(np.abs(K.numpy(a))), 0.0, atol=3e-14)
    if nsites == 1:
        # Sampling only |0> has exactly zero covariance but the true derivative in
        # the missing |1> direction is finite. No masked log-score reference.
        repeated = K.convert_to_tensor([[1]] * 128)
        ap = problem.amplitudes(theta, repeated)
        dp = problem.derivatives(theta, repeated)
        hpp = problem.local_batch(theta, repeated, 0.0)[2]
        ss, ff, _, _ = standard_moments(ap, dp, hpp)
        np.testing.assert_array_equal(K.numpy(ss), np.zeros((4, 4)))
        np.testing.assert_array_equal(K.numpy(ff), np.zeros(4))
        np.testing.assert_allclose(K.numpy(exact_s[1, 1]), 1.0, atol=3e-14)
        np.testing.assert_allclose(K.numpy(exact_f[1]), 1j, atol=3e-14)
    # Independent chains and replicas quantify MCMC uncertainty of S and F.
    estimates_s, estimates_f, corrected_s, corrected_f = [], [], [], []
    nrep, samples = 32, 4096
    estimate = K.jit(lambda c, rng: problem.estimate(theta, c, rng, 0.5))
    burn = K.jit(lambda c, rng: problem.metropolis(theta, c, rng, 128))
    for seed in range(700, 700 + nrep):
        chains, key = burn(
            problem.initial_chains(samples), K.set_random_state(seed, get_only=True)
        )
        ss, ff, _, weights, counts, _, _ = estimate(chains, key)
        # Eq. (11), with normalized weights p/r (norm is known only here).
        factor = (
            samples
            * float(K.numpy(K.sum(counts * weights) / samples)) ** 2
            / (samples - 1)
        )
        estimates_s.append(K.numpy(ss))
        estimates_f.append(K.numpy(ff))
        corrected_s.append(factor * K.numpy(ss))
        corrected_f.append(factor * K.numpy(ff))
    for name, estimates, reference in [
        ("S", estimates_s, exact_s),
        ("F", estimates_f, exact_f),
        ("Eq11_S", corrected_s, exact_s),
        ("Eq11_F", corrected_f, exact_f),
    ]:
        values, reference = np.array(estimates), K.numpy(reference)
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(nrep)
        # A simultaneous 5-SEM check over at most 100 components, with floating
        # point tolerance for exactly vanishing variances.
        np.testing.assert_allclose(
            (mean - reference) / (5 * sem + 1e-12),
            0.0,
            rtol=0,
            atol=1.0,
            err_msg=f"Nodal {name} differs by more than five replica SEM",
        )


def check_trajectory(nsites):
    """Compare staged steps with a host loop, including chains, RNG and ESS."""
    K = tc.backend
    problem = SpinProblem(nsites)
    theta0 = initial_parameters(problem)
    chains0, key0 = K.jit(lambda c, r: problem.metropolis(theta0, c, r, 128))(
        problem.initial_chains(128), K.set_random_state(100, get_only=True)
    )
    step_sizes = np.array([0.001, 0.001, 0.001, 0.0005])
    for q in [0.0, 0.5]:
        rhs = make_rhs(problem, q)
        step = make_rk45(rhs) if nsites == 1 else make_heun(rhs)
        theta, chains, key, trajectory = make_trajectory(step, step_sizes)(
            theta0, chains0, key0
        )
        expected, current, rng = theta0, chains0, key0
        rows = [np.append(K.numpy(theta0), 1.0)]
        for dt in step_sizes:
            expected, current, rng, ess = step(expected, current, rng, dt)
            rows.append(np.append(K.numpy(expected), K.numpy(ess)))
        np.testing.assert_allclose(K.numpy(trajectory), rows, rtol=0, atol=3e-12)
        np.testing.assert_allclose(
            K.numpy(theta), K.numpy(expected), rtol=0, atol=3e-12
        )
        np.testing.assert_array_equal(K.numpy(chains), K.numpy(current))
        np.testing.assert_array_equal(K.numpy(key), K.numpy(rng))


def check_step_size():
    """Halve the single-spin blurred RK45 step across the full benchmark interval."""
    K = tc.backend
    problem = SpinProblem(1)
    theta0 = initial_parameters(problem)
    chains, key = K.jit(lambda c, r: problem.metropolis(theta0, c, r, 128))(
        problem.initial_chains(8192), K.set_random_state(100, get_only=True)
    )
    basis, _, observable, _, _, _ = exact_operators(problem)
    step = make_rk45(make_rhs(problem, 0.5))
    curves = []
    for dt in [0.001, 0.0005]:
        nsteps = round(2.0 / dt)
        trajectory = make_trajectory(step, np.full(nsteps, dt))(theta0, chains, key)[-1]
        states = K.vmap(lambda p: problem.amplitudes(p, basis))(trajectory[:, :-1])
        values = K.real(
            K.sum(K.conj(states) * (states @ K.transpose(observable)), axis=1)
        ) / K.sum(K.abs(states) ** 2, axis=1)
        values = K.numpy(values)
        exact = -np.sin(2 * dt * np.arange(nsteps + 1))
        np.testing.assert_allclose(values, exact, rtol=0, atol=1e-10)
        curves.append(values)
        print(
            f"Single-spin dt={dt:g}: max |Z-Z_exact|={np.max(np.abs(values - exact)):.3e}"
        )
    np.testing.assert_allclose(curves[0], curves[1][::2], rtol=0, atol=1e-10)
    print(
        f"Step halving: max |Z_dt-Z_dt/2|={np.max(np.abs(curves[0] - curves[1][::2])):.3e}"
    )


def main():
    """Validate Hamiltonians, derivatives, TC circuits, and nodal MC estimates."""
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    for nsites in [1, 4]:
        check_problem(nsites)
        check_nodal_estimator(nsites)
        check_trajectory(nsites)
    check_step_size()
    print("All physics, nodal-estimator and fixed-step trajectory checks passed.")


if __name__ == "__main__":
    main()
