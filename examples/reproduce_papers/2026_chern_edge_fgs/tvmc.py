"""
Born sampling and gauge-projected PEPS-tVMC, following PRX Quantum 7, 033035.

The standard SR equation is Eq. (13); minSR uses the published uncentered
form, Eq. (17), with the real-time factor -i restored as in Eq. (18).
"""

import math

import tensorcircuit as tc


def sr_solve(scores, energies, weights, projector, regulator=1e-8):
    """Solve in the gauge-orthogonal space with a positive Cholesky system."""
    K = tc.backend
    mean_score = K.sum(weights[:, None] * scores, axis=0)
    mean_energy = K.sum(weights * energies)
    centered = scores - mean_score
    residual_energy = energies - mean_energy
    metric = K.adjoint(centered) @ (weights[:, None] * centered)
    force = K.adjoint(centered) @ (weights * residual_energy)
    projected = projector @ metric @ projector
    projected = (projected + K.adjoint(projected)) / 2
    rhs = projector @ force
    identity = K.eye(metric.shape[0], dtype="complex128")
    # Unit eigenvalues in discarded gauge directions implement the constrained
    # solve without choosing a discontinuous basis for its orthogonal complement.
    system = projected + regulator * projector + identity - projector
    velocity = projector @ K.solve(system, rhs, assume_a="pos")
    residual = K.norm(metric @ velocity - force) ** 2 / K.norm(force) ** 2
    return velocity, mean_energy, residual


def minsr_solve(scores, energies, weights, regulator=1e-8):
    """Published uncentered minSR; use only in the underdetermined regime."""
    K = tc.backend
    root = K.sqrt(weights)
    design = root[:, None] * scores
    rhs = root * energies
    metric = design @ K.adjoint(design)
    metric = (metric + K.adjoint(metric)) / 2
    y = K.solve(
        metric + regulator * K.eye(metric.shape[0], dtype="complex128"),
        rhs,
        assume_a="pos",
    )
    velocity = K.adjoint(design) @ y
    centered = scores - K.sum(weights[:, None] * scores, axis=0)
    local_residual = centered @ velocity - (energies - K.sum(weights * energies))
    force = K.adjoint(centered) @ (weights * energies)
    residual = (
        K.norm(K.adjoint(centered) @ (weights * local_residual)) ** 2
        / K.norm(force) ** 2
    )
    return velocity, K.sum(weights * energies), residual


class MonteCarlo:
    """Independent number-conserving Metropolis chains with random bond proposals."""

    def __init__(
        self, problem, chains=64, draws=32, sweeps=2, solver="sr", regulator=1e-8
    ):
        if chains < 2 or min(draws, sweeps) < 1 or solver not in ("sr", "minsr"):
            raise ValueError(
                "At least two chains, positive draws/sweeps, and sr/minsr are required."
            )
        self.problem, self.peps = problem, problem.peps
        self.chains, self.draws, self.sweeps = chains, draws, sweeps
        self.solver, self.regulator = solver, regulator
        if solver == "minsr" and chains * draws >= self.peps.nparams:
            raise ValueError(
                "minSR requires fewer sampled rows than parameters; use SR."
            )
        self.capacity = min(
            chains * draws, math.comb(problem.nsites, problem.particles)
        )
        self.compress_samples = problem.nsites <= 20
        if self.compress_samples:
            self.powers = tc.backend.convert_to_tensor(
                [2**i for i in range(problem.nsites)]
            )

    def initial_chains(self, key):
        K = tc.backend
        random_values = K.stateful_randu(key, (self.chains, self.problem.nsites))
        return K.cast(
            K.argsort(random_values, axis=1) < self.problem.particles, "int64"
        )

    def advance(self, theta, chains, key, steps):
        """One transition satisfies detailed balance for the actual amplitude function."""
        K = tc.backend
        logp = 2 * K.log(K.abs(self.peps.batch_amplitude(theta, chains)))

        def step(carry, _):
            state, current, rng, accepted = carry
            rng, draw = K.random_split(rng)
            random_values = K.stateful_randu(draw, (self.chains, 2))
            bond = K.cast(random_values[:, 0] * self.problem.left.shape[0], "int64")
            permutations = self.problem.permutations[bond]
            proposal = K.vmap(lambda s, p: s[p], vectorized_argnums=(0, 1))(
                state, permutations
            )
            proposed = 2 * K.log(K.abs(self.peps.batch_amplitude(theta, proposal)))
            changed = K.sum(K.abs(proposal - state), axis=1) > 0
            accept = K.log(random_values[:, 1]) < proposed - current
            return (
                K.where(accept[:, None], proposal, state),
                K.where(accept, proposed, current),
                rng,
                accepted + K.sum(K.cast(accept & changed, "float64")),
            )

        chains, _, key, accepted = K.scan(
            step, K.arange(steps), (chains, logp, key, K.cast(0.0, "float64"))
        )
        return chains, key, accepted / (steps * self.chains)

    def sample(self, theta, chains, key):
        K = tc.backend
        samples = K.zeros((self.draws, self.chains, self.problem.nsites), dtype="int64")

        def draw(carry, i):
            state, rng, buffer, accepted = carry
            state, rng, rate = self.advance(
                theta, state, rng, self.sweeps * self.problem.left.shape[0]
            )
            buffer = K.scatter(buffer, K.reshape(i, (1, 1)), state[None])
            return state, rng, buffer, accepted + rate

        chains, key, samples, accepted = K.scan(
            draw, K.arange(self.draws), (chains, key, samples, K.cast(0.0, "float64"))
        )
        return (
            K.reshape(samples, (-1, self.problem.nsites)),
            chains,
            key,
            accepted / self.draws,
        )

    def rows(self, samples):
        """Coalesce only configurations actually sampled; never enumerate the basis."""
        K = tc.backend
        if self.compress_samples:
            labels = samples @ self.powers
            labels, counts = K.unique_with_counts(
                labels, size=self.capacity, fill_value=labels[0]
            )
            configurations = K.mod(labels[:, None] // self.powers[None, :], 2)
            return configurations, counts / K.sum(counts)
        return samples, K.ones((samples.shape[0],), dtype="float64") / samples.shape[0]

    def estimate(self, theta, chains, key, potential=0.0):
        K = tc.backend
        samples, chains, key, accepted = self.sample(theta, chains, key)
        configurations, weights = self.rows(samples)
        _, scores = self.peps.batch_scores(theta, configurations)
        energies = self.problem.local_batch(theta, configurations, potential)
        if self.solver == "sr":
            direction, energy, residual = sr_solve(
                scores,
                energies,
                weights,
                self.peps.gauge_projector(theta),
                self.regulator,
            )
        else:
            direction, energy, residual = minsr_solve(
                scores, energies, weights, self.regulator
            )
        blocks = K.mean(
            K.reshape(
                K.cast(samples, "float64"),
                (self.draws, self.chains, self.problem.nsites),
            ),
            axis=0,
        )
        density = K.mean(blocks, axis=0)
        error = K.sqrt(
            K.sum((blocks - density) ** 2, axis=0) / (self.chains * (self.chains - 1))
        )
        variance = K.real(K.sum(weights * K.abs(energies - energy) ** 2))
        diagnostics = K.concat(
            [
                K.real(energy)[None],
                variance[None],
                residual[None],
                accepted[None],
                density,
                error,
            ]
        )
        return direction, diagnostics, chains, key

    def rk4_step(self, theta, chains, key, dt, potential=0.0, imaginary=False):
        """Re-equilibrate chains at each RK stage; carry RNG state between stages."""
        factor = -1.0 if imaginary else -1j
        d1, diagnostics, chains, key = self.estimate(theta, chains, key, potential)
        d2, _, chains, key = self.estimate(
            theta + factor * dt * d1 / 2, chains, key, potential
        )
        d3, _, chains, key = self.estimate(
            theta + factor * dt * d2 / 2, chains, key, potential
        )
        d4, _, chains, key = self.estimate(
            theta + factor * dt * d3, chains, key, potential
        )
        theta = self.peps.normalize(
            theta + factor * dt * (d1 + 2 * d2 + 2 * d3 + d4) / 6
        )
        return theta, chains, key, diagnostics

    def trajectory(self, theta, chains, key, dt, steps, potential=0.0, imaginary=False):
        """A complete fixed-step trajectory executes within a backend scan."""
        K = tc.backend
        history = K.zeros((steps, 4 + 2 * self.problem.nsites), dtype="float64")

        def step(carry, index):
            params, states, rng, buffer = carry
            params, states, rng, diagnostics = self.rk4_step(
                params, states, rng, dt, potential, imaginary
            )
            buffer = K.scatter(buffer, K.reshape(index, (1, 1)), diagnostics[None])
            return params, states, rng, buffer

        return K.scan(step, K.arange(steps), (theta, chains, key, history))
