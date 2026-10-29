"""Reproduction of "Real-Time Dynamics in Two Dimensions with Tensor Network States
via Time-Dependent Variational Monte Carlo Method"
Link: https://arxiv.org/abs/2512.06768
Description:
This script reproduces the Figure 2(b,c) method from the paper using TensorCircuit-NG.
A scaled 6 x 6 corner quench compares independent PEPS-tVMC with exact Gaussian
densities; it does not establish chiral edge propagation at the paper's scale.
"""

import argparse
import cmath
import itertools
import math
from pathlib import Path
import time

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import matplotlib.pyplot as plt
import numpy as np
import tencirpauli as tcp
import tensorcircuit as tc

DEFAULT_SEED = 17


class FermionPEPS:
    """Even-parity PEPS with physical order i=y*columns+x and upward y."""

    def __init__(
        self,
        rows,
        columns,
        bond_dim=2,
        exact_optimizer="omeco-8-48",
    ):
        if rows < 2 or columns < 2 or bond_dim < 2 or bond_dim % 2:
            raise ValueError("Use a rectangle of at least 2 x 2 and an even D >= 2.")
        self.rows, self.columns = rows, columns
        self.nsites = rows * columns
        self.bond_dim = bond_dim
        self.shapes, self.offsets, self.allowed, self.coordinates = [], [0], [], []
        for y in range(rows):
            for x in range(columns):
                shape = (
                    2,
                    bond_dim if x else 1,
                    bond_dim if x + 1 < columns else 1,
                    bond_dim if y else 1,
                    bond_dim if y + 1 < rows else 1,
                )
                coordinates = []
                for virtual in itertools.product(*(range(d) for d in shape[1:])):
                    coordinates.append((sum(virtual) % 2,) + virtual)
                indices = []
                for coordinate in coordinates:
                    index = 0
                    for k, d in zip(coordinate, shape):
                        index = index * d + k
                    indices.append(index)
                self.shapes.append(shape)
                self.coordinates.append(coordinates)
                self.allowed.append(tc.backend.convert_to_tensor(indices))
                self.offsets.append(self.offsets[-1] + len(indices))
        self.nparams = self.offsets[-1]
        self.bonds = []
        for y in range(rows):
            for x in range(columns):
                i = y * columns + x
                if x + 1 < columns:
                    self.bonds.append((i, i + 1, 2, 1))
                if y + 1 < rows:
                    self.bonds.append((i, i + columns, 4, 3))
        self.prepare_exact_path(exact_optimizer)
        self.batch_amplitude = tc.backend.vmap(self.amplitude, vectorized_argnums=1)
        self.value_derivative = tc.backend.value_and_grad(
            lambda theta, occupation: tc.backend.real(self.amplitude(theta, occupation))
        )
        self.batch_scores = tc.backend.vmap(self.scores, vectorized_argnums=1)

    def random_parameters(self, key):
        """Generic complex tensors avoid the stationary product-state manifold."""
        K = tc.backend
        k1, k2 = K.random_split(key)
        theta = K.stateful_randn(k1, [self.nparams]) + 1j * K.stateful_randn(
            k2, [self.nparams]
        )
        return self.normalize(theta)

    def normalize(self, theta):
        """Remove irrelevant tensor scales without changing the physical state."""
        K = tc.backend
        return K.concat(
            [
                theta[a:b] / K.norm(theta[a:b])
                for a, b in zip(self.offsets[:-1], self.offsets[1:])
            ]
        )

    def tensors(self, theta):
        """Expand parity blocks to physical,left,right,down,up tensors."""
        K = tc.backend
        return [
            K.reshape(
                K.scatter(
                    K.zeros([math.prod(shape)], dtype="complex128"),
                    indices[:, None],
                    theta[a:b],
                ),
                shape,
            )
            for shape, indices, a, b in zip(
                self.shapes, self.allowed, self.offsets[:-1], self.offsets[1:]
            )
        ]

    def sliced_tensors(self, theta, occupation):
        """Absorb every sampled physical/virtual fermionic swap exactly once."""
        K = tc.backend
        grid = K.reshape(occupation, (self.rows, self.columns))
        right = K.sum(grid, axis=1)[:, None] - K.cumsum(grid, axis=1)
        tensors = self.tensors(theta)
        sliced = []
        for i, tensor in enumerate(tensors):
            y, x = divmod(i, self.columns)
            sign = 1 - 2 * K.mod(right[y, x] * K.arange(tensor.shape[-1]), 2)
            sliced.append(tensor[occupation[i]] * sign[None, None, None, :])
        return sliced

    def amplitude(self, theta, occupation):
        """Contract a sampled single-layer PEPS along the cached pairwise path."""
        K = tc.backend
        tensors = self.sliced_tensors(theta, occupation)
        values = [K.reshape(t, shape) for t, shape in zip(tensors, self.exact_shapes)]
        for a, b, axes in self.exact_steps:
            result = K.tensordot(values[a], values[b], axes)
            values = [v for i, v in enumerate(values) if i not in (a, b)]
            values.append(result)
        return K.reshape(values[0], ())

    def prepare_exact_path(self, optimizer="omeco-8-48"):
        """Search once per topology; reuse backend tensordot steps for every sample."""
        labels = [[None] * 4 for _ in self.shapes]
        for bond, (i, j, ai, aj) in enumerate(self.bonds):
            labels[i][ai - 1] = labels[j][aj - 1] = bond
        inputs = [[ix for ix in site if ix is not None] for site in labels]
        self.exact_shapes = [[self.bond_dim] * len(site) for site in inputs]
        finder = tc.get_contractor(optimizer).keywords["optimizer"]
        path = finder(inputs, [], {i: self.bond_dim for i in range(len(self.bonds))})
        current, steps = [list(site) for site in inputs], []
        for a, b in path:
            common = [ix for ix in current[a] if ix in current[b]]
            axes = (
                [current[a].index(ix) for ix in common],
                [current[b].index(ix) for ix in common],
            )
            remaining = [ix for ix in current[a] + current[b] if ix not in common]
            current = [site for i, site in enumerate(current) if i not in (a, b)]
            current.append(remaining)
            steps.append((a, b, axes))
        if current != [[]]:
            raise ValueError(
                "The contraction path must reduce the network to a scalar."
            )
        self.exact_steps = steps

    def scores(self, theta, occupation):
        """Holomorphic derivative of log amplitude, with JAX's complex convention."""
        amplitude = self.amplitude(theta, occupation)
        _, derivative = self.value_derivative(theta, occupation)
        return amplitude, derivative / amplitude


class Hofstadter:
    """Open spinless-fermion hopping; the published negative Peierls exponent."""

    def __init__(self, peps, particles):
        if particles % 2 or not 0 < particles < peps.nsites:
            raise ValueError("This even-parity example requires an even 0 < N < sites.")
        self.peps, self.particles = peps, particles
        self.nsites = peps.nsites
        self.corner = (peps.rows - 1) * peps.columns
        K = tc.backend
        pairs = [(i, j) for i, j, _, _ in peps.bonds]
        self.left = K.convert_to_tensor([i for i, _ in pairs])
        self.right = K.convert_to_tensor([j for _, j in pairs])
        phases, terms = [], []
        for i, j in pairs:
            phase = (
                -cmath.exp(-2j * math.pi * (i % peps.columns) / 3)
                if j - i == peps.columns
                else -1.0
            )
            phases.append(phase)
            terms.extend(
                [
                    (((i, "create"), (j, "annihilate")), phase),
                    (((j, "create"), (i, "annihilate")), complex(phase).conjugate()),
                ]
            )
        self.hopping = K.stack([K.cast(h, "complex128") for h in phases])
        self.operator = tcp.FermionOperator.from_terms(self.nsites, terms)
        self.pin_operator = tcp.FermionOperator.from_terms(
            self.nsites, [(((self.corner, "create"), (self.corner, "annihilate")), 1.0)]
        )
        self.pauli = self.operator.map_fermions()
        x_masks, z_masks = [], []
        for term in self.pauli.terms:
            codes = term.word.to_codes()
            x_masks.append(tuple(int(code in (1, 2)) for code in codes))
            z_masks.append(tuple(int(code in (2, 3)) for code in codes))
        unique = list(dict.fromkeys(x_masks))
        groups = [unique.index(mask) for mask in x_masks]
        self.flips = K.convert_to_tensor(unique)
        self.z_masks = K.convert_to_tensor(z_masks)
        # Pauli terms avoid the full-state dimension limit of backend_mvp_plan.
        self.coefficients = K.convert_to_tensor(
            [
                term.coefficient * (-1j) ** sum(x * z for x, z in zip(xs, zs))
                for term, xs, zs in zip(self.pauli.terms, x_masks, z_masks)
            ]
        )
        self.grouping = K.transpose(K.onehot(K.convert_to_tensor(groups), len(unique)))
        self.local_batch = K.vmap(self.local_energy, vectorized_argnums=1)

    def local_energy(self, theta, occupation, potential=0.0):
        """Evaluate connected amplitudes using TenCirPauli's mapped Pauli terms."""
        K = tc.backend
        signs = 1 - 2 * K.mod(self.z_masks @ occupation, 2)
        matrix_elements = self.grouping @ (self.coefficients * signs)
        connected = K.mod(occupation[None, :] + self.flips, 2)
        amplitudes = self.peps.batch_amplitude(theta, connected)
        amplitudes = K.where(
            K.sum(connected, axis=1) == self.particles, amplitudes, 0.0
        )
        amplitude = self.peps.amplitude(theta, occupation)
        return (
            K.sum(matrix_elements * amplitudes) / amplitude
            + potential * occupation[self.corner]
        )

    def one_body(self, potential=0.0):
        """Single-particle Hamiltonian for the independent FGS reference only."""
        K = tc.backend
        h = K.scatter(
            K.zeros((self.nsites, self.nsites), dtype="complex128"),
            K.stack([self.left, self.right], axis=1),
            self.hopping,
        )
        h = h + K.adjoint(h)
        pin = K.onehot(self.corner, self.nsites)
        return h + potential * pin[:, None] * pin[None, :]


def sr_moments_solve(metric, force, regulator=1e-8):
    """Solve centered ridge SR; exact gauge nulls need no explicit projection.

    Both the covariance and force have zero components along every exact
    gauge direction. Adding a positive scalar ridge preserves that null
    space and selects a direction orthogonal to it, without a gauge SVD.
    """
    K = tc.backend
    metric = (metric + K.adjoint(metric)) / 2
    identity = K.eye(metric.shape[0], dtype="complex128")
    velocity = K.solve(metric + regulator * identity, force, assume_a="pos")
    norm2 = K.norm(force) ** 2
    residual = K.norm(metric @ velocity - force) ** 2 / K.where(norm2 > 0, norm2, 1.0)
    return velocity, residual


def sr_cg_solve(scores, energies, weights, regulator=1e-8, tolerance=1e-8, maxiter=512):
    """
    Centered ridge SR with matrix-free Fisher products and zero-start CG.

    Every iterate lies in the centered score row space. Exact gauge-null
    directions are excluded without constructing a gauge basis or projector.
    Storage is O(N P), each iteration costs O(N P), and no P x P or N x N
    metric is formed. Return the true relative linear-system residual as well
    as the unregularized squared force residual used by the other solvers.
    """
    if regulator <= 0 or tolerance <= 0 or maxiter < 1:
        raise ValueError("CG requires positive regulator, tolerance and maxiter.")
    K = tc.backend
    mean_energy = K.sum(weights * energies)
    centered = scores - K.sum(weights[:, None] * scores, axis=0)
    force = K.adjoint(centered) @ (weights * (energies - mean_energy))

    def fisher(vector):
        return K.adjoint(centered) @ (weights * (centered @ vector))

    def inner(vector):
        return K.real(K.sum(K.conj(vector) * vector))

    norm2 = inner(force)
    denominator = K.where(norm2 > 0, norm2, 1.0)

    def step(carry, _):
        vector, residual, direction, squared, iterations = carry

        def update():
            product = fisher(direction) + regulator * direction
            alpha = squared / K.real(K.sum(K.conj(direction) * product))
            next_residual = residual - alpha * product
            next_squared = inner(next_residual)
            return (
                vector + alpha * direction,
                next_residual,
                next_residual + (next_squared / squared) * direction,
                next_squared,
                iterations + 1,
            )

        return K.cond(squared > tolerance**2 * norm2, update, lambda: carry)

    velocity, _, _, _, iterations = K.scan(
        step,
        K.arange(maxiter),
        (
            K.zeros(force.shape, dtype="complex128"),
            force,
            force,
            norm2,
            K.cast(0, "int64"),
        ),
    )
    unregularized = fisher(velocity) - force
    linear_residual = K.sqrt(inner(unregularized + regulator * velocity) / denominator)
    return (
        velocity,
        mean_energy,
        inner(unregularized) / denominator,
        linear_residual,
        iterations,
    )


class MonteCarlo:
    """Fixed-number Metropolis chains and exact-contraction SR dynamics."""

    def __init__(
        self,
        problem,
        chains=10240,
        sweeps=1,
        solver="dense",
        regulator=1e-5,
        contraction_batch_size=2048,
        cg_tolerance=1e-8,
        cg_maxiter=8192,
    ):
        if chains < 2 or sweeps < 1 or contraction_batch_size < 1:
            raise ValueError("Use at least two chains and positive sweep/batch sizes.")
        if solver not in ("dense", "cg") or regulator <= 0:
            raise ValueError("Use dense or cg SR with a positive regulator.")
        self.problem, self.peps = problem, problem.peps
        self.chains, self.sweeps = chains, sweeps
        self.solver, self.regulator = solver, regulator
        self.contraction_batch_size = contraction_batch_size
        self.cg_tolerance, self.cg_maxiter = cg_tolerance, cg_maxiter

    def initial_chains(self, key):
        K = tc.backend
        random_values = K.stateful_randu(key, (self.chains, self.problem.nsites))
        return K.cast(
            K.argsort(random_values, axis=1) < self.problem.particles, "int64"
        )

    def advance(self, theta, chains, key, steps):
        """Swap a uniformly selected occupied/empty pair, accepting by PEPS weight.

        At fixed particle number the proposal probability 1/[N(L-N)] is
        identical in both directions, including for non-neighbouring sites.
        """
        K = tc.backend
        logp = 2 * K.log(K.abs(self.peps.batch_amplitude(theta, chains)))

        def step(carry, _):
            state, current, rng, accepted = carry
            rng, draw = K.random_split(rng)
            random_values = K.stateful_randu(draw, (self.chains, 3))
            occupied = K.argmax(
                K.cumsum(state, axis=1)
                > random_values[:, 0, None] * self.problem.particles,
                axis=1,
            )
            empty = K.argmax(
                K.cumsum(1 - state, axis=1)
                > random_values[:, 1, None]
                * (self.problem.nsites - self.problem.particles),
                axis=1,
            )
            proposal = (
                state
                - K.cast(K.onehot(occupied, self.problem.nsites), "int64")
                + K.cast(K.onehot(empty, self.problem.nsites), "int64")
            )
            proposed = 2 * K.log(K.abs(self.peps.batch_amplitude(theta, proposal)))
            accept = K.log(random_values[:, 2]) < proposed - current
            return (
                K.where(accept[:, None], proposal, state),
                K.where(accept, proposed, current),
                rng,
                accepted + K.sum(K.cast(accept, "float64")),
            )

        chains, _, key, accepted = K.scan(
            step, K.arange(steps), (chains, logp, key, K.cast(0.0, "float64"))
        )
        return chains, key, accepted / (steps * self.chains)

    def sampled_moments(self, theta, configurations, potential=0.0):
        """Bound contraction intermediates while retaining all sampled SR rows."""
        K = tc.backend

        def evaluate(states):
            _, scores = self.peps.batch_scores(theta, states)
            return scores, self.problem.local_batch(theta, states, potential)

        count, batch = configurations.shape[0], self.contraction_batch_size
        if count <= batch:
            return evaluate(configurations)
        padding = (-count) % batch
        states = K.concat([configurations, K.tile(configurations[:1], (padding, 1))])
        blocks = K.reshape(states, (-1, batch, self.problem.nsites))
        _, (scores, energies) = K.jaxy_scan(
            lambda carry, block: (carry, evaluate(block)), None, blocks
        )
        return (
            K.reshape(scores, (-1, self.peps.nparams))[:count],
            K.reshape(energies, (-1,))[:count],
        )

    def sr_statistics(self, theta, configurations, weights, potential=0.0):
        """
        Stream weighted covariance blocks without storing the full score matrix.

        Weighted parallel-variance merging avoids subtracting two large raw
        second moments. Padding has zero weight and does not change the estimator.
        """
        K = tc.backend
        count = configurations.shape[0]
        batch = min(count, self.contraction_batch_size)
        padding = (-count) % batch
        states = K.concat([configurations, K.tile(configurations[:1], (padding, 1))])
        probabilities = K.concat([weights, K.zeros((padding,), dtype="float64")])
        blocks = (
            K.reshape(states, (-1, batch, self.problem.nsites)),
            K.reshape(probabilities, (-1, batch)),
        )

        def accumulate(carry, block):
            mass, mean, mean_e, metric, force = carry
            states, probability = block
            _, scores = self.peps.batch_scores(theta, states)
            energies = self.problem.local_batch(theta, states, potential)
            block_mass = K.sum(probability)
            denominator = K.where(block_mass > 0, block_mass, 1.0)
            block_mean = K.sum(probability[:, None] * scores, axis=0) / denominator
            block_energy = K.sum(probability * energies) / denominator
            centered = scores - block_mean
            centered_energy = energies - block_energy
            total = mass + block_mass
            total_denominator = K.where(total > 0, total, 1.0)
            fraction = block_mass / total_denominator
            correction = mass * fraction
            difference, difference_e = block_mean - mean, block_energy - mean_e
            metric = (
                metric
                + K.adjoint(centered) @ (probability[:, None] * centered)
                + correction * K.conj(difference)[:, None] * difference[None, :]
            )
            force = (
                force
                + K.adjoint(centered) @ (probability * centered_energy)
                + correction * K.conj(difference) * difference_e
            )
            return (
                total,
                mean + fraction * difference,
                mean_e + fraction * difference_e,
                metric,
                force,
            ), energies

        zero = K.zeros((self.peps.nparams,), dtype="complex128")
        initial = (
            K.cast(0.0, "float64"),
            zero,
            K.cast(0.0, "complex128"),
            K.zeros((self.peps.nparams, self.peps.nparams), dtype="complex128"),
            zero,
        )
        (mass, _, energy, metric, force), energies = K.jaxy_scan(
            accumulate, initial, blocks
        )
        return metric / mass, force / mass, energy, K.reshape(energies, (-1,))[:count]

    def estimate(self, theta, chains, key, potential=0.0):
        """Sample the PEPS Born distribution and solve the centered SR equations."""
        K = tc.backend
        chains, key, accepted = self.advance(
            theta, chains, key, self.sweeps * self.peps.nsites
        )
        weights = K.ones((self.chains,), dtype="float64") / self.chains
        if self.solver == "dense":
            metric, force, energy, energies = self.sr_statistics(
                theta, chains, weights, potential
            )
            direction, residual = sr_moments_solve(metric, force, self.regulator)
        else:
            scores, energies = self.sampled_moments(theta, chains, potential)
            direction, energy, residual, linear_residual, _ = sr_cg_solve(
                scores,
                energies,
                weights,
                self.regulator,
                self.cg_tolerance,
                self.cg_maxiter,
            )
            direction = K.where(
                linear_residual <= self.cg_tolerance * 10,
                direction,
                K.cast(float("nan"), "complex128"),
            )
        variance = K.real(K.sum(weights * K.abs(energies - energy) ** 2))
        diagnostics = K.stack([K.real(energy), variance, residual, accepted])
        return direction, diagnostics, chains, key

    def rk4_step(self, theta, chains, key, dt, potential=0.0, imaginary=False):
        """Re-equilibrate chains at each RK stage; carry RNG state between stages."""
        K = tc.backend
        factor = -1.0 if imaginary else -1j
        offsets = K.convert_to_tensor([0.0, 0.5, 0.5, 1.0])
        weights = K.convert_to_tensor([1.0, 2.0, 2.0, 1.0])

        def stage(carry, index):
            previous, total, states, rng, first = carry
            direction, diagnostics, states, rng = self.estimate(
                theta + factor * dt * offsets[index] * previous, states, rng, potential
            )
            return (
                direction,
                total + weights[index] * direction,
                states,
                rng,
                K.where(index == 0, diagnostics, first),
            )

        zero = K.zeros(theta.shape, dtype="complex128")
        _, total, chains, key, diagnostics = K.scan(
            stage,
            K.arange(4),
            (
                zero,
                zero,
                chains,
                key,
                K.zeros((4,), dtype="float64"),
            ),
        )
        theta = self.peps.normalize(theta + factor * dt * total / 6)
        return theta, chains, key, diagnostics

    def trajectory(
        self,
        theta,
        chains,
        key,
        dt,
        steps,
        potential=0.0,
        imaginary=False,
    ):
        """A complete fixed-step trajectory executes within a backend scan."""
        K = tc.backend
        history = K.zeros((steps, 4), dtype="float64")

        def step(carry, index):
            params, states, rng, buffer = carry
            if imaginary:
                direction, diagnostics, states, rng = self.estimate(
                    params, states, rng, potential
                )
                params = self.peps.normalize(params - dt * direction)
            else:
                params, states, rng, diagnostics = self.rk4_step(
                    params, states, rng, dt, potential, imaginary
                )
            buffer = K.scatter(buffer, K.reshape(index, (1, 1)), diagnostics[None])
            return params, states, rng, buffer

        return K.scan(step, K.arange(steps), (theta, chains, key, history))


def enlarge_bond(theta, source, target, seed, noise=0.05):
    """Embed parity blocks exactly, then activate the new entries with seeded noise."""
    K = tc.backend
    positions = []
    for offset, small, large in zip(
        target.offsets, source.coordinates, target.coordinates
    ):
        lookup = {coordinate: index for index, coordinate in enumerate(large)}
        positions.extend(offset + lookup[coordinate] for coordinate in small)
    enlarged = K.scatter(
        K.zeros((target.nparams,), dtype="complex128"),
        K.reshape(K.convert_to_tensor(positions), (-1, 1)),
        theta,
    )
    return target.normalize(
        enlarged + noise * target.random_parameters(K.get_random_state(seed))
    )


def fixed_number_alpha(h, particles):
    """
    Explicitly occupy the lowest `particles` single-particle orbitals.

    C = alpha @ alpha^dagger = diag(I - rho, rho.T), where
    rho[i,j] = <c[j]^dagger c[i]>. A zero-chemical-potential FGS ground
    state would not, in general, have the requested particle number.
    """
    K = tc.backend
    modes = h.shape[0] // 2
    _, orbitals = K.eigh(2 * h[:modes, :modes])
    empty = orbitals[:, particles:]
    occupied = K.conj(orbitals[:, :particles])
    top = K.concat([empty, K.zeros((modes, particles), dtype="complex128")], axis=1)
    bottom = K.concat(
        [K.zeros((modes, modes - particles), dtype="complex128"), occupied], axis=1
    )
    return K.concat([top, bottom], axis=0)


def gaussian_reference(problem, physical_time):
    K = tc.backend
    initial = problem.one_body(-1.0)
    zero = K.zeros(initial.shape, dtype="complex128")
    nambu = (
        K.concat(
            [
                K.concat([initial, zero], axis=1),
                K.concat([zero, -K.transpose(initial)], axis=1),
            ],
            axis=0,
        )
        / 2
    )
    alpha = fixed_number_alpha(nambu, problem.particles)
    orbitals = K.eigh(initial)[1][:, : problem.particles]
    h = problem.one_body()
    orbitals = K.expm(-1j * physical_time * h) @ orbitals
    nambu = (
        K.concat(
            [K.concat([h, zero], axis=1), K.concat([zero, -K.transpose(h)], axis=1)],
            axis=0,
        )
        / 2
    )
    state = tc.FGSSimulator(problem.nsites, alpha=alpha)
    state.evol_hamiltonian(2 * physical_time * nambu)
    return state.alpha, orbitals


def born_draw(correlation, randoms):
    """Exact number-conserving Gaussian sampling by Wick's conditional update.

    For unmeasured j,k, observing n_i=1 subtracts C_ji C_ik / C_ii;
    observing n_i=0 adds C_ji C_ik / (1-C_ii). This rank-one update
    avoids a QR decomposition after each occupation measurement.
    """
    K = tc.backend
    n = randoms.shape[0]

    def measure(carry, i):
        current, occupation = carry
        probability = K.clip(K.real(current[i, i]), 0.0, 1.0)
        value = randoms[i] >= 1 - probability
        selected = K.where(value, probability, 1 - probability)
        column = current[:, i]
        current = current + (K.where(value, -1.0, 1.0) / selected) * (
            column[:, None] * K.conj(column[None, :])
        )
        mask = K.cast(K.arange(n) != i, "float64")
        current = current * mask[:, None] * mask[None, :]
        current = current + K.cast(value, "float64") * (
            (1 - mask)[:, None] * (1 - mask)[None, :]
        )
        occupation = K.scatter(
            occupation, K.reshape(i, (1, 1)), K.reshape(K.cast(value, "int32"), (1,))
        )
        return current, occupation

    _, occupation = K.scan(
        measure, K.arange(n), (correlation, K.zeros((n,), dtype="int32"))
    )
    return occupation


def determinant(orbitals, occupation, particles):
    K = tc.backend
    indices = K.sort(K.argsort(-K.cast(occupation, "int32"))[:particles])
    return K.det(K.gather1d(orbitals, indices))


def importance_statistics(ratios, configurations):
    """Raw self-normalized PEPS density and delete-block jackknife uncertainty."""
    ratios = ratios / np.max(np.abs(ratios))
    weights = np.abs(ratios) ** 2
    weighted = weights[:, None] * configurations
    total = np.concatenate(([ratios.sum(), weights.sum()], weighted.sum(axis=0)))

    def moments(sums, count):
        fidelity = np.abs(sums[0]) ** 2 / (count * sums[1].real)
        return np.concatenate(([1 - fidelity], sums[2:].real / sums[1].real))

    estimate = moments(total, len(ratios))
    blocks = np.array_split(np.arange(len(ratios)), 32)
    deleted = []
    for indices in blocks:
        removed = np.concatenate(
            (
                [ratios[indices].sum(), weights[indices].sum()],
                weighted[indices].sum(axis=0),
            )
        )
        deleted.append(moments(total - removed, len(ratios) - len(indices)))
    deleted = np.asarray(deleted)
    stderr = np.sqrt(31 / 32 * np.sum((deleted - deleted.mean(axis=0)) ** 2, axis=0))
    return estimate, stderr, weights


def measure(problem, theta, physical_time, samples=8192, seed=8201, batch_size=2048):
    """FGS only proposes validation samples; every plotted PEPS weight uses its own amplitude."""
    K = tc.backend
    alpha, orbitals = gaussian_reference(problem, physical_time)
    correlation = orbitals @ K.adjoint(orbitals)
    reference_density = np.real(np.diag(np.asarray(correlation)))
    fgs = tc.FGSSimulator(problem.nsites, alpha=alpha)
    np.testing.assert_allclose(
        np.asarray(fgs.get_cmatrix())[problem.nsites :, problem.nsites :],
        np.asarray(K.transpose(correlation)),
        atol=1e-12,
    )

    def draw(parameters, occupied, cmat, uniforms):
        states = K.vmap(lambda u: born_draw(cmat, u))(uniforms)
        reference = K.vmap(lambda s: determinant(occupied, s, problem.particles))(
            states
        )
        return problem.peps.batch_amplitude(parameters, states) / reference, states

    draw_batch = K.jit(draw)
    mesh = Mesh(np.asarray(jax.devices()), ("chains",))
    sharded = NamedSharding(mesh, PartitionSpec("chains", None))
    replicated = NamedSharding(mesh, PartitionSpec())
    theta, orbitals, correlation = [
        jax.device_put(np.asarray(value), replicated)
        for value in (theta, orbitals, correlation)
    ]
    rng = np.random.default_rng(seed)
    ratios, configurations = [], []
    started = time.perf_counter()
    for _ in range((samples + batch_size - 1) // batch_size):
        uniforms = jax.device_put(rng.random((batch_size, problem.nsites)), sharded)
        ratios_batch, states = draw_batch(theta, orbitals, correlation, uniforms)
        np.testing.assert_array_equal(np.asarray(states).sum(axis=1), problem.particles)
        values = np.asarray(ratios_batch)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Non-finite importance ratios")
        ratios.append(values)
        configurations.append(np.asarray(states))
    ratios = np.concatenate(ratios)[:samples]
    configurations = np.concatenate(configurations)[:samples]
    estimate, stderr, weights = importance_statistics(ratios, configurations)
    density = configurations.mean(axis=0)
    error = configurations.std(axis=0, ddof=1) / np.sqrt(samples)
    np.testing.assert_array_less(abs(density - reference_density), 6 * error + 0.002)
    difference = estimate[1:] - reference_density
    report = {
        "time": physical_time,
        "samples": samples,
        "seed": seed,
        "batch_size": batch_size,
        "proposal": "Exact Gaussian conditional rank-one updates, validated against FGSSimulator",
        "estimated_infidelity": float(estimate[0]),
        "jackknife_stderr": float(stderr[0]),
        "density_rmse": float(np.sqrt(np.mean(difference**2))),
        "density_max_error": float(np.max(abs(difference))),
        "density_stderr_rms": float(np.sqrt(np.mean(stderr[1:] ** 2))),
        "density_stderr_max": float(np.max(stderr[1:])),
        "weight_effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
        "max_weight_fraction": float(weights.max() / weights.sum()),
        "seconds_including_compilation": time.perf_counter() - started,
        "estimator": "Self-normalized importance density; no reference-density correction or smoothing.",
    }
    arrays = {
        "density": estimate[1:],
        "density_stderr": stderr[1:],
        "exact_density": reference_density,
    }
    return report, arrays


def distribute(theta, states, key):
    """Shard independent chains; replicate shared PEPS parameters and random state."""
    devices = jax.devices()
    if states.shape[0] % len(devices):
        raise ValueError(
            "The chain count must be divisible by the visible device count."
        )
    mesh = Mesh(np.asarray(devices), ("chains",))
    replicated = NamedSharding(mesh, PartitionSpec())
    sharded = NamedSharding(mesh, PartitionSpec("chains", None))
    return (
        jax.device_put(theta, replicated),
        jax.device_put(states, sharded),
        jax.device_put(key, replicated),
    )


def equilibrate(engine, theta, seed):
    """Start fresh fixed-number chains and thermalize against the represented PEPS."""
    K = tc.backend
    key, draw = K.random_split(K.get_random_state(seed))
    theta, states, key = distribute(theta, engine.initial_chains(draw), key)
    warm = K.jit(lambda p, s, k: engine.advance(p, s, k, 8 * engine.peps.nsites))
    states, key, _ = warm(theta, states, key)
    return theta, states, key


def advance_phase(engine, state, dt, steps, imaginary, label):
    """Use ten-step compiled chunks, checking finite diagnostics at each boundary."""
    K = tc.backend
    evolve = K.jit(
        lambda p, s, k: engine.trajectory(
            p, s, k, dt, 10, -1.0 if imaginary else 0.0, imaginary
        )
    )
    start = time.perf_counter()
    for step in range(0, steps, 10):
        theta, states, key, diagnostics = evolve(*state)
        diagnostics = np.asarray(diagnostics)
        if not np.all(np.isfinite(diagnostics)) or not np.all(
            np.isfinite(np.asarray(theta))
        ):
            raise FloatingPointError("Nonfinite trajectory or unconverged SR solve.")
        state = theta, states, key
        print(
            f"{label}: {step + 10}/{steps}, E={diagnostics[-1, 0]:.8f}, "
            f"elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )
    return state


def simulate(
    samples=10240, measurement_samples=4194304, seed=DEFAULT_SEED, solver="dense"
):
    """Prepare from random tensors, evolve to t=8, and independently measure four frames."""
    K = tc.backend
    # Preserve the published default streams while letting --seed shift every stage.
    bond_seed = 7109 + (seed - DEFAULT_SEED)
    measurement_seed = 9147 + (seed - DEFAULT_SEED)
    batch = min(samples, 2560 * len(jax.devices()))
    small = FermionPEPS(6, 6, 2)
    _, draw = K.random_split(K.get_random_state(seed))
    theta = small.random_parameters(draw)
    engine = MonteCarlo(
        Hofstadter(small, 24),
        chains=min(samples, 2048),
        solver=solver,
        regulator=1e-3,
        contraction_batch_size=batch,
    )
    state = equilibrate(engine, theta, seed + 1)
    theta, _, _ = advance_phase(engine, state, 0.05, 100, True, "prepare D=2")
    peps = FermionPEPS(6, 6, 4)
    theta = enlarge_bond(
        jax.device_put(np.asarray(theta), jax.devices()[0]), small, peps, seed=bond_seed
    )
    problem = Hofstadter(peps, 24)
    for phase, (count, steps, ridge) in enumerate(
        ((min(samples, 4096), 100, 1e-3), (samples, 60, 1e-4))
    ):
        engine = MonteCarlo(
            problem, count, solver=solver, regulator=ridge, contraction_batch_size=batch
        )
        state = equilibrate(engine, theta, seed + 2 + phase)
        theta, _, _ = advance_phase(
            engine, state, 0.05, steps, True, f"prepare D=4/{phase}"
        )
    engine = MonteCarlo(
        problem, samples, solver=solver, regulator=1e-5, contraction_batch_size=batch
    )
    state = equilibrate(engine, theta, seed + 4)
    times = np.asarray([0.0, 2.0, 4.0, 8.0])
    time_step = 0.2
    records, measured = [], []
    for i, physical_time in enumerate(times):
        if i:
            steps = int(round((physical_time - times[i - 1]) / time_step))
            state = advance_phase(
                engine, state, time_step, steps, False, f"evolve to t={physical_time:g}"
            )
        record, arrays = measure(
            problem,
            state[0],
            physical_time,
            samples=measurement_samples,
            seed=measurement_seed,
            batch_size=2048 * len(jax.devices()),
        )
        print(
            f"t={physical_time:g}: RMSE={record['density_rmse']:.6g}, "
            f"max error={record['density_max_error']:.6g}, "
            f"infidelity={record['estimated_infidelity']:.6g}",
            flush=True,
        )
        records.append(record)
        measured.append(arrays)
    orbitals = K.eigh(problem.one_body())[1][:, : problem.particles]
    summary = {
        name: np.stack([frame[name] for frame in measured])
        for name in ("density", "density_stderr", "exact_density")
    }
    summary.update(
        times=times,
        background=np.asarray(K.sum(K.abs(orbitals) ** 2, axis=1)),
        infidelity=np.asarray([r["estimated_infidelity"] for r in records]),
        infidelity_stderr=np.asarray([r["jackknife_stderr"] for r in records]),
        effective_samples=np.asarray(
            [r["weight_effective_sample_size"] for r in records]
        ),
        samples=samples,
        measurement_samples=measurement_samples,
        time_step=time_step,
        seed=seed,
        solver=solver,
    )
    return summary


def plot(summary, output):
    """Draw PEPS and exact densities on the same fixed color scale."""
    times, background = summary["times"], summary["background"]
    density, exact = summary["density"], summary["exact_density"]
    np.testing.assert_allclose(density.sum(axis=1), 24.0, atol=1e-10)
    np.testing.assert_allclose(exact.sum(axis=1), 24.0, atol=1e-10)
    fig, axes = plt.subplots(2, len(times), figsize=(10.2, 5.2), squeeze=False)
    fig.subplots_adjust(
        left=0.10, right=0.86, bottom=0.08, top=0.82, hspace=0.16, wspace=0.12
    )
    for column, physical_time in enumerate(times):
        for row, values in enumerate((exact - background, density - background)):
            ax = axes[row, column]
            ax.imshow(
                values[column].reshape(6, 6),
                origin="lower",
                interpolation="nearest",
                cmap="viridis_r",
                vmin=0.0,
                vmax=0.04,
            )
            ax.set_xticks([0, 5])
            ax.set_yticks([0, 5])
            ax.tick_params(labelsize=8, length=2)
            if row == 0:
                ax.set_title(f"t = {physical_time:g}", fontsize=12)
            if column == 0:
                ax.set_ylabel(("Exact (FGS)", "PEPS-tVMC")[row], fontsize=11)
    fig.colorbar(
        axes[0, 0].images[0],
        cax=fig.add_axes([0.885, 0.08, 0.018, 0.74]),
        label="Density excess δn",
        extend="both",
    )
    fig.suptitle(
        "6 × 6 corner quench · 24 fermions · D = 4\n"
        "Independent PEPS measurements; shared 0–0.04 color scale",
        fontsize=13,
    )
    fig.savefig(output / "result.png", dpi=180)
    plt.close(fig)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--samples", type=int, default=10240)
    parser.add_argument("--measurement-samples", type=int, default=4194304)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--solver", choices=("dense", "cg"), default="dense")
    args = parser.parse_args()
    if args.samples < 2 or args.measurement_samples < 64:
        parser.error("Use at least two chains and 64 independent measurement samples.")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with np.load(args.output / "summary.npz") as saved:
            summary = dict(saved)
    else:
        summary = simulate(
            args.samples, args.measurement_samples, args.seed, args.solver
        )
        np.savez_compressed(args.output / "summary.npz", **summary)
    plot(summary, args.output)
    print(f"Simulation, measurement and plotting: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
