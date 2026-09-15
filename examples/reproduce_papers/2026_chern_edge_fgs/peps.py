"""
Number-projected fermionic PEPS in the swap-gate formulation.

The physical order is i = y * columns + x, with y increasing upward.
Physical legs leave toward the upper left; a leg at (x, y) crosses the
upward virtual bonds to its left. See Eqs. (2), (3), (8) of
https://arxiv.org/abs/2506.20106. Virtual indices alternate even/odd parity.
"""

import cmath
import itertools
import math

import tencirpauli as tcp
import tensorcircuit as tc
from peps_boundary_mps import apply_grid_row_dmrg, peps_partition_function


class FermionPEPS:
    """Open, even-parity PEPS; pack only parity-allowed tensor entries."""

    def __init__(self, rows, columns, bond_dim=2, boundary_dim=None):
        if rows < 2 or columns < 2 or bond_dim < 2 or bond_dim % 2:
            raise ValueError("Use a rectangle of at least 2 x 2 and an even D >= 2.")
        if boundary_dim is not None and boundary_dim < 1:
            raise ValueError("The boundary-MPS cap must be positive.")
        self.rows, self.columns = rows, columns
        self.nsites = rows * columns
        self.bond_dim, self.boundary_dim = bond_dim, boundary_dim
        self.contraction_key = tc.backend.get_random_state(0)
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
        self.batch_amplitude = tc.backend.vmap(self.amplitude, vectorized_argnums=1)
        self.value_derivative = tc.backend.value_and_grad(
            lambda theta, occupation: tc.backend.real(self.amplitude(theta, occupation))
        )
        self.batch_scores = tc.backend.vmap(self.scores, vectorized_argnums=1)
        self._make_gauge_indices()

    def _make_gauge_indices(self):
        """Sparse generators of parity-preserving virtual GL(D/2) x GL(D/2)."""
        output, source, columns = [], [], []
        lookups = [dict(zip(c, range(len(c)))) for c in self.coordinates]
        column = 0
        for i, j, ai, aj in self.bonds:
            for a in range(self.bond_dim):
                for b in range(self.bond_dim):
                    if (a - b) % 2:
                        continue
                    for site, axis, old, new, sign in (
                        (i, ai, a, b, 1),
                        (j, aj, b, a, -1),
                    ):
                        for k, coordinate in enumerate(self.coordinates[site]):
                            if coordinate[axis] == new:
                                changed = list(coordinate)
                                changed[axis] = old
                                output.append(self.offsets[site] + k)
                                source.append(
                                    self.offsets[site] + lookups[site][tuple(changed)]
                                )
                                columns.append((column, sign))
                    column += 1
        self.gauge_output = tc.backend.convert_to_tensor(output)
        self.gauge_source = tc.backend.convert_to_tensor(source)
        self.gauge_columns = tc.backend.convert_to_tensor([c for c, _ in columns])
        self.gauge_signs = tc.backend.convert_to_tensor([s for _, s in columns])
        self.ngenerators = column
        self.physical_index = tc.backend.convert_to_tensor(
            [c[0] for site in self.coordinates for c in site]
        )

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
        """
        Contract a sampled single-layer fermionic PEPS.

        boundary_dim=None uses TensorCircuit's exact contractor. Finite caps call
        TensorCircuit-NG's existing variational boundary-MPS contractor.
        """
        K = tc.backend
        tensors = self.sliced_tensors(theta, occupation)
        if self.boundary_dim is not None:
            return peps_partition_function(
                self.padded_grid(tensors),
                self.boundary_dim,
                self.contraction_key,
                num_sweeps=2,
            )
        nodes = [tc.Gate(tensor) for tensor in tensors]
        for i, j, ai, aj in self.bonds:
            nodes[i][ai - 1] ^ nodes[j][aj - 1]
        result = tc.contractor(nodes, ignore_edge_order=True).tensor
        return K.reshape(result, ())

    def padded_grid(self, tensors):
        """Adapt sliced tensors to the existing contractor's uniform grid layout."""
        K = tc.backend
        padded = []
        for tensor in tensors:
            values = K.transpose(tensor, (2, 3, 0, 1))
            coordinates = list(itertools.product(*(range(d) for d in values.shape)))
            grid = K.scatter(
                K.zeros((self.bond_dim,) * 4, dtype="complex128"),
                K.convert_to_tensor(coordinates),
                K.reshape(values, (-1,)),
            )
            padded.append(grid)
        return K.reshape(
            K.stack(padded), (self.rows, self.columns) + (self.bond_dim,) * 4
        )

    def boundary_scores(self, theta, occupation):
        """Hole environments, Eq. (12) of arXiv:2506.20106; no AD through compression."""
        K = tc.backend
        sliced = self.sliced_tensors(theta, occupation)
        grid = self.padded_grid(sliced)
        chi, D = self.boundary_dim, self.bond_dim
        initial = K.scatter(
            K.zeros((chi, D, chi), dtype="complex128"),
            K.convert_to_tensor([[0, 0, 0]]),
            K.ones((1,), dtype="complex128"),
        )
        initial = K.stack([initial] * self.columns)
        lower, upper = [initial], [initial]
        key = self.contraction_key
        for y in range(self.rows - 1):
            key, draw = K.random_split(key)
            value, _, _ = apply_grid_row_dmrg(lower[-1], grid[y], chi, draw)
            lower.append(value)
            value, _, _ = apply_grid_row_dmrg(
                upper[-1],
                K.transpose(grid[self.rows - 1 - y], (0, 2, 1, 3, 4)),
                chi,
                draw,
            )
            upper.append(value)
        upper = upper[::-1]
        endpoint = K.scatter(
            K.zeros((chi, chi, D), dtype="complex128"),
            K.convert_to_tensor([[0, 0, 0]]),
            K.ones((1,), dtype="complex128"),
        )
        scores = []
        for y in range(self.rows):
            left = [endpoint]
            right = [endpoint]
            for x in range(self.columns):
                left.append(
                    K.einsum(
                        "abl,adA,buB,dulr->ABr",
                        left[-1],
                        lower[y][x],
                        upper[y][x],
                        grid[y, x],
                    )
                )
                k = self.columns - 1 - x
                right.append(
                    K.einsum(
                        "ABr,adA,buB,dulr->abl",
                        right[-1],
                        lower[y][k],
                        upper[y][k],
                        grid[y, k],
                    )
                )
            right = right[::-1]
            for x in range(self.columns):
                i = y * self.columns + x
                hole = K.einsum(
                    "abl,ABr,adA,buB->lrdu",
                    left[x],
                    right[x + 1],
                    lower[y][x],
                    upper[y][x],
                )
                shape = self.shapes[i]
                hole = hole[: shape[1], : shape[2], : shape[3], : shape[4]]
                denominator = K.sum(hole * sliced[i])
                sign = 1 - 2 * K.mod(
                    K.sum(occupation[i + 1 : (y + 1) * self.columns])
                    * K.arange(shape[-1]),
                    2,
                )
                derivative = (
                    K.onehot(occupation[i], 2)[:, None, None, None, None]
                    * (hole * sign[None, None, None, :] / denominator)[None]
                )
                scores.append(K.reshape(derivative, (-1,))[self.allowed[i]])
        return self.amplitude(theta, occupation), K.concat(scores)

    def scores(self, theta, occupation):
        """Holomorphic derivative of log amplitude, with JAX's complex convention."""
        if self.boundary_dim is not None:
            return self.boundary_scores(theta, occupation)
        amplitude = self.amplitude(theta, occupation)
        _, derivative = self.value_derivative(theta, occupation)
        return amplitude, derivative / amplitude

    def gauge_vectors(self, theta):
        """Include virtual gauge, global scale, and fixed-number scale directions."""
        K = tc.backend
        indices = K.stack([self.gauge_output, self.gauge_columns], axis=1)
        vectors = K.scatter(
            K.zeros((self.nparams, self.ngenerators), dtype="complex128"),
            indices,
            theta[self.gauge_source] * self.gauge_signs,
        )
        return K.concat(
            [vectors, theta[:, None], (theta * self.physical_index)[:, None]], axis=1
        )

    def gauge_projector(self, theta):
        """Rank-revealing orthogonal projector removes dependent gauge generators."""
        K = tc.backend
        vectors = self.gauge_vectors(theta)
        u, singular, _, _ = K.svd(vectors)
        active = K.cast(K.real(singular) > K.real(singular[0]) * 1e-10, "complex128")
        return K.eye(self.nparams, dtype="complex128") - (u * active) @ K.adjoint(u)


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
        phases, permutations, terms = [], [], []
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
            permutation = list(range(self.nsites))
            permutation[i], permutation[j] = permutation[j], permutation[i]
            permutations.append(permutation)
        self.hopping = K.stack([K.cast(h, "complex128") for h in phases])
        self.permutations = K.convert_to_tensor(permutations)
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
