import typing
from typing import cast
from scipy.sparse import coo_matrix
import numpy as np
import tensorcircuit as tc
from .lattice import AbstractLattice


def generate_heisenberg_hamiltonian(
    lattice: AbstractLattice, j_coupling: float = 1.0
) -> coo_matrix:
    """
    Generates the sparse matrix of the Heisenberg Hamiltonian for a given lattice.

    The Heisenberg Hamiltonian is defined as:
    H = J * Σ_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j)
    where the sum is over all unique nearest-neighbor pairs <i,j>.

    :param lattice: An instance of a class derived from AbstractLattice,
        which provides the geometric information of the system.
    :type lattice: AbstractLattice
    :param j_coupling: The coupling constant for the Heisenberg interaction. Defaults to 1.0.
    :type j_coupling: float, optional
    :return: The Hamiltonian represented as a SciPy COO sparse matrix.
    :rtype: coo_matrix
    """
    num_sites = lattice.num_sites
    if num_sites == 0:
        return coo_matrix((0, 0))

    neighbor_pairs = lattice.get_neighbor_pairs(k=1, unique=True)
    if not neighbor_pairs:
        return coo_matrix((2**num_sites, 2**num_sites))

    pauli_map = {"X": 1, "Y": 2, "Z": 3}

    ls: typing.List[typing.List[int]] = []
    weights: typing.List[float] = []

    for i, j in neighbor_pairs:
        xx_string = [0] * num_sites
        xx_string[i] = pauli_map["X"]
        xx_string[j] = pauli_map["X"]
        ls.append(xx_string)
        weights.append(j_coupling)

        yy_string = [0] * num_sites
        yy_string[i] = pauli_map["Y"]
        yy_string[j] = pauli_map["Y"]
        ls.append(yy_string)
        weights.append(j_coupling)

        zz_string = [0] * num_sites
        zz_string[i] = pauli_map["Z"]
        zz_string[j] = pauli_map["Z"]
        ls.append(zz_string)
        weights.append(j_coupling)

    hamiltonian_matrix = tc.quantum.PauliStringSum2COO(ls, weight=weights, numpy=True)

    return cast(coo_matrix ,hamiltonian_matrix)


def generate_rydberg_hamiltonian(
    lattice: AbstractLattice, omega: float, delta: float, c6: float
) -> coo_matrix:
    """
    Generates the sparse matrix of the Rydberg atom array Hamiltonian.

    The Hamiltonian is defined as:
    H = Σ_i (Ω/2)X_i - Σ_i δ(1 - Z_i)/2 + Σ_{i<j} V_ij (1-Z_i)/2 (1-Z_j)/2
      = Σ_i (Ω/2)X_i + Σ_i (δ/2)Z_i + Σ_{i<j} (V_ij/4)(Z_iZ_j - Z_i - Z_j)
    where V_ij = C6 / |r_i - r_j|^6.

    Note: Constant energy offset terms (proportional to the identity operator)
    are ignored in this implementation.

    :param lattice: An instance of a class derived from AbstractLattice,
        which provides site coordinates and the distance matrix.
    :type lattice: AbstractLattice
    :param omega: The Rabi frequency (Ω) of the driving laser field.
    :type omega: float
    :param delta: The laser detuning (δ).
    :type delta: float
    :param c6: The Van der Waals interaction coefficient (C6).
    :type c6: float
    :return: The Hamiltonian represented as a SciPy COO sparse matrix.
    :rtype: coo_matrix
    """
    num_sites = lattice.num_sites
    if num_sites == 0:
        return coo_matrix((0, 0))

    pauli_map = {"X": 1, "Y": 2, "Z": 3}
    ls: typing.List[typing.List[int]] = []
    weights: typing.List[float] = []

    for i in range(num_sites):
        x_string = [0] * num_sites
        x_string[i] = pauli_map["X"]
        ls.append(x_string)
        weights.append(omega / 2.0)

    z_coefficients = np.zeros(num_sites)

    for i in range(num_sites):
        z_coefficients[i] += delta / 2.0

    dist_matrix = lattice.distance_matrix

    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            distance = dist_matrix[i, j]

            if distance < 1e-9:
                continue

            interaction_strength = c6 / (distance**6)
            coefficient = interaction_strength / 4.0

            zz_string = [0] * num_sites
            zz_string[i] = pauli_map["Z"]
            zz_string[j] = pauli_map["Z"]
            ls.append(zz_string)
            weights.append(coefficient)

            z_coefficients[i] -= coefficient
            z_coefficients[j] -= coefficient

    for i in range(num_sites):
        if abs(z_coefficients[i]) > 1e-9:
            z_string = [0] * num_sites
            z_string[i] = pauli_map["Z"]
            ls.append(z_string)
            weights.append(float(z_coefficients[i]))

    hamiltonian_matrix = tc.quantum.PauliStringSum2COO(ls, weight=weights, numpy=True)

    return cast(coo_matrix,hamiltonian_matrix)
