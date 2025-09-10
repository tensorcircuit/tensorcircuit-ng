from typing import Any, List, Tuple, Union
import numpy as np
from ..cons import dtypestr, backend
from ..quantum import PauliStringSum2COO
from .lattice import AbstractLattice


def _create_empty_sparse_matrix(shape: Tuple[int, int]) -> Any:
    """
    Helper function to create a backend-agnostic empty sparse matrix.
    """
    indices = backend.convert_to_tensor(backend.zeros((0, 2), dtype="int32"))
    values = backend.convert_to_tensor(backend.zeros((0,), dtype=dtypestr))  # type: ignore
    return backend.coo_sparse_matrix(indices=indices, values=values, shape=shape)  # type: ignore


def heisenberg_hamiltonian(
    lattice: AbstractLattice,
    j_coupling: Union[float, List[float], Tuple[float, ...]] = 1.0,
    interaction_scope: str = "neighbors",
) -> Any:
    r"""
    Generates the sparse matrix of the Heisenberg Hamiltonian for a given lattice.

    The Heisenberg Hamiltonian is defined as:
    :math:`H = J\sum_{i,j} (X_i X_j + Y_i Y_j + Z_i Z_j)`
    where the sum is over a specified set of interacting pairs {i,j}.

    :param lattice: An instance of a class derived from AbstractLattice,
        which provides the geometric information of the system.
    :type lattice: AbstractLattice
    :param j_coupling: The coupling constants. Can be a single float for an
        isotropic model (Jx=Jy=Jz) or a list/tuple of 3 floats for an
        anisotropic model (Jx, Jy, Jz). Defaults to 1.0.
    :type j_coupling: Union[float, List[float], Tuple[float, ...]], optional
    :param interaction_scope: Defines the range of interactions.
        - "neighbors": Includes only nearest-neighbor pairs (default).
        - "all": Includes all unique pairs of sites.
    :type interaction_scope: str, optional
    :return: The Hamiltonian as a backend-agnostic sparse matrix.
    :rtype: Any
    """
    num_sites = lattice.num_sites
    if interaction_scope == "neighbors":
        neighbor_pairs = lattice.get_neighbor_pairs(k=1, unique=True)
    elif interaction_scope == "all":
        neighbor_pairs = lattice.get_all_pairs()
    else:
        raise ValueError(
            f"Invalid interaction_scope: '{interaction_scope}'. "
            "Must be 'neighbors' or 'all'."
        )

    if isinstance(j_coupling, (float, int)):
        js = [float(j_coupling)] * 3
    else:
        if len(j_coupling) != 3:
            raise ValueError("j_coupling must be a float or a list/tuple of 3 floats.")
        js = [float(j) for j in j_coupling]

    if not neighbor_pairs:
        return _create_empty_sparse_matrix(shape=(2**num_sites, 2**num_sites))
    if num_sites == 0:
        raise ValueError("Cannot generate a Hamiltonian for a lattice with zero sites.")

    pauli_map = {"X": 1, "Y": 2, "Z": 3}

    ls: List[List[int]] = []
    weights: List[float] = []

    pauli_terms = ["X", "Y", "Z"]
    for i, j in neighbor_pairs:
        for idx, pauli_char in enumerate(pauli_terms):
            if abs(js[idx]) > 1e-9:
                string = [0] * num_sites
                string[i] = pauli_map[pauli_char]
                string[j] = pauli_map[pauli_char]
                ls.append(string)
                weights.append(js[idx])

    hamiltonian_matrix = PauliStringSum2COO(ls, weight=weights, numpy=False)

    return hamiltonian_matrix


def rydberg_hamiltonian(
    lattice: AbstractLattice, omega: float, delta: float, c6: float
) -> Any:
    r"""
    Generates the sparse matrix of the Rydberg atom array Hamiltonian.

    The Hamiltonian is defined as:
    .. math::

    H = \sum_i \frac{\Omega}{2} X_i
        - \sum_i \frac{\delta}{2} \bigl(1 - Z_i \bigr)
        + \sum_{i<j} \frac{V_{ij}}{4} \bigl(1 - Z_i \bigr)\bigl(1 - Z_j \bigr)

      = \sum_i \frac{\Omega}{2} X_i
        + \sum_i \frac{\delta}{2} Z_i
        + \sum_{i<j} \frac{V_{ij}}{4}\,\bigl(Z_i Z_j - Z_i - Z_j \bigr)

    where :math:`V_{ij} = C6 / |r_i - r_j|^6`.

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
    :return: The Hamiltonian as a backend-agnostic sparse matrix.
    :rtype: Any
    """
    num_sites = lattice.num_sites
    if num_sites == 0:
        raise ValueError("Cannot generate a Hamiltonian for a lattice with zero sites.")

    pauli_map = {"X": 1, "Y": 2, "Z": 3}
    ls: List[List[int]] = []
    weights: List[float] = []

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

            # The interaction term V_ij * n_i * n_j, when expanded using
            # n_i = (1-Z_i)/2, becomes (V_ij/4)*(I - Z_i - Z_j + Z_i*Z_j).
            # This contributes a positive term (+V_ij/4) to the ZZ interaction,
            # but negative terms (-V_ij/4) to the single-site Z_i and Z_j operators.

            z_coefficients[i] -= coefficient
            z_coefficients[j] -= coefficient

    for i in range(num_sites):
        if abs(z_coefficients[i]) > 1e-9:
            z_string = [0] * num_sites
            z_string[i] = pauli_map["Z"]
            ls.append(z_string)
            weights.append(z_coefficients[i])  # type: ignore

    hamiltonian_matrix = PauliStringSum2COO(ls, weight=weights, numpy=False)

    return hamiltonian_matrix
