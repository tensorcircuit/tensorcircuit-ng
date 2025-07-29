from typing import List, Set, Tuple

from .lattice import AbstractLattice


def get_nn_gate_layers(lattice: AbstractLattice) -> List[List[Tuple[int, int]]]:
    """
    Partitions nearest-neighbor pairs into compatible layers for parallel
    gate application using a greedy edge-coloring algorithm.

    In quantum circuits, a single qubit cannot participate in more than one
    two-qubit gate simultaneously. This function takes a lattice geometry,
    finds its nearest-neighbor graph, and partitions the edges of that graph
    (the neighbor pairs) into the minimum number of sets ("layers") where
    no two edges in a set share a vertex.

    This is essential for efficiently scheduling gates in algorithms like
    Trotterized Hamiltonian evolution.

    :Example:

    >>> import numpy as np
    >>> from tensorcircuit.templates.lattice import SquareLattice
    >>> sq_lattice = SquareLattice(size=(2, 2), pbc=False)
    >>> gate_layers = get_nn_gate_layers(sq_lattice)
    >>> print(gate_layers)
    [[[0, 1], [2, 3]], [[0, 2], [1, 3]]]

    :param lattice: An initialized `AbstractLattice` object from which to
        extract nearest-neighbor connectivity.
    :type lattice: AbstractLattice
    :return: A list of layers. Each layer is a list of tuples, where each
        tuple represents a nearest-neighbor pair (i, j) of site indices.
        All pairs within a layer are non-overlapping.
    :rtype: List[List[Tuple[int, int]]]
    """
    uncolored_edges: Set[Tuple[int, int]] = set(
        lattice.get_neighbor_pairs(k=1, unique=True)
    )

    layers: List[List[Tuple[int, int]]] = []

    while uncolored_edges:
        current_layer: List[Tuple[int, int]] = []
        qubits_in_this_layer: Set[int] = set()
        edges_to_remove: Set[Tuple[int, int]] = set()

        for edge in sorted(list(uncolored_edges)):
            i, j = edge
            if i not in qubits_in_this_layer and j not in qubits_in_this_layer:
                current_layer.append(edge)
                qubits_in_this_layer.add(i)
                qubits_in_this_layer.add(j)
                edges_to_remove.add(edge)

        layers.append(sorted(current_layer))
        uncolored_edges -= edges_to_remove

    return layers
