from typing import List, Set, Tuple

import pytest
import numpy as np

from tensorcircuit.templates.circuit_utils import get_nn_gate_layers
from tensorcircuit.templates.lattice import (
    AbstractLattice,
    ChainLattice,
    HoneycombLattice,
    SquareLattice,
)


class MockLattice(AbstractLattice):
    """A mock lattice class for testing purposes to precisely control neighbors."""

    def __init__(self, neighbor_pairs: List[Tuple[int, int]]):
        super().__init__(dimensionality=0)
        self._neighbor_pairs = neighbor_pairs

    def get_neighbor_pairs(
        self, k: int = 1, unique: bool = True
    ) -> List[Tuple[int, int]]:
        return self._neighbor_pairs

    def _build_lattice(self, *args, **kwargs) -> None:
        pass

    def _build_neighbors(self, max_k: int = 1, **kwargs) -> None:
        pass

    def _compute_distance_matrix(self) -> np.ndarray:
        return np.array([])


def _validate_layers(
    lattice: AbstractLattice, layers: List[List[Tuple[int, int]]]
) -> None:
    """
    A helper function to scientifically validate the output of get_nn_gate_layers.
    """
    expected_edges = set(lattice.get_neighbor_pairs(k=1, unique=True))
    actual_edges = set(tuple(sorted(edge)) for layer in layers for edge in layer)

    assert (
        expected_edges == actual_edges
    ), "Completeness check failed: The set of all edges in the layers must "
    "exactly match the lattice's unique nearest-neighbor pairs."

    for i, layer in enumerate(layers):
        qubits_in_layer: Set[int] = set()
        for edge in layer:
            q1, q2 = edge
            assert (
                q1 not in qubits_in_layer
            ), f"Compatibility check failed: Qubit {q1} is reused in layer {i}."
            qubits_in_layer.add(q1)
            assert (
                q2 not in qubits_in_layer
            ), f"Compatibility check failed: Qubit {q2} is reused in layer {i}."
            qubits_in_layer.add(q2)


@pytest.mark.parametrize(
    "lattice_instance",
    [
        SquareLattice(size=(3, 2), pbc=False),
        SquareLattice(size=(3, 3), pbc=True),
        HoneycombLattice(size=(2, 2), pbc=False),
    ],
    ids=[
        "SquareLattice_3x2_OBC",
        "SquareLattice_3x3_PBC",
        "HoneycombLattice_2x2_OBC",
    ],
)
def test_various_lattices_layering(lattice_instance: AbstractLattice):
    """Tests gate layering for various standard lattice types."""
    layers = get_nn_gate_layers(lattice_instance)
    assert len(layers) > 0, "Layers should not be empty for non-trivial lattices."
    _validate_layers(lattice_instance, layers)


def test_1d_chain_pbc():
    """Test layering on a 1D chain with periodic boundaries (a cycle graph)."""
    lattice_even = ChainLattice(size=(6,), pbc=True)
    layers_even = get_nn_gate_layers(lattice_even)
    _validate_layers(lattice_even, layers_even)

    lattice_odd = ChainLattice(size=(5,), pbc=True)
    layers_odd = get_nn_gate_layers(lattice_odd)
    assert len(layers_odd) == 3, "A 5-site cycle graph should be 3-colorable."
    _validate_layers(lattice_odd, layers_odd)


def test_custom_star_graph():
    """Test layering on a custom lattice forming a star graph."""
    star_edges = [(0, 1), (0, 2), (0, 3)]
    lattice = MockLattice(star_edges)
    layers = get_nn_gate_layers(lattice)
    assert len(layers) == 3, "A star graph S_4 requires 3 layers."
    _validate_layers(lattice, layers)


def test_edge_cases():
    """Test various edge cases: empty, single-site, and no-edge lattices."""
    empty_lattice = MockLattice([])
    layers = get_nn_gate_layers(empty_lattice)
    assert layers == [], "Layers should be empty for an empty lattice."

    single_site_lattice = MockLattice([])
    layers = get_nn_gate_layers(single_site_lattice)
    assert layers == [], "Layers should be empty for a single-site lattice."

    disconnected_lattice = MockLattice([])
    layers = get_nn_gate_layers(disconnected_lattice)
    assert layers == [], "Layers should be empty for a lattice with no neighbors."

    single_edge_lattice = MockLattice([(0, 1)])
    layers = get_nn_gate_layers(single_edge_lattice)
    # The tuple inside the list might be (0, 1) or (1, 0) after sorting.
    # We check for the sorted version to be deterministic.
    assert layers == [[(0, 1)]]
    _validate_layers(single_edge_lattice, layers)
