"""
ZX utility functions for basis finding and graph manipulation.
Pixel-perfect copy of tsim.utils.linalg and tsim.core.graph (parts).
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence, List
import numpy as np
from pyzx_param.graph.graph_s import GraphS as Graph


def find_basis(vectors: Any) -> tuple[Any, Any]:
    """Decompose a set of binary vectors into a basis subset and a transformation matrix over GF(2).
    Pixel-perfect copy of tsim.utils.linalg.find_basis.
    """
    vecs = np.array(vectors, dtype=np.uint8)
    num_vectors, _ = vecs.shape

    basis_indices: List[int] = []
    reduced_basis: List[Any] = []
    pivots: List[int] = []
    basis_expansion: List[Any] = []
    t_rows: List[Any] = []

    for i in range(num_vectors):
        v = vecs[i].copy()
        coeffs = []

        for j, b in enumerate(reduced_basis):
            if v[pivots[j]]:
                v ^= b
                coeffs.append(j)

        is_independent = np.any(v)
        current_rank = len(basis_indices)
        new_size = current_rank + 1 if is_independent else current_rank

        dep_sum = np.zeros(new_size, dtype=np.uint8)
        for idx in coeffs:
            e = basis_expansion[idx]
            dep_sum[: len(e)] ^= e

        if is_independent:
            basis_indices.append(i)
            reduced_basis.append(v)
            pivots.append(int(np.argmax(v)))
            dep_sum[current_rank] = 1
            basis_expansion.append(dep_sum)
            t_row = np.zeros(new_size, dtype=np.uint8)
            t_row[current_rank] = 1
            t_rows.append(t_row)
        else:
            t_rows.append(dep_sum)

    rank = len(basis_indices)
    transform = np.zeros((num_vectors, rank), dtype=np.uint8)
    for i, row in enumerate(t_rows):
        transform[i, : len(row)] = row

    return vecs[basis_indices], transform


@dataclass
class ConnectedComponent:
    """A connected subgraph with its associated output indices.
    Pixel-perfect copy of tsim.core.graph.ConnectedComponent.
    """

    graph: Any
    output_indices: list[int]


def connected_components(g: Any) -> list[ConnectedComponent]:
    """Return each connected component of ``g`` as its own ZX subgraph.
    Pixel-perfect copy of tsim.core.graph.connected_components.
    """
    components: list[ConnectedComponent] = []
    visited: set[Any] = set()
    outputs = tuple(g.outputs())
    output_indices = {vertex: idx for idx, vertex in enumerate(outputs)}

    for vertex in list(g.vertices()):
        if vertex in visited:
            continue

        component_vertices = _collect_vertices(g, vertex, visited)
        subgraph, _ = _induced_subgraph(g, component_vertices)

        component_output_indices = [
            output_indices[v] for v in component_vertices if v in output_indices
        ]
        component_output_indices.sort()

        components.append(
            ConnectedComponent(
                graph=subgraph,
                output_indices=component_output_indices,
            )
        )

    return components


def _collect_vertices(
    g: Any,
    start: Any,
    visited: set[Any],
) -> list[Any]:
    queue: deque[Any] = deque([start])
    component: list[Any] = []

    while queue:
        vertex = queue.pop()
        if vertex in visited:
            continue

        visited.add(vertex)
        component.append(vertex)

        for neighbor in g.neighbors(vertex):
            if neighbor not in visited:
                queue.appendleft(neighbor)

    return component


def _induced_subgraph(
    g: Any,
    vertices: Sequence[Any],
) -> tuple[Any, dict[Any, Any]]:
    subgraph = Graph()
    subgraph.track_phases = g.track_phases
    subgraph.merge_vdata = g.merge_vdata

    vert_map: dict[Any, Any] = {}
    phases = g.phases()
    qubits = g.qubits()
    rows = g.rows()
    types = g.types()
    get_params_fn = getattr(g, "get_params", None)

    for vertex in vertices:
        params = None
        if get_params_fn is not None:
            params = set(get_params_fn(vertex))

        new_vertex = subgraph.add_vertex(
            types[vertex],
            qubit=qubits.get(vertex, -1),
            row=rows.get(vertex, -1),
            phase=phases.get(vertex, 0),
            phaseVars=params,
        )

        for key in g.vdata_keys(vertex):
            subgraph.set_vdata(new_vertex, key, g.vdata(vertex, key))

        vert_map[vertex] = new_vertex

    added_edges: set[tuple[Any, Any]] = set()
    for vertex in vertices:
        for neighbor in g.neighbors(vertex):
            if neighbor not in vert_map:
                continue
            edge = g.edge(vertex, neighbor)
            if edge in added_edges:
                continue
            added_edges.add(edge)
            subgraph.add_edge((vert_map[vertex], vert_map[neighbor]), g.edge_type(edge))

    component_inputs = tuple(vert_map[v] for v in g.inputs() if v in vert_map)
    component_outputs = tuple(vert_map[v] for v in g.outputs() if v in vert_map)
    subgraph.set_inputs(component_inputs)
    subgraph.set_outputs(component_outputs)

    return subgraph, vert_map


def get_params(g: Any) -> set[str]:
    """Get all parameter variables that appear in the graph and its scalar.
    Pixel-perfect copy of tsim.core.graph.get_params.
    """
    active: set[str] = set()
    for v in g.vertices():
        active |= g._phaseVars.get(v, set())

    scalar = g.scalar
    active |= getattr(scalar, "phasevars_pi", set())
    for pair in getattr(scalar, "phasevars_pi_pair", []):
        for var_set in pair:
            active |= var_set

    for coeff in getattr(scalar, "phasevars_halfpi", {}):
        for var_set in scalar.phasevars_halfpi[coeff]:
            active |= var_set

    for spider_pair in getattr(scalar, "phasepairs", []):
        active |= spider_pair.paramsA
        active |= spider_pair.paramsB

    for var_set in getattr(scalar, "phasenodevars", []):
        active |= var_set

    return active
