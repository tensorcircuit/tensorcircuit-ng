"""
Converter from TensorCircuit to PyZX.
Includes graph preparation and reduction utilities.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Dict, List, Optional, cast, Callable, Sequence

import numpy as np
import pyzx_param as pyzx
from pyzx_param.graph.scalar import Scalar
from pyzx_param.graph.graph_s import GraphS
from pyzx_param.utils import VertexType, EdgeType

from ..abstractcircuit import AbstractCircuit
from .. import gates as tcgates
from .utils import find_basis
from .noise_model import (
    pauli_channel_1_probs,
    pauli_channel_2_probs,
    error_probs,
)


def is_pauli(matrix: Any) -> Optional[str]:
    """
    Check if a matrix is a Pauli matrix.

    :param matrix: The matrix to check.
    :type matrix: np.ndarray
    :return: The name of the Pauli matrix ('i', 'x', 'y', 'z') or None.
    :rtype: Optional[str]
    """
    for name, gate in zip(
        ["i", "x", "y", "z"], [tcgates.i(), tcgates.x(), tcgates.y(), tcgates.z()]  # type: ignore
    ):
        if np.allclose(matrix, gate.tensor, atol=1e-5):
            return name
    return None


@dataclass
class SamplingGraph:
    graph: Any
    error_transform: Any
    channel_probs: list[Any]
    num_outputs: int
    num_detectors: int
    num_error_bits: int
    observables: list[int] = field(default_factory=list)


@dataclass
class GraphRepresentation:
    graph: GraphS = field(default_factory=GraphS)
    rec: List[int] = field(default_factory=list)
    silent_rec: List[int] = field(default_factory=list)
    detectors: List[int] = field(default_factory=list)
    observables_dict: Dict[int, int] = field(default_factory=dict)
    first_vertex: Dict[int, int] = field(default_factory=dict)
    last_vertex: Dict[int, int] = field(default_factory=dict)
    channel_probs: List[Any] = field(default_factory=list)
    correlated_error_probs: List[float] = field(default_factory=list)
    num_error_bits: int = 0
    num_correlated_error_bits: int = 0

    @property
    def observables(self) -> list[int]:
        """
        List of observable vertices.

        :return: List of vertex indices.
        :rtype: list[int]
        """
        return [self.observables_dict[i] for i in sorted(self.observables_dict)]

    def add_edge(self, e: Any, t: Any = EdgeType.SIMPLE) -> None:
        self.graph.add_edge(e, t)

    def remove_edge(self, e: Any) -> None:
        self.graph.remove_edge(e)

    def add_vertex(
        self, t: Any = VertexType.Z, qubit: int = -1, row: float = -1, phase: Any = 0
    ) -> int:
        """
        Add a vertex to the graph.

        :param t: Vertex type, defaults to VertexType.Z.
        :type t: Any, optional
        :param qubit: Qubit index, defaults to -1.
        :type qubit: int, optional
        :param row: Row index for layout, defaults to -1.
        :type row: float, optional
        :param phase: Vertex phase, defaults to 0.
        :type phase: Any, optional
        :return: The index of the new vertex.
        :rtype: int
        """
        v = self.graph.add_vertex(t, qubit, row)
        self.graph.set_phase(v, phase)
        return v

    def remove_vertex(self, v: Any) -> None:
        self.graph.remove_vertex(v)

    def remove_vertices(self, vertices: list[Any]) -> None:
        for v in vertices:
            self.graph.remove_vertex(v)

    def remove_edges(self, edges: list[tuple[Any, Any]]) -> None:
        for e in edges:
            self.graph.remove_edge(e)

    def vertex_set(self) -> Any:
        return self.graph.vertex_set()

    def edge_set(self) -> Any:
        return self.graph.edge_set()

    def num_vertices(self) -> int:
        return self.graph.num_vertices()  # type: ignore[no-any-return]

    def num_edges(self) -> int:
        return self.graph.num_edges()  # type: ignore[no-any-return]

    def incident_edges(self, v: Any) -> Any:
        """
        Get edges incident to a vertex.

        :param v: Vertex index.
        :type v: Any
        :return: Iterable of edges.
        :rtype: Any
        """
        return self.graph.incident_edges(v)

    def edge_st(self, e: Any) -> Any:
        """
        Get the endpoints of an edge.

        :param e: Edge.
        :type e: Any
        :return: Tuple of vertex indices (v1, v2).
        :rtype: Any
        """
        return self.graph.edge_st(e)

    def add_edges(self, edges: list[tuple[Any, Any]], t: Any = EdgeType.SIMPLE) -> None:
        for e in edges:
            self.graph.add_edge(e, t)

    def add_edge_table(self, etab: dict[tuple[Any, Any], list[int]]) -> None:
        for (v1, v2), ets in etab.items():
            for et in ets:
                if et != 0:
                    self.graph.add_edge((v1, v2), et)

    def qubit(self, v: Any) -> Any:
        return self.graph.qubit(v)

    def set_qubit(self, v: Any, q: Any) -> None:
        self.graph.set_qubit(v, q)

    def row(self, v: Any) -> Any:
        return self.graph.row(v)

    def is_ground(self, v: Any) -> bool:
        return self.graph.is_ground(v)  # type: ignore[no-any-return]

    def vertex_degree(self, v: Any) -> int:
        return self.graph.vertex_degree(v)  # type: ignore[no-any-return]

    def remove_isolated_vertices(self) -> None:
        for v in list(self.graph.vertices()):
            if self.graph.vertex_degree(v) == 0:
                self.graph.remove_vertex(v)

    def get_params(self, v: Any) -> Any:
        return self.graph.get_params(v)

    def edges(self, *args: Any) -> Any:
        return self.graph.edges(*args)

    def set_inputs(self, v: Any) -> None:
        self.graph.set_inputs(v)

    def set_outputs(self, v: Any) -> None:
        self.graph.set_outputs(v)

    def phase(self, v: Any) -> Any:
        return self.graph.phase(v)

    def set_phase(self, v: Any, p: Any) -> None:
        self.graph.set_phase(v, p)

    def add_to_phase(self, v: Any, p: Any, params: Any = None) -> None:
        self.graph.add_to_phase(v, p, params)

    def set_ground(self, v: Any, g: bool = True) -> None:
        self.graph.set_ground(v, g)

    def update_phase_index(self, v1: Any, v2: Any) -> None:
        if hasattr(self.graph, "update_phase_index"):
            self.graph.update_phase_index(v1, v2)

    def fuse_phases(self, v1: Any, v2: Any) -> None:
        if hasattr(self.graph, "fuse_phases"):
            self.graph.fuse_phases(v1, v2)

    def set_row(self, v: Any, r: Any) -> None:
        self.graph.set_row(v, r)

    def neighbors(self, v: Any) -> Any:
        return self.graph.neighbors(v)

    def to_tensor(self) -> Any:
        return self.graph.to_tensor()

    def types(self) -> Any:
        return self.graph.types()

    def phases(self) -> Any:
        return self.graph.phases()

    def qubits(self) -> Any:
        return self.graph.qubits()

    def rows(self) -> Any:
        return self.graph.rows()

    def vdata_keys(self, v: Any) -> Any:
        return self.graph.vdata_keys(v)

    def vdata(self, v: Any, key: str) -> Any:
        return self.graph.vdata(v, key)

    def set_vdata(self, v: Any, key: str, val: Any) -> None:
        self.graph.set_vdata(v, key, val)

    def get_auto_simplify(self) -> bool:
        return self.graph.get_auto_simplify()

    def set_auto_simplify(self, v: bool) -> None:
        self.graph.set_auto_simplify(v)

    def is_multigraph(self) -> bool:
        return self.graph.is_multigraph()  # type: ignore

    def edge(self, v1: Any, v2: Any) -> Any:
        return self.graph.edge(v1, v2)

    def edge_type(self, e: Any) -> Any:
        return self.graph.edge_type(e)

    def set_edge_type(self, e: Any, t: Any) -> None:
        self.graph.set_edge_type(e, t)

    def vertices(self) -> Any:
        return self.graph.vertices()

    def copy(self) -> GraphRepresentation:
        new_b = replace(self)
        assert isinstance(new_b, GraphRepresentation)
        new_b.graph = cast(GraphS, self.graph.copy())
        new_b.rec = list(self.rec)
        new_b.silent_rec = list(self.silent_rec)
        new_b.detectors = list(self.detectors)
        new_b.observables_dict = dict(self.observables_dict)
        new_b.first_vertex = dict(self.first_vertex)
        new_b.last_vertex = dict(self.last_vertex)
        new_b.channel_probs = list(self.channel_probs)
        new_b.correlated_error_probs = list(self.correlated_error_probs)
        return new_b

    def inputs(self) -> Any:
        return self.graph.inputs()

    def outputs(self) -> Any:
        return self.graph.outputs()

    def type(self, v: Any) -> Any:
        return self.graph.type(v)

    def set_type(self, v: Any, t: Any) -> None:
        self.graph.set_type(v, t)

    @property
    def _phaseVars(self) -> Any:
        return self.graph._phaseVars

    @property
    def scalar(self) -> Any:
        return self.graph.scalar

    @scalar.setter
    def scalar(self, v: Any) -> None:
        self.graph.scalar = v

    @property
    def track_phases(self) -> bool:
        return self.graph.track_phases

    @track_phases.setter
    def track_phases(self, v: bool) -> None:
        self.graph.track_phases = v

    @property
    def merge_vdata(self) -> Any:
        return self.graph.merge_vdata

    @merge_vdata.setter
    def merge_vdata(self, v: Any) -> None:
        self.graph.merge_vdata = v


def last_row(b: GraphRepresentation, qubit: int) -> float:
    """
    Get the row index of the last vertex on a qubit lane.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: The qubit index.
    :type qubit: int
    :return: Highest row index on the lane.
    :rtype: float
    """
    return float(b.graph.row(b.last_vertex[qubit]))  # type: ignore[no-any-return]


def last_edge(b: GraphRepresentation, qubit: int) -> Any:
    """
    Get the last edge on a qubit lane.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: The qubit index.
    :type qubit: int
    :return: The last edge index.
    :rtype: Any
    """
    return list(b.graph.incident_edges(b.last_vertex[qubit]))[0]


def add_dummy(
    b: GraphRepresentation, qubit: int, row: float | int | None = None
) -> int:
    """
    Add a dummy boundary vertex to a qubit lane.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: The qubit index.
    :type qubit: int
    :param row: Row index, defaults to last_row + 1.
    :type row: float | int | None, optional
    :return: The index of the new vertex.
    :rtype: int
    """
    if row is None:
        row = last_row(b, qubit) + 1
    v1 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=row)
    b.last_vertex[qubit] = v1
    return v1


def add_lane(b: GraphRepresentation, qubit: int) -> int:
    """
    Initialize a new qubit lane with two boundary vertices.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: The qubit index.
    :type qubit: int
    :return: The index of the first vertex.
    :rtype: int
    """
    v1 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=0)
    v2 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=1)
    b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
    b.first_vertex[qubit], b.last_vertex[qubit] = v1, v2
    return v1


def ensure_lane(b: GraphRepresentation, qubit: int) -> None:
    """
    Ensure a qubit lane exists in the graph.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: The qubit index.
    :type qubit: int
    """
    if qubit not in b.last_vertex:
        add_lane(b, qubit)


def x_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """
    Apply an X spider with the given phase to a qubit.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: The qubit index.
    :type qubit: int
    :param phase: Phase in units of π.
    :type phase: Fraction
    """
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.X)
    b.graph.set_phase(v1, phase)
    # Correctly handle lane update
    v2 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=b.graph.row(v1) + 1)
    b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
    b.last_vertex[qubit] = v2


def z_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """
    Apply a Z spider with the given phase to a qubit.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: The qubit index.
    :type qubit: int
    :param phase: Phase in units of π.
    :type phase: Fraction
    """
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.Z)
    b.graph.set_phase(v1, phase)
    # Correctly handle lane update
    v2 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=b.graph.row(v1) + 1)
    b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
    b.last_vertex[qubit] = v2


def x_gate(b: GraphRepresentation, qubit: int) -> None:
    """Apply Pauli X gate."""
    x_phase(b, qubit, Fraction(1, 1))


def y_gate(b: GraphRepresentation, qubit: int) -> None:
    """Apply Pauli Y gate."""
    z_gate(b, qubit)
    x_gate(b, qubit)
    b.graph.scalar.add_phase(Fraction(1, 2))


def z_gate(b: GraphRepresentation, qubit: int) -> None:
    """Apply Pauli Z gate."""
    z_phase(b, qubit, Fraction(1, 1))


def h_gate(b: GraphRepresentation, qubit: int) -> None:
    """
    Apply Hadamard gate.
    """
    ensure_lane(b, qubit)
    e = last_edge(b, qubit)
    b.graph.set_edge_type(
        e,
        (
            EdgeType.HADAMARD
            if b.graph.edge_type(e) == EdgeType.SIMPLE
            else EdgeType.SIMPLE
        ),
    )


def h_xy(b: GraphRepresentation, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the X and Y axes (instead of X and Z)."""
    x_gate(b, qubit)
    _s_gate(b, qubit)
    b.graph.scalar.add_phase(Fraction(-1, 4))


def h_yz(b: GraphRepresentation, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the Y and Z axes (instead of X and Z)."""
    sqrt_x(b, qubit)
    z_gate(b, qubit)
    b.graph.scalar.add_phase(Fraction(-1, 4))


def sqrt_x(b: GraphRepresentation, qubit: int) -> None:
    """
    Apply SQRT_X gate.
    """
    x_phase(b, qubit, Fraction(1, 2))


def sqrt_x_dag(b: GraphRepresentation, qubit: int) -> None:
    """
    Apply SQRT_X_DAG gate.
    """
    x_phase(b, qubit, Fraction(-1, 2))


def sqrt_y(b: GraphRepresentation, qubit: int) -> None:
    """
    Apply SQRT_Y gate.
    """
    z_gate(b, qubit)
    h_gate(b, qubit)
    b.graph.scalar.add_phase(Fraction(1, 4))


def sqrt_y_dag(b: GraphRepresentation, qubit: int) -> None:
    """
    Apply SQRT_Y_DAG gate.
    """
    h_gate(b, qubit)
    z_gate(b, qubit)
    b.graph.scalar.add_phase(Fraction(-1, 4))


def sqrt_z(b: GraphRepresentation, qubit: int) -> None:
    """
    Apply SQRT_Z gate.
    """
    _s_gate(b, qubit)


def sqrt_z_dag(b: GraphRepresentation, qubit: int) -> None:
    """
    Apply SQRT_Z_DAG gate.
    """
    _s_dag_gate(b, qubit)


def y_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """
    Apply Y-basis spider with the given phase.
    """
    h_yz(b, qubit)
    z_phase(b, qubit, phase)
    h_yz(b, qubit)


def r_z(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply R_Z rotation gate with given phase (in units of π)."""
    z_phase(b, qubit, phase)
    b.graph.scalar.add_phase(-phase / 2)


def r_x(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply R_X rotation gate with given phase (in units of π)."""
    x_phase(b, qubit, phase)
    b.graph.scalar.add_phase(-phase / 2)


def r_y(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply R_Y rotation gate with given phase (in units of π)."""
    h_yz(b, qubit)
    r_z(b, qubit, phase)
    h_yz(b, qubit)


def u3(
    b: GraphRepresentation,
    qubit: int,
    theta: Fraction,
    phi: Fraction,
    lambda_: Fraction,
) -> None:
    """Apply U3 gate: U3(θ,φ,λ) = R_Z(φ)·R_Y(θ)·R_Z(λ)."""
    r_z(b, qubit, lambda_)
    r_y(b, qubit, theta)
    r_z(b, qubit, phi)
    b.graph.scalar.add_phase((phi + lambda_) / 2)


def _cx_cz(b: GraphRepresentation, is_cx: bool, control: int, target: int) -> None:
    ensure_lane(b, control)
    ensure_lane(b, target)
    v1 = b.last_vertex[control]
    v3 = b.last_vertex[target]

    row = max(b.graph.row(v1), b.graph.row(v3))

    b.graph.set_row(v1, row)
    b.graph.set_row(v3, row)

    # CX: Z on control, X on target, SIMPLE edge
    # CZ: Z on control, Z on target, HADAMARD edge
    if is_cx:
        b.graph.set_type(v1, VertexType.Z)
        v2 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=control, row=row + 1)
        b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
        b.last_vertex[control] = v2

        b.graph.set_type(v3, VertexType.X)
        v4 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=target, row=row + 1)
        b.graph.add_edge((v3, v4), EdgeType.SIMPLE)
        b.last_vertex[target] = v4
        b.graph.add_edge((v1, v3), EdgeType.SIMPLE)
    else:
        b.graph.set_type(v1, VertexType.Z)
        v2 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=control, row=row + 1)
        b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
        b.last_vertex[control] = v2

        b.graph.set_type(v3, VertexType.Z)
        v4 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=target, row=row + 1)
        b.graph.add_edge((v3, v4), EdgeType.SIMPLE)
        b.last_vertex[target] = v4
        b.graph.add_edge((v1, v3), EdgeType.HADAMARD)
    b.graph.scalar.add_power(1)


def _m(b: GraphRepresentation, qubit: int, p: float = 0, silent: bool = False) -> None:
    if p > 0:
        x_error(b, qubit, p)
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.Z)
    if not silent:
        b.graph.set_phase(v1, f"rec[{len(b.rec)}]")
        b.rec.append(v1)
    else:
        b.graph.set_phase(v1, f"m[{len(b.silent_rec)}]")
        b.silent_rec.append(v1)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
    b.graph.scalar.add_power(-1)


def _r(b: GraphRepresentation, qubit: int, perform_trace: bool) -> None:
    if qubit not in b.last_vertex:
        v1 = add_lane(b, qubit)
        b.graph.set_type(v1, VertexType.X)
        b.graph.scalar.add_power(-1)
    else:
        if perform_trace:
            _m(b, qubit, silent=True)
        row = last_row(b, qubit)
        v1 = b.last_vertex[qubit]
        b.graph.set_type(v1, VertexType.X)
        edges = list(b.graph.incident_edges(v1))
        [b.graph.remove_edge(e) for e in edges]
        v2 = add_dummy(b, qubit, row + 1)
        b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
        b.graph.scalar.add_power(-1)


def detector(b: GraphRepresentation, rec: list[int]) -> None:
    """
    Add a detector vertex defined by a set of measurement outcomes.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param rec: Indices of measurement records involved in the detector.
    :type rec: list[int]
    """
    rec_vertices = [b.rec[r] for r in rec]
    row = min((b.graph.row(v) for v in rec_vertices), default=0) - 0.5
    v0 = b.graph.add_vertex(VertexType.X, qubit=-1, row=row)
    b.graph.set_phase(v0, f"det[{len(b.detectors)}]")
    for v in rec_vertices:
        b.graph.add_edge((v0, v))
    b.detectors.append(v0)


def observable_include(b: GraphRepresentation, rec: list[int], idx: int) -> None:
    """
    Add an observable vertex defined by a set of measurement outcomes.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param rec: Indices of measurement records involved in the observable.
    :type rec: list[int]
    :param idx: Index of the observable.
    :type idx: int
    """
    idx = int(idx)
    rec_vertices = [b.rec[r] for r in rec]
    if idx not in b.observables_dict:
        row = min((b.graph.row(v) for v in rec_vertices), default=0) - 0.5
        v0 = b.graph.add_vertex(VertexType.X, qubit=-1, row=row)
        b.graph.set_phase(v0, f"obs[{idx}]")
        b.observables_dict[idx] = v0
    v0 = b.observables_dict[idx]
    for v in rec_vertices:
        b.graph.add_edge((v0, v))


def depolarize1(b: GraphRepresentation, qubit: int, p: float) -> None:
    """
    Apply single-qubit depolarizing channel.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: Qubit index.
    :type qubit: int
    :param p: Depolarizing probability.
    :type p: float
    """
    pauli_channel_1(b, qubit, p / 3, p / 3, p / 3)


def depolarize2(b: GraphRepresentation, q1: int, q2: int, p: float) -> None:
    """
    Apply two-qubit depolarizing channel.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param q1: First qubit index.
    :type q1: int
    :param q2: Second qubit index.
    :type q2: int
    :param p: Depolarizing probability.
    :type p: float
    """
    b.channel_probs.append(pauli_channel_2_probs(*([p / 15] * 15)))
    for i in range(4):
        _error(
            b,
            q1 if i < 2 else q2,
            VertexType.Z if i % 2 == 0 else VertexType.X,
            f"e{b.num_error_bits + i}",
        )
    b.num_error_bits += 4


def pauli_channel_1(
    b: GraphRepresentation, qubit: int, px: float = 0, py: float = 0, pz: float = 0
) -> None:
    """
    Apply single-qubit Pauli channel.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: Qubit index.
    :type qubit: int
    :param px: Probability of X error, defaults to 0.
    :type px: float, optional
    :param py: Probability of Y error, defaults to 0.
    :type py: float, optional
    :param pz: Probability of Z error, defaults to 0.
    :type pz: float, optional
    """
    b.channel_probs.append(pauli_channel_1_probs(px, py, pz))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits + 1}")
    b.num_error_bits += 2


def pauli_channel_2(
    b: GraphRepresentation,
    qubit_i: int,
    qubit_j: int,
    pix: float = 0,
    piy: float = 0,
    piz: float = 0,
    pxi: float = 0,
    pxx: float = 0,
    pxy: float = 0,
    pxz: float = 0,
    pyi: float = 0,
    pyx: float = 0,
    pyy: float = 0,
    pyz: float = 0,
    pzi: float = 0,
    pzx: float = 0,
    pzy: float = 0,
    pzz: float = 0,
) -> None:
    """
    Apply two-qubit Pauli channel.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit_i: First qubit index.
    :type qubit_i: int
    :param qubit_j: Second qubit index.
    :type qubit_j: int
    :param pix: Probability of error IX.
    ...
    """
    b.channel_probs.append(
        pauli_channel_2_probs(
            pix, piy, piz, pxi, pxx, pxy, pxz, pyi, pyx, pyy, pyz, pzi, pzx, pzy, pzz
        )
    )
    _error(b, qubit_i, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit_i, VertexType.X, f"e{b.num_error_bits + 1}")
    _error(b, qubit_j, VertexType.Z, f"e{b.num_error_bits + 2}")
    _error(b, qubit_j, VertexType.X, f"e{b.num_error_bits + 3}")
    b.num_error_bits += 4


def x_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.channel_probs.append(error_probs(p))
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def z_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.channel_probs.append(error_probs(p))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def _error(b: GraphRepresentation, qubit: int, error_type: int, phase: str) -> None:
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2), EdgeType.SIMPLE)
    b.graph.set_type(v1, error_type)
    b.graph.set_phase(v1, phase)


def m(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Z basis with optional bit-flip error probability p."""
    if invert:
        x_phase(b, qubit, Fraction(1, 1))
    _m(b, qubit, p, silent=False)
    if invert:
        x_phase(b, qubit, Fraction(1, 1))


def y_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    """Apply Y error with probability p."""
    b.channel_probs.append(error_probs(p))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def mr(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Z-basis demolition measurement (optionally noisy)."""
    if p > 0:
        x_error(b, qubit, p)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit, perform_trace=False)


def mrx(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """X-basis demolition measurement (optionally noisy)."""
    h_gate(b, qubit)
    if p > 0:
        x_error(b, qubit, p)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit, perform_trace=False)
    h_gate(b, qubit)


def mry(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Y-basis demolition measurement (optionally noisy)."""
    h_yz(b, qubit)
    if p > 0:
        x_error(b, qubit, p)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit, perform_trace=False)
    h_yz(b, qubit)


def mrz(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Z-basis demolition measurement (optionally noisy)."""
    mr(b, qubit, p=p, invert=invert)


def mx(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """
    Apply X-basis measurement.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: Qubit index.
    :type qubit: int
    :param p: Error probability, defaults to 0.
    :type p: float, optional
    :param invert: Whether to invert measurement result, defaults to False.
    :type invert: bool, optional
    """
    h_gate(b, qubit)
    m(b, qubit, p=p, invert=invert)
    h_gate(b, qubit)


def my(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """
    Apply Y-basis measurement.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: Qubit index.
    :type qubit: int
    :param p: Error probability, defaults to 0.
    :type p: float, optional
    :param invert: Whether to invert measurement result, defaults to False.
    :type invert: bool, optional
    """
    h_yz(b, qubit)
    m(b, qubit, p=p, invert=invert)
    h_yz(b, qubit)


def reset_z(b: GraphRepresentation, qubit: int, p: float = 0) -> None:
    """
    Reset qubit to |0> state (Z-basis reset).

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: Qubit index.
    :type qubit: int
    :param p: Error probability, defaults to 0.
    :type p: float, optional
    """
    if p > 0:
        x_error(b, qubit, p)
    _r(b, qubit, perform_trace=True)


def reset_x(b: GraphRepresentation, qubit: int) -> None:
    """
    Reset qubit in X basis.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: Qubit index.
    :type qubit: int
    """
    if qubit in b.last_vertex:
        h_gate(b, qubit)
    reset_z(b, qubit)
    h_gate(b, qubit)


def reset_y(b: GraphRepresentation, qubit: int) -> None:
    """
    Reset qubit in Y basis.

    :param b: The graph representation.
    :type b: GraphRepresentation
    :param qubit: Qubit index.
    :type qubit: int
    """
    if qubit in b.last_vertex:
        h_yz(b, qubit)
    reset_z(b, qubit)
    h_yz(b, qubit)


def _swap(b: GraphRepresentation, q1: int, q2: int) -> None:
    ensure_lane(b, q1)
    ensure_lane(b, q2)
    b.last_vertex = {**b.last_vertex, q1: b.last_vertex[q2], q2: b.last_vertex[q1]}


def _y_gate(b: GraphRepresentation, q: int) -> None:
    z_phase(b, q, Fraction(1, 1))
    x_phase(b, q, Fraction(1, 1))
    b.graph.scalar.add_phase(Fraction(1, 2))


def _h_xy(b: GraphRepresentation, q: int) -> None:
    z_phase(b, q, Fraction(1, 1))
    x_phase(b, q, Fraction(1, 1))
    z_phase(b, q, Fraction(1, 2))
    b.graph.scalar.add_phase(Fraction(-1, 4))


def _s_gate(b: GraphRepresentation, q: int) -> None:
    z_phase(b, q, Fraction(1, 2))


def _s_dag_gate(b: GraphRepresentation, q: int) -> None:
    z_phase(b, q, Fraction(-1, 2))


def _t_gate(b: GraphRepresentation, q: int) -> None:
    z_phase(b, q, Fraction(1, 4))


def _t_dag_gate(b: GraphRepresentation, q: int) -> None:
    z_phase(b, q, Fraction(-1, 4))


def _sqrt_z_gate(b: GraphRepresentation, q: int) -> None:
    z_phase(b, q, Fraction(1, 2))


# NOTE on Gate Naming Convention:
# -----------------------------
# We distinguish between demolition measurements (resets) and parametric rotations:
# - RX, RY, RZ (no underscore): Demolition measurements (resets) in X, Y, Z bases.
#   These map to reset_x, reset_y, and reset_z internally.
# - R_X, R_Y, R_Z (with underscore): Parametric rotation gates with 'theta' parameter.


GATE_TABLE: Dict[str, tuple[Callable[..., Any], int]] = {
    # ---- Pauli gates -----------------------------------------------------------
    "I": (lambda b, q: None, 1),
    "X": (x_gate, 1),
    "Y": (y_gate, 1),
    "Z": (z_gate, 1),
    # ---- Non-Clifford gates ---------------------------------------------------
    "S": (_s_gate, 1),
    "SD": (_s_dag_gate, 1),
    "S_DAG": (_s_dag_gate, 1),
    "T": (_t_gate, 1),
    "TD": (_t_dag_gate, 1),
    "T_DAG": (_t_dag_gate, 1),
    "SQRT_X": (sqrt_x, 1),
    "SQRT_X_DAG": (sqrt_x_dag, 1),
    "SQRT_Y": (sqrt_y, 1),
    "SQRT_Y_DAG": (sqrt_y_dag, 1),
    "H": (h_gate, 1),
    "H_XY": (_h_xy, 1),
    "SQRT_Z": (_sqrt_z_gate, 1),
    "H_YZ": (h_yz, 1),
    "H_XZ": (h_gate, 1),
    "R_Z": (r_z, 1),
    "R_X": (r_x, 1),
    "R_Y": (r_y, 1),
    "U3": (u3, 1),
    # ---- Two-qubit gates ------------------------------------------------------
    "CNOT": (lambda b, c, t: _cx_cz(b, True, c, t), 2),
    "CX": (lambda b, c, t: _cx_cz(b, True, c, t), 2),
    "CZ": (lambda b, c, t: _cx_cz(b, False, c, t), 2),
    "SWAP": (_swap, 2),
    # ---- Collapsing gates -----------------------------------------------------
    "M": (m, 1),
    "R": (reset_z, 1),
    "MEASURE": (m, 1),
    "RESET": (reset_z, 1),
    "RESET_X": (reset_x, 1),
    "RESET_Y": (reset_y, 1),
    "RESET_Z": (reset_z, 1),
    "MR": (mr, 1),
    "MRX": (mrx, 1),
    "MRY": (mry, 1),
    "MRZ": (mr, 1),
    "MX": (mx, 1),
    "MY": (my, 1),
    "MZ": (m, 1),
    "RX": (reset_x, 1),
    "RY": (reset_y, 1),
    "RZ": (reset_z, 1),
}


def circuit_to_zx(
    c: AbstractCircuit, force_measure_all: bool = False
) -> GraphRepresentation:
    """
    Convert a TensorCircuit AbstractCircuit to a ZX-calculus GraphRepresentation.

    :param c: The source circuit.
    :type c: AbstractCircuit
    :param force_measure_all: Whether to force measurements on all qubits at the end, defaults to False.
    :type force_measure_all: bool, optional
    :return: The ZX graph representation of the circuit.
    :rtype: GraphRepresentation
    """
    b = GraphRepresentation()
    b.graph.track_phases = True
    n = c._nqubits
    if hasattr(c, "_merge_qir"):
        merged_qir = c._merge_qir()
    else:
        merged_qir_with_pos = sorted(
            [(float(i), d) for i, d in enumerate(c._qir)]
            + [
                (
                    float(d.get("pos", len(c._qir)))
                    + (
                        0.5
                        if d.get("name", "").upper()
                        in ["DETECTOR", "OBSERVABLE_INCLUDE"]
                        else 0.01
                    ),
                    d,
                )
                for d in getattr(c, "_extra_qir", [])
            ],
            key=lambda x: x[0],
        )
        merged_qir = [d for _, d in merged_qir_with_pos]
    for i, d in enumerate(merged_qir):
        name, index, params = (
            str(d.get("name", "")).upper(),
            list(d.get("index", ())),
            d.get("parameters", {}),
        )
        p = d.get("p", params.get("p", 0.0))
        px = d.get("px", params.get("px", 0.0))
        py = d.get("py", params.get("py", 0.0))
        pz = d.get("pz", params.get("pz", 0.0))
        probs = d.get("probs", params.get("probs"))
        if name in ["DEPOLARIZE1", "DEPOLARIZING"]:
            if p > 0:
                depolarize1(b, index[0], p)
            else:
                pauli_channel_1(b, index[0], px, py, pz)
        elif name in ["DEPOLARIZE2", "DEPOLARIZING2"]:
            if p > 0:
                depolarize2(b, index[0], index[1], p)
            else:
                pauli_channel_2(b, index[0], index[1], px, py, pz)
        elif name in ["PAULI_CHANNEL_1", "PAULI"]:
            if probs is not None:
                b.channel_probs.append(np.array(probs, dtype=np.float64))
                _error(b, index[0], VertexType.Z, f"e{b.num_error_bits}")
                _error(b, index[0], VertexType.X, f"e{b.num_error_bits + 1}")
                b.num_error_bits += 2
            else:
                pauli_channel_1(b, index[0], px, py, pz)
        elif name in ["PAULI_CHANNEL_2"]:
            pauli_channel_2(b, index[0], index[1], px, py, pz)
        elif name == "X_ERROR":
            x_error(b, index[0], p if p > 0 else px)
        elif name == "Y_ERROR":
            y_error(b, index[0], p if p > 0 else py)
        elif name == "Z_ERROR":
            z_error(b, index[0], p if p > 0 else pz)
        elif name == "DETECTOR":
            detector(b, index)
        elif name == "OBSERVABLE_INCLUDE":
            observable_include(
                b, index, d.get("observable_index", params.get("index", 0))
            )
        elif name in ["QUBIT_COORDS", "SHIFT_COORDS", "TICK"]:
            continue
        elif name in GATE_TABLE:
            func, num_qubits = GATE_TABLE[name]
            if name in ["R_X", "R_Y", "R_Z"]:
                theta = params.get("theta", params.get("phi", params.get("phase", 0.0)))
                if isinstance(theta, (float, int)):
                    theta = Fraction(theta) / np.pi
            elif name == "U3":
                theta = params.get("theta", 0.0)
                phi = params.get("phi", 0.0)
                lam = params.get("lambda", params.get("lam", 0.0))
                if isinstance(theta, (float, int)):
                    theta = Fraction(theta) / np.pi
                if isinstance(phi, (float, int)):
                    phi = Fraction(phi) / np.pi
                if isinstance(lam, (float, int)):
                    lam = Fraction(lam) / np.pi

            for i_target in range(0, len(index), num_qubits):
                chunk = index[i_target : i_target + num_qubits]
                if name in [
                    "M",
                    "R",
                    "MEASURE",
                    "RESET",
                    "MR",
                    "MRX",
                    "MRY",
                    "MRZ",
                    "MX",
                    "MY",
                    "MZ",
                ]:
                    func(b, *chunk, p=p)
                elif name in ["R_X", "R_Y", "R_Z"]:
                    func(b, *chunk, theta)
                elif name == "U3":
                    func(b, *chunk, theta, phi, lam)
                else:
                    func(b, *chunk)
        elif name == "":
            continue
        else:
            raise ValueError(
                f"Unknown instruction name: '{name}' in {d}. GATE_TABLE keys: {list(GATE_TABLE.keys())}"
            )
    if force_measure_all:
        for i in range(n):
            _m(b, i)
    for i in range(n):
        ensure_lane(b, i)
    b.graph.set_inputs(tuple(b.first_vertex[i] for i in sorted(b.first_vertex)))
    b.graph.set_outputs(tuple(b.last_vertex[i] for i in sorted(b.last_vertex)))
    return b


def build_sampling_graph(
    built: GraphRepresentation,
    sample_detectors: bool,
    pauli: Optional[Dict[int, str]] = None,
) -> GraphS:
    """
    Build a doubled sampling graph from a ZX representation.

    :param built: The input ZX representation.
    :type built: GraphRepresentation
    :param sample_detectors: Whether to prepare for detector/observable sampling or measurement records.
    :type sample_detectors: bool
    :param pauli: Optional Pauli string to insert at the junction for expectation values.
        Dictionary mapping qubit index to Pauli operator ('I', 'X', 'Y', 'Z').
    :type pauli: Optional[Dict[int, str]]
    :return: The prepared ZX graph.
    :rtype: GraphS
    """
    g = built.graph.copy()

    # Initialize un-initialized first vertices to the 0 state
    for v in built.first_vertex.values():
        if g.type(v) == VertexType.BOUNDARY:
            g.set_type(v, VertexType.X)

    # Clean up last row
    if built.last_vertex:
        max_row = max(g.row(v) for v in built.last_vertex.values())
        for q in built.last_vertex:
            g.set_row(built.last_vertex[q], max_row)

    num_m = len(built.rec)
    outputs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    g.set_outputs(tuple(outputs))

    # Graph doubling: composed with adjoint
    g_adj = g.adjoint()

    if pauli is not None:
        # Insert Pauli operators at the junction
        # We modify the output boundary types of g before composition
        outputs = list(g.outputs())
        for v in outputs:
            op = pauli.get(int(g.qubit(v)), "I").upper()
            if op == "I":
                continue
            nbs = list(g.neighbors(v))
            if not nbs:
                continue
            nb = nbs[0]
            et = g.edge_type((nb, v))
            g.remove_edge((nb, v))

            if op == "Z":
                mid = g.add_vertex(VertexType.Z, qubit=g.qubit(v), row=g.row(v) - 0.05)
                g.set_phase(mid, Fraction(1, 1))
                g.add_edge((nb, mid), et)
                g.add_edge((mid, v), EdgeType.SIMPLE)
            elif op == "X":
                mid = g.add_vertex(VertexType.X, qubit=g.qubit(v), row=g.row(v) - 0.05)
                g.set_phase(mid, Fraction(1, 1))
                g.add_edge((nb, mid), et)
                g.add_edge((mid, v), EdgeType.SIMPLE)
            elif op == "Y":
                # Y = i X Z (up to phase, which we'll handle)
                # We insert X and Z
                z_mid = g.add_vertex(
                    VertexType.Z, qubit=g.qubit(v), row=g.row(v) - 0.06
                )
                g.set_phase(z_mid, Fraction(1, 1))
                x_mid = g.add_vertex(
                    VertexType.X, qubit=g.qubit(v), row=g.row(v) - 0.03
                )
                g.set_phase(x_mid, Fraction(1, 1))

                g.add_edge((nb, z_mid), et)
                g.add_edge((z_mid, x_mid), EdgeType.SIMPLE)
                g.add_edge((x_mid, v), EdgeType.SIMPLE)
                # Y = iXZ. We add pi/2 phase to the scalar.
                g.scalar.add_phase(Fraction(1, 2))

    g.compose(g_adj)

    l2v = defaultdict(list)
    a2v = defaultdict(list)
    for v in g.vertices():
        pv = g._phaseVars[v]
        if len(pv) == 1:
            p = list(pv)[0]
            if any(x in p for x in ["det", "obs", "rec", "m"]):
                l2v[p].append(v)
            if any(x in p for x in ["det", "obs"]):
                a2v[p].append(v)

    outputs = [0] * num_m if not sample_detectors else []

    # Connect all rec[i] vertices
    for i in range(num_m):
        label = f"rec[{i}]"
        vertices = l2v[label]
        if not vertices:
            continue
        assert len(vertices) == 2
        v0, v1 = vertices
        if not g.connected(v0, v1):
            g.add_edge((v0, v1))
        g.set_phase(v0, 0)
        g.set_phase(v1, 0)
        # Substitute the rec variable with 0 as they cancel each other in doubling if not measured
        # Actually in sampling, we keep rec variables to build the measurement IR
        if not sample_detectors:
            v3 = g.add_vertex(VertexType.BOUNDARY, qubit=-1, row=i + 1)
            outputs[i] = v3
            g.add_edge((v0, v3))

    # Connect all m[i] vertices
    for i in range(len(built.silent_rec)):
        label = f"m[{i}]"
        vertices = l2v[label]
        assert len(vertices) == 2
        v0, v1 = vertices
        if not g.connected(v0, v1):
            g.add_edge((v0, v1))
        g.set_phase(v0, 0)
        g.set_phase(v1, 0)

    if not sample_detectors:
        # Remove detectors and observables
        for vs in a2v.values():
            assert len(vs) == 2
            for v in vs:
                g.remove_vertex(v)
    else:
        # Keep annotations but remove adjoint copies
        for vs in a2v.values():
            assert len(vs) == 2
            g.remove_vertex(vs.pop())

        labels = [f"det[{i}]" for i in range(len(built.detectors))] + [
            f"obs[{i}]" for i in sorted(built.observables_dict.keys())
        ]
        for label in labels:
            vs = a2v[label]
            assert len(vs) == 1
            v = vs[0]
            vb = g.add_vertex(
                VertexType.BOUNDARY, qubit=-2 if "det" in label else -2.5, row=g.row(v)
            )
            g.add_edge((v, vb))
            g.set_phase(v, 0)
            outputs.append(vb)

    g.set_outputs(tuple(outputs))
    return cast(GraphS, g)


def build_amplitude_graph(
    built: GraphRepresentation, bitstring: Sequence[int]
) -> GraphS:
    """
    Build a scalar graph representing the amplitude <bitstring|psi>.

    :param built: The input ZX representation.
    :type built: GraphRepresentation
    :param bitstring: The target bitstring as a sequence of 0s and 1s.
    :type bitstring: Sequence[int]
    :return: The scalar ZX graph.
    :rtype: GraphS
    """
    g = built.graph.copy()
    for v in built.first_vertex.values():
        if g.type(v) == VertexType.BOUNDARY:
            g.set_type(v, VertexType.X)
            g.set_phase(v, 0)

    outputs = list(g.outputs())
    for i, v in enumerate(outputs):
        val = 1 if i < len(bitstring) and bitstring[i] == 1 else 0
        # outputs in GraphRepresentation are typically Z-spiders or boundaries
        # We set them to X-spiders with phase pi * bit to project onto |bit>
        g.set_type(v, VertexType.X)
        g.set_phase(v, Fraction(val, 1))

    g.set_inputs(tuple())
    g.set_outputs(tuple())
    return cast(GraphS, g)


def transform_error_basis(g: Any, num_e: int | None = None) -> tuple[Any, Any]:
    """
    Transform error bit variables from original 'e' basis to a reduced 'f' basis.

    :param g: The ZX graph containing error variables.
    :type g: Any
    :param num_e: Total number of error bits, defaults to None.
    :type num_e: int, optional
    :return: A tuple of (transformed_graph, basis_transformation_matrix).
    :rtype: tuple[Any, Any]
    """
    p_v = [
        v
        for v in g.vertices()
        if v in g._phaseVars and any(var.startswith("e") for var in g._phaseVars[v])
    ]
    if not p_v:
        return g, np.zeros((0, num_e if num_e else 0), dtype=np.uint8)
    e_idx = [
        [int(var[1:]) for var in g._phaseVars[v] if var.startswith("e")] for v in p_v
    ]
    n_e = max((max(indices) for indices in e_idx), default=0) + 1
    if num_e:
        n_e = max(n_e, num_e)
    e_mat = np.zeros((len(e_idx), n_e), dtype=np.uint8)
    for r, idxs in enumerate(e_idx):
        e_mat[r, idxs] = 1
    basis, transform = find_basis(e_mat)
    for v, row in zip(p_v, transform):
        g._phaseVars[v] = {f"f{j}" for j in np.nonzero(row)[0]}
    return g, basis


def squash_graph(g: Any) -> None:
    """
    Compact the graph layout by re-assigning qubit and row indices.

    :param g: The ZX graph.
    :type g: Any
    """
    outputs = list(g.outputs())
    if not outputs:
        return
    n_o = len(outputs)
    for r, v in enumerate(outputs):
        g.set_row(v, r)
    occ, placed, q = {(n_o, r) for r in range(n_o)}, set(outputs), deque(outputs)
    while q:
        curr = q.popleft()
        cq, cr = int(g.qubit(curr)), int(g.row(curr))
        for nb in g.neighbors(curr):
            if nb in placed:
                continue
            tq, tr = cq - 1, cr
            if (tq, tr) in occ:
                for off in range(1, 1000):
                    if (tq, tr + off) not in occ:
                        tr += off
                        break
                    if (tq, tr - off) not in occ and tr - off >= 0:
                        tr -= off
                        break
            g.set_qubit(nb, tq)
            g.set_row(nb, tr)
            occ.add((tq, tr))
            placed.add(nb)
            q.append(nb)


def prepare_graph(
    circuit: AbstractCircuit,
    *,
    sample_detectors: bool,
    force_measure_all: bool = False,
    pauli: Optional[Dict[int, str]] = None,
    reset_scalar: bool = True,
) -> SamplingGraph:
    """
    Prepare a circuit for sampling by converting to ZX and reducing.

    :param circuit: The input circuit.
    :type circuit: AbstractCircuit
    :param sample_detectors: Whether to prepare for detector/observable sampling.
    :type sample_detectors: bool
    :param force_measure_all: Whether to force final measurements, defaults to False.
    :type force_measure_all: bool, optional
    :param pauli: Optional Pauli string to insert at the junction.
    :type pauli: Optional[Dict[int, str]]
    :return: A SamplingGraph object containing the reduced graph and metadata.
    :rtype: SamplingGraph
    """
    built = circuit_to_zx(circuit, force_measure_all=force_measure_all)
    graph = build_sampling_graph(built, sample_detectors=sample_detectors, pauli=pauli)
    pyzx.full_reduce(graph, paramSafe=True)
    squash_graph(graph)
    graph, error_transform = transform_error_basis(graph, num_e=built.num_error_bits)
    if reset_scalar:
        graph.scalar = Scalar()
    return SamplingGraph(
        graph,
        error_transform,
        built.channel_probs,
        len(graph.outputs()),
        len(built.detectors),
        built.num_error_bits,
        built.observables,
    )
