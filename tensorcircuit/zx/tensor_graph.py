"""Backend-native contraction of ZX graphs with runtime phase inputs."""

from cmath import exp
from dataclasses import dataclass
from math import pi, sqrt
from typing import Any, Sequence

import jax
import pyzx_param as zx
from networkx.utils import UnionFind
from pyzx_param.utils import EdgeType, VertexType

from ..cons import backend, dtypestr


@jax.tree_util.register_pytree_node_class
@dataclass
class TensorGraph:
    """A static contraction plan with differentiable phase-port tensors."""

    operands: tuple[Any, ...]
    steps: tuple[Any, ...]
    parameter_ports: tuple[Any, ...]
    scalar: complex

    def tree_flatten(self) -> Any:
        return self.operands, (self.steps, self.parameter_ports, self.scalar)

    @classmethod
    def tree_unflatten(cls, metadata: Any, operands: Any) -> "TensorGraph":
        steps, parameter_ports, scalar = metadata
        return cls(tuple(operands), steps, parameter_ports, scalar)

    def evaluate(self, values: Any) -> Any:
        def single(bits: Any) -> Any:
            tensors = list(self.operands)
            for position, indices in self.parameter_ports:
                parity = (
                    backend.sum(
                        backend.gather1d(bits, backend.convert_to_tensor(indices))
                    )
                    % 2
                )
                tensors[position] = tensors[position] * backend.stack(
                    [backend.ones((), dtype=dtypestr), 1 - 2 * parity]
                )
            for positions, equation in self.steps:
                operands = [tensors.pop(i) for i in positions]
                tensors.append(backend.einsum(equation, *operands))
            return self.scalar * tensors[0]

        return backend.vmap(single)(values)


def compile_tensor_graph(
    graph: Any, phase_weights: Sequence[Any], params: Sequence[str] = ()
) -> TensorGraph:
    """Protect runtime phases as open legs, simplify, and plan a contraction."""
    import opt_einsum as oe

    g = graph.copy()
    zx.to_gh(g)
    inputs = tuple(v for v in g.inputs() if g.type(v) == VertexType.BOUNDARY)
    outputs = tuple(v for v in g.outputs() if g.type(v) == VertexType.BOUNDARY)
    g.set_inputs(inputs)
    ports = {}
    parameter_indices = {name: i for i, name in enumerate(params)}
    for v in list(g.vertices()):
        index = g.vdata(v, "tensor_phase", None)
        variables = set(g.get_params(v))
        if index is None and not variables:
            continue
        phase = float(g.phase(v)) + ("1" in variables)
        weight = backend.convert_to_tensor([1.0, exp(1j * pi * phase)])
        if index is not None:
            dynamic = phase_weights[index]
            if g.vdata(v, "conjugate_phase", False):
                dynamic = backend.conj(dynamic)
            weight = weight * dynamic
        indices = tuple(
            parameter_indices[p]
            for p in sorted(variables - {"1"})
            if p in parameter_indices
        )
        port = g.add_vertex(VertexType.BOUNDARY)
        g.add_edge((v, port), EdgeType.SIMPLE)
        ports[port] = (backend.cast(weight, dtypestr), indices)
        g.set_phase(v, 0)
        g.set_params(v, set())
    g.set_outputs(outputs + tuple(ports))
    zx.full_reduce(g, paramSafe=True)
    zx.to_gh(g)

    # Z spiders share binary indices; do not materialize degree-sized tensors.
    labels = UnionFind(g.vertices())
    for edge in g.edges():
        if g.edge_type(edge) == EdgeType.SIMPLE:
            labels.union(*g.edge_st(edge))
    roots = dict.fromkeys(labels[v] for v in g.vertices())
    symbols = {label: oe.get_symbol(i) for i, label in enumerate(roots)}
    operands: list[Any] = []
    terms = []
    parameter_ports = []
    for v in g.vertices():
        symbol = symbols[labels[v]]
        if v in ports:
            weight, indices = ports[v]
            if indices:
                parameter_ports.append((len(operands), indices))
            operands.append(weight)
            terms.append(symbol)
        elif g.type(v) == VertexType.Z:
            operands.append(
                backend.convert_to_tensor(
                    [1.0, exp(1j * pi * float(g.phase(v)))], dtype=dtypestr
                )
            )
            terms.append(symbol)
        elif g.type(v) != VertexType.BOUNDARY:
            raise ValueError(
                "Runtime phase contraction requires Z spiders and boundaries"
            )
    hadamard = backend.convert_to_tensor([[1, 1], [1, -1]], dtype=dtypestr) / sqrt(2.0)
    for edge in g.edges():
        if g.edge_type(edge) == EdgeType.HADAMARD:
            operands.append(hadamard)
            terms.append("".join(symbols[labels[v]] for v in g.edge_st(edge)))
    external = []
    for i, v in enumerate(outputs + inputs):
        symbol = oe.get_symbol(len(symbols) + i)
        operands.append(backend.eye(2, dtype=dtypestr))
        terms.append(symbols[labels[v]] + symbol)
        external.append(symbol)
    if not operands:
        operands.append(backend.ones((), dtype=dtypestr))
        terms.append("")
    expression = ",".join(terms) + "->" + "".join(external)
    plan = oe.contract_expression(
        expression, *(tuple(t.shape) for t in operands), optimize="greedy"
    )
    steps = tuple((tuple(c[0]), c[2]) for c in plan.contraction_list)
    return TensorGraph(
        tuple(operands), steps, tuple(parameter_ports), complex(g.scalar.to_number())
    )
