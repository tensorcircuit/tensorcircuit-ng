"""
Module for functions adding layers of circuits (deprecated)
"""

import logging
import sys
from typing import Any, Optional, List, Sequence, Union

import networkx as nx

from ..templates import layers

logger = logging.getLogger(__name__)

try:
    import cirq
    import numpy as np
except ImportError as e:
    logger.warning(e)
    logger.warning("Therefore some functionality in %s may not work" % __name__)


thismodule = sys.modules[__name__]

Tensor = Any
Graph = Any  # nx.Graph
Symbol = Any  # sympy.Symbol

_resolve = layers._resolve


## below is similar layer but in cirq API instead of tensrocircuit native API
## special notes to the API, the arguments order are different due to historical reason compared to tc layers API
## and we have no attention to further maintain the cirq codebase below, availability is not guaranteend


def generate_qubits(g: Graph) -> List[Any]:
    return sorted([v for _, v in g.nodes.data("qubit")])


try:
    basis_rotation = {
        "x": (cirq.H, cirq.H),
        "y": (cirq.rx(-np.pi / 2), cirq.rx(np.pi / 2)),
        "z": None,
    }

    def generate_cirq_double_gate(gates: str) -> None:
        d1, d2 = gates[0], gates[1]
        r1, r2 = basis_rotation[d1], basis_rotation[d2]

        def f(
            circuit: cirq.Circuit,
            qubit1: cirq.GridQubit,
            qubit2: cirq.GridQubit,
            symbol: Symbol,
        ) -> cirq.Circuit:
            if r1 is not None:
                circuit.append(r1[0](qubit1))
            if r2 is not None:
                circuit.append(r2[0](qubit2))
            circuit.append(cirq.CNOT(qubit1, qubit2))
            circuit.append(cirq.rz(symbol)(qubit2))
            circuit.append(cirq.CNOT(qubit1, qubit2))
            if r1 is not None:
                circuit.append(r1[1](qubit1))
            if r2 is not None:
                circuit.append(r2[1](qubit2))
            return circuit

        f.__doc__ = """%sgate""" % gates
        setattr(thismodule, "cirq" + gates + "gate", f)

    def cirqswapgate(
        circuit: cirq.Circuit,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        symbol: Symbol,
    ) -> cirq.Circuit:
        circuit.append(cirq.SwapPowGate(exponent=symbol)(qubit1, qubit2))
        return circuit

    def cirqcnotgate(
        circuit: cirq.Circuit,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        symbol: Symbol,
    ) -> cirq.Circuit:
        circuit.append(cirq.CNOT(qubit1, qubit2))
        return circuit

    def generate_cirq_gate_layer(gate: str) -> None:
        r"""
        $$e^{-i\theta \sigma}$$

        :param gate:
        :type gate: str
        :return:
        """

        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            symbol0 = _resolve(symbol[0])
            if not qubits:
                qubits = generate_qubits(g)
            rotation = getattr(cirq, gate)
            if isinstance(rotation, cirq.Gate):
                circuit.append(rotation.on_each(qubits))
            else:  # function
                circuit.append(rotation(2.0 * symbol0).on_each(qubits))
            return circuit

        f.__doc__ = """%slayer""" % gate
        f.__repr__ = """%slayer""" % gate  # type: ignore
        f.__trainable__ = False if isinstance(getattr(cirq, gate), cirq.Gate) else True  # type: ignore
        setattr(thismodule, "cirq" + gate + "layer", f)

    def generate_cirq_any_gate_layer(gate: str) -> None:
        r"""
        $$e^{-i\theta \sigma}$$

        :param gate:
        :type gate: str
        :return:
        """

        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            if not qubits:
                qubits = generate_qubits(g)
            rotation = getattr(cirq, gate)
            for i, q in enumerate(qubits):
                circuit.append(rotation(2.0 * symbol[i])(q))
            return circuit

        f.__doc__ = """any%slayer""" % gate
        f.__repr__ = """any%slayer""" % gate  # type: ignore
        f.__trainable__ = True  # type: ignore
        setattr(thismodule, "cirqany" + gate + "layer", f)

    def generate_cirq_double_gate_layer(gates: str) -> None:
        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            symbol0 = _resolve(symbol)
            for e in g.edges:
                qubit1 = g.nodes[e[0]]["qubit"]
                qubit2 = g.nodes[e[1]]["qubit"]
                getattr(thismodule, "cirq" + gates + "gate")(
                    circuit, qubit1, qubit2, -symbol0 * g[e[0]][e[1]]["weight"] * 2
                )  ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
            return circuit

        f.__doc__ = """%slayer""" % gates
        f.__repr__ = """%slayer""" % gates  # type: ignore
        f.__trainable__ = True  # type: ignore
        setattr(thismodule, "cirq" + gates + "layer", f)

    def generate_cirq_any_double_gate_layer(gates: str) -> None:
        """
        The following function should be used to generate layers with special case.
        As its soundness depends on the nature of the task or problem, it doesn't always make sense.

        :param gates:
        :type gates: str
        :return:
        """

        def f(
            circuit: cirq.Circuit,
            g: Graph,
            symbol: Symbol,
            qubits: Optional[Sequence[Any]] = None,
        ) -> cirq.Circuit:
            for i, e in enumerate(g.edges):
                qubit1 = g.nodes[e[0]]["qubit"]
                qubit2 = g.nodes[e[1]]["qubit"]
                getattr(thismodule, "cirq" + gates + "gate")(
                    circuit, qubit1, qubit2, -symbol[i] * g[e[0]][e[1]]["weight"] * 2
                )  ## should be better as * 2 # e^{-i\theta H}, H=-ZZ
            return circuit

        f.__doc__ = """any%slayer""" % gates
        f.__repr__ = """any%slayer""" % gates  # type: ignore
        f.__trainable__ = True  # type: ignore
        setattr(thismodule, "cirqany" + gates + "layer", f)

    import itertools

    for gate in ["rx", "ry", "rz", "H"]:
        generate_cirq_gate_layer(gate)
        if gate != "H":
            generate_cirq_any_gate_layer(gate)

    for gates_tuple in itertools.product(*[["x", "y", "z"] for _ in range(2)]):
        gates = gates_tuple[0] + gates_tuple[1]
        generate_cirq_double_gate(gates)
        generate_cirq_double_gate_layer(gates)
        generate_cirq_any_double_gate_layer(gates)

    generate_cirq_double_gate_layer("swap")
    generate_cirq_any_double_gate_layer("swap")
    generate_cirq_double_gate_layer("cnot")

except NameError as e:
    logger.warning(e)
    logger.warning("cirq layer generation disabled")
