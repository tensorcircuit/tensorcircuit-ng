"""
Compiler passes for symbolic circuits.
"""

from typing import Any, Dict, Sequence, Tuple

from ..symbolcircuit import SymbolCircuit


def lightcone_compile(
    circuit: SymbolCircuit, observable_qubits: Sequence[int]
) -> Tuple[SymbolCircuit, Dict[str, Any]]:
    """
    Compile a SymbolCircuit to the causal cone induced by a final observable's
    qubit support.

    The compiled circuit retains the original SymPy symbols, so callers may
    pass the complete symbol-to-value binding dictionary to ``to_circuit``;
    bindings for removed gates are ignored.

    :param circuit: Symbolic circuit to compile.
    :type circuit: SymbolCircuit
    :param observable_qubits: Qubit support of the final observable.
    :type observable_qubits: Sequence[int]
    :return: Compiled circuit and mapping information. ``observable_qubits``
        contains the remapped observable support, and
        ``logical_physical_mapping`` maps retained input qubits to compiled
        qubits.
    :rtype: Tuple[SymbolCircuit, Dict[str, Any]]
    :raises ValueError: If the observable indices are empty, duplicated, or out
        of range.
    :raises NotImplementedError: If the circuit uses non-default inputs,
        channels, or extra QIR instructions.
    """
    if not observable_qubits:
        raise ValueError("observable_qubits must contain at least one qubit")
    normalized_observable_qubits = [
        q if q >= 0 else circuit._nqubits + q for q in observable_qubits
    ]
    if any(q < 0 or q >= circuit._nqubits for q in normalized_observable_qubits):
        raise ValueError("observable_qubits contains an index outside the circuit")
    if len(set(normalized_observable_qubits)) != len(normalized_observable_qubits):
        raise ValueError("observable_qubits must not contain duplicates")
    if circuit.inputs is not None:
        raise NotImplementedError(
            "lightcone_compile requires the default product input state"
        )
    if circuit._extra_qir:
        raise NotImplementedError(
            "lightcone_compile does not support extra QIR instructions"
        )

    qir = circuit.to_qir()
    if any(instruction.get("is_channel", False) for instruction in qir):
        raise NotImplementedError(
            "lightcone_compile does not support channel instructions"
        )

    normalized_qir = []
    for instruction in qir:
        normalized_instruction = dict(instruction)
        normalized_instruction["index"] = tuple(
            q if q >= 0 else circuit._nqubits + q for q in instruction["index"]
        )
        normalized_qir.append(normalized_instruction)

    active = set(normalized_observable_qubits)
    kept = []
    for instruction in reversed(normalized_qir):
        support = set(instruction["index"])
        if active & support:
            kept.append(instruction)
            active |= support
    kept.reverse()

    active_qubits = sorted(active)
    qubit_mapping = {old: new for new, old in enumerate(active_qubits)}
    reduced_qir = []
    for instruction in kept:
        reduced_instruction = dict(instruction)
        reduced_instruction["index"] = tuple(
            qubit_mapping[index] for index in instruction["index"]
        )
        if "parameters" in reduced_instruction:
            reduced_instruction["parameters"] = dict(reduced_instruction["parameters"])
        reduced_qir.append(reduced_instruction)

    compiled = SymbolCircuit(len(active_qubits))
    compiled.append_from_qir(reduced_qir)
    info = {
        "logical_physical_mapping": qubit_mapping,
        "observable_qubits": [qubit_mapping[q] for q in normalized_observable_qubits],
    }
    return compiled, info
