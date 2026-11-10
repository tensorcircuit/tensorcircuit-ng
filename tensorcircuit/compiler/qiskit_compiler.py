"""
compiler interface via qiskit
"""

import re
from typing import Any, Dict, Optional, Sequence

import sympy

from ..abstractcircuit import AbstractCircuit
from ..circuit import Circuit
from ..symbolcircuit import SymbolCircuit
from ..translation import (
    get_qiskit_qasm,
    qiskit2tc,
    qiskit_from_qasm_str_ordered_measure,
)


def _free_pi(s: str) -> str:
    # dirty trick to get rid of pi in openqasm from qiskit
    rs = []
    pistr = "3.141592653589793"
    s = s.replace("pi", pistr)
    for r in s.split("\n"):
        inc = re.search(r"\(.*\)", r)
        if inc is None:
            rs.append(r)
        else:
            v = r[inc.start() : inc.end()]
            v = eval(v)
            if not isinstance(v, tuple):
                r = r[: inc.start()] + "(" + str(v) + ")" + r[inc.end() :]
            else:  # u gate case
                r = r[: inc.start()] + str(v) + r[inc.end() :]
            rs.append(r)
    return "\n".join(rs)


def _comment_qasm(s: str) -> str:
    """
    return the qasm str in comment format

    :param s: qasm string to wrap line-by-line in ``//`` comments
    :type s: str
    :return: the input qasm with each line prefixed by ``//``, bookended by
        ``//circuit begins`` and ``//circuit ends`` markers
    :rtype: str
    """
    nslist = []
    nslist.append("//circuit begins")
    for line in s.split("\n"):
        nslist.append("//" + line)
    nslist.append("//circuit ends")
    return "\n".join(nslist)


def _comment_dict(d: Dict[int, int], name: str = "logical_physical_mapping") -> str:
    """
    save a dict in commented qasm

    :param d: mapping serialized as ``// k : v`` lines
    :type d: Dict[int, int]
    :param name: header label used for the begin/end marker lines, defaults
        to "logical_physical_mapping"
    :type name: str, optional
    :return: the dict rendered as commented qasm under a named header, with
        ``//<name> begins`` and ``//<name> ends`` bookends
    :rtype: str
    """
    nslist = []
    nslist.append("//%s begins" % name)
    for k, v in d.items():
        nslist.append("// " + str(k) + " : " + str(v))
    nslist.append("//%s ends" % name)
    return "\n".join(nslist)


def _get_positional_logical_mapping_from_qiskit(qc: Any) -> Dict[int, int]:
    # the position in the result bitstring is the classical bit index
    positional_logical_mapping = {}
    for inst in qc.data:
        if inst[0].name == "measure":
            positional_logical_mapping[qc.find_bit(inst[2][0]).index] = qc.find_bit(
                inst[1][0]
            ).index

    return positional_logical_mapping


def _get_logical_physical_mapping_from_qiskit(
    qc_after: Any, qc_before: Any = None
) -> Dict[int, int]:
    """
    get ``logical_physical_mapping`` from qiskit Circuit by comparing the circuit after and before compiling

    :param qc_after: qiskit ``QuantumCircuit`` after compiling
    :type qc_after: Any
    :param qc_before: qiskit ``QuantumCircuit`` before compiling,
        if None, measure(q, q) is assumed
    :type qc_before: Any
    :return: logical_physical_mapping
    :rtype: Dict[int, int]
    """
    logical_physical_mapping = {}
    for inst in qc_after.data:
        if inst[0].name == "measure":
            if qc_before is None:
                logical_q = qc_after.find_bit(inst[2][0]).index
            else:
                for instb in qc_before.data:
                    if (
                        instb[0].name == "measure"
                        and qc_before.find_bit(instb[2][0]).index
                        == qc_after.find_bit(inst[2][0]).index
                    ):
                        logical_q = qc_before.find_bit(instb[1][0]).index
                        break
            logical_physical_mapping[logical_q] = qc_after.find_bit(inst[1][0]).index
    return logical_physical_mapping


def _add_measure_all_if_none(qc: Any) -> Any:
    for inst in qc.data:
        if inst[0].name == "measure":
            break
    else:
        qc.measure_all()
    return qc


def qiskit_compile(
    circuit: Any,
    info: Optional[Dict[str, Any]] = None,
    output: str = "tc",
    compiled_options: Optional[Dict[str, Any]] = None,
    remove_barriers: bool = True,
) -> Any:
    """
    compile the circuit using ``qiskit.transpile`` method with some tricks and hacks.

    A ``SymbolCircuit`` with free symbols is compiled with the symbols kept, into
    a ``SymbolCircuit`` on the compiled qubits with the same symbols, so that it is
    compiled once and bound by ``to_circuit`` for each set of parameters, e.g.
    submitted with ``compiling=False`` to run on the same physical qubits.

    :Example:

    >>> p = tc.cloud.apis.get_device("lqcloud::<device>").list_properties()
    >>> options = {"target": properties_to_target(p),
    ...            "optimization_level": 2, "seed_transpiler": 42}
    >>> csc, info = qiskit_compile(sc, compiled_options=options)
    >>> c = csc.to_circuit({theta: 0.3})

    :param circuit: circuit in ``tc.Circuit``, ``tc.SymbolCircuit`` or
        ``qiskit.QuantumCircuit`` form, where the ``inputs`` state of ``tc.Circuit``
        is prepared by ``initialize``
    :type circuit: Any
    :param info: info for qubit mappings, defaults to None
    :type info: Optional[Dict[str, Any]], optional
    :param output: output circuit format, defaults to "tc", i.e. a ``tc.Circuit``,
        or a ``tc.SymbolCircuit`` if the compiled circuit has parameters
    :type output: str, optional
    :param compiled_options: ``qiskit.transpile`` options in a dict, defaults to None
    :type compiled_options: Optional[Dict[str, Any]], optional
    :param remove_barriers: whether remove the barriers after compiling,
        defaults to True
    :type remove_barriers: bool, optional
    :return: Tuple containing the output circuit and the qubit mapping info dict.
        ``positional_logical_mapping`` in the info refers to the original circuit,
        keyed by the classical bit; the output circuit keeps the classical bits,
        so its own ``get_positional_logical_mapping`` gives the compiled qubits.
    :rtype: Any
    """
    from qiskit.compiler import transpile
    from qiskit.transpiler.passes import RemoveBarriers

    symbols = {}
    if isinstance(circuit, SymbolCircuit):
        symbols = {str(s): s for s in circuit.free_symbols()}
    if isinstance(circuit, AbstractCircuit):
        circuit = circuit.to_qiskit(enable_instruction=True, enable_inputs=True)
    elif isinstance(circuit, str):
        circuit = qiskit_from_qasm_str_ordered_measure(circuit)
    # else qiskit circuit
    circuit = _add_measure_all_if_none(circuit)
    if compiled_options is None:
        compiled_options = {
            "basis_gates": ["h", "rz", "cx"],
            "optimization_level": 2,  # 3 can induce bugs...
        }
    ncircuit = transpile(circuit, **compiled_options)
    if remove_barriers:
        ncircuit = RemoveBarriers()(ncircuit)

    if output.lower() in ["qasm", "openqasm"]:
        r0 = get_qiskit_qasm(ncircuit)

    elif output.lower() in ["qiskit", "ibm"]:
        r0 = ncircuit

    elif output.lower() in ["tc", "tensorcircuit"] and ncircuit.parameters:
        # the parameters back to the symbols of the same names
        r0 = qiskit2tc(
            ncircuit,
            ncircuit.num_qubits,
            circuit_constructor=SymbolCircuit,
            binding_params={
                p: symbols.get(p.name, sympy.Symbol(p.name))
                for p in ncircuit.parameters
            },
        )

    elif output.lower() in ["tc", "tensorcircuit"]:
        s = _free_pi(get_qiskit_qasm(ncircuit))
        r0 = Circuit.from_openqasm(  # type: ignore
            s,
            keep_measure_order=True,
        )

    else:
        raise ValueError("Unknown output format: %s" % output)

    r1 = {}
    nlpm = _get_logical_physical_mapping_from_qiskit(ncircuit, circuit)
    # new_logical_physical_mapping
    if info is not None and "logical_physical_mapping" in info:
        r1["logical_physical_mapping"] = {
            k: nlpm[v] for k, v in info["logical_physical_mapping"].items()
        }
    else:
        r1["logical_physical_mapping"] = nlpm
    if info is not None and "positional_logical_mapping" in info:
        r1["positional_logical_mapping"] = info["positional_logical_mapping"]
    else:  # info is none, assume circuit is the logical circuit
        r1["positional_logical_mapping"] = _get_positional_logical_mapping_from_qiskit(
            circuit
        )
    # TODO(@refraction-ray): more info to be added into r1 dict

    return (r0, r1)


def properties_to_target(
    properties: Dict[str, Any], basis_gates: Optional[Sequence[str]] = None
) -> Any:
    """
    Build the qiskit ``Target`` of a device with the error rates in its properties,
    for the noise aware layout of ``qiskit.transpile`` at ``optimization_level``
    2 or 3, which places the circuit on the qubits of the highest estimated success
    probability. A qubit without its single qubit or readout errors, or a coupler
    without its ``CZErrRate``, is left out and never used. The layout is the best
    one only for a circuit placed without routing; when SWAPs are needed, the
    qubits are chosen by the routing and depend on ``seed_transpiler``.

    :Example:

    >>> d = tc.cloud.apis.get_device("lqcloud::QZ03")
    >>> target = properties_to_target(d.list_properties())
    >>> qiskit_compile(c, compiled_options={"target": target, "optimization_level": 2})

    :param properties: the device properties by ``list_properties``, with the
        standardized ``bits`` and ``links``
    :type properties: Dict[str, Any]
    :param basis_gates: qiskit names of the one and two qubit gates of the target,
        each two qubit gate with the ``CZErrRate`` of the coupler, defaults to None,
        i.e. the ``native_gates`` of the properties
    :type basis_gates: Optional[Sequence[str]], optional
    :return: the qiskit ``Target``, also supporting measure and reset
    :rtype: Any
    """
    from qiskit.circuit import Measure, Reset
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
    from qiskit.transpiler import InstructionProperties, Target

    if basis_gates is None:
        basis_gates = properties["native_gates"]
    # qiskit < 2 (e.g. 0.46 and 1.x): VF2Layout may drop a usable qubit instead of
    # a left out one, so the layout can be worse and depends on the seed when any
    # qubit is left out; qiskit >= 2 is recommended
    bits = {
        q: b
        for q, b in properties["bits"].items()
        if all(
            b.get(k) is not None
            for k in ["SingleQubitErrRate", "ReadoutF0Err", "ReadoutF1Err"]
        )
    }
    if not bits:
        raise ValueError("No qubit with known errors in the device properties")
    links = {}
    for (q1, q2), link in properties["links"].items():
        if link.get("CZErrRate") is not None and q1 in bits and q2 in bits:
            links[(q1, q2)] = links[(q2, q1)] = link["CZErrRate"]
    gates = get_standard_gate_name_mapping()
    target = Target(num_qubits=max(properties["bits"]) + 1)
    for name in basis_gates:
        if gates[name].num_qubits == 1:
            errors = {(q,): b["SingleQubitErrRate"] for q, b in bits.items()}
        else:
            errors = links  # type: ignore
        target.add_instruction(
            gates[name], {k: InstructionProperties(error=e) for k, e in errors.items()}
        )
    target.add_instruction(
        Measure(),
        {
            (q,): InstructionProperties(
                error=(b["ReadoutF0Err"] + b["ReadoutF1Err"]) / 2
            )
            for q, b in bits.items()
        },
    )
    target.add_instruction(Reset(), {(q,): None for q in bits})
    return target
