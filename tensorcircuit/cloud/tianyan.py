"""
Cloud provider from TianYan (中电信天衍量子计算云平台)
https://qc.zdxlz.com
"""

from collections import Counter
from datetime import datetime
import logging
from typing import Any, cast, Dict, List, Optional, Sequence, Union

from .abstraction import Device, sep, Task
from ..abstractcircuit import AbstractCircuit
from ..utils import is_sequence

logger = logging.getLogger(__name__)

try:
    from cqlib import TianYanPlatform
    from cqlib.quantum_platform import QuantumLanguage
    from cqlib.utils import QasmToQcis
except ImportError:
    TianYanPlatform = None  # type: ignore
    QuantumLanguage = None  # type: ignore
    QasmToQcis = None  # type: ignore

_SIMULATOR_DEVICES = {"tianyan_sw", "tianyan_s", "tianyan_tn"}
_REQUIRED_PLATFORM_ATTRIBUTES = (
    "_send_request",
    "QUERY_EXP_PATH",
    "MACHINE_LIST_PATH",
)


def _assert_cqlib() -> None:
    if TianYanPlatform is None:
        raise ImportError(
            "cqlib is required for TianYan provider. "
            'Install it via: pip install "cqlib>=1.3.10,<1.4"'
        )
    missing = [
        attribute
        for attribute in _REQUIRED_PLATFORM_ATTRIBUTES
        if not hasattr(TianYanPlatform, attribute)
    ]
    if missing:
        raise ImportError(
            "Installed cqlib is incompatible with TianYan provider; "
            "missing TianYanPlatform attributes: %s" % ", ".join(missing)
        )


def _get_platform(token: str, machine_name: Optional[str] = None) -> "TianYanPlatform":
    _assert_cqlib()
    return TianYanPlatform(login_key=token, machine_name=machine_name)  # type: ignore


def _circuit_to_qcis(circuit: Any) -> str:
    """Convert a TensorCircuit or qiskit circuit to QCIS string."""
    if isinstance(circuit, str):
        return circuit

    # Try to get qiskit circuit directly
    try:
        from qiskit import qasm2
        from qiskit.circuit import QuantumCircuit

        if isinstance(circuit, QuantumCircuit):
            qasm = qasm2.dumps(circuit)
            return _qasm_to_qcis(qasm)
    except ImportError:
        pass

    # Try AbstractCircuit / TensorCircuit
    if isinstance(circuit, AbstractCircuit):
        return _tc_qir_to_qcis(circuit)

    to_openqasm = getattr(circuit, "to_openqasm", None)
    if callable(to_openqasm):
        return _qasm_to_qcis(to_openqasm())

    raise ValueError(
        "Unsupported circuit type for TianYan provider: %s" % type(circuit)
    )


def _qasm_to_qcis(qasm: str) -> str:
    """Convert OpenQASM 2 source with cqlib's supported converter."""
    _assert_cqlib()
    return str(QasmToQcis().convert_to_qcis(qasm))  # type: ignore


def _source_to_qcis(
    source: Union[str, Sequence[str]], lang: str
) -> Union[str, List[str]]:
    """Normalize explicitly supplied source code to QCIS."""
    normalized_lang = lang.strip().upper()
    if normalized_lang == "QCIS":
        if isinstance(source, str):
            return source
        return list(source)
    if normalized_lang in {"QASM", "OPENQASM", "OPENQASM2"}:
        if isinstance(source, str):
            return _qasm_to_qcis(source)
        return [_qasm_to_qcis(item) for item in source]
    raise ValueError(f"Unsupported TianYan source language: {lang}")


def _validate_circuit_topology(
    circuit: AbstractCircuit,
    graph: dict[int, set[int]],
    available_qubits: set[int],
) -> None:
    """
    Raise ValueError if the circuit is incompatible with the device topology,
    i.e. it uses unavailable qubits or multi-qubit gates on unconnected pairs.
    """
    for inst in circuit.to_qir():
        indices = tuple(inst.get("index", ()))
        for idx in indices:
            if idx not in available_qubits:
                raise ValueError(
                    "Gate %s acts on qubit %s, which is unavailable on this "
                    "device; compile the circuit for the device first, "
                    "e.g. via tensorcircuit.compiler" % (inst.get("name"), idx)
                )
        for i, q1 in enumerate(indices):
            for q2 in indices[i + 1 :]:
                if q1 != q2 and q2 not in graph.get(q1, set()):
                    raise ValueError(
                        "Gate %s acts on unconnected physical qubits (%s, %s) "
                        "for this device; compile and map the circuit to the "
                        "device topology first, e.g. via tensorcircuit.compiler"
                        % (inst.get("name"), q1, q2)
                    )


def _get_device_topology(
    pf: "TianYanPlatform", device_name: str
) -> tuple[dict[int, set[int]], set[int]]:
    """Get topology graph and available qubits from device config."""
    config = pf.download_config(machine=device_name) or {}
    overview = config.get("overview", {})

    # Build adjacency graph from coupler map
    coupler_map = overview.get("coupler_map", {})
    graph: dict[int, set[int]] = {}
    for _coupler, qubits in coupler_map.items():
        q1 = int(qubits[0][1:])  # Remove 'Q' prefix
        q2 = int(qubits[1][1:])
        graph.setdefault(q1, set()).add(q2)
        graph.setdefault(q2, set()).add(q1)

    # Get all qubits
    all_qubits = set()
    for q_str in overview.get("qubits", []):
        if q_str.startswith("Q"):
            all_qubits.add(int(q_str[1:]))

    # Remove disabled qubits
    disabled = overview.get("disabledQubits", "")
    if disabled:
        for q_str in disabled.split(","):
            q_str = q_str.strip()
            if q_str.startswith("Q"):
                all_qubits.discard(int(q_str[1:]))

    return graph, all_qubits


def _tc_qir_to_qcis(circuit: AbstractCircuit) -> str:
    """
    Convert TensorCircuit QIR to QCIS string using cqlib.Circuit.
    Measurements are always emitted as terminal measurements in record
    order; mid-circuit measurement semantics are not preserved.
    """
    qir = circuit.to_qir()
    extra_qir = circuit._extra_qir
    measure_instructions = [inst for inst in extra_qir if inst.get("name") == "measure"]
    measure_instructions.sort(key=lambda inst: inst.get("record_index", 0))

    _assert_cqlib()
    from cqlib.circuits import Circuit as CqlibCircuit

    # Collect all logical qubits used
    all_logical_qubits = set()
    for inst in qir:
        for idx in inst.get("index", ()):
            all_logical_qubits.add(idx)
    for inst in measure_instructions:
        idx = inst.get("index", [None])[0]
        if idx is not None:
            all_logical_qubits.add(idx)

    sorted_logical = sorted(all_logical_qubits)

    cqlib_circuit = CqlibCircuit(qubits=sorted_logical)

    for inst in qir:
        name = inst.get("name", "").lower()
        index = inst.get("index", ())
        parameters = inst.get("parameters", {})

        resolved_params = {}
        for key, value in parameters.items():
            try:
                if hasattr(value, "item"):
                    value = float(value.item())
                elif hasattr(value, "numpy"):
                    value = float(value.numpy().item())
                else:
                    value = float(value)
                resolved_params[key] = value
            except (TypeError, ValueError):
                resolved_params[key] = value

        if name == "h":
            cqlib_circuit.h(index[0])
        elif name == "x":
            cqlib_circuit.x(index[0])
        elif name == "y":
            cqlib_circuit.y(index[0])
        elif name == "z":
            cqlib_circuit.z(index[0])
        elif name == "s":
            cqlib_circuit.s(index[0])
        elif name == "sd":
            cqlib_circuit.sd(index[0])
        elif name == "t":
            cqlib_circuit.t(index[0])
        elif name == "td":
            cqlib_circuit.td(index[0])
        elif name == "rx":
            cqlib_circuit.rx(index[0], resolved_params.get("theta", 0))
        elif name == "ry":
            cqlib_circuit.ry(index[0], resolved_params.get("theta", 0))
        elif name == "rz":
            cqlib_circuit.rz(index[0], resolved_params.get("theta", 0))
        elif name in ("cnot", "cx"):
            cqlib_circuit.cx(index[0], index[1])
        elif name == "cz":
            cqlib_circuit.cz(index[0], index[1])
        elif name == "swap":
            cqlib_circuit.swap(index[0], index[1])
        elif name == "ccx":
            cqlib_circuit.ccx(index[0], index[1], index[2])
        else:
            raise ValueError("Unsupported gate for cqlib.Circuit: %s" % name)

    measure_qubits = [inst.get("index", [None])[0] for inst in measure_instructions]
    measure_qubits = [idx for idx in measure_qubits if idx is not None]
    if measure_qubits:
        cqlib_circuit.measure(measure_qubits)

    return str(cqlib_circuit.qcis)


def _normalize_task_state(state: Any) -> Optional[str]:
    if state is None:
        return None
    normalized = str(state).strip().lower()
    if normalized in {
        "completed",
        "complete",
        "success",
        "succeeded",
        "done",
        "finished",
    }:
        return "completed"
    if normalized in {
        "failed",
        "failure",
        "error",
        "cancelled",
        "canceled",
        "stopped",
        "terminated",
    }:
        return "failed"
    if normalized in {
        "pending",
        "queued",
        "submitted",
        "waiting",
        "running",
        "executing",
        "processing",
    }:
        return "pending"
    return None


def _query_experiment_once(
    pf: "TianYanPlatform", query_id: str
) -> List[Dict[str, Any]]:
    """Query the SDK result endpoint once without its built-in polling loop."""
    response = pf._send_request(  # pylint: disable=protected-access
        path=pf.QUERY_EXP_PATH,
        data={"query_ids": [query_id]},
        method="POST",
    )
    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError("Invalid TianYan query response: missing data object")
    results = data.get("experimentResultModelList")
    if results is None:
        return []
    if not isinstance(results, list):
        raise ValueError("Invalid TianYan query response: results must be a list")
    return results


def _parse_result(result_item: Dict[str, Any], device: Device) -> Dict[str, Any]:
    """
    Parse a single result item from cqlib query_experiment into
    TensorCircuit task details format.
    """
    task_id = result_item.get("experimentTaskId", "")
    result_status = result_item.get("resultStatus") or []
    probability = result_item.get("probability")

    raw_state = next(
        (
            result_item[key]
            for key in (
                "state",
                "status",
                "taskStatus",
                "experimentStatus",
                "runStatus",
            )
            if result_item.get(key) is not None
        ),
        None,
    )
    error = next(
        (
            str(result_item[key])
            for key in ("err", "error", "errorMessage", "failReason")
            if result_item.get(key)
        ),
        "",
    )
    state = _normalize_task_state(raw_state)
    if state is None:
        if error:
            state = "failed"
        elif raw_state is not None:
            state = "failed"
            error = f"Unknown TianYan task status: {raw_state}"
        else:
            # The SDK result endpoint only returns an item after reaching a terminal state.
            state = "completed"

    # Parse counts
    # Bit ordering convention verified against the TianYan simulator (2026-07):
    # shot bits are emitted left-to-right in measure_order, matching the tc
    # simulation convention (qubit 0 is the most significant bit).
    if result_status and len(result_status) > 1:
        measure_order = result_status[0]  # e.g. [0, 1]
        shots_data = result_status[1:]  # each shot is a list of bits
        total_shots = len(shots_data)

        # Convert to counts dict: bitstring -> count
        # Bits are in the order of measure_order
        counts: Counter[str] = Counter()
        for shot in shots_data:
            bitstring = "".join(str(b) for b in shot)
            counts[bitstring] += 1
        counts_dict = dict(counts)
    else:
        total_shots = 0
        counts_dict = {}
        measure_order = []

    details = {
        "id": task_id,
        "state": state,
        "results": counts_dict,
        "shots": total_shots,
        "measure_order": measure_order,
        "probability": probability,
        "device": str(device),
    }
    if error:
        details["err"] = error
    return details


def _query_machine_list_once(pf: "TianYanPlatform") -> List[Dict[str, Any]]:
    """Query the machine list endpoint once, returning raw machine dicts."""
    response = pf._send_request(  # pylint: disable=protected-access
        path=pf.MACHINE_LIST_PATH
    )
    data = response.get("data")
    if not isinstance(data, list):
        raise ValueError("Invalid TianYan machine list response: missing data list")
    return data


def list_devices(token: Optional[str] = None, **kws: Any) -> List[Device]:
    _assert_cqlib()
    if token is None:
        raise ValueError("TianYan provider requires a token (login_key).")
    pf = _get_platform(token)
    devices = []
    for item in _query_machine_list_once(pf):
        machine_name = item.get("code") or item.get("machineName")
        if not isinstance(machine_name, str) or not machine_name:
            raise ValueError(
                "Invalid TianYan machine list response: missing machine name"
            )
        devices.append(Device.from_name("tianyan" + sep + machine_name))
    return devices


def list_properties(device: Device, token: Optional[str] = None) -> Dict[str, Any]:
    _assert_cqlib()
    if token is None:
        raise ValueError("TianYan provider requires a token (login_key).")

    pf = _get_platform(token, machine_name=device.name)
    # Try to download device config if available
    properties: Dict[str, Any] = {
        "id": device.name,
        "provider": "tianyan",
    }
    try:
        # download_config may return machine topology / calibration data
        config = pf.download_config(machine=device.name)
        if config:
            properties.update(config)
    except Exception as e:
        logger.debug("Failed to download config for %s: %s", device.name, e)

    # Standardize topology data into "links" and "bits" dicts
    # (compatible with tencent provider format)
    _standardize_properties(properties)

    # Native gates supported by the TensorCircuit-to-QCIS conversion path;
    # QCIS sources supplied directly may use the full cqlib gate set
    properties["native_gates"] = [
        "h",
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "cx",
        "cz",
        "swap",
        "ccx",
        "s",
        "sd",
        "t",
        "td",
    ]
    return properties


def _standardize_properties(properties: Dict[str, Any]) -> None:
    """
    Convert raw TianYan config into standardized 'links' and 'bits' dicts
    compatible with tencent provider format.
    """
    overview = properties.get("overview", {})
    coupler_map = overview.get("coupler_map", {})
    two_qubit = properties.get("twoQubitGate", {})
    cz_gate = two_qubit.get("czGate", {})
    cz_err = cz_gate.get("gate error", {})
    cz_err_list = cz_err.get("param_list", [])
    cz_err_qubits = cz_err.get("qubit_used", [])

    # Build links dict: {(q1, q2): {'CZErrRate': ...}, ...}
    links: Dict[tuple[int, int], Dict[str, Any]] = {}
    for g_name, qubits in coupler_map.items():
        if len(qubits) >= 2:
            q1 = int(qubits[0][1:])  # Remove 'Q' prefix
            q2 = int(qubits[1][1:])
            # Find CZ error rate for this coupler
            err_rate = None
            for idx, g_id in enumerate(cz_err_qubits):
                if g_id == g_name and idx < len(cz_err_list):
                    err_rate = cz_err_list[idx]
                    break
            link_data: Dict[str, Any] = {}
            if err_rate is not None:
                link_data["CZErrRate"] = err_rate
            links[(q1, q2)] = link_data
            links[(q2, q1)] = link_data.copy()

    properties["links"] = links

    # Build bits dict: {qubit_id: {T1: ..., T2: ..., ...}, ...}
    bits: Dict[int, Dict[str, Any]] = {}

    def _extract_param(parent: Dict[str, Any], qubit_key: str = "Q") -> None:
        """Extract (param_list, qubit_used) pairs from parent dict into bits."""
        for param_name, data in parent.items():
            if isinstance(data, dict) and "param_list" in data and "qubit_used" in data:
                for idx, qubit_id in enumerate(data["qubit_used"]):
                    if idx < len(data["param_list"]):
                        q_idx = (
                            int(qubit_id[1:])
                            if qubit_id.startswith(qubit_key)
                            else int(qubit_id)
                        )
                        bits.setdefault(q_idx, {})
                        # Map to tencent-compatible key names
                        if param_name == "T1":
                            bits[q_idx]["T1"] = data["param_list"][idx]
                        elif param_name == "T2":
                            bits[q_idx]["T2"] = data["param_list"][idx]
                        elif "gate error" in param_name:
                            bits[q_idx]["SingleQubitErrRate"] = data["param_list"][idx]
                        elif "Readout Error" in param_name:
                            bits[q_idx]["ReadoutError"] = data["param_list"][idx]
                        elif "|0> readout fidelity" in param_name:
                            bits[q_idx]["ReadoutF0Err"] = 1.0 - data["param_list"][idx]
                        elif "|1> readout fidelity" in param_name:
                            bits[q_idx]["ReadoutF1Err"] = 1.0 - data["param_list"][idx]
                        elif "f01" in param_name:
                            bits[q_idx]["Freqency"] = data["param_list"][idx]

    # Extract from relatime (T1, T2)
    _extract_param(properties.get("relatime", {}))
    # Extract from singleQubit (gate error, etc.)
    _extract_param(properties.get("singleQubit", {}))
    # Extract from readout.readoutArray
    readout_array = properties.get("readout", {}).get("readoutArray", {})
    _extract_param(readout_array)
    # Extract from qubit.frequency (f01 frequency)
    qubit_freq = properties.get("qubit", {}).get("frequency", {})
    _extract_param(qubit_freq)

    # Fallback: if bits is empty, gather qubit IDs from links
    if not bits:
        for qubit_1, qubit_2 in links.keys():
            bits.setdefault(qubit_1, {})
            bits.setdefault(qubit_2, {})

    properties["bits"] = bits


def _normalize_shots(shots: Union[int, Sequence[int]], task_count: int) -> int:
    if task_count <= 0:
        raise ValueError("At least one TianYan task must be submitted")
    if not isinstance(shots, int):
        shot_values = list(shots)
        if len(shot_values) != task_count:
            raise ValueError(
                f"Expected {task_count} shots values, got {len(shot_values)}"
            )
        if any(value != shot_values[0] for value in shot_values[1:]):
            raise ValueError(
                "TianYan batch submission requires the same shots for every circuit"
            )
        normalized = shot_values[0]
    else:
        normalized = shots
    if (
        not isinstance(normalized, int)
        or isinstance(normalized, bool)
        or normalized <= 0
    ):
        raise ValueError("shots must be a positive integer")
    return normalized


def submit_task(
    device: Device,
    token: str,
    lang: str = "QCIS",
    shots: Union[int, Sequence[int]] = 1024,
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    lab_id: Optional[str] = None,
    exp_name: Optional[str] = None,
    **kws: Any,
) -> Union[Task, List[Task]]:
    """
    Submit task via TianYan provider.

    Measurements are always submitted as terminal measurements in their
    recorded order; mid-circuit measurement semantics are not preserved.

    Circuits submitted to real hardware must already respect the device
    topology: compile and map the circuit for the device first, e.g. via
    :py:mod:`tensorcircuit.compiler`. A ``ValueError`` is raised when a
    TensorCircuit circuit is incompatible with the device topology.
    Topology validation only applies to TensorCircuit circuits; for qiskit
    circuits and direct sources, compatibility is the user's responsibility.

    :param device: Target device
    :param token: Login key for TianYan platform
    :param lang: Language of an explicitly supplied source, ``QCIS`` or ``OPENQASM``
    :param shots: Number of measurement shots
    :param circuit: TensorCircuit or qiskit circuit object(s)
    :param source: Direct QCIS or OpenQASM source string(s)
    :param lab_id: Optional lab ID for experiment collection
    :param exp_name: Optional experiment name
    """
    _assert_cqlib()
    if source is not None:
        source = _source_to_qcis(source, lang)

    pf = _get_platform(token, machine_name=device.name)

    # If source is not provided, convert circuit to QCIS
    if source is None:
        if circuit is None:
            raise ValueError("Either `circuit` or `source` must be provided.")

        topology = None
        if device.name not in _SIMULATOR_DEVICES:
            topology = _get_device_topology(pf, device.name)

        def validate(c: Any) -> None:
            if topology is not None and isinstance(c, AbstractCircuit):
                graph, available_qubits = topology
                _validate_circuit_topology(c, graph, available_qubits)

        if is_sequence(circuit):
            sources = []
            for c in cast(Sequence[AbstractCircuit], circuit):
                validate(c)
                sources.append(_circuit_to_qcis(c))
            source = sources
        else:
            validate(circuit)
            source = _circuit_to_qcis(circuit)

    # Ensure exp_name has a default
    if exp_name is None:
        exp_name = f"tc_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Submit
    if is_sequence(source):
        sources = list(source)
        num_shots = _normalize_shots(shots, len(sources))
        query_ids = pf.submit_experiment(
            circuit=sources,
            language=QuantumLanguage.QCIS,  # type: ignore
            lab_id=lab_id,
            lab_name=exp_name,
            num_shots=num_shots,
        )
        if not isinstance(query_ids, (list, tuple)):
            raise ValueError("TianYan batch submission did not return task IDs")
        if len(query_ids) != len(sources):
            raise ValueError(
                f"TianYan returned {len(query_ids)} task IDs for "
                f"{len(sources)} submitted circuits"
            )
        tasks = []
        for qid, src in zip(query_ids, sources):
            if not isinstance(qid, str) or not qid:
                raise ValueError("TianYan returned an invalid task ID")
            task = Task(id_=qid, device=device)
            task.add_details(source=src)
            tasks.append(task)
        return tasks
    else:
        num_shots = _normalize_shots(shots, 1)
        query_id = pf.submit_job(
            circuit=source,  # type: ignore
            exp_name=exp_name,
            lab_id=lab_id,
            num_shots=num_shots,
            language=QuantumLanguage.QCIS,  # type: ignore
        )
        if isinstance(query_id, (list, tuple)):
            if len(query_id) != 1:
                raise ValueError("TianYan single submission returned multiple task IDs")
            query_id = query_id[0]
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("TianYan submission did not return a valid task ID")
        task = Task(id_=query_id, device=device)
        task.add_details(source=source)
        return task


def resubmit_task(task: Task, token: str) -> Task:
    _assert_cqlib()
    if task.device is None:
        raise ValueError("Task must have an associated device for resubmission.")
    pf = _get_platform(token, machine_name=task.device.name)
    try:
        # cqlib re-execute task
        new_query_id = pf.re_execute_task(task.id_)
        if isinstance(new_query_id, list):
            new_query_id = new_query_id[0]
        return Task(id_=new_query_id, device=task.device)
    except Exception as e:
        raise ValueError("Failed to resubmit task %s: %s" % (task.id_, e))


def remove_task(task: Task, token: str) -> Any:
    """Raise because cqlib configures no task-cancellation endpoint."""
    raise NotImplementedError(
        "TianYan provider does not support remove_task because cqlib "
        "has no cancellation endpoint"
    )


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    """Raise because cqlib does not expose a task-listing API."""
    raise NotImplementedError(
        "TianYan provider does not support list_tasks because cqlib has no listing API"
    )


def get_task_details(
    task: Task, device: Device, token: str, prettify: bool
) -> Dict[str, Any]:
    """
    Get task details from TianYan platform.
    """
    _assert_cqlib()
    pf = _get_platform(token, machine_name=device.name)
    result = _query_experiment_once(pf, task.id_)

    if not result:
        return {
            "id": task.id_,
            "state": "pending",
            "results": {},
            "shots": 0,
            "device": str(device),
            "source": task.more_details.get("source", ""),
        }

    # result is a list of dicts, one per submitted circuit
    parsed = _parse_result(result[0], device)

    # Include frontend source code if available
    parsed["source"] = task.more_details.get("source", "")

    if prettify:
        # Make datetime more readable if present
        if "at" in parsed and isinstance(parsed["at"], (int, float)):
            parsed["at"] = datetime.fromtimestamp(parsed["at"] / 1e6)

    return parsed
