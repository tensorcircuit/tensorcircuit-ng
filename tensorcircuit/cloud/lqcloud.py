"""
Cloud provider from LQCloud (逻辑比特量子云)
https://cloud.logicalqubit.com
"""

from typing import Any, Dict, List, Optional, Tuple
import gzip
import json
import logging

import requests

from .abstraction import Device, sep, Task
from .config import lqcloud_base_url
from .utils import SubmissionRejected, SubmissionUnknown
from ..abstractcircuit import AbstractCircuit
from ..compiler.qiskit_compiler import qiskit_compile
from ..utils import is_sequence

logger = logging.getLogger(__name__)

native_gates = ["cz", "rz", "sx", "x"]
# qiskit names of the native gates to the LQCloud ones. Flipping the rotation sense
# of ``rz``, or of ``xhalf``, is unobservable with only these gates, |0> inputs and
# Z-basis readout, since it amounts to the complex conjugation of the circuit; the
# ``rz`` angle scale is observable and confirmed on hardware by routed circuits with
# arbitrary angles. Another non-diagonal native gate, e.g. ``yhalf`` or ``su2``,
# would make the sign observable and need its convention confirmed on hardware.
_lq_gates = {"cz": "cz", "rz": "rz", "sx": "xhalf", "x": "x"}
default_seed = 42
compilation_schema_version = 1

# LQCloud task states to the unified ones
_states = {
    "queued": "pending",
    "pending": "pending",
    "retrying": "pending",
    "running": "pending",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "failed",
}
_lqcloud_fields = [
    "status",
    "progress",
    "queue_position",
    "error",
    "error_type",
    "retry_count",
    "created_at",
    "started_at",
    "completed_at",
    "execution_time",
    "cost",
    "currency",
]


def _lqcloud() -> Any:
    try:
        import lqcloud
    except ImportError as e:
        raise ImportError(
            "lqcloud is required for the LQCloud provider, which needs Python 3.11 "
            'or 3.12. Install it via: pip install "lqcloud>=0.5.1,<0.6"'
        ) from e
    return lqcloud


def _provider(token: Optional[str]) -> Any:
    if not token:
        raise ValueError(
            "LQCloud provider requires an API key, set it via "
            '`tc.cloud.apis.set_token(key, provider="lqcloud")`'
        )
    # the credential and url only come from TensorCircuit, never from the SDK
    # environment variables or account file, and the SDK job store stays off
    return _lqcloud().LQCloudProvider(
        api_key=token, url=lqcloud_base_url, interactive=False, store=False
    )


def _backend(device: Device, token: Optional[str]) -> Any:
    return _provider(token).get_backend(device.name)


def list_devices(token: Optional[str] = None, **kws: Any) -> List[Device]:
    """
    List the LQCloud devices available to the account.

    :param token: the LQCloud API key, defaults to None
    :type token: Optional[str], optional
    :param kws: unused, for the uniform provider interface
    :type kws: Any
    :return: the devices
    :rtype: List[Device]
    """
    return [
        Device.from_name("lqcloud" + sep + b["name"])
        for b in _provider(token).get_backends()
    ]


def _error_rate(fidelity: Optional[float]) -> Optional[float]:
    return None if fidelity is None else 1.0 - fidelity


def list_properties(device: Device, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the device configuration, with the calibration standardized as
    ``bits`` (``T1``, ``T2`` in μs, readout and single qubit error rates) and
    ``links`` (``CZErrRate`` of each coupled pair in both directions).

    :param device: the device
    :type device: Device
    :param token: the LQCloud API key, defaults to None
    :type token: Optional[str], optional
    :return: the device configuration with ``bits``, ``links``, ``native_gates``
        and ``calibration_time``
    :rtype: Dict[str, Any]
    """
    return _properties(device.name, _backend(device, token).config)


def _properties(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    calibration = config.get("properties") or {}
    bits = {}
    for m in calibration.get("qubit_metrics") or []:
        bits[m["id"]] = {
            "T1": m.get("t1"),
            "T2": m.get("t2"),
            "ReadoutF0Err": _error_rate(m.get("measure_f0")),
            "ReadoutF1Err": _error_rate(m.get("measure_f1")),
            "SingleQubitErrRate": _error_rate(m.get("xeb_fidelity")),
        }
    cz_fidelity = {
        tuple(sorted(m["qubits"])): m.get("cz_fidelity")
        for m in calibration.get("coupler_metrics") or []
    }
    links = {}
    for q1, q2 in config["topology"]["coupling_map"]:
        err = _error_rate(cz_fidelity.get(tuple(sorted([q1, q2]))))
        links[(q1, q2)] = {"CZErrRate": err}
        links[(q2, q1)] = {"CZErrRate": err}
    properties = dict(config)
    properties.update(
        id=name,
        provider="lqcloud",
        bits=bits,
        links=links,
        native_gates=list(native_gates),
        calibration_time=calibration.get("last_update"),
    )
    return properties


def _check_circuit(circuit: Any) -> None:
    if not isinstance(circuit, AbstractCircuit):
        raise ValueError(
            "LQCloud provider only supports TensorCircuit circuits, got %s"
            % type(circuit).__name__
        )
    if any(d.get("is_channel", False) for d in circuit.to_qir()):
        raise ValueError("Noise channels cannot be submitted to the LQCloud hardware")
    instructions = {d["name"] for d in circuit._extra_qir} - {"measure", "barrier"}
    if instructions:
        raise ValueError(
            "Unsupported instructions for the LQCloud provider: %s"
            % ", ".join(sorted(instructions))
        )
    params = getattr(circuit, "circuit_param", {})
    if params.get("dim") not in [None, 2]:
        raise ValueError("LQCloud provider only supports qubit circuits")
    inputs = [
        k
        for k in ["mps_inputs", "tensors", "dminputs", "mpo_dminputs"]
        if params.get(k) is not None
    ]
    if inputs:
        raise ValueError(
            "Unsupported circuit inputs for the LQCloud provider: %s, "
            "only a state vector ``inputs`` is supported" % ", ".join(inputs)
        )


def _readout_windows(circuit: AbstractCircuit) -> List[Dict[str, Any]]:
    """
    Group the measurements into readout windows, each a set of measurements read
    out together after a barrier. A measurement followed by no gate on its qubit is
    terminal and belongs to the final window after all gates, which is the only
    window supported now; all qubits are measured in it if no measurement is given.

    :return: the windows in time order, each with its ``kind`` and ``records``,
        i.e. ``{record_index: qubit}`` of its measurements
    :rtype: List[Dict[str, Any]]
    """
    qir = circuit.to_qir()
    last_gate: Dict[int, int] = {}
    for i, d in enumerate(qir):
        for q in d["index"]:
            last_gate[q] = i
    measured = set()
    for d in circuit._extra_qir:
        if d["name"] != "measure":
            continue
        q = d["index"][0]
        if q in measured or d.get("pos", len(qir)) <= last_gate.get(q, -1):
            raise ValueError(
                "Mid-circuit or repeated measurement on qubit %s is not supported "
                "by the LQCloud provider yet" % q
            )
        measured.add(q)
    return [{"kind": "final", "records": circuit.get_positional_logical_mapping()}]


def _logical_qiskit(circuit: AbstractCircuit, records: Dict[int, int]) -> Any:
    # the gates and barriers, then the final window: a barrier over all qubits,
    # so that no operation is placed after any measurement, and the measurements
    from qiskit import QuantumCircuit

    qc = circuit.to_qiskit(enable_instruction=True, enable_inputs=True)
    lqc = QuantumCircuit(qc.num_qubits, max(records) + 1)
    for inst in qc.data:
        if inst.operation.name != "measure":
            lqc.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
    lqc.barrier()
    for r, q in records.items():
        lqc.measure(q, r)
    return lqc


def _native_instructions(qc: Any) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Translate the compiled qiskit circuit into native instructions on the active
    physical qubits, i.e. those with any operation besides barriers. The resets
    before any operation, e.g. from the state preparation, are dropped since the
    SDK resets every qubit at the beginning.
    """
    ops: List[Dict[str, Any]] = []
    dropped = []
    started = set()
    measured = set()
    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            ops.append({"name": "barrier", "qubits": qubits})
            continue
        if any(q in measured for q in qubits):
            raise ValueError(
                "Compiled circuit has operation %s after a measurement on qubits %s"
                % (name, qubits)
            )
        if name == "reset":
            if qubits[0] in started:
                raise ValueError("Reset after operations is unsupported on LQCloud")
            dropped.append(qubits[0])
            continue
        started.update(qubits)
        if name == "measure":
            measured.add(qubits[0])
            ops.append(
                {
                    "name": "measure",
                    "qubits": qubits,
                    "clbits": [qc.find_bit(inst.clbits[0]).index],
                }
            )
        elif name in _lq_gates:
            ops.append(
                {
                    "name": _lq_gates[name],
                    "qubits": qubits,
                    "params": [float(p) for p in inst.operation.params],
                }
            )
        else:
            raise ValueError("Unsupported gate %s in the compiled circuit" % name)
    for op in ops:
        if op["name"] == "barrier":
            op["qubits"] = [q for q in op["qubits"] if q in started]
    return [op for op in ops if op["qubits"]], dropped


def compile_circuit(
    circuit: AbstractCircuit,
    properties: Dict[str, Any],
    shots: int = 1024,
    compiling: bool = True,
    compiled_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """
    Compile the circuit locally into native gates on the physical qubits of the
    device, since LQCloud runs the submitted gates as they are.
    All measurements are read out in the final window after all gates, in the
    order of the classical bits, and the result bitstring is indexed by the
    classical bit, i.e. the ``record_index`` of the measurement.

    :param circuit: the circuit, with terminal measurements only
    :type circuit: AbstractCircuit
    :param properties: the device properties by :py:func:`list_properties`, whose
        ``native_gates`` and ``links`` are compiled to
    :type properties: Dict[str, Any]
    :param shots: number of shots, defaults to 1024
    :type shots: int, optional
    :param compiling: whether choose the layout and route on the device topology;
        if False, qubit ``i`` of the circuit runs on physical qubit ``i``,
        defaults to True
    :type compiling: bool, optional
    :param compiled_options: ``qiskit.transpile`` options overriding the defaults,
        e.g. ``initial_layout``, defaults to None
    :type compiled_options: Optional[Dict[str, Any]], optional
    :return: the positional_logical_mapping from the classical bit to the physical
        qubit, and the compilation record, i.e. the program given to the SDK, which
        still adds e.g. the initial resets and the readout pulses before sending
    :rtype: Tuple[Dict[int, int], Dict[str, Any]]
    """
    import qiskit

    _check_circuit(circuit)
    windows = _readout_windows(circuit)
    records = windows[-1]["records"]
    options: Dict[str, Any] = {"basis_gates": list(properties["native_gates"])}
    if compiling:
        options.update(
            coupling_map=[list(link) for link in properties["links"]],
            optimization_level=2,
            seed_transpiler=default_seed,
        )
    else:
        options["optimization_level"] = 0
    options.update(compiled_options or {})
    compiled, _ = qiskit_compile(
        _logical_qiskit(circuit, records),
        output="qiskit",
        compiled_options=options,
        remove_barriers=False,
    )
    instructions, dropped = _native_instructions(compiled)
    physical_qubits = sorted(
        {q for op in instructions if op["name"] != "barrier" for q in op["qubits"]}
    )
    compilation = {
        "schema_version": compilation_schema_version,
        "shots": shots,
        "physical_qubits": physical_qubits,
        "n_clbits": max(records) + 1,
        "instructions": instructions,
        "windows": [
            {"kind": w["kind"], "records": sorted(w["records"])} for w in windows
        ],
        "dropped_resets": dropped,
        "compiler": {
            "name": "qiskit",
            "version": qiskit.__version__,
            "compiling": compiling,
            "options": options,
        },
        "calibration_time": properties["calibration_time"],
    }
    return _mapping(compilation), compilation


def _mapping(compilation: Dict[str, Any]) -> Dict[int, int]:
    # positional_logical_mapping from the classical bit to the physical qubit
    return dict(
        sorted(
            (op["clbits"][0], op["qubits"][0])
            for op in compilation["instructions"]
            if op["name"] == "measure"
        )
    )


def prepare_task(
    device: Device,
    token: Optional[str],
    circuit: Optional[AbstractCircuit] = None,
    source: Optional[str] = None,
    shots: int = 1024,
    compiling: bool = True,
    compiled_options: Optional[Dict[str, Any]] = None,
    **kws: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Locally compile the circuit into the program sent by :py:func:`submit_task`.

    :param device: the device
    :type device: Device
    :param token: the LQCloud API key
    :type token: Optional[str]
    :param circuit: the circuit, defaults to None
    :type circuit: Optional[AbstractCircuit], optional
    :param source: unsupported, raises if given, defaults to None
    :type source: Optional[str], optional
    :param shots: number of shots, defaults to 1024
    :type shots: int, optional
    :param compiling: whether choose the layout and route on the device topology,
        see :py:func:`submit_task`, defaults to True
    :type compiling: bool, optional
    :param compiled_options: ``qiskit.transpile`` options overriding the defaults,
        defaults to None
    :type compiled_options: Optional[Dict[str, Any]], optional
    :param kws: other options are unsupported and raise an error
    :type kws: Any
    :return: the keyword arguments of :py:func:`submit_task` with the compilation
        to send, and what is prepared for the circuit, i.e. its ``compilation`` and
        ``positional_logical_mapping``
    :rtype: Tuple[Dict[str, Any], List[Dict[str, Any]]]
    """
    _check_options(kws)
    if source is not None:
        raise ValueError("LQCloud provider does not support `source`, use `circuit`")
    if circuit is None:
        raise ValueError("`circuit` must be provided")
    if is_sequence(circuit) or is_sequence(shots):
        raise ValueError(
            "LQCloud provider does not support batch submission yet, "
            "submit the circuits one by one"
        )
    _, compilation = compile_circuit(
        circuit, list_properties(device, token), shots, compiling, compiled_options
    )
    return {"compilation": compilation}, [_prepared(compilation)]


def _check_options(kws: Dict[str, Any]) -> None:
    if kws:
        raise ValueError(
            "Unsupported options for the LQCloud provider: %s" % ", ".join(sorted(kws))
        )


def _prepared(compilation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "positional_logical_mapping": _mapping(compilation),
        "compilation": compilation,
    }


def _lq_circuit(compilation: Dict[str, Any]) -> Any:
    index = {q: i for i, q in enumerate(compilation["physical_qubits"])}
    qc = _lqcloud().QuantumCircuit(len(index), compilation["n_clbits"])
    for op in compilation["instructions"]:
        qubits = [index[q] for q in op["qubits"]]
        if op["name"] == "measure":
            qc.measure(qubits[0], op["clbits"][0])
        elif op["name"] == "barrier":
            qc.barrier(qubits)
        else:
            getattr(qc, op["name"])(*op.get("params", []), *qubits)
    return qc


def _capture_posts(session: Any) -> List[Any]:
    # the requests posted by the session, as sent to the server
    posts: List[Any] = []

    def hook(r: Any, *args: Any, **kws: Any) -> None:
        if r.request.method == "POST":
            posts.append(r.request)

    session.hooks["response"].append(hook)
    return posts


def _payload(request: Any) -> Any:
    body = request.body
    if request.headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    return json.loads(body)


def _send(backend: Any, qc: Any, shots: int, initial_layout: List[int]) -> Any:
    """
    Run the circuit once, where the SDK retries the request with an idempotency
    key. Local validation failures before the request are raised as they are.
    """
    lq = _lqcloud()
    try:
        job = backend.run(qc, shots=shots, initial_layout=initial_layout)
    except lq.AuthenticationError as e:
        raise SubmissionRejected(str(e)) from e
    except lq.JobError as e:
        # an HTTP error response, otherwise an unexpected response body
        response = getattr(e.__context__, "response", None)
        if (
            isinstance(e.__context__, requests.HTTPError)
            and response is not None
            and 400 <= response.status_code < 500
            and response.status_code != 408
            and e.error_type != "idempotency_in_flight"
        ):
            raise SubmissionRejected(str(e)) from e
        raise SubmissionUnknown(str(e)) from e
    except lq.LQCloudError as e:
        # transient network failures after retries, while e.g. CircuitError is local
        if isinstance(e.__cause__, requests.RequestException):
            raise SubmissionUnknown(str(e)) from e
        raise
    except requests.RequestException as e:
        raise SubmissionUnknown(repr(e)) from e
    if not isinstance(job.job_id, str) or not job.job_id:
        raise SubmissionUnknown("LQCloud did not return a valid task id")
    return job


def submit_task(
    device: Device,
    token: str,
    shots: int = 1024,
    circuit: Optional[AbstractCircuit] = None,
    compiling: bool = True,
    compiled_options: Optional[Dict[str, Any]] = None,
    compilation: Optional[Dict[str, Any]] = None,
    **kws: Any,
) -> Task:
    """
    Submit a circuit to LQCloud. The circuit is compiled locally with qiskit into
    the native gates ``rz``, ``xhalf`` (i.e. ``sx``), ``x`` and ``cz`` on the
    physical qubits, which LQCloud runs without further compiling, see
    :py:func:`compile_circuit`. The compilation is kept in
    ``task.prepared["compilation"]``, and the request body as sent to the server,
    i.e. with the rewrites by the SDK such as the readout pulses before
    measurements, in ``task.prepared["payload"]``, both also in the task store.

    Only terminal measurements are supported for now; noise channels, resets,
    batch submissions and readout correction by the provider are unsupported.
    A state vector ``inputs`` is synthesized by qiskit into a state preparation
    circuit, whose depth grows exponentially with the number of qubits, so it is
    only practical on a few qubits; prepare basis states with ``x`` gates instead.

    :param device: the device
    :type device: Device
    :param token: the LQCloud API key
    :type token: str
    :param shots: number of shots, defaults to 1024
    :type shots: int, optional
    :param circuit: the circuit, optionally with a state vector ``inputs``
    :type circuit: Optional[AbstractCircuit], optional
    :param compiling: whether choose the layout and route on the device topology;
        if False, qubit ``i`` of the circuit runs on physical qubit ``i``,
        defaults to True
    :type compiling: bool, optional
    :param compiled_options: ``qiskit.transpile`` options overriding the defaults,
        e.g. ``{"initial_layout": [5, 6]}``, defaults to None
    :type compiled_options: Optional[Dict[str, Any]], optional
    :param compilation: a compilation to send directly, e.g. by
        :py:func:`prepare_task`, defaults to None
    :type compilation: Optional[Dict[str, Any]], optional
    :param kws: other options are unsupported and raise an error
    :type kws: Any
    :return: the task
    :rtype: Task
    """
    _check_options(kws)
    if compilation is None:
        _, (prepared,) = prepare_task(
            device, token, circuit, None, shots, compiling, compiled_options
        )
        compilation = prepared["compilation"]
    backend = _backend(device, token)
    posts = _capture_posts(backend.provider.session)
    job = _send(
        backend,
        _lq_circuit(compilation),
        compilation["shots"],
        compilation["physical_qubits"],
    )
    task = Task(job.job_id, device=device)
    # the task is accepted, so a missing or unreadable payload must not lose it
    payload = None
    if posts:
        try:
            payload = _payload(posts[-1])
        except (ValueError, OSError, TypeError) as e:
            logger.warning("cannot read the request body sent: %r", e)
    task.prepared = dict(_prepared(compilation), payload=payload)
    return task


def resubmit_task(task: Task, token: str) -> Task:
    """
    Run the compilation of the task, i.e. ``task.prepared["compilation"]``,
    again as a new task.

    :param task: the task to rerun
    :type task: Task
    :param token: the LQCloud API key
    :type token: str
    :return: the new task
    :rtype: Task
    """
    if task.prepared is None:
        raise ValueError(
            "The compilation of task %s is unknown, resubmit the circuit instead"
            % task.id_
        )
    return submit_task(
        task.get_device(), token, compilation=task.prepared["compilation"]
    )


def _job(task: Task, device: Device, token: Optional[str]) -> Any:
    provider = _provider(token)
    return _lqcloud().job.Job(
        task.id_, provider, device.name, base_url=provider.url, use_async=True
    )


def remove_task(task: Task, token: str) -> Any:
    """
    Cancel a queued task, while a running task cannot be cancelled.

    :param task: the task
    :type task: Task
    :param token: the LQCloud API key
    :type token: str
    :return: the cancellation response of the SDK
    :rtype: Any
    """
    return _job(task, task.get_device(), token).cancel()


# the compiled circuit is already on the physical qubits, which run as they are
def task_metadata(details: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return {"positional_logical_mapping": prepared.get("positional_logical_mapping")}


def get_task_details(
    task: Task, device: Device, token: Optional[str], prettify: bool
) -> Dict[str, Any]:
    """
    Get the task details by a single query, with the unified ``state``
    (``pending``, ``completed`` or ``failed``), the raw counts as ``results``,
    the per-shot ``memory`` if returned, and the provider fields in ``lqcloud``.

    :param task: the task
    :type task: Task
    :param device: the device of the task
    :type device: Device
    :param token: the LQCloud API key
    :type token: Optional[str]
    :param prettify: unused, the details hold no python objects to prettify
    :type prettify: bool
    :return: the task details
    :rtype: Dict[str, Any]
    """
    info = _job(task, device, token).status_info()
    status = info.get("status")
    if status not in _states:
        raise ValueError("Unknown LQCloud task status: %s" % status)
    details: Dict[str, Any] = {
        "id": task.id_,
        "state": _states[status],
        "device": str(device),
        "results": {},
        "shots": None,
        "lqcloud": {k: info.get(k) for k in _lqcloud_fields},
    }
    if details["state"] == "failed":
        details["err"] = info.get("error") or status
    if status == "completed":
        data = info.get("result")
        if not isinstance(data, dict):
            raise ValueError(
                "Unsupported LQCloud result of task %s: %r" % (task.id_, data)
            )
        result = _lqcloud().Result(device.name, task.id_, status, data, info)
        details["results"] = result.get_raw_counts()
        details["shots"] = data.get("shots")
        if result.result_format == "memory":
            details["memory"] = result.get_memory()
        details["lqcloud"]["result_format"] = result.result_format
    return details
