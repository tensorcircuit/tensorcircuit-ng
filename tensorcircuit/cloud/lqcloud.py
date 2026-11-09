"""
Cloud provider from LQCloud (逻辑比特量子云)
https://cloud.logicalqubit.com
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import gzip
import json
import logging

import requests

from .abstraction import Device, sep, Task
from .config import lqcloud_base_url
from .utils import SubmissionRejected, SubmissionUnknown
from ..abstractcircuit import AbstractCircuit
from ..compiler.qiskit_compiler import qiskit_compile, properties_to_target
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
# the options of the SDK ``backend.run`` allowed in run_options, since the SDK
# ignores unknown options silently
_run_options = {"dynamic_decoupling", "result_format"}

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
    # some couplers are listed without their qubits, e.g. on AGate-100
    cz_fidelity = {
        tuple(sorted(m["qubits"])): m.get("cz_fidelity")
        for m in calibration.get("coupler_metrics") or []
        if "qubits" in m
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


def _two_state(properties: Dict[str, Any]) -> bool:
    # whether the SDK reads out by ``x21`` then ``measure`` on the device, by the
    # rule of ``CloudBackend._resolve_measurement_kind`` in the SDK
    backend = _lqcloud().provider.cloud_provider.CloudBackend
    types = backend._normalize_supported_measurement_types(
        properties.get("supported_measurement_types")
    )
    return "measure" not in types


def _check_circuit(circuit: Any) -> None:
    if not isinstance(circuit, AbstractCircuit):
        raise ValueError(
            "LQCloud provider only supports TensorCircuit circuits, got %s"
            % type(circuit).__name__
        )
    if any(d.get("is_channel", False) for d in circuit.to_qir()):
        raise ValueError("Noise channels cannot be submitted to the LQCloud hardware")
    instructions = {d["name"] for d in circuit._extra_qir} - {
        "measure",
        "reset",
        "barrier",
    }
    if instructions:
        raise ValueError(
            "Unsupported instructions for the LQCloud provider: %s"
            % ", ".join(sorted(instructions))
        )
    if any(d.get("parameters", {}).get("p") for d in circuit._extra_qir):
        raise ValueError(
            "Readout flip probabilities of measurements are for simulation only "
            "and cannot be submitted to the LQCloud hardware"
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


def _logical_qiskit(circuit: AbstractCircuit) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Arrange the circuit into readout windows, each a set of measurements read out
    together between barriers over all qubits. A measurement followed by no other
    operation on its qubit is terminal and read out in the final window after all
    gates, where all qubits are measured if no measurement is given. The other
    measurements are read out in mid-circuit windows, one for each run of them
    with no other operation in between, after which the measured qubits go on in
    the measured states without a reset, unless one is given.

    :return: the qiskit circuit, and the windows in time order, each with its
        ``kind``, ``"mid"`` or ``"final"``, and ``records``, i.e. the classical
        bits of its measurements
    :rtype: Tuple[Any, List[Dict[str, Any]]]
    """
    from qiskit import QuantumCircuit

    # Whether mid-circuit measurements run depends on the device and the qubits, and
    # the task fails with ``internal_error`` otherwise. The check is left to the
    # cloud on purpose, without any local check or layout around it.
    qc = circuit.to_qiskit(enable_instruction=True, enable_inputs=True)
    data = [
        (
            inst.operation,
            [qc.find_bit(q).index for q in inst.qubits],
            [qc.find_bit(c).index for c in inst.clbits],
        )
        for inst in qc.data
    ]
    clbits = [c[0] for op, _, c in data if op.name == "measure"]
    repeated = sorted({c for c in clbits if clbits.count(c) > 1})
    if repeated:
        raise ValueError(
            "Classical bits %s are written by more than one measurement" % repeated
        )
    terminal = set()
    later = set()
    for i in reversed(range(len(data))):
        op, qubits, _ = data[i]
        if op.name == "barrier":
            continue
        if op.name == "measure" and qubits[0] not in later:
            terminal.add(i)
        later.update(qubits)
    final = {data[i][2][0]: data[i][1][0] for i in terminal}
    if not clbits:
        final = {q: q for q in range(qc.num_qubits)}
    lqc = QuantumCircuit(qc.num_qubits, max(clbits + list(final)) + 1)
    windows: List[Dict[str, Any]] = []
    window: Optional[Dict[str, Any]] = None
    for i, (op, qubits, cs) in enumerate(data):
        if i in terminal:
            continue
        if op.name == "measure":
            if window is None:
                window = {"kind": "mid", "records": []}
                windows.append(window)
                lqc.barrier()
            window["records"].append(cs[0])
            lqc.measure(qubits[0], cs[0])
            continue
        if window is not None:
            lqc.barrier()
            window = None
        lqc.append(op, qubits)
    if final:
        # the barrier also closes a mid-circuit window followed only by final readouts
        lqc.barrier()
        for r, q in sorted(final.items()):
            lqc.measure(q, r)
        windows.append({"kind": "final", "records": sorted(final)})
    return lqc, windows


def _native_instructions(
    qc: Any, two_state: bool
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Translate the compiled qiskit circuit into native instructions on the active
    physical qubits, i.e. those with any operation besides barriers. The resets
    before any operation, e.g. from the state preparation, are dropped since the
    SDK resets every qubit at the beginning. With the two-state readout, i.e.
    ``x21`` then ``measure`` by the SDK, a measured |1> is left in |2>, so
    ``x21`` is added before the next operation on the qubit unless it is a reset.
    """
    ops: List[Dict[str, Any]] = []
    dropped = []
    started = set()
    shelved: Set[int] = set()
    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            ops.append({"name": "barrier", "qubits": qubits})
            continue
        if name == "reset" and qubits[0] not in started:
            dropped.append(qubits[0])
            continue
        for q in sorted(shelved.intersection(qubits)):
            shelved.remove(q)
            if name != "reset":
                ops.append({"name": "x21", "qubits": [q]})
        started.update(qubits)
        if name == "reset":
            ops.append({"name": "reset", "qubits": qubits})
        elif name == "measure":
            if two_state:
                shelved.add(qubits[0])
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
    noise_aware: bool = True,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """
    Compile the circuit locally into native gates on the physical qubits of the
    device, since LQCloud runs the submitted gates as they are.
    The terminal measurements are read out in the final window after all gates, in
    the order of the classical bits, and the others in mid-circuit windows, see
    below. The result bitstring is indexed by the classical bit, i.e. the
    ``record_index`` of the measurement, which must differ for each measurement.

    A measured qubit goes on in the measured state, and is reset only by a
    ``reset_instruction``, which runs as the native reset. On a device with only the
    two-state readout, where a measured |1> is left in |2>, ``x21`` is added before
    the next operation on the qubit unless it is a reset. This restore is imperfect,
    so prefer a reset when the state after the measurement is not needed. Whether
    mid-circuit measurements run depends on the device and the qubits, e.g. on the
    measurements of the other qubits sharing a readout line, and a circuit that
    cannot run fails with the ``internal_error`` of the task, so it may depend on
    the layout chosen.

    :param circuit: the circuit
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
    :param noise_aware: whether choose the layout by the error rates of the device
        when compiling, with the qubits and couplers of unknown errors unused, see
        :py:func:`tensorcircuit.compiler.qiskit_compiler.properties_to_target`,
        unless ``coupling_map`` or ``basis_gates`` is in ``compiled_options``;
        the errors of the qubits and couplers used are kept in the compilation,
        defaults to True
    :type noise_aware: bool, optional
    :return: the positional_logical_mapping from the classical bit to the physical
        qubit, and the compilation record, i.e. the program given to the SDK, which
        still adds e.g. the initial resets and the readout pulses before sending
    :rtype: Tuple[Dict[int, int], Dict[str, Any]]
    """
    import qiskit

    _check_circuit(circuit)
    logical, windows = _logical_qiskit(circuit)
    options: Dict[str, Any] = {"basis_gates": list(properties["native_gates"])}
    # the coupling_map or basis_gates given by the user is kept
    noise_aware = noise_aware and not {"coupling_map", "basis_gates"} & set(
        compiled_options or {}
    )
    if compiling:
        options.update(optimization_level=2, seed_transpiler=default_seed)
        if noise_aware:
            options["target"] = properties_to_target(
                properties, options.pop("basis_gates")
            )
        else:
            options["coupling_map"] = [list(link) for link in properties["links"]]
    else:
        # without a coupling map, level 1 keeps the qubits and merges single qubit gates
        options["optimization_level"] = 1
    options.update(compiled_options or {})
    compiled, _ = qiskit_compile(
        logical,
        output="qiskit",
        compiled_options=options,
        remove_barriers=False,
    )
    two_state = _two_state(properties)
    instructions, dropped = _native_instructions(compiled, two_state)
    physical_qubits = sorted(
        {q for op in instructions if op["name"] != "barrier" for q in op["qubits"]}
    )
    compilation = {
        "schema_version": compilation_schema_version,
        "shots": shots,
        "physical_qubits": physical_qubits,
        "n_clbits": logical.num_clbits,
        "instructions": instructions,
        "windows": windows,
        "dropped_resets": dropped,
        # the readout the x21 restores are compiled for, checked when sending
        "two_state_readout": two_state,
        "compiler": {
            "name": "qiskit",
            "version": qiskit.__version__,
            "compiling": compiling,
            "noise_aware": compiling and noise_aware,
            # the target is built from the properties, see ``calibration``
            "options": {k: v for k, v in options.items() if k != "target"},
        },
        "calibration_time": properties["calibration_time"],
        "calibration": _calibration(properties, instructions, physical_qubits),
    }
    return _mapping(compilation), compilation


def _calibration(
    properties: Dict[str, Any], instructions: List[Dict[str, Any]], qubits: List[int]
) -> Dict[str, Any]:
    # the errors of the qubits and couplers used, by which the layout is chosen
    couplers = sorted(
        {tuple(sorted(op["qubits"])) for op in instructions if op["name"] == "cz"}
    )
    return {
        "bits": [dict(properties["bits"].get(q) or {}, qubit=q) for q in qubits],
        "links": [
            dict(properties["links"].get(c) or {}, qubits=list(c)) for c in couplers
        ],
    }


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
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    source: Optional[str] = None,
    shots: Union[int, Sequence[int]] = 1024,
    compiling: bool = True,
    compiled_options: Optional[
        Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]
    ] = None,
    noise_aware: bool = True,
    run_options: Optional[Dict[str, Any]] = None,
    **kws: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Locally compile the circuits into the program sent by :py:func:`submit_task`.

    :param device: the device
    :type device: Device
    :param token: the LQCloud API key
    :type token: Optional[str]
    :param circuit: the circuit, or a list of circuits for a batch, defaults to None
    :type circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]], optional
    :param source: unsupported, raises if given, defaults to None
    :type source: Optional[str], optional
    :param shots: number of shots, or one per circuit with the same value for
        a batch, defaults to 1024
    :type shots: Union[int, Sequence[int]], optional
    :param compiling: whether choose the layout and route on the device topology,
        see :py:func:`submit_task`, defaults to True
    :type compiling: bool, optional
    :param compiled_options: ``qiskit.transpile`` options overriding the defaults,
        or one per circuit, defaults to None
    :type compiled_options: Optional[Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]], optional
    :param noise_aware: whether choose the layout by the error rates of the device,
        see :py:func:`compile_circuit`, defaults to True
    :type noise_aware: bool, optional
    :param run_options: the options of the SDK ``backend.run`` for all circuits,
        kept in the compilation as ``run_options``, see :py:func:`submit_task`,
        defaults to None
    :type run_options: Optional[Dict[str, Any]], optional
    :param kws: options of other providers, ignored
    :type kws: Any
    :return: the keyword arguments of :py:func:`submit_task` with the compilation
        to send, a list for a batch, and what is prepared for each circuit, i.e.
        its ``compilation`` and ``positional_logical_mapping``
    :rtype: Tuple[Dict[str, Any], List[Dict[str, Any]]]
    """
    if source is not None:
        raise ValueError("LQCloud provider does not support `source`, use `circuit`")
    if circuit is None:
        raise ValueError("`circuit` must be provided")
    circuits = list(circuit) if is_sequence(circuit) else [circuit]  # type: ignore
    properties = list_properties(device, token)
    compilations = [
        dict(
            compile_circuit(c, properties, s, compiling, o, noise_aware)[1],
            run_options=dict(run_options or {}),
        )
        for c, s, o in zip(
            circuits,
            _per_circuit(shots, len(circuits), "shots"),
            _per_circuit(compiled_options, len(circuits), "compiled_options"),
        )
    ]
    send = compilations if is_sequence(circuit) else compilations[0]
    return {"compilation": send}, [_prepared(c) for c in compilations]


def _per_circuit(value: Any, n: int, name: str) -> List[Any]:
    if not is_sequence(value):
        return [value] * n
    if len(value) != n:
        raise ValueError("`%s` has %d entries for %d circuits" % (name, len(value), n))
    return list(value)


def _prepared(compilation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "positional_logical_mapping": _mapping(compilation),
        "compilation": compilation,
    }


def _lq_circuit(compilation: Dict[str, Any], layout: List[int]) -> Any:
    # the circuit on the qubits of ``layout``, the initial_layout given to the SDK
    index = {q: i for i, q in enumerate(layout)}
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


def _capture_posts(session: Any, batch: bool) -> List[Any]:
    # the requests posted by the session, as sent to the server
    posts: List[Any] = []

    def hook(r: Any, *args: Any, **kws: Any) -> None:
        if r.request.method == "POST":
            posts.append(r.request)
            # the SDK takes any 400 on a batch for an old server and silently
            # submits each circuit as a separate task, which costs more
            if batch and r.status_code == 400:
                raise SubmissionRejected(
                    "LQCloud rejected the batch submission (HTTP 400): %s" % r.text
                )

    session.hooks["response"].append(hook)
    return posts


def _payload(request: Any) -> Any:
    body = request.body
    if request.headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    return json.loads(body)


def _circuit_payload(body: Dict[str, Any], i: int) -> Dict[str, Any]:
    # the batch request body with only the entry of circuit ``i``
    command = body["command"]
    return dict(body, command=dict(command, circuits=[command["circuits"][i]]))


def _send(
    backend: Any,
    qc: Any,
    shots: int,
    initial_layout: List[int],
    run_options: Dict[str, Any],
) -> Any:
    """
    Run the circuit once, where the SDK retries the request with an idempotency
    key. Local validation failures before the request are raised as they are.
    """
    lq = _lqcloud()
    try:
        job = backend.run(qc, shots=shots, initial_layout=initial_layout, **run_options)
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
    shots: Union[int, Sequence[int]] = 1024,
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    compiling: bool = True,
    compiled_options: Optional[
        Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]
    ] = None,
    compilation: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    noise_aware: bool = True,
    run_options: Optional[Dict[str, Any]] = None,
    **kws: Any,
) -> Union[Task, List[Task]]:
    """
    Submit a circuit to LQCloud. The circuit is compiled locally with qiskit into
    the native gates ``rz``, ``xhalf`` (i.e. ``sx``), ``x`` and ``cz`` on the
    physical qubits, which LQCloud runs without further compiling, see
    :py:func:`compile_circuit`. By default the layout is chosen by the error rates
    of the device. The compilation is kept in
    ``task.prepared["compilation"]``, and the request body as sent to the server,
    i.e. with the rewrites by the SDK such as the readout pulses before
    measurements, in ``task.prepared["payload"]``, both also in the task store.

    A list of circuits is submitted as one LQCloud batch task, with one Task per
    circuit, whose id is ``"<batch task id>#<circuit index>"`` and whose payload
    holds only its own entry of the batch request. The circuits of a batch share
    the shots, while each may have its own ``compiled_options``, e.g. its
    ``initial_layout``. If LQCloud rejects the batch, the SDK fallback to a
    separate task per circuit is not used.

    Measurements may be mid-circuit or repeated, each to its own classical bit,
    and resets run as the native reset, see :py:func:`compile_circuit`; noise
    channels and readout correction by the provider are unsupported.
    A state vector ``inputs`` is synthesized by qiskit into a state preparation
    circuit, whose depth grows exponentially with the number of qubits, so it is
    only practical on a few qubits; prepare basis states with ``x`` gates instead.

    :param device: the device
    :type device: Device
    :param token: the LQCloud API key
    :type token: str
    :param shots: number of shots, or one per circuit with the same value for
        a batch, defaults to 1024
    :type shots: Union[int, Sequence[int]], optional
    :param circuit: the circuit, optionally with a state vector ``inputs``, or
        a list of circuits for a batch
    :type circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]], optional
    :param compiling: whether choose the layout and route on the device topology;
        if False, qubit ``i`` of the circuit runs on physical qubit ``i``,
        defaults to True
    :type compiling: bool, optional
    :param compiled_options: ``qiskit.transpile`` options overriding the defaults,
        e.g. ``{"initial_layout": [5, 6]}``, or one per circuit, defaults to None
    :type compiled_options: Optional[Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]], optional
    :param compilation: a compilation to send directly, or a list for a batch,
        e.g. by :py:func:`prepare_task`, defaults to None
    :type compilation: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]], optional
    :param noise_aware: whether choose the layout by the error rates of the device,
        see :py:func:`compile_circuit`, defaults to True
    :type noise_aware: bool, optional
    :param run_options: options of the SDK ``backend.run``, shared by the circuits
        of a batch and kept in the compilation, only ``dynamic_decoupling`` and
        ``result_format`` for now, e.g.
        ``{"dynamic_decoupling": {"enabled": True, "m": 4, "dd_type": "CPMG"}}`` to
        fill idle windows such as mid-circuit readouts, with its ``qubits``, if
        given, on the physical qubits, defaults to None
    :type run_options: Optional[Dict[str, Any]], optional
    :param kws: options of other providers, ignored
    :type kws: Any
    :return: the task, or a list of tasks for a batch
    :rtype: Union[Task, List[Task]]
    """
    # billing by the LQCloud docs for SDK 0.5.1: (¥5 + ¥0.005 × shots) × multiplier
    # per task, and (¥3 × circuits + ¥0.005 × total shots) × multiplier per batch,
    # where the multiplier depends on the device and the subscription; failed
    # tasks are free, and the charged cost is in ``details()["lqcloud"]``
    if compilation is None:
        send_kws, _ = prepare_task(
            device,
            token,
            circuit,
            None,
            shots,
            compiling,
            compiled_options,
            noise_aware,
            run_options,
        )
        compilation = send_kws["compilation"]
    batch = is_sequence(compilation)
    compilations = list(compilation) if batch else [compilation]  # type: ignore
    batch_shots = {c["shots"] for c in compilations}
    if len(batch_shots) > 1:
        raise ValueError(
            "The circuits of an LQCloud batch share the shots, got %s"
            % sorted(batch_shots)
        )
    options = [c.get("run_options", {}) for c in compilations]
    if any(o != options[0] for o in options):
        raise ValueError(
            "The circuits of an LQCloud batch share the run options, got %s" % options
        )
    unknown = sorted(set(options[0]) - _run_options)
    if unknown:
        raise ValueError(
            "Unsupported LQCloud run options %s, only %s"
            % (unknown, sorted(_run_options))
        )
    # the circuits of a batch share the initial_layout given to the SDK, i.e.
    # all their physical qubits, on which each runs as compiled
    layout = sorted({q for c in compilations for q in c["physical_qubits"]})
    backend = _backend(device, token)
    two_state = _two_state(backend.config)
    if any(c["two_state_readout"] != two_state for c in compilations):
        raise ValueError(
            "The compilation does not match the current readout of %s, compile "
            "the circuit again" % device.name
        )
    sdk_options = options[0]
    dd = sdk_options.get("dynamic_decoupling")
    if isinstance(dd, dict) and dd.get("qubits") is not None:
        # physical qubits to their positions in the initial_layout, as the SDK takes
        index = {q: i for i, q in enumerate(layout)}
        sdk_options = dict(
            sdk_options,
            dynamic_decoupling=dict(dd, qubits=[index[q] for q in dd["qubits"]]),
        )
    posts = _capture_posts(backend.provider.session, batch)
    qcs = [_lq_circuit(c, layout) for c in compilations]
    job = _send(
        backend, qcs if batch else qcs[0], compilations[0]["shots"], layout, sdk_options
    )
    # the tasks are accepted, so a missing or unreadable payload must not lose them
    payloads: List[Any] = [None] * len(compilations)
    if posts:
        try:
            body = _payload(posts[-1])
            if batch:
                payloads = [_circuit_payload(body, i) for i in range(len(qcs))]
            else:
                payloads = [body]
        except (ValueError, OSError, TypeError, KeyError, IndexError) as e:
            logger.warning("cannot read the request body sent: %r", e)
    tasks = []
    for i, (c, payload) in enumerate(zip(compilations, payloads)):
        task = Task("%s#%d" % (job.job_id, i) if batch else job.job_id, device=device)
        task.circuit_index = i
        task.prepared = dict(_prepared(c), payload=payload)
        tasks.append(task)
    return tasks if batch else tasks[0]


def resubmit_task(task: Task, token: str) -> Task:
    """
    Run the compilation of the task, i.e. ``task.prepared["compilation"]``,
    again as a new task, which is a single task also for a circuit of a batch.

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
    return submit_task(  # type: ignore
        task.get_device(), token, compilation=task.prepared["compilation"]
    )


def _job(job_id: str, device: Device, token: Optional[str]) -> Any:
    provider = _provider(token)
    return _lqcloud().job.Job(
        job_id, provider, device.name, base_url=provider.url, use_async=True
    )


def _split_id(task: Task) -> Tuple[str, Optional[int]]:
    # the LQCloud task id, and the circuit index for a circuit of a batch
    job_id, _, index = task.id_.partition("#")
    return job_id, int(index) if index else None


def remove_task(task: Task, token: str) -> Any:
    """
    Cancel a queued task, while a running task cannot be cancelled. Cancelling
    a circuit of a batch cancels the whole batch.

    :param task: the task
    :type task: Task
    :param token: the LQCloud API key
    :type token: str
    :return: the cancellation response of the SDK
    :rtype: Any
    """
    return _job(_split_id(task)[0], task.get_device(), token).cancel()


# the compiled circuit is already on the physical qubits, which run as they are
def task_metadata(details: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return {"positional_logical_mapping": prepared.get("positional_logical_mapping")}


def get_task_details(
    task: Task, device: Device, token: Optional[str], prettify: bool
) -> Dict[str, Any]:
    """
    Get the task details by a single query, with the unified ``state``
    (``pending``, ``completed`` or ``failed``), the raw counts as ``results``,
    the per-shot ``memory`` if returned, and the provider fields in ``lqcloud``,
    with the other fields of the result as returned in ``lqcloud["result"]``.
    For a circuit of a batch, the state, results and ``lqcloud["result"]`` are
    its own, while the other provider fields, e.g. ``cost``, are the ones of the
    whole batch task.

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
    job_id, index = _split_id(task)
    info = _job(job_id, device, token).status_info()
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
        batch = data.get("result_format") == "batch" or "results" in data
        if batch != (index is not None):
            raise ValueError(
                "LQCloud task %s is %s, query a circuit of a batch as `<id>#<index>`"
                % (job_id, "a batch" if batch else "not a batch")
            )
        if batch:
            data = data["results"][index]
            if not data.get("success", True):
                details["state"] = "failed"
                details["err"] = data.get("error") or "failed"
                return details
        result = _lqcloud().Result(device.name, task.id_, status, data, info)
        details["results"] = result.get_raw_counts()
        details["shots"] = data.get("shots", info["result"].get("shots"))
        if result.result_format == "memory":
            details["memory"] = result.get_memory()
        details["lqcloud"]["result_format"] = result.result_format
        # the other fields of the result as returned, e.g. dynamic_decoupling_info
        details["lqcloud"]["result"] = {
            k: v for k, v in data.items() if k not in ["counts", "raw_counts", "memory"]
        }
    return details
