"""
Tests for the LQCloud provider.

Offline tests run the real SDK against a fake server. Set ``TC_CLOUD_TEST=1``
and ``TC_TOKEN_LQCLOUD`` to run online tests, and additionally
``TC_CLOUD_HARDWARE_TEST=1`` for paid hardware tests.
"""

import gzip
import importlib
import importlib.util
import json
import os
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import pytest
import sympy

import tensorcircuit as tc
from tensorcircuit.cloud import apis, lqcloud, store, wrapper
from tensorcircuit.cloud.abstraction import Device, TaskFailed
from tensorcircuit.cloud.config import lqcloud_base_url
from tensorcircuit.cloud.utils import SubmissionRejected, SubmissionUnknown

HAS_SDK = (
    importlib.util.find_spec("lqcloud") is not None
    and importlib.util.find_spec("qiskit") is not None
)
CLOUD_TESTS_ENABLED = (
    HAS_SDK
    and bool(os.getenv("TC_TOKEN_LQCLOUD"))
    and os.getenv("TC_CLOUD_TEST") == "1"
)
HARDWARE_TESTS_ENABLED = (
    CLOUD_TESTS_ENABLED and os.getenv("TC_CLOUD_HARDWARE_TEST") == "1"
)
requires_sdk = pytest.mark.skipif(not HAS_SDK, reason="lqcloud or qiskit missing")
requires_cloud = pytest.mark.skipif(
    not CLOUD_TESTS_ENABLED,
    reason="set TC_TOKEN_LQCLOUD and TC_CLOUD_TEST=1 to run cloud tests",
)
requires_hardware = pytest.mark.skipif(
    not HARDWARE_TESTS_ENABLED,
    reason="set TC_CLOUD_HARDWARE_TEST=1 to run paid hardware tests",
)

DEVICE = "lqcloud::FAKE6"
XHALF = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2


def fake_config():
    # 2 x 3 grid: 0 - 1 - 2 over 3 - 4 - 5
    edges = [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]]
    return {
        "name": "FAKE6",
        "status": "active",
        "type": "real_qpu",
        "qubits": 6,
        "billing_multiplier": 1.0,
        "supported_measurement_types": ["measure2"],
        "topology": {"type": "grid", "coupling_map": edges},
        "properties": {
            "last_update": "2026-11-08 06:00:00",
            "qubit_metrics": [
                {
                    "id": q,
                    "t1": 50.0,
                    "t2": 20.0,
                    "measure_f0": 0.98,
                    "measure_f1": 0.95,
                    "xeb_fidelity": 0.999,
                }
                for q in range(6)
            ],
            "coupler_metrics": [
                {"id": i, "qubits": e, "cz_fidelity": 0.99} for i, e in enumerate(edges)
            ],
        },
    }


def fake_properties():
    return lqcloud._properties("FAKE6", fake_config())


def response(body, status=200):
    import requests

    r = requests.Response()
    r.status_code = status
    r._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    r.headers["Content-Type"] = "application/json"
    r.url = lqcloud_base_url
    return r


class FakeLQCloud:
    """
    Fake LQCloud server behind ``requests.adapters.HTTPAdapter.send``, recording the
    requests as sent. ``submission`` is the response of the task submission, an
    exception to raise, or a function of the request body, and ``info`` is the task
    status, or a function of the task id.
    """

    def __init__(self):
        self.config = fake_config()
        self.requests = []
        self.submission = response({"task_id": "task_0001", "status": "queued"})
        self.info = {"status": "queued"}

    def __call__(self, request):
        r = self.answer(request)
        r.request = request
        return r

    def answer(self, request):
        method, url, body = request.method, request.url, request.body
        if request.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        self.requests.append(
            {
                "method": method,
                "url": url,
                "body": json.loads(body) if body else None,
                "headers": dict(request.headers),
            }
        )
        path = urlsplit(url).path
        if method == "GET" and path == "/api/v1/qpus":
            return response({"qpus": [self.config]})
        if method == "POST" and path == "/api/v1/tasks/async":
            if isinstance(self.submission, Exception):
                raise self.submission
            if callable(self.submission):
                return self.submission(self.requests[-1]["body"])
            return self.submission
        if method == "GET" and path.startswith("/api/v1/tasks/async/"):
            task_id = path.split("/")[-1]
            info = self.info(task_id) if callable(self.info) else self.info
            return response(dict(info, task_id=task_id))
        if method == "POST" and path.endswith("/cancel"):
            return response({"status": "cancelled"})
        raise AssertionError("unexpected request %s %s" % (method, url))

    def submissions(self):
        return [
            r
            for r in self.requests
            if r["method"] == "POST" and r["url"].endswith("/tasks/async")
        ]


@pytest.fixture
def fake_lqcloud(monkeypatch):
    import requests
    from lqcloud import http_session

    fake = FakeLQCloud()
    monkeypatch.setattr(
        requests.adapters.HTTPAdapter,
        "send",
        lambda adapter, request, **kws: fake(request),
    )
    monkeypatch.setattr(http_session, "time", SimpleNamespace(sleep=lambda s: None))
    monkeypatch.setenv("TC_TOKEN_LQCLOUD", "tc-key")
    # the SDK credentials must never be used
    monkeypatch.setenv("LQCLOUD_API_KEY", "sdk-env-key")
    monkeypatch.setenv("LQCLOUD_URL", "https://sdk-env.invalid")
    return fake


def marginal(c, order):
    n = c._nqubits
    p = np.real(tc.backend.numpy(c.probability())).reshape([2] * n)
    p = p.sum(axis=tuple(q for q in range(n) if q not in order))
    return p.transpose([sorted(order).index(q) for q in order]).reshape(-1)


def ideal(c):
    return marginal(c, list(c.get_positional_logical_mapping().values()))


def simulate(compilation):
    # the distribution of the classical bits by the native program
    index = {q: i for i, q in enumerate(compilation["physical_qubits"])}
    c = tc.Circuit(len(index))
    measures = {}
    for op in compilation["instructions"]:
        qubits = [index[q] for q in op["qubits"]]
        if op["name"] == "measure":
            measures[op["clbits"][0]] = qubits[0]
        elif op["name"] == "rz":
            c.rz(qubits[0], theta=op["params"][0])
        elif op["name"] == "xhalf":
            c.unitary(qubits[0], unitary=XHALF)
        elif op["name"] == "x":
            c.x(qubits[0])
        elif op["name"] == "cz":
            c.cz(*qubits)
    return marginal(c, [measures[i] for i in range(compilation["n_clbits"])])


def triangle_circuit():
    # the cycle of three qubits needs routing on a grid
    c = tc.Circuit(4)
    c.ry(0, theta=0.9)
    c.cx(0, 1)
    c.rx(1, theta=1.3)
    c.cx(1, 2)
    c.rz(2, theta=0.8)
    c.cx(2, 0)
    c.x(3)
    c.barrier_instruction(0, 1)
    c.measure_instruction(2)
    c.measure_instruction(0)
    c.measure_instruction(1)
    return c


@requires_sdk
def test_compile_routes_and_reads_out_in_final_window():
    c = triangle_circuit()
    mapping, comp = lqcloud.compile_circuit(c, fake_properties(), shots=100)
    ops = comp["instructions"]
    assert {op["name"] for op in ops} <= {
        "rz",
        "xhalf",
        "x",
        "cz",
        "barrier",
        "measure",
    }
    # positions are the classical bits, mapped to physical qubits
    assert comp["n_clbits"] == 3 and comp["windows"] == [
        {"kind": "final", "records": [0, 1, 2]}
    ]
    measures = [op for op in ops if op["name"] == "measure"]
    assert [op["clbits"][0] for op in measures] == [0, 1, 2]
    assert mapping == {op["clbits"][0]: op["qubits"][0] for op in measures}
    # one barrier over all active qubits right before the measurements
    final = ops.index(measures[0]) - 1
    assert ops[final]["name"] == "barrier"
    assert sorted(ops[final]["qubits"]) == comp["physical_qubits"]
    assert all(op["name"] == "measure" for op in ops[final + 1 :])
    edges = {tuple(sorted(e)) for e in fake_config()["topology"]["coupling_map"]}
    assert all(tuple(sorted(op["qubits"])) in edges for op in ops if op["name"] == "cz")
    np.testing.assert_allclose(simulate(comp), ideal(c), atol=1e-5)
    assert comp["shots"] == 100


@requires_sdk
def test_compile_inputs_with_initial_layout():
    rng = np.random.default_rng(1)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    c = tc.Circuit(3, inputs=psi)
    c.h(0)
    c.cx(0, 1)
    mapping, comp = lqcloud.compile_circuit(
        c, fake_properties(), compiled_options={"initial_layout": [5, 4, 3]}
    )
    # without measurements, all qubits are measured, routing may permute them
    assert sorted(mapping) == [0, 1, 2] and sorted(mapping.values()) == [3, 4, 5]
    assert comp["physical_qubits"] == [3, 4, 5]
    # the state preparation starts with resets, done by the SDK anyway
    assert sorted(comp["dropped_resets"]) == [3, 4, 5]
    assert "reset" not in {op["name"] for op in comp["instructions"]}
    np.testing.assert_allclose(simulate(comp), ideal(c), atol=1e-5)


@requires_sdk
def test_compile_chooses_the_layout_by_the_errors():
    config = fake_config()
    for m in config["properties"]["coupler_metrics"]:
        m["cz_fidelity"] = 0.999 if m["qubits"] == [4, 5] else 0.95
    properties = lqcloud._properties("FAKE6", config)
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    _, comp = lqcloud.compile_circuit(c, properties)
    assert comp["physical_qubits"] == [4, 5] and comp["compiler"]["noise_aware"]
    # the errors it was chosen by are kept, and the target is not
    assert "target" not in comp["compiler"]["options"]
    assert [b["qubit"] for b in comp["calibration"]["bits"]] == [4, 5]
    ((link,),) = [comp["calibration"]["links"]]
    assert link["qubits"] == [4, 5]
    np.testing.assert_allclose(link["CZErrRate"], 0.001)
    np.testing.assert_allclose(simulate(comp), ideal(c), atol=1e-5)
    _, comp = lqcloud.compile_circuit(c, properties, noise_aware=False)
    assert not comp["compiler"]["noise_aware"]
    assert comp["compiler"]["options"]["coupling_map"]
    # the coupling map given by the user is kept
    _, comp = lqcloud.compile_circuit(
        c, properties, compiled_options={"coupling_map": [[0, 3], [3, 0]]}
    )
    assert comp["physical_qubits"] == [0, 3] and not comp["compiler"]["noise_aware"]


@requires_sdk
def test_compile_without_compiling_keeps_qubits():
    c = tc.Circuit(3)
    c.h(1)
    c.cx(1, 2)
    c.measure_instruction(2)
    c.measure_instruction(1)
    mapping, comp = lqcloud.compile_circuit(c, fake_properties(), compiling=False)
    assert mapping == {0: 2, 1: 1}
    assert comp["physical_qubits"] == [1, 2]
    np.testing.assert_allclose(simulate(comp), ideal(c), atol=1e-5)


def program(compilation):
    # the native program as a circuit, where x21 restores the measured |1>
    index = {q: i for i, q in enumerate(compilation["physical_qubits"])}
    c = tc.Circuit(len(index))
    for op in compilation["instructions"]:
        qubits = [index[q] for q in op["qubits"]]
        if op["name"] == "measure":
            c.measure_instruction(qubits[0])
            c._extra_qir[-1]["record_index"] = op["clbits"][0]
        elif op["name"] == "reset":
            c.reset_instruction(qubits[0])
        elif op["name"] == "rz":
            c.rz(qubits[0], theta=op["params"][0])
        elif op["name"] == "xhalf":
            c.unitary(qubits[0], unitary=XHALF)
        elif op["name"] in ["x", "cz"]:
            getattr(c, op["name"])(*qubits)
    return c


def record_distribution(c):
    # the exact joint distribution of the measurement records by classical bits
    clbits = [d["record_index"] for d in c._extra_qir if d["name"] == "measure"]
    for i in range(len(clbits)):
        c.detector_instruction([i])
    p = np.real(tc.backend.numpy(c.detector_probabilities()))
    return p.reshape([2] * len(clbits)).transpose(np.argsort(clbits)).reshape(-1)


def mid_circuit():
    c = tc.Circuit(3)
    c.h(0)
    c.cx(0, 2)
    c.measure_instruction(0)
    c.measure_instruction(2)
    # the measured qubits go on without a reset, unless one is given
    c.rx(0, theta=0.8)
    c.cx(2, 1)
    c.reset_instruction(2)
    c.ry(2, theta=0.7)
    c.measure_instruction(0)
    c.measure_instruction(2)
    c.measure_instruction(1)
    return c


@requires_sdk
@pytest.mark.parametrize("types, n_x21", [(["measure", "measure2"], 0), (["2"], 2)])
# the second layout routes the measured qubit 2 by a swap after the readout
@pytest.mark.parametrize("layout", [None, [0, 5, 1]])
def test_compile_mid_circuit_measurements(types, n_x21, layout):
    config = fake_config()
    config["supported_measurement_types"] = types
    properties = lqcloud._properties("FAKE6", config)
    c = mid_circuit()
    mapping, comp = lqcloud.compile_circuit(
        c, properties, compiled_options=layout and {"initial_layout": layout}
    )
    assert comp["two_state_readout"] == bool(n_x21)
    assert comp["windows"] == [
        {"kind": "mid", "records": [0, 1]},
        {"kind": "final", "records": [2, 3, 4]},
    ]
    ops = comp["instructions"]
    names = [op["name"] for op in ops]
    assert names.count("measure") == 5 and len(mapping) == 5
    assert names.count("reset") == 1 and names.count("x21") == n_x21
    # the mid-circuit window is between barriers over the physical qubits holding
    # the circuit qubits at that time, here 3, so not a routing ancilla, e.g. the
    # qubit 4 by the second layout
    first, last = names.index("measure"), names.index("measure") + 1
    assert names[last] == "measure"
    for i in [first - 1, last + 1]:
        assert ops[i]["name"] == "barrier" and len(ops[i]["qubits"]) == 3
        assert {op["qubits"][0] for op in ops[first : last + 1]} <= set(
            ops[i]["qubits"]
        )
    if n_x21:
        # x21 restores the measured qubits before their next operation
        restored = [op["qubits"][0] for op in ops if op["name"] == "x21"]
        assert sorted(restored) == sorted(
            op["qubits"][0] for op in ops[first : last + 1]
        )
        assert names.index("x21") > last + 1
    np.testing.assert_allclose(
        record_distribution(program(comp)), record_distribution(c), atol=1e-5
    )


@requires_sdk
@pytest.mark.parametrize(
    "ops, names",
    [
        ("xMRM", ["x", "barrier", "measure", "barrier", "reset", "barrier", "measure"]),
        ("xMM", ["x", "barrier", "measure", "barrier", "x21", "measure"]),
        (
            "xMMxM",
            ["x", "barrier", "measure", "x21", "measure", "barrier"]
            + ["x21", "x", "barrier", "measure"],
        ),
    ],
)
def test_compile_restores_after_two_state_readout(ops, names):
    # x21 before each operation after a measurement but a reset
    c = tc.Circuit(1)
    for op in ops:
        {"x": c.x, "M": c.measure_instruction, "R": c.reset_instruction}[op](0)
    _, comp = lqcloud.compile_circuit(c, fake_properties(), compiling=False)
    assert [op["name"] for op in comp["instructions"]] == names


def _same_clbit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)
    qc.measure(0, 0)
    return tc.Circuit.from_qiskit(qc)


def _readout_flip():
    c = tc.Circuit(1)
    c.measure_instruction(0, p=0.1)
    return c


def _detector():
    c = tc.Circuit(1)
    c.measure_instruction(0)
    c.detector_instruction([0])
    return c


def _channel():
    c = tc.DMCircuit(2)
    c.h(0)
    c.depolarizing(0, px=0.1, py=0.1, pz=0.1)
    return c


@requires_sdk
@pytest.mark.parametrize(
    "circuit, message",
    [
        (_same_clbit, "Classical bits \\[0\\]"),
        (_readout_flip, "Readout flip"),
        (_detector, "detector"),
        (_channel, "Noise channels"),
        (lambda: "OPENQASM 2.0;", "TensorCircuit circuits"),
    ],
)
def test_unsupported_circuits_raise(circuit, message):
    with pytest.raises(ValueError, match=message):
        lqcloud.compile_circuit(circuit(), fake_properties())


@requires_sdk
def test_submit_and_query_through_sdk(fake_lqcloud, tmp_path):
    c = triangle_circuit()
    s = store.TaskStore(tmp_path / "tasks")
    t = apis.submit_task(device=DEVICE, circuit=c, shots=100, store=s)
    (sent,) = fake_lqcloud.submissions()
    # the credential and url only come from TensorCircuit
    assert {r["headers"]["X-API-Key"] for r in fake_lqcloud.requests} == {"tc-key"}
    assert {urlsplit(r["url"]).netloc for r in fake_lqcloud.requests} == {
        urlsplit(lqcloud_base_url).netloc
    }
    comp = t.prepared["compilation"]
    circuit = sent["body"]["command"]["circuit"]
    assert circuit["initial_layout"] == comp["physical_qubits"]
    assert circuit["n_clbits"] == 3 and circuit["shots"] == 100
    # the SDK adds the resets and the two-state readout
    names = [op["name"] for op in circuit["instructions"]]
    assert names[: len(comp["physical_qubits"])] == ["reset"] * len(
        comp["physical_qubits"]
    )
    assert names.count("x21") == names.count("measure") == 3
    # the request body as sent, here compressed by the SDK
    assert sent["headers"]["Content-Encoding"] == "gzip"
    assert t.prepared["payload"] == sent["body"]
    # the readout wiring of the request matches the one derived from the compilation
    measures = [op for op in circuit["instructions"] if op["name"] == "measure"]
    assert {op["clbits"][0]: op["qubits"][0] for op in measures} == t.prepared[
        "positional_logical_mapping"
    ]

    for status in ["queued", "pending", "retrying", "running"]:
        fake_lqcloud.info = {"status": status, "queue_position": 1}
        d = t.details()
        assert d["state"] == "pending" and d["lqcloud"]["status"] == status
    fake_lqcloud.info = {
        "status": "completed",
        "cost": 5.0,
        "currency": "CNY",
        "result": {"counts": {"010": 60, "100": 40}, "shots": 100},
    }
    assert t.results() == {"010": 60, "100": 40}
    d = t.details()
    meta = d["tensorcircuit"]
    assert "memory" not in d
    assert meta["positional_logical_mapping"] == lqcloud._mapping(comp)

    record = s.get(t.local_id)
    assert record["state"] == "completed" and record["prepared"] == t.prepared
    assert record["details"]["lqcloud"]["cost"] == 5.0
    assert record["versions"]["lqcloud"] == sys.modules["lqcloud"].__version__
    restored = s.task(t.local_id)
    assert restored.prepared == t.prepared
    assert restored.details()["tensorcircuit"] == meta


@requires_sdk
@pytest.mark.parametrize("types, n_x21", [(["measure"], 0), (["2state"], 2)])
def test_sent_payload_is_the_request_body(fake_lqcloud, types, n_x21):
    # the readout pulses are decided by the SDK, and only read from what is sent
    fake_lqcloud.config["supported_measurement_types"] = types
    c = tc.Circuit(2)
    c.x(0)
    t = apis.submit_task(device=DEVICE, circuit=c, store=False)
    (sent,) = fake_lqcloud.submissions()
    assert "Content-Encoding" not in sent["headers"]
    assert t.prepared["payload"] == sent["body"]
    names = [
        op["name"] for op in t.prepared["payload"]["command"]["circuit"]["instructions"]
    ]
    assert names.count("x21") == n_x21


@requires_sdk
def test_mid_circuit_payload_and_run_options(fake_lqcloud):
    dd = {"enabled": True, "m": 4, "dd_type": "CPMG", "qubits": [5]}
    t = apis.submit_task(
        device=DEVICE,
        circuit=mid_circuit(),
        compiling=False,
        compiled_options={"initial_layout": [3, 5, 4]},
        run_options={"dynamic_decoupling": dd},
        store=False,
    )
    (sent,) = fake_lqcloud.submissions()
    section = sent["body"]["command"]["circuit"]
    comp = t.prepared["compilation"]
    assert comp["run_options"] == {"dynamic_decoupling": dd}
    # the physical qubit 5, sent as its position 2 in the initial_layout [3, 4, 5],
    # which the SDK maps back in the request, and would reject 5 as out of range
    assert section["initial_layout"] == [3, 4, 5]
    assert section["dynamic_decoupling"]["qubits"] == [5]
    assert section["dynamic_decoupling"]["m"] == 4
    # the restores by TensorCircuit, then the readout pulses by the SDK
    names = [op["name"] for op in section["instructions"]]
    assert names.count("x21") == 2 + 5 and names.count("measure") == 5
    info = {"num_windows": 1, "insertions": []}
    fake_lqcloud.info = {
        "status": "completed",
        "result": {
            "counts": {"00000": 10},
            "shots": 10,
            "dynamic_decoupling_info": info,
        },
    }
    result = t.details()["lqcloud"]["result"]
    assert result["dynamic_decoupling_info"] == info and "counts" not in result

    # the rerun sends the run options again
    apis.resubmit_task(t)
    assert fake_lqcloud.submissions()[-1]["body"]["command"]["circuit"] == section
    device = Device.from_name(DEVICE)
    with pytest.raises(ValueError, match="share the run options"):
        lqcloud.submit_task(
            device, "tc-key", compilation=[comp, dict(comp, run_options={})]
        )
    # the x21 restores are compiled for the two-state readout
    fake_lqcloud.config["supported_measurement_types"] = ["measure"]
    with pytest.raises(ValueError, match="current readout"):
        apis.resubmit_task(t)
    assert len(fake_lqcloud.submissions()) == 2


@requires_sdk
def test_provider_ignores_sdk_account_and_store(fake_lqcloud):
    p = lqcloud._provider("tc-key")
    assert p.api_key == "tc-key" and p.url == lqcloud_base_url and p.store is None
    with pytest.raises(ValueError, match="set_token"):
        lqcloud._provider(None)


@requires_sdk
@pytest.mark.parametrize(
    "info, state, err",
    [
        ({"status": "cancelled"}, "failed", "cancelled"),
        ({"status": "failed", "error": "hardware fault"}, "failed", "hardware fault"),
    ],
)
def test_failed_states(fake_lqcloud, info, state, err):
    fake_lqcloud.info = info
    t = apis.get_task("task_0001", device=DEVICE)
    d = apis.get_task_details(t)
    assert d["state"] == state and d["err"] == err
    with pytest.raises(TaskFailed, match=err):
        t.results(blocked=True)


@requires_sdk
def test_unknown_state_and_offloaded_result_raise(fake_lqcloud):
    t = apis.get_task("task_0001", device=DEVICE)
    fake_lqcloud.info = {"status": "paused"}
    with pytest.raises(ValueError, match="Unknown LQCloud task status"):
        apis.get_task_details(t)
    fake_lqcloud.info = {"status": "completed", "result": "https://oss/result.json"}
    with pytest.raises(ValueError, match="Unsupported LQCloud result"):
        apis.get_task_details(t)


@requires_sdk
def test_memory_results(fake_lqcloud):
    memory = ["0" * 12, "1" * 12, "1" * 12]
    fake_lqcloud.info = {
        "status": "completed",
        "result": {"memory": memory, "result_format": "memory", "shots": 3},
    }
    t = apis.get_task("task_0001", device=DEVICE)
    d = t.details()
    assert d["results"] == {"0" * 12: 1, "1" * 12: 2}
    assert d["memory"] == memory


def _http_error(status, error_type="validation_error"):
    return response({"detail": {"error": "nope", "error_type": error_type}}, status)


def _connection_error():
    import requests

    return requests.ConnectionError("connection reset")


@requires_sdk
@pytest.mark.parametrize(
    "answer, error",
    [
        (lambda: _http_error(400), SubmissionRejected),
        (lambda: _http_error(401), SubmissionRejected),
        (lambda: _http_error(409, "idempotency_in_flight"), SubmissionUnknown),
        (lambda: _http_error(502), SubmissionUnknown),
        (_connection_error, SubmissionUnknown),
        (lambda: response(b"<html></html>"), SubmissionUnknown),
        (lambda: response({"status": "queued"}), SubmissionUnknown),
    ],
)
def test_submission_errors_are_classified(fake_lqcloud, tmp_path, answer, error):
    fake_lqcloud.submission = answer()
    c = tc.Circuit(2)
    c.x(0)
    s = store.TaskStore(tmp_path / "tasks")
    with pytest.raises(error):
        apis.submit_task(device=DEVICE, circuit=c, store=s)
    records = s.find()
    if error is SubmissionUnknown:
        # the task may be accepted: kept with the exact program sent
        assert [r["submission_status"] for r in records] == ["unknown"]
        assert s.get(records[0]["local_id"])["prepared"]["compilation"][
            "physical_qubits"
        ]
    else:
        assert records == []


@requires_sdk
def test_local_failures_are_not_submitted(fake_lqcloud, tmp_path):
    c = tc.Circuit(2)
    c.x(0)
    s = store.TaskStore(tmp_path / "tasks")
    with pytest.raises(ValueError, match="shots"):
        apis.submit_task(device=DEVICE, circuit=c, shots=0, store=s)
    with pytest.raises(ValueError, match="share the shots"):
        apis.submit_task(device=DEVICE, circuit=[c, c], shots=[100, 200], store=s)
    # the SDK would ignore a misspelled option silently
    with pytest.raises(ValueError, match="Unsupported LQCloud run options"):
        apis.submit_task(
            device=DEVICE, circuit=c, run_options={"dynamic_decouplng": {}}, store=s
        )
    assert fake_lqcloud.submissions() == [] and s.find() == []


@requires_sdk
def test_unreadable_payload_keeps_the_accepted_task(
    fake_lqcloud, tmp_path, monkeypatch
):
    def unreadable(request):
        raise ValueError("unknown body")

    monkeypatch.setattr(lqcloud, "_payload", unreadable)
    c = tc.Circuit(2)
    c.x(0)
    s = store.TaskStore(tmp_path / "tasks")
    t = apis.submit_task(device=DEVICE, circuit=c, store=s)
    assert t.prepared["payload"] is None and "compilation" in t.prepared
    assert s.get(t.local_id)["remote_id"] == t.id_


@requires_sdk
def test_batch_is_one_task_with_a_task_per_circuit(fake_lqcloud, tmp_path):
    c0 = tc.Circuit(3)
    c0.x(2)
    c0.measure_instruction(2)
    c0.measure_instruction(0)
    c1 = tc.Circuit(2)
    c1.h(0)
    c1.cx(0, 1)
    s = store.TaskStore(tmp_path / "tasks")
    ts = apis.submit_task(
        device=DEVICE,
        circuit=[c0, c1],
        shots=[100, 100],
        compiled_options=[{"initial_layout": [0, 1, 2]}, {"initial_layout": [4, 5]}],
        store=s,
    )
    (sent,) = fake_lqcloud.submissions()
    command = sent["body"]["command"]
    assert command["action"] == "run_circuit_batch" and command["shots"] == 100
    assert [t.id_ for t in ts] == ["task_0001#0", "task_0001#1"]
    comps = [t.prepared["compilation"] for t in ts]
    assert comps[0]["physical_qubits"] == [0, 2] and comps[1]["physical_qubits"] == [
        4,
        5,
    ]
    for t, section in zip(ts, command["circuits"]):
        # each circuit runs on its own physical qubits within the shared layout
        assert section["initial_layout"] == [0, 2, 4, 5]
        assert t.prepared["payload"]["command"]["circuits"] == [section]
        measures = [op for op in section["instructions"] if op["name"] == "measure"]
        assert {op["clbits"][0]: op["qubits"][0] for op in measures} == t.prepared[
            "positional_logical_mapping"
        ]
        assert s.get(t.local_id)["remote_id"] == t.id_

    fake_lqcloud.info = {
        "status": "completed",
        "cost": 4.0,
        "result": {
            "result_format": "batch",
            "shots": 100,
            "results": [
                {"counts": {"10": 100}, "shots": 100, "success": True},
                {"success": False, "error": "readout fault"},
            ],
        },
    }
    assert ts[0].results() == {"10": 100}
    assert ts[0].details()["lqcloud"]["cost"] == 4.0
    with pytest.raises(TaskFailed, match="readout fault"):
        ts[1].results(blocked=True)
    with pytest.raises(ValueError, match="is a batch"):
        apis.get_task_details(apis.get_task("task_0001", device=DEVICE))
    # a circuit of a batch reruns alone
    fake_lqcloud.submission = response({"task_id": "task_0002", "status": "queued"})
    t2 = apis.resubmit_task(ts[0])
    assert (
        t2.id_ == "task_0002"
        and "circuit" in fake_lqcloud.submissions()[-1]["body"]["command"]
    )


@requires_sdk
def test_rejected_batch_does_not_fall_back(fake_lqcloud, tmp_path):
    fake_lqcloud.submission = response(
        {"detail": "batch is not allowed", "error_type": "validation_error"}, 400
    )
    c = tc.Circuit(2)
    c.x(0)
    s = store.TaskStore(tmp_path / "tasks")
    with pytest.raises(SubmissionRejected, match="batch is not allowed"):
        apis.submit_task(device=DEVICE, circuit=[c, c], store=s)
    assert len(fake_lqcloud.submissions()) == 1 and s.find() == []


@requires_sdk
@pytest.mark.parametrize("readout_cals, nposts", [("api", 1), ("system", 2)])
def test_batch_expectation_ps_mitigates_the_measured_qubits(
    fake_lqcloud, monkeypatch, with_readout_errors, readout_cals, nposts
):
    errors = {q: (0.01 * (q + 1), 0.03 * (q + 1)) for q in range(6)}
    for m in fake_lqcloud.config["properties"]["qubit_metrics"]:
        m["measure_f0"], m["measure_f1"] = [1 - e for e in errors[m["id"]]]
    commands = {}

    def submit(body):
        task_id = "task_%04d" % len(commands)
        commands[task_id] = body["command"]
        return response({"task_id": task_id, "status": "queued"})

    def info(task_id):
        command = commands[task_id]
        results = []
        for section in command["circuits"]:
            measures = {
                op["clbits"][0]: op["qubits"][0]
                for op in section["instructions"]
                if op["name"] == "measure"
            }
            qubits = [measures[i] for i in range(len(measures))]
            program = dict(section, physical_qubits=section["initial_layout"])
            probs = with_readout_errors(
                simulate(dict(program, n_clbits=len(qubits))), qubits, errors
            )
            counts = {
                format(i, "0%db" % len(qubits)): int(round(p * command["shots"]))
                for i, p in enumerate(probs)
            }
            results.append({"counts": counts, "success": True})
        return {
            "status": "completed",
            "result": {"result_format": "batch", "results": results},
        }

    fake_lqcloud.submission = submit
    fake_lqcloud.info = info
    compiling = []
    compile_circuit = lqcloud.compile_circuit

    def record(c, properties, shots, flag, *args):
        compiling.append(flag)
        return compile_circuit(c, properties, shots, flag, *args)

    monkeypatch.setattr(lqcloud, "compile_circuit", record)
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    r = wrapper.batch_expectation_ps(
        c,
        [[3, 3], [1, 1], [2, 2]],
        device=DEVICE,
        shots=10000,
        readout_cals=readout_cals,
    )
    np.testing.assert_allclose(r, [1, 1, -1], atol=1e-3)
    # one batch each, on the qubits of the wrapper, the options of tencent ignored
    assert len(fake_lqcloud.submissions()) == nposts
    assert set(compiling) == {False}


@requires_sdk
def test_batch_expectation_ps_without_errors_compiles_on_the_topology(fake_lqcloud):
    for m in fake_lqcloud.config["properties"]["qubit_metrics"]:
        del m["xeb_fidelity"]
    captured = []

    def run(cs, shots):
        captured.extend(cs)
        return [{"000": shots} for _ in cs]

    c = tc.Circuit(3)
    c.h(0)
    c.cx(0, 2)
    wrapper.batch_expectation_ps(
        c, [[3, 0, 3]], device=DEVICE, with_rem=False, batch_submit_func=run
    )
    links = fake_properties()["links"]
    pairs = [
        tuple(d["index"])
        for cc in captured
        for d in cc.to_qir()
        if len(d["index"]) == 2
    ]
    assert pairs and all(p in links for p in pairs)


@requires_sdk
def test_missing_readout_error_raises_before_submission(fake_lqcloud):
    for m in fake_lqcloud.config["properties"]["qubit_metrics"][:2]:
        del m["measure_f1"]
    c = tc.Circuit(2)
    c.h(0)
    with pytest.raises(ValueError, match="No readout error"):
        # on the qubits 0 and 1 as given, which the default compiling avoids
        wrapper.batch_expectation_ps(
            c, [[3, 0]], device=DEVICE, compile_func=lambda circuit: (circuit, {})
        )
    assert fake_lqcloud.submissions() == []


@requires_sdk
def test_resubmit_sends_the_stored_compilation(fake_lqcloud, tmp_path):
    c = triangle_circuit()
    s = store.TaskStore(tmp_path / "tasks")
    t = apis.submit_task(device=DEVICE, circuit=c, store=s)
    restored = s.task(t.local_id)
    restored.prepared = None
    fake_lqcloud.submission = response({"task_id": "task_0002", "status": "queued"})
    t2 = apis.resubmit_task(restored)
    first, second = fake_lqcloud.submissions()
    assert first["body"]["command"] == second["body"]["command"]
    assert t2.id_ == "task_0002" and t2.prepared == t.prepared
    # the given task is not changed by the rerun
    assert restored.prepared is None
    assert s.get(t2.local_id)["parent_id"] == t.local_id


@requires_sdk
@pytest.mark.parametrize("method", ["qiskit", "default"])
def test_symbol_circuit_compiled_once_runs_on_its_qubits(fake_lqcloud, method):
    from tensorcircuit.compiler.qiskit_compiler import (
        properties_to_target,
        qiskit_compile,
    )

    ts = sympy.symbols("t0:4", real=True)
    sc = tc.SymbolCircuit(3)
    sc.rx(0, theta=ts[0])
    sc.cx(0, 1)
    sc.ry(1, theta=2 * ts[1] + 0.3)
    sc.cx(1, 2)
    sc.rz(2, theta=ts[2])
    sc.rx(2, theta=ts[3] - ts[0])
    # a line of interactions, placed without routing, so that the noise aware
    # layout does not depend on the seed
    sc.cx(2, 1)
    for m in fake_lqcloud.config["properties"]["coupler_metrics"]:
        m["cz_fidelity"] = 0.999 if m["qubits"] in [[3, 4], [4, 5]] else 0.95
    p = apis.list_properties(device=DEVICE)
    if method == "qiskit":
        options = {
            "target": properties_to_target(p),
            "optimization_level": 2,
            "seed_transpiler": 42,
        }
        csc, info = qiskit_compile(sc, compiled_options=options)
    else:
        # gates other than the native ones, translated by the provider
        csc, info = tc.compiler.DefaultCompiler(properties=p)(sc)
    layout = sorted(info["logical_physical_mapping"].values())
    assert layout == [3, 4, 5]
    # the device prefers other qubits later, while the compiled circuit stays
    for m in fake_lqcloud.config["properties"]["coupler_metrics"]:
        m["cz_fidelity"] = 0.999 if m["qubits"] in [[0, 1], [1, 2]] else 0.95
    for seed in range(2):
        values = dict(zip(ts, np.random.default_rng(seed).normal(size=4)))
        t = apis.submit_task(
            device=DEVICE, circuit=csc.to_circuit(values), compiling=False, store=False
        )
        comp = t.prepared["compilation"]
        assert comp["physical_qubits"] == layout
        np.testing.assert_allclose(
            simulate(comp), ideal(sc.to_circuit(values)), atol=1e-5
        )

    # the same for batch_expectation_ps, given the compiled circuit and its info
    def run(cs, shots):
        rs = []
        for c in cs:
            t = apis.submit_task(device=DEVICE, circuit=c, compiling=False, store=False)
            comp = t.prepared["compilation"]
            assert comp["physical_qubits"] == layout
            probs = simulate(comp)
            n = comp["n_clbits"]
            rs.append({format(i, "0%db" % n): q * shots for i, q in enumerate(probs)})
        return rs

    pss = [[3, 3, 0], [0, 1, 1], [2, 0, 3]]
    r = wrapper.batch_expectation_ps(
        csc,
        pss,
        device=DEVICE,
        with_rem=False,
        batch_submit_func=run,
        params=values,
        info=info,
    )
    np.testing.assert_allclose(
        r, wrapper.batch_expectation_ps(sc, pss, params=values), atol=1e-4
    )


@requires_sdk
def test_list_properties_and_cancel(fake_lqcloud):
    # a coupler without its qubits is skipped
    fake_lqcloud.config["properties"]["coupler_metrics"].append(
        {"id": 7, "cz_fidelity": 0.99}
    )
    del fake_lqcloud.config["properties"]["qubit_metrics"][2]["measure_f1"]
    d = Device.from_name(DEVICE)
    p = apis.list_properties(device=d)
    assert p["native_gates"] == ["cz", "rz", "sx", "x"]
    np.testing.assert_allclose(p["bits"][2]["ReadoutF0Err"], 0.02)
    assert p["bits"][2]["ReadoutF1Err"] is None
    np.testing.assert_allclose(p["links"][(4, 1)]["CZErrRate"], 0.01)
    assert (0, 4) not in p["links"]
    assert apis.list_devices(provider="lqcloud") == [d]
    assert apis.remove_task(apis.get_task("task_0001", device=d))
    assert fake_lqcloud.requests[-1]["url"].endswith("/api/v1/jobs/task_0001/cancel")


def test_missing_sdk_reports_installation(monkeypatch):
    monkeypatch.setitem(sys.modules, "lqcloud", None)
    with pytest.raises(ImportError, match="pip install"):
        lqcloud._lqcloud()
    assert "lqcloud" in apis.avail_providers


@requires_sdk
def test_store_records_the_sdk_version_before_its_import(monkeypatch):
    version = importlib.import_module("lqcloud").__version__
    monkeypatch.delitem(sys.modules, "lqcloud")
    assert store._versions(["lqcloud"])["lqcloud"] == version


def _hardware_run(c, device="QZ03", **kws):
    t = apis.submit_task(
        provider="lqcloud", device=device, circuit=c, shots=1000, **kws
    )
    r = t.results()
    d = t.details()
    print(t.id_, d["lqcloud"]["cost"], r)
    return t, r, d


def _tvd(counts, probs):
    n = sum(counts.values())
    keys = [format(i, "0%db" % int(np.log2(len(probs)))) for i in range(len(probs))]
    return 0.5 * sum(abs(counts.get(k, 0) / n - p) for k, p in zip(keys, probs))


@requires_cloud
def test_cloud_list_properties():
    p = apis.list_properties(device="lqcloud::QZ03")
    assert p["links"] and p["bits"] and p["calibration_time"]


@requires_hardware
@pytest.mark.parametrize("device", ["QZ03", "MQ02"])
def test_hardware_partial_measurement_bit_order(device):
    c = tc.Circuit(4)
    c.x(2)
    c.measure_instruction(2)
    c.measure_instruction(0)
    c.measure_instruction(1)
    _, r, d = _hardware_run(c, device)
    assert max(r, key=r.get) == "100" and r["100"] > 800
    assert len(d["tensorcircuit"]["positional_logical_mapping"]) == 3


@requires_hardware
def test_hardware_routed_distribution():
    c = triangle_circuit()
    t, r, _ = _hardware_run(c)
    assert t.prepared["compilation"]["compiler"]["version"]
    # the leading outcomes, which a wrong bit order or mapping would change
    assert set(sorted(r, key=r.get)[-2:]) == {"000", "111"}
    assert _tvd(r, ideal(c)) < 0.2


@requires_hardware
def test_hardware_initial_layout_and_inputs():
    # the generic state preparation is deep even for basis states, keep it small
    c = tc.Circuit(2, inputs=np.array([0.0, 0.0, 1.0, 0.0]))
    c.cx(0, 1)
    t, r, _ = _hardware_run(c, compiled_options={"initial_layout": [24, 30]})
    assert t.prepared["compilation"]["physical_qubits"] == [24, 30]
    assert t.prepared["compilation"]["dropped_resets"]
    assert max(r, key=r.get) == "11" and r["11"] > 750


@requires_hardware
def test_hardware_memory_format():
    c = tc.Circuit(12)
    for q in range(0, 12, 2):
        c.x(q)
    _, r, d = _hardware_run(c)
    assert len(d["memory"]) == 1000
    assert max(r, key=r.get) == "10" * 6


@requires_hardware
def test_hardware_batch_layouts_and_bit_order():
    # each circuit on its own qubits, overlapping on qubit 30
    c0 = tc.Circuit(3)
    c0.x(2)
    c0.measure_instruction(2)
    c0.measure_instruction(0)
    c0.measure_instruction(1)
    c1 = tc.Circuit(2)
    c1.x(0)
    c1.cx(0, 1)
    c2 = tc.Circuit(1)
    c2.x(0)
    ts = apis.submit_task(
        provider="lqcloud",
        device="QZ03",
        circuit=[c0, c1, c2],
        shots=1000,
        compiled_options=[
            {"initial_layout": [30, 17, 1]},
            {"initial_layout": [24, 30]},
            {"initial_layout": [5]},
        ],
    )
    rs = [t.results() for t in ts]
    print([t.id_ for t in ts], ts[0].details()["lqcloud"]["cost"], rs)
    assert [max(r, key=r.get) for r in rs] == ["100", "11", "1"]


@requires_hardware
def test_hardware_batch_expectation_ps():
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    r = wrapper.batch_expectation_ps(
        c, [[3, 3], [1, 1], [2, 2]], device="lqcloud::QZ03", shots=1000
    )
    print(r)
    np.testing.assert_allclose(r, [1, 1, -1], atol=0.2)


@requires_hardware
def test_hardware_mid_circuit_measurement():
    cs = [tc.Circuit(1), tc.Circuit(1)]
    for c in cs:
        c.x(0)
        c.measure_instruction(0)
    cs[0].x(0)
    cs[1].reset_instruction(0)
    for c in cs:
        c.measure_instruction(0)
    ts = apis.submit_task(provider="lqcloud", device="MQ02", circuit=cs, shots=1000)
    rs = [t.results() for t in ts]
    print([t.id_ for t in ts], ts[0].details()["lqcloud"]["cost"], rs)
    # the x21 restore before the x leaves 30-40% of the measured |1> in |2>,
    # and the reset leaves 1-6%, depending on the qubit
    assert rs[0]["10"] > 500 and rs[1]["10"] > 850


@requires_hardware
def test_hardware_noise_aware_ghz():
    c = tc.Circuit(5)
    c.h(0)
    for i in range(4):
        c.cx(i, i + 1)
    t, r, _ = _hardware_run(c)
    assert t.prepared["compilation"]["compiler"]["noise_aware"]
    assert set(sorted(r, key=r.get)[-2:]) == {"00000", "11111"}
    assert (r.get("00000", 0) + r.get("11111", 0)) / sum(r.values()) > 0.7
