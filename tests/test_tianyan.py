"""Tests for the TianYan cloud provider.

Offline tests run by default. Set ``TC_CLOUD_TEST=1`` and ``TC_TOKEN_TIANYAN``
to run online simulator tests. Real-device tests additionally require
``TC_CLOUD_HARDWARE_TEST=1``.
"""

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

import tensorcircuit as tc
from tensorcircuit.cloud import apis, tianyan
from tensorcircuit.cloud.abstraction import Device, Task

# token is picked up automatically by the cloud env token system
LOGIN_KEY = os.getenv("TC_TOKEN_TIANYAN")
HAS_CQLIB = tianyan.TianYanPlatform is not None
CLOUD_TESTS_ENABLED = (
    HAS_CQLIB and bool(LOGIN_KEY) and os.getenv("TC_CLOUD_TEST") == "1"
)
HARDWARE_TESTS_ENABLED = (
    CLOUD_TESTS_ENABLED and os.getenv("TC_CLOUD_HARDWARE_TEST") == "1"
)
FAKE_QUANTUM_LANGUAGE = SimpleNamespace(QCIS="QCIS")

SIMULATOR = "tianyan_sw"  # full-amplitude simulator, no topology constraints
REAL_DEVICE = "tianyan176"  # superconducting hardware with topology constraints

# keep shots low on real hardware to save quota
TEST_SHOTS_SIM = 1000
TEST_SHOTS_REAL = 100

requires_cqlib = pytest.mark.skipif(not HAS_CQLIB, reason="cqlib is not installed")
requires_cloud = pytest.mark.skipif(
    not CLOUD_TESTS_ENABLED,
    reason="set TC_TOKEN_TIANYAN and TC_CLOUD_TEST=1 to run cloud tests",
)
requires_hardware = pytest.mark.skipif(
    not HARDWARE_TESTS_ENABLED,
    reason="set TC_CLOUD_HARDWARE_TEST=1 to run hardware tests",
)


class _FakePlatform:
    QUERY_EXP_PATH = "/query"
    MACHINE_LIST_PATH = "/machines"

    def __init__(
        self,
        response: Dict[str, Any],
        batch_query_ids: Optional[List[str]] = None,
        single_query_ids: Optional[List[str]] = None,
    ):
        self.response = response
        self.calls: List[Dict[str, Any]] = []
        self.submitted = False
        self.batch_query_ids = batch_query_ids or []
        self.single_query_ids = single_query_ids or ["single-task"]
        self.last_submission: Dict[str, Any] = {}

    def _send_request(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return self.response

    def submit_job(self, **kwargs: Any) -> List[str]:
        self.submitted = True
        self.last_submission = kwargs
        return self.single_query_ids

    def submit_experiment(self, **kwargs: Any) -> List[str]:
        self.submitted = True
        self.last_submission = kwargs
        return self.batch_query_ids


def assert_is_counts(counts: Any) -> None:
    assert isinstance(counts, dict) and len(counts) > 0
    for bitstring, cnt in counts.items():
        assert isinstance(bitstring, str)
        assert isinstance(cnt, int)
        assert cnt >= 0


def assert_bell_state_like(counts: Dict[str, int], tol: float = 0.3) -> None:
    total = sum(counts.values())
    ratio = (counts.get("00", 0) + counts.get("11", 0)) / total
    assert ratio > 1 - tol, f"Bell 00+11 ratio too low: {ratio:.2f}, counts={counts}"


@requires_cqlib
def test_parameterized_qasm_uses_cqlib_converter() -> None:
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
rz(pi/2) q[0];
measure q[0] -> c[0];
"""

    source = tianyan._qasm_to_qcis(qasm)

    assert "RZ Q0 1.570796" in source
    assert source.splitlines()[-1] == "M Q0"


def test_incompatible_cqlib_fails_fast() -> None:
    class IncompatiblePlatform:
        pass

    with patch.object(tianyan, "TianYanPlatform", IncompatiblePlatform):
        with pytest.raises(ImportError, match="_send_request"):
            tianyan._assert_cqlib()


def test_topology_validation_accepts_connected_circuit() -> None:
    graph = {0: {1}, 1: {0, 2}, 2: {1}}
    circuit = tc.Circuit(2)
    circuit.cx(0, 1)

    tianyan._validate_circuit_topology(circuit, graph, set(graph))


def test_topology_validation_rejects_unconnected_pair() -> None:
    graph = {0: {2}, 1: set(), 2: {0}}
    circuit = tc.Circuit(2)
    circuit.cx(0, 1)

    with pytest.raises(ValueError, match="unconnected physical qubits"):
        tianyan._validate_circuit_topology(circuit, graph, set(graph))


def test_topology_validation_rejects_unavailable_qubit() -> None:
    graph = {0: {1}, 1: {0}}
    circuit = tc.Circuit(2)
    circuit.cx(0, 1)

    with pytest.raises(ValueError, match="unavailable on this device"):
        tianyan._validate_circuit_topology(circuit, graph, {0})


@requires_cqlib
def test_partial_measurement_preserves_qubits_and_order() -> None:
    circuit = tc.Circuit(3)
    circuit.h(0)
    circuit.x(1)
    circuit.z(2)
    circuit.measure_instruction(2, 0)

    source = tianyan._tc_qir_to_qcis(circuit)
    measurements = [line for line in source.splitlines() if line.startswith("M ")]

    assert measurements == ["M Q2", "M Q0"]


@requires_cqlib
def test_unsupported_qir_gate_fails_without_fallback() -> None:
    circuit = tc.Circuit(1)
    circuit.i(0)

    with pytest.raises(ValueError, match="Unsupported gate"):
        tianyan._tc_qir_to_qcis(circuit)


def test_get_task_details_queries_once_and_returns_pending() -> None:
    platform = _FakePlatform({"data": {"experimentResultModelList": []}})
    device = Device.from_name("tianyan::offline-status")
    task = Task("pending-task", device=device)
    task.add_details(source="H Q0")

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
    ):
        details = tianyan.get_task_details(task, device, "token", False)

    assert details["state"] == "pending"
    assert details["source"] == "H Q0"
    assert len(platform.calls) == 1


def test_get_task_details_maps_failure_state() -> None:
    result = {
        "experimentTaskId": "failed-task",
        "status": "FAILED",
        "errorMessage": "backend failure",
        "resultStatus": [],
    }
    platform = _FakePlatform({"data": {"experimentResultModelList": [result]}})
    device = Device.from_name("tianyan::offline-failure")
    task = Task("failed-task", device=device)

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
    ):
        details = tianyan.get_task_details(task, device, "token", False)

    assert details["state"] == "failed"
    assert details["err"] == "backend failure"


def test_result_without_explicit_state_is_completed() -> None:
    result = {
        "experimentTaskId": "completed-task",
        "resultStatus": [[0], [0], [1]],
    }
    details = tianyan._parse_result(
        result, Device.from_name("tianyan::offline-completed")
    )

    assert details["state"] == "completed"
    assert details["results"] == {"0": 1, "1": 1}


def test_topology_failure_stops_before_submission() -> None:
    platform = _FakePlatform({})
    device = Device.from_name("tianyan::hardware")
    circuit = tc.Circuit(2)
    circuit.cx(0, 1)

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
        patch.object(
            tianyan,
            "_get_device_topology",
            side_effect=ValueError("topology unavailable"),
        ),
    ):
        with pytest.raises(ValueError, match="topology unavailable"):
            tianyan.submit_task(device, "token", circuit=circuit)

    assert not platform.submitted


def test_incompatible_circuit_stops_before_submission() -> None:
    platform = _FakePlatform({})
    device = Device.from_name("tianyan::hardware-incompatible")
    circuit = tc.Circuit(3)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    graph = {0: {1}, 1: {0, 2}, 2: {1}}

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
        patch.object(tianyan, "_get_device_topology", return_value=(graph, set(graph))),
    ):
        with pytest.raises(ValueError, match="unconnected physical qubits"):
            tianyan.submit_task(device, "token", circuit=circuit)

    assert not platform.submitted


def test_batch_submission_on_hardware_validates_each_circuit() -> None:
    platform = _FakePlatform({}, batch_query_ids=["task-1", "task-2"])
    device = Device.from_name("tianyan::hardware-batch")
    circuit_1 = tc.Circuit(2)
    circuit_1.cx(0, 1)
    circuit_2 = tc.Circuit(3)
    circuit_2.cx(1, 2)
    graph = {0: {1}, 1: {0, 2}, 2: {1}}

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
        patch.object(tianyan, "QuantumLanguage", FAKE_QUANTUM_LANGUAGE),
        patch.object(tianyan, "_circuit_to_qcis", return_value="M Q0"),
        patch.object(tianyan, "_get_device_topology", return_value=(graph, set(graph))),
    ):
        tasks = tianyan.submit_task(
            device, "token", circuit=[circuit_1, circuit_2], shots=[100, 100]
        )

    assert len(tasks) == 2
    assert platform.last_submission["num_shots"] == 100


def test_batch_submission_rejects_different_shots() -> None:
    platform = _FakePlatform({}, batch_query_ids=["task-1", "task-2"])
    device = Device.from_name("tianyan::tianyan_sw")

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
        patch.object(tianyan, "QuantumLanguage", FAKE_QUANTUM_LANGUAGE),
    ):
        with pytest.raises(ValueError, match="requires the same shots"):
            tianyan.submit_task(
                device,
                "token",
                source=["H Q0", "X Q0"],
                shots=[100, 200],
            )

    assert not platform.submitted


def test_batch_submission_rejects_shots_length_mismatch() -> None:
    platform = _FakePlatform({}, batch_query_ids=["task-1", "task-2"])
    device = Device.from_name("tianyan::tianyan_sw")

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
        patch.object(tianyan, "QuantumLanguage", FAKE_QUANTUM_LANGUAGE),
    ):
        with pytest.raises(ValueError, match="Expected 2 shots values"):
            tianyan.submit_task(
                device,
                "token",
                source=["H Q0", "X Q0"],
                shots=[100],
            )

    assert not platform.submitted


def test_batch_submission_validates_returned_task_ids() -> None:
    platform = _FakePlatform({}, batch_query_ids=["task-1"])
    device = Device.from_name("tianyan::tianyan_sw")

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
        patch.object(tianyan, "QuantumLanguage", FAKE_QUANTUM_LANGUAGE),
    ):
        with pytest.raises(ValueError, match="1 task IDs for 2"):
            tianyan.submit_task(device, "token", source=["H Q0", "X Q0"], shots=100)


@requires_cqlib
def test_openqasm_source_is_converted_before_submission() -> None:
    platform = _FakePlatform({}, single_query_ids=["qasm-task"])
    device = Device.from_name("tianyan::tianyan_sw")
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
"""

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
    ):
        tianyan.submit_task(device, "token", source=qasm, lang="OPENQASM", shots=100)

    submitted = platform.last_submission["circuit"]
    assert "Y2M Q0" in submitted
    assert "RZ Q0" in submitted
    assert submitted.splitlines()[-1] == "M Q0"


def test_unsupported_source_language_fails_before_platform_creation() -> None:
    device = Device.from_name("tianyan::tianyan_sw")

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform") as get_platform,
    ):
        with pytest.raises(ValueError, match="Unsupported TianYan source"):
            tianyan.submit_task(device, "token", source="H Q0", lang="UNKNOWN")

    get_platform.assert_not_called()


@requires_cqlib
def test_qiskit_circuit_uses_qasm2_dumps() -> None:
    qasm = "OPENQASM 2.0;\nqreg q[1];\nh q[0];"

    class FakeQuantumCircuit:
        pass

    qiskit_module = ModuleType("qiskit")
    qiskit_module.qasm2 = SimpleNamespace(dumps=lambda circuit: qasm)  # type: ignore
    circuit_module = ModuleType("qiskit.circuit")
    circuit_module.QuantumCircuit = FakeQuantumCircuit  # type: ignore

    with patch.dict(
        sys.modules,
        {"qiskit": qiskit_module, "qiskit.circuit": circuit_module},
    ):
        source = tianyan._circuit_to_qcis(FakeQuantumCircuit())

    assert "Y2M Q0" in source
    assert "RZ Q0" in source


def test_list_tasks_reports_unsupported_operation() -> None:
    device = Device.from_name("tianyan::tianyan_sw")
    with pytest.raises(NotImplementedError, match="no listing API"):
        tianyan.list_tasks(device, "token")


def test_remove_task_reports_unsupported_operation() -> None:
    device = Device.from_name("tianyan::tianyan_sw")
    task = Task("cancel-task", device=device)
    with pytest.raises(NotImplementedError, match="no cancellation endpoint"):
        tianyan.remove_task(task, "token")


def test_list_devices_parses_machine_names() -> None:
    platform = _FakePlatform(
        {
            "data": [
                {"id": 1, "isToll": 1, "status": 0, "code": "tianyan_sw"},
                {"id": 2, "isToll": 2, "status": 0, "code": "tianyan176"},
            ]
        }
    )

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
    ):
        devices = tianyan.list_devices("token")

    assert [d.name for d in devices] == ["tianyan_sw", "tianyan176"]


def test_list_devices_rejects_missing_machine_name() -> None:
    platform = _FakePlatform({"data": [{"id": 1, "status": 0}]})

    with (
        patch.object(tianyan, "_assert_cqlib"),
        patch.object(tianyan, "_get_platform", return_value=platform),
    ):
        with pytest.raises(ValueError, match="missing machine name"):
            tianyan.list_devices("token")


@requires_cloud
def test_provider_is_registered() -> None:
    provider = apis.get_provider("tianyan")
    assert str(provider) == "tianyan"


@requires_cloud
def test_cloud_list_devices() -> None:
    devices = apis.list_devices(provider="tianyan")
    names = [str(d) for d in devices]
    assert f"tianyan::{SIMULATOR}" in names
    assert f"tianyan::{REAL_DEVICE}" in names


@requires_cloud
def test_cloud_get_device() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")
    assert isinstance(device, Device)
    assert device.name == SIMULATOR
    assert device.provider.name == "tianyan"


@requires_cloud
def test_cloud_list_properties_simulator() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")
    props = apis.list_properties(device)

    assert "native_gates" in props
    assert isinstance(props["links"], dict)
    assert isinstance(props["bits"], dict)
    if props["links"]:
        (q1, _), info = next(iter(props["links"].items()))
        assert isinstance(q1, int)
        assert isinstance(info, dict)
    if props["bits"]:
        qid, info = next(iter(props["bits"].items()))
        assert isinstance(qid, int)
        assert isinstance(info, dict)


@requires_cloud
def test_cloud_native_gates() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")
    gates = device.native_gates()
    assert "h" in gates
    assert "cx" in gates


@requires_hardware
def test_cloud_list_properties_real_device() -> None:
    device = apis.get_device(f"tianyan::{REAL_DEVICE}")
    props = apis.list_properties(device)

    assert len(props["links"]) > 0
    assert len(props["bits"]) > 0
    sample_bit = next(iter(props["bits"].values()))
    assert any(k in sample_bit for k in ["T1", "T2", "ReadoutError", "Freqency"])


@requires_hardware
def test_cloud_topology_real_device() -> None:
    device = apis.get_device(f"tianyan::{REAL_DEVICE}")
    topo = device.topology()
    assert len(topo) > 0
    for edge in topo[:3]:
        assert len(edge) == 2
        assert isinstance(edge[0], int)
        assert isinstance(edge[1], int)


@requires_cloud
def test_cloud_submit_and_query_bell_task() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")

    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    c.measure_instruction(0, 1)

    task = apis.submit_task(circuit=c, device=device, shots=TEST_SHOTS_SIM)
    assert isinstance(task, Task)
    assert task.id_

    counts = task.results(blocked=True)
    details = task.details()

    assert "state" in details
    assert "results" in details
    assert "shots" in details
    assert details.get("source"), "details should contain the submitted QCIS source"
    assert details["state"] == "completed"
    assert_is_counts(counts)
    assert sum(counts.values()) == TEST_SHOTS_SIM
    assert_bell_state_like(counts, tol=0.1)


@requires_cloud
def test_cloud_asymmetric_state_bit_ordering() -> None:
    """An X on qubit 0 must yield {"10": shots}, matching the tc bit ordering.

    Guards against silent bit-reversal regressions in result parsing; the
    Bell state test is insensitive to mirrored bitstrings.
    """
    device = apis.get_device(f"tianyan::{SIMULATOR}")

    c = tc.Circuit(2)
    c.x(0)
    c.measure_instruction(0, 1)

    task = apis.submit_task(circuit=c, device=device, shots=100)
    counts = task.results(blocked=True)

    assert counts == {"10": 100}


@requires_cloud
def test_cloud_batch_submit() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")

    circuits = []
    for _ in range(3):
        c = tc.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.measure_instruction(0, 1)
        circuits.append(c)

    tasks = apis.submit_task(circuit=circuits, device=device, shots=100)
    assert isinstance(tasks, list)
    assert len(tasks) == 3
    for t in tasks:
        assert isinstance(t, Task)
        assert_is_counts(t.results(blocked=True))


@requires_cloud
def test_cloud_no_circuit_no_source_raises() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")
    with pytest.raises(ValueError):
        apis.submit_task(device=device, shots=100)


@requires_cloud
def test_cloud_direct_qcis_source() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")
    qcis = "H Q0\nX Q1\nCZ Q0 Q1\nM Q0\nM Q1"

    t = apis.submit_task(source=qcis, device=device, shots=100)
    assert_is_counts(t.results(blocked=True))


@requires_cloud
def test_cloud_single_qubit_circuit() -> None:
    device = apis.get_device(f"tianyan::{SIMULATOR}")

    c = tc.Circuit(1)
    c.h(0)
    c.measure_instruction(0)

    t = apis.submit_task(circuit=c, device=device, shots=50)
    assert_is_counts(t.results(blocked=True))


@requires_hardware
def test_hardware_submit_on_connected_physical_qubits() -> None:
    device = apis.get_device(f"tianyan::{REAL_DEVICE}")
    q1, q2 = sorted(device.topology()[0])

    c = tc.Circuit(max(q1, q2) + 1)
    c.h(q1)
    c.cx(q1, q2)
    c.measure_instruction(q1, q2)

    task = apis.submit_task(circuit=c, device=device, shots=TEST_SHOTS_REAL)
    assert isinstance(task, Task)
    assert task.id_

    counts = task.results(blocked=True)
    assert_is_counts(counts)
    assert sum(counts.values()) == TEST_SHOTS_REAL


@requires_hardware
def test_hardware_incompatible_circuit_raises() -> None:
    device = apis.get_device(f"tianyan::{REAL_DEVICE}")
    edges = {tuple(sorted(e)) for e in device.topology()}
    nqubits = max(max(e) for e in edges) + 1
    pair = next(
        (q1, q2)
        for q1 in range(nqubits)
        for q2 in range(q1 + 1, nqubits)
        if (q1, q2) not in edges
    )

    c = tc.Circuit(pair[1] + 1)
    c.cx(*pair)
    c.measure_instruction(*pair)

    with pytest.raises(ValueError, match="unconnected physical qubits"):
        apis.submit_task(circuit=c, device=device, shots=TEST_SHOTS_REAL)
