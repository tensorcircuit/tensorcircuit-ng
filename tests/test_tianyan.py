"""Tests for the TianYan cloud provider.

Offline tests run by default. Set ``TC_CLOUD_TEST=1`` and ``LOGIN_KEY`` to run
online simulator tests. Real-device tests additionally require
``TC_CLOUD_HARDWARE_TEST=1``.
"""

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import tensorcircuit as tc
from tensorcircuit.cloud import apis, tianyan
from tensorcircuit.cloud.abstraction import Device, Task

LOGIN_KEY = os.getenv("LOGIN_KEY")
HAS_CQLIB = tianyan.TianYanPlatform is not None
RUN_CLOUD_TESTS = os.getenv("TC_CLOUD_TEST") == "1"
RUN_HARDWARE_TESTS = os.getenv("TC_CLOUD_HARDWARE_TEST") == "1"
CLOUD_TESTS_ENABLED = HAS_CQLIB and bool(LOGIN_KEY) and RUN_CLOUD_TESTS
HARDWARE_TESTS_ENABLED = CLOUD_TESTS_ENABLED and RUN_HARDWARE_TESTS
FAKE_QUANTUM_LANGUAGE = SimpleNamespace(QCIS="QCIS")

SIMULATOR = "tianyan_sw"  # full-amplitude simulator, no topology constraints
REAL_DEVICE = "tianyan176"  # superconducting hardware with topology constraints

# keep shots low on real hardware to save quota
TEST_SHOTS_SIM = 1000
TEST_SHOTS_REAL = 100


@unittest.skipUnless(HAS_CQLIB, "cqlib is required for TianYan conversion tests")
class TestQasmConversion(unittest.TestCase):
    """Offline tests for the official OpenQASM converter."""

    def test_parameterized_qasm_uses_cqlib_converter(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
rz(pi/2) q[0];
measure q[0] -> c[0];
"""

        source = tianyan._qasm_to_qcis(qasm)

        self.assertIn("RZ Q0 1.570796", source)
        self.assertEqual(source.splitlines()[-1], "M Q0")


class TestOfflineProvider(unittest.TestCase):
    """Offline tests for status queries and topology validation."""

    def test_incompatible_cqlib_fails_fast(self) -> None:
        class IncompatiblePlatform:
            pass

        with patch.object(tianyan, "TianYanPlatform", IncompatiblePlatform):
            with self.assertRaisesRegex(ImportError, "_send_request"):
                tianyan._assert_cqlib()

    def test_topology_validation_accepts_connected_circuit(self) -> None:
        graph = {0: {1}, 1: {0, 2}, 2: {1}}
        circuit = tc.Circuit(2)
        circuit.cx(0, 1)

        tianyan._validate_circuit_topology(circuit, graph, set(graph))

    def test_topology_validation_rejects_unconnected_pair(self) -> None:
        graph = {0: {2}, 1: set(), 2: {0}}
        circuit = tc.Circuit(2)
        circuit.cx(0, 1)

        with self.assertRaisesRegex(ValueError, "unconnected physical qubits"):
            tianyan._validate_circuit_topology(circuit, graph, set(graph))

    def test_topology_validation_rejects_unavailable_qubit(self) -> None:
        graph = {0: {1}, 1: {0}}
        circuit = tc.Circuit(2)
        circuit.cx(0, 1)

        with self.assertRaisesRegex(ValueError, "unavailable on this device"):
            tianyan._validate_circuit_topology(circuit, graph, {0})

    @unittest.skipUnless(HAS_CQLIB, "cqlib is required for QCIS conversion")
    def test_partial_measurement_preserves_qubits_and_order(self) -> None:
        circuit = tc.Circuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.z(2)
        circuit.measure_instruction(2, 0)

        source = tianyan._tc_qir_to_qcis(circuit)
        measurements = [line for line in source.splitlines() if line.startswith("M ")]

        self.assertEqual(measurements, ["M Q2", "M Q0"])

    @unittest.skipUnless(HAS_CQLIB, "cqlib is required for QCIS conversion")
    def test_unsupported_qir_gate_fails_without_fallback(self) -> None:
        circuit = tc.Circuit(1)
        circuit.i(0)

        with self.assertRaisesRegex(ValueError, "Unsupported gate"):
            tianyan._tc_qir_to_qcis(circuit)

    def test_get_task_details_queries_once_and_returns_pending(self) -> None:
        platform = _FakePlatform({"data": {"experimentResultModelList": []}})
        device = Device.from_name("tianyan::offline-status")
        task = Task("pending-task", device=device)
        task.add_details(source="H Q0")

        with (
            patch.object(tianyan, "_assert_cqlib"),
            patch.object(tianyan, "_get_platform", return_value=platform),
        ):
            details = tianyan.get_task_details(task, device, "token", False)

        self.assertEqual(details["state"], "pending")
        self.assertEqual(details["source"], "H Q0")
        self.assertEqual(len(platform.calls), 1)

    def test_get_task_details_maps_failure_state(self) -> None:
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

        self.assertEqual(details["state"], "failed")
        self.assertEqual(details["err"], "backend failure")

    def test_result_without_explicit_state_is_completed(self) -> None:
        result = {
            "experimentTaskId": "completed-task",
            "resultStatus": [[0], [0], [1]],
        }
        details = tianyan._parse_result(
            result, Device.from_name("tianyan::offline-completed")
        )

        self.assertEqual(details["state"], "completed")
        self.assertEqual(details["results"], {"0": 1, "1": 1})

    def test_topology_failure_stops_before_submission(self) -> None:
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
            with self.assertRaisesRegex(ValueError, "topology unavailable"):
                tianyan.submit_task(device, "token", circuit=circuit)

        self.assertFalse(platform.submitted)

    def test_incompatible_circuit_stops_before_submission(self) -> None:
        platform = _FakePlatform({})
        device = Device.from_name("tianyan::hardware-incompatible")
        circuit = tc.Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        graph = {0: {1}, 1: {0, 2}, 2: {1}}

        with (
            patch.object(tianyan, "_assert_cqlib"),
            patch.object(tianyan, "_get_platform", return_value=platform),
            patch.object(
                tianyan, "_get_device_topology", return_value=(graph, set(graph))
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unconnected physical qubits"):
                tianyan.submit_task(device, "token", circuit=circuit)

        self.assertFalse(platform.submitted)

    def test_batch_submission_on_hardware_validates_each_circuit(self) -> None:
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
            patch.object(
                tianyan, "_get_device_topology", return_value=(graph, set(graph))
            ),
        ):
            tasks = tianyan.submit_task(
                device, "token", circuit=[circuit_1, circuit_2], shots=[100, 100]
            )

        self.assertEqual(len(tasks), 2)
        self.assertEqual(platform.last_submission["num_shots"], 100)

    def test_batch_submission_rejects_different_shots(self) -> None:
        platform = _FakePlatform({}, batch_query_ids=["task-1", "task-2"])
        device = Device.from_name("tianyan::tianyan_sw")

        with (
            patch.object(tianyan, "_assert_cqlib"),
            patch.object(tianyan, "_get_platform", return_value=platform),
            patch.object(tianyan, "QuantumLanguage", FAKE_QUANTUM_LANGUAGE),
        ):
            with self.assertRaisesRegex(ValueError, "requires the same shots"):
                tianyan.submit_task(
                    device,
                    "token",
                    source=["H Q0", "X Q0"],
                    shots=[100, 200],
                )

        self.assertFalse(platform.submitted)

    def test_batch_submission_rejects_shots_length_mismatch(self) -> None:
        platform = _FakePlatform({}, batch_query_ids=["task-1", "task-2"])
        device = Device.from_name("tianyan::tianyan_sw")

        with (
            patch.object(tianyan, "_assert_cqlib"),
            patch.object(tianyan, "_get_platform", return_value=platform),
            patch.object(tianyan, "QuantumLanguage", FAKE_QUANTUM_LANGUAGE),
        ):
            with self.assertRaisesRegex(ValueError, "Expected 2 shots values"):
                tianyan.submit_task(
                    device,
                    "token",
                    source=["H Q0", "X Q0"],
                    shots=[100],
                )

        self.assertFalse(platform.submitted)

    def test_batch_submission_validates_returned_task_ids(self) -> None:
        platform = _FakePlatform({}, batch_query_ids=["task-1"])
        device = Device.from_name("tianyan::tianyan_sw")

        with (
            patch.object(tianyan, "_assert_cqlib"),
            patch.object(tianyan, "_get_platform", return_value=platform),
            patch.object(tianyan, "QuantumLanguage", FAKE_QUANTUM_LANGUAGE),
        ):
            with self.assertRaisesRegex(ValueError, "1 task IDs for 2"):
                tianyan.submit_task(device, "token", source=["H Q0", "X Q0"], shots=100)

    @unittest.skipUnless(HAS_CQLIB, "cqlib is required for OpenQASM conversion")
    def test_openqasm_source_is_converted_before_submission(self) -> None:
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
            tianyan.submit_task(
                device, "token", source=qasm, lang="OPENQASM", shots=100
            )

        submitted = platform.last_submission["circuit"]
        self.assertIn("Y2M Q0", submitted)
        self.assertIn("RZ Q0", submitted)
        self.assertEqual(submitted.splitlines()[-1], "M Q0")

    def test_unsupported_source_language_fails_before_platform_creation(self) -> None:
        device = Device.from_name("tianyan::tianyan_sw")

        with (
            patch.object(tianyan, "_assert_cqlib"),
            patch.object(tianyan, "_get_platform") as get_platform,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported TianYan source"):
                tianyan.submit_task(device, "token", source="H Q0", lang="UNKNOWN")

        get_platform.assert_not_called()

    @unittest.skipUnless(HAS_CQLIB, "cqlib is required for OpenQASM conversion")
    def test_qiskit_circuit_uses_qasm2_dumps(self) -> None:
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

        self.assertIn("Y2M Q0", source)
        self.assertIn("RZ Q0", source)

    def test_list_tasks_reports_unsupported_operation(self) -> None:
        device = Device.from_name("tianyan::tianyan_sw")
        with self.assertRaisesRegex(NotImplementedError, "no listing API"):
            tianyan.list_tasks(device, "token")

    def test_remove_task_reports_unsupported_operation(self) -> None:
        device = Device.from_name("tianyan::tianyan_sw")
        task = Task("cancel-task", device=device)
        with self.assertRaisesRegex(NotImplementedError, "no cancellation endpoint"):
            tianyan.remove_task(task, "token")

    def test_list_devices_parses_machine_names(self) -> None:
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

        self.assertEqual([d.name for d in devices], ["tianyan_sw", "tianyan176"])

    def test_list_devices_rejects_missing_machine_name(self) -> None:
        platform = _FakePlatform({"data": [{"id": 1, "status": 0}]})

        with (
            patch.object(tianyan, "_assert_cqlib"),
            patch.object(tianyan, "_get_platform", return_value=platform),
        ):
            with self.assertRaisesRegex(ValueError, "missing machine name"):
                tianyan.list_devices("token")


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


@unittest.skipUnless(
    CLOUD_TESTS_ENABLED,
    "Set LOGIN_KEY and TC_CLOUD_TEST=1 to run TianYan cloud tests",
)
class TianYanTestBase(unittest.TestCase):
    """Provide token isolation and common cloud-result assertions."""

    def setUp(self) -> None:
        assert LOGIN_KEY is not None
        token_patch = patch.dict(apis.saved_token, {"tianyan::": LOGIN_KEY})
        token_patch.start()
        self.addCleanup(token_patch.stop)

    def assertIsCounts(self, counts: Any) -> None:
        """Assert the value is a valid counts dict: {bitstring: int_count}."""
        self.assertIsInstance(counts, dict)
        self.assertGreater(len(counts), 0)
        for bitstring, cnt in counts.items():
            self.assertIsInstance(bitstring, str)
            self.assertIsInstance(cnt, int)
            self.assertGreaterEqual(cnt, 0)

    def assertBellStateLike(self, counts: Dict[str, int], tol: float = 0.3) -> None:
        """
        Assert counts are close to a Bell state distribution (00 and 11 dominant).
        tol: upper bound of the allowed noise ratio (0.1 for simulator, 0.5 for hardware)
        """
        total = sum(counts.values())
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        self.assertGreater(
            p00 + p11,
            1 - tol,
            f"Bell state 00+11 ratio too low: {p00+p11:.2f}, counts={counts}",
        )


class TestProviderAndDevices(TianYanTestBase):
    """Test provider-level APIs: device listing, properties, topology."""

    def test_01_set_provider(self) -> None:
        """The provider should be registered."""
        provider = apis.get_provider("tianyan")
        self.assertEqual(str(provider), "tianyan")

    def test_02_list_devices(self) -> None:
        """list_devices should return a non-empty list including simulator and hardware."""
        devices = apis.list_devices(provider="tianyan")
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)

        names = [str(d) for d in devices]
        self.assertIn(f"tianyan::{SIMULATOR}", names)
        self.assertIn(f"tianyan::{REAL_DEVICE}", names)

    def test_03_get_device(self) -> None:
        """get_device should return a Device object from a string name."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")
        self.assertIsInstance(device, Device)
        self.assertEqual(device.name, SIMULATOR)
        self.assertEqual(device.provider.name, "tianyan")

    def test_04_list_properties_simulator(self) -> None:
        """list_properties (simulator) should return standardized properties."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")
        props = apis.list_properties(device, token=LOGIN_KEY)

        self.assertIn("native_gates", props)
        self.assertIn("links", props)
        self.assertIn("bits", props)

        # links format: {(q1, q2): {'CZErrRate': ...}, ...}
        links = props["links"]
        self.assertIsInstance(links, dict)
        if links:
            (q1, _), info = next(iter(links.items()))
            self.assertIsInstance(q1, int)
            self.assertIsInstance(info, dict)

        # bits format: {qubit_id: {T1: ..., ...}, ...}
        bits = props["bits"]
        self.assertIsInstance(bits, dict)
        if bits:
            qid, info = next(iter(bits.items()))
            self.assertIsInstance(qid, int)
            self.assertIsInstance(info, dict)

    @unittest.skipUnless(
        HARDWARE_TESTS_ENABLED,
        "Set TC_CLOUD_HARDWARE_TEST=1 to run TianYan hardware tests",
    )
    def test_05_list_properties_real_device(self) -> None:
        """list_properties (hardware) should include topology and calibration data."""
        device = apis.get_device(f"tianyan::{REAL_DEVICE}")
        props = apis.list_properties(device, token=LOGIN_KEY)

        self.assertIn("links", props)
        self.assertIn("bits", props)
        self.assertGreater(len(props["links"]), 0)
        self.assertGreater(len(props["bits"]), 0)

        sample_bit = next(iter(props["bits"].values()))
        self.assertTrue(
            any(k in sample_bit for k in ["T1", "T2", "ReadoutError", "Freqency"]),
            f"unexpected bits data: {sample_bit}",
        )

    @unittest.skipUnless(
        HARDWARE_TESTS_ENABLED,
        "Set TC_CLOUD_HARDWARE_TEST=1 to run TianYan hardware tests",
    )
    def test_06_topology(self) -> None:
        """Device.topology() should return coupling edges compatible with qiskit."""
        device = apis.get_device(f"tianyan::{REAL_DEVICE}")
        topo = device.topology()
        self.assertIsInstance(topo, list)
        self.assertGreater(len(topo), 0)
        for edge in topo[:3]:
            self.assertEqual(len(edge), 2)
            self.assertIsInstance(edge[0], int)
            self.assertIsInstance(edge[1], int)

    def test_07_native_gates(self) -> None:
        """Device.native_gates() should return the supported native gate list."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")
        gates = device.native_gates()
        self.assertIsInstance(gates, list)
        self.assertIn("h", gates)
        self.assertIn("cx", gates)


class TestTaskSubmissionSimulator(TianYanTestBase):
    """Test the full task lifecycle on the simulator: submit, query, parse."""

    def test_11_submit_and_query_bell_task(self) -> None:
        """Submit a Bell state task and verify the complete lifecycle."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")

        c = tc.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.measure_instruction(0, 1)

        task = apis.submit_task(circuit=c, device=device, shots=TEST_SHOTS_SIM)
        self.assertIsInstance(task, Task)
        self.assertTrue(task.id_)

        counts = task.results(blocked=True)
        details = task.details()

        self.assertIn("state", details)
        self.assertIn("results", details)
        self.assertIn("shots", details)
        self.assertIn(
            "source", details, "details should contain the submitted QCIS source"
        )
        self.assertTrue(details["source"])
        self.assertEqual(details["state"], "completed")
        self.assertIsCounts(counts)
        self.assertEqual(sum(counts.values()), TEST_SHOTS_SIM)
        self.assertBellStateLike(counts, tol=0.1)

    def test_12_asymmetric_state_bit_ordering(self) -> None:
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

        self.assertEqual(counts, {"10": 100})


@unittest.skipUnless(
    HARDWARE_TESTS_ENABLED,
    "Set TC_CLOUD_HARDWARE_TEST=1 to run TianYan hardware tests",
)
class TestTaskSubmissionRealDevice(TianYanTestBase):
    """Test task submission on real hardware with topology validation and noise."""

    def test_21_submit_on_connected_physical_qubits(self) -> None:
        """Build a circuit directly on a connected physical pair and submit it."""
        device = apis.get_device(f"tianyan::{REAL_DEVICE}")
        q1, q2 = sorted(device.topology()[0])

        c = tc.Circuit(max(q1, q2) + 1)
        c.h(q1)
        c.cx(q1, q2)
        c.measure_instruction(q1, q2)

        task = apis.submit_task(circuit=c, device=device, shots=TEST_SHOTS_REAL)
        self.assertIsInstance(task, Task)
        self.assertTrue(task.id_)

        counts = task.results(blocked=True)
        self.assertIsCounts(counts)
        self.assertEqual(sum(counts.values()), TEST_SHOTS_REAL)

    def test_22_incompatible_circuit_raises_on_hardware(self) -> None:
        """A circuit violating the device topology must raise before submission."""
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

        with self.assertRaisesRegex(ValueError, "unconnected physical qubits"):
            apis.submit_task(circuit=c, device=device, shots=TEST_SHOTS_REAL)


class TestBatchSubmission(TianYanTestBase):
    """Test batch task submission."""

    def test_31_batch_submit(self) -> None:
        """Submit a batch of identical circuits."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")

        circuits = []
        for _ in range(3):
            c = tc.Circuit(2)
            c.h(0)
            c.cx(0, 1)
            c.measure_instruction(0, 1)
            circuits.append(c)

        tasks = apis.submit_task(circuit=circuits, device=device, shots=100)
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 3)

        for t in tasks:
            self.assertIsInstance(t, Task)
            counts = t.results(blocked=True)
            self.assertIsCounts(counts)


class TestEdgeCases(TianYanTestBase):
    """Test edge cases and error handling."""

    def test_41_no_circuit_no_source(self) -> None:
        """Submitting without circuit or source should raise."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")
        with self.assertRaises(ValueError):
            apis.submit_task(device=device, shots=100)

    def test_42_direct_qcis_source(self) -> None:
        """Submit a QCIS source string directly."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")
        qcis = "H Q0\nX Q1\nCZ Q0 Q1\nM Q0\nM Q1"

        t = apis.submit_task(source=qcis, device=device, shots=100)
        counts = t.results(blocked=True)
        self.assertIsCounts(counts)

    def test_43_single_qubit_circuit(self) -> None:
        """A single-qubit circuit needs no topology validation; use the simulator."""
        device = apis.get_device(f"tianyan::{SIMULATOR}")

        c = tc.Circuit(1)
        c.h(0)
        c.measure_instruction(0)

        t = apis.submit_task(circuit=c, device=device, shots=50)
        counts = t.results(blocked=True)
        self.assertIsCounts(counts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
