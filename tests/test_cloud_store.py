import gzip
import json
import logging
import os
import sys

import pytest
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit.cloud import apis, local, store, wrapper
from tensorcircuit.cloud.store import TaskStore


@pytest.fixture
def task_store(tmp_path, monkeypatch):
    monkeypatch.setattr(apis, "default_store", None)
    monkeypatch.setattr(store, "default_store_path", str(tmp_path / "default"))
    return TaskStore(tmp_path / "tasks")


def bell():
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    return c


def test_store_disabled_by_default(task_store, tmp_path):
    t = apis.submit_task(device="local::testing", circuit=bell(), shots=10)
    meta = t.details()["tensorcircuit"]
    assert t.local_id is None
    assert meta["archive"] == {
        "requested": False,
        "saved": False,
        "local_id": None,
        "error": None,
    }
    assert meta["positional_logical_mapping"] == {0: 0, 1: 1}
    assert not os.path.exists(tmp_path / "default")
    assert not os.path.exists(task_store.path)


def test_metadata_unknown_without_provider_metadata(task_store, monkeypatch):
    # e.g. quafu, which does not tell whether it remaps the qubits
    monkeypatch.delattr(local, "task_metadata")
    t = apis.submit_task(device="local::testing", circuit=bell(), shots=10)
    meta = t.details()["tensorcircuit"]
    assert meta["schema_version"] == 1 and meta["archive"]["requested"] is False
    assert meta["positional_logical_mapping"] is None


def test_local_rejects_source(task_store):
    with pytest.raises(ValueError, match="source"):
        apis.submit_task(
            device="local::testing", source=bell().to_openqasm(), store=task_store
        )
    assert task_store.find() == []


def test_store_records_submission_and_result(task_store):
    c = tc.Circuit(3)
    c.multicontrol(0, 1, 2, ctrl=[1, 0], unitary=tc.gates._x_matrix)
    c.rx(1, theta=0.3)
    c.measure_instruction(0)
    c.depolarizing(2, px=0.1, py=0.1, pz=0.1)
    t = apis.submit_task(device="local::testing", circuit=c, shots=20, store=task_store)

    r = task_store.get(t.local_id)
    assert r["integrity"] == "complete"
    assert r["submission"]["status"] == "submitted"
    assert r["remote_id"] == t.id_ and r["state"] is None
    assert r["request"]["shots"] == 20
    assert r["versions"]["tensorcircuit"] == tc.__version__
    # only the SDKs of the provider are recorded besides qiskit
    assert set(r["versions"]) <= {"tensorcircuit", "qiskit"}
    qir = r["request"]["circuit"]["qir"]
    assert [g["name"] for g in qir] == ["multicontrol", "rx", "depolarizing"]
    assert qir[2]["is_channel"] and qir[2]["parameters"] == {
        "px": 0.1,
        "py": 0.1,
        "pz": 0.1,
    }
    assert r["request"]["circuit"]["extra_qir"][0]["name"] == "measure"

    counts = t.results()
    # multicontrol flips q2 only, and only q0 is measured
    assert counts == {"0": 20}
    r = task_store.get(remote_id=t.id_, provider="local")
    assert r["state"] == "completed"
    assert r["details"]["results"] == counts
    assert r["tensorcircuit"]["positional_logical_mapping"] == {0: 0}
    assert r["prepared"]["positional_logical_mapping"] == {0: 0}

    t1 = task_store.task(t.local_id)
    assert t1.id_ == t.id_ and t1.local_id == t.local_id
    meta = t1.details()["tensorcircuit"]
    assert meta["positional_logical_mapping"] == {0: 0}


def test_set_store_global_and_override(task_store):
    apis.set_store(task_store)
    apis.submit_task(device="local::testing", circuit=bell(), shots=10)
    apis.submit_task(device="local::testing", circuit=bell(), shots=10, store=False)
    assert len(task_store.find()) == 1

    wrapper.batch_expectation_ps(
        bell(),
        [[3, 3], [1, 1]],
        device="local::testing",
        shots=10,
        with_rem=False,
        compile_func=lambda circuit: (circuit, {}),
    )
    assert len(task_store.find()) == 3

    assert apis.set_store(True).path == store.default_store_path
    apis.set_store(False)
    assert apis.get_store() is None


def test_store_find_and_export(task_store, tmp_path):
    ts = apis.submit_task(
        device="local::testing", circuit=[bell(), bell()], shots=10, store=task_store
    )
    t = apis.submit_task(device="local::testing", circuit=bell(), store=task_store)
    t.details()

    batch_id = task_store.get(ts[0].local_id)["batch_id"]
    rows = task_store.find(batch_id=batch_id)
    assert [r["local_id"] for r in rows] == [x.local_id for x in ts]
    assert [r["local_id"] for r in task_store.find(state="completed")] == [t.local_id]
    assert len(task_store.find(device="local::testing", limit=2)) == 2
    created = task_store.get(t.local_id)["created_at"]
    assert [r["local_id"] for r in task_store.find(since=created)] == [t.local_id]
    assert task_store.find(until=created - 1e3) == []

    path = tmp_path / "out.jsonl.gz"
    assert task_store.export(path, batch_id=batch_id) == 2
    with gzip.open(path, "rt") as f:
        records = [json.loads(line) for line in f]
    assert [r["local_id"] for r in records] == [x.local_id for x in ts]
    assert records[0]["request"]["circuit"]["nqubits"] == 2


def test_store_write_failure_keeps_tasks(task_store, tmp_path, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    with caplog.at_level(logging.WARNING):
        t = apis.submit_task(
            device="local::testing", circuit=bell(), shots=10, store=blocker / "tasks"
        )
    assert "task store write failure" in caplog.text
    archive = t.details()["tensorcircuit"]["archive"]
    assert archive["requested"] and not archive["saved"] and archive["error"]
    assert sum(t.results().values()) == 10


def test_store_unserializable_circuit_still_submits(task_store, caplog):
    c = tc.Circuit(2)
    c.h(0)
    zz = np.kron(tc.gates._z_matrix, tc.gates._z_matrix).reshape([2, 2, 2, 2])
    c.mpo(0, 1, mpo=tc.quantum.QuOperator.from_tensor(zz, [0, 1], [2, 3]))
    with caplog.at_level(logging.WARNING):
        t = apis.submit_task(
            device="local::testing", circuit=c, shots=10, store=task_store
        )
    assert "cannot serialize the circuit" in caplog.text
    assert sum(t.results().values()) == 10
    r = task_store.get(t.local_id)
    assert r["state"] == "completed" and r["request"]["circuit"] is None
    assert r["request"]["circuit_error"]


def test_store_drops_tasks_certainly_not_submitted(task_store):
    with pytest.raises(ValueError, match="Unsupported provider"):
        apis.submit_task(device="nowhere::qpu", circuit=bell(), store=task_store)
    with pytest.raises(ValueError, match="Unsupported device"):
        apis.submit_task(device="local::nowhere", circuit=bell(), store=task_store)
    assert task_store.find() == []
    assert os.listdir(task_store.records_path) == []


def test_store_interrupted_and_corrupt_records(task_store, monkeypatch):
    t = apis.submit_task(device="local::testing", circuit=bell(), store=task_store)

    write_body = TaskStore._write_body

    def fail(self, record):
        raise OSError("disk full")

    monkeypatch.setattr(TaskStore, "_write_body", fail)
    meta = t.details()["tensorcircuit"]
    assert not meta["archive"]["saved"] and "disk full" in meta["archive"]["error"]
    r = task_store.get(t.local_id)
    # the index holds the latest state, while the body is not updated
    assert r["integrity"] == "pending" and r["state"] == "completed"
    assert r["details"] is None
    monkeypatch.setattr(TaskStore, "_write_body", write_body)

    with open(task_store._body_path(t.local_id), "wb") as f:
        f.write(b"broken")
    r = task_store.get(t.local_id)
    assert r["integrity"] == "corrupt" and r["remote_id"] == t.id_
    assert not t.details()["tensorcircuit"]["archive"]["saved"]


def test_store_interrupted_binding_keeps_remote_id(task_store, monkeypatch):
    write_body = TaskStore._write_body

    def fail_after_intent(self, record):
        if record["submission"]["status"] != "intent":
            raise OSError("disk full")
        return write_body(self, record)

    monkeypatch.setattr(TaskStore, "_write_body", fail_after_intent)
    t = apis.submit_task(device="local::testing", circuit=bell(), store=task_store)
    assert t.local_id is None
    monkeypatch.setattr(TaskStore, "_write_body", write_body)

    (row,) = task_store.find()
    r = task_store.get(row["local_id"])
    assert r["integrity"] == "pending" and r["remote_id"] == t.id_
    assert r["submission"]["status"] == "submitted"
    t1 = task_store.task(row["local_id"])
    assert t1.id_ == t.id_
    # a later update writes back the indexed remote id with a complete body
    assert t1.details()["tensorcircuit"]["archive"]["saved"]
    r = task_store.get(row["local_id"])
    assert r["integrity"] == "complete" and r["remote_id"] == t.id_
