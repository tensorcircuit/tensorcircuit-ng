"""
Tests for the Qboson provider.

Offline tests run the real Kaiwu SDK against a fake server. Set
``TC_CLOUD_TEST=1``, ``TC_CLOUD_HARDWARE_TEST=1`` and ``TC_TOKEN_QBOSON`` as
``<user id>:<SDK code>`` to run the paid hardware tests.
"""

import importlib.util
import io
import itertools
import os
import pickle
import zlib
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from tensorcircuit.cloud import apis, qboson, store
from tensorcircuit.cloud.abstraction import Task, TaskFailed
from tensorcircuit.cloud.utils import SubmissionRejected, SubmissionUnknown

HAS_SDK = importlib.util.find_spec("kaiwu") is not None
HARDWARE_TESTS_ENABLED = (
    HAS_SDK
    and bool(os.getenv("TC_TOKEN_QBOSON"))
    and os.getenv("TC_CLOUD_TEST") == "1"
    and os.getenv("TC_CLOUD_HARDWARE_TEST") == "1"
)
requires_sdk = pytest.mark.skipif(not HAS_SDK, reason="kaiwu missing")
requires_hardware = pytest.mark.skipif(
    not HARDWARE_TESTS_ENABLED,
    reason="set TC_TOKEN_QBOSON, TC_CLOUD_TEST=1 and TC_CLOUD_HARDWARE_TEST=1 "
    "to run paid hardware tests",
)

DEVICE = "qboson::cim"
TOKEN = "1234:fake-sdk-code"
RESULT_URL = "https://oss.invalid/result.pkl"


def response(body, status=200):
    import json

    import requests

    r = requests.Response()
    r.status_code = status
    r._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    r.headers["Content-Type"] = "application/json"
    return r


def result_pickle(spins):
    spins = np.asarray(spins, dtype="<f8")
    return pickle.dumps(
        {
            "format": "ndarray-zlib-v1",
            "shape": spins.shape,
            "dtype": "<f8",
            "data": zlib.compress(spins.tobytes()),
        }
    )


class FakeQboson:
    """
    Fake Qboson server behind ``requests.adapters.HTTPAdapter.send``. ``tasks``
    maps the task name to its matrix as uploaded, ``status`` is the ``data`` of
    the task query, ``creation`` the response of the task creation, or an
    exception to raise, and ``solutions`` the spins of the result.
    """

    def __init__(self):
        self.requests = []
        self.tasks = {}
        self.uploads = {}
        self.status = {"code": "2002", "task_status": "计算中"}
        self.creation = None
        self.solutions = None

    def __call__(self, request):
        r = self.answer(request)
        r.request = request
        r.url = request.url
        return r

    def answer(self, request):
        parts = urlsplit(request.url)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.requests.append((request.method, parts.path, query))
        if parts.path.endswith("/get_sdk_task_result/"):
            assert request.headers["Authorization"].startswith("1234/")
            if query["task_name"] not in self.tasks:
                return response({"code": "10320", "data": None, "msg": "数据不存在"})
            data = dict(self.status, desc="", failed_desc="", time_consume=None)
            if data["code"] == "1000":
                data["result"] = RESULT_URL
            return response({"code": "0", "data": data, "msg": "success"})
        if parts.path.endswith("/oss_signed_url/"):
            url = "https://oss.invalid/" + query["filename"]
            return response(
                {"code": 2000, "data": {"url": url, "oss_file_url": url}, "msg": "ok"}
            )
        if request.method == "PUT":
            body = request.body
            self.uploads[parts.path.strip("/")] = (
                body.getvalue() if hasattr(body, "getvalue") else body
            )
            return response(b"")
        if parts.path.endswith("/create_sdk_task/"):
            form = {k: v[0] for k, v in parse_qs(request.body).items()}
            creation = self.creation
            if callable(creation):
                creation = creation(form)
            if isinstance(creation, Exception):
                raise creation
            if creation is not None:
                return creation
            self.tasks[form["task_name"]] = self.uploads[form["file_name"]]
            return response({"code": "0", "data": {"result_id": "r"}, "msg": "ok"})
        if request.url == RESULT_URL:
            return response(result_pickle(self.solutions))
        raise AssertionError("unexpected request %s %s" % (request.method, request.url))

    def creations(self):
        return [r for r in self.requests if r[1].endswith("/create_sdk_task/")]

    def matrix(self, name):
        d = pickle.loads(self.tasks[name])
        return np.frombuffer(zlib.decompress(d["data"]), dtype=d["dtype"]).reshape(
            d["shape"]
        )


@pytest.fixture
def fake_qboson(monkeypatch, tmp_path):
    import requests
    from kaiwu.cim import _cim_optimizer

    fake = FakeQboson()
    monkeypatch.setattr(
        requests.adapters.HTTPAdapter,
        "send",
        lambda adapter, request, **kws: fake(request),
    )
    # a license of the fake account instead of the one of the SDK server
    monkeypatch.setattr(qboson, "_licensed", TOKEN)
    monkeypatch.setattr(_cim_optimizer, "ensure_license", lambda *a, **k: None)
    monkeypatch.setattr(
        _cim_optimizer,
        "_read_license",
        lambda: ({"user_id": 1234, "qubits": 999999, "exp": 4102444800}, "x" * 30),
    )
    monkeypatch.setattr(qboson, "_checkpoint_dir", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TC_TOKEN_QBOSON", TOKEN)
    return fake


def maxcut():
    # the cut of the square with a diagonal is at most 4, by {0, 2} and {1, 3}
    w = np.zeros([4, 4])
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        w[i, j] = w[j, i] = 1.0
    return -w


@requires_sdk
def test_submit_and_query_through_sdk(fake_qboson, tmp_path):
    from kaiwu.common import CheckpointManager

    j = maxcut()
    s = store.TaskStore(tmp_path / "tasks")
    CheckpointManager.save_dir = "user-dir"
    t = apis.submit_task(device=DEVICE, source=j, store=s)
    # the checkpoint directory of the user's own SDK use is kept
    assert CheckpointManager.save_dir == "user-dir"
    CheckpointManager.save_dir = None
    assert t.id_.startswith("tc-") and len(fake_qboson.creations()) == 1
    np.testing.assert_array_equal(fake_qboson.matrix(t.id_), j)
    assert t.details()["state"] == "pending"
    fake_qboson.status = {"code": "2001", "task_status": "校验中"}
    assert t.details()["state"] == "pending"

    fake_qboson.status = {"code": "1000", "task_status": "计算成功"}
    fake_qboson.solutions = [[1, -1, 1, -1], [-1, 1, -1, 1], [1, 1, -1, -1]]
    d = t.details()
    assert d["memory"] == ["0101", "1010", "0011"]
    assert t.results() == {"0101": 1, "1010": 1, "0011": 1}
    assert d["qboson"]["task_status"] == "计算成功"

    record = s.get(t.local_id)
    assert record["state"] == "completed" and record["request"]["circuit"] is None
    # the task is found again from the store by its name and matrix
    restored = s.task(t.local_id)
    assert restored.results() == t.results()


@requires_sdk
def test_fields_by_an_auxiliary_spin(fake_qboson, tmp_path):
    rng = np.random.default_rng(1)
    j = np.triu(rng.integers(-3, 4, size=(4, 4)), 1).astype(float)
    j = j + j.T
    h = rng.integers(-3, 4, size=4).astype(float)
    s = store.TaskStore(tmp_path / "tasks")
    t = apis.submit_task(device=DEVICE, source={"J": j, "h": h}, store=s)
    assert isinstance(t, Task) and len(s.find()) == 1
    assert s.find()[0]["batch_id"] is None
    m = fake_qboson.matrix(t.id_)
    assert m.shape == (5, 5)

    def energy(s):
        return -s @ j @ s - h @ s

    spins = [np.array(s) for s in itertools.product([-1.0, 1.0], repeat=5)]
    e = [-s @ m @ s for s in spins]
    ground = [s for s, x in zip(spins, e) if np.isclose(x, min(e))]
    # both spin flipped ground states of the matrix sent decode to the optimum
    assert len(ground) == 2
    fake_qboson.status = {"code": "1000"}
    fake_qboson.solutions = ground
    best = min(
        itertools.product([-1.0, 1.0], repeat=4), key=lambda s: energy(np.array(s))
    )
    bits = "".join("0" if v > 0 else "1" for v in best)
    assert t.results() == {bits: 2}


@requires_sdk
def test_failed_and_unknown_states(fake_qboson):
    t = apis.submit_task(device=DEVICE, source=maxcut(), store=False)
    fake_qboson.status = {"code": "4001", "task_status": "校验失败"}
    with pytest.raises(TaskFailed, match="校验失败"):
        t.results()
    fake_qboson.status = {"code": "3001", "task_status": "排队中"}
    with pytest.raises(ValueError, match="Unknown Qboson task status"):
        t.details()
    fake_qboson.tasks.clear()
    with pytest.raises(ValueError, match="not found"):
        t.details()


@requires_sdk
@pytest.mark.parametrize(
    "creation, error",
    [
        (response({"code": 4000, "data": None, "msg": "资源不足"}), SubmissionRejected),
        ("connection", SubmissionUnknown),
        (response({"detail": "boom"}, 502), SubmissionUnknown),
        (response(b"<html></html>"), SubmissionUnknown),
    ],
)
def test_submission_errors_are_classified(fake_qboson, tmp_path, creation, error):
    import requests

    if creation == "connection":
        creation = requests.ConnectionError("reset")
    fake_qboson.creation = creation
    s = store.TaskStore(tmp_path / "tasks")
    with pytest.raises(error):
        apis.submit_task(device=DEVICE, source=maxcut(), store=s)
    records = s.find()
    if error is SubmissionUnknown:
        assert [r["submission_status"] for r in records] == ["unknown"]
        assert s.get(records[0]["local_id"])["prepared"]["problem"]["task_name"]
    else:
        assert records == []


@requires_sdk
@pytest.mark.parametrize(
    "source, kws, message",
    [
        ([[0.0, 1.0], [1.0, 0.0]], {}, "batch"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), {}, "symmetric"),
        (np.eye(2), {}, "zero diagonal"),
        ({"J": np.zeros([2, 2]), "h": [1.0]}, {}, "length 2"),
        ({"J": np.zeros([2, 2])}, {}, "keys"),
        (np.zeros([1000, 1000]), {}, None),
        ({"J": np.zeros([1000, 1000]), "h": np.zeros(1000)}, {}, "exceed"),
        (np.zeros([2, 2]), {"shots": 10}, "optimization mode"),
    ],
)
def test_local_failures_are_not_submitted(fake_qboson, source, kws, message):
    if message is None:
        apis.submit_task(device=DEVICE, source=source, store=False, **kws)
        return
    with pytest.raises(ValueError, match=message):
        apis.submit_task(device=DEVICE, source=source, store=False, **kws)
    assert fake_qboson.creations() == []


@requires_sdk
def test_batch(fake_qboson, tmp_path):
    import scipy.sparse

    j = maxcut()
    s = store.TaskStore(tmp_path / "tasks")
    # a tuple is a batch, like a list
    ts = apis.submit_task(
        device=DEVICE,
        source=(scipy.sparse.csr_matrix(j), {"J": j, "h": np.ones(4)}),
        store=s,
    )
    assert [t.circuit_index for t in ts] == [0, 1]
    assert len({t.id_ for t in ts}) == 2 and len(fake_qboson.creations()) == 2
    assert [r["batch_id"] for r in s.find()][0] is not None
    with pytest.raises(NotImplementedError, match="submit_task"):
        apis.resubmit_task(s.task(ts[0].local_id))
    assert len(fake_qboson.creations()) == 2

    # the size is checked before a sparse matrix is made dense
    with pytest.raises(ValueError, match="exceed"):
        apis.submit_task(
            device=DEVICE, source=scipy.sparse.eye(10**6, format="csr"), store=False
        )

    # a failure after the first task names the submitted one
    names = []

    def second_rejected(form):
        names.append(form["task_name"])
        if len(names) == 2:
            return response({"code": 4000, "data": None, "msg": "资源不足"})
        return None

    fake_qboson.creation = second_rejected
    with pytest.raises(SubmissionUnknown) as e:
        apis.submit_task(device=DEVICE, source=[j, j], store=False)
    assert len(names) == 2 and names[0] in fake_qboson.tasks
    assert names[0] in str(e.value)


@requires_sdk
def test_result_unpickler_loads_no_class(fake_qboson):
    class Evil:
        def __reduce__(self):
            return (print, ("never",))

    with pytest.raises(pickle.UnpicklingError):
        qboson._ResultUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()


@requires_sdk
def test_task_without_problem_raises(fake_qboson):
    t = apis.get_task("tc-0123", device=DEVICE)
    with pytest.raises(ValueError, match="set_store"):
        t.details()


def test_missing_sdk_reports_installation(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "kaiwu", None)
    with pytest.raises(ImportError, match="pip install"):
        qboson._kaiwu()
    assert "qboson" in apis.avail_providers
    with pytest.raises(ValueError, match="user id"):
        qboson._optimizer("no-colon", "t")


@requires_hardware
def test_hardware_maxcut():
    # a random 8-vertex graph with a maximal cut of 10, by brute force
    rng = np.random.default_rng(7)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if rng.random() < 0.5]
    w = np.zeros([8, 8])
    for i, j in edges:
        w[i, j] = w[j, i] = 1.0
    t = apis.submit_task(device=DEVICE, source=-w)
    r = t.results()
    print(t.id_, r)
    cuts = [sum(b[i] != b[j] for i, j in edges) for b in r]
    assert max(cuts) == 10
