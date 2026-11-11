import sys
import os
import copy
import json
import logging
import re
import stat
import time
from types import SimpleNamespace
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit.cloud import apis, store, wrapper
from tensorcircuit.cloud.abstraction import Task, TaskFailed
from tensorcircuit.cloud.config import tencent_base_url
from tensorcircuit.cloud.utils import SubmissionUnknown
from tensorcircuit.results import counts

requires_cloud = pytest.mark.skipif(
    os.getenv("TC_CLOUD_TEST") != "1",
    reason="set TC_CLOUD_TEST=1 and a tencent token to run cloud tests",
)


@pytest.fixture(autouse=True)
def restore_cloud_defaults():
    yield
    apis.set_provider("tencent")
    apis.set_device("tencent::simulator:tc")


@pytest.fixture
def token_store(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    monkeypatch.setenv("TC_AUTH_PATH", str(path))
    for k in list(os.environ):
        if k.startswith("TC_TOKEN"):
            monkeypatch.delenv(k)
    monkeypatch.setattr(apis, "saved_token", {})
    return path


class FakeTencent:
    """
    Record requests posted to tencent endpoints and reply canned responses,
    a response can be a callable of the request json.
    """

    def __init__(self):
        self.responses = {}
        self.calls = []

    def __call__(self, url, json=None, headers=None, **kws):
        endpoint = url[len(tencent_base_url) :].split("?")[0]
        self.calls.append(
            {
                "endpoint": endpoint,
                "json": json,
                "headers": headers,
                "timeout": kws.get("timeout"),
            }
        )
        response = self.responses[endpoint]
        if callable(response):
            response = response(json)
        response = copy.deepcopy(response)
        return SimpleNamespace(status_code=200, json=lambda: response)

    def requests_to(self, endpoint):
        return [call["json"] for call in self.calls if call["endpoint"] == endpoint]


@pytest.fixture
def fake_tencent(token_store, monkeypatch):
    requests = pytest.importorskip("requests")
    server = FakeTencent()
    monkeypatch.setattr(requests, "post", server)
    return server


@requires_cloud
def test_get_token():
    assert apis.get_token(provider="Tencent") is not None
    p = apis.get_provider("tencent")
    assert p.get_token() is not None
    assert p.get_device("simulator:tc").get_token() is not None


@requires_cloud
def test_list_devices():
    print(apis.list_devices())
    p = apis.get_provider()
    print(p.list_devices())
    print(p.list_devices(state="on"))


def test_get_device():
    d1 = apis.get_device(device="tencent::hello")
    assert d1.name == "hello"
    assert d1.provider.name == "tencent"
    d2 = apis.get_device(device="hello")
    assert d2.name == "hello"
    assert d2.provider.name == "tencent"
    p = apis.get_provider()
    d3 = p.get_device("tencent::hello")
    assert d3.name == "hello"
    assert d3.provider.name == "tencent"
    d4 = p.get_device("hello")
    assert d4.name == "hello"
    assert d4.provider.name == "tencent"


def test_get_device_cache():
    d1 = apis.get_device("local::testing")
    d2 = apis.get_device(provider="local", device="testing")
    apis.set_provider("local")
    d3 = apis.get_device("testing")
    assert id(d1) == id(d2)
    assert id(d3) == id(d1)
    apis.set_provider("tencent")
    d4 = apis.get_device("testing")
    assert d4.provider.name == "tencent"
    assert id(d4) != id(d1)


@requires_cloud
def test_list_properties():
    d = apis.get_device(device="simulator:aer")
    print(d.list_properties())
    print(apis.list_properties(device="simulator:aer"))
    with pytest.raises(ValueError):
        apis.list_properties(device="hell")


@requires_cloud
def test_submit_task():
    c = tc.Circuit(3)
    c.H(0)
    c.H(1)
    c.H(2)
    t = apis.submit_task(device="simulator:tc", circuit=c)
    r = t.details()
    assert r["state"] in ["pending", "completed"]
    print(t.results(blocked=True))
    assert t.get_logical_physical_mapping() == {0: 0, 1: 1, 2: 2}


@requires_cloud
def test_resubmit_task():
    c = tc.Circuit(3)
    c.H(0)
    c.H(1)
    t = apis.submit_task(device="simulator:aer", circuit=c)
    time.sleep(15)
    t1 = apis.resubmit_task(t)
    print(t.details())
    print(t1.details(wait=True))


@requires_cloud
def test_get_task():
    apis.set_device("simulator:tcn1")
    c = tc.Circuit(2)
    c.cx(0, 1)
    t = apis.submit_task(circuit=c)
    t1 = apis.get_task(t.id_)
    assert t1.id_ == t.id_
    t2 = apis.get_device("tencent::simulator:tcn1").get_task(t.id_)
    assert t2.id_ == t.id_

    apis.set_device()


@requires_cloud
def test_list_tasks():
    d = apis.get_device(device="simulator:aer")
    print(d.list_tasks())
    print(apis.list_tasks(device="simulator:tc"))


def test_local_list_device():
    dd = apis.list_devices(provider="local")
    assert dd[0].name == "testing"


def test_local_submit_task():
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)

    t = apis.submit_task(device="local::testing", circuit=c, shots=2048)
    r = t.results(blocked=True)
    assert counts.kl_divergence({"00": 0.5, "11": 0.5}, r) < 0.1
    print(t.details())
    print(t.get_device())


def test_local_list_tasks():
    print(apis.list_tasks(provider="local"))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_local_measurement_semantics(backend):
    c = tc.Circuit(3)
    c.x(0)
    c.measure_instruction(2)
    c.measure_instruction(0)
    c.measure_instruction(0)
    t = apis.submit_task(device="local::testing", circuit=c, shots=10)
    # bit i is the i-th measurement, unmeasured q1 is not in the result
    assert t.results() == {"011": 10}
    # unwritten classical bits are 0
    c._extra_qir[1]["record_index"] = 3
    t = apis.submit_task(device="local::testing", circuit=c, shots=10)
    assert t.results() == {"0011": 10}
    c.h(0)
    with pytest.raises(ValueError, match="gates after a measurement"):
        apis.submit_task(device="local::testing", circuit=c, shots=10)


def test_local_batch_submit():
    apis.set_provider("local")
    c = tc.Circuit(2)
    c.h(1)
    c.ry(1, theta=0.8)

    ts = apis.submit_task(device="testing", circuit=[c, c])
    print(ts[0].results())

    apis.set_device("testing")
    ts = apis.submit_task(circuit=[c, c])
    print(ts[1].results())
    print(ts[1].details())
    apis.set_provider("tencent")


@requires_cloud
def test_batch_exp_ps():
    pss = [[1, 0], [0, 3]]
    c = tc.Circuit(2)
    c.h(0)
    c.x(1)
    np.testing.assert_allclose(wrapper.batch_expectation_ps(c, pss), [1, -1], atol=1e-5)
    np.testing.assert_allclose(
        wrapper.batch_expectation_ps(c, pss, ws=[1, -0.5]), 1.5, atol=1e-5
    )
    np.testing.assert_allclose(
        wrapper.batch_expectation_ps(c, pss, device="simulator:tcn1"),
        [1, -1],
        atol=1e-1,
    )
    np.testing.assert_allclose(
        wrapper.batch_expectation_ps(c, pss, device="local::default", with_rem=False),
        [1, -1],
        atol=1e-1,
    )


def test_batch_exp_ps_with_idle_qubit():
    c = tc.Circuit(2)
    c.h(0)
    result = wrapper.batch_expectation_ps(
        c,
        [[0, 3]],
        device="local::default",
        with_rem=False,
        shots=100,
        compile_func=lambda circuit: (circuit, {}),
    )
    np.testing.assert_allclose(result, [1])


def test_batch_exp_ps_groups_qubit_wise_commuting_strings():
    n = 3
    c = tc.Circuit(n)
    for i in range(n):
        c.rx(i, theta=0.3 + 0.4 * i)
        c.rz(i, theta=0.2 * i)
    c.cx(0, 1)
    c.cx(1, 2)
    captured = []

    def run(cs, shots):
        captured.extend(cs)
        rs = []
        for ci in cs:
            probs = np.real(tc.backend.numpy(ci.probability()))
            rs.append({format(k, "0%db" % n): p * shots for k, p in enumerate(probs)})
        return rs

    pss = [
        [3, 3, 0],
        [1, 0, 0],
        [0, 3, 3],
        [2, 2, 0],
        [1, 1, 0],
        [0, 1, 0],
        [3, 0, 0],
        [0, 0, 1],
    ]
    r = wrapper.batch_expectation_ps(
        c,
        pss,
        device="local::default",
        with_rem=False,
        compile_func=lambda circuit: (circuit, {}),
        batch_submit_func=run,
    )
    np.testing.assert_allclose(r, wrapper.batch_expectation_ps(c, pss), atol=1e-5)
    # the Z, X and Y strings, instead of one circuit per string
    assert len(captured) == 3


def _exact_counts(cs, shots, keep=1.0):
    # exact counts of the measured classical bits, as a provider returns, under the
    # global depolarizing noise keeping ``keep`` of the state
    rs = []
    for ci in cs:
        m = ci.get_positional_logical_mapping()
        order = [m[i] for i in range(len(m))]
        n = ci._nqubits
        probs = np.real(tc.backend.numpy(ci.probability())).reshape([2] * n)
        probs = probs.sum(axis=tuple(q for q in range(n) if q not in order))
        probs = probs.transpose([sorted(order).index(q) for q in order])
        probs = keep * probs.reshape(-1) + (1 - keep) / probs.size
        rs.append(
            {format(k, "0%db" % len(order)): p * shots for k, p in enumerate(probs)}
        )
    return rs


def test_batch_exp_ps_symbol_circuit_params():
    pytest.importorskip("qiskit")
    import sympy

    ts = sympy.symbols("t0:3", real=True)
    sc = tc.SymbolCircuit(3)
    for i in range(3):
        sc.rx(i, theta=ts[i])
    sc.cx(0, 1)
    sc.cx(1, 2)
    sc.rz(2, theta=2 * ts[0] - 0.4)
    sc.ry(0, theta=ts[2])
    pss = [[3, 3, 0], [1, 0, 1], [0, 2, 3]]
    ws = [1.0, -0.5, 2.0]
    params = [dict(zip(ts, v)) for v in [[0.3, -1.2, 0.8], [2.1, 0.5, -0.7]]]
    exact = np.array(
        [wrapper.batch_expectation_ps(sc.to_circuit(b), pss) for b in params]
    )
    np.testing.assert_allclose(
        wrapper.batch_expectation_ps(sc, pss, params=params), exact, atol=1e-5
    )
    captured = []

    def run(cs, shots):
        captured.extend(cs)
        return _exact_counts(cs, shots)

    kws = dict(device="local::default", with_rem=False, batch_submit_func=run)
    chain = {"coupling_map": [[i, i + 1] for i in range(4)]}
    # compiled once in the call, all bindings submitted together
    r = wrapper.batch_expectation_ps(
        sc, pss, params=params, compile_func=tc.compiler.DefaultCompiler(chain), **kws
    )
    np.testing.assert_allclose(r, exact, atol=1e-4)
    assert len(captured) == 3 * len(params)
    r = wrapper.batch_expectation_ps(
        sc,
        pss,
        ws=ws,
        params=params[1],
        compile_func=tc.compiler.DefaultCompiler(chain),
        **kws,
    )
    np.testing.assert_allclose(r, exact[1] @ ws, atol=1e-4)
    # compiled beforehand onto other qubits, which is not compiled again
    csc, info = tc.compiler.DefaultCompiler(dict(chain, initial_layout=[4, 2, 3]))(sc)
    assert sorted(info["logical_physical_mapping"].values()) == [2, 3, 4]

    def compile_again(c):
        raise AssertionError("compiled again")

    r = wrapper.batch_expectation_ps(
        csc, pss, ws=ws, params=params, info=info, compile_func=compile_again, **kws
    )
    np.testing.assert_allclose(r, exact @ ws, atol=1e-4)
    r = wrapper.batch_expectation_ps(csc, pss, ws=ws, params=params, info=info)
    np.testing.assert_allclose(r, exact @ ws, atol=1e-4)
    assert not np.iscomplexobj(tc.backend.numpy(r))
    with pytest.raises(ValueError, match="SymbolCircuit"):
        wrapper.batch_expectation_ps(sc.to_circuit(params[0]), pss, params=params)


def test_batch_exp_ps_variants(highp):
    pytest.importorskip("qiskit")
    from tensorcircuit.results.qem import mitigators

    c = tc.Circuit(3)
    for i in range(3):
        c.rx(i, theta=0.3 + 0.4 * i)
    c.cx(0, 1)
    c.cx(1, 2)
    c.rz(2, theta=0.7)
    pss = [[3, 3, 0], [1, 0, 1], [0, 2, 3]]
    exact = tc.backend.numpy(wrapper.batch_expectation_ps(c, pss))
    cc, info = tc.compiler.DefaultCompiler(
        {"coupling_map": [[i, i + 1] for i in range(3)], "initial_layout": [3, 1, 2]}
    )(c)
    m = mitigators.Chain(mitigators.CDR(2, seed=1), mitigators.Twirl(3, seed=2))
    calls = []

    def run(cs, shots):
        calls.append(len(cs))
        return _exact_counts(cs, shots, keep=0.6)

    noisy = wrapper.batch_expectation_ps(
        cc,
        pss,
        device="local::default",
        with_rem=False,
        info=info,
        batch_submit_func=run,
        variants=m,
    )
    ideal = wrapper.batch_expectation_ps(cc, pss, info=info, variants=m)
    assert np.shape(noisy) == np.shape(ideal) == (3, 3, 3)
    # the variants of all the measurement bases in one batch
    assert len(calls) == 1
    np.testing.assert_allclose(noisy, 0.6 * tc.backend.numpy(ideal), atol=1e-8)
    np.testing.assert_allclose(m.combine(noisy, ideal), exact, atol=1e-8)
    np.testing.assert_allclose(m.combine(ideal, ideal), exact, atol=1e-8)
    r = wrapper.batch_expectation_ps(
        cc, pss, ws=[1.0, -0.5, 2.0], info=info, variants=lambda ci: [[ci, ci]]
    )
    np.testing.assert_allclose(r, [[exact @ [1.0, -0.5, 2.0]] * 2], atol=1e-8)
    with pytest.raises(ValueError, match="info"):
        wrapper.batch_expectation_ps(cc, pss, variants=m)


@requires_cloud
def test_batch_submit_template():
    run = tc.cloud.wrapper.batch_submit_template(
        device="simulator:tc", batch_limit=2, prior=10
    )
    cs = []
    for i in range(4):
        c = tc.Circuit(4)
        c.h(i)
        cs.append(c)

    rs = run(cs[:3], prior=1)
    assert len(rs) == 3
    rs = run(cs[:4])
    assert len(rs) == 4


@requires_cloud
def test_allz_batch():
    n = 5

    def makec(inputs, params):
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=inputs[i])
        for i in range(n):
            c.rz(i, theta=params[0, i])
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n):
            c.rx(i, theta=params[1, i])
        return c

    pss = []
    for i in range(n):
        ps = [0 for _ in range(n)]
        ps[i] = 3
        pss.append(ps)

    def exp_val(c, device=None):
        rs = tc.cloud.wrapper.batch_expectation_ps(c, pss, device)
        return tc.backend.stack(rs)

    def qmlf(inputs, params, device=None):
        c = makec(inputs, params)
        return exp_val(c, device)

    inputs = tc.array_to_tensor(np.array([0, 1, 0, 1, 0]))
    params = np.ones([2, n])
    print(qmlf(inputs, params, device="9gmon"))
    print(qmlf(inputs, params))


def test_token_default_in_memory_only(token_store):
    apis.set_token("tok-tencent", provider="tencent")
    apis.get_provider("local").set_token("tok-local")
    apis.get_device("local::testing").set_token("tok-testing")
    assert not token_store.exists()
    assert apis.get_token(provider="tencent") == "tok-tencent"
    assert apis.get_device("local::testing").get_token() == "tok-testing"
    assert apis.get_token(provider="local", device="other") == "tok-local"


def test_token_cached_file(token_store):
    token_store.write_text(json.dumps({"tencent::": apis.b64encode_s("tok-legacy")}))
    assert apis.get_token(provider="tencent") == "tok-legacy"

    apis.set_token("tok-device", provider="tencent", device="20xmon", cached=True)
    on_disk = json.loads(token_store.read_text())
    assert apis.b64decode_s(on_disk["tencent::"]) == "tok-legacy"
    assert apis.b64decode_s(on_disk["tencent::20xmon"]) == "tok-device"
    if os.name == "posix":
        assert stat.S_IMODE(token_store.stat().st_mode) == 0o600

    apis.saved_token.clear()  # as in a new process
    assert apis.get_token(provider="tencent", device="20xmon") == "tok-device"
    assert apis.get_token(provider="tencent", device="9gmon") == "tok-legacy"


def test_token_lookup_order(token_store, monkeypatch):
    apis.set_token("tok-file", provider="tencent", cached=True)
    apis.saved_token.clear()
    monkeypatch.setenv("TC_TOKEN", "tok-generic")
    assert apis.get_token(provider="tencent") == "tok-file"
    monkeypatch.setenv("TC_TOKEN_TENCENT", "tok-env")
    assert apis.get_token(provider="tencent") == "tok-env"
    apis.set_token("tok-session", provider="tencent")
    assert apis.get_token(provider="tencent") == "tok-session"
    assert apis.get_token(provider="tencent", device="9gmon") == "tok-session"
    apis.set_token("tok-9gmon", provider="tencent", device="9gmon")
    assert apis.get_token(provider="tencent", device="9gmon") == "tok-9gmon"


def test_token_clear(token_store):
    apis.set_token("tok-tencent", provider="tencent", cached=True)
    apis.set_token("tok-9gmon", provider="tencent", device="9gmon", cached=True)
    apis.set_token("tok-local", provider="local", cached=True)

    apis.set_token(provider="tencent", device="9gmon", clear=True)
    assert apis.get_token(provider="tencent", device="9gmon") == "tok-tencent"
    apis.set_token("tok-9gmon", provider="tencent", device="9gmon", cached=True)
    apis.set_token(provider="tencent", clear=True)
    assert apis.get_token(provider="tencent", device="9gmon") is None
    apis.saved_token.clear()
    assert apis.get_token(provider="local") == "tok-local"

    apis.set_token(clear=True)
    assert apis.get_token(provider="local") is None
    assert not token_store.exists()


@pytest.mark.parametrize("token", [123, "", "  ", "tok-secret\n", "tok\r-secret"])
def test_token_rejects_invalid_input(token_store, token):
    with pytest.raises((TypeError, ValueError)) as e:
        apis.set_token(token, provider="tencent")
    assert "secret" not in str(e.value)
    assert apis.saved_token == {}


def test_token_save_failure_warns(token_store, tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("TC_AUTH_PATH", str(blocker / "auth.json"))
    with caplog.at_level(logging.WARNING):
        apis.set_token("tok-secret", provider="tencent", cached=True)
    assert "failure" in caplog.text
    assert "tok-secret" not in caplog.text
    assert apis.get_token(provider="tencent") == "tok-secret"


def test_tencent_list_devices_redacts_proxy(fake_tencent, monkeypatch, caplog):
    cloud_utils = pytest.importorskip("tensorcircuit.cloud.utils")
    monkeypatch.setattr(
        cloud_utils, "proxy", "http://user:pass-secret@proxy.local:8080", raising=False
    )
    fake_tencent.responses["device/find"] = {"devices": [{"id": "20xmon"}]}
    with caplog.at_level(logging.DEBUG, logger="tensorcircuit.cloud"):
        ds = apis.list_devices(provider="tencent", token="tok-secret")
    assert [str(d) for d in ds] == ["tencent::20xmon"]
    assert fake_tencent.calls[0]["headers"]["Authorization"] == "Bearer tok-secret"
    assert "proxy.local" in caplog.text
    assert "pass-secret" not in caplog.text and "tok-secret" not in caplog.text


def test_tencent_submit_contract(fake_tencent):
    fake_tencent.responses["task/submit"] = {"tasks": [{"id": "t1"}]}
    apis.set_token("tok-tencent", provider="tencent")
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)

    t = apis.submit_task(circuit=c, shots=100)
    assert isinstance(t, Task)
    assert str(t) == "tencent::simulator:tc~~t1"
    call = fake_tencent.calls[-1]
    assert call["endpoint"] == "task/submit"
    assert call["headers"]["Authorization"] == "Bearer tok-tencent"
    assert call["json"] == {
        "device": "simulator:tc?o=3",
        "shots": 100,
        "source": c.to_openqasm(),
        "version": "1",
        "lang": "OPENQASM",
        "prior": 1,
        "remarks": None,
        "group": None,
    }

    apis.submit_task(
        circuit=c,
        enable_qos_qubit_mapping=False,
        enable_qos_gate_decomposition=False,
        qos_dry_run=True,
    )
    assert fake_tencent.calls[-1]["json"]["device"] == "simulator:tc?o=0&dry"
    assert fake_tencent.calls[-1]["json"]["shots"] == 1024


def test_tencent_batch_submit_contract(fake_tencent):
    c = tc.Circuit(1)
    c.x(0)
    fake_tencent.responses["task/submit"] = {
        "tasks": [{"id": "t1"}, {"err": "quota exceeded"}, {"id": "t3"}]
    }
    ts = apis.submit_task(device="tencent::20xmon", circuit=[c, c, c], shots=10)
    assert [t.id_ for t in ts] == ["t1", "t3"]
    payload = fake_tencent.calls[-1]["json"]
    assert [p["shots"] for p in payload] == [10, 10, 10]
    assert {p["device"] for p in payload} == {"20xmon?o=3"}

    fake_tencent.responses["task/submit"] = {"tasks": [{"err": "quota exceeded"}]}
    with pytest.raises(ValueError, match="All tasks submitted failed"):
        apis.submit_task(device="tencent::20xmon", circuit=[c])
    fake_tencent.responses["task/submit"] = {"err": "invalid token"}
    with pytest.raises(ValueError, match="invalid token"):
        apis.submit_task(circuit=c)


def test_tencent_task_details_contract(fake_tencent):
    fake_tencent.responses["task/detail"] = {
        "task": {
            "id": "t1",
            "state": "completed",
            "shots": 100,
            "device": "20xmon",
            "result": {"counts": {"00": 30, "11": 70}},
            "optimization": {"pairs": {"0": "3", "1": "4"}},
        }
    }
    t = apis.get_task("tencent::20xmon~~t1")
    assert t.details()["results"] == {"00": 30, "11": 70}
    assert t.get_logical_physical_mapping() == {0: 3, 1: 4}
    assert list(t.results().items()) == [("11", 70), ("00", 30)]
    assert fake_tencent.calls[-1]["json"] == {"id": "t1"}
    assert fake_tencent.calls[-1]["headers"]["Authorization"] == "Bearer ANY;0"

    fake_tencent.responses["task/detail"] = {
        "task": {"id": "t1", "state": "failed", "err": "chip offline"}
    }
    with pytest.raises(TaskFailed, match="chip offline"):
        t.results()
    fake_tencent.responses["task/detail"] = {"err": "no such task"}
    with pytest.raises(ValueError, match="no such task"):
        t.details()


@pytest.mark.parametrize("readout_cals, nbatches", [("api", [1]), ("system", [2, 1])])
def test_tencent_batch_expectation_ps_contract(
    fake_tencent, with_readout_errors, readout_cals, nbatches
):
    pytest.importorskip("qiskit")
    submitted = {}
    errors = {0: (0.02, 0.08), 1: (0.04, 0.12)}

    def submit(payload):
        tasks = []
        for p in payload:
            submitted[str(len(submitted))] = p
            tasks.append({"id": str(len(submitted) - 1)})
        return {"tasks": tasks}

    def detail(payload):
        p = submitted[payload["id"]]
        # counts are indexed by the classical bits, whatever qubits qiskit chose
        c = tc.Circuit.from_openqasm(p["source"], keep_measure_order=True)
        order = list(c.get_positional_logical_mapping().values())
        n = c._nqubits
        probs = np.real(tc.backend.numpy(c.probability())).reshape([2] * n)
        probs = probs.sum(axis=tuple(q for q in range(n) if q not in order))
        probs = probs.transpose([sorted(order).index(q) for q in order]).reshape(-1)
        probs = with_readout_errors(probs, order, errors)
        result = {
            format(i, "0%db" % len(order)): int(round(pr * p["shots"]))
            for i, pr in enumerate(probs)
            if pr > 1e-6
        }
        task = {"id": payload["id"], "state": "completed", "result": {"counts": result}}
        return {"task": {**p, **task}}

    fake_tencent.responses["task/submit"] = submit
    fake_tencent.responses["task/detail"] = detail
    fake_tencent.responses["device/detail"] = {
        "device": {
            "id": "2xmon",
            "qubits": 2,
            "links": [{"A": 0, "B": 1, "CZErrRate": 0.01}],
            "bits": [
                {
                    "Qubit": q,
                    "T1": 30.0,
                    "SingleQubitErrRate": 0.001,
                    "ReadoutF0Err": e0,
                    "ReadoutF1Err": e1,
                }
                for q, (e0, e1) in errors.items()
            ],
        }
    }
    c = tc.Circuit(2)
    c.h(0)
    c.x(1)
    r = wrapper.batch_expectation_ps(
        c,
        [[1, 0], [0, 3]],
        device="tencent::2xmon",
        shots=10000,
        readout_cals=readout_cals,
    )
    # the readout errors are mitigated on the measured physical qubits
    np.testing.assert_allclose(r, [1, -1], atol=1e-3)
    batches = fake_tencent.requests_to("task/submit")
    # X0 and Z1 are measured by one circuit, after the two calibration circuits
    # of the system calibration
    assert [len(b) for b in batches] == nbatches
    assert {(p["device"], p["shots"]) for b in batches for p in b} == {
        ("2xmon?o=0", 10000)
    }


def test_tencent_store_and_measurement_metadata(fake_tencent, tmp_path):
    # q1 is measured twice, so there are more classical bits than qubits
    c = tc.Circuit(2)
    c.x(0)
    c.measure_instruction(1)
    c.measure_instruction(0)
    c.h(1)
    c.measure_instruction(1)
    fake_tencent.responses["task/submit"] = {
        "tasks": [{"id": "t1"}, {"err": "quota exceeded"}]
    }
    fake_tencent.responses["task/detail"] = {
        "task": {
            "id": "t1",
            "state": "completed",
            "result": {"counts": {"010": 6, "011": 4}},
            "optimization": {"pairs": {"0": "3", "1": "4"}},
        }
    }
    s = store.TaskStore(tmp_path / "tasks")
    (t,) = apis.submit_task(device="tencent::20xmon", circuit=[c, c], store=s)
    source = fake_tencent.requests_to("task/submit")[0][0]["source"]
    assert "creg c[3];" in source and "measure q[1] -> c[2];" in source

    d = t.details(prettify=True)
    meta = d["tensorcircuit"]
    assert d["results"] == {"010": 6, "011": 4}
    assert meta["positional_logical_mapping"] == {0: 4, 1: 3, 2: 4}
    assert meta["archive"]["saved"]

    # the circuit rejected in the batch is certainly not submitted
    (accepted,) = s.find(device="tencent::20xmon")
    assert accepted["local_id"] == t.local_id and accepted["state"] == "completed"
    assert s.get(t.local_id)["tensorcircuit"] == {
        k: v for k, v in meta.items() if k != "archive"
    }

    # the raw details are queried for archival only once the state changed
    n_queries = len(fake_tencent.requests_to("task/detail"))
    assert n_queries == 2
    t.details(prettify=True)
    assert len(fake_tencent.requests_to("task/detail")) == n_queries + 1


def test_tencent_source_takes_precedence_in_records(fake_tencent, tmp_path):
    fake_tencent.responses["task/submit"] = {"tasks": [{"id": "t1"}, {"id": "t2"}]}
    s = store.TaskStore(tmp_path / "tasks")
    sources = [tc.Circuit(1).to_openqasm(), tc.Circuit(2).to_openqasm()]
    ts = apis.submit_task(
        device="tencent::20xmon", circuit=tc.Circuit(3), source=sources, store=s
    )
    assert [t.id_ for t in ts] == ["t1", "t2"]
    records = [s.get(t.local_id) for t in ts]
    assert [r["request"]["source"] for r in records] == sources
    assert all(r["request"]["circuit"] is None for r in records)
    assert [t.prepared for t in ts] == [
        {"positional_logical_mapping": None, "source": s} for s in sources
    ]


def test_tencent_frontend_compiling_mapping_end_to_end(fake_tencent, tmp_path):
    pytest.importorskip("qiskit")
    # q0 = q1 = 1 and q2 = 0, routing cx(0, 2) on a line needs a swap;
    # q1 is not measured and q2 is measured twice
    c = tc.Circuit(3)
    c.x(0)
    c.x(1)
    c.x(2)
    c.cx(0, 2)
    c.measure_instruction(2)
    c.measure_instruction(0)
    c.measure_instruction(2)
    ideal = {0: "1", 1: "1", 2: "0"}
    pairs = {0: 5, 1: 3, 2: 4}

    def run_sent():
        # run the circuit sent to the provider, counts are in classical bit order
        source = fake_tencent.requests_to("task/submit")[0]["source"]
        measures = {
            int(b): int(q)
            for q, b in re.findall(r"measure q\[(\d+)\] -> \w+\[(\d+)\];", source)
        }
        unitary = "\n".join(
            line for line in source.split("\n") if not line.startswith("measure")
        )
        state = tc.Circuit.from_openqasm(unitary).state()
        bits = format(int(np.argmax(np.abs(state))), "03b")
        return "".join(bits[measures[i]] for i in sorted(measures)), measures

    fake_tencent.responses["task/submit"] = {"tasks": [{"id": "t1"}]}
    fake_tencent.responses["task/detail"] = lambda json: {
        "task": {
            "id": "t1",
            "state": "completed",
            "result": {"counts": {run_sent()[0]: 100}},
            "optimization": {"pairs": {str(k): str(v) for k, v in pairs.items()}},
        }
    }
    s = store.TaskStore(tmp_path / "tasks")
    t = apis.submit_task(
        device="tencent::20xmon",
        circuit=c,
        compiled=True,
        compiled_options={
            "basis_gates": ["x", "h", "rz", "cx"],
            "optimization_level": 0,
            "coupling_map": [[0, 1], [1, 0], [1, 2], [2, 1]],
            "initial_layout": [0, 1, 2],
            "seed_transpiler": 42,
        },
        store=s,
    )
    result, measures = run_sent()
    assert fake_tencent.requests_to("task/submit")[0]["source"].count("cx") >= 4
    for task in [t, s.task(t.local_id)]:
        d = task.details()
        meta = d["tensorcircuit"]
        assert d["results"] == {result: 100}
        # positions are fixed by the submitted circuit, compiling changes qubits only
        positional_logical = c.get_positional_logical_mapping()
        assert positional_logical == {0: 2, 1: 0, 2: 2}
        assert {i: result[i] for i in range(3)} == {
            i: ideal[q] for i, q in positional_logical.items()
        }
        assert meta["positional_logical_mapping"] == {
            i: pairs[q] for i, q in measures.items()
        }
        assert set(meta) == {
            "schema_version",
            "positional_logical_mapping",
            "archive",
        }


def lost_request(json):
    requests = pytest.importorskip("requests")
    raise requests.exceptions.Timeout("read timed out")


def test_tencent_paid_submission_sent_once(fake_tencent, tmp_path):
    pytest.importorskip("qiskit")
    c = tc.Circuit(2)
    c.x(0)
    c.measure_instruction(0)
    s = store.TaskStore(tmp_path / "tasks")
    fake_tencent.responses["task/submit"] = lost_request
    with pytest.raises(SubmissionUnknown, match="read timed out"):
        # the frontend compiling swaps the qubits
        apis.submit_task(
            device="tencent::20xmon",
            circuit=c,
            compiled=True,
            compiled_options={
                "basis_gates": ["x", "cx"],
                "optimization_level": 0,
                "coupling_map": [[0, 1], [1, 0]],
                "initial_layout": [1, 0],
            },
            store=s,
        )
    (call,) = fake_tencent.calls
    assert call["endpoint"] == "task/submit" and call["timeout"] == 60
    assert "measure q[1] -> c[0];" in call["json"]["source"]
    # the task may run on the provider, so its record is kept with the mapping
    # of the sent circuit
    (row,) = s.find(submission_status="unknown")
    r = s.get(row["local_id"])
    assert r["remote_id"] is None and "read timed out" in r["submission"]["error"]
    assert r["prepared"]["positional_logical_mapping"] == {0: 1}
    assert "measure q[1] -> c[0];" in r["prepared"]["source"]

    # bind the task found on the provider; queries are still retried
    responses = [
        {
            "task": {
                "id": "t1",
                "state": "completed",
                "result": {"counts": {"1": 10}},
                "optimization": {"pairs": {"0": "3", "1": "4"}},
            }
        }
    ]
    fake_tencent.responses["task/detail"] = lambda json: (
        responses.pop() if len(fake_tencent.calls) > 2 else lost_request(json)
    )
    t = s.bind(row["local_id"], "t1")
    assert str(t) == "tencent::20xmon~~t1"
    assert t.details()["tensorcircuit"]["positional_logical_mapping"] == {0: 4}
    assert len(fake_tencent.requests_to("task/detail")) == 2
    r = s.get(t.local_id)
    assert r["submission"]["status"] == "submitted" and r["state"] == "completed"

    # explicitly rejected, so certainly not submitted
    fake_tencent.responses["task/submit"] = {"err": "invalid token"}
    with pytest.raises(ValueError, match="invalid token"):
        apis.submit_task(device="tencent::20xmon", circuit=c, store=s)
    assert [x["local_id"] for x in s.find()] == [t.local_id]


@pytest.mark.parametrize(
    "response",
    [
        [],
        {"tasks": []},
        {"tasks": [{"id": "t1"}]},
        {"tasks": [{"id": "t1"}, {}]},
        {"tasks": [{"err": "quota exceeded"}, "t2"]},
    ],
)
def test_tencent_unclear_answer_keeps_tasks_unknown(fake_tencent, tmp_path, response):
    s = store.TaskStore(tmp_path / "tasks")
    fake_tencent.responses["task/submit"] = response
    circuits = [tc.Circuit(1), tc.Circuit(1)]
    with pytest.raises(SubmissionUnknown):
        apis.submit_task(device="tencent::20xmon", circuit=circuits, store=s)
    assert len(fake_tencent.calls) == 1
    rows = s.find(submission_status="unknown")
    assert len(rows) == 2
    for row in rows:
        assert json.dumps(response) in s.get(row["local_id"])["submission"]["error"]


def test_tencent_resubmit_recorded_with_parent(fake_tencent, tmp_path):
    c = tc.Circuit(2)
    c.measure_instruction(1)
    s = store.TaskStore(tmp_path / "tasks")
    fake_tencent.responses["task/submit"] = {"tasks": [{"id": "t1"}]}
    fake_tencent.responses["task/start"] = {"tasks": [{"id": "t2"}]}
    t = apis.submit_task(device="tencent::20xmon", circuit=c, shots=100, store=s)
    t2 = t.resubmit()
    assert str(t2) == "tencent::20xmon~~t2"
    call = fake_tencent.calls[-1]
    assert call["endpoint"] == "task/start" and call["timeout"] == 60
    assert call["json"] == {"id": "t1"}
    parent, r = s.get(t.local_id), s.get(t2.local_id)
    assert r["parent_id"] == t.local_id and r["remote_id"] == "t2"
    assert r["request"] == dict(parent["request"], resubmit_of="t1")
    assert r["prepared"] == parent["prepared"] == t2.prepared
    assert t2.prepared["positional_logical_mapping"] == {0: 1}

    # the parent is found by the remote id
    fake_tencent.responses["task/start"] = lost_request
    with pytest.raises(SubmissionUnknown):
        apis.resubmit_task("tencent::20xmon~~t1", store=s)
    assert len(fake_tencent.requests_to("task/start")) == 2
    (row,) = s.find(submission_status="unknown")
    r = s.get(row["local_id"])
    assert r["parent_id"] == t.local_id and r["prepared"] == parent["prepared"]
