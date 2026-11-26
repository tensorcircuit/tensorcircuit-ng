from collections import OrderedDict
from functools import partial
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import numpy as np
import networkx as nx

import tensorcircuit as tc
from tensorcircuit.noisemodel import NoiseConf
from tensorcircuit.results import qem
from tensorcircuit.results.qem import (
    zne_option,
    apply_zne,
    dd_option,
    apply_dd,
    apply_rc,
)
from tensorcircuit.results.qem import benchmark_circuits, qem_methods


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_benchmark_circuits(backend):
    # QAOA
    graph = [(2, 0), (0, 3), (1, 2)]
    weight = [1] * len(graph)
    params = np.array([[1, 1]])

    _ = benchmark_circuits.QAOA_circuit(graph, weight, params)

    # mirror circuit
    # return circuit and ideal counts {"01000":1}
    _, _ = benchmark_circuits.mirror_circuit(
        depth=5, two_qubit_gate_prob=1, connectivity_graph=nx.complete_graph(3), seed=20
    )

    # GHZ circuit
    _ = benchmark_circuits.generate_ghz_circuit(10)

    # Werner-state with linear complexity
    #  {'1000': 0.25, '0100': 0.25, '0010': 0.25, '0001': 0.25}
    _ = benchmark_circuits.generate_w_circuit(5)

    # RB cirucit
    _ = benchmark_circuits.generate_rb_circuits(2, 7)[0]


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_zne(backend):
    c = tc.Circuit(2)
    for _ in range(3):
        c.rx(range(2), theta=0.4)

    error1 = tc.channels.generaldepolarizingchannel(0.01, 1)
    noise_conf = NoiseConf()
    noise_conf.add_noise("rx", error1)

    def execute(circuit):
        value = circuit.expectation_ps(z=[0], noise_conf=noise_conf, nmc=10000)
        return value

    random_state = np.random.RandomState(0)
    noise_scaling_function = partial(
        zne_option.scaling.fold_gates_at_random,
        # fidelities = {"single": 1.0},
        random_state=random_state,
    )
    factory = zne_option.inference.PolyFactory(scale_factors=[1, 3, 5], order=1)
    # factory = zne_option.inference.ExpFactory(scale_factors=[1,1.5,2],asymptote=0.)
    # factory = zne_option.inference.RichardsonFactory(scale_factors=[1,1.5,2])
    # factory = zne_option.inference.AdaExpFactory(steps=5, asymptote=0.)

    result = apply_zne(
        circuit=c,
        executor=execute,
        factory=factory,
        scale_noise=noise_scaling_function,
        num_to_average=1,
    )

    ideal_value = c.expectation_ps(z=[0])
    mit_value = result

    np.testing.assert_allclose(ideal_value, mit_value, atol=4e-2)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dd(backend):
    c = tc.Circuit(2)
    for _ in range(3):
        c.rx(range(2), theta=0.4)

    def execute(circuit):
        value = circuit.expectation_ps(z=[0])
        return value

    def execute2(circuit):
        key = tc.backend.get_random_state(42)
        count = circuit.sample(
            batch=1000, allow_state=True, format_="count_dict_bin", random_generator=key
        )
        return count

    ideal_value = c.expectation_ps(z=[0])

    mit_value, dd_circuit = apply_dd(
        circuit=c,
        executor=execute,
        rule=["X", "X"],
        rule_args={"spacing": -1},
        full_output=True,
        ignore_idle_qubit=True,
        fulldd=False,
    )
    assert np.isfinite(mit_value)
    np.testing.assert_allclose(ideal_value, mit_value, atol=1e-5)
    assert dd_circuit.circuit_param["nqubits"] == c.circuit_param["nqubits"]

    mit_count, dd_circuit2 = apply_dd(
        circuit=c,
        executor=execute2,
        rule=dd_option.rules.xyxy,
        rule_args={"spacing": -1},
        full_output=True,
        ignore_idle_qubit=True,
        fulldd=True,
        iscount=True,
    )
    assert isinstance(mit_count, dict)
    assert len(dd_circuit2.to_qir()) > 0

    # wash circuit based on use_qubits and washout iden gates
    pruned = qem.prune_ddcircuit(c, qlist=list(range(c.circuit_param["nqubits"])))
    assert pruned.circuit_param["nqubits"] == c.circuit_param["nqubits"]


def _initial_state_circuit(kind):
    K = tc.backend
    psi = np.kron(np.array([1, 1j, 2, -0.5j]) / 2.5, [0, 1])
    kwargs = {"split": {"max_singular_values": 4}}
    circuit_type = tc.Circuit
    if kind == "dense":
        kwargs["inputs"] = K.convert_to_tensor(psi)
    elif kind == "mps":
        source = tc.Circuit(3, inputs=K.convert_to_tensor(psi))
        kwargs["mps_inputs"] = source.quvector()
    elif kind == "tensors":
        kwargs["tensors"] = [
            K.reshape(K.convert_to_tensor(v), [1, 2, 1])
            for v in [np.array([1, 1j]) / np.sqrt(2), [0, 1], [0, 1]]
        ]
    elif kind == "mixed":
        circuit_type = tc.DMCircuit
        kwargs["dminputs"] = K.convert_to_tensor(
            0.7 * np.outer(psi, psi.conj()) + 0.3 * np.eye(8) / 8
        )
    c = circuit_type(3, **kwargs)
    c.z(0)
    for _ in range(4):
        c.z(1)
    c.ry(0, theta=0.31)
    return c


def _density_matrix(c):
    if c.is_dm:
        return tc.backend.numpy(c.densitymatrix())
    state = tc.backend.numpy(c.state())
    return np.outer(state, state.conj())


@pytest.mark.parametrize("method", ["dd", "zne"])
@pytest.mark.parametrize("kind", ["default", "dense", "mps", "tensors", "mixed"])
def test_qem_preserves_initial_state(npb, kind, method):
    c = _initial_state_circuit(kind)
    expected = _density_matrix(c)

    def execute(rebuilt):
        assert type(rebuilt) is type(c)
        assert rebuilt.circuit_param["split"] == c.circuit_param["split"]
        np.testing.assert_allclose(_density_matrix(rebuilt), expected, atol=1e-6)
        return float(tc.backend.numpy(tc.backend.real(rebuilt.expectation_ps(z=[2]))))

    if method == "dd":
        result, rebuilt = apply_dd(c, execute, rule=["X", "X"], full_output=True)
        execute(rebuilt)
        execute(
            qem.prune_ddcircuit(qem.add_dd(c, dd_option.rules.xx), qem.used_qubits(c))
        )
    else:
        factory = zne_option.inference.RichardsonFactory(scale_factors=[1.0, 3.0])
        result = apply_zne(c, execute, factory=factory)
    np.testing.assert_allclose(result, execute(c), atol=1e-6)
    np.testing.assert_allclose(_density_matrix(c), expected, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_rc(backend):
    c = tc.Circuit(2)
    for _ in range(3):
        c.rx(range(2), theta=0.4)
        c.cnot(0, 1)

    def execute(circuit):
        value = circuit.expectation_ps(z=[0])
        return value

    def execute2(circuit):
        key = tc.backend.get_random_state(42)
        count = circuit.sample(
            batch=1000, allow_state=True, format_="count_dict_bin", random_generator=key
        )
        return count

    ideal_value = c.expectation_ps(z=[0])

    mit_value, circuit_list = apply_rc(
        circuit=c, executor=execute, num_to_average=6, simplify=False
    )
    assert np.isfinite(mit_value)
    np.testing.assert_allclose(ideal_value, mit_value, atol=1e-5)
    assert len(circuit_list) == 6
    assert circuit_list[0].circuit_param["nqubits"] == c.circuit_param["nqubits"]

    mit_count, circuit_list2 = apply_rc(
        circuit=c, executor=execute2, num_to_average=6, simplify=True, iscount=True
    )
    assert isinstance(mit_count, dict)
    assert len(circuit_list2) == 6

    # generate a circuit with rc
    rc_c = qem.rc_circuit(c)
    assert rc_c.circuit_param["nqubits"] == c.circuit_param["nqubits"]
    assert len(rc_c.to_qir()) > 0


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
@pytest.mark.parametrize(
    "double_precision,custom,warm_angle,separate_calls",
    [
        (False, False, 0.0, False),
        (False, False, np.pi / 2, True),
        (True, True, 0.0, True),
        (True, True, np.pi / 2, False),
    ],
)
def test_rc_cache_matrix_equivalence(
    backend, double_precision, custom, warm_angle, separate_calls, request, monkeypatch
):
    if double_precision:
        request.getfixturevalue("highp")
    monkeypatch.setattr(qem_methods, "candidate_dict", OrderedDict())
    paulis = [np.asarray(g.tensor) for g in tc.gates.pauli_gates]

    def add_gate(circuit, angle, indices):
        if custom:
            matrix = np.diag(np.exp(-0.5j * angle * np.array([1, -1, -1, 1])))
            h = np.kron(np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.eye(2))
            circuit.unitary(*indices, unitary=h @ matrix @ h, name="shared")
        else:
            circuit.rzz(*indices, theta=angle)
        return tc.backend.numpy(
            tc.backend.reshapem(circuit.to_qir()[-1]["gate"].tensor)
        )

    warm = tc.Circuit(2)
    first = add_gate(warm, warm_angle, (1, 0))
    target = tc.Circuit(2) if separate_calls else warm
    second = add_gate(target, 0.37, (0, 1))
    matrices = iter([first, second])

    def choose(candidates):
        matrix = next(matrices)
        for a, b, c, d in candidates:
            twirled = (
                np.kron(paulis[c], paulis[d]) @ matrix @ np.kron(paulis[a], paulis[b])
            )
            phase = np.trace(matrix.conj().T @ twirled) / 4
            np.testing.assert_allclose(abs(phase), 1, atol=2e-6)
            np.testing.assert_allclose(twirled, phase * matrix, atol=2e-6)
        return candidates[1]

    monkeypatch.setattr(qem_methods, "choice", choose)
    if separate_calls:
        qem.rc_circuit(warm)
    expected = tc.backend.numpy(target.matrix())
    actual = tc.backend.numpy(qem.rc_circuit(target).matrix())
    phase = np.trace(expected.conj().T @ actual) / 4
    np.testing.assert_allclose(abs(phase), 1, atol=3e-6)
    np.testing.assert_allclose(actual, phase * expected, atol=3e-6)


def test_rc_cache_reuse_and_eviction(npb, monkeypatch):
    cache = OrderedDict()
    monkeypatch.setattr(qem_methods, "candidate_dict", cache)
    monkeypatch.setattr(qem_methods, "_RC_CACHE_MAXSIZE", 2)
    circuit = tc.Circuit(2)
    circuit.rzz(0, 1, theta=0.0)
    circuit.rzz(0, 1, theta=0.37)
    qem.rc_circuit(circuit)
    keys = list(cache)
    candidates = list(cache.values())
    apply_rc(circuit, lambda c: 0.0, num_to_average=3, simplify=False)
    assert list(cache) == keys
    assert all(cache[key] is value for key, value in zip(keys, candidates))

    warm = tc.Circuit(2)
    warm.rzz(0, 1, theta=0.0)
    qem.rc_circuit(warm)
    assert list(cache) == keys[::-1]
    cold = tc.Circuit(2)
    cold.rzz(0, 1, theta=0.71)
    qem.rc_circuit(cold)
    assert len(cache) == 2
    assert keys[0] in cache and keys[1] not in cache


def test_rc_candidates_follow_circuit_order(npb, monkeypatch):
    source = tc.Circuit(2)
    source.cnot(0, 1)
    source.h(0)
    circuit = tc.Circuit(2)
    circuit.unitary(0, 1, unitary=source.matrix(), name="composite")
    candidates = qem_methods.rc_candidates(circuit.to_qir()[0]["gate"])
    assert (1, 0, 3, 1) in candidates
    assert (3, 1, 1, 0) not in candidates

    monkeypatch.setattr(qem_methods, "candidate_dict", OrderedDict())
    choices = iter(candidates)
    monkeypatch.setattr(qem_methods, "choice", lambda _: next(choices))
    expected = tc.backend.numpy(circuit.matrix())
    for _ in candidates:
        actual = tc.backend.numpy(qem.rc_circuit(circuit).matrix())
        phase = np.trace(expected.conj().T @ actual) / 4
        np.testing.assert_allclose(abs(phase), 1, atol=1e-6)
        np.testing.assert_allclose(actual, phase * expected, atol=1e-6)


def test_rc_candidates_reject_near_identity_twirls(npb):
    circuit = tc.Circuit(2)
    circuit.rzz(0, 1, theta=2e-6)
    candidates = qem_methods.rc_candidates(circuit.to_qir()[0]["gate"])
    assert (0, 1, 0, 1) not in candidates


@pytest.mark.parametrize("kind", ["default", "dense", "mps", "tensors", "mixed"])
def test_rc_preserves_initial_state(npb, kind, monkeypatch):
    circuit = _initial_state_circuit(kind)
    circuit.cnot(0, 2)
    expected = _density_matrix(circuit)

    def choose_identity(candidates):
        assert (0, 0, 0, 0) in candidates
        return (0, 0, 0, 0)

    def check_rebuilt(rebuilt):
        assert type(rebuilt) is type(circuit)
        assert rebuilt.circuit_param["split"] == circuit.circuit_param["split"]
        np.testing.assert_allclose(_density_matrix(rebuilt), expected, atol=1e-6)
        return 0.0

    monkeypatch.setattr(qem_methods, "choice", choose_identity)
    check_rebuilt(qem.rc_circuit(circuit))
    for simplify in (False, True):
        _, rebuilt = apply_rc(circuit, check_rebuilt, simplify=simplify)
        check_rebuilt(rebuilt[0])


def test_rc_keeps_instructions(npb):
    def sequence(circuit):
        merged = circuit.to_qir(enable_instruction=True)
        return [
            (d["name"], list(d["index"]))
            for d in merged
            if d["name"] in ["cnot", "measure", "reset", "barrier"]
        ]

    c = tc.Circuit(3)
    c.h(0)
    c.cnot(0, 1)
    c.measure_instruction(1)
    c.reset_instruction(1)
    c.barrier_instruction(0, 1, 2)
    c.cnot(1, 2)
    c.measure_instruction(2)
    c.measure_instruction(0)
    expected = sequence(c)
    rc = qem.rc_circuit(c)
    assert sequence(rc) == expected
    pruned = qem.prune_ddcircuit(rc, [0, 1, 2])
    assert sequence(pruned) == expected
    assert rc.to_qiskit(enable_instruction=True).count_ops()["measure"] == 3
    for rebuilt in (rc, pruned):
        rebuilt.measure_instruction(1)
        assert rebuilt._extra_qir[-1]["record_index"] == 3
    with pytest.raises(ValueError, match="at the end"):
        apply_rc(c, lambda _: 0.0)

    c = tc.Circuit(2)
    c.cnot(0, 1)
    c.measure_instruction(1)
    _, rebuilt = apply_rc(c, lambda _: 0.0, simplify=True)
    assert sequence(rebuilt[0]) == [("cnot", [0, 1]), ("measure", [1])]
    rebuilt[0].measure_instruction(0)
    assert rebuilt[0]._extra_qir[-1]["record_index"] == 1


def _matrix_up_to_phase(c, expected):
    actual = tc.backend.numpy(c.matrix())
    phase = np.trace(expected.conj().T @ actual) / len(expected)
    np.testing.assert_allclose(abs(phase), 1, atol=1e-6)
    np.testing.assert_allclose(actual, phase * expected, atol=1e-6)


def test_mitigators_expand(npb, highp):
    M = qem.mitigators
    c = tc.Circuit(3)
    c.h(0)
    c.rz(1, theta=0.4)
    c.cz(0, 1)
    c.rz(0, theta=-0.3)
    c.cnot(1, 2)
    c.rz(1, theta=-np.pi / 2)
    c.measure_instruction(2)
    expected = tc.backend.numpy(c.matrix())

    def names(ci):
        return [d["name"] for d in ci.to_qir(enable_instruction=True)]

    twirl = M.Twirl(8, seed=1)
    cs = twirl.expand(c)
    assert len({ci.to_openqasm() for ci in cs}) > 1
    assert [ci.to_openqasm() for ci in cs] == [
        ci.to_openqasm() for ci in twirl.expand(c)
    ]
    for ci in cs:
        _matrix_up_to_phase(ci, expected)
        assert names(ci)[-1] == "measure"

    cdr = M.CDR(3, sigma=0.3, seed=2)
    cs = cdr.expand(c)
    assert len(cs) == 4 and cs[0] is c
    thetas = [
        [float(d["parameters"]["theta"]) for d in ci.to_qir() if d["name"] == "rz"]
        for ci in cs
    ]
    assert all(names(ci) == names(c) for ci in cs)
    assert np.all(np.abs(np.diff(thetas, axis=0))[:, :2] > 0)
    # the Clifford angles are kept, not to change the pulses after compiling
    np.testing.assert_allclose(np.array(thetas)[:, 2], -np.pi / 2)
    # the rz gates in common are perturbed the same, e.g. in another basis
    c2 = tc.Circuit.from_qir(c.to_qir(enable_instruction=True), c.circuit_param)
    c2.rz(2, theta=0.1)
    for ci, cj in zip(cs[1:], cdr.expand(c2)[1:]):
        np.testing.assert_allclose(
            [d["parameters"]["theta"] for d in cj.to_qir() if d["name"] == "rz"][:3],
            [d["parameters"]["theta"] for d in ci.to_qir() if d["name"] == "rz"],
        )

    cs = M.Fold([1, 3]).expand(c)
    _matrix_up_to_phase(cs[1], expected)
    assert names(cs[1]).count("cz") == 3 and names(cs[1]).count("cnot") == 3
    assert names(cs[1]).count("barrier") == 4

    chain = M.Chain(M.CDR(2, seed=3), M.Twirl(4, seed=4))
    nested = chain.expand(c)
    assert chain.shape == (3, 4)
    assert np.shape(tc.backend.tree_map(lambda _: 0, nested)) == (3, 4)


def test_mitigators_combine():
    M = qem.mitigators
    rng = np.random.default_rng(0)
    ideal = rng.uniform(-1, 1, size=[2, 5, 3])
    # two terms with their own Pauli noise rescaling, and the twirls averaged
    scaled = np.array([0.7, 0.4])[:, None, None] * ideal
    noisy = 0.05 + scaled + rng.normal(scale=0.01, size=[2, 5, 1]) * [1, -1, 0]
    chain = M.Chain(M.CDR(4, intercept=True), M.Twirl(3))
    np.testing.assert_allclose(
        chain.combine(noisy, ideal), np.mean(ideal, -1)[:, 0], atol=1e-10
    )
    np.testing.assert_allclose(
        M.CDR(4).combine(scaled[..., 0], ideal[..., 0]),
        ideal[:, 0, 0],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        M.Fold([1, 3, 5], order=2).combine([[0.9, 0.7, 0.3]], None), [0.925]
    )


def test_cdr_uninformative_training(caplog):
    M = qem.mitigators
    # a term vanishing by symmetry, a constant term and an informative term
    ideal = np.array(
        [[1e-17, -2e-17, 3e-17], [0.5, 0.5, 0.5], [0.6, 0.3, -0.2]], dtype=float
    )
    noisy = np.array([[0.02, 0.01, -0.01], [0.4, 0.4, 0.4], [0.48, 0.24, -0.16]])
    noisy = noisy + 0.03
    # a term with no signal left, uncorrelated with the varying ideal values
    ideal = np.concatenate([ideal, [[0.6, 0.3, -0.2]]])
    noisy = np.concatenate([noisy, [[0.1, -0.05, 0.05]]])
    for intercept in [False, True]:
        r = M.CDR(2, intercept=intercept).combine(noisy, ideal)
        assert np.all(np.isfinite(r))
        # the vanishing and no signal terms kept unscaled, the constant term
        # fitted without ``a``
        np.testing.assert_allclose(r[[0, 1, 3]], [0.05, 0.5, 0.1])
    np.testing.assert_allclose(r[2], 0.6, atol=1e-10)
    assert "[[0], [3]]" in caplog.text
    with pytest.raises(ValueError, match="intercept"):
        M.CDR(1, intercept=True)
