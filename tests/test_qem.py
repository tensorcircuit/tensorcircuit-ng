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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("double_precision", [False, True])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("warm_angle", [0.0, np.pi / 2])
@pytest.mark.parametrize("separate_calls", [False, True])
def test_rc_cache_matrix_equivalence(
    backend, double_precision, custom, warm_angle, separate_calls, request, monkeypatch
):
    if double_precision:
        request.getfixturevalue("highp")
    monkeypatch.setattr(qem_methods, "candidate_dict", {})
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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("simplify", [False, True])
def test_rc_cache_noiseless_expectation(backend, simplify, monkeypatch):
    monkeypatch.setattr(qem_methods, "candidate_dict", {})
    monkeypatch.setattr(qem_methods, "choice", lambda candidates: candidates[1])
    warm = tc.Circuit(2)
    warm.rzz(0, 1, theta=0)
    qem.rc_circuit(warm)
    target = tc.Circuit(2)
    target.h(0)
    target.rzz(0, 1, theta=np.pi / 4)
    result, circuits = apply_rc(
        target,
        executor=lambda circuit: circuit.expectation_ps(y=[0]),
        num_to_average=3,
        simplify=simplify,
    )
    np.testing.assert_allclose(result, np.sqrt(0.5), atol=2e-6)
    expected = tc.backend.numpy(target.state())
    for circuit in circuits:
        actual = tc.backend.numpy(circuit.state())
        np.testing.assert_allclose(abs(np.vdot(expected, actual)) ** 2, 1, atol=3e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_rc_cache_reuses_identical_matrix(backend, monkeypatch):
    monkeypatch.setattr(qem_methods, "candidate_dict", {})
    original = qem_methods.rc_candidates
    calls = []

    def counted(gate):
        calls.append(gate)
        return original(gate)

    monkeypatch.setattr(qem_methods, "rc_candidates", counted)
    circuit = tc.Circuit(2)
    circuit.rzz(0, 1, theta=0.37)
    circuit.rzz(1, 0, theta=0.37)
    qem.rc_circuit(circuit)
    qem.rc_circuit(circuit)
    assert len(calls) == 1
