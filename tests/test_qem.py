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
from tensorcircuit.results.qem import benchmark_circuits


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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("method", ["dd", "zne"])
def test_qem_custom_initial_state(backend, method):
    c = tc.Circuit(1, inputs=tc.backend.convert_to_tensor([0, 1]))
    c.z(0)

    def execute(circuit):
        return float(tc.backend.numpy(tc.backend.real(circuit.expectation_ps(z=[0]))))

    if method == "dd":
        result = apply_dd(c, execute, rule=["X", "X"])
    else:
        factory = zne_option.inference.RichardsonFactory(scale_factors=[1.0, 3.0])
        result = apply_zne(c, execute, factory=factory)
    np.testing.assert_allclose(execute(c), -1.0, atol=1e-6)
    np.testing.assert_allclose(result, -1.0, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("kind", ["default", "dense", "mps", "tensors", "mixed"])
@pytest.mark.parametrize("fulldd,ignore_idle", [(False, True), (True, False)])
def test_dd_preserves_initial_state(backend, kind, fulldd, ignore_idle):
    c = _initial_state_circuit(kind)
    expected = _density_matrix(c)

    def execute(rebuilt):
        assert type(rebuilt) is type(c)
        assert rebuilt.circuit_param["nqubits"] == 3
        assert rebuilt.circuit_param["split"] == c.circuit_param["split"]
        np.testing.assert_allclose(_density_matrix(rebuilt), expected, atol=1e-6)
        return float(tc.backend.numpy(tc.backend.real(rebuilt.expectation_ps(z=[2]))))

    added = qem.add_dd(c, dd_option.rules.xx)
    execute(added)
    execute(qem.prune_ddcircuit(added, qem.used_qubits(c)))
    result, rebuilt = apply_dd(
        c,
        execute,
        rule=["X", "X"],
        full_output=True,
        fulldd=fulldd,
        ignore_idle_qubit=ignore_idle,
    )
    np.testing.assert_allclose(result, execute(c), atol=1e-6)
    execute(rebuilt)
    np.testing.assert_allclose(_density_matrix(c), expected, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("kind", ["default", "dense", "mps", "tensors", "mixed"])
def test_zne_preserves_initial_state(backend, kind):
    c = _initial_state_circuit(kind)
    expected = _density_matrix(c)
    executed = []

    def execute(rebuilt):
        assert type(rebuilt) is type(c)
        assert rebuilt.circuit_param["nqubits"] == 3
        assert rebuilt.circuit_param["split"] == c.circuit_param["split"]
        np.testing.assert_allclose(_density_matrix(rebuilt), expected, atol=1e-6)
        executed.append(rebuilt)
        return float(tc.backend.numpy(tc.backend.real(rebuilt.expectation_ps(z=[2]))))

    factory = zne_option.inference.RichardsonFactory(scale_factors=[1.0, 3.0, 5.0])
    result = apply_zne(c, execute, factory=factory)
    assert len(executed) == 3
    assert len(executed[-1].to_qir()) > len(c.to_qir())
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
