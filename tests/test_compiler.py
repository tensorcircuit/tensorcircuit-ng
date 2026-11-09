import sys
import os
import pytest
import numpy as np
import networkx as nx
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_qsikit_compiler():
    try:
        import qiskit as _
    except ImportError:
        pytest.skip("qiskit is not installed")

    from tensorcircuit.compiler.qiskit_compiler import qiskit_compile

    c = tc.Circuit(2)
    c.x(1)
    c.cx(0, 1)

    c1, info = qiskit_compile(
        c,
        info=None,
        output="qasm",
        compiled_options={
            "basis_gates": ["cz", "rz", "h"],
            "optimization_level": 3,
            "coupling_map": [[0, 2], [2, 0], [1, 0], [0, 1]],
        },
    )
    assert "cz" in c1
    print(info["logical_physical_mapping"])

    c = tc.Circuit(2)
    c.x(1)
    c.cx(0, 1)
    c.measure_instruction(1)
    c1, info = qiskit_compile(
        c,
        info=None,
        output="tc",
        compiled_options={
            "basis_gates": ["cx", "rz", "h"],
            "optimization_level": 3,
            "coupling_map": [[0, 2], [2, 0], [1, 0], [0, 1]],
            "initial_layout": [1, 2],
        },
    )
    for inst in c1.to_qir():
        if inst["name"] == "h":
            assert inst["index"][0] == 2
    print(c1.draw())
    assert info["logical_physical_mapping"][1] in [0, 2]
    print(info)
    c2, info2 = qiskit_compile(
        c1,
        info=info,
        output="tc",
        compiled_options={
            "basis_gates": ["cx", "rz", "h"],
            "optimization_level": 3,
            "coupling_map": [[0, 2], [2, 0], [1, 0], [0, 1]],
        },
    )
    assert info2["positional_logical_mapping"] == {0: 1}
    print(c2.draw())


def test_qiskit_compile_positions_follow_classical_bits():
    qiskit = pytest.importorskip("qiskit")
    from tensorcircuit.compiler.qiskit_compiler import qiskit_compile

    qc = qiskit.QuantumCircuit(3, 2)
    qc.x(0)
    qc.measure(0, 1)
    qc.measure(2, 0)
    c1, info = qiskit_compile(
        qc,
        output="tc",
        compiled_options={
            "basis_gates": ["x", "h", "rz", "cx"],
            "coupling_map": [[0, 1], [1, 0], [1, 2], [2, 1]],
            "initial_layout": [2, 1, 0],
        },
    )
    assert info["positional_logical_mapping"] == {1: 0, 0: 2}
    # the compiled circuit keeps the classical bits on the compiled qubits
    assert c1.get_positional_logical_mapping() == {1: 2, 0: 0}


def test_composed_compiler():
    from tensorcircuit.compiler import DefaultCompiler

    c = tc.Circuit(3)
    c.rx(0)
    c.cx(0, 1)
    c.cz(1, 0)
    c.rxx(0, 2, theta=0.2)
    c.measure_instruction(2)
    c.measure_instruction(0)
    default_compiler = DefaultCompiler()
    c1, info = default_compiler(c)
    print(c1.draw())
    assert c1.gate_count_by_condition(lambda qir: qir["name"] == "cnot") == 3
    assert info["positional_logical_mapping"][0] == 2

    default_compiler = DefaultCompiler(
        {
            "basis_gates": ["h", "rz", "cz"],
            "optimization_level": 2,
            "coupling_map": [[0, 1], [1, 2]],
        }
    )

    c1, info = default_compiler(c)
    assert c1.gate_count_by_condition(lambda qir: qir["name"] == "cnot") == 0
    print(info)


def test_replace_r():
    c = tc.Circuit(3)
    c.rz(0, theta=0.1)
    c.cx(0, 2)
    c.ry(1)
    c.rxx(1, 0, theta=0.2)
    c.rx(0, theta=3.9)
    c.ry(1, theta=-0.2)
    c.rzz(1, 0, theta=-0.3)
    c.ryy(1, 0, theta=-0.6)
    c.rx(2)

    print(c.draw())
    c1 = tc.compiler.simple_compiler.replace_r(c)
    print(c1.draw())
    np.testing.assert_allclose(c.matrix(), c1.matrix(), atol=1e-5)


def test_default_compiler():
    c = tc.Circuit(3)
    c.cx(0, 1)
    c.rx(0, theta=1e-5)
    c.x(1)
    c.y(1)
    c.z(1)
    c.h(1)
    c.cz(2, 0)
    c.h(1)
    c.cz(2, 0)
    c.s(2)
    c.sd(2)
    c.s(2)
    c.s(2)
    c.y(2)
    c.ry(2, theta=0.1)
    c.t(2)
    c.td(2)
    c.ry(2, theta=-0.1)
    c.rz(1, theta=0.3)

    c1, _ = tc.compiler.simple_compiler.simple_compile(c)
    print(c1.draw())
    assert c1.gate_count() == 3


def test_default_compiler_places_on_the_lowest_errors():
    pytest.importorskip("qiskit")
    from tensorcircuit.compiler.qiskit_compiler import properties_to_target

    bit = {"SingleQubitErrRate": 1e-3, "ReadoutF0Err": 0.01, "ReadoutF1Err": 0.02}
    # a chain 0 - 1 - 2 - 3 - 4 - 5, where qubit 4 and coupler 2 - 3 have
    # unknown errors and are never used, though 3 - 4 looks the best
    bits = {q: dict(bit) for q in range(6)}
    bits[4]["SingleQubitErrRate"] = None
    errors = {(0, 1): 0.2, (1, 2): 0.01, (2, 3): None, (3, 4): 1e-4, (4, 5): 1e-4}
    links = {}
    for (a, b), e in errors.items():
        links[(a, b)] = links[(b, a)] = {"CZErrRate": e}
    properties = {"bits": bits, "links": links}
    target = properties_to_target(properties, ["u3", "h", "cx", "cz"])
    assert not target.instruction_supported("cz", (2, 3))
    assert not target.instruction_supported("measure", (4,))
    assert target.instruction_supported("cx", (2, 1))

    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    # seeded, as the layout on qiskit < 2 depends on the seed, see properties_to_target
    compiler = tc.compiler.DefaultCompiler(
        {"seed_transpiler": 0}, properties=properties
    )
    c1, info = compiler(c)
    assert sorted(info["logical_physical_mapping"].values()) == [1, 2]
    np.testing.assert_allclose(
        c1.probability().reshape([2] * c1._nqubits).sum(axis=(0, 3, 4, 5)),
        [[0.5, 0], [0, 0.5]],
        atol=1e-5,
    )
    # the coupling map given by the user is kept
    _, info = tc.compiler.DefaultCompiler(
        {"coupling_map": [[0, 1], [1, 0]]}, properties=properties
    )(c)
    assert sorted(info["logical_physical_mapping"].values()) == [0, 1]


def test_merge_same_axis_rotation():
    c = tc.Circuit(1)
    c.ry(0, theta=0.3)
    c.ry(0, theta=0.4)
    c1 = tc.compiler.simple_compiler.merge(c)
    assert c1.gate_count() == 1
    np.testing.assert_allclose(c.matrix(), c1.matrix(), atol=1e-7)

    c = tc.Circuit(2)
    c.cry(0, 1, theta=0.3)
    c.cry(0, 1, theta=0.4)
    qir = tc.compiler.simple_compiler.merge(c.to_qir())
    merged = [q for q in qir if q["name"].startswith("cr")]
    assert len(merged) == 1
    assert merged[0]["name"] == "cry"
    np.testing.assert_allclose(merged[0]["parameters"]["theta"], 0.7, atol=1e-7)


def test_merge_two_qubit_rotation():
    for g in ["rxx", "rzz", "ryy"]:
        c = tc.Circuit(2)
        getattr(c, g)(0, 1, theta=0.3)
        getattr(c, g)(0, 1, theta=0.4)
        c1 = tc.compiler.simple_compiler.merge(c)
        assert c1.gate_count() == 1
        ref = tc.Circuit(2)
        getattr(ref, g)(0, 1, theta=0.7)
        np.testing.assert_allclose(c.matrix(), c1.matrix(), atol=1e-7)
        np.testing.assert_allclose(c.matrix(), ref.matrix(), atol=1e-7)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize(
    "case", ["matching", "different_angle", "different_matrix", "both", "nonunitary"]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_merge_adjoint_requires_inverse(backend, case, reverse):
    first = tc.Circuit(1)
    second = tc.Circuit(1)
    if case == "nonunitary":
        first.any(0, unitary=np.diag([2.0, 1.0]))
        second.any(0, unitary=np.diag([2.0, 1.0]))
    else:
        first.exp1(0, unitary=tc.gates.x().tensor, theta=0.2)
        generator = (
            tc.gates.z() if case in ("different_matrix", "both") else tc.gates.x()
        )
        theta = 0.7 if case in ("different_angle", "both") else 0.2
        second.exp1(0, unitary=generator.tensor, theta=theta)
    adjoint = second.inverse()
    pair = adjoint.append(first) if reverse else first.append(adjoint)
    c = tc.Circuit(1)
    c.h(0)
    c.append(pair)
    expected = tc.backend.numpy(c.matrix())
    compiled, _ = tc.compiler.simple_compiler.simple_compile(c)
    merged = tc.compiler.simple_compiler.merge(c)
    merged_qir = tc.compiler.simple_compiler.merge(c.to_qir())
    reconstructed = tc.Circuit.from_qir(merged_qir, {"nqubits": 1})
    for result in (compiled, merged, reconstructed):
        np.testing.assert_allclose(
            tc.backend.numpy(result.matrix()), expected, atol=1e-6
        )
        assert result.gate_count() == (1 if case == "matching" else 3)
    assert c.gate_count() == 3


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
@pytest.mark.parametrize("gate", ["s", "t"])
@pytest.mark.parametrize("reverse", [False, True])
def test_merge_fixed_adjoint_pair(backend, gate, reverse):
    c = tc.Circuit(1)
    names = (gate + "d", gate) if reverse else (gate, gate + "d")
    for name in names:
        getattr(c, name)(0)
    merged = tc.compiler.simple_compiler.merge(c)
    assert merged.gate_count() == 0
    np.testing.assert_allclose(tc.backend.numpy(c.matrix()), np.eye(2), atol=1e-6)


@pytest.mark.parametrize("instruction", ["measure_instruction", "reset_instruction"])
@pytest.mark.parametrize("gate_count,position", [(1, 0), (3, 1), (3, 2)])
def test_simple_compile_rejects_midcircuit_instruction(
    npb, instruction, gate_count, position
):
    c = tc.Circuit(1)
    for i in range(gate_count):
        if i == position:
            getattr(c, instruction)(0)
        c.x(0)
    original = [dict(d) for d in c._extra_qir]
    with pytest.raises(ValueError, match=f"only the final position {gate_count}"):
        tc.compiler.simple_compiler.simple_compile(c)
    assert c._extra_qir == original
    assert c.gate_count() == gate_count


@pytest.mark.parametrize("program", ["empty", "single", "cancel", "reduce", "expand"])
def test_simple_compile_preserves_terminal_instructions(npb, program):
    c = tc.Circuit(2)
    if program == "single":
        c.x(0)
    elif program == "cancel":
        c.x(0)
        c.x(0)
    elif program == "reduce":
        c.h(0)
        c.h(0)
        c.x(0)
    elif program == "expand":
        c.rx(0, theta=0.3)
    c.measure_instruction(1, 0)
    c.reset_instruction(0)
    c.measure_instruction(0)
    original = [dict(d) for d in c._extra_qir]
    expected = tc.backend.numpy(c.state())
    compiled, _ = tc.compiler.simple_compiler.simple_compile(c)
    assert c._extra_qir == original
    assert compiled._extra_qir == [
        {**d, "pos": compiled.gate_count()} for d in original
    ]
    compiled._extra_qir[0]["index"] = [0]
    assert c._extra_qir == original
    np.testing.assert_allclose(tc.backend.numpy(compiled.state()), expected, atol=1e-6)
    repeated, _ = tc.compiler.simple_compiler.simple_compile(c)
    assert c._extra_qir == original
    assert repeated._extra_qir == [
        {**d, "pos": repeated.gate_count()} for d in original
    ]


def test_simple_compile_preserves_measurement_export(npb):
    qiskit = pytest.importorskip("qiskit")
    c = tc.Circuit(1)
    c.h(0)
    c.h(0)
    c.x(0)
    c.measure_instruction(0)
    original = c.to_qiskit(enable_instruction=True)
    compiled, _ = tc.compiler.simple_compiler.simple_compile(c)
    assert c.to_qiskit(enable_instruction=True) == original
    assert compiled.gate_count() == 1
    for qc in (original, compiled.to_qiskit(enable_instruction=True)):
        state = qiskit.quantum_info.Statevector.from_instruction(
            qc.remove_final_measurements(inplace=False)
        )
        np.testing.assert_allclose(state.probabilities(), [0, 1], atol=1e-6)


def _routing_error(circuit, routed, info, seed=0):
    """Compare seeded random-input amplitudes, including global phase."""
    n = circuit._nqubits
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    state /= np.linalg.norm(state)
    physical_input = (
        state.reshape([2] * n).transpose(np.argsort(info["initial_layout"])).reshape(-1)
    )
    original = tc.Circuit.from_qir(
        circuit.to_qir(),
        dict(circuit.circuit_param, inputs=state, mps_inputs=None, tensors=None),
    )
    compiled = tc.Circuit.from_qir(
        routed.to_qir(),
        dict(
            routed.circuit_param, inputs=physical_input, mps_inputs=None, tensors=None
        ),
    )
    expected = (
        tc.backend.numpy(original.state())
        .reshape([2] * n)
        .transpose(np.argsort(info["final_layout"]))
        .reshape(-1)
    )
    actual = tc.backend.numpy(compiled.state())
    return float(np.max(np.abs(actual - expected)))


def _routing_circuit():
    rng = np.random.default_rng(19)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    c = tc.Circuit(4)
    c.rx(0, theta=0.31, name="rz")
    c.cx(3, 0)
    c.rxx(0, 2, theta=-0.4)
    c.any(3, 1, unitary=unitary)
    c.rz(0, theta=0.7)
    c.rzz(3, 1, theta=-0.2)
    c.rzz(3, 1, theta=0.8)
    c.cphase(1, 3, theta=0.3)
    c.crz(2, 0, theta=0.6)
    c.cz(0, 3)
    c.diagonal(3, 0, diag=np.exp(1j * np.array([0.1, 0.4, -0.2, 0.8])))
    c.s(2)
    c.td(1)
    c.ry(3, theta=-0.1)
    c.any(2, 0, unitary=unitary)
    c.rz(-1, theta=0.5)
    c.cx(-1, 0)
    return c


def _permuted_routing_matrix(circuit, info):
    n = circuit._nqubits
    initial = np.argsort(info["initial_layout"])
    final = np.argsort(info["final_layout"])
    return (
        tc.backend.numpy(circuit.matrix())
        .reshape([2] * (2 * n))
        .transpose(list(final) + list(n + initial))
        .reshape(2**n, 2**n)
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
@pytest.mark.parametrize("strategy", ["greedy", "swap_network", "auto"])
@pytest.mark.parametrize(
    "commute,fuse", [(False, False), (False, True), (True, False), (True, True)]
)
def test_linear_route_operator(backend, strategy, commute, fuse):
    c = _routing_circuit()
    original = [(d["index"], d["name"], d["gatef"]) for d in c.to_qir()]
    routed, info = tc.compiler.linear_route(
        c,
        initial_layout=[2, 0, 3, 1],
        strategy=strategy,
        commute_diagonal=commute,
        fuse=fuse,
    )
    np.testing.assert_allclose(
        tc.backend.numpy(routed.matrix()), _permuted_routing_matrix(c, info), atol=2e-6
    )
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)
    assert all(
        len(d["index"]) == 1 or abs(d["index"][1] - d["index"][0]) == 1
        for d in routed.to_qir()
    )
    assert sorted(info["final_layout"]) == list(range(4))
    assert sorted(i for origins in info["origins"] for i in origins) == list(
        range(len(original))
    )
    assert [(d["index"], d["name"], d["gatef"]) for d in c.to_qir()] == original
    if not commute and not fuse:
        assert [i for origins in info["origins"] for i in origins] == list(
            range(len(original))
        )


@pytest.mark.parametrize("backend", [lf("tfb"), lf("torchb")])
def test_linear_route_backend_tensors(backend):
    c = _routing_circuit()
    routed, info = tc.compiler.linear_route(c)
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)
    assert all(tc.backend.is_tensor(d["gate"].tensor) for d in routed.to_qir())


def test_linear_route_mapping_and_pipeline(npb):
    c = _routing_circuit()
    previous = {"logical_physical_mapping": {7: 2, 9: 0}, "tag": "retained"}
    compile_pass = tc.compiler.Compiler(
        tc.compiler.linear_route,
        {"initial_layout": [2, 0, 3, 1], "strategy": "greedy"},
    )
    routed, first = compile_pass(c, previous)
    assert previous == {"logical_physical_mapping": {7: 2, 9: 0}, "tag": "retained"}
    assert first["tag"] == "retained"
    assert first["logical_physical_mapping"] == {
        7: first["final_layout"][2],
        9: first["final_layout"][0],
    }
    second_routed, second = tc.compiler.linear_route(
        routed, first, initial_layout=[1, 3, 0, 2]
    )
    assert second["logical_physical_mapping"] == {
        q: second["final_layout"][p]
        for q, p in first["logical_physical_mapping"].items()
    }
    np.testing.assert_allclose(
        _routing_error(routed, second_routed, second), 0, atol=2e-6
    )
    broken = dict(first, final_layout=first["final_layout"][::-1])
    assert _routing_error(c, routed, broken) > 0.01


def test_linear_route_fusion_and_phase(npb):
    c = tc.Circuit(4)
    c.cz(0, 1)
    c.rx(3, theta=0.3)
    c.cx(1, 0)
    c.rzz(0, 3, theta=0.6)
    c.rzz(0, 3, theta=-0.4)
    kwargs = {"initial_layout": list(range(4)), "commute_diagonal": False}
    raw, raw_info = tc.compiler.linear_route(c, fuse=False, **kwargs)
    fused, info = tc.compiler.linear_route(c, **kwargs)
    assert info["num_swaps"] > 0
    assert info["num_two_qubit_gates"] < raw_info["num_two_qubit_gates"]
    assert (0, 2) in info["origins"]
    np.testing.assert_allclose(_routing_error(c, fused, info), 0, atol=2e-6)
    np.testing.assert_allclose(_routing_error(c, raw, raw_info), 0, atol=2e-6)
    phased = tc.Circuit.from_qir(fused.to_qir(), {"nqubits": 4})
    phased.any(0, unitary=1j * np.eye(2))
    assert _routing_error(c, phased, info) > 0.1


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
@pytest.mark.parametrize("fuse", [False, True])
def test_linear_route_preserves_unmerged_gates(backend, fuse):
    split = {"max_singular_values": 2}
    c = tc.Circuit(2)
    c.cx(1, 0)
    c.rz(0, theta=0.2)
    c.exp1(1, 0, unitary=tc.gates._zz_matrix, theta=0.3, split=split)
    routed, info = tc.compiler.linear_route(c, initial_layout=[0, 1], fuse=fuse)
    before, after = c.to_qir(), routed.to_qir()
    assert info["num_swaps"] == 0
    assert [d["index"] for d in after] == [(1, 0), (0,), (1, 0)]
    assert [d["name"] for d in after] == ["cnot", "rz", "exp1"]
    assert [d["gatef"] for d in after] == [d["gatef"] for d in before]
    assert after[2]["split"] == split
    assert len(routed._nodes) == len(c._nodes)
    for original, compiled in zip(before, after):
        for key, value in original.get("parameters", {}).items():
            np.testing.assert_allclose(
                tc.backend.numpy(compiled["parameters"][key]), tc.backend.numpy(value)
            )
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)


def test_linear_route_preserves_reversed_gate_export(npb):
    pytest.importorskip("qiskit")
    c = tc.Circuit(2)
    c.cx(1, 0)
    routed, _ = tc.compiler.linear_route(c, initial_layout=[0, 1], fuse=False)
    assert routed.to_qiskit() == c.to_qiskit()
    assert routed.to_openqasm() == c.to_openqasm()


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
@pytest.mark.parametrize("first", [(0, 1), (1, 0)])
@pytest.mark.parametrize("second", [(0, 1), (1, 0)])
def test_linear_route_fuses_both_orientations(backend, first, second):
    c = tc.Circuit(3)
    c.cx(*first)
    c.ry(2, theta=0.4)
    c.crx(*second, theta=0.3)
    routed, info = tc.compiler.linear_route(c, initial_layout=[0, 1, 2])
    assert info["num_two_qubit_gates"] == 1
    assert info["origins"] == [(0, 2), (1,)]
    assert routed.to_qir()[0]["index"] == (0, 1)
    np.testing.assert_allclose(
        tc.backend.numpy(routed.matrix()), tc.backend.numpy(c.matrix()), atol=2e-6
    )


@pytest.mark.parametrize("gate", ["oz", "orz"])
def test_linear_route_open_control_diagonal(npb, gate):
    c = tc.Circuit(4)
    getattr(c, gate)(3, 0, **({"theta": 0.37} if gate == "orz" else {}))
    c.cz(1, 2)
    routed, info = tc.compiler.linear_route(
        c, initial_layout=list(range(4)), strategy="greedy", fuse=False
    )
    assert info["origins"][0] == (1,)
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)


def test_linear_route_auto_and_layout(npb):
    labels = [4, 0, 3, 1, 5, 2]
    c = tc.Circuit(6)
    for i, j in zip(labels[:-1], labels[1:]):
        c.rzz(i, j, theta=0.2)
    routed, info = tc.compiler.linear_route(c)
    assert info["bandwidth"]["initial"] == 1
    assert info["cutwidth"]["initial"] == 1
    assert info["num_swaps"] == 0
    assert info["num_two_qubit_gates"] == 5
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)
    c = _routing_circuit()
    _, greedy = tc.compiler.linear_route(c, strategy="greedy")
    _, network = tc.compiler.linear_route(c, strategy="swap_network")
    _, auto = tc.compiler.linear_route(c)
    assert auto["num_two_qubit_gates"] == min(
        greedy["num_two_qubit_gates"], network["num_two_qubit_gates"]
    )
    assert auto["strategy"] == (
        "greedy"
        if greedy["num_two_qubit_gates"] <= network["num_two_qubit_gates"]
        else "swap_network"
    )
    _, repeated = tc.compiler.linear_route(c)
    assert repeated == auto


@pytest.mark.parametrize("case", ["disabled", "no_diagonal"])
def test_linear_route_auto_skips_identical_candidate(npb, monkeypatch, case):
    c = tc.Circuit(4)
    c.cx(3, 0)
    c.crx(1, 3, theta=0.2)
    if case == "disabled":
        c.rzz(0, 2, theta=0.3)
    created = []
    router_type = tc.compiler.linear_routing._Router
    original_init = router_type.__init__

    def counted_init(self, layout, fuse):
        created.append(layout)
        original_init(self, layout, fuse)

    monkeypatch.setattr(router_type, "__init__", counted_init)
    routed, info = tc.compiler.linear_route(c, commute_diagonal=case != "disabled")
    assert len(created) == 1
    assert info["strategy"] == "greedy"
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)


@pytest.mark.parametrize("n", [3, 6, 7])
def test_linear_route_swap_network_all_pairs(npb, n):
    c = tc.Circuit(n)
    for i in range(n):
        for j in range(i + 1, n):
            c.crz(j, i, theta=0.1 * (i + 2 * j))
    routed, info = tc.compiler.linear_route(
        c, strategy="swap_network", initial_layout=list(range(n)), fuse=False
    )
    assert info["num_swaps"] <= n * (n - 1) // 2
    assert info["origins"][-1]
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)


@pytest.mark.parametrize("n", [1, 4])
def test_linear_route_empty_and_onsite(npb, n):
    c = tc.Circuit(n)
    routed, info = tc.compiler.linear_route(c)
    assert info["num_swaps"] == 0
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=1e-7)
    c.rz(0, theta=0.3)
    c.diagonal(n - 1, diag=np.array([1, 1j]))
    routed, info = tc.compiler.linear_route(c)
    np.testing.assert_allclose(_routing_error(c, routed, info), 0, atol=2e-6)


@pytest.mark.parametrize(
    "case",
    [
        "arity",
        "mpo",
        "channel",
        "measure",
        "reset",
        "inputs",
        "mps_inputs",
        "tensors",
        "qudit",
        "symbolic",
    ],
)
def test_linear_route_unsupported(npb, case):
    c = tc.Circuit(3)
    if case == "arity":
        c.toffoli(0, 1, 2)
    elif case == "mpo":
        c.mpo(0, 1, mpo=tc.Circuit(2).quoperator())
    elif case == "channel":
        c.depolarizing(0, px=0.1, py=0.1, pz=0.1, status=0.5)
    elif case in ("measure", "reset"):
        getattr(c, case + "_instruction")(0)
    elif case == "inputs":
        c = tc.Circuit(3, inputs=np.eye(1, 8).reshape(-1))
    elif case == "mps_inputs":
        c = tc.Circuit(3, mps_inputs=c.quvector())
    elif case == "tensors":
        c = tc.Circuit(3, tensors=[np.array([1, 0]).reshape(1, 2, 1)] * 3)
    elif case == "qudit":
        c = tc.Circuit(3, dim=3)
    elif case == "symbolic":
        c = tc.SymbolCircuit(3)
    with pytest.raises((NotImplementedError, TypeError)):
        tc.compiler.linear_route(c)


@pytest.mark.parametrize("layout", [[0, 0], [0], [1, 2], [0.0, 1.0]])
def test_linear_route_invalid_layout(npb, layout):
    with pytest.raises(ValueError, match="permutation"):
        tc.compiler.linear_route(tc.Circuit(2), initial_layout=layout)


def test_linear_route_invalid_options(npb):
    with pytest.raises(ValueError, match="strategy"):
        tc.compiler.linear_route(tc.Circuit(2), strategy="unknown")
    with pytest.raises(ValueError, match="unknown routing options"):
        tc.compiler.linear_route(tc.Circuit(2), compiled_options={"typo": True})
    with pytest.raises(ValueError, match="previous mapping"):
        tc.compiler.linear_route(tc.Circuit(2), {"logical_physical_mapping": {0: 2}})


def test_linear_route_qaoa_mps(npb):
    n = 12
    rng = np.random.default_rng(7)
    edges = list(nx.random_regular_graph(3, n, seed=8).edges())
    h, weights = rng.normal(size=n), rng.normal(size=len(edges))
    c = tc.Circuit(n)
    for q in range(n):
        c.h(q)
    for gamma, beta in [(0.3, 0.2), (0.4, 0.1)]:
        for q in range(n):
            c.rz(q, theta=2 * gamma * h[q])
        for (i, j), w in zip(edges, weights):
            c.rzz(i, j, theta=2 * gamma * w)
        for q in range(n):
            c.rx(q, theta=2 * beta)
    routed, info = tc.compiler.linear_route(c)
    mps = tc.MPSCircuit.from_qir(
        routed.to_qir(), {"nqubits": n, "split": {"max_singular_values": 64}}
    )
    expected = (
        tc.backend.numpy(c.state())
        .reshape([2] * n)
        .transpose(np.argsort(info["final_layout"]))
        .reshape(-1)
    )
    np.testing.assert_allclose(
        tc.backend.numpy(mps.wavefunction()), expected, atol=4e-6
    )
    np.testing.assert_allclose(mps.get_fidelity(), 1, atol=1e-6)


def test_linear_route_gradient(jaxb):
    def state(theta, route):
        c = tc.Circuit(3)
        c.h(0)
        c.rzz(2, 0, theta=theta)
        c.cphase(0, 2, theta=theta * 0.4)
        c.rx(1, theta=theta * 0.2)
        if route:
            c, info = tc.compiler.linear_route(c, initial_layout=[1, 2, 0])
            logical = info["final_layout"][0]
        else:
            logical = 0
        return tc.backend.real(c.expectation_ps(x=[logical]))

    theta = tc.backend.convert_to_tensor(0.37)
    reference = tc.backend.jit(tc.backend.value_and_grad(lambda t: state(t, False)))
    routed = tc.backend.jit(tc.backend.value_and_grad(lambda t: state(t, True)))
    np.testing.assert_allclose(routed(theta), reference(theta), atol=2e-6)
