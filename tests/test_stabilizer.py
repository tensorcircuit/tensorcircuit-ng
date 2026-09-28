import os
import sys

import numpy as np
import pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc

# Skip all tests if stim is not installed
stim = pytest.importorskip("stim")


def test_basic_gates():
    c = tc.StabilizerCircuit(2)
    c.h(0)
    c.cnot(0, 1)
    results = c.measure(0, 1)
    print(results)
    results = c.measure(0, 1)
    print(results)
    assert len(results) == 2
    assert all(r in [0, 1] for r in results)


def test_bell_state():
    c = tc.StabilizerCircuit(2)
    c.H(1)
    c.cx(1, 0)
    # Test Z measurements correlation
    samples = c.sample(batch=1000)
    assert samples.shape == (1000, 2)
    # Assert on the joint outcome histogram: Bell state yields only (0,0) and (1,1).
    joint = samples[:, 0].astype(int) * 2 + samples[:, 1].astype(int)
    hist = np.bincount(joint, minlength=4)
    # correlated outcomes dominate, anti-correlated outcomes ~ 0
    assert hist[0] + hist[3] == 1000
    assert hist[1] == 0 and hist[2] == 0
    # both parity branches should occur with nonzero support
    assert hist[0] > 0 and hist[3] > 0


def test_ghz_state():
    c = tc.StabilizerCircuit(3)
    c.h(0)
    c.cnot(0, 1)
    c.cnot(1, 2)
    # Test expectation values
    exp_z = c.expectation_ps(z=[0, 1])
    np.testing.assert_allclose(exp_z, 1.0, atol=1e-6)


def test_stabilizer_operations():
    c = tc.StabilizerCircuit(2)
    # Test all supported gates
    for gate in ["h", "x", "y", "z", "s", "sdg"]:
        getattr(c, gate)(0)
    c.cnot(0, 1)
    c.cz(0, 1)
    c.swap(0, 1)
    print(c.current_circuit())


def test_sample_expectation():
    c = tc.StabilizerCircuit(2)
    c.h(0)
    c.cnot(0, 1)
    # Test sampling-based expectation
    exp = c.sample_expectation_ps(z=[0, 1], shots=1000)
    np.testing.assert_allclose(exp, 1.0, atol=0.1)


def test_invalid_gates():
    c = tc.StabilizerCircuit(1)
    with pytest.raises(ValueError):
        c.t(0)


def test_qir():
    c = tc.StabilizerCircuit(2)
    c.h(0)
    c.cnot(0, 1)
    qir = c.to_qir()
    assert len(qir) == 2
    assert qir[0]["name"] == "h"
    assert qir[1]["name"] == "cnot"
    print(qir)
    c1 = tc.Circuit.from_qir(qir)
    np.testing.assert_allclose(c1.expectation_ps(z=(0, 1)), 1, atol=1e-5)


def test_cond_measure():
    c = tc.StabilizerCircuit(3)

    # Prepare Bell pair between qubits 1 and 2
    c.H(1)
    c.CNOT(1, 2)

    # Prepare state to teleport on qubit 0 (can only be Clifford state)
    c.X(0)

    # Teleportation circuit
    c.CNOT(0, 1)
    c.H(0)

    # Measure qubits 0 and 1
    r0 = c.cond_measure(0)
    r1 = c.cond_measure(1)
    # Apply corrections based on measurements
    if r0 == 1:
        c.Z(2)
    if r1 == 1:
        c.X(2)

    # Verify teleported state
    final = c.measure(2)
    np.testing.assert_allclose(final, 1)


def test_post_select():
    c = tc.StabilizerCircuit(2)
    c.h(0)
    c.s(0)
    c.cx(0, 1)
    c.post_select(1, keep=1)
    np.testing.assert_allclose(c.expectation_ps(z=[0]), -1, atol=1e-5)


def test_to_openqasm():
    c = tc.StabilizerCircuit(3)
    c.sd(0)
    c.cz(0, 1)
    c.s(2)
    c.measure(0, 1)

    # Test basic circuit conversion
    qasm = c.to_openqasm()
    print(qasm)

    c1 = tc.StabilizerCircuit.from_openqasm(qasm)
    print(c1.draw())


def test_ee():
    c = tc.Circuit(8)
    for i in range(3):
        c.h(i)
        c.cx(i, i + 4)
        c.sd(i + 2)
    ee0 = tc.quantum.entanglement_entropy(c.state(), list(range(4)))
    c1 = tc.StabilizerCircuit.from_openqasm(c.to_openqasm())
    ee1 = c1.entanglement_entropy(list(range(4)))
    np.testing.assert_allclose(ee0, ee1, atol=1e-6)


def test_random_gates():
    c = tc.StabilizerCircuit(4)
    c.random_gate(0, 1, recorded=True)
    c.random_gate(2, 3)
    c.random_gate(1, 2)
    ee = float(c.entanglement_entropy(list(range(2))))
    assert np.isfinite(ee)
    assert 0.0 <= ee <= 2.0
    # recorded gate is appended to the stim circuit, unrecorded gates are not
    n_recorded = len(c.current_circuit())
    assert n_recorded > 0
    c2 = tc.StabilizerCircuit(4)
    c2.random_gate(0, 1)
    c2.random_gate(2, 3)
    c2.random_gate(1, 2)
    assert len(c2.current_circuit()) == 0
    print(ee)
    print(n_recorded)


def test_circuit_state():
    c = tc.StabilizerCircuit(2)
    c.h(1)
    c1 = tc.Circuit(2)
    c1.h(1)
    np.testing.assert_allclose(c.state(), c1.state(), atol=1e-5)


def test_circuit_inputs():
    c = tc.StabilizerCircuit(2, inputs=[stim.PauliString("XX"), stim.PauliString("ZZ")])
    c.cnot(0, 1)
    c.h(0)
    np.testing.assert_allclose(c.expectation_ps(z=[0]), 1, atol=1e-6)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 1, atol=1e-6)


def test_depolarize():
    r = []
    for _ in range(40):
        c = tc.StabilizerCircuit(2)
        c.h(0)
        c.depolarizing(0, 1, p=0.2)
        c.h(0)
        r.append(c.expectation_ps(z=[0]))
    assert 4 < np.sum(r) < 39


def test_tableau_inputs():
    c = tc.StabilizerCircuit(2)
    c.x(1)
    c.s(1)
    it = c.current_inverse_tableau()
    c1 = tc.StabilizerCircuit(2, tableau_inputs=it)
    c1.s(1)
    c1.x(1)
    np.testing.assert_allclose(c1.state()[0], 1, atol=1e-6)


def test_mipt():
    resource = [stim.Tableau.random(2) for _ in range(1000)]

    def ruc(n, nlayer, p):
        c = tc.StabilizerCircuit(n)
        status = np.random.choice(1000, size=[n, nlayer], replace=True)
        for j in range(nlayer):
            for i in range(0, n, 2):
                c.tableau_gate(i, (i + 1) % n, tableau=resource[status[i, j]])
            for i in range(1, n, 2):
                c.tableau_gate(i, (i + 1) % n, tableau=resource[status[i, j]])
            mask = np.random.random(n) < p
            ids = list(np.where(mask)[0])
            c.cond_measure_many(*ids)
        return c.entanglement_entropy(list(range(n // 2)))

    print(ruc(50, 10, 0.1))

    # entropy should be finite and within a sane range for the MIPT phase
    ee = float(ruc(50, 10, 0.1))
    assert np.isfinite(ee)
    assert 0.0 <= ee <= 25.0


def test_measure_with_prob():
    c = tc.StabilizerCircuit(3)
    c.h(0)
    c.cnot(0, 1)
    m, p = c.measure(0, 2, with_prob=True)
    np.testing.assert_allclose(p, 0.5, atol=1e-6)
    print(m)


def test_measure_with_prob_correlated():
    # The reported probability must be the true JOINT probability of the outcome.
    # For entangled qubits, peek_z==0 on every qubit does NOT imply independent
    # randomness: correlated outcomes share a single random bit. The pre-fix
    # heuristic (0.5 ** (#peek_z==0 qubits)) returned 0.25/0.125 here instead of 0.5.

    # Bell pair: measuring both qubits yields only (0,0) or (1,1), each with prob 0.5.
    for _ in range(50):
        c = tc.StabilizerCircuit(2)
        c.h(0)
        c.cnot(0, 1)
        m, p = c.measure(0, 1, with_prob=True)
        np.testing.assert_allclose(p, 0.5, atol=1e-6)
        assert tuple(int(x) for x in np.atleast_1d(np.array(m))) in [(0, 0), (1, 1)]

    # GHZ on 3 qubits: only (0,0,0) or (1,1,1), each with prob 0.5 (was 0.125 pre-fix).
    for _ in range(50):
        c = tc.StabilizerCircuit(3)
        c.h(0)
        c.cnot(0, 1)
        c.cnot(1, 2)
        m, p = c.measure(0, 1, 2, with_prob=True)
        np.testing.assert_allclose(p, 0.5, atol=1e-6)
        assert tuple(int(x) for x in np.atleast_1d(np.array(m))) in [
            (0, 0, 0),
            (1, 1, 1),
        ]

    # Two genuinely independent random qubits: 4 equiprobable outcomes, each 0.25.
    # Guards against over-correcting (the independent case must stay 0.25).
    for _ in range(50):
        c = tc.StabilizerCircuit(2)
        c.h(0)
        c.h(1)
        _, p = c.measure(0, 1, with_prob=True)
        np.testing.assert_allclose(p, 0.25, atol=1e-6)

    # Deterministic product state: prob 1.0.
    c = tc.StabilizerCircuit(3)
    _, p = c.measure(0, 1, 2, with_prob=True)
    np.testing.assert_allclose(p, 1.0, atol=1e-6)


def test_entanglement_entropy_dual_and_validation():
    n = 4
    stc = tc.StabilizerCircuit(n)
    stc.h(0)
    stc.cx(0, 1)
    stc.cx(1, 2)
    stc.cx(2, 3)

    # legacy cut (keep) == keyword keep
    ref = stc.entanglement_entropy([0, 1])
    assert np.isclose(ref, stc.entanglement_entropy(subsystem_to_keep=[0, 1]))
    # trace_out complement == keep
    assert np.isclose(ref, stc.entanglement_entropy(subsystems_to_trace_out=[2, 3]))
    # matches quantum.entanglement_entropy on the same state when both keep [0,1]
    c = tc.Circuit(n)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.cx(2, 3)
    q_ee = float(
        np.real(tc.quantum.entanglement_entropy(c.state(), subsystem_to_keep=[0, 1]))
    )
    assert np.isclose(ref, q_ee, atol=1e-6)
    # None (full system) -> entropy 0
    assert np.isclose(stc.entanglement_entropy(), 0.0)

    # validation
    with pytest.raises(ValueError, match="only one of"):
        stc.entanglement_entropy(subsystem_to_keep=[0], subsystems_to_trace_out=[1])
    with pytest.raises(ValueError, match="out of range"):
        stc.entanglement_entropy([n])


@pytest.mark.parametrize("input_kind", ["stabilizers", "tableau", "both"])
@pytest.mark.parametrize(
    "preparation,observable,expected",
    [
        ("X 0", {"z": [0]}, -1),
        ("H 0\nS 0", {"y": [0]}, 1),
        ("H 0\nCX 0 1\nX 1", {"z": [0, 1]}, -1),
    ],
)
def test_replay_initial_state(npb, input_kind, preparation, observable, expected):
    reference = stim.TableauSimulator()
    reference.do(stim.Circuit(preparation))
    n = reference.num_qubits
    kwargs = {}
    if input_kind in ("stabilizers", "both"):
        kwargs["inputs"] = reference.canonical_stabilizers()
    if input_kind == "both":
        kwargs["inputs"] = [
            stim.PauliString("I" * i + "Z" + "I" * (n - i - 1)) for i in range(n)
        ]
    if input_kind in ("tableau", "both"):
        kwargs["tableau_inputs"] = reference.current_inverse_tableau()
    c = tc.StabilizerCircuit(n, **kwargs)
    np.testing.assert_allclose(c.expectation_ps(**observable), expected)
    np.testing.assert_allclose(
        c.sample_expectation_ps(**observable, shots=16), expected
    )
    replay = stim.TableauSimulator()
    replay.do(c.current_circuit())
    assert replay.canonical_stabilizers() == reference.canonical_stabilizers()
    if "z" in observable:
        bits = c.sample(batch=16).astype(int)
        np.testing.assert_array_equal(
            np.prod(1 - 2 * bits, axis=1), np.full(16, expected)
        )
    c.x(0)
    reference.x(0)
    replay = stim.TableauSimulator()
    replay.do(c.current_circuit())
    assert replay.canonical_stabilizers() == reference.canonical_stabilizers()


@pytest.mark.parametrize("indices", [(2,), (2, 0), (3, 1, 0)])
@pytest.mark.parametrize("random", [False, True])
def test_replay_tableau_targets(npb, monkeypatch, indices, random):
    local = stim.Circuit("H 0\nS 0")
    if len(indices) > 1:
        local.append("CX", [0, len(indices) - 1])
    tableau = stim.Tableau.from_circuit(local)
    c = tc.StabilizerCircuit(4)
    c.x(1)
    if random:
        monkeypatch.setattr(stim.Tableau, "random", lambda n: tableau.copy())
        c.random_gate(*indices, recorded=True)
    else:
        c.tableau_gate(*indices, tableau=tableau, recorded=True)
    replay = stim.TableauSimulator()
    replay.do(c.current_circuit())
    replay.set_num_qubits(4)
    live = c.current_simulator().copy()
    live.set_num_qubits(4)
    assert replay.canonical_stabilizers() == live.canonical_stabilizers()
    recorded = c.current_circuit().copy()
    recorded.append("I", [3])
    assert (
        stim.Tableau.from_circuit(recorded) == live.current_inverse_tableau().inverse()
    )


def test_replay_tableau_samples(npb):
    c = tc.StabilizerCircuit(2)
    c.tableau_gate(1, tableau=stim.Tableau.from_named_gate("X"), recorded=True)
    np.testing.assert_array_equal(c.measure(0, 1), [0, 1])
    np.testing.assert_array_equal(c.sample(batch=16), np.tile([0, 1], (16, 1)))
    np.testing.assert_allclose(c.sample_expectation_ps(z=[0], shots=16), 1)
    np.testing.assert_allclose(c.sample_expectation_ps(z=[1], shots=16), -1)


@pytest.mark.parametrize("basis", ["x", "y", "z", "identity"])
@pytest.mark.parametrize("measure_many", [False, True])
def test_replay_expectation_excludes_history(npb, basis, measure_many):
    c = tc.StabilizerCircuit(2)
    c.x(0)
    if measure_many:
        c.cond_measure_many(0, 1)
    else:
        c.cond_measurement(0)
    if basis in ("x", "y"):
        c.h(0)
    if basis == "y":
        c.s(0)
    observable = {} if basis == "identity" else {basis: [0]}
    expected = 1 if basis == "identity" else -1
    before = c.current_circuit().copy()
    np.testing.assert_allclose(c.expectation_ps(**observable), expected)
    np.testing.assert_allclose(
        c.sample_expectation_ps(**observable, shots=16), expected
    )
    assert c.current_circuit() == before
    assert c.current_circuit().num_measurements == (2 if measure_many else 1)


def test_replay_preserves_noise_sampling(npb, monkeypatch):
    compile_sampler = stim.Circuit.compile_sampler

    def seeded_sampler(circuit):
        return compile_sampler(circuit, seed=17)

    monkeypatch.setattr(stim.Circuit, "compile_sampler", seeded_sampler)
    c = tc.StabilizerCircuit(1, inputs=[stim.PauliString("-Z")])
    c.depolarizing(0, p=0.75)
    reference = stim.Circuit("X 0\nDEPOLARIZE1(0.75) 0\nM 0")
    expected = reference.compile_sampler().sample(256)
    np.testing.assert_array_equal(c.sample(batch=256), expected)
    np.testing.assert_array_equal(np.unique(expected), [False, True])
    np.testing.assert_allclose(
        c.sample_expectation_ps(z=[0], shots=256), np.mean(1 - 2 * expected.astype(int))
    )
    assert c.current_circuit().num_measurements == 0


@pytest.mark.parametrize("input_kind", ["stabilizers", "tableau"])
@pytest.mark.parametrize("first_access", ["sample", "expectation", "circuit"])
def test_replay_lazy_preparation(npb, monkeypatch, input_kind, first_access):
    original = stim.Tableau.to_circuit
    calls = []

    def counted(tableau, *args, **kwargs):
        calls.append(tableau.copy())
        return original(tableau, *args, **kwargs)

    monkeypatch.setattr(stim.Tableau, "to_circuit", counted)
    kwargs = (
        {"inputs": [stim.PauliString("-Z")]}
        if input_kind == "stabilizers"
        else {"tableau_inputs": stim.Tableau.from_named_gate("X")}
    )
    c = tc.StabilizerCircuit(1, **kwargs)
    np.testing.assert_allclose(c.expectation_ps(z=[0]), -1)
    c.x(0)
    np.testing.assert_allclose(c.expectation_ps(z=[0]), 1)
    np.testing.assert_array_equal(c.measure(0), [0])
    assert not calls
    if first_access == "sample":
        np.testing.assert_array_equal(c.sample(batch=4), np.zeros((4, 1)))
    elif first_access == "expectation":
        np.testing.assert_allclose(c.sample_expectation_ps(z=[0], shots=4), 1)
    else:
        c.current_circuit()
    assert len(calls) == 1
    expected = original(calls[0]) + stim.Circuit("X 0")
    recorded = c.current_circuit()
    assert recorded == expected
    c.x(0)
    assert c.current_circuit() is recorded
    assert recorded == expected + stim.Circuit("X 0")
    np.testing.assert_array_equal(c.sample(batch=4), np.ones((4, 1)))
    np.testing.assert_allclose(c.sample_expectation_ps(z=[0], shots=4), -1)
    assert len(calls) == 1


def test_replay_lazy_preparation_preserves_unrecorded_operations(npb):
    c = tc.StabilizerCircuit(1, inputs=[stim.PauliString("-Z")])
    c.tableau_gate(0, tableau=stim.Tableau.from_named_gate("X"))
    np.testing.assert_allclose(c.expectation_ps(z=[0]), 1)
    np.testing.assert_allclose(c.sample_expectation_ps(z=[0], shots=4), -1)


@pytest.mark.parametrize("indices", [(0, 1, 2), (2, 1, 0), (4, 0, 2)])
def test_replay_batched_tableau_mapping(npb, indices):
    tableau = stim.Tableau.from_circuit(stim.Circuit("H 0\nS 1\nCX 0 2\nCX 2 1"))
    c = tc.StabilizerCircuit(
        5,
        inputs=[
            stim.PauliString("-ZIIII"),
            stim.PauliString("IZIII"),
            stim.PauliString("IIZII"),
            stim.PauliString("IIIZI"),
            stim.PauliString("IIIIZ"),
        ],
    )
    expected = c.current_tableau().to_circuit() + stim.Circuit("H 3")
    c.h(3)
    for instruction in tableau.to_circuit():
        expected.append(
            instruction.name,
            [indices[target.value] for target in instruction.targets_copy()],
            instruction.gate_args_copy(),
        )
    c.tableau_gate(*indices, tableau=tableau, recorded=True)
    assert c.current_circuit() == expected
    reference = stim.TableauSimulator()
    reference.do(expected)
    assert reference.canonical_stabilizers() == c.current_sim.canonical_stabilizers()


def test_replay_lazy_preparation_order(npb):
    c = tc.StabilizerCircuit(1, inputs=[stim.PauliString("-Z")])
    c.h(0)
    np.testing.assert_allclose(c.expectation_ps(x=[0]), -1)
    np.testing.assert_allclose(c.sample_expectation_ps(x=[0], shots=4), -1)
    c.h(0)
    np.testing.assert_array_equal(c.sample(batch=4), np.ones((4, 1)))
