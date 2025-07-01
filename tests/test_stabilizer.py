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
    counts = np.sum(samples, axis=0)
    # Should be roughly equal number of 00 and 11 states
    assert abs(counts[0] - counts[1]) < 50


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
    assert abs(exp - 1.0) < 0.1


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
    print(c.entanglement_entropy(list(range(2))))
    print(len(c.current_circuit()))


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
    for _ in range(20):
        c = tc.StabilizerCircuit(2)
        c.h(0)
        c.depolarizing(0, 1, p=0.2)
        c.h(0)
        r.append(c.expectation_ps(z=[0]))
    assert 4 < np.sum(r) < 20


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
