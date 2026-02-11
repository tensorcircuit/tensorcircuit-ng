import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorcircuit as tc
from tensorcircuit.u1circuit import U1Circuit
from tensorcircuit.stabilizercircuit import StabilizerCircuit
from tensorcircuit.mpscircuit import MPSCircuit


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_qir_roundtrip(backend):
    n = 4
    k = 2
    c = U1Circuit(n, k=k, filled=[0, 1])
    c.rz(0, theta=0.5)
    c.cz(0, 1)
    c.iswap(1, 2, theta=0.8)
    c.cphase(2, 3, theta=0.4)

    qir = c.to_qir()
    assert len(qir) == 4

    # Reconstruct from QIR
    c2 = U1Circuit.from_qir(qir, c.circuit_param)

    # Verify expectations
    for i in range(n):
        np.testing.assert_allclose(
            tc.backend.numpy(c.expectation_z(i)),
            tc.backend.numpy(c2.expectation_z(i)),
            atol=1e-5,
        )


@pytest.mark.parametrize("backend", [lf("npb")])
def test_stabilizer_qir_roundtrip(backend):
    n = 4
    c = StabilizerCircuit(n)
    c.h(0)
    c.cnot(0, 1)
    c.s(1)

    qir = c.to_qir()
    assert len(qir) == 3

    c2 = StabilizerCircuit.from_qir(qir, c.circuit_param)

    # Compare state vectors (TableauSimulator to_state_vector)
    np.testing.assert_allclose(
        tc.backend.numpy(c.state()), tc.backend.numpy(c2.state()), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
def test_mps_qir_roundtrip(backend):
    n = 4
    c = MPSCircuit(n)
    c.h(0)
    c.cnot(0, 1)
    c.rz(1, theta=0.8)

    qir = c.to_qir()
    assert len(qir) == 3

    c2 = MPSCircuit.from_qir(qir, c.circuit_param)

    np.testing.assert_allclose(
        tc.backend.numpy(c.wavefunction()),
        tc.backend.numpy(c2.wavefunction()),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb")])
def test_circuit_to_stabilizer(backend):
    n = 2
    c = tc.Circuit(n)
    c.h(0)
    c.cnot(0, 1)
    c.s(1)
    c.x(0)

    qir = c.to_qir()
    # Convert to StabilizerCircuit
    c_stab = StabilizerCircuit.from_qir(qir, circuit_params={"nqubits": n})

    np.testing.assert_allclose(
        tc.backend.numpy(c.state()), tc.backend.numpy(c_stab.state()), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
def test_u1_to_mps(backend):
    n = 4
    c_u1 = U1Circuit(n, k=2, filled=[0, 1])
    c_u1.rz(0, theta=0.5)
    c_u1.cz(0, 1)
    c_u1.iswap(1, 2, theta=1.2)
    c_u1.rzz(3, 2, theta=-0.9)

    qir = c_u1.to_qir()

    c_mps = MPSCircuit(n)
    # Match initial state of U1 ([0,1] filled)
    c_mps.x(0)
    c_mps.x(1)
    c_mps.append_from_qir(qir)

    np.testing.assert_allclose(
        tc.backend.numpy(c_u1.to_dense()),
        tc.backend.numpy(c_mps.wavefunction().reshape([-1])),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb")])
def test_mps_to_circuit(backend):
    n = 2
    c_mps = MPSCircuit(n)
    c_mps.h(0)
    c_mps.cnot(0, 1)
    c_mps.ryy(1, 0, theta=0.9)
    c_mps.x(1)

    qir = c_mps.to_qir()
    c_std = tc.Circuit.from_qir(qir, circuit_params={"nqubits": n})

    np.testing.assert_allclose(
        tc.backend.numpy(c_mps.wavefunction().reshape([-1])),
        tc.backend.numpy(c_std.state()),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb")])
def test_dm_to_u1(backend):
    n = 2
    c_dm = tc.DMCircuit(n)
    c_dm.rz(0, theta=0.5)
    c_dm.cz(0, 1)
    c_dm.cphase(1, 0, theta=-1.1)

    qir = c_dm.to_qir()
    c_u1 = U1Circuit(n, k=0)
    c_u1.append_from_qir(qir)

    dm = tc.backend.numpy(c_dm.state())
    st = tc.backend.numpy(c_u1.to_dense())
    np.testing.assert_allclose(dm, np.outer(st, np.conj(st)), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb")])
def test_mps_to_stabilizer(backend):
    n = 2
    c_mps = MPSCircuit(n)
    c_mps.h(0)
    c_mps.cnot(0, 1)
    c_mps.s(1)

    qir = c_mps.to_qir()
    c_stab = StabilizerCircuit.from_qir(qir, circuit_params={"nqubits": n})

    np.testing.assert_allclose(
        tc.backend.numpy(c_mps.wavefunction().reshape([-1])),
        tc.backend.numpy(c_stab.state()),
        atol=1e-5,
    )
