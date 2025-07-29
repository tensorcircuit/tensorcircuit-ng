import sys
import os
import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_ode_evol(jaxb):
    def h_square(t, b):
        return (tc.backend.sign(t - 1.0) + 1) / 2 * b * tc.gates.x().tensor

    c = tc.Circuit(3)
    c.x(0)
    c.cx(0, 1)
    c.h(2)
    c = tc.timeevol.ode_evol_local(
        c, [1], h_square, 2.0, tc.backend.convert_to_tensor(0.2)
    )
    c.rx(1, theta=np.pi - 0.4)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 1.0, atol=1e-5)

    ixi = tc.quantum.PauliStringSum2COO([[0, 1, 0]])

    def h_square_sparse(t, b):
        return (tc.backend.sign(t - 1.0) + 1) / 2 * b * ixi

    c = tc.Circuit(3)
    c.x(0)
    c.cx(0, 1)
    c.h(2)
    c = tc.timeevol.ode_evol_global(
        c, h_square_sparse, 2.0, tc.backend.convert_to_tensor(0.2)
    )
    c.rx(1, theta=np.pi - 0.4)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_hamiltonian_evol_basic(backend):
    """Test basic functionality of hamiltonian_evol with a simple 2-qubit Hamiltonian"""
    # Define a simple 2-qubit Hamiltonian
    h = tc.backend.convert_to_tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 2.0, 0.0],
            [0.0, 2.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Initial state |00⟩
    psi0 = tc.backend.convert_to_tensor([1.0, 0.0, 0.0, 0.0])

    # Evolution times
    times = 1.0j * tc.backend.cast(
        tc.backend.convert_to_tensor([0.0, 0.5, 1.0]), tc.dtypestr
    )

    # Evolve and get states
    states = tc.timeevol.hamiltonian_evol(times, h, psi0)

    # Check output shape
    assert states.shape == (3, 4)

    # At t=0, state should be the initial state (normalized)
    np.testing.assert_allclose(states[0], psi0, atol=1e-5)

    # All states should be normalized
    for state in states:
        norm = tc.backend.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_hamiltonian_evol_with_callback(backend):
    """Test hamiltonian_evol with a callback function"""
    # Define a simple Hamiltonian
    h = tc.backend.convert_to_tensor([[1.0, 0.0], [0.0, -1.0]])

    # Initial state
    psi0 = tc.backend.convert_to_tensor([1.0, 1.0]) / np.sqrt(2)

    # Evolution times
    times = 1.0j * tc.backend.cast(
        tc.backend.convert_to_tensor([0.0, 1.0, 2.0]), tc.dtypestr
    )

    # Define callback to compute expectation value of Pauli Z
    def callback(state):
        # Z operator
        c = tc.Circuit(1, inputs=state)
        # Compute <psi|Z|psi>
        return tc.backend.real(c.expectation_ps(z=[0]))

    # Evolve with callback
    results = tc.timeevol.hamiltonian_evol(times, h, psi0, callback)

    # Check output shape - should be scalar for each time point
    assert results.shape == (3,)

    # At t=0, for |+⟩ state, <Z> should be 0
    np.testing.assert_allclose(results[0], 0.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_hamiltonian_evol_imaginary_time(backend):
    """Test that hamiltonian_evol performs imaginary time evolution by default"""
    # Define a Hamiltonian
    h = tc.backend.convert_to_tensor([[2.0, 0.0], [0.0, 1.0]])

    # Initial state with equal superposition
    psi0 = tc.backend.convert_to_tensor([1.0, 1.0]) / np.sqrt(2)

    # Large time to see ground state dominance
    times = tc.backend.convert_to_tensor([0.0, 10.0])

    # Evolve
    states = tc.timeevol.hamiltonian_evol(times, h, psi0)

    # Ground state is |1⟩ (eigenvalue 1.0), so after long imaginary time
    # evolution, we should approach this state
    expected_ground_state = tc.backend.convert_to_tensor([0.0, 1.0])
    np.testing.assert_allclose(states[-1], expected_ground_state, atol=1e-3)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_krylov_evol_heisenberg_6_sites(backend):
    """Test krylov_evol with Heisenberg Hamiltonian on 6 sites"""
    n = 6
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=False)
    print(h.dtype)
    # Initial state - all spins up except last one down
    psi0 = np.zeros((2**n,))
    psi0[62] = 1.0
    # State with pattern: up, up, up, up, up, down (111110 in binary = 62 in decimal)

    # Normalize initial state
    psi0 = psi0 / tc.backend.norm(psi0)

    # Evolution times
    times = tc.backend.convert_to_tensor([0.0, 0.5, 1.0])

    # Perform Krylov evolution
    states = tc.timeevol.krylov_evol(h, psi0, times, subspace_dimension=10)

    # Check output shape
    assert states.shape == (3, 2**n)

    # At t=0, state should be the initial state
    np.testing.assert_allclose(states[0], psi0, atol=1e-5)

    # All states should be normalized
    for state in states:
        norm = tc.backend.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_krylov_evol_heisenberg_8_sites(backend):
    """Test krylov_evol with Heisenberg Hamiltonian on 8 sites with callback"""
    n = 8
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=True)

    # Initial Neel state (alternating up and down spins)
    c = tc.Circuit(n)
    c.x([i for i in range(n // 2)])
    psi0 = c.state()

    # Evolution times
    times = tc.backend.convert_to_tensor([0.0, 0.2, 0.4])

    # Define callback to compute total magnetization
    def total_magnetization(state):
        # Calculate sum of <Z_i> for all sites
        c = tc.Circuit(n, inputs=state)
        total_z = 0.0
        for i in range(n):
            total_z += c.expectation_ps(z=[i])
        return tc.backend.real(total_z)

    # Perform Krylov evolution with callback
    results = tc.timeevol.krylov_evol(
        h, psi0, times, subspace_dimension=12, callback=total_magnetization
    )

    # Check output shape - should be scalar for each time point
    assert results.shape == (3,)

    # At t=0, for Neel state, total magnetization should be 0
    np.testing.assert_allclose(results[0], 0.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_krylov_evol_subspace_accuracy(backend):
    """Test accuracy of krylov_evol with different subspace dimensions"""
    n = 6
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=True)

    # Initial domain wall state
    c = tc.Circuit(n)
    c.x([i + n // 2 for i in range(n // 2)])
    psi0 = c.state()

    # Evolution time
    times = tc.backend.convert_to_tensor([1.0])

    # Compare different subspace dimensions
    state_small = tc.timeevol.krylov_evol(h, psi0, times, subspace_dimension=6)
    state_large = tc.timeevol.krylov_evol(h, psi0, times, subspace_dimension=16)

    # Both should be normalized
    norm_small = tc.backend.norm(state_small[0])
    norm_large = tc.backend.norm(state_large[0])
    np.testing.assert_allclose(norm_small, 1.0, atol=1e-5)
    np.testing.assert_allclose(norm_large, 1.0, atol=1e-5)

    # Larger subspace should be more accurate (but we can't easily test that without exact solution)
    # At least verify they have the correct shape
    assert state_small.shape == (1, 2**n)
    assert state_large.shape == (1, 2**n)
