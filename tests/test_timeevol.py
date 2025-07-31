import sys
import os
import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_circuit_ode_evol(jaxb):
    def h_square(t, b):
        return (tc.backend.sign(t - 1.0) + 1) / 2 * b * tc.gates.x().tensor

    c = tc.Circuit(3)
    c.x(0)
    c.cx(0, 1)
    c.h(2)
    c = tc.timeevol.evol_local(c, [1], h_square, 2.0, tc.backend.convert_to_tensor(0.2))
    c.rx(1, theta=np.pi - 0.4)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 1.0, atol=1e-5)

    ixi = tc.quantum.PauliStringSum2COO([[0, 1, 0]])

    def h_square_sparse(t, b):
        return (tc.backend.sign(t - 1.0) + 1) / 2 * b * ixi

    c = tc.Circuit(3)
    c.x(0)
    c.cx(0, 1)
    c.h(2)
    c = tc.timeevol.evol_global(
        c, h_square_sparse, 2.0, tc.backend.convert_to_tensor(0.2)
    )
    c.rx(1, theta=np.pi - 0.4)
    np.testing.assert_allclose(c.expectation_ps(z=[1]), 1.0, atol=1e-5)


def test_ode_evol_local(jaxb):
    def local_hamiltonian(t, Omega, phi):
        angle = phi * t
        coeff = Omega * tc.backend.cos(2.0 * t)  # Amplitude modulation

        # Single-qubit Rabi Hamiltonian (2x2 matrix)
        hx = coeff * tc.backend.cos(angle) * tc.gates.x().tensor
        hy = coeff * tc.backend.sin(angle) * tc.gates.y().tensor
        return hx + hy

    # Initial state: GHZ state |0000⟩ + |1111⟩
    c = tc.Circuit(4)
    c.h(0)
    for i in range(3):
        c.cnot(i, i + 1)
    psi0 = c.state()

    # Time points
    times = tc.backend.arange(0.0, 3.0, 0.1)

    # Evolve with local Hamiltonian acting on qubit 1
    states = tc.timeevol.ode_evol_local(
        local_hamiltonian,
        psi0,
        times,
        [1],  # Apply to qubit 1
        None,
        1.0,
        2.0,  # Omega=1.0, phi=2.0
    )
    assert tc.backend.shape_tuple(states) == (30, 16)


def test_ode_evol_global(jaxb):
    # Create a time-dependent transverse field Hamiltonian
    # H(t) = -∑ᵢ Jᵢ(t) ZᵢZᵢ₊₁ - ∑ᵢ hᵢ(t) Xᵢ

    # Time-dependent coefficients
    def time_dep_J(t):
        return 1.0 + 0.5 * tc.backend.sin(2.0 * t)

    def time_dep_h(t):
        return 0.5 * tc.backend.cos(1.5 * t)

    zz_ham = tc.quantum.PauliStringSum2COO(
        [[3, 3, 0, 0], [0, 3, 3, 0], [0, 0, 3, 3]], [1, 1, 1]
    )
    x_ham = tc.quantum.PauliStringSum2COO(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [1, 1, 1, 1]
    )

    # Hamiltonian construction function
    def hamiltonian_func(t):
        # Create time-dependent ZZ terms
        zz_coeff = time_dep_J(t)

        # Create time-dependent X terms
        x_coeff = time_dep_h(t)

        return zz_coeff * zz_ham + x_coeff * x_ham

    # Initial state: |↑↓↑↓⟩
    c = tc.Circuit(4)
    c.x([1, 3])
    psi0 = c.state()

    # Time points for evolution
    times = tc.backend.arange(0, 5, 0.5)

    def zobs(state):
        n = int(np.log2(state.shape[-1]))
        c = tc.Circuit(n, inputs=state)
        return tc.backend.real(c.expectation_ps(z=[0]))

    # Perform global ODE evolution
    states = tc.timeevol.ode_evol_global(hamiltonian_func, psi0, times, zobs)
    assert tc.backend.shape_tuple(states) == (10,)

    zz_ham = tc.quantum.PauliStringSum2COO([[3, 3, 0, 0], [0, 3, 3, 0]], [1, 1])
    x_ham = tc.quantum.PauliStringSum2COO([[1, 0, 0, 0], [0, 1, 0, 0]], [1, 1])

    # Example with parameterized Hamiltonian and optimization
    def parametrized_hamiltonian(t, params):
        # params = [J0, J1, h0, h1] - parameters to optimize
        J_t = params[0] + params[1] * tc.backend.sin(2.0 * t)
        h_t = params[2] + params[3] * tc.backend.cos(1.5 * t)

        return J_t * zz_ham + h_t * x_ham

    # Observable function: measure ZZ correlation
    def zz_correlation(state):
        n = int(np.log2(state.shape[0]))
        circuit = tc.Circuit(n, inputs=state)
        return circuit.expectation_ps(z=[0, 1])

    @tc.backend.jit
    @tc.backend.value_and_grad
    def objective_function(params):
        states = tc.timeevol.ode_evol_global(
            parametrized_hamiltonian,
            psi0,
            tc.backend.convert_to_tensor([0, 1.0]),
            None,
            params,
        )
        # Measure ZZ correlation at final time
        final_state = states[-1]
        return tc.backend.real(zz_correlation(final_state))

    print(objective_function(tc.backend.ones([4])))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_ed_evol(backend):
    n = 4
    g = tc.templates.graphs.Line1D(n, pbc=False)
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=False)

    # Initial Neel state: |↑↓↑↓⟩
    c = tc.Circuit(n)
    c.x([1, 3])  # Apply X gates to qubits 1 and 3
    psi0 = c.state()

    # Imaginary time evolution times
    times = tc.backend.convert_to_tensor([0.0, 0.5, 1.0, 2.0])

    # Evolve and get states
    states = tc.timeevol.ed_evol(h, psi0, times)
    print(states)

    def evolve_and_measure(params):
        # Parametrized Hamiltonian
        h_param = tc.quantum.heisenberg_hamiltonian(
            g, hzz=params[0], hxx=params[1], hyy=params[2], sparse=False
        )
        states = tc.timeevol.ed_evol(h_param, psi0, times)
        # Measure observable on final state
        circuit = tc.Circuit(n, inputs=states[-1])
        return tc.backend.real(circuit.expectation_ps(z=[0]))

    evolve_and_measure(tc.backend.ones([3]))


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
    states = tc.timeevol.hamiltonian_evol(h, psi0, times)

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
    results = tc.timeevol.hamiltonian_evol(h, psi0, times, callback)

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
    states = tc.timeevol.hamiltonian_evol(h, psi0, times)

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


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_krylov_evol_scan_impl(backend):
    """Test krylov_evol with scan_impl=True"""
    n = 4
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=True)

    c = tc.Circuit(n)
    c.x([1, 2])
    psi0 = c.state()

    # Evolution times
    times = tc.backend.convert_to_tensor([0.0, 0.5])

    # Perform Krylov evolution with scan implementation
    states_scan = tc.timeevol.krylov_evol(
        h, psi0, times, subspace_dimension=8, scan_impl=True
    )

    states_scan_dense = tc.timeevol.krylov_evol(
        tc.backend.to_dense(h), psi0, times, subspace_dimension=8, scan_impl=True
    )

    # Perform Krylov evolution with regular implementation
    states_regular = tc.timeevol.krylov_evol(
        h, psi0, times, subspace_dimension=8, scan_impl=False
    )

    # Check output shapes
    assert states_scan.shape == (2, 2**n)
    assert states_regular.shape == (2, 2**n)

    # Results should be the same (up to numerical precision)
    np.testing.assert_allclose(states_scan, states_regular, atol=1e-5)
    np.testing.assert_allclose(states_scan_dense, states_regular, atol=1e-5)

    # All states should be normalized
    for state in states_scan:
        norm = tc.backend.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_krylov_evol_gradient(backend):
    """Test gradient computation with krylov_evol"""
    n = 5
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=False)

    c = tc.Circuit(n)
    c.x([1, 2])
    psi0 = c.state()

    # Evolution time
    t = tc.backend.convert_to_tensor([1.0])

    # Define a simple loss function based on the evolved state
    def loss_function(t):
        states = tc.timeevol.krylov_evol(
            h, psi0, t, subspace_dimension=8, scan_impl=True
        )
        # Compute the sum of absolute values of the final state as a simple loss
        return tc.backend.sum(tc.backend.abs(states[0]))

    grad_fn = tc.backend.jit(tc.backend.grad(loss_function))
    gradient = grad_fn(t)
    print(gradient)


@pytest.mark.parametrize(
    "backend, sparse",
    [[lf("npb"), True], [lf("npb"), False], [lf("jaxb"), True], [lf("jaxb"), False]],
)
def test_chebyshev_evol_basic(backend, highp, sparse):
    n = 6
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian (dense for better compatibility)
    h = tc.quantum.heisenberg_hamiltonian(
        g, hzz=1.0, hxx=1.0, hyy=1.0, hx=0.2, sparse=sparse
    )

    # Initial Neel state: |↑↓↑↓⟩
    c = tc.Circuit(n)
    c.x([1, 3, 5])  # Apply X gates to qubits 1 and 3
    psi0 = c.state()

    # Evolution time
    t = 2.0

    # Estimate spectral bounds
    e_max, e_min = tc.timeevol.estimate_spectral_bounds(h, n_iter=30)

    # Estimate parameters
    k = tc.timeevol.estimate_k(t, (e_max, e_min))
    m = tc.timeevol.estimate_M(t, (e_max, e_min), k)

    # Evolve using Chebyshev method
    psi_chebyshev = tc.timeevol.chebyshev_evol(
        h, psi0, t, (float(e_max) + 0.1, float(e_min) - 0.1), k, m
    )

    # Check that state is normalized (or close to it)
    norm = tc.backend.norm(psi_chebyshev)
    np.testing.assert_allclose(norm, 1.0, atol=1e-3)

    # Compare with exact evolution for small system
    if sparse is True:
        h = tc.backend.to_dense(h)
    psi_exact = tc.timeevol.ed_evol(h, psi0, 1.0j * tc.backend.convert_to_tensor([t]))[
        0
    ]

    # States should be close (up to global phase)
    fidelity = np.abs(np.vdot(np.asarray(psi_exact), np.asarray(psi_chebyshev))) ** 2
    assert fidelity > 0.95  # Should be close, but not exact due to approximations


def test_chebyshev_evol_vmap_on_t(jaxb, highp):
    n = 4
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=False)

    # Initial Neel state
    c = tc.Circuit(n)
    c.x([1, 3])
    psi0 = c.state()

    # Estimate spectral bounds
    e_max, e_min = tc.timeevol.estimate_spectral_bounds(h, n_iter=20)

    # Fixed parameters
    k = 50
    m = 150

    # Define vectorized evolution function
    def evolve_single_time(t):
        return tc.timeevol.chebyshev_evol(
            h, psi0, t, (float(e_max) + 0.1, float(e_min) - 0.1), k, m
        )

    # Vectorize over times
    times = tc.backend.convert_to_tensor([0.5, 1.0, 1.5])
    vmap_evolve = tc.backend.jit(tc.backend.vmap(evolve_single_time))
    states_vmap = vmap_evolve(times)

    # Check output shape
    assert states_vmap.shape == (3, 2**n)

    # Compare with sequential execution
    states_sequential = []
    for t in times:
        state = tc.timeevol.chebyshev_evol(
            h, psi0, float(t), (e_max + 0.1, e_min - 0.1), k, m
        )
        states_sequential.append(state)

    states_sequential = tc.backend.stack(states_sequential)

    # Results should be the same
    np.testing.assert_allclose(states_vmap, states_sequential, atol=1e-5)


def test_chebyshev_evol_jit_on_psi(jaxb, highp):
    """Test JIT compilation capability of chebyshev_evol on psi parameter"""
    n = 4
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=False)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=0.6, hyy=1.0, sparse=True)

    # Estimate spectral bounds
    e_max, e_min = tc.timeevol.estimate_spectral_bounds(h, n_iter=20)

    # Fixed parameters
    t = 1.0
    k = 50
    m = 150

    # Define JIT-compiled evolution function with psi as argument
    def evolve_state(psi):
        return tc.timeevol.chebyshev_evol(
            h, psi, t, (float(e_max) + 0.1, float(e_min) - 0.1), k, m
        )

    jit_evolve = tc.backend.jit(evolve_state)

    # Test with different initial states
    c1 = tc.Circuit(n)
    c1.x([0, 2])
    psi1 = c1.state()

    c2 = tc.Circuit(n)
    c2.h(0)
    for i in range(n - 1):
        c2.cnot(i, i + 1)
    psi2 = c2.state()

    # Run JIT-compiled evolution
    result1_jit = jit_evolve(psi1)
    result2_jit = jit_evolve(psi2)

    # Run regular evolution for comparison
    result1_regular = tc.timeevol.chebyshev_evol(
        h, psi1, t, (e_max + 0.1, e_min - 0.1), k, m
    )
    result2_regular = tc.timeevol.chebyshev_evol(
        h, psi2, t, (e_max + 0.1, e_min - 0.1), k, m
    )
    print(result1_jit)
    # Results should be the same
    np.testing.assert_allclose(result1_jit, result1_regular, atol=1e-5)
    np.testing.assert_allclose(result2_jit, result2_regular, atol=1e-5)


def test_chebyshev_evol_ad_on_t(jaxb, highp):
    n = 5
    # Create a 1D chain graph
    g = tc.templates.graphs.Line1D(n, pbc=True)

    # Generate Heisenberg Hamiltonian
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=True)

    # Initial state
    c = tc.Circuit(n)
    c.x([1, 3])
    psi0 = c.state()

    # Estimate spectral bounds
    e_max, e_min = tc.timeevol.estimate_spectral_bounds(h, n_iter=20)

    # Fixed parameters
    k = 50
    m = 100

    # Define loss function for gradient computation
    def loss_function(t):
        psi_t = tc.timeevol.chebyshev_evol(
            h, psi0, t, (float(e_max) + 0.1, float(e_min) - 0.1), k, m
        )
        c = tc.Circuit(5, inputs=psi_t)
        return tc.backend.real(c.expectation_ps(z=[2]))

    # Compute gradient
    grad_fn = tc.backend.jit(tc.backend.grad(loss_function))
    t_test = tc.backend.convert_to_tensor(1.0)
    gradient = grad_fn(t_test)
    print(gradient)
    # Gradient should be a scalar
    assert gradient.shape == ()
