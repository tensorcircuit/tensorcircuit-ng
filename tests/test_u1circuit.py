import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import tensorcircuit as tc

# We will implement this soon
from tensorcircuit.u1circuit import U1Circuit


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_correctness(backend):
    # n=4, k=2, filled=[0, 1] -> |1100>
    sim = U1Circuit(4, 2, filled=[0, 1])

    # Test initial expectation
    # |1100> -> Z0=-1, Z1=-1, Z2=1, Z3=1
    assert np.allclose(tc.backend.numpy(sim.expectation_z(0)), -1.0)
    assert np.allclose(tc.backend.numpy(sim.expectation_z(2)), 1.0)

    # Test Swap
    sim.swap(1, 2)  # |1010>
    assert np.allclose(tc.backend.numpy(sim.expectation_z(1)), 1.0)
    assert np.allclose(tc.backend.numpy(sim.expectation_z(2)), -1.0)

    # Test Rz (phase only)
    sim.rz(0, theta=np.pi)  # |1010> -> -|1010> (global phase doesn't change Z)
    assert np.allclose(tc.backend.numpy(sim.expectation_z(0)), -1.0)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_u1circuit_ad_gradients(backend):
    def loss(theta):
        sim = U1Circuit(2, 1, filled=[0])  # |10>
        sim.iswap(0, 1, theta=theta)
        # |10> -> cos(theta)|10> + i sin(theta)|01>
        # <Z0> = cos^2(theta)*(-1) + sin^2(theta)*(1) = -cos(2*theta)
        return tc.backend.real(sim.expectation_z(0))

    vg = tc.backend.value_and_grad(loss)
    theta_v = tc.backend.convert_to_tensor(0.5, dtype="float32")
    val, grad = vg(theta_v)

    # Expected Val: -cos(theta_v * pi) approx -0.58778 (for theta=0.5, pi*theta=pi/2, so -cos(pi/2)=0)
    # Wait, if theta=0.5, th*pi = pi/2, -cos(pi/2) = 0.
    # d/dth (-cos(th*pi)) = pi * sin(th*pi). For th=0.5, pi * sin(pi/2) = pi.
    assert np.allclose(
        tc.backend.numpy(val), -np.cos(tc.backend.numpy(theta_v) * np.pi), atol=1e-4
    )
    assert np.allclose(
        tc.backend.numpy(grad),
        np.pi * np.sin(tc.backend.numpy(theta_v) * np.pi),
        atol=1e-4,
    )


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_u1circuit_jit(backend):
    @tc.backend.jit
    def run(theta):
        sim = U1Circuit(4, 2, filled=[0, 1])
        sim.iswap(1, 2, theta=theta)
        return tc.backend.real(sim.expectation_z(1))

    res = run(tc.backend.convert_to_tensor(0.2, dtype="float32"))
    assert res is not None


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_u1circuit_vmap(backend):
    def run(theta):
        sim = U1Circuit(2, 1, filled=[0])
        sim.iswap(0, 1, theta=theta)
        return tc.backend.real(sim.expectation_z(0))

    vrun = tc.backend.vmap(run)
    thetas = tc.backend.convert_to_tensor([0.0, 0.5, 1.0], dtype="float32")
    results = vrun(thetas)
    assert np.allclose(tc.backend.numpy(results), [-1.0, 0.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb")])
def test_u1circuit_errors(backend):
    sim = U1Circuit(2, 1, filled=[0])
    with pytest.raises(ValueError, match=r"not U\(1\) conserving"):
        sim.x(0)
    with pytest.raises(ValueError, match=r"not U\(1\) conserving"):
        sim.h(1)
    with pytest.raises(ValueError, match=r"not U\(1\) conserving"):
        sim.rx(0, theta=0.1)

    # Test filled validation
    with pytest.raises(ValueError, match="match length of 'filled'"):
        U1Circuit(4, k=1, filled=[0, 1])
    with pytest.raises(ValueError, match="Duplicate indices"):
        U1Circuit(4, filled=[0, 0])
    with pytest.raises(ValueError, match="out of range"):
        U1Circuit(4, filled=[2, 4])

    # Test expectation error
    with pytest.raises(NotImplementedError, match="Please use expectation_ps"):
        sim.expectation([tc.gates.z(), [0]])

    # Test initialization with only k
    sim3 = U1Circuit(4, k=2)
    assert sim3.circuit_param["filled"] == [0, 1]
    assert np.allclose(tc.backend.numpy(sim3.expectation_ps(z=[0, 1])), 1.0)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_diagonal_gate(backend):
    # Test case 1: 2 qubits
    n, k = 2, 1
    filled = [0]
    diag = tc.backend.convert_to_tensor([1.0, 2.0, 3.0, 4.0])

    u1c = U1Circuit(n, k, filled=filled)
    u1c.diagonal(0, 1, diag=diag)

    c = tc.Circuit(n)
    c.x(0)
    c.diagonal(0, 1, diag=diag)

    assert np.allclose(
        tc.backend.numpy(u1c.to_dense()),
        tc.backend.numpy(tc.backend.reshape(c.state(), [-1])),
        atol=1e-5,
    )

    # Test case 2: 3 qubits
    n, k = 4, 2
    filled = [0, 2]
    diag3 = tc.backend.convert_to_tensor(np.exp(1j * np.random.rand(8)))

    u1c3 = U1Circuit(n, k, filled=filled)
    u1c3.diagonal(0, 1, 2, diag=diag3)

    c3 = tc.Circuit(n)
    c3.x(0)
    c3.x(2)
    c3.diagonal(0, 1, 2, diag=diag3)

    assert np.allclose(
        tc.backend.numpy(u1c3.to_dense()),
        tc.backend.numpy(tc.backend.reshape(c3.state(), [-1])),
        atol=1e-5,
    )

    # Test case 3: 4 qubits
    n, k = 5, 2
    filled = [1, 3]
    diag4 = tc.backend.convert_to_tensor(np.exp(1j * np.random.rand(16)))

    u1c4 = U1Circuit(n, k, filled=filled)
    u1c4.diagonal(0, 1, 2, 4, diag=diag4)

    c4 = tc.Circuit(n)
    c4.x(1)
    c4.x(3)
    c4.diagonal(0, 1, 2, 4, diag=diag4)

    assert np.allclose(
        tc.backend.numpy(u1c4.to_dense()),
        tc.backend.numpy(tc.backend.reshape(c4.state(), [-1])),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_expectation_odd_y(backend):
    # Odd number of Y in Pauli string
    n, k = 4, 2
    filled = [0, 1]

    u1c = U1Circuit(n, k, filled=filled)
    c = tc.Circuit(n)
    c.x(0)
    c.x(1)

    # Create some entanglement
    u1c.iswap(0, 2, theta=0.5)
    c.iswap(0, 2, theta=0.5)
    u1c.iswap(1, 3, theta=0.7)
    c.iswap(1, 3, theta=0.7)

    ps = {"y": [0, 2], "z": [1]}
    assert np.allclose(
        tc.backend.numpy(u1c.expectation_ps(**ps)),
        tc.backend.numpy(c.expectation_ps(**ps)),
        atol=1e-5,
    )

    ps_odd = {"y": [0], "x": [1]}
    assert np.abs(tc.backend.numpy(u1c.expectation_ps(**ps_odd))) < 1e-6
    assert np.abs(tc.backend.numpy(c.expectation_ps(**ps_odd))) < 1e-6

    ps_complex = {"y": [0, 1, 2], "x": [3]}
    assert np.allclose(
        tc.backend.numpy(u1c.expectation_ps(**ps_complex)),
        tc.backend.numpy(c.expectation_ps(**ps_complex)),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_expectation_ps(backend):
    # |1010>
    sim = U1Circuit(4, 2, filled=[0, 2])
    # Z0 * Z1 = -1 * 1 = -1
    # Note: expectation_ps in TC usually returns complex
    res = sim.expectation_ps(z=[0, 1])
    assert np.allclose(tc.backend.numpy(res), -1.0, atol=1e-5)

    # test X0Y1 - Y0X1 (non-zero for entangling states)
    sim.iswap(0, 1, theta=np.pi / 4)
    # |10> -> 1/sqrt(2) (|10> - i|01>)
    # <X0Y1> = <1/sqrt(2)(|10> - i|01|) | X0Y1 | 1/sqrt(2)(|10> - i|01|)>
    # X0Y1 |10> = |0>(-i)|0> = -i|00> NO, Y1 |0> = i|1>
    # Correct mapping:
    # |10> -> Z0=-1, Z1=1
    # |01> -> Z0=1, Z1=-1
    # <X0Y1 - Y0X1> ... let's just check non-zero
    val = sim.expectation_ps(x=[0], y=[1]) - sim.expectation_ps(y=[0], x=[1])
    assert np.abs(tc.backend.numpy(val)) > 0.1


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_vs_circuit_comparison(backend):
    """Compare U1Circuit with regular Circuit for various gate sequences."""
    n = 4
    k = 2

    # Test 1: Simple RZ + CZ sequence
    u1c = U1Circuit(n, k, filled=[0, 1])
    c = tc.Circuit(n)
    c.X(0)
    c.X(1)

    # Apply same gates
    u1c.rz(0, theta=0.5)
    c.rz(0, theta=0.5)
    u1c.cz(0, 1)
    u1c.iswap(0, 2, theta=1.9)
    u1c.cz(2, 3)
    u1c.rzz(2, 0, theta=0.3)
    u1c.cphase(1, 2, theta=0.8)
    c.cz(0, 1)
    c.iswap(0, 2, theta=1.9)
    c.cz(2, 3)
    c.rzz(2, 0, theta=0.3)
    c.cphase(1, 2, theta=0.8)

    # Compare Z expectations
    for i in range(n):
        u1_exp = tc.backend.numpy(u1c.expectation_z(i))
        c_exp = tc.backend.numpy(c.expectation([tc.gates.z(), [i]]))
        assert np.allclose(u1_exp, c_exp, atol=1e-5), f"Mismatch at qubit {i}"

    # Test 2: Swap gates
    u1c2 = U1Circuit(n, filled=[1, 2])  # k inferred from filled
    c2 = tc.Circuit(n)
    c2.X(1)
    c2.X(2)

    u1c2.swap(1, 2)
    c2.swap(1, 2)

    for i in range(n):
        u1_exp = tc.backend.numpy(u1c2.expectation_z(i))
        c_exp = tc.backend.numpy(c2.expectation([tc.gates.z(), [i]]))
        assert np.allclose(u1_exp, c_exp, atol=1e-5)

    # Test 3: ISwap gates
    u1c3 = U1Circuit(n, k, filled=[0, 3])
    c3 = tc.Circuit(n)
    c3.X(0)
    c3.X(3)

    u1c3.iswap(0, 3)
    c3.iswap(0, 3)
    u1c3.cphase(1, 3)
    c3.cphase(1, 3)

    for i in range(n):
        u1_exp = tc.backend.numpy(u1c3.expectation_z(i))
        c_exp = tc.backend.numpy(c3.expectation([tc.gates.z(), [i]]))
        assert np.allclose(u1_exp, c_exp, atol=1e-5)

    # Test 4: Complex sequence with RZZ and CPhase
    u1c4 = U1Circuit(n, k, filled=[1, 3])
    c4 = tc.Circuit(n)
    c4.X(1)
    c4.X(3)

    u1c4.rzz(1, 3, theta=0.7)
    c4.rzz(1, 3, theta=0.7)
    u1c4.rz(1, theta=0.3)
    c4.rz(1, theta=0.3)
    u1c4.cphase(1, 3, theta=0.5)
    c4.cphase(1, 3, theta=0.5)

    for i in range(n):
        u1_exp = tc.backend.numpy(u1c4.expectation_z(i))
        c_exp = tc.backend.numpy(c4.expectation([tc.gates.z(), [i]]))
        assert np.allclose(u1_exp, c_exp, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb")])
def test_u1circuit_init_flexibility(backend):
    """Test that U1Circuit can be initialized with either k or filled."""
    n = 4

    # Test 1: Only k provided
    u1c1 = U1Circuit(n, k=2)
    assert u1c1._k == 2
    assert u1c1._dim == 6  # C(4,2) = 6

    # Test 2: Only filled provided (k inferred)
    u1c2 = U1Circuit(n, filled=[0, 1, 2])
    assert u1c2._k == 3
    assert u1c2._dim == 4  # C(4,3) = 4

    # Test 3: Both provided (should use k, verify consistency)
    u1c3 = U1Circuit(n, k=2, filled=[1, 3])
    assert u1c3._k == 2

    # Test 4: Neither provided (should raise error)
    with pytest.raises(ValueError, match="Either 'k' or 'filled' must be provided"):
        U1Circuit(n)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_to_dense(backend):
    """Test to_dense method converts U(1) state to full Hilbert space."""
    n = 4
    k = 2

    # Test 1: Initial state |1100> (filled=[0,1]) should have amplitude 1 at index 12
    # TC uses reversed bit ordering: qubit 0 is leftmost (highest) bit
    u1c = U1Circuit(n, k, filled=[0, 1])
    dense = u1c.to_dense()
    dense_np = tc.backend.numpy(dense)

    assert dense.shape == (2**n,)
    assert np.allclose(np.abs(dense_np[12]), 1.0)  # |1100> = 12 in TC ordering
    assert np.allclose(np.sum(np.abs(dense_np) ** 2), 1.0)  # Normalized

    # Test 2: After swap gate, compare with regular Circuit
    u1c2 = U1Circuit(n, k, filled=[0, 2])  # |0101> = 5
    c = tc.Circuit(n)
    c.X(0)
    c.X(2)

    u1c2.swap(0, 1)
    c.swap(0, 1)

    dense2 = u1c2.to_dense()
    c_state = c.state()
    c_state_flat = tc.backend.reshape(c_state, [-1])

    assert np.allclose(
        tc.backend.numpy(dense2), tc.backend.numpy(c_state_flat), atol=1e-5
    )

    # Test 3: After iswap gate, compare expectation values
    u1c3 = U1Circuit(n, k, filled=[1, 2])
    c3 = tc.Circuit(n)
    c3.X(1)
    c3.X(2)

    u1c3.iswap(1, 2, theta=-1.3)
    c3.iswap(1, 2, theta=-1.3)

    for i in range(n):
        u1_exp = tc.backend.numpy(u1c3.expectation_z(i))
        c_exp = tc.backend.numpy(c3.expectation([tc.gates.z(), [i]]))
        assert np.allclose(u1_exp, c_exp, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_probability(backend):
    """Test probability methods."""
    n = 4
    k = 2

    # Test 1: Initial state should have probability 1 at one basis state
    u1c = U1Circuit(n, k, filled=[0, 1])
    p = u1c.probability()
    p_np = tc.backend.numpy(p)

    assert p.shape == (6,)  # C(4,2) = 6 basis states
    assert np.allclose(np.sum(p_np), 1.0)
    assert np.sum(p_np > 0.5) == 1  # Only one state has probability 1

    # Test 2: probability_full should have same structure but in 2^n space
    p_full = u1c.probability_full()
    p_full_np = tc.backend.numpy(p_full)

    assert p_full.shape == (2**n,)
    assert np.allclose(np.sum(p_full_np), 1.0)
    # Only 6 (= C(4,2)) states should have non-zero probability potential
    # but initially only 1 state is occupied
    assert np.sum(p_full_np > 0.5) == 1


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_sample(backend):
    """Test sample method."""
    n = 4
    k = 2

    # Test 1: Single sample (default)
    u1c = U1Circuit(n, k, filled=[0, 1])
    sample = u1c.sample()
    binary, _ = sample

    assert binary.shape == (n,)
    # Check that sample has exactly k=2 ones (U(1) conservation)
    assert np.sum(tc.backend.numpy(binary)) == k

    # Test 2: Batch sampling
    samples = u1c.sample(batch=10)
    assert len(samples) == 10
    for s, _ in samples:
        assert np.sum(tc.backend.numpy(s)) == k

    # Test 3: Sample with format='sample_int'
    samples_int = u1c.sample(batch=100, format="sample_int")
    samples_int_np = tc.backend.numpy(samples_int)

    # All samples should be valid k-particle states
    for s in samples_int_np:
        # Count bits set to 1
        bit_count = bin(int(s)).count("1")
        assert bit_count == k


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_measure(backend):
    """Test measure method."""
    n = 4
    k = 2

    # Test 1: Measure from known state
    u1c = U1Circuit(n, k, filled=[0, 1])  # |0011>
    outcome, prob = u1c.measure(0, 1, with_prob=True)

    outcome_np = tc.backend.numpy(outcome)
    # Qubit 0 and 1 are filled, so should measure 1, 1
    assert np.allclose(outcome_np, [1.0, 1.0])
    assert np.allclose(tc.backend.numpy(prob), 1.0)

    # Test 2: After superposition, outcomes vary
    u1c2 = U1Circuit(n, k, filled=[0, 1])
    u1c2.iswap(0, 2, theta=np.pi / 4)  # Create superposition

    # Run multiple measurements and check U(1) conservation
    for _ in range(10):
        outcome, _ = u1c2.measure(0, 1, 2, 3)
        outcome_np = tc.backend.numpy(outcome)
        # Total should always be k=2
        assert np.sum(outcome_np) == k


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_dtype_agnostic(backend, highp):
    """Test that U1Circuit works with complex128 dtype."""
    n = 4
    k = 2

    # With highp fixture, dtype is complex128
    u1c = U1Circuit(n, k, filled=[0, 1])
    u1c.iswap(0, 1, theta=0.5)
    exp = tc.backend.numpy(u1c.expectation_z(0))

    # Verify result is precise
    np.testing.assert_allclose(exp, -1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_state_matching_parameterized_gates(backend):
    """Test state matching between U1Circuit and Circuit with parameterized gates."""
    n = 6
    k = 3

    # Various theta values to test
    thetas = [0.1, 0.5, 1.0, np.pi / 4, np.pi / 2]

    for theta in thetas:
        # Test RZZ with different theta
        u1c = U1Circuit(n, k, filled=[0, 2, 4])
        c = tc.Circuit(n)
        for i in [0, 2, 4]:
            c.x(i)

        u1c.rzz(0, 1, theta=theta)
        c.rzz(0, 1, theta=theta)

        u1_dense = tc.backend.numpy(u1c.to_dense())
        c_state = tc.backend.numpy(tc.backend.reshape(c.state(), [-1]))
        assert np.allclose(
            u1_dense, c_state, atol=1e-5
        ), f"RZZ mismatch at theta={theta}"

        # Test RZ with different theta
        u1c2 = U1Circuit(n, k, filled=[1, 3, 5])
        c2 = tc.Circuit(n)
        for i in [1, 3, 5]:
            c2.x(i)

        u1c2.rz(1, theta=theta)
        c2.rz(1, theta=theta)

        u1_dense2 = tc.backend.numpy(u1c2.to_dense())
        c_state2 = tc.backend.numpy(tc.backend.reshape(c2.state(), [-1]))
        assert np.allclose(
            u1_dense2, c_state2, atol=1e-5
        ), f"RZ mismatch at theta={theta}"

        # Test iswap with different theta
        u1c3 = U1Circuit(n, k, filled=[0, 1, 2])
        c3 = tc.Circuit(n)
        for i in [0, 1, 2]:
            c3.x(i)

        u1c3.iswap(1, 2, theta=theta)
        c3.iswap(1, 2, theta=theta)

        u1_dense3 = tc.backend.numpy(u1c3.to_dense())
        c_state3 = tc.backend.numpy(tc.backend.reshape(c3.state(), [-1]))
        assert np.allclose(
            u1_dense3, c_state3, atol=1e-5
        ), f"iswap mismatch at theta={theta}"

        # Test cphase with different theta
        u1c4 = U1Circuit(n, k, filled=[0, 3, 5])
        c4 = tc.Circuit(n)
        for i in [0, 3, 5]:
            c4.x(i)

        u1c4.cphase(0, 3, theta=theta)
        c4.cphase(0, 3, theta=theta)

        u1_dense4 = tc.backend.numpy(u1c4.to_dense())
        c_state4 = tc.backend.numpy(tc.backend.reshape(c4.state(), [-1]))
        assert np.allclose(
            u1_dense4, c_state4, atol=1e-5
        ), f"cphase mismatch at theta={theta}"


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_expectation_ps_comprehensive(backend):
    """Comprehensive test of expectation_ps comparing U1Circuit with Circuit."""
    n = 6
    k = 3

    # Create entangled state using U(1) conserving gates only
    filled = [0, 2, 4]
    u1c = U1Circuit(n, k, filled=filled)
    c = tc.Circuit(n)
    for i in filled:
        c.x(i)

    # Apply U(1) conserving gates (excluding fsim since Circuit doesn't have it)
    u1c.iswap(0, 1, theta=0.5)
    c.iswap(0, 1, theta=0.5)

    u1c.iswap(2, 3, theta=0.7)
    c.iswap(2, 3, theta=0.7)

    u1c.rzz(4, 5, theta=0.4)
    c.rzz(4, 5, theta=0.4)

    u1c.cphase(0, 2, theta=0.3)
    c.cphase(0, 2, theta=0.3)

    # Test Z expectations
    for i in range(n):
        u1_z = tc.backend.numpy(u1c.expectation_ps(z=[i]))
        c_z = tc.backend.numpy(c.expectation_ps(z=[i]))
        assert np.allclose(u1_z, c_z, atol=1e-5), f"Z[{i}] mismatch"

    # Test ZZ expectations
    for i in range(n - 1):
        u1_zz = tc.backend.numpy(u1c.expectation_ps(z=[i, i + 1]))
        c_zz = tc.backend.numpy(c.expectation_ps(z=[i, i + 1]))
        assert np.allclose(u1_zz, c_zz, atol=1e-5), f"ZZ[{i},{i+1}] mismatch"

    # Test XX expectations (requires flipping bits - only valid if result stays in U1 sector)
    # XX on adjacent qubits with different occupation
    u1_xx = tc.backend.numpy(u1c.expectation_ps(x=[0, 1]))
    c_xx = tc.backend.numpy(c.expectation_ps(x=[0, 1]))
    assert np.allclose(u1_xx, c_xx, atol=1e-5), "XX[0,1] mismatch"

    # Test YY expectations
    u1_yy = tc.backend.numpy(u1c.expectation_ps(y=[0, 1]))
    c_yy = tc.backend.numpy(c.expectation_ps(y=[0, 1]))
    assert np.allclose(u1_yy, c_yy, atol=1e-5), "YY[0,1] mismatch"

    # Test XY expectations
    u1_xy = tc.backend.numpy(u1c.expectation_ps(x=[0], y=[1]))
    c_xy = tc.backend.numpy(c.expectation_ps(x=[0], y=[1]))
    assert np.allclose(u1_xy, c_xy, atol=1e-5), "XY[0,1] mismatch"

    # Test longer Pauli strings
    u1_xyz = tc.backend.numpy(u1c.expectation_ps(x=[0], y=[1], z=[2]))
    c_xyz = tc.backend.numpy(c.expectation_ps(x=[0], y=[1], z=[2]))
    assert np.allclose(u1_xyz, c_xyz, atol=1e-5), "XYZ mismatch"


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_all_gate_types(backend):
    """Test all supported gate types with state comparison."""
    n = 4
    k = 2

    gate_tests = [
        # (gate_name, qubit_indices, kwargs, filled_qubits)
        ("rz", [0], {"theta": 0.5}, [0, 1]),
        ("rz", [1], {"theta": -0.3}, [0, 2]),
        ("rzz", [0, 1], {"theta": 0.7}, [0, 1]),
        ("rzz", [1, 2], {"theta": -0.4}, [1, 3]),
        ("cz", [0, 1], {}, [0, 1]),
        ("cz", [1, 3], {}, [1, 3]),
        ("cphase", [0, 1], {"theta": 0.6}, [0, 1]),
        ("cphase", [2, 3], {"theta": -0.2}, [0, 3]),
        ("swap", [0, 1], {}, [0, 2]),
        ("swap", [1, 2], {}, [1, 3]),
        ("iswap", [0, 1], {"theta": 0.5}, [0, 2]),
        ("iswap", [2, 3], {"theta": 1.0}, [0, 3]),
    ]

    for gate_name, indices, kwargs, filled in gate_tests:
        u1c = U1Circuit(n, k, filled=filled)
        c = tc.Circuit(n)
        for i in filled:
            c.x(i)

        # Apply gate to both circuits
        getattr(u1c, gate_name)(*indices, **kwargs)
        getattr(c, gate_name)(*indices, **kwargs)
        # Compare states
        u1_dense = tc.backend.numpy(u1c.to_dense())
        c_state = tc.backend.numpy(tc.backend.reshape(c.state(), [-1]))
        assert np.allclose(
            u1_dense, c_state, atol=1e-5
        ), f"State mismatch for {gate_name}({indices}, {kwargs}) with filled={filled}"


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_u1circuit_filled_jit(backend):
    # This test verifies that different 'filled' tensors do not trigger re-compilation
    compilation_count = 0

    def get_expectation(filled):
        nonlocal compilation_count
        compilation_count += 1
        # Create U1Circuit inside JIT
        c = U1Circuit(4, 2, filled=filled)
        return c.expectation_ps(z=[0])

    f_jit = tc.backend.jit(get_expectation)

    # First call: compiles
    res1 = f_jit(tc.backend.convert_to_tensor([0, 1], dtype="int32"))
    assert compilation_count == 1
    assert np.allclose(tc.backend.numpy(res1), -1.0)

    # Second call: should use cached JIT-ted function
    res2 = f_jit(tc.backend.convert_to_tensor([1, 2], dtype="int32"))
    assert compilation_count == 1
    # |0110> -> Z0=1, Z1=-1, Z2=-1, Z3=1
    assert np.allclose(tc.backend.numpy(res2), 1.0)

    # Third call: another variation
    res3 = f_jit(tc.backend.convert_to_tensor([0, 2], dtype="int32"))
    assert compilation_count == 1
    # |1010> -> Z0=-1, Z1=1, Z2=-1, Z3=1
    assert np.allclose(tc.backend.numpy(res3), -1.0)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_different_particle_numbers(backend):
    """Test U1Circuit with different particle numbers k."""
    n = 6

    for k in [1, 2, 3, 4, 5]:
        # Create filled list with k particles
        filled = list(range(k))

        u1c = U1Circuit(n, k=k, filled=filled)
        c = tc.Circuit(n)
        for i in filled:
            c.x(i)

        # Apply some gates
        if k >= 2:
            u1c.swap(0, 1)
            c.swap(0, 1)
            u1c.rzz(0, 1, theta=0.5)
            c.rzz(0, 1, theta=0.5)

        u1c.rz(0, theta=0.3)
        c.rz(0, theta=0.3)

        # Compare states
        u1_dense = tc.backend.numpy(u1c.to_dense())
        c_state = tc.backend.numpy(tc.backend.reshape(c.state(), [-1]))
        assert np.allclose(u1_dense, c_state, atol=1e-5), f"Mismatch for k={k}"


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_gradient_with_exact_circuit(backend):
    n = 4
    k = 2

    def loss(param):
        u1c = U1Circuit(n, k, filled=[0, 1])
        u1c.rzz(0, 2, theta=param[0])
        u1c.rzz(1, 3, theta=param[1])
        u1c.iswap(0, 1, theta=param[2])
        u1c.iswap(2, 3, theta=param[3])
        u1c.rz(0, theta=param[4])
        u1c.rz(1, theta=param[5])
        return tc.backend.real(u1c.expectation_ps(z=[0]))

    def loss2(param):
        c = tc.Circuit(n)
        for i in [0, 1]:
            c.x(i)
        c.rzz(0, 2, theta=param[0])
        c.rzz(1, 3, theta=param[1])
        c.iswap(0, 1, theta=param[2])
        c.iswap(2, 3, theta=param[3])
        c.rz(0, theta=param[4])
        c.rz(1, theta=param[5])
        return tc.backend.real(c.expectation_ps(z=[0]))

    grad1 = tc.backend.grad(loss)(tc.backend.ones([6]))
    grad2 = tc.backend.grad(loss2)(tc.backend.ones([6]))
    assert np.allclose(grad1, grad2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_inverse(backend):
    # n=4, k=2, filled=[0, 1] -> |1100>
    sim = U1Circuit(4, 2, filled=[0, 1])
    sim.iswap(0, 1, theta=0.8)
    # verify k is propagated and state is NOT (starts from zero/filled state)
    inv_sim = sim.inverse()
    assert inv_sim.circuit_param.get("k") == 2
    assert "filled" in inv_sim.circuit_param
    # verify functional inversion
    sim.append(inv_sim)
    # After U and U^dagger, state should be back to |1100>
    assert np.allclose(tc.backend.numpy(sim.expectation_z(0)), -1.0, atol=1e-5)
    assert np.allclose(tc.backend.numpy(sim.expectation_z(2)), 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_rdm(backend):
    n, k = 4, 2
    filled = [0, 1]
    subsystem = [0, 1]

    u1c = U1Circuit(n, k, filled=filled)
    u1c.iswap(0, 2, theta=0.5)
    u1c.iswap(1, 3, theta=0.3)

    # 1. Test RDM blocks
    blocks = u1c.reduced_density_matrix(subsystem, return_blocks=True)
    # kA can be 0, 1, 2
    # But for n=4, k=2, if we keep [0, 1], kA can only be:
    # kA=2 (k_B=0), kA=1 (k_B=1), kA=0 (k_B=2)
    assert len(blocks) <= 3
    for b in blocks:
        assert b.shape[0] == b.shape[1]
        # Trace of each block is the probability of that sector
        assert tc.backend.numpy(tc.backend.real(tc.backend.trace(b))) >= 0

    # 2. Test full RDM vs dense
    rdm_u1 = u1c.reduced_density_matrix(subsystem)

    c = tc.Circuit(n)
    c.x(0)
    c.x(1)
    c.iswap(0, 2, theta=0.5)
    c.iswap(1, 3, theta=0.3)

    traceout = [2, 3]
    rdm_dense = tc.quantum.reduced_density_matrix(
        tc.backend.reshape(c.state(), [-1]), traceout
    )

    assert np.allclose(tc.backend.numpy(rdm_u1), tc.backend.numpy(rdm_dense), atol=1e-5)

    # 3. Test non-contiguous subsystem
    subsystem2 = [0, 2]
    rdm_u1_2 = u1c.reduced_density_matrix(subsystem2)
    traceout2 = [1, 3]
    rdm_dense_2 = tc.quantum.reduced_density_matrix(
        tc.backend.reshape(c.state(), [-1]), traceout2
    )
    assert np.allclose(
        tc.backend.numpy(rdm_u1_2), tc.backend.numpy(rdm_dense_2), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1circuit_entropy(backend):
    n, k = 6, 3
    subsystem = [0, 1, 2]

    u1c = U1Circuit(n, k, filled=[0, 1, 2])
    u1c.iswap(0, 3, theta=0.5)
    u1c.iswap(1, 4, theta=0.5)
    u1c.iswap(2, 5, theta=0.5)

    entropy = u1c.entanglement_entropy(subsystem)

    c = tc.Circuit(n)
    for i in range(3):
        c.x(i)
    c.iswap(0, 3, theta=0.5)
    c.iswap(1, 4, theta=0.5)
    c.iswap(2, 5, theta=0.5)

    rdm_dense = tc.quantum.reduced_density_matrix(
        tc.backend.reshape(c.state(), [-1]), [3, 4, 5]
    )
    entropy_dense = tc.quantum.entropy(rdm_dense)

    assert np.allclose(
        tc.backend.numpy(entropy), tc.backend.numpy(entropy_dense), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_u1circuit_entropy_vmap(backend):
    def run(theta):
        c = U1Circuit(4, 2, filled=[0, 1])
        c.iswap(0, 2, theta=theta)
        return tc.backend.real(c.entanglement_entropy([0, 1]))

    vrun = tc.backend.vmap(run)
    results = vrun(tc.backend.convert_to_tensor([0.0, 0.5], dtype="float32"))
    assert results.shape == (2,)
    assert np.allclose(tc.backend.numpy(results[0]), 0.0, atol=1e-5)
    assert tc.backend.numpy(results[1]) > 0.0


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_u1circuit_entropy_ad(backend):
    def loss_u1(theta):
        u1c = U1Circuit(4, 2, filled=[0, 1])
        u1c.iswap(0, 2, theta=theta)
        return tc.backend.real(u1c.entanglement_entropy([0, 1]))

    def loss_circuit(theta):
        c = tc.Circuit(4)
        c.x(0)
        c.x(1)
        c.iswap(0, 2, theta=theta)
        rdm = tc.quantum.reduced_density_matrix(
            tc.backend.reshape(c.state(), [-1]), [2, 3]
        )
        return tc.backend.real(tc.quantum.entropy(rdm))

    vg_u1 = tc.backend.value_and_grad(loss_u1)
    vg_circuit = tc.backend.value_and_grad(loss_circuit)

    theta_v = tc.backend.convert_to_tensor(0.2, dtype="float32")
    val_u1, grad_u1 = vg_u1(theta_v)
    val_c, grad_c = vg_circuit(theta_v)

    assert np.allclose(tc.backend.numpy(val_u1), tc.backend.numpy(val_c), atol=1e-5)
    assert np.allclose(tc.backend.numpy(grad_u1), tc.backend.numpy(grad_c), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_u1circuit_expectation_ps_ad(backend):
    def loss_u1(theta):
        u1c = U1Circuit(4, 2, filled=[0, 1])
        u1c.iswap(0, 2, theta=theta)
        # Use non-overlapping indices for regular circuit comparison
        return tc.backend.real(u1c.expectation_ps(x=[0], y=[1], z=[2]))

    def loss_circuit(theta):
        c = tc.Circuit(4)
        c.x(0)
        c.x(1)
        c.iswap(0, 2, theta=theta)
        return tc.backend.real(c.expectation_ps(x=[0], y=[1], z=[2]))

    vg_u1 = tc.backend.value_and_grad(loss_u1)
    vg_circuit = tc.backend.value_and_grad(loss_circuit)

    theta_v = tc.backend.convert_to_tensor(0.3, dtype="float32")
    val_u1, grad_u1 = vg_u1(theta_v)
    val_c, grad_c = vg_circuit(theta_v)

    assert np.allclose(tc.backend.numpy(val_u1), tc.backend.numpy(val_c), atol=1e-5)
    assert np.allclose(tc.backend.numpy(grad_u1), tc.backend.numpy(grad_c), atol=1e-5)
