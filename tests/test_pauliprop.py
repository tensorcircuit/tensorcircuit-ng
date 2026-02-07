import numpy as np
import pytest
import tensorcircuit as tc
from tensorcircuit.pauliprop import PauliPropagationEngine, pauli_propagation


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_initialization(request, backend):
    request.getfixturevalue(backend)
    N, k = 4, 3
    pp = PauliPropagationEngine(N, k)
    # Check if basis size is correct
    # P_k = 1 (empty) + 4*3 (1-loc) + 6*9 (2-loc) + 4*27 (3-loc) = 1 + 12 + 54 + 108 = 175
    assert pp.dim == 175
    assert pp.neighbor_map.shape == (175 + 1, 4, 4)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_string_to_code(request, backend):
    request.getfixturevalue(backend)
    N, k = 4, 2
    pp = PauliPropagationEngine(N, k)
    from tensorcircuit.pauliprop import SparsePauliPropagationEngine

    spp = SparsePauliPropagationEngine(N, k)

    s = ((0, 1), (3, 3))  # Z0 Z1
    idx = pp.string_to_code(s)
    assert pp.basis[idx] == s

    code = spp.string_to_code(s)
    # Z is 3. Bit pattern for Z at q=0 and q=1:
    # word 0: (3 << 0) | (3 << 2) = 3 | 12 = 15
    assert tc.backend.numpy(code)[0] == 15


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_initial_state(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 3, 2
    pp = PauliPropagationEngine(N, k)
    # H = 0.5 Z0 + 0.3 Z1 Z2
    structures = np.array([[3, 0, 0], [0, 3, 3]])
    weights = np.array([0.5, 0.3])
    state = pp.get_initial_state(structures, weights)

    state_np = K.numpy(state)

    idx1 = pp.string_to_idx[((0,), (3,))]
    idx2 = pp.string_to_idx[((1, 2), (3, 3))]
    assert np.isclose(state_np[idx1], 0.5)
    assert np.isclose(state_np[idx2], 0.3)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_apply_gate_1q(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 2, 2
    pp = PauliPropagationEngine(N, k)
    # H = Z0
    state = pp.get_initial_state(np.array([[3, 0]]), np.array([1.0]))
    # Apply X0. X Z X = -Z.
    state = pp.apply_gate(state, "x", [0])
    idx = pp.string_to_idx[((0,), (3,))]
    assert np.allclose(K.numpy(state)[idx], -1.0)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_expectation_correctness(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 2, 2
    c = tc.Circuit(N)
    c.h(0)
    c.cx(0, 1)
    c.ry(1, theta=0.3)
    c.rxx(0, 1, theta=-1.2)

    obs = [(1.0, "ZZ")]
    res = pauli_propagation(c, obs, k=k)

    exact = c.expectation_ps(z=[0, 1])
    assert np.allclose(K.numpy(res), K.numpy(exact), atol=1e-5)


@pytest.mark.parametrize("backend", ["jaxb"])
def test_gradients(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 3, 3

    def loss(theta):
        c = tc.Circuit(N)
        c.rx(0, theta=theta[0])
        c.ryy(1, 0, theta=theta[1])
        c.cx(0, 1)
        obs = [(1.0, "ZZI")]
        res = pauli_propagation(c, obs, k=k)
        return K.real(res)

    theta = K.convert_to_tensor(np.array([0.1, 0.2]))
    grad_fn = K.grad(loss)
    g = grad_fn(theta)

    def loss_exact(theta):
        c = tc.Circuit(N)
        c.rx(0, theta=theta[0])
        c.ryy(1, 0, theta=theta[1])
        c.cx(0, 1)
        return K.real(c.expectation_ps(z=[0, 1]))

    g_exact = K.grad(loss_exact)(theta)
    assert np.allclose(K.numpy(g), K.numpy(g_exact), atol=1e-5)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_light_cone_exactness(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N = 6
    k = 4
    c = tc.Circuit(N)
    for i in range(3):
        c.cx(i, i + 1)
        c.rxx(i + 1, i, theta=0.6)
    obs = [(1.0, "ZZIIII")]

    res = pauli_propagation(c, obs, k=k)
    exact = c.expectation_ps(z=[0, 1])
    assert np.allclose(K.numpy(res), K.numpy(exact), atol=1e-5)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_hamiltonian_complex(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 4, 2
    structures = np.array([[3, 0, 0, 0], [0, 3, 0, 0], [3, 3, 0, 0]])
    weights = np.array([1.0, 1.0, 0.5])

    pp = PauliPropagationEngine(N, k)
    state = pp.get_initial_state(structures, weights)

    assert np.allclose(K.numpy(pp.expectation(state)), 2.5)

    state = pp.apply_gate(state, "x", [0])
    assert np.allclose(K.numpy(pp.expectation(state)), -0.5)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_scan_interface(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 4, 3
    pp = PauliPropagationEngine(N, k)

    def layer(c, params):
        for i in range(N):
            c.rx(i, theta=params[i])
        for i in range(N - 1):
            c.cx(i, i + 1)

    ham_struct = np.zeros((N, N), dtype=int)
    for i in range(N):
        ham_struct[i, i] = 3
    ham_weights = np.ones(N)

    params = K.convert_to_tensor(np.ones((2, N)), dtype="complex64")
    val_scan = pp.compute_expectation_scan(ham_struct, ham_weights, layer, params)

    state = pp.get_initial_state(ham_struct, ham_weights)
    # Reverse the order manually for manual verification
    for l in [1, 0]:
        c_l = tc.Circuit(N)
        layer(c_l, params[l])
        ops = c_l.to_qir()
        for op in reversed(ops):
            state = pp.apply_gate(state, op["name"], op["index"], op.get("parameters"))

    val_manual = pp.expectation(state)
    assert np.allclose(K.numpy(val_scan), K.numpy(val_manual), atol=1e-5)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_pauli_propagation_struct_weights(request, backend):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 3, 2
    c = tc.Circuit(N)
    c.h(0)
    c.cx(0, 1)

    # H = Z0 Z1
    structures = np.array([[3, 3, 0]])
    weights = np.array([1.0])

    res = pauli_propagation(c, structures, weights, k=k)
    exact = c.expectation_ps(z=[0, 1])
    assert np.allclose(K.numpy(res), K.numpy(exact), atol=1e-5)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_subset_jump(request, backend):
    """
    Specifically test the case where a Pauli string moves between subsets.
    In k=2, Z1 Z3 ->(SWAP 3,4)-> Z1 Z4.
    The old implementation would fail here because subset {1, 3} does not contain qubit 4.
    """
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 5, 2

    # Starting with Z1 Z4
    # We apply SWAP(3, 4) in the circuit.
    # Heisenberg picture (backward): SWAP(3, 4) (Z1 Z4) SWAP(3, 4) = Z1 Z3
    c = tc.Circuit(N)
    c.swap(3, 4)

    # We want to measure Z1 Z4 at the end.
    # In PPE, we initialize with Z1 Z4 and propagate backward.
    obs = [(1.0, "IZIIZ")]  # Z1 Z4

    res = pauli_propagation(c, obs, k=k)
    # The expectation value should be 1.0 (identity state)
    assert np.allclose(K.numpy(res), 1.0)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_sparse_engine_match_comprehensive(request, backend, highp):
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 4, 3
    buffer_size = 500
    from tensorcircuit.pauliprop import SparsePauliPropagationEngine

    dense_pp = PauliPropagationEngine(N, k)
    sparse_pp = SparsePauliPropagationEngine(N, k, buffer_size=buffer_size)

    # Initial state: Z0 + 0.5 Z1 Z2
    structures = np.zeros((2, N), dtype=int)
    structures[0, 0] = 3
    structures[1, 1] = 3
    structures[1, 2] = 3
    weights = K.convert_to_tensor(np.array([1.0, 0.5]), dtype="complex64")

    s_dense = dense_pp.get_initial_state(structures, weights)
    s_sparse = sparse_pp.get_initial_state(structures, weights)

    # Apply a variety of gates
    theta = 0.4
    gates = [
        ("rx", (0,), {"theta": theta}),
        ("ryy", (1, 2), {"theta": -0.3}),
        ("cnot", (0, 1), {}),
        ("rzz", (0, 3), {"theta": 0.5}),
        ("rxx", (2, 3), {"theta": 1.2}),
        ("h", (1,), {}),
    ]

    for name, wires, params in gates:
        s_dense = dense_pp.apply_gate(s_dense, name, wires, params)
        s_sparse = sparse_pp.apply_gate(s_sparse, name, wires, params)

    e_dense = dense_pp.expectation(s_dense)
    e_sparse = sparse_pp.expectation(s_sparse)

    assert np.allclose(K.numpy(e_dense), K.numpy(e_sparse), atol=1e-5)


@pytest.mark.parametrize("backend", ["jaxb"])
def test_sparse_engine_scan_match(request, backend, highp):
    """Verify Sparse engine works with compute_expectation_scan."""
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 4, 2
    buffer_size = 200
    from tensorcircuit.pauliprop import SparsePauliPropagationEngine

    sparse_pp = SparsePauliPropagationEngine(N, k, buffer_size=buffer_size)
    dense_pp = PauliPropagationEngine(N, k)

    def layer(c, params):
        for i in range(N - 1):
            c.rxx(i, i + 1, theta=params[i])

    ham_struct = np.zeros((N, N), dtype=int)
    for i in range(N):
        ham_struct[i, i] = 3
    ham_weights = np.ones(N)

    params = K.convert_to_tensor(np.ones((2, N - 1)), dtype="complex64")

    val_scan_sparse = sparse_pp.compute_expectation_scan(
        ham_struct, ham_weights, layer, params
    )
    val_scan_dense = dense_pp.compute_expectation_scan(
        ham_struct, ham_weights, layer, params
    )

    assert np.allclose(K.numpy(val_scan_sparse), K.numpy(val_scan_dense), atol=1e-5)


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_truncation_analytical(request, backend, highp):
    """
    Directly verify that for insufficient k, the approximation is exactly the same
    as the analytical derived Pauli string and then truncated.
    """
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 3, 1
    theta = 0.6

    # Initial state: Z1
    # Apply RXX(0, 1)
    # Heisenberg: exp(i theta/2 X0 X1) Z1 exp(-i theta/2 X0 X1)
    # = cos(theta) Z1 + sin(theta) Y1 X0
    # Truncated to k=1: cos(theta) Z1

    # 1. Dense Engine
    pp = PauliPropagationEngine(N, k)
    state = pp.get_initial_state(np.array([[0, 3, 0]]), np.array([1.0]))
    state = pp.apply_gate(state, "rxx", [0, 1], {"theta": theta})

    idx_z1 = pp.string_to_code(((1,), (3,)))
    val = K.numpy(state)[idx_z1]
    assert np.allclose(val, np.cos(theta), atol=1e-5)

    # All other k=1 terms should be 0
    # The total weight should be |cos(theta)|^2 ? No, the state vector stores coefficients.
    # Sum of absolute values of k=1 terms should be |cos(theta)|
    # Exclude SINK index
    assert np.allclose(
        np.sum(np.abs(K.numpy(state[:-1]))), np.abs(np.cos(theta)), atol=1e-5
    )

    # 2. Sparse Engine
    from tensorcircuit.pauliprop import SparsePauliPropagationEngine

    spp = SparsePauliPropagationEngine(N, k, buffer_size=100)
    s_state = spp.get_initial_state(np.array([[0, 3, 0]]), np.array([1.0]))
    s_state = spp.apply_gate(s_state, "rxx", [0, 1], {"theta": theta})

    # Expectation on |0> state sums only Z strings
    # Z1 is the only Z string here
    assert np.allclose(K.numpy(spp.expectation(s_state)), np.cos(theta), atol=1e-5)

    # Verify that only the Z1 term remains with correct coefficient
    # (Sparse engine might have multiple entries or empty ones, but aggregate_and_truncate cleans it)
    s_codes, s_coeffs = s_state
    assert K.numpy(s_codes[0]) == spp.string_to_code(((1,), (3,)))
    assert np.allclose(
        K.numpy(K.sum(K.abs(s_coeffs))), np.abs(np.cos(theta)), atol=1e-5
    )


@pytest.mark.parametrize("backend", ["npb", "jaxb"])
def test_truncation_and_buffer_sparse(request, backend, highp):
    """
    Verify Sparse engine truncation for both k (locality) and buffer_size (number of strings).
    """
    request.getfixturevalue(backend)
    K = tc.backend
    N, k = 4, 2
    buffer_size = 2

    from tensorcircuit.pauliprop import SparsePauliPropagationEngine

    spp = SparsePauliPropagationEngine(N, k, buffer_size=buffer_size)

    # Initial: Z0
    # Apply RX(0, theta1) -> cos(theta1) Z0 + sin(theta1) Y0
    # Apply RXX(0, 1, theta2) ->
    # cos(theta1) [cos(theta2) Z0 + sin(theta2) Y0 X1]  (loc 1, loc 2)
    # + sin(theta1) [cos(theta2) Y0 - sin(theta2) Z0 X1] (loc 1, loc 2)
    # All are loc <= 2, so k=2 doesn't truncate yet.
    # Total terms: 4.
    # But buffer_size = 2. It should keep the 2 terms with largest coefficients.

    theta1 = 0.4
    theta2 = 0.8
    s_state = spp.get_initial_state(np.array([[3, 0, 0, 0]]), np.array([1.0]))
    s_state = spp.apply_gate(s_state, "rx", [0], {"theta": theta1})
    s_state = spp.apply_gate(s_state, "rxx", [0, 1], {"theta": theta2})

    # Expected coefficients before buffer truncation:
    # c1 = cos(theta1)cos(theta2)  (Z0)
    # c2 = cos(theta1)sin(theta2)  (Y0 X1)
    # c3 = sin(theta1)cos(theta2)  (Y0)
    # c4 = -sin(theta1)sin(theta2) (Z0 X1)

    c = np.array(
        [
            np.cos(theta1) * np.cos(theta2),
            np.cos(theta1) * np.sin(theta2),
            np.sin(theta1) * np.cos(theta2),
            -np.sin(theta1) * np.sin(theta2),
        ]
    )
    c_abs = np.abs(c)
    top2_idx = np.argsort(c_abs)[-2:]
    expected_sum_abs = np.sum(c_abs[top2_idx])

    s_codes, s_coeffs = s_state
    print(s_codes)
    assert np.allclose(K.numpy(K.sum(K.abs(s_coeffs))), expected_sum_abs, atol=1e-5)
