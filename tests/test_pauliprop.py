import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.pauliprop import PauliPropagationEngine, pauli_propagation


def test_initialization(jaxb):
    N, k = 4, 3
    pp = PauliPropagationEngine(N, k)
    assert pp.num_subsets == 4  # 4C3 = 4
    assert pp.subset_arr.shape == (4, 3)


def test_initial_state(jaxb):
    N, k = 3, 2
    pp = PauliPropagationEngine(N, k)
    # H = 0.5 Z0 + 0.3 Z1 Z2
    structures = np.array([[3, 0, 0], [0, 3, 3]])
    weights = np.array([0.5, 0.3])
    state = pp.get_initial_state(structures, weights)

    # State shape (3, 16)
    assert state[0, 12] == 0.5
    assert state[2, 15] == 0.3


def test_apply_gate_1q(jaxb):
    N, k = 2, 2
    pp = PauliPropagationEngine(N, k)
    # H = Z0
    state = pp.get_initial_state(np.array([[3, 0]]), np.array([1.0]))
    # Apply X0. X Z X = -Z.
    state = pp.apply_gate(state, "x", [0])
    assert jnp.allclose(state[0, 12], -1.0)


def test_expectation_correctness(jaxb):
    N, k = 2, 2
    c = tc.Circuit(N)
    c.h(0)
    c.cx(0, 1)

    obs = [(1.0, "ZZ")]
    res = pauli_propagation(c, obs, k=k)

    exact = c.expectation_ps(z=[0, 1])
    assert jnp.allclose(res, exact, atol=1e-5)


def test_gradients(jaxb):
    N, k = 3, 3

    def loss(theta):
        c = tc.Circuit(N)
        c.rx(0, theta=theta[0])
        c.ry(1, theta=theta[1])
        c.cx(0, 1)
        obs = [(1.0, "ZZI")]
        res = pauli_propagation(c, obs, k=k)
        return jnp.real(res).real

    theta = jnp.array([0.1, 0.2])
    grad_fn = jax.grad(loss)
    g = grad_fn(theta)

    def loss_exact(theta):
        c = tc.Circuit(N)
        c.rx(0, theta=theta[0])
        c.ry(1, theta=theta[1])
        c.cx(0, 1)
        return jnp.real(c.expectation_ps(z=[0, 1])).real

    g_exact = jax.grad(loss_exact)(theta)
    assert jnp.allclose(g, g_exact, atol=1e-5)


def test_light_cone_exactness(jaxb):
    N = 6
    k = 4
    c = tc.Circuit(N)
    for i in range(3):
        c.cx(i, i + 1)
    obs = [(1.0, "ZZIIII")]

    res = pauli_propagation(c, obs, k=k)
    exact = c.expectation_ps(z=[0, 1])
    assert jnp.allclose(res, exact, atol=1e-5)


def test_hamiltonian_complex(jaxb):
    N, k = 4, 2
    structures = np.array([[3, 0, 0, 0], [0, 3, 0, 0], [3, 3, 0, 0]])
    weights = np.array([1.0, 1.0, 0.5])

    pp = PauliPropagationEngine(N, k)
    state = pp.get_initial_state(structures, weights)

    assert jnp.allclose(pp.expectation(state), 2.5)

    state = pp.apply_gate(state, "x", [0])
    assert jnp.allclose(pp.expectation(state), -0.5)


def test_scan_interface(jaxb):
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

    params = jnp.ones((2, N))
    val_scan = pp.compute_expectation_scan(ham_struct, ham_weights, layer, params)

    state = pp.get_initial_state(ham_struct, ham_weights)
    for l in [1, 0]:
        c_l = tc.Circuit(N)
        layer(c_l, params[l])
        ops = c_l.to_qir()
        for op in reversed(ops):
            state = pp.apply_gate(state, op["name"], op["index"], op.get("parameters"))

    val_manual = pp.expectation(state)
    assert jnp.allclose(val_scan, val_manual)


def test_pauli_propagation_struct_weights(jaxb):
    N, k = 3, 2
    c = tc.Circuit(N)
    c.h(0)
    c.cx(0, 1)

    # H = Z0 Z1
    structures = np.array([[3, 3, 0]])
    weights = np.array([1.0])

    res = pauli_propagation(c, structures, weights, k=k)
    exact = c.expectation_ps(z=[0, 1])
    assert jnp.allclose(res, exact, atol=1e-5)
