import sys
import os
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_rgate(highp):
    np.testing.assert_almost_equal(
        tc.gates.r_gate(1, 2, 3).tensor, tc.gates.rgate_theoretical(1, 2, 3).tensor
    )


# regression test for jax 0.10.0
def test_builtin_gate_registry_survives_jax_dtype_switch(jaxb):
    @tc.backend.jit
    def f(x):
        return tc.backend.sum(tc.gates.h().tensor) + tc.backend.sum(x)

    v = f(tc.backend.ones([1]))
    assert np.isfinite(float(np.asarray(tc.backend.numpy(tc.backend.real(v)))))

    with tc.runtime_dtype("complex128"):
        h = tc.gates.h()
        assert tc.backend.dtype(h.tensor) == "complex128"


def test_phase_gate():
    c = tc.Circuit(1)
    c.h(0)
    c.phase(0, theta=np.pi / 2)
    np.testing.assert_allclose(c.state()[1], 0.7071j, atol=1e-4)


def test_cu_gate():
    c = tc.Circuit(2)
    c.cu(0, 1, theta=np.pi / 2, phi=-np.pi / 4, lbd=np.pi / 4)
    m = c.matrix()
    print(m)
    np.testing.assert_allclose(m[2:, 2:], tc.gates._wroot_matrix, atol=1e-5)
    np.testing.assert_allclose(m[:2, :2], np.eye(2), atol=1e-5)


def test_get_u_parameter(highp):
    for _ in range(6):
        hermitian = np.random.uniform(size=[2, 2])
        hermitian += np.conj(np.transpose(hermitian))
        unitary = tc.backend.expm(hermitian * 1.0j)
        params = tc.gates.get_u_parameter(unitary)
        unitary2 = tc.gates.u_gate(theta=params[0], phi=params[1], lbd=params[2])
        ans = unitary2.tensor
        unitary = unitary / np.exp(1j * np.angle(unitary[0, 0]))
        np.testing.assert_allclose(unitary, ans, atol=1e-3)


def test_ided_gate():
    g = tc.gates.rx.ided()
    np.testing.assert_allclose(
        tc.backend.reshapem(g(theta=0.3).tensor),
        np.kron(np.eye(2), tc.gates.rx(theta=0.3).tensor),
        atol=1e-5,
    )
    g1 = tc.gates.rx.ided(before=False)
    np.testing.assert_allclose(
        tc.backend.reshapem(g1(theta=0.3).tensor),
        np.kron(tc.gates.rx(theta=0.3).tensor, np.eye(2)),
        atol=1e-5,
    )


def test_fsim_gate():
    theta = 0.2
    phi = 0.3
    c = tc.Circuit(2)
    c.iswap(0, 1, theta=-theta)
    c.cphase(0, 1, theta=-phi)
    m = c.matrix()
    ans = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.95105654 + 0.0j, 0.0 - 0.309017j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 0.309017j, 0.95105654 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.9553365 - 0.29552022j],
        ]
    )
    np.testing.assert_allclose(m, ans, atol=1e-5)
    print(m)


def test_exp_gate():
    c = tc.Circuit(2)
    c.exp(
        0,
        1,
        unitary=tc.gates.array_to_tensor(
            np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        ),
        theta=tc.gates.num_to_tensor(np.pi / 2),
    )
    np.testing.assert_allclose(c.wavefunction()[0], -1j)


def test_any_gate():
    c = tc.Circuit(2)
    c.any(0, unitary=np.eye(2))
    np.testing.assert_allclose(c.expectation((tc.gates.z(), [0])), 1.0)


def test_iswap_gate():
    t = tc.gates.iswap_gate().tensor
    ans = np.array([[1.0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1.0]])
    np.testing.assert_allclose(t, ans.reshape([2, 2, 2, 2]), atol=1e-5)
    t = tc.gates.iswap_gate(theta=0).tensor
    np.testing.assert_allclose(t, np.eye(4).reshape([2, 2, 2, 2]), atol=1e-5)


def test_gate_list():
    assert tc.Circuit.sgates == tc.abstractcircuit.sgates


def test_controlled():
    xgate = tc.gates.x
    cxgate = xgate.controlled()
    ccxgate = cxgate.controlled()
    assert ccxgate.n == "ccx"
    assert ccxgate.ctrl == [1, 1]
    np.testing.assert_allclose(
        ccxgate().tensor, tc.backend.reshape2(tc.gates._toffoli_matrix)
    )
    ocxgate = cxgate.ocontrolled()
    c = tc.Circuit(3)
    c.x(0)
    c.any(1, 0, 2, unitary=ocxgate())
    np.testing.assert_allclose(c.expectation([tc.gates.z(), [2]]), -1, atol=1e-5)
    print(c.to_qir()[1])


def test_variable_controlled():
    crxgate = tc.gates.rx.controlled()
    c = tc.Circuit(2)
    c.x(0)
    tc.Circuit.crx_my = tc.Circuit.apply_general_variable_gate_delayed(crxgate)
    c.crx_my(0, 1, theta=0.3)
    np.testing.assert_allclose(
        c.expectation([tc.gates.z(), [1]]), 0.95533645, atol=1e-5
    )
    assert c.to_qir()[1]["name"] == "crx"


def test_adjoint_gate():
    np.testing.assert_allclose(
        tc.gates.sd().tensor, tc.backend.adjoint(tc.gates._s_matrix)
    )
    assert tc.gates.td.n == "td"


def test_rxx_gate():
    c1 = tc.Circuit(3)
    c1.rxx(0, 1, theta=1.0)
    c1.ryy(0, 2, theta=0.5)
    c1.rzz(0, 1, theta=-0.5)
    c2 = tc.Circuit(3)
    c2.exp1(0, 1, theta=1.0 / 2, unitary=tc.gates._xx_matrix)
    c2.exp1(0, 2, theta=0.5 / 2, unitary=tc.gates._yy_matrix)
    c2.exp1(0, 1, theta=-0.5 / 2, unitary=tc.gates._zz_matrix)
    np.testing.assert_allclose(c1.state(), c2.state(), atol=1e-5)


def test_matrix_for_gate_no_mutation(npb):
    g = tc.gates.ry(np.pi)
    before = tc.backend.copy(g.tensor)
    _ = tc.gates.matrix_for_gate(g)
    after = g.tensor
    assert np.array_equal(before, after)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_batched_unitary_parameterized_gates(backend):
    parameters = tc.backend.convert_to_tensor(
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float32,
        )
    )
    actual = tc.gates.batched_unitary(
        tc.gates.u,
        vectorized_argnames=("theta", "phi", "lbd"),
        theta=parameters[:, 0],
        phi=parameters[:, 1],
        lbd=parameters[:, 2],
    )
    expected = tc.backend.stack(
        [
            tc.backend.reshapem(
                tc.gates.u_gate(
                    theta=parameters[i, 0],
                    phi=parameters[i, 1],
                    lbd=parameters[i, 2],
                ).tensor
            )
            for i in range(3)
        ]
    )
    assert tc.backend.shape_tuple(actual) == (3, 2, 2)
    np.testing.assert_allclose(
        tc.backend.numpy(actual), tc.backend.numpy(expected), atol=1e-6
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_batched_unitary_shared_and_custom_parameters(backend):
    theta = tc.backend.convert_to_tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    unitary = tc.gates.array_to_tensor(tc.gates._zz_matrix)
    actual = tc.gates.batched_unitary(
        tc.gates.exp_gate,
        vectorized_argnames="theta",
        unitary=unitary,
        theta=theta,
    )
    expected = tc.backend.stack(
        [
            tc.backend.reshapem(
                tc.gates.exp_gate(unitary=unitary, theta=theta[i]).tensor
            )
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(
        tc.backend.numpy(actual), tc.backend.numpy(expected), atol=1e-5
    )

    def custom_tensor_gate(theta):
        return tc.backend.reshapem(tc.gates.rx_gate(theta=theta).tensor)

    custom = tc.gates.batched_unitary(
        custom_tensor_gate, vectorized_argnames="theta", theta=theta
    )
    reference = tc.gates.batched_unitary("rx", vectorized_argnames="theta", theta=theta)
    np.testing.assert_allclose(
        tc.backend.numpy(custom), tc.backend.numpy(reference), atol=1e-6
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
def test_batched_unitary_su4(backend):
    theta = tc.backend.convert_to_tensor(
        np.arange(45, dtype=np.float32).reshape(3, 15) / 100
    )
    actual = tc.gates.batched_unitary("su4", vectorized_argnames="theta", theta=theta)
    theta_complex = tc.gates.num_to_tensor(theta)
    pauli_ops = tc.gates.array_to_tensor(
        tc.gates._ix_matrix,
        tc.gates._iy_matrix,
        tc.gates._iz_matrix,
        tc.gates._xi_matrix,
        tc.gates._xx_matrix,
        tc.gates._xy_matrix,
        tc.gates._xz_matrix,
        tc.gates._yi_matrix,
        tc.gates._yx_matrix,
        tc.gates._yy_matrix,
        tc.gates._yz_matrix,
        tc.gates._zi_matrix,
        tc.gates._zx_matrix,
        tc.gates._zy_matrix,
        tc.gates._zz_matrix,
    )
    expected = tc.backend.stack(
        [
            tc.backend.expm(
                -tc.backend.i()
                * tc.backend.sum(
                    tc.backend.stack(
                        [theta_complex[batch, i] * pauli_ops[i] for i in range(15)]
                    ),
                    axis=0,
                )
            )
            for batch in range(3)
        ]
    )
    assert tc.backend.shape_tuple(actual) == (3, 4, 4)
    np.testing.assert_allclose(
        tc.backend.numpy(actual), tc.backend.numpy(expected), atol=1e-5
    )


def test_batched_unitary_jax_autodiff(jaxb):
    theta = tc.backend.convert_to_tensor(
        np.arange(30, dtype=np.float32).reshape(2, 15) / 100
    )

    def batched_loss(parameters):
        matrices = tc.gates.batched_unitary(
            tc.gates.su4,
            vectorized_argnames="theta",
            theta=parameters,
        )
        return tc.backend.real(
            tc.backend.sum(matrices[:, 0, 0] + 0.37 * matrices[:, 0, 1])
        )

    def loop_loss(parameters):
        matrices = tc.backend.stack(
            [
                tc.backend.reshapem(tc.gates.su4_gate(theta=parameters[i]).tensor)
                for i in range(2)
            ]
        )
        return tc.backend.real(
            tc.backend.sum(matrices[:, 0, 0] + 0.37 * matrices[:, 0, 1])
        )

    batched_value, batched_grad = tc.backend.jit(
        tc.backend.value_and_grad(batched_loss)
    )(theta)
    loop_value, loop_grad = tc.backend.jit(tc.backend.value_and_grad(loop_loss))(theta)
    np.testing.assert_allclose(
        tc.backend.numpy(batched_value), tc.backend.numpy(loop_value), atol=1e-5
    )
    np.testing.assert_allclose(
        tc.backend.numpy(batched_grad), tc.backend.numpy(loop_grad), atol=1e-5
    )


def test_batched_unitary_validation(npb):
    with pytest.raises(ValueError, match="at least one"):
        tc.gates.batched_unitary("rx", vectorized_argnames=(), theta=np.ones(2))
    with pytest.raises(ValueError, match="unknown gate"):
        tc.gates.batched_unitary(
            "unknown", vectorized_argnames="theta", theta=np.ones(2)
        )
    with pytest.raises(ValueError, match="missing vectorized"):
        tc.gates.batched_unitary("rx", vectorized_argnames="theta")
    with pytest.raises(ValueError, match="batch dimensions must match"):
        tc.gates.batched_unitary(
            tc.gates.u,
            vectorized_argnames=("theta", "phi"),
            theta=np.ones(2),
            phi=np.ones(3),
        )
