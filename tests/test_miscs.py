# pylint: disable=invalid-name

import sys
import os
from functools import partial
import numpy as np
import tensorflow as tf
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit import experimental
from tensorcircuit.quantum import PauliString2COO, PauliStringSum2COO
from tensorcircuit.applications.vqes import construct_matrix_v2
from tensorcircuit.applications.physics.baseline import TFIM1Denergy, Heisenberg1Denergy

i, x, y, z = [t.tensor for t in tc.gates.pauli_gates]

# note i is in use!

check_pairs = [
    ([0, 0], np.eye(4)),
    ([0, 1], np.kron(i, x)),
    ([2, 1], np.kron(y, x)),
    ([3, 1], np.kron(z, x)),
    ([3, 2, 2, 0], np.kron(np.kron(np.kron(z, y), y), i)),
    ([0, 1, 1, 1], np.kron(np.kron(np.kron(i, x), x), x)),
]


def test_about():
    print(tc.about())


def test_cite():
    print(tc.cite())
    print(tc.cite("aps"))


def test_ps2coo(tfb):
    for l, a in check_pairs:
        r1 = PauliString2COO(tf.constant(l, dtype=tf.int64))
        np.testing.assert_allclose(tc.backend.to_dense(r1), a, atol=1e-5)


def test_pss2coo(tfb):
    l = [t[0] for t in check_pairs[:4]]
    a = sum([t[1] for t in check_pairs[:4]])
    r1 = PauliStringSum2COO(tf.constant(l, dtype=tf.int64))
    np.testing.assert_allclose(tc.backend.to_dense(r1), a, atol=1e-5)
    l = [t[0] for t in check_pairs[4:]]
    a = sum([t[1] for t in check_pairs[4:]])
    r1 = PauliStringSum2COO(tf.constant(l, dtype=tf.int64), weight=[0.5, 1])
    a = check_pairs[4][1] * 0.5 + check_pairs[5][1] * 1.0
    np.testing.assert_allclose(tc.backend.to_dense(r1), a, atol=1e-5)


def test_sparse(benchmark, tfb):
    def sparse(h):
        return PauliStringSum2COO(h)

    h = [[1 for _ in range(12)], [2 for _ in range(12)]]
    h = tf.constant(h, dtype=tf.int64)
    sparse(h)
    benchmark(sparse, h)


def test_dense(benchmark, tfb):
    def dense(h):
        return construct_matrix_v2(h, dtype=tf.complex64)

    h = [[1 for _ in range(12)], [2 for _ in range(12)]]
    h = [[1.0] + hi for hi in h]
    dense(h)
    benchmark(dense, h)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_adaptive_vmap(backend):
    def f(x):
        return x**2

    x = tc.backend.ones([30, 2])

    vf = experimental.adaptive_vmap(f, chunk_size=6)
    np.testing.assert_allclose(vf(x), tc.backend.ones([30, 2]), atol=1e-5)

    vf2 = experimental.adaptive_vmap(f, chunk_size=7)
    np.testing.assert_allclose(vf2(x), tc.backend.ones([30, 2]), atol=1e-5)

    def f2(x):
        return tc.backend.sum(x)

    vf3 = experimental.adaptive_vmap(f2, chunk_size=7)
    np.testing.assert_allclose(vf3(x), 2 * tc.backend.ones([30]), atol=1e-5)

    vf3_jit = tc.backend.jit(vf3)
    np.testing.assert_allclose(vf3_jit(x), 2 * tc.backend.ones([30]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_adaptive_vmap_mul_io(backend):
    def f(x, y, a):
        return x + y + a

    vf = experimental.adaptive_vmap(f, chunk_size=6, vectorized_argnums=(0, 1))
    x = tc.backend.ones([30, 2])
    a = tc.backend.ones([2])
    # jax vmap has some weird behavior in terms of keyword arguments...
    # TODO(@refraction-ray): further investigate jax vmap behavior with kwargs
    np.testing.assert_allclose(vf(x, x, a), 3 * tc.backend.ones([30, 2]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_qng(backend):
    n = 6

    def f(params):
        params = tc.backend.reshape(params, [4, n])
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, params)
        return c.state()

    params = tc.backend.ones([4 * n])
    fim = experimental.qng(f)(params)
    assert tc.backend.shape_tuple(fim) == (4 * n, 4 * n)
    print(experimental.dynamics_matrix(f)(params))


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dynamic_rhs(backend):
    h1 = tc.array_to_tensor(tc.gates._z_matrix)

    def f(param):
        c = tc.Circuit(1)
        c.rx(0, theta=param)
        return c.state()

    rhsf = experimental.dynamics_rhs(f, h1)
    np.testing.assert_allclose(rhsf(tc.backend.ones([])), -np.sin(1.0) / 2, atol=1e-5)

    h2 = tc.backend.coo_sparse_matrix(
        indices=tc.array_to_tensor(np.array([[0, 0], [1, 1]]), dtype="int64"),
        values=tc.array_to_tensor(np.array([1, -1])),
        shape=[2, 2],
    )

    rhsf = experimental.dynamics_rhs(f, h2)
    np.testing.assert_allclose(rhsf(tc.backend.ones([])), -np.sin(1.0) / 2, atol=1e-5)


@pytest.mark.parametrize("backend", ["tensorflow", "jax"])
def test_two_qng_approaches(backend):
    n = 6
    nlayers = 2
    with tc.runtime_backend(backend) as K:
        with tc.runtime_dtype("complex128"):

            def state(params):
                params = K.reshape(params, [2 * nlayers, n])
                c = tc.Circuit(n)
                c = tc.templates.blocks.example_block(c, params, nlayers=nlayers)
                return c.state()

            params = K.ones([2 * nlayers * n])
            params = K.cast(params, "float32")
            n1 = experimental.qng(state)(params)
            n2 = experimental.qng2(state)(params)
            np.testing.assert_allclose(n1, n2, atol=1e-7)


def test_arg_alias():
    @partial(tc.utils.arg_alias, alias_dict={"theta": ["alpha", "gamma"]})
    def f(theta: float, beta: float) -> float:
        """
        f doc

        :param theta: theta angle
        :type theta: float
        :param beta: beta angle
        :type beta: float
        :return: sum angle
        :rtype: float
        """
        return theta + beta

    np.testing.assert_allclose(f(beta=0.2, alpha=0.1), 0.3, atol=1e-5)
    print(f.__doc__)
    assert len(f.__doc__.strip().split("\n")) == 12

    @partial(tc.utils.arg_alias, alias_dict={"theta": "alpha"})
    def g(theta: float) -> float:
        """
        g doc

        :param theta: theta angle
        :type theta: float
        :return: theta
        """
        return theta

    np.testing.assert_allclose(g(alpha=1.0), 1.0, atol=1e-5)
    assert "alpha: alias for the argument ``theta``" in g.__doc__


def test_finite_difference_tf(tfb):
    def f(param1, param2):
        n = 4
        c = tc.Circuit(n)
        for i in range(n):
            c.rx(i, theta=param1[i])
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=param2[i])
        r = [c.expectation_ps(z=[i]) for i in range(n)]
        return tc.backend.stack(r)

    def fsum(param1, param2):
        return tc.backend.mean(f(param1, param2))

    p1 = tf.ones([4])
    p2 = tf.ones([3])
    g1, g2 = tc.backend.value_and_grad(fsum)(p1, p2)

    f1 = experimental.finite_difference_differentiator(
        f, argnums=(0, 1), shifts=(np.pi / 2, 2)
    )

    def fsum1(param1, param2):
        return tc.backend.mean(f1(param1, param2))

    g3, g4 = tc.backend.value_and_grad(fsum1)(p1, p2)

    np.testing.assert_allclose(g1, g3, atol=1e-5)
    np.testing.assert_allclose(g2, g4, atol=1e-5)


def test_energy_baseline():
    print(TFIM1Denergy(10))
    print(Heisenberg1Denergy(10))


def test_jax_function_load(jaxb, tmp_path):
    K = tc.backend

    @K.jit
    def f(weights):
        c = tc.Circuit(3)
        c.rx(range(3), theta=weights)
        return K.real(c.expectation_ps(z=[0]))

    print(f(K.ones([3])))

    experimental.jax_jitted_function_save(
        os.path.join(tmp_path, "temp.bin"), f, K.ones([3])
    )

    f_load = tc.experimental.jax_jitted_function_load(
        os.path.join(tmp_path, "temp.bin")
    )
    np.testing.assert_allclose(f_load(K.ones([3])), 0.5403, atol=1e-4)


def test_distrubuted_contractor(jaxb):
    def nodes_fn(params):
        c = tc.Circuit(4)
        c.rx(range(4), theta=params["x"])
        c.cnot([0, 1, 2], [1, 2, 3])
        c.ry(range(4), theta=params["y"])
        return c.expectation_before([tc.gates.z(), [-1]], reuse=False)

    params = {"x": np.ones([4]), "y": 0.3 * np.ones([4])}
    dc = experimental.DistributedContractor(
        nodes_fn,
        params,
        {
            "slicing_reconf_opts": {"target_size": 2**3},
            "max_repeats": 8,
            "minimize": "write",
            "parallel": False,
        },
    )
    value, grad = dc.value_and_grad(params)
    assert grad["y"].shape == (4,)

    def baseline(params):
        c = tc.Circuit(4)
        c.rx(range(4), theta=params["x"])
        c.cnot([0, 1, 2], [1, 2, 3])
        c.ry(range(4), theta=params["y"])
        return c.expectation_ps(z=[-1])

    np.testing.assert_allclose(value, baseline(params), atol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_runtime_nodes_capture(backend):
    with tc.cons.runtime_nodes_capture() as captured:
        c = tc.Circuit(3)
        c.h(0)
        c.amplitude("010")
    len(captured["nodes"]) == 7


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_function_nodes_capture(backend):
    @tc.cons.function_nodes_capture
    def exp(theta):
        c = tc.Circuit(3)
        c.h(0)
        return c.expectation_ps(z=[-3], reuse=False)

    assert len(exp(0.3)) == 9


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_parameter_shift_grad(backend):
    def f(params):
        c = tc.Circuit(2)
        c.rx(0, theta=params[0])
        c.ry(1, theta=params[1])
        c.cnot(0, 1)
        return tc.backend.real(c.expectation_ps(z=[0, 1]))

    params = tc.array_to_tensor(np.array([0.1, 0.2]))

    # Standard AD gradient
    g_ad = tc.backend.grad(f)(params)

    # Parameter shift gradient
    g_ps = experimental.parameter_shift_grad(f)(params)

    np.testing.assert_allclose(g_ad, g_ps, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_parameter_shift_grad_v2(backend):
    # v2 is mainly for jax and supports randomness
    def f(params):
        c = tc.Circuit(2)
        c.rx(0, theta=params[0])
        c.ry(1, theta=params[1])
        return tc.backend.real(c.expectation_ps(z=[0]))

    params = tc.array_to_tensor(np.array([0.5, 0.5]))
    g_ps = experimental.parameter_shift_grad_v2(f)(params)
    g_ad = tc.backend.grad(f)(params)
    np.testing.assert_allclose(g_ps, g_ad, atol=1e-5)


def test_broadcast_py_object_single_process(jaxb):
    # In a single process environment, broadcast should just return the object
    # though it uses jax.experimental.multihost_utils.broadcast_one_to_all
    obj = {"a": 1, "b": [1, 2, 3]}
    res = experimental.broadcast_py_object(obj)
    assert res == obj


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_jax_jitted_function_save_load_v2(backend, tmp_path):
    K = tc.backend

    @K.jit
    def f(x):
        return x**2 + 1.0

    x = K.ones([2])
    path = os.path.join(tmp_path, "f.bin")
    experimental.jax_jitted_function_save(path, f, x)

    f_load = experimental.jax_jitted_function_load(path)
    np.testing.assert_allclose(f_load(x), f(x), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_qng_options(backend):
    def f(params):
        c = tc.Circuit(1)
        c.rx(0, theta=params[0])
        return c.state()

    params = tc.backend.ones([1])
    # test different options in qng to hit more lines
    qng_fn = experimental.qng(f, mode="fwd")
    qng_fn(params)

    qng_fn2 = experimental.qng(f, mode="rev")
    qng_fn2(params)

    qng_fn3 = experimental.qng(f, kernel="dynamics", postprocess=None)
    qng_fn3(params)


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_qng2_options(backend):
    def f(params):
        c = tc.Circuit(1)
        c.rx(0, theta=params[0])
        return c.state()

    params = tc.backend.ones([1])
    qng_fn = experimental.qng2(f, mode="fwd")
    qng_fn(params)

    qng_fn2 = experimental.qng2(f, mode="rev")
    qng_fn2(params)

    qng_fn3 = experimental.qng2(f, kernel="dynamics", postprocess=None)
    qng_fn3(params)


def test_vis_extra():
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    tex = tc.vis.qir2tex(c.to_qir(), 2)
    assert "\\qw" in tex

    assert tc.vis.gate_name_trans("ccnot") == (2, "not")
    assert tc.vis.gate_name_trans("h") == (0, "h")


def test_cons_extra(jaxb):
    # set_function_backend
    @tc.cons.set_function_backend("jax")
    def f():
        return tc.backend.name

    # set_function_dtype
    @tc.cons.set_function_dtype("complex128")
    def g():
        return tc.dtypestr


def test_ascii_art():
    # hit some lines in asciiart.py

    try:
        tc.set_ascii("wrong")
    except AttributeError:
        pass

    # lucky() is only available after set_ascii
    assert not hasattr(tc, "lucky")


def test_utils_extra():
    from tensorcircuit import utils

    # return_partial
    f = lambda x: [x, x**2, x**3]
    f1 = utils.return_partial(f, return_argnums=1)
    assert f1(2) == 4
    f2 = utils.return_partial(f, return_argnums=[0, 2])
    assert f2(2) == (2, 8)

    # append
    f3 = utils.append(lambda x: x**2, lambda x: x + 1)
    assert f3(2) == 5

    # is_m1mac
    utils.is_m1mac()

    # is_sequence, is_number
    assert utils.is_sequence([1])
    assert utils.is_number(1.0)

    # benchmark
    def h(x):
        return x + 1

    utils.benchmark(h, 1.0, tries=2)
