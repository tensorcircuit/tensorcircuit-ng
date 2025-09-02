# pylint: disable=invalid-name

import os
import sys

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

# see https://stackoverflow.com/questions/56307329/how-can-i-parametrize-tests-to-run-with-different-fixtures-in-pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_basics(backend):
    c = tc.QuditCircuit(2, 3)
    c.x(0)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("10")), np.array(1.0))
    c.csum(0, 1)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("11")), np.array(1.0))
    c.csum(0, 1)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("12")), np.array(1.0))

    c = tc.QuditCircuit(2, 3)
    c.x(0)
    c.x(0)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("20")), np.array(1.0))
    c.csum(0, 1, cv=1)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("21")), np.array(0.0))
    c.csum(0, 1, cv=2)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("21")), np.array(1.0))


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_measure(backend):
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    c.x(1)
    c.csum(0, 1)
    assert c.measure(1)[0] in [0, 1, 2]


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_large_scale_sample(backend):
    L = 30
    d = 3
    c = tc.QuditCircuit(L, d)
    c.h(0)
    for i in range(L - 1):
        c.csum(i, i + 1)

    batch = 1024
    results = c.sample(
        allow_state=False, batch=batch, format="count_dict_bin", jittable=False
    )

    k0, k1, k2 = "0" * L, "1" * L, "2" * L
    c0, c1, c2 = results.get(k0, 0), results.get(k1, 0), results.get(k2, 0)
    assert c0 + c1 + c2 == batch

    probs = np.array([c0, c1, c2], dtype=float) / batch
    np.testing.assert_allclose(probs, np.ones(3) / 3, rtol=0.2, atol=0.0)

    for a, b in [(c0, c1), (c1, c2), (c0, c2)]:
        ratio = (a + 1e-12) / (b + 1e-12)
        assert 0.8 <= ratio <= 1.25


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_expectation(backend):
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    np.testing.assert_allclose(
        tc.backend.numpy(c.expectation((tc.quditgates._z_matrix_func(3), [0]))),
        0,
        atol=1e-7,
    )


def test_complex128(highp, tfb):
    c = tc.QuditCircuit(2, 3)
    c.h(1)
    c.rx(0, theta=1j)
    c.wavefunction()
    np.testing.assert_allclose(
        c.expectation((tc.quditgates._z_matrix_func(3), [1])), 0, atol=1e-15
    )


def test_single_qubit():
    c = tc.QuditCircuit(1, 10)
    c.h(0)
    w = c.state()[0]
    np.testing.assert_allclose(
        w,
        np.array(
            [
                1,
            ]
            * 10
        )
        / np.sqrt(10),
        atol=1e-4,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_expectation_between_two_states_qudit(backend):
    dim = 3
    X3 = tc.quditgates._x_matrix_func(dim)
    Y3 = tc.quditgates._y_matrix_func(dim)  # ZX/i
    Z3 = tc.quditgates._z_matrix_func(dim)
    H3 = tc.quditgates._h_matrix_func(dim)
    X3_dag = np.conjugate(X3.T)

    e0 = np.array([1.0, 0.0, 0.0], dtype=np.complex64)
    e1 = np.array([0.0, 1.0, 0.0], dtype=np.complex64)
    val = tc.expectation((tc.gates.Gate(Y3), [0]), ket=e0, bra=e1, dim=dim)
    omega = np.exp(2j * np.pi / dim)
    expected = omega / 1j
    np.testing.assert_allclose(tc.backend.numpy(val), expected, rtol=1e-6, atol=1e-6)

    c = tc.QuditCircuit(3, dim)
    c.unitary(0, unitary=tc.gates.Gate(H3))
    c.ry(1, theta=0.8, j=0, k=1)
    state = c.wavefunction()
    x1z2 = [(tc.gates.Gate(X3), [0]), (tc.gates.Gate(Z3), [1])]
    e1 = c.expectation(*x1z2)
    e2 = tc.expectation(*x1z2, ket=state, bra=state, normalization=True, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e2), tc.backend.numpy(e1))

    c = tc.QuditCircuit(3, dim)
    c.unitary(0, unitary=tc.gates.Gate(H3))
    c.ry(1, theta=0.8 + 0.7j, j=0, k=1)
    state = c.wavefunction()
    e1 = c.expectation(*x1z2) / (tc.backend.norm(state) ** 2)
    e2 = tc.expectation(*x1z2, ket=state, normalization=True, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e2), tc.backend.numpy(e1))

    c1 = tc.QuditCircuit(2, dim)
    c1.unitary(1, unitary=tc.gates.Gate(X3))
    s1 = c1.state()

    c2 = tc.QuditCircuit(2, dim)
    c2.unitary(0, unitary=tc.gates.Gate(X3))
    s2 = c2.state()

    c3 = tc.QuditCircuit(2, dim)
    c3.unitary(1, unitary=tc.gates.Gate(H3))
    s3 = c3.state()

    x1x2_fixed = [(tc.gates.Gate(X3), [0]), (tc.gates.Gate(X3_dag), [1])]
    e = tc.expectation(*x1x2_fixed, ket=s1, bra=s2, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e), 1.0)

    e2 = tc.expectation(*x1x2_fixed, ket=s3, bra=s2, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e2), 1.0 / np.sqrt(3))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("cpb")])
def test_any_inputs_state_qudit_true_gates(backend):
    dim = 3
    Xd = tc.quditgates._x_matrix_func(dim)
    Zd = tc.quditgates._z_matrix_func(dim)
    omega = np.exp(2j * np.pi / dim)

    def idx(j0, j1):
        return dim * j0 + j1

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(2, 0)] = 1.0
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), 1.0 + 0j, rtol=1e-6, atol=1e-6)

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(0, 0)] = 1.0
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), omega, rtol=1e-6, atol=1e-6)

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(1, 0)] = 1.0
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), omega**2, rtol=1e-6, atol=1e-6)

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(0, 0)] = 1 / np.sqrt(2)
    vec[idx(1, 0)] = 1 / np.sqrt(2)
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), -0.5 + 0j, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("cpb")])
def test_postselection(backend):
    c = tc.QuditCircuit(3, 3)
    c.h(1)
    c.h(2)
    c.mid_measurement(1, 1)
    c.mid_measurement(2, 1)
    s = c.wavefunction()
    np.testing.assert_allclose(tc.backend.numpy(s[4]).real, 1.0 / 3.0, rtol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_unitary(backend):
    c = tc.QuditCircuit(2, dim=3, inputs=np.eye(9))
    c.x(0)
    c.z(1)
    answer = tc.backend.numpy(
        np.kron(tc.quditgates._x_matrix_func(3), tc.quditgates._z_matrix_func(3))
    )
    np.testing.assert_allclose(
        tc.backend.numpy(c.wavefunction().reshape([9, 9])), answer, atol=1e-4
    )


def test_probability():
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    c.h(1)
    np.testing.assert_allclose(c.probability(), np.ones(9) / 9, atol=1e-5)


def test_circuit_add_demo():
    dim = 3
    c = tc.QuditCircuit(2, dim=dim)
    c.x(0)  # |00> -> |10>
    c2 = tc.QuditCircuit(2, dim=dim, mps_inputs=c.quvector())
    c2.x(0)  # |00> -> |20>
    answer = np.zeros(dim * dim, dtype=np.complex64)
    answer[dim * 2 + 0] = 1.0
    np.testing.assert_allclose(c2.wavefunction(), answer, atol=1e-4)
    c3 = tc.QuditCircuit(2, dim=dim)
    c3.x(0)
    c3.replace_mps_inputs(c.quvector())
    np.testing.assert_allclose(c3.wavefunction(), answer, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_circuit_matrix(backend):
    c = tc.QuditCircuit(2, 3)
    c.x(1)
    c.csum(0, 1)

    U = c.matrix()
    U_np = tc.backend.numpy(U)

    row_10 = U_np[3]
    expected_row_10 = np.zeros(9, dtype=row_10.dtype)
    expected_row_10[4] = 1.0
    np.testing.assert_allclose(row_10, expected_row_10, atol=1e-5)

    state = tc.backend.numpy(c.state())
    expected_state = np.zeros(9, dtype=state.dtype)
    expected_state[1] = 1.0
    np.testing.assert_allclose(state, expected_state, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_batch_sample(backend):
    c = tc.QuditCircuit(3, 3)
    c.h(0)
    c.csum(0, 1)
    print(c.sample())
    print(c.sample(batch=8, status=np.random.uniform(size=[8, 3])))
    print(c.sample(batch=8))
    print(c.sample(random_generator=tc.backend.get_random_state(42)))
    print(c.sample(allow_state=True))
    print(c.sample(batch=8, allow_state=True))
    print(
        c.sample(
            batch=8, allow_state=True, random_generator=tc.backend.get_random_state(42)
        )
    )
    print(
        c.sample(
            batch=8,
            allow_state=True,
            status=np.random.uniform(size=[8]),
            format="sample_bin",
        )
    )
    print(
        c.sample(
            batch=8,
            allow_state=False,
            status=np.random.uniform(size=[8, 3]),
            format="sample_bin",
        )
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_sample_format(backend):
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    c.csum(0, 1)
    key = tc.backend.get_random_state(42)
    for allow_state in [False, True]:
        print("allow_state: ", allow_state)
        for batch in [None, 1, 3]:
            print("  batch: ", batch)
            for format_ in [
                None,
                "sample_bin",
                "count_vector",
                "count_dict_bin",
            ]:
                print("    format: ", format_)
                print(
                    "      ",
                    c.sample(
                        batch=batch,
                        allow_state=allow_state,
                        format_=format_,
                        random_generator=key,
                    ),
                )


def test_sample_representation():
    _ALPHBET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    c = tc.QuditCircuit(1, 36)
    (result,) = c.sample(1, format="count_dict_bin").keys()
    assert result == _ALPHBET[0]

    for i in range(1, 35):
        c.x(0)
        (result,) = c.sample(1, format="count_dict_bin").keys()
        assert result == _ALPHBET[i]
