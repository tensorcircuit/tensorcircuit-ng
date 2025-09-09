import sys
import os
import numpy as np

import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

import tensorcircuit as tc

from tensorcircuit.quditgates import (
    i_matrix_func,
    x_matrix_func,
    z_matrix_func,
    h_matrix_func,
    s_matrix_func,
    rx_matrix_func,
    ry_matrix_func,
    rz_matrix_func,
    swap_matrix_func,
    rzz_matrix_func,
    rxx_matrix_func,
    u8_matrix_func,
    cphase_matrix_func,
    csum_matrix_func,
    _is_prime,
)


def is_unitary(M):
    M = tc.backend.numpy(M)
    Mc = M.astype(np.complex128, copy=False)
    I = np.eye(M.shape[0], dtype=np.complex128)
    return np.allclose(Mc.conj().T @ Mc, I, atol=1e-5, rtol=1e-5) and np.allclose(
        Mc @ Mc.conj().T, I, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("d", [2, 3, 4, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_I_X_Z_shapes_and_unitarity(d, backend, highp):
    I = i_matrix_func(d)
    X = x_matrix_func(d)
    Z = z_matrix_func(d)
    assert I.shape == (d, d) and X.shape == (d, d) and Z.shape == (d, d)
    assert is_unitary(X)
    assert is_unitary(Z)
    np.testing.assert_allclose(tc.backend.numpy(I), np.eye(d), atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_X_is_right_cyclic_shift(d, backend, highp):
    X = x_matrix_func(d)
    X = tc.backend.numpy(X)
    for j in range(d):
        v = np.zeros(d)
        v[j] = 1
        out = X @ v
        expected = np.zeros(d)
        expected[(j + 1) % d] = 1
        np.testing.assert_allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_Z_diagonal_and_value(d, backend, highp):
    omega = np.exp(2j * np.pi / d)
    Z = z_matrix_func(d, omega)
    np.testing.assert_allclose(
        tc.backend.numpy(Z), np.diag([omega**j for j in range(d)]), atol=1e-5
    )
    assert is_unitary(Z)


# @pytest.mark.parametrize("d", [2, 3, 5])
# def test_Y_equals_ZX_over_i(d, highp):
#     Y = _y_matrix_func(d)
#     ZX_over_i = (_z_matrix_func(d) @ _x_matrix_func(d)) / 1j
#     np.testing.assert_allclose(Y, ZX_over_i, atol=1e-5)
#     assert is_unitary(Y)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_H_is_fourier_like_and_unitary(d, backend, highp):
    H = h_matrix_func(d)
    assert H.shape == (d, d)
    assert is_unitary(H)
    omega = np.exp(2j * np.pi / d)
    F = (1 / np.sqrt(d)) * np.array(
        [[omega ** (j * k) for k in range(d)] for j in range(d)]
    ).T
    np.testing.assert_allclose(tc.backend.numpy(H), F, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_S_is_diagonal(d, backend, highp):
    S = s_matrix_func(d)
    np.testing.assert_allclose(S, np.diag(np.diag(S)), atol=1e-5)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_RX_RY_only_affect_subspace(d, backend, highp):
    theta = 0.7
    j, k = 0, 1
    RX = rx_matrix_func(d, theta, j, k)
    RY = ry_matrix_func(d, theta, j, k)
    assert is_unitary(RX) and is_unitary(RY)
    RX, RY = tc.backend.numpy(RX), tc.backend.numpy(RY)
    for t in range(d):
        if t not in (j, k):
            e = np.zeros(d)
            e[t] = 1
            outx = RX @ e
            outy = RY @ e
            np.testing.assert_allclose(outx, e, atol=1e-5)
            np.testing.assert_allclose(outy, e, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_RZ_phase_on_single_level(backend, highp):
    d, theta, j = 5, 1.234, 2
    RZ = rz_matrix_func(d, theta, j)
    assert is_unitary(RZ)
    diag = np.ones(d, dtype=np.complex64)
    diag[j] = np.exp(1j * theta)
    assert np.allclose(RZ, np.diag(diag), atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_SWAP_permutation(d, backend, highp):
    SW = swap_matrix_func(d)
    D = d * d
    assert SW.shape == (D, D)
    assert is_unitary(SW)
    SW = tc.backend.numpy(SW)
    for i in range(min(d, 3)):
        for j in range(min(d, 3)):
            v = np.zeros(D)
            v[i * d + j] = 1
            out = SW @ v
            exp = np.zeros(D)
            exp[j * d + i] = 1
            np.testing.assert_allclose(out, exp, atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_RZZ_diagonal(d, backend, highp):
    theta = 0.37
    RZZ = rzz_matrix_func(d, theta, j1=0, k1=1, j2=0, k2=1)
    assert is_unitary(RZZ)

    D = d * d
    I = np.eye(D, dtype=np.complex128)

    idx_a = 0 * d + 0
    idx_b = 1 * d + 1

    for t in range(D):
        if t not in (idx_a, idx_b):
            np.testing.assert_allclose(RZZ[t], I[t], atol=1e-5)

    np.testing.assert_allclose(RZZ[idx_a, idx_a], np.exp(-1j * theta / 2), atol=1e-5)
    np.testing.assert_allclose(RZZ[idx_b, idx_b], np.exp(+1j * theta / 2), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_RXX_selected_block(backend, highp):
    d = 4
    theta = 0.81
    j1, k1 = 0, 2
    j2, k2 = 1, 3
    RXX = rxx_matrix_func(d, theta, j1, k1, j2, k2)
    assert is_unitary(RXX)
    D = d * d
    I = np.eye(D)
    idx_a = j1 * d + j2
    idx_b = k1 * d + k2
    for t in range(D):
        for s in range(D):
            if {t, s} & {idx_a, idx_b}:
                continue
            np.testing.assert_allclose(RXX[t, s], I[t, s], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_CPHASE_blocks(d, backend, highp):
    omega = np.exp(2j * np.pi / d)
    Z = z_matrix_func(d, omega)
    M = cphase_matrix_func(d, cv=None, omega=omega)
    for a in range(d):
        rs = a * d
        block = M[rs : rs + d, rs : rs + d]
        Za = np.linalg.matrix_power(Z, a)
        np.testing.assert_allclose(block, Za, atol=1e-5)
    assert is_unitary(M)

    cv = 1
    M2 = cphase_matrix_func(d, cv=cv, omega=omega)
    for a in range(d):
        rs = a * d
        block = M2[rs : rs + d, rs : rs + d]
        if a == cv:
            np.testing.assert_allclose(block, Z, atol=1e-5)
        else:
            np.testing.assert_allclose(block, np.eye(d), atol=1e-5)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_CSUM_blocks(d, backend, highp):
    X = x_matrix_func(d)
    M = csum_matrix_func(d, cv=None)
    for a in range(d):
        rs = a * d
        block = M[rs : rs + d, rs : rs + d]
        Xa = np.linalg.matrix_power(X, a)
        np.testing.assert_allclose(block, Xa, atol=1e-5)
    assert is_unitary(M)

    cv = 2 % d
    M2 = csum_matrix_func(d, cv=cv)
    for a in range(d):
        rs = a * d
        block = M2[rs : rs + d, rs : rs + d]
        if a == cv:
            np.testing.assert_allclose(block, X, atol=1e-5)
        else:
            np.testing.assert_allclose(block, np.eye(d), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_CSUM_mapping_small_d(backend, highp):
    d = 3
    M = csum_matrix_func(d)
    M = tc.backend.numpy(M)
    for r in range(d):
        for s in range(d):
            v = np.zeros(d * d)
            v[r * d + s] = 1
            out = M @ v
            exp = np.zeros(d * d)
            exp[r * d + ((r + s) % d)] = 1
            np.testing.assert_allclose(out, exp, atol=1e-5)


def test_rotation_index_errors(highp):
    d = 4
    with pytest.raises(ValueError):
        rx_matrix_func(d, 0.1, j=-1, k=1)
    with pytest.raises(ValueError):
        ry_matrix_func(d, 0.1, j=0, k=4)
    with pytest.raises(ValueError):
        rx_matrix_func(d, 0.1, j=2, k=2)


def test_CPHASE_CSUM_cv_range(highp):
    d = 5
    with pytest.raises(ValueError):
        cphase_matrix_func(d, cv=-1)
    with pytest.raises(ValueError):
        cphase_matrix_func(d, cv=d)
    with pytest.raises(ValueError):
        csum_matrix_func(d, cv=-1)
    with pytest.raises(ValueError):
        csum_matrix_func(d, cv=d)


def test__is_prime_edge_and_composites():
    assert _is_prime(2) is True
    assert _is_prime(3) is True
    assert _is_prime(4) is False
    assert _is_prime(5) is True
    assert _is_prime(25) is False
    assert _is_prime(29) is True


def test_two_qudit_builders_index_validation():
    d = 3
    theta = 0.1
    with pytest.raises(ValueError):
        rzz_matrix_func(d, theta, j1=0, k1=1, j2=0, k2=3)
    with pytest.raises(ValueError):
        rxx_matrix_func(d, theta, j1=0, k1=1, j2=3, k2=0)
    with pytest.raises(ValueError):
        rzz_matrix_func(d, theta, j1=0, k1=0, j2=1, k2=1)
    with pytest.raises(ValueError):
        rxx_matrix_func(d, theta, j1=2, k1=2, j2=0, k2=0)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u8_prime_dimension_and_qubit_case(backend):
    with pytest.raises(ValueError):
        u8_matrix_func(d=4)
    with pytest.raises(ValueError):
        u8_matrix_func(d=9, gamma=1, z=0, eps=0)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u8_qutrit_correct_phases_and_gamma_zero_allowed(backend):
    d = 3
    U3 = u8_matrix_func(d=d, gamma=2, z=1, eps=0)
    zeta = np.exp(2j * np.pi / 9)
    expected3 = np.diag([zeta**0, zeta**1, zeta**8])
    assert np.allclose(tc.backend.numpy(U3), expected3, atol=1e-12)

    U3_g0 = u8_matrix_func(d=d, gamma=0, z=1, eps=2)
    U3_g0 = tc.backend.numpy(U3_g0)
    assert U3_g0.shape == (3, 3)
    assert np.allclose(U3_g0, np.diag(np.diag(U3_g0)), atol=1e-12)
    assert np.allclose(U3_g0.conj().T @ U3_g0, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u8_p_greater_than_3_matches_closed_form(backend):
    d = 5
    gamma, z, eps = 2, 1, 3

    inv_12 = pow(12, -1, d)  # 12 \equiv 2 (mod 5), inverse = 3
    vks = [0] * d
    for k in range(1, d):
        k_ = k % d
        term_inner = ((6 * z) % d + ((2 * k_ - 3) % d) * gamma % d) % d
        term = (gamma + (k_ * term_inner) % d) % d
        vk = ((inv_12 * k_) % d) * term % d
        vk = (vk + (eps * k_) % d) % d
        vks[k] = vk

    omega = np.exp(2j * np.pi / d)
    expected5 = np.diag([omega**v for v in vks])

    U5 = u8_matrix_func(d=d, gamma=gamma, z=z, eps=eps)
    assert np.allclose(tc.backend.numpy(U5), expected5, atol=1e-12)
    assert sum(vks) % d == 0


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u8_parameter_normalization_and_custom_omega(backend):
    d = 5
    U_modded = u8_matrix_func(d=d, gamma=2, z=1, eps=3)
    U_unnormalized = u8_matrix_func(
        d=d, gamma=7, z=-4, eps=13
    )  # 7\equiv 2, -4\equiv 1, 13\equiv 3 (mod 5)
    assert np.allclose(U_modded, U_unnormalized, atol=1e-12)

    d = 7
    gamma, z, eps = 3, 2, 1
    inv_12 = pow(12, -1, d)
    vks = [0] * d
    for k in range(1, d):
        k_ = k % d
        term_inner = ((6 * z) % d + ((2 * k_ - 3) % d) * gamma % d) % d
        term = (gamma + (k_ * term_inner) % d) % d
        vk = ((inv_12 * k_) % d) * term % d
        vk = (vk + (eps * k_) % d) % d
        vks[k] = vk

    omega_custom = np.exp(2j * np.pi / d) * np.exp(0j)
    U7_custom = u8_matrix_func(d=d, gamma=gamma, z=z, eps=eps, omega=omega_custom)
    expected7_custom = np.diag([omega_custom**v for v in vks])
    assert np.allclose(tc.backend.numpy(U7_custom), expected7_custom, atol=1e-12)
