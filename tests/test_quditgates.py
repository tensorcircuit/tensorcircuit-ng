import sys
import os
import pytest
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

from tensorcircuit.quditgates import (
    _i_matrix_func,
    _x_matrix_func,
    _z_matrix_func,
    _h_matrix_func,
    _s_matrix_func,
    _rx_matrix_func,
    _ry_matrix_func,
    _rz_matrix_func,
    _swap_matrix_func,
    _rzz_matrix_func,
    _rxx_matrix_func,
    _u8_matrix_func,
    _cphase_matrix_func,
    _csum_matrix_func,
    _cached_matrix,
    _is_prime,
    SINGLE_BUILDERS,
    TWO_BUILDERS,
)


def is_unitary(M):
    Mc = M.astype(np.complex128, copy=False)
    I = np.eye(M.shape[0], dtype=np.complex128)
    return np.allclose(Mc.conj().T @ Mc, I, atol=1e-5, rtol=1e-5) and np.allclose(
        Mc @ Mc.conj().T, I, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_I_X_Z_shapes_and_unitarity(d, highp):
    I = _i_matrix_func(d)
    X = _x_matrix_func(d)
    Z = _z_matrix_func(d)
    assert I.shape == (d, d) and X.shape == (d, d) and Z.shape == (d, d)
    assert is_unitary(X)
    assert is_unitary(Z)
    np.testing.assert_allclose(I, np.eye(d), atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_X_is_right_cyclic_shift(d, highp):
    X = _x_matrix_func(d)
    for j in range(d):
        v = np.zeros(d)
        v[j] = 1
        out = X @ v
        expected = np.zeros(d)
        expected[(j + 1) % d] = 1
        np.testing.assert_allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_Z_diagonal_and_value(d, highp):
    omega = np.exp(2j * np.pi / d)
    Z = _z_matrix_func(d, omega)
    np.testing.assert_allclose(Z, np.diag([omega**j for j in range(d)]), atol=1e-5)
    assert is_unitary(Z)


# @pytest.mark.parametrize("d", [2, 3, 5])
# def test_Y_equals_ZX_over_i(d, highp):
#     Y = _y_matrix_func(d)
#     ZX_over_i = (_z_matrix_func(d) @ _x_matrix_func(d)) / 1j
#     np.testing.assert_allclose(Y, ZX_over_i, atol=1e-5)
#     assert is_unitary(Y)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_H_is_fourier_like_and_unitary(d, highp):
    H = _h_matrix_func(d)
    assert H.shape == (d, d)
    assert is_unitary(H)
    omega = np.exp(2j * np.pi / d)
    F = (1 / np.sqrt(d)) * np.array(
        [[omega ** (j * k) for k in range(d)] for j in range(d)]
    ).T
    np.testing.assert_allclose(
        H.astype(np.complex128), F.astype(np.complex128), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("d", [2, 3, 5])
def test_S_is_diagonal(d, highp):
    S = _s_matrix_func(d)
    np.testing.assert_allclose(S, np.diag(np.diag(S)), atol=1e-5)


@pytest.mark.parametrize("d", [3, 5])
def test_RX_RY_only_affect_subspace(d, highp):
    theta = 0.7
    j, k = 0, 1
    RX = _rx_matrix_func(d, theta, j, k)
    RY = _ry_matrix_func(d, theta, j, k)
    assert is_unitary(RX) and is_unitary(RY)
    for t in range(d):
        if t not in (j, k):
            e = np.zeros(d)
            e[t] = 1
            outx = RX @ e
            outy = RY @ e
            np.testing.assert_allclose(outx, e, atol=1e-5)
            np.testing.assert_allclose(outy, e, atol=1e-5)


def test_RZ_phase_on_single_level(highp):
    d, theta, j = 5, 1.234, 2
    RZ = _rz_matrix_func(d, theta, j)
    assert is_unitary(RZ)
    diag = np.ones(d, dtype=np.complex64)
    diag[j] = np.exp(1j * theta)
    assert np.allclose(RZ, np.diag(diag), atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_SWAP_permutation(d, highp):
    SW = _swap_matrix_func(d)
    D = d * d
    assert SW.shape == (D, D)
    assert is_unitary(SW)
    for i in range(min(d, 3)):
        for j in range(min(d, 3)):
            v = np.zeros(D)
            v[i * d + j] = 1
            out = SW @ v
            exp = np.zeros(D)
            exp[j * d + i] = 1
            np.testing.assert_allclose(out, exp, atol=1e-5)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_RZZ_diagonal(d, highp):
    theta = 0.37
    RZZ = _rzz_matrix_func(d, theta, j1=0, k1=1, j2=0, k2=1)
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


def test_RXX_selected_block(highp):
    d = 4
    theta = 0.81
    j1, k1 = 0, 2
    j2, k2 = 1, 3
    RXX = _rxx_matrix_func(d, theta, j1, k1, j2, k2)
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
def test_CPHASE_blocks(d, highp):
    omega = np.exp(2j * np.pi / d)
    Z = _z_matrix_func(d, omega)
    M = _cphase_matrix_func(d, cv=None, omega=omega)
    for a in range(d):
        rs = a * d
        block = M[rs : rs + d, rs : rs + d]
        Za = np.linalg.matrix_power(Z, a)
        np.testing.assert_allclose(block, Za, atol=1e-5)
    assert is_unitary(M)

    cv = 1
    M2 = _cphase_matrix_func(d, cv=cv, omega=omega)
    for a in range(d):
        rs = a * d
        block = M2[rs : rs + d, rs : rs + d]
        if a == cv:
            np.testing.assert_allclose(block, Z, atol=1e-5)
        else:
            np.testing.assert_allclose(block, np.eye(d), atol=1e-5)


@pytest.mark.parametrize("d", [3, 5])
def test_CSUM_blocks(d, highp):
    X = _x_matrix_func(d)
    M = _csum_matrix_func(d, cv=None)
    for a in range(d):
        rs = a * d
        block = M[rs : rs + d, rs : rs + d]
        Xa = np.linalg.matrix_power(X, a)
        np.testing.assert_allclose(block, Xa, atol=1e-5)
    assert is_unitary(M)

    cv = 2 % d
    M2 = _csum_matrix_func(d, cv=cv)
    for a in range(d):
        rs = a * d
        block = M2[rs : rs + d, rs : rs + d]
        if a == cv:
            np.testing.assert_allclose(block, X, atol=1e-5)
        else:
            np.testing.assert_allclose(block, np.eye(d), atol=1e-5)


def test_CSUM_mapping_small_d(highp):
    d = 3
    M = _csum_matrix_func(d)
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
        _rx_matrix_func(d, 0.1, j=-1, k=1)
    with pytest.raises(ValueError):
        _ry_matrix_func(d, 0.1, j=0, k=4)
    with pytest.raises(ValueError):
        _rx_matrix_func(d, 0.1, j=2, k=2)


def test_U8_errors_and_values(highp):
    with pytest.raises(ValueError):
        _u8_matrix_func(d=4)
    with pytest.raises(ValueError):
        _u8_matrix_func(d=3, gamma=0.0)
    d = 3
    U = _u8_matrix_func(d, gamma=2.0, z=1.0, eps=0.0)
    omega = np.exp(2j * np.pi / d)
    expected = np.diag([omega**0, omega**1, omega**8])
    assert np.allclose(U, expected, atol=1e-5)


def test_CPHASE_CSUM_cv_range(highp):
    d = 5
    with pytest.raises(ValueError):
        _cphase_matrix_func(d, cv=-1)
    with pytest.raises(ValueError):
        _cphase_matrix_func(d, cv=d)
    with pytest.raises(ValueError):
        _csum_matrix_func(d, cv=-1)
    with pytest.raises(ValueError):
        _csum_matrix_func(d, cv=d)


def test_cached_matrix_identity_and_x(highp):
    A1 = _cached_matrix("single", "I", d=3, omega=None, key=())
    A2 = _cached_matrix("single", "I", d=3, omega=None, key=())
    assert A1 is A2

    X1 = _cached_matrix("single", "RX", d=3, omega=None, key=(0.1, 0, 1))
    X2 = _cached_matrix("single", "RX", d=3, omega=None, key=(0.2, 0, 1))
    assert X1 is not X2
    assert X1.shape == (3, 3) and X2.shape == (3, 3)


def test_builders_smoke(highp):
    d = 3
    for name, (sig, _) in SINGLE_BUILDERS.items():
        defaults = {
            "theta": 0.1,
            "gamma": 0.1,
            "z": 0.1,
            "eps": 0.1,
            "j": 0,
            "k": 1,  # ensure j != k when present
        }
        key = tuple(defaults.get(s, 0) for s in sig)
        M = _cached_matrix("single", name, d, None, key)
        assert M.shape == (d, d)

    for name, (sig, _) in TWO_BUILDERS.items():
        defaults = {"theta": 0.1, "j1": 0, "k1": 1, "j2": 0, "k2": 1, "cv": None}
        key = tuple(defaults[s] for s in sig)
        M = _cached_matrix("two", name, d, None, key)
        assert M.shape == (d * d, d * d)


def test_cached_matrix_np_scalar_and_non_scalar_omega():
    d = 3
    if "RX" in SINGLE_BUILDERS:
        name = "RX"
        sig, _ = SINGLE_BUILDERS[name]
        defaults = {"theta": np.float64(0.314159265), "j": 0, "k": 1}
        key = tuple(defaults.get(s, 0) for s in sig if s != "none")
        M = _cached_matrix("single", name, d, None, key)
        assert M.shape == (d, d)

    name = "I" if "I" in SINGLE_BUILDERS else next(iter(SINGLE_BUILDERS.keys()))
    sig, _ = SINGLE_BUILDERS[name]
    key = tuple(0 for s in sig if s != "none")
    M2 = _cached_matrix("single", name, d, np.array([0.5]), key)
    assert M2.shape == (d, d)


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
        _rzz_matrix_func(d, theta, j1=0, k1=1, j2=0, k2=3)
    with pytest.raises(ValueError):
        _rxx_matrix_func(d, theta, j1=0, k1=1, j2=3, k2=0)
    with pytest.raises(ValueError):
        _rzz_matrix_func(d, theta, j1=0, k1=0, j2=1, k2=1)
    with pytest.raises(ValueError):
        _rxx_matrix_func(d, theta, j1=2, k1=2, j2=0, k2=0)


def test_u8_requires_prime_dimension():
    with pytest.raises(ValueError):
        _u8_matrix_func(d=9, gamma=1.0, z=0.0, eps=0.0)
