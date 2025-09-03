# tests/test_quditgates.py
import numpy as np
import pytest

from tensorcircuit.quditgates import (
    _i_matrix_func,
    _x_matrix_func,
    _z_matrix_func,
    _y_matrix_func,
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
    SINGLE_BUILDERS,
    TWO_BUILDERS,
    npdtype,
)

if npdtype in (np.complex64, np.float32):
    ATOL = 1e-6
    RTOL = 1e-6
else:
    ATOL = 1e-12
    RTOL = 1e-12


def is_unitary(M, atol=None, rtol=None):
    if atol is None or rtol is None:
        atol = ATOL
        rtol = RTOL
    Mc = M.astype(np.complex128, copy=False)
    I = np.eye(M.shape[0], dtype=np.complex128)
    return np.allclose(Mc.conj().T @ Mc, I, atol=atol, rtol=rtol) and np.allclose(
        Mc @ Mc.conj().T, I, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_I_X_Z_shapes_and_unitarity(d):
    I = _i_matrix_func(d)
    X = _x_matrix_func(d)
    Z = _z_matrix_func(d)
    assert I.shape == (d, d) and X.shape == (d, d) and Z.shape == (d, d)
    assert is_unitary(X)
    assert is_unitary(Z)
    assert np.allclose(I, np.eye(d, dtype=npdtype), atol=ATOL)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_X_is_right_cyclic_shift(d):
    X = _x_matrix_func(d)
    for j in range(d):
        v = np.zeros(d, dtype=npdtype)
        v[j] = 1
        out = X @ v
        expected = np.zeros(d, dtype=npdtype)
        expected[(j + 1) % d] = 1
        assert np.allclose(out, expected, atol=ATOL)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_Z_diagonal_and_value(d):
    omega = np.exp(2j * np.pi / d)
    Z = _z_matrix_func(d, omega)
    assert np.allclose(Z, np.diag([omega**j for j in range(d)]), atol=ATOL)
    assert is_unitary(Z)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_Y_equals_ZX_over_i(d):
    Y = _y_matrix_func(d)
    ZX_over_i = (_z_matrix_func(d) @ _x_matrix_func(d)) / 1j
    assert np.allclose(Y, ZX_over_i, atol=ATOL)
    assert is_unitary(Y)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_H_is_fourier_like_and_unitary(d):
    H = _h_matrix_func(d)
    assert H.shape == (d, d)
    assert is_unitary(H)
    omega = np.exp(2j * np.pi / d)
    F = (1 / np.sqrt(d)) * np.array(
        [[omega ** (j * k) for k in range(d)] for j in range(d)], dtype=npdtype
    ).T
    assert np.allclose(
        H.astype(np.complex128), F.astype(np.complex128), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("d", [2, 3, 5])
def test_S_is_diagonal(d):
    S = _s_matrix_func(d)
    assert np.allclose(S, np.diag(np.diag(S)), atol=ATOL)


@pytest.mark.parametrize("d", [3, 5])
def test_RX_RY_only_affect_subspace(d):
    theta = 0.7
    j, k = 0, 1
    RX = _rx_matrix_func(d, theta, j, k)
    RY = _ry_matrix_func(d, theta, j, k)
    assert is_unitary(RX) and is_unitary(RY)
    for t in range(d):
        if t not in (j, k):
            e = np.zeros(d, dtype=npdtype)
            e[t] = 1
            outx = RX @ e
            outy = RY @ e
            assert np.allclose(outx, e, atol=ATOL)
            assert np.allclose(outy, e, atol=ATOL)


def test_RZ_phase_on_single_level():
    d, theta, j = 5, 1.234, 2
    RZ = _rz_matrix_func(d, theta, j)
    assert is_unitary(RZ)
    diag = np.ones(d, dtype=npdtype)
    diag[j] = np.exp(1j * theta)
    assert np.allclose(RZ, np.diag(diag), atol=ATOL)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_SWAP_permutation(d):
    SW = _swap_matrix_func(d)
    D = d * d
    assert SW.shape == (D, D)
    assert is_unitary(SW)
    for i in range(min(d, 3)):
        for j in range(min(d, 3)):
            v = np.zeros(D, dtype=npdtype)
            v[i * d + j] = 1
            out = SW @ v
            exp = np.zeros(D, dtype=npdtype)
            exp[j * d + i] = 1
            assert np.allclose(out, exp, atol=ATOL)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_RZZ_diagonal(d):
    theta = 0.37
    RZZ = _rzz_matrix_func(d, theta)
    assert is_unitary(RZZ)
    assert np.allclose(RZZ, np.diag(np.diag(RZZ)), atol=ATOL)  # 对角阵


def test_RXX_selected_block():
    d = 4
    theta = 0.81
    j1, k1 = 0, 2
    j2, k2 = 1, 3
    RXX = _rxx_matrix_func(d, theta, j1, k1, j2, k2)
    assert is_unitary(RXX)
    D = d * d
    I = np.eye(D, dtype=npdtype)
    idx_a = j1 * d + j2
    idx_b = k1 * d + k2
    for t in range(D):
        for s in range(D):
            if {t, s} & {idx_a, idx_b}:
                continue
            assert np.isclose(RXX[t, s], I[t, s], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d", [3, 5])
def test_CPHASE_blocks(d):
    omega = np.exp(2j * np.pi / d)
    Z = _z_matrix_func(d, omega)
    M = _cphase_matrix_func(d, cv=None, omega=omega)
    for a in range(d):
        rs = a * d
        block = M[rs : rs + d, rs : rs + d]
        Za = np.linalg.matrix_power(Z, a)
        assert np.allclose(block, Za, atol=ATOL)
    assert is_unitary(M)

    cv = 1
    M2 = _cphase_matrix_func(d, cv=cv, omega=omega)
    for a in range(d):
        rs = a * d
        block = M2[rs : rs + d, rs : rs + d]
        if a == cv:
            assert np.allclose(block, Z, atol=ATOL)
        else:
            assert np.allclose(block, np.eye(d, dtype=npdtype), atol=ATOL)


@pytest.mark.parametrize("d", [3, 5])
def test_CSUM_blocks(d):
    X = _x_matrix_func(d)
    M = _csum_matrix_func(d, cv=None)
    for a in range(d):
        rs = a * d
        block = M[rs : rs + d, rs : rs + d]
        Xa = np.linalg.matrix_power(X, a)
        assert np.allclose(block, Xa, atol=ATOL)
    assert is_unitary(M)

    cv = 2 % d
    M2 = _csum_matrix_func(d, cv=cv)
    for a in range(d):
        rs = a * d
        block = M2[rs : rs + d, rs : rs + d]
        if a == cv:
            assert np.allclose(block, X, atol=ATOL)
        else:
            assert np.allclose(block, np.eye(d, dtype=npdtype), atol=ATOL)


def test_CSUM_mapping_small_d():
    d = 3
    M = _csum_matrix_func(d)
    for r in range(d):
        for s in range(d):
            v = np.zeros(d * d, dtype=npdtype)
            v[r * d + s] = 1
            out = M @ v
            exp = np.zeros(d * d, dtype=npdtype)
            exp[r * d + ((r + s) % d)] = 1
            assert np.allclose(out, exp, atol=ATOL)


def test_rotation_index_errors():
    d = 4
    with pytest.raises(ValueError):
        _rx_matrix_func(d, 0.1, j=-1, k=1)
    with pytest.raises(ValueError):
        _ry_matrix_func(d, 0.1, j=0, k=4)
    with pytest.raises(ValueError):
        _rx_matrix_func(d, 0.1, j=2, k=2)


def test_U8_errors_and_values():
    with pytest.raises(ValueError):
        _u8_matrix_func(d=4)
    with pytest.raises(ValueError):
        _u8_matrix_func(d=3, gamma=0.0)
    d = 3
    U = _u8_matrix_func(d, gamma=2.0, z=1.0, eps=0.0)
    omega = np.exp(2j * np.pi / d)
    expected = np.diag([omega**0, omega**1, omega**8])
    assert np.allclose(U, expected, atol=ATOL)


def test_CPHASE_CSUM_cv_range():
    d = 5
    with pytest.raises(ValueError):
        _cphase_matrix_func(d, cv=-1)
    with pytest.raises(ValueError):
        _cphase_matrix_func(d, cv=d)
    with pytest.raises(ValueError):
        _csum_matrix_func(d, cv=-1)
    with pytest.raises(ValueError):
        _csum_matrix_func(d, cv=d)


def test_cached_matrix_identity_and_x():
    A1 = _cached_matrix("single", "I", d=3, omega=None, key=())
    A2 = _cached_matrix("single", "I", d=3, omega=None, key=())
    assert A1 is A2

    X1 = _cached_matrix("single", "RX", d=3, omega=None, key=(0.1, 0, 1))
    X2 = _cached_matrix("single", "RX", d=3, omega=None, key=(0.2, 0, 1))
    assert X1 is not X2
    assert X1.shape == (3, 3) and X2.shape == (3, 3)


def test_builders_smoke():
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
