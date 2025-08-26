from typing import Any, Optional

import numpy as np
from sympy import mod_inverse, Mod

from .cons import npdtype

Tensor = Any


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3, 5, 7):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    r = int(n**0.5) + 1
    for i in range(5, r, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


def _i_matrix_func(d: int) -> Tensor:
    matrix = np.zeros((d, d), dtype=npdtype)
    for i in range(d):
        matrix[i, i] = 1.0
    return matrix


def _x_matrix_func(d: int) -> Tensor:
    r"""
    X_d\ket{j} = \ket{(j + 1) mod d}
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[(j + 1) % d, j] = 1.0
    return matrix


def _z_matrix_func(d: int, omega: float) -> Tensor:
    r"""
    Z_d\ket{j} = \omega^{j}\ket{j}
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[j, j] = omega**j
    return matrix


def _h_matrix_func(d: int, omega: float) -> Tensor:
    r"""
    H_d\ket{j} = \frac{1}{\sqrt{d}}\sum_{k=0}^{d-1}\omega^{jk}\ket{k}
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        for k in range(d):
            matrix[j, k] = omega ** (j * k) / np.sqrt(d)
    return matrix.T


def _s_matrix_func(d: int, omega: float) -> Tensor:
    r"""
    S_d\ket{j} = \omega^{j(j + p_d) / 2}\ket{j}
    """
    _pd = 0 if d % 2 == 0 else 1
    matrix = np.zeros((d, d), dtype=complex)
    for j in range(d):
        phase_exp = (j * (j + _pd)) / 2
        matrix[j, j] = omega**phase_exp
    return matrix


def _u8_matrix_func(
    d: int, gamma: float = 2.0, z: float = 1.0, eps: float = 0.0
) -> Tensor:
    if not _is_prime(d):
        raise ValueError(
            f"Dimension d={d} is not prime, U8 gate requires a prime dimension."
        )
    if gamma == 0.0:
        raise ValueError("gamma must be non-zero")

    vks = [0] * d
    if d == 3:
        vks = [0, 1, 8]
    else:
        try:
            inv_12 = mod_inverse(12, d)
        except ValueError:
            raise ValueError(
                f"Inverse of 12 mod {d} does not exist. Choose a prime d that does not divide 12."
            )

        for i in range(1, d):
            a = inv_12 * i * (gamma + i * (6 * z + (2 * i - 3) * gamma)) + eps * i
            vks[i] = Mod(a, d)

    # print(vks)
    sum_vks = Mod(sum(vks), d)
    if sum_vks != 0:
        raise ValueError(
            f"Sum of v_k's is not 0 mod {d}. Got {sum_vks}. Check parameters."
        )

    omega = np.exp(2j * np.pi / d)
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[j, j] = omega ** vks[j]
    return matrix


def _cphase_matrix_func(d: int, cv: Optional[int] = None) -> Tensor:
    r"""
    Qudit Controlled-z gate
    \ket{r}\ket{s} \rightarrow \omega^{rs}\ket{r}\ket{s} = \ket{r}Z^r\ket{s}

    This gate is also called SUMZ gate, where Z represents Z_d gate.
              ┌─                                          ─┐
              │ I_d      0        0         ...     0      │
              │ 0       Z_d       0         ...     0      │
     SUMZ_d = │ 0        0       Z_d^2      ...     0      │
              │ .        .        .         .       .      │
              │ 0        0        0         ...  Z_d^{d-1} │
              └                                           ─┘
    """
    size = d**2
    omega = np.exp(2j * np.pi / d)
    z_matrix = _z_matrix_func(d=d, omega=omega)

    if cv is None:
        z_pows = [np.eye(d, dtype=npdtype)]
        for _ in range(1, d):
            z_pows.append(z_pows[-1] @ z_matrix)

        matrix = np.zeros((size, size), dtype=npdtype)
        for a in range(d):
            rs = a * d
            matrix[rs : rs + d, rs : rs + d] = z_pows[a]
        return matrix

    if not (0 <= cv < d):
        raise ValueError(f"cv must be in [0, {d - 1}], got {cv}")

    matrix = np.eye(size, dtype=npdtype)
    rs = cv * d
    matrix[rs : rs + d, rs : rs + d] = z_matrix

    return matrix


def _csum_matrix_func(d: int, cv: Optional[int] = None) -> Tensor:
    r"""
    Qudit Controlled-NOT gate
    \ket{r}\ket{s} \rightarrow \ket{r}\ket{r+s} = \ket{r}X^r\ket{s} = \ket{r}\ket{(r+s) mod d}

    This gate is also called SUMX gate, where X represents X_d gate.
              ┌─                                          ─┐
              │ I_d      0        0         ...     0      │
              │ 0       X_d       0         ...     0      │
     SUMX_d = │ 0        0       X_d^2      ...     0      │
              │ .        .        .         .       .      │
              │ 0        0        0         ...  X_d^{d-1} │
              └                                           ─┘
    """
    size = d**2
    x_matrix = _x_matrix_func(d=d)

    if cv is None:
        x_pows = [np.eye(d, dtype=npdtype)]
        for _ in range(1, d):
            x_pows.append(x_pows[-1] @ x_matrix)

        matrix = np.zeros((size, size), dtype=npdtype)
        for a in range(d):
            rs = a * d
            matrix[rs : rs + d, rs : rs + d] = x_pows[a]
        return matrix

    if not (0 <= cv < d):
        raise ValueError(f"cv must be in [0, {d - 1}], got {cv}")
    matrix = np.eye(size, dtype=npdtype)
    rs = cv * d
    matrix[rs : rs + d, rs : rs + d] = x_matrix

    return matrix
