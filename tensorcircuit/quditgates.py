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


def _z_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    Z_d\ket{j} = \omega^{j}\ket{j}
    """
    omega = np.exp(2j * np.pi / d) if omega is None else omega
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[j, j] = omega**j
    return matrix


def _y_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    Generalized Pauli-Y (Y) Gate for qudits.

    The Y gate represents a combination of the X and Z gates, generalizing the Pauli-Y gate
    from qubits to higher dimensions. It is defined as

    .. math::

            Y = \frac{1}{i}\, Z \cdot X,

    where the generalized Pauli-X and Pauli-Z gates are applied to the target qudits.
    """
    return np.matmul(_z_matrix_func(d, omega=omega), _x_matrix_func(d)) / 1j


def _h_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    H_d\ket{j} = \frac{1}{\sqrt{d}}\sum_{k=0}^{d-1}\omega^{jk}\ket{k}
    """
    omega = np.exp(2j * np.pi / d) if omega is None else omega
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        for k in range(d):
            matrix[j, k] = omega ** (j * k) / np.sqrt(d)
    return matrix.T


def _s_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    S_d\ket{j} = \omega^{j(j + p_d) / 2}\ket{j}
    """
    omega = np.exp(2j * np.pi / d) if omega is None else omega
    _pd = 0 if d % 2 == 0 else 1
    matrix = np.zeros((d, d), dtype=complex)
    for j in range(d):
        phase_exp = (j * (j + _pd)) / 2
        matrix[j, j] = omega**phase_exp
    return matrix


def _check_rotation(d: int, j: int, k: int) -> None:
    if not (0 <= j < d) or not (0 <= k < d):
        raise ValueError(f"Indices j={j}, k={k} must satisfy 0 <= j,k < d (d={d}).")
    if j == k:
        raise ValueError("RX rotation requires two distinct levels j != k.")


def _rx_matrix_func(d: int, theta: float, j: int = 0, k: int = 1) -> Tensor:
    r"""
    Rotation-X (RX) Gate for qudits.

    The RX gate represents a rotation about the X-axis of the Bloch sphere in a qudit system.
    For a qubit (2-level system), the matrix representation is given by

    .. math::

            RX(\theta) =
            \begin{pmatrix}
            \cos(\theta/2) & -i\sin(\theta/2) \\
            -i\sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

    For higher-dimensional qudits, the RX gate affects only the specified two levels (indexed by
    \(j\) and \(k\)), leaving all other levels unchanged.

    Args:
        d (int): Dimension of the qudit Hilbert space.
        theta (float): Rotation angle θ.
        j (int): First level index (default 0).
        k (int): Second level index (default 1).

    Returns:
        Tensor: A (d x d) numpy array of dtype `npdtype` representing the RX gate.
    """
    _check_rotation(d, j, k)
    matrix = np.eye(d, dtype=npdtype)
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    matrix[j, j] = c
    matrix[k, k] = c
    matrix[j, k] = -1j * s
    matrix[k, j] = -1j * s
    return matrix


def _ry_matrix_func(d: int, theta: float, j: int = 0, k: int = 1) -> Tensor:
    r"""
    Rotation-Y (RY) Gate for qudits.

    Acts as a standard qubit RY(θ) on the two-level subspace spanned by |j> and |k>,
    and as identity on all other levels:

    .. math::

            RY(\theta) =
            \begin{pmatrix}
            \cos(\theta/2) & -\sin(\theta/2) \\
            \sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

    Args:
        d (int): Dimension of the qudit Hilbert space.
        theta (float): Rotation angle θ.
        j (int): First level index (default 0).
        k (int): Second level index (default 1).

    Returns:
        Tensor: A (d x d) numpy array of dtype `npdtype` representing the RY gate.
    """
    _check_rotation(d, j, k)
    matrix = np.eye(d, dtype=npdtype)
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    matrix[j, j] = c
    matrix[k, k] = c
    matrix[j, k] = -s
    matrix[k, j] = s
    return matrix


def _rz_matrix_func(d: int, theta: float, j: int = 0) -> Tensor:
    r"""
    Rotation-Z (RZ) Gate for qudits.

    .. math::

            RZ(\theta) =
            \begin{pmatrix}
            e^{-i\theta/2} & 0 \\
            0 & e^{i\theta/2}
            \end{pmatrix}

    For qudits (d >= 2), apply a phase e^{iθ} only to level |j>, leaving others unchanged:
        (RZ_d)_{mm} = e^{iθ} if m == j else 1

    Args:
        d (int): Dimension of the qudit Hilbert space.
        theta (float): Rotation angle θ.
        j (int): First level index (default 0).
        k (int): Second level index (default 1).

    Returns:
        Tensor: A (d x d) numpy array of dtype `npdtype` representing the RZ gate.
    """
    matrix = np.eye(d, dtype=npdtype)
    matrix[j, j] = np.exp(1j * theta)
    return matrix


def _swap_matrix_func(d: int) -> Tensor:
    r"""
    SWAP gate for two qudits of dimensions d.

    Exchanges the states |i⟩|j⟩ -> |j⟩|i⟩.

    Args:
        d (int): Dimension of the qudit.

    Returns:
        Tensor: A numpy array representing the SWAP gate.
    """
    D = d * d
    matrix = np.zeros((D, D), dtype=npdtype)
    for i in range(d):
        for j in range(d):
            idx_in = i * d + j
            idx_out = j * d + i
            matrix[idx_out, idx_in] = 1.0
    return matrix


def _rzz_matrix_func(d: int, theta: float) -> Tensor:
    r"""
    Two-qudit RZZ(\theta) gate for qudits.

    .. math::

        Z_H = \mathrm{diag}(d-1,\, d-3,\, \ldots,\,-(d-1))
    .. math::
        RZZ(\theta) = \exp\!\left(-i \tfrac{\theta}{2} \, \bigl(Z_H \otimes Z_H\bigr)\right)

    For :math:`d=2`, this reduces to the standard qubit RZZ gate.

    Args:
        d (int): Dimension of the qudits (assumed equal for both).
        theta (float): Rotation angle.

    Returns:
        Tensor: A ``(d*d, d*d)`` numpy array representing the RZZ gate.
    """
    lam = np.array(
        [d - 1 - 2 * j for j in range(d)], dtype=float
    )  # [d-1, d-3, ..., -(d-1)]
    D = d * d
    diag = np.empty(D, dtype=npdtype)
    idx = 0
    for a in range(d):
        for b in range(d):
            diag[idx] = np.exp(-1j * (theta / 2.0) * lam[a] * lam[b])
            idx += 1
    return np.diag(diag)


def _rxx_matrix_func(
    d: int, theta: float, j1: int = 0, k1: int = 1, j2: int = 0, k2: int = 1
) -> Tensor:
    r"""
    Two-qudit RXX(θ) on a selected two-state subspace.

    Acts like a qubit RXX on the subspace spanned by |j1, j2> and |k1, k2>:
    
    .. math::

        RXX(\theta) =
        \begin{pmatrix}
        \cos\!\left(\tfrac{\theta}{2}\right) & -i \sin\!\left(\tfrac{\theta}{2}\right) \\
        -i \sin\!\left(\tfrac{\theta}{2}\right) & \cos\!\left(\tfrac{\theta}{2}\right)
        \end{pmatrix}
    All other basis states are unchanged.

    Args:
        d (int): Dimension for both qudits (assumed equal).
        theta (float): Rotation angle.
        j1, k1 (int): Levels on qudit-1.
        j2, k2 (int): Levels on qudit-2.

    Returns:
        Tensor: A ``(d*d, d*d)`` numpy array representing the RXX gate.
    """
    D = d * d
    M = np.eye(D, dtype=npdtype)

    # flatten basis index: |a,b> ↦ a*d + b
    idx_a = j1 * d + j2
    idx_b = k1 * d + k2

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    # Overwrite the chosen 2x2 block
    M[idx_a, idx_a] = c
    M[idx_b, idx_b] = c
    M[idx_a, idx_b] = -1j * s
    M[idx_b, idx_a] = -1j * s

    return M


def _u8_matrix_func(
    d: int,
    gamma: float = 2.0,
    z: float = 1.0,
    eps: float = 0.0,
    omega: Optional[float] = None,
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

    omega = np.exp(2j * np.pi / d) if omega is None else omega
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[j, j] = omega ** vks[j]
    return matrix


def _cphase_matrix_func(
    d: int, cv: Optional[int] = None, omega: Optional[float] = None
) -> Tensor:
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
    omega = np.exp(2j * np.pi / d) if omega is None else omega
    size = d**2
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
