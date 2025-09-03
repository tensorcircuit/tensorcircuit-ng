from functools import lru_cache
from typing import Any, Optional, Tuple

import numpy as np
from sympy import mod_inverse, Mod

from .cons import npdtype

Tensor = Any


SINGLE_BUILDERS = {
    "I": (("none",), lambda d, omega, **kw: _i_matrix_func(d)),
    "X": (("none",), lambda d, omega, **kw: _x_matrix_func(d)),
    "Y": (("none",), lambda d, omega, **kw: _y_matrix_func(d, omega)),
    "Z": (("none",), lambda d, omega, **kw: _z_matrix_func(d, omega)),
    "H": (("none",), lambda d, omega, **kw: _h_matrix_func(d, omega)),
    "RX": (
        ("theta", "j", "k"),
        lambda d, omega, **kw: _rx_matrix_func(d, kw["theta"], kw["j"], kw["k"]),
    ),
    "RY": (
        ("theta", "j", "k"),
        lambda d, omega, **kw: _ry_matrix_func(d, kw["theta"], kw["j"], kw["k"]),
    ),
    "RZ": (
        ("theta", "j"),
        lambda d, omega, **kw: _rz_matrix_func(d, kw["theta"], kw["j"]),
    ),
    "U8": (
        ("gamma", "z", "eps"),
        lambda d, omega, **kw: _u8_matrix_func(
            d, kw["gamma"], kw["z"], kw["eps"], omega
        ),
    ),
}

TWO_BUILDERS = {
    "RXX": (
        ("theta", "j1", "k1", "j2", "k2"),
        lambda d, omega, **kw: _rxx_matrix_func(
            d, kw["theta"], kw["j1"], kw["k1"], kw["j2"], kw["k2"]
        ),
    ),
    "RZZ": (("theta",), lambda d, omega, **kw: _rzz_matrix_func(d, kw["theta"])),
    "CPHASE": (("cv",), lambda d, omega, **kw: _cphase_matrix_func(d, kw["cv"], omega)),
    "CSUM": (("cv",), lambda d, omega, **kw: _csum_matrix_func(d, kw["cv"])),
}


@lru_cache(maxsize=None)
def _cached_matrix(
    kind: str,
    name: str,
    d: int,
    omega: Optional[float] = None,
    key: Optional[tuple[Any, ...]] = (),
) -> Tensor:
    """
    Build and cache a matrix using a registered builder function.

    Looks up a builder in ``SINGLE_BUILDERS`` (for single–qudit gates) or
    ``TWO_BUILDERS`` (for two–qudit gates) according to ``kind``, constructs the
    matrix, and caches the result via ``functools.lru_cache``.

    :param kind: Either ``"single"`` (use ``SINGLE_BUILDERS``) or ``"two"`` (use ``TWO_BUILDERS``).
    :type kind: str
    :param name: Builder name to look up in the chosen dictionary.
    :type name: str
    :param d: Dimension of the (sub)system.
    :type d: int
    :param omega: Optional frequency/scaling parameter passed to the builder.
    :type omega: Optional[float]
    :param key: Tuple of extra parameters matched positionally to the builder's signature.
    :type key: Optional[tuple[Any, ...]]
    :return: Matrix built by the selected builder.
    :rtype: Tensor
    :raises KeyError: If the builder ``name`` is not found.
    :raises TypeError: If ``key`` does not match the builder’s expected parameters.
    :raises ValueError: If ``key`` does not match the builder’s expected parameters.
    """
    builders = SINGLE_BUILDERS if kind == "single" else TWO_BUILDERS
    try:
        sig, builder = builders[name]
    except KeyError as e:
        raise KeyError(f"Unknown builder '{name}' for kind '{kind}'") from e

    extras: Tuple[Any, ...] = () if key is None else key  # normalized & typed
    kwargs = {k: v for k, v in zip(sig, extras)}
    return builder(d, omega, **kwargs)


def _is_prime(n: int) -> bool:
    """
    Check whether a number is prime.

    :param n: Integer to test.
    :type n: int
    :return: ``True`` if ``n`` is prime, else ``False``.
    :rtype: bool
    """
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
    """
    Identity matrix of size ``d``.

    :param d: Qudit dimension.
    :type d: int
    :return: ``(d, d)`` identity matrix.
    :rtype: Tensor
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for i in range(d):
        matrix[i, i] = 1.0
    return matrix


def _x_matrix_func(d: int) -> Tensor:
    r"""
    Generalized Pauli-X on a ``d``-level system.

    .. math:: X_d\lvert j \rangle = \lvert (j+1) \bmod d \rangle

    :param d: Qudit dimension.
    :type d: int
    :return: ``(d, d)`` matrix for :math:`X_d`.
    :rtype: Tensor
    """
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[(j + 1) % d, j] = 1.0
    return matrix


def _z_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    Generalized Pauli-Z on a ``d``-level system.

    .. math:: Z_d\lvert j \rangle = \omega^{j}\lvert j \rangle,\quad \omega=e^{2\pi i/d}

    :param d: Qudit dimension.
    :type d: int
    :param omega: Optional primitive ``d``-th root of unity. Defaults to :math:`e^{2\pi i/d}`.
    :type omega: Optional[float]
    :return: ``(d, d)`` matrix for :math:`Z_d`.
    :rtype: Tensor
    """
    omega = np.exp(2j * np.pi / d) if omega is None else omega
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        matrix[j, j] = omega**j
    return matrix


def _y_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    Generalized Pauli-Y (Y) gate for qudits.

    Defined (up to a global phase) via :math:`Y \propto Z\,X`.

    :param d: Qudit dimension.
    :type d: int
    :param omega: Optional primitive ``d``-th root of unity used by ``Z``.
    :type omega: Optional[float]
    :return: ``(d, d)`` matrix for :math:`Y`.
    :rtype: Tensor
    """
    return np.matmul(_z_matrix_func(d, omega=omega), _x_matrix_func(d)) / 1j


def _h_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    Discrete Fourier transform (Hadamard-like) on ``d`` levels.

    .. math:: H_d\lvert j \rangle = \frac{1}{\sqrt{d}} \sum_{k=0}^{d-1} \omega^{jk}\lvert k \rangle

    :param d: Qudit dimension.
    :type d: int
    :param omega: Optional primitive ``d``-th root of unity. Defaults to :math:`e^{2\pi i/d}`.
    :type omega: Optional[float]
    :return: ``(d, d)`` matrix for :math:`H_d`.
    :rtype: Tensor
    """
    omega = np.exp(2j * np.pi / d) if omega is None else omega
    matrix = np.zeros((d, d), dtype=npdtype)
    for j in range(d):
        for k in range(d):
            matrix[j, k] = omega ** (j * k) / np.sqrt(d)
    return matrix.T


def _s_matrix_func(d: int, omega: Optional[float] = None) -> Tensor:
    r"""
    Diagonal phase gate ``S_d`` on ``d`` levels.

    .. math:: S_d\lvert j \rangle = \omega^{j(j+p_d)/2}\lvert j \rangle,\quad p_d = (d \bmod 2)

    :param d: Qudit dimension.
    :type d: int
    :param omega: Optional primitive ``d``-th root of unity. Defaults to :math:`e^{2\pi i/d}`.
    :type omega: Optional[float]
    :return: ``(d, d)`` diagonal matrix for :math:`S_d`.
    :rtype: Tensor
    """
    omega = np.exp(2j * np.pi / d) if omega is None else omega
    _pd = 0 if d % 2 == 0 else 1
    matrix = np.zeros((d, d), dtype=complex)
    for j in range(d):
        phase_exp = (j * (j + _pd)) / 2
        matrix[j, j] = omega**phase_exp
    return matrix


def _check_rotation(d: int, j: int, k: int) -> None:
    """
    Validate rotation subspace indices for a ``d``-level system.

    :param d: Qudit dimension.
    :type d: int
    :param j: First level index.
    :type j: int
    :param k: Second level index.
    :type k: int
    :raises ValueError: If indices are out of range or if ``j == k``.
    """
    if not (0 <= j < d) or not (0 <= k < d):
        raise ValueError(f"Indices j={j}, k={k} must satisfy 0 <= j,k < d (d={d}).")
    if j == k:
        raise ValueError("R- rotation requires two distinct levels j != k.")


def _rx_matrix_func(d: int, theta: float, j: int = 0, k: int = 1) -> Tensor:
    r"""
    Rotation-X (``RX``) gate on a selected two-level subspace of a qudit.

    Acts like the qubit :math:`RX(\theta)` on levels ``j`` and ``k``, identity elsewhere.

    :param d: Qudit dimension.
    :type d: int
    :param theta: Rotation angle :math:`\theta`.
    :type theta: float
    :param j: First level index.
    :type j: int
    :param k: Second level index.
    :type k: int
    :return: ``(d, d)`` matrix for :math:`RX(\theta)` on the ``j,k`` subspace.
    :rtype: Tensor
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
    Rotation-Y (``RY``) gate on a selected two-level subspace of a qudit.

    :param d: Qudit dimension.
    :type d: int
    :param theta: Rotation angle :math:`\theta`.
    :type theta: float
    :param j: First level index.
    :type j: int
    :param k: Second level index.
    :type k: int
    :return: ``(d, d)`` matrix for :math:`RY(\theta)` on the ``j,k`` subspace.
    :rtype: Tensor
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
    Rotation-Z (``RZ``) gate for qudits.

    For qubits it reduces to the usual :math:`RZ(\theta)`. For general ``d``, it
    applies a phase :math:`e^{i\theta}` to level ``j`` and leaves others unchanged.

    :param d: Qudit dimension.
    :type d: int
    :param theta: Rotation angle :math:`\theta`.
    :type theta: float
    :param j: Level index receiving the phase.
    :type j: int
    :return: ``(d, d)`` diagonal matrix implementing :math:`RZ(\theta)` on level ``j``.
    :rtype: Tensor
    """
    matrix = np.eye(d, dtype=npdtype)
    matrix[j, j] = np.exp(1j * theta)
    return matrix


def _swap_matrix_func(d: int) -> Tensor:
    """
    SWAP gate for two qudits of dimension ``d``.

    Exchanges basis states ``|i⟩|j⟩ → |j⟩|i⟩``.

    :param d: Qudit dimension (for each register).
    :type d: int
    :return: ``(d*d, d*d)`` matrix representing SWAP.
    :rtype: Tensor
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
    Two-qudit ``RZZ(\theta)`` interaction for qudits.

    .. math:: RZZ(\theta) = \exp\!\left(-i \tfrac{\theta}{2} (Z_H \otimes Z_H)\right)

    :param d: Dimension of each qudit (assumed equal).
    :type d: int
    :param theta: Rotation angle.
    :type theta: float
    :return: ``(d*d, d*d)`` matrix representing :math:`RZZ(\theta)`.
    :rtype: Tensor
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
    Two-qudit ``RXX(\theta)`` on a selected two-state subspace.

    Acts like a qubit :math:`RXX` on the subspace spanned by ``|j1, j2⟩`` and ``|k1, k2⟩``.

    :param d: Dimension of each qudit (assumed equal).
    :type d: int
    :param theta: Rotation angle.
    :type theta: float
    :param j1: Level on qudit-1.
    :type j1: int
    :param k1: Level on qudit-1.
    :type k1: int
    :param j2: Level on qudit-2.
    :type j2: int
    :param k2: Level on qudit-2.
    :type k2: int
    :return: ``(d*d, d*d)`` matrix representing :math:`RXX(\theta)` on the selected subspace.
    :rtype: Tensor
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
    r"""
    ``U8`` diagonal single-qudit gate for prime dimensions.

    Defined for prime ``d`` with phases determined by modular polynomials depending
    on parameters :math:`\gamma, z, \epsilon`.

    :param d: Qudit dimension (must be prime).
    :type d: int
    :param gamma: Gate parameter (must be non-zero).
    :type gamma: float
    :param z: Gate parameter.
    :type z: float
    :param eps: Gate parameter.
    :type eps: float
    :param omega: Optional primitive :math:`d`-th root of unity. Defaults to :math:`e^{2\pi i/d}`.
    :type omega: Optional[float]
    :return: ``(d, d)`` diagonal matrix of dtype ``npdtype``.
    :rtype: Tensor
    :raises ValueError: If ``d`` is not prime; if ``gamma==0``;
     if 12 has no modular inverse mod ``d``; or if the computed exponents do not sum to 0 mod ``d``.
    """
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
    Qudit controlled-phase (``CPHASE``) gate.
    Implements ``|r⟩|s⟩ → ω^{rs}|r⟩|s⟩``; optionally condition on a specific control value ``cv``.
              ┌─                                          ─┐
              │ I_d      0        0         ...     0      │
              │ 0       Z_d       0         ...     0      │
     SUMZ_d = │ 0        0       Z_d^2      ...     0      │
              │ .        .        .         .       .      │
              │ 0        0        0         ...  Z_d^{d-1} │
              └                                           ─┘

    :param d: Qudit dimension (for each register).
    :type d: int
    :param cv: Optional control value in ``[0, d-1]``. If ``None``, builds the full SUMZ block-diagonal.
    :type cv: Optional[int]
    :param omega: Optional primitive ``d``-th root of unity for ``Z_d``.
    :type omega: Optional[float]
    :return: ``(d*d, d*d)`` matrix representing the controlled-phase.
    :rtype: Tensor
    :raises ValueError: If ``cv`` is provided and is outside ``[0, d-1]``.
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
    Qudit controlled-sum (``CSUM`` / ``SUMX``) gate.
    Implements ``|r⟩|s⟩ → |r⟩|r+s (\bmod d)⟩``; optionally condition on a specific control value ``cv``.
              ┌─                                          ─┐
              │ I_d      0        0         ...     0      │
              │ 0       X_d       0         ...     0      │
     SUMX_d = │ 0        0       X_d^2      ...     0      │
              │ .        .        .         .       .      │
              │ 0        0        0         ...  X_d^{d-1} │
              └                                           ─┘

    :param d: Qudit dimension (for each register).
    :type d: int
    :param cv: Optional control value in ``[0, d-1]``. If ``None``, builds the full SUMX block-diagonal.
    :type cv: Optional[int]
    :return: ``(d*d, d*d)`` matrix representing the controlled-sum.
    :rtype: Tensor
    :raises ValueError: If ``cv`` is provided and is outside ``[0, d-1]``.
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
