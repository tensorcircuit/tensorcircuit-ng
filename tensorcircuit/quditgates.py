from typing import Any, Optional, Tuple

import numpy as np

from .cons import backend, dtypestr
from .gates import num_to_tensor

Tensor = Any

SINGLE_BUILDERS = {
    "I": (("none",), lambda d, omega, **kw: _i_matrix_func(d)),
    "X": (("none",), lambda d, omega, **kw: _x_matrix_func(d)),
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
    return backend.eye(d, dtype=dtypestr)


def _x_matrix_func(d: int) -> Tensor:
    r"""
    Generalized Pauli-X on a ``d``-level system.

    .. math:: X_d\lvert j \rangle = \lvert (j+1) \bmod d \rangle

    :param d: Qudit dimension.
    :type d: int
    :return: ``(d, d)`` matrix for :math:`X_d`.
    :rtype: Tensor
    """
    m = np.roll(np.eye(d), shift=1, axis=0)
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)


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
    m = np.diag(omega ** np.arange(d))
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)


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
    j, k = np.arange(d), np.arange(d)
    m = omega ** np.outer(j, k) / np.sqrt(d)
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)


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
    pd = 0 if d % 2 == 0 else 1

    j = np.arange(d)
    m = np.diag(omega ** ((j * (j + pd)) / 2))
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)


def _check_rotation_indices(
    d: int, *indices: int, distinct_pairs: bool = False
) -> None:
    """
    Validate that indices are within [0, d-1] and optionally form distinct pairs.

    :param d: Qudit dimension.
    :type d: int
    :param indices: Indices to validate.
    :type indices: int
    :param distinct_pairs: If True, enforce that (indices[0], indices[1])
                           ≠ (indices[2], indices[3]) for 4 indices.
    :type distinct_pairs: bool
    :raises ValueError: If indices are invalid.
    """
    for idx in indices:
        if not (0 <= idx < d):
            raise ValueError(f"Index {idx} must satisfy 0 <= index < d (d={d}).")

    if len(indices) == 2 and indices[0] == indices[1]:
        raise ValueError("Rotation requires two distinct levels: j != k.")

    if distinct_pairs and len(indices) == 4:
        j1, k1, j2, k2 = indices
        if j1 == k1 and j2 == k2:
            raise ValueError(
                "Selected basis states must be different: (j1, j2) ≠ (k1, k2)."
            )


def _two_level_projectors(
    d: int, j: int, k: Optional[int] = None
) -> Tuple[Tensor, ...]:
    r"""
    Construct projectors for single- or two-level subspaces in a ``d``-level qudit.

    :param d: Qudit dimension.
    :type d: int
    :param j: First level index.
    :type j: int
    :param k: Optional second level index. If None, only projectors for ``j`` are returned.
    :type k: Optional[int]
    :return:
        - If ``k is None``: ``(I, Pjj)``
        - Else: ``(I, Pjj, Pkk, Pjk, Pkj)``
    :rtype: Tuple[Tensor, ...]
    """
    I = backend.eye(d, dtype=dtypestr)
    ej = I[:, j]
    Pjj = backend.outer_product(ej, ej)

    if k is None:
        return I, Pjj

    ek = I[:, k]
    Pkk = backend.outer_product(ek, ek)
    Pjk = backend.outer_product(ej, ek)
    Pkj = backend.outer_product(ek, ej)
    return I, Pjj, Pkk, Pjk, Pkj


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
    _check_rotation_indices(d, j, k)
    I, Pjj, Pkk, Pjk, Pkj = _two_level_projectors(d, j, k)
    theta = num_to_tensor(theta)
    c = backend.cos(theta / 2.0)
    s = backend.sin(theta / 2.0)
    return I + (c - 1.0) * (Pjj + Pkk) + (-1j * s) * (Pjk + Pkj)


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
    _check_rotation_indices(d, j, k)
    I, Pjj, Pkk, Pjk, Pkj = _two_level_projectors(d, j, k)
    theta = num_to_tensor(theta)
    c = backend.cos(theta / 2.0)
    s = backend.sin(theta / 2.0)
    return I + (c - 1.0) * (Pjj + Pkk) - s * Pjk + s * Pkj


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
    I, Pjj = _two_level_projectors(d, j, k=None)
    theta = num_to_tensor(theta)
    phase = backend.exp(1j * theta)
    return I + (phase - 1.0) * Pjj


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
    I = np.eye(D, dtype=dtypestr)
    m = I.reshape(d, d, d, d).transpose(1, 0, 2, 3).reshape(D, D)
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)


def _rzz_matrix_func(
    d: int, theta: float, j1: int = 0, k1: int = 1, j2: int = 0, k2: int = 1
) -> Tensor:
    r"""
    Two-qudit ``RZZ(\theta)`` on a selected two-state subspace.

    Acts like a qubit :math:`RZZ(\theta)=\exp(-i\,\tfrac{\theta}{2}\,\sigma_z)` on the
    two-dimensional subspace spanned by ``|j1, j2⟩`` and ``|k1, k2⟩``,
    and as identity elsewhere. The resulting block is diagonal with phases
    :math:`\mathrm{diag}(e^{-i\theta/2},\, e^{+i\theta/2})`.

    :param d: Dimension of each qudit (assumed equal).
    :type d: int
    :param theta: Rotation angle.
    :type theta: float
    :param j1: Level on qudit-1 for the first basis state.
    :type j1: int
    :param k1: Level on qudit-1 for the second basis state.
    :type k1: int
    :param j2: Level on qudit-2 for the first basis state.
    :type j2: int
    :param k2: Level on qudit-2 for the second basis state.
    :type k2: int
    :return: ``(d*d, d*d)`` matrix representing subspace :math:`RZZ(\theta)`.
    :rtype: Tensor
    :raises ValueError: If indices are out of range or select the same basis state.
    """
    _check_rotation_indices(d, j1, k1, j2, k2, distinct_pairs=True)
    idx_a = j1 * d + j2
    idx_b = k1 * d + k2
    theta = num_to_tensor(theta)
    phase_minus = backend.exp(-1j * theta / 2.0)
    phase_plus = backend.exp(+1j * theta / 2.0)

    I = backend.eye(d * d, dtype=dtypestr)
    ea = I[:, idx_a]
    eb = I[:, idx_b]
    Paa = backend.outer_product(ea, ea)
    Pbb = backend.outer_product(eb, eb)
    return I + (phase_minus - 1.0) * Paa + (phase_plus - 1.0) * Pbb


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
    _check_rotation_indices(d, j1, k1, j2, k2, distinct_pairs=True)
    idx_a = j1 * d + j2
    idx_b = k1 * d + k2
    theta = num_to_tensor(theta)
    c = backend.cos(theta / 2.0)
    s = backend.sin(theta / 2.0)

    I = backend.eye(d * d, dtype=dtypestr)
    ea = I[:, idx_a]
    eb = I[:, idx_b]
    Paa = backend.outer_product(ea, ea)
    Pbb = backend.outer_product(eb, eb)
    Pab = backend.outer_product(ea, eb)
    Pba = backend.outer_product(eb, ea)
    return I + (c - 1.0) * (Paa + Pbb) + (-1j * s) * (Pab + Pba)


def _u8_matrix_func(
    d: int,
    gamma: float = 2.0,
    z: float = 1.0,
    eps: float = 0.0,
    omega: Optional[float] = None,
) -> Tensor:
    r"""
    ``U8`` diagonal single-qudit gate for prime dimensions.

    This gate represents a canonical nontrivial diagonal Clifford element
    in prime-dimensional qudit systems. Together with generalized Pauli
    operators, it generates the full single-qudit Clifford group. In the
    qubit case (``d=2``), it reduces to the well-known π/8 gate. For higher
    prime dimensions, the phases are defined through modular polynomials
    depending on :math:`\gamma, z, \epsilon`. Its explicit inclusion ensures
    coverage of the complete Clifford generating set across prime qudit
    dimensions.

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
    :raises ValueError: If ``d`` is not prime; if ``gamma==0``; if 12 has no modular
        inverse mod ``d``; or if the computed exponents do not sum to 0 mod ``d``.
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
            inv_12 = pow(12, -1, d)
        except ValueError:
            raise ValueError(
                f"Inverse of 12 mod {d} does not exist. Choose a prime d that does not divide 12."
            )

        for i in range(1, d):
            a = inv_12 * i * (gamma + i * (6 * z + (2 * i - 3) * gamma)) + eps * i
            vks[i] = int(a) % d

    if sum(vks) % d != 0:
        raise ValueError(
            f"Sum of v_k's is not 0 mod {d}. Got {sum(vks) % d}. Check parameters."
        )

    omega = np.exp(2j * np.pi / d) if omega is None else omega
    m = np.diag([omega ** vks[j] for j in range(d)])
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)


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
    r = np.arange(d).reshape(-1, 1)
    s = np.arange(d).reshape(1, -1)

    if cv is None:
        phase = omega ** (r * s)
    else:
        if not (0 <= cv < d):
            raise ValueError(f"cv must be in [0, {d - 1}], got {cv}")
        phase = 1 + (r == cv) * (omega**s - 1)

    diag = np.ravel(phase)
    m = np.diag(diag)
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)


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
    I = np.eye(d)

    if cv is None:
        blocks = [np.roll(I, shift=r, axis=0) for r in range(d)]
        m = np.block(
            [
                [blocks[r] if r == c else np.zeros((d, d)) for c in range(d)]
                for r in range(d)
            ]
        )
        return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)

    if not (0 <= cv < d):
        raise ValueError(f"cv must be in [0, {d - 1}], got {cv}")

    X = np.roll(I, shift=1, axis=0)
    m = np.kron(I, I) + np.kron(np.outer(I[:, cv], I[:, cv]), (X - I))
    return backend.cast(backend.convert_to_tensor(m), dtype=dtypestr)
