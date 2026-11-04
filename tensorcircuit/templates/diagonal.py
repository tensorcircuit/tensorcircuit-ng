"""
Z-polynomial diagonals and elementwise diagonal evolution.

For a fixed Hamiltonian, compute its complete energy vector once and reuse it
for different evolution times. Pad changing term lists with zero-weight
identity terms to reuse one JIT shape. Precision follows ``tc.set_dtype``:
energies use the global real dtype and diagonal action the global complex dtype.
"""

from numbers import Integral
from typing import Any, Optional

import numpy as np

from ..cons import backend, dtypestr, rdtypestr
from ..quantities import _fwht

Tensor = Any


def _check_n(n: int) -> None:
    if not isinstance(n, Integral) or not 0 <= n <= 30:
        raise ValueError("n must be a static integer in [0, 30] for int32 indexing.")


def zpoly_masks(
    structures: Tensor, n: Optional[int] = None, *, encoding: str = "pauli"
) -> Tensor:
    """
    Encode products of Z operators as int32 bit masks.

    Qubit 0 is the most significant bit: a term supported on ``S`` has mask
    ``sum(2**(n - 1 - i) for i in S)``. The identity has mask zero.

    :param structures: With ``encoding="pauli"`` (default), an integer array of
        shape ``[T, n]`` using TensorCircuit's Pauli encoding, restricted to
        ``0=I`` and ``3=Z`` (unlike the 0/1 indicators in
        ``QAOA_ansatz_for_Ising``). Backend tensors may be traced JIT/vmap inputs;
        their values must already satisfy this restriction. Lists and NumPy
        arrays are checked for unsupported Pauli codes on the host.
        With ``encoding="supports"``, a host-side sequence of index sequences,
        e.g. ``[[0], [1, 2], []]`` for ``Z0``, ``Z1 Z2``, and the identity.
        Indices must be integral, in range, and unique within each term.
    :type structures: Tensor or Sequence[Sequence[int]]
    :param n: Static number of qubits, between 0 and 30. Inferred from the last
        dimension for Pauli input; required for supports input. JAX's int32
        index normalization also represents the vector length in int32, so
        ``2**31`` would overflow. This bound is independent of available memory.
    :type n: Optional[int]
    :param encoding: ``"pauli"`` or ``"supports"``; a static Python option.
    :type encoding: str
    :return: Backend int32 tensor of shape ``[T]``. Changing Pauli tensor values
        at fixed shape and dtype does not require a new JIT specialization.
    :rtype: Tensor

    .. code-block:: python

        masks = tc.templates.diagonal.zpoly_masks([[3, 0], [0, 3], [3, 3]])
        # masks represents Z0, Z1, Z0 Z1, respectively: [2, 1, 3]
    """
    if encoding == "supports":
        if n is None:
            raise ValueError("n is required for supports encoding.")
        _check_n(n)
        masks = []
        for support in structures:
            if any(not isinstance(i, Integral) or not 0 <= int(i) < n for i in support):
                raise ValueError("Support indices must be integers in [0, n).")
            if len(set(support)) != len(support):
                raise ValueError("A support must not contain duplicate indices.")
            masks.append(sum(1 << (n - 1 - int(i)) for i in support))
        return backend.convert_to_tensor(masks, dtype="int32")
    if encoding != "pauli":
        raise ValueError("encoding must be 'pauli' or 'supports'.")

    if isinstance(structures, np.ndarray) or not backend.is_tensor(structures):
        structures = np.asarray(structures)
        if not np.all((structures == 0) | (structures == 3)):
            raise ValueError("Pauli structures must contain only 0 (I) and 3 (Z).")
    structures = backend.convert_to_tensor(structures)
    shape = backend.shape_tuple(structures)
    if len(shape) != 2:
        raise ValueError("Pauli structures must have shape [T, n].")
    if n is None:
        n = shape[1]
    _check_n(n)
    if shape[1] != n:
        raise ValueError("Pauli structures width must equal n.")
    positions = backend.convert_to_tensor(
        [1 << (n - 1 - i) for i in range(n)], dtype="int32"
    )
    return backend.cast(
        backend.sum(backend.cast(structures == 3, "int32") * positions, axis=-1),
        "int32",
    )


def zpoly_diagonal(masks: Tensor, weights: Tensor, n: int) -> Tensor:
    """
    Compute the complete diagonal of a real Z-polynomial Hamiltonian.

    Returns ``D[x] = sum_t weights[t] * (-1)**popcount(x & masks[t])`` in
    ``Circuit.state()`` order (qubit 0 is the most significant bit). Weights
    with repeated masks are added; mask zero is a constant term, so padding
    with ``(mask=0, weight=0)`` has no effect.

    Coefficients are scattered into a length ``2**n`` vector and transformed
    by an unnormalized fast Walsh-Hadamard transform. Time is
    ``O(T + n * 2**n)`` and working memory is ``O(T + 2**n)``; no
    ``[T, 2**n]`` or ``[2**n, n]`` intermediate is constructed.

    :param masks: Integer tensor of shape ``[T]``, potentially traced. Values
        must lie in ``[0, 2**n)``; out-of-range values are undefined and are
        not checked here. Use :func:`zpoly_masks` to encode valid structures.
    :type masks: Tensor
    :param weights: Real coefficients of shape ``[T]``, converted to the global
        real dtype ``rdtypestr`` configured by ``tc.set_dtype``.
    :type weights: Tensor
    :param n: Static Python integer between 0 and 30, limited by int32
        vector-length normalization as described in :func:`zpoly_masks`.
    :type n: int
    :return: Real tensor of shape ``[2**n]``, differentiable with respect to
        weights, in the global real dtype. Both inputs support vmap; their
        values may change without recompilation at fixed shape and dtype.
    :rtype: Tensor
    """
    _check_n(n)
    masks = backend.convert_to_tensor(masks)
    if not backend.dtype(masks).startswith(("int", "uint")):
        raise ValueError("masks must have an integer dtype.")
    masks = backend.cast(masks, "int32")
    weights = backend.convert_to_tensor(weights, dtype=rdtypestr)
    shape = backend.shape_tuple(weights)
    if len(shape) != 1 or backend.shape_tuple(masks) != shape:
        raise ValueError("masks and weights must have the same shape [T].")
    values = backend.scatter(
        backend.zeros([1 << n], dtype=rdtypestr),
        backend.reshape(masks, [-1, 1]),
        weights,
        mode="add",
    )
    return _fwht(values)


def apply_diagonal(
    state: Tensor, diag: Tensor, *, theta: Optional[Tensor] = None
) -> Tensor:
    """
    Apply a diagonal operator or its exponential elementwise to a state.

    With ``theta=None``, return ``state * diag``; otherwise return
    ``state * exp(-1j * theta * diag)``. In particular, ``theta=0`` gives the
    original state, whereas ``theta=None`` applies the supplied diagonal.

    :param state: State tensor of any shape, converted to the global complex
        dtype ``dtypestr`` configured by ``tc.set_dtype``.
    :type state: Tensor
    :param diag: Diagonal entries (or generator eigenvalues when theta is
        supplied), with the same number of elements as state. Reshaped to
        match state and converted to the global complex dtype; no assumption
        is made about local Hilbert dimensions.
    :type diag: Tensor
    :param theta: Optional scalar evolution time; keyword-only. May be a
        traced tensor. Use vmap for multiple times or states.
    :type theta: Optional[Tensor]
    :return: Tensor with the same shape as state, in the global complex dtype.
        Both paths are JIT-friendly and differentiable; the presence of theta
        selects the path statically.
    :rtype: Tensor

    .. code-block:: python

        diagonal = tc.templates.diagonal
        masks = diagonal.zpoly_masks([[3, 0], [0, 3], [3, 3]])
        energies = diagonal.zpoly_diagonal(masks, [2.0, 3.0, 5.0], n=2)
        state = tc.backend.ones([4], dtype="complex64") / 2
        evolved = diagonal.apply_diagonal(state, energies, theta=0.2)
    """
    state = backend.convert_to_tensor(state, dtype=dtypestr)
    diag = backend.convert_to_tensor(diag, dtype=dtypestr)
    if backend.sizen(state) != backend.sizen(diag):
        raise ValueError("state and diag must have the same number of elements.")
    diag = backend.reshape(diag, backend.shape_tuple(state))
    if theta is None:
        return state * diag
    theta = backend.convert_to_tensor(theta, dtype=dtypestr)
    if backend.sizen(theta) != 1:
        raise ValueError("theta must be a scalar; use vmap for multiple times.")
    return state * backend.exp(-1j * backend.reshape(theta, []) * diag)
