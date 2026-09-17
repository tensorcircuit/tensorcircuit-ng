"""
Pure-tensor operator-string validation shared by TN algorithm builders.
"""

from typing import Any, Tuple

import numpy as np

from .symmetry import SectorIndex

Array = Any


def default_pauli_table() -> np.ndarray[Any, Any]:
    """
    Return TensorCircuit's ``I, X, Y, Z`` operator table.

    :return: Operator matrices ordered as identity, X, Y, and Z.
    :rtype: np.ndarray[Any, Any]
    """
    return np.asarray(  # type: ignore[no-any-return]
        (
            ((1, 0), (0, 1)),
            ((0, 1), (1, 0)),
            ((0, -1j), (1j, 0)),
            ((1, 0), (0, -1)),
        ),
        dtype=np.complex64,
    )


def physical_dims_from_indices(physical_indices: Tuple[Any, ...]) -> Tuple[int, ...]:
    """
    Extract positive dimensions from integers or explicit sector indices.

    :param physical_indices: Local dimensions or explicit sector indices.
    :type physical_indices: Tuple[Any, ...]
    :return: Positive physical dimension at each site.
    :rtype: Tuple[int, ...]
    """
    if not physical_indices:
        raise ValueError("physical_indices must contain at least one site")
    if all(isinstance(index, SectorIndex) for index in physical_indices):
        symmetry = physical_indices[0].symmetry
        if any(
            index.symmetry != symmetry or index.flow != 1 for index in physical_indices
        ):
            raise ValueError("physical indices must share a symmetry and have flow +1")
        return tuple(index.dimension for index in physical_indices)
    if not all(isinstance(index, (int, np.integer)) for index in physical_indices):
        raise TypeError(
            "physical_indices must be all integers or all SectorIndex objects"
        )
    dimensions = tuple(int(index) for index in physical_indices)
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("physical dimensions must be positive")
    return dimensions


def validate_operator_input(
    operator_codes: Array, physical_dims: Tuple[int, ...], operator_table: Any
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Validate concrete fixed operator strings and their local operator table.

    :param operator_codes: Integer operator codes with shape
        ``(n_terms, n_sites)``.
    :type operator_codes: Array
    :param physical_dims: Physical dimension at each site.
    :type physical_dims: Tuple[int, ...]
    :param operator_table: Local operator matrices, or ``None`` for Pauli
        matrices.
    :type operator_table: Any
    :return: Validated NumPy operator codes and local operator table.
    :rtype: Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    """
    codes = np.asarray(operator_codes)
    if codes.ndim != 2 or codes.shape[1] != len(physical_dims):
        raise ValueError("operator_codes must have shape (n_terms, n_sites)")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("operator_codes must use an integer dtype")
    table = (
        default_pauli_table() if operator_table is None else np.asarray(operator_table)
    )
    if table.ndim != 3 or table.shape[1] != table.shape[2]:
        raise ValueError("operator_table must have shape (n_operators, d, d)")
    if any(d != table.shape[1] for d in physical_dims):
        raise ValueError(
            "a single operator_table requires identical physical dimensions"
        )
    if np.any(codes < 0) or np.any(codes >= table.shape[0]):
        raise ValueError("operator_codes contains an operator outside operator_table")
    if not np.allclose(table[0], np.eye(table.shape[1], dtype=table.dtype)):
        raise ValueError("operator_table entry 0 must be the identity")
    return codes, table


def validate_coefficient_indices(
    coefficient_indices: Any, nterms: int
) -> Tuple[np.ndarray[Any, Any], int]:
    """
    Validate static term-to-dynamic-coefficient grouping indices.

    :param coefficient_indices: Term-to-coefficient map, or ``None`` for one
        coefficient per term.
    :type coefficient_indices: Any
    :param nterms: Number of operator terms.
    :type nterms: int
    :return: Contiguous coefficient indices and the number of coefficients.
    :rtype: Tuple[np.ndarray[Any, Any], int]
    """
    if coefficient_indices is None:
        indices = np.arange(nterms, dtype=np.int32)
    else:
        indices = np.asarray(coefficient_indices)
        if indices.shape != (nterms,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                "coefficient_indices must be an integer tensor of shape (n_terms,)"
            )
        if np.any(indices < 0):
            raise ValueError("coefficient_indices must be non-negative")
    ncoefficients = int(np.max(indices)) + 1 if nterms else 0
    if set(indices.tolist()) != set(range(ncoefficients)):
        raise ValueError(
            "coefficient_indices must use contiguous indices starting at zero"
        )
    return indices.astype(np.int32), ncoefficients


def canonical_site_operator_tables(
    table: np.ndarray[Any, Any], physical_indices: Tuple[Any, ...]
) -> Tuple[np.ndarray[Any, Any], ...]:
    """
    Put one shared user-basis operator table into each canonical site basis.

    :param table: Shared local operator table in user-basis order.
    :type table: np.ndarray[Any, Any]
    :param physical_indices: Local dimensions or sector indices.
    :type physical_indices: Tuple[Any, ...]
    :return: One canonical-basis operator table per site.
    :rtype: Tuple[np.ndarray[Any, Any], ...]
    """
    tables = []
    for index in physical_indices:
        permutation = getattr(index, "canonical_to_basis", ())
        if permutation:
            tables.append(table[:, permutation, :][:, :, permutation])
        else:
            tables.append(table)
    return tuple(tables)


def validate_symmetric_operator_groups(
    codes: np.ndarray[Any, Any],
    coefficient_indices: np.ndarray[Any, Any],
    site_tables: Tuple[np.ndarray[Any, Any], ...],
    physical_indices: Tuple[Any, ...],
) -> None:
    """
    Check charge transfer by sector-wise QR, without a dense many-body tensor.

    Each charge sector carries at most one column per tied operator string.
    QR preserves cancellation between strings; the tolerance is relative to
    the normalized prefix Frobenius norm.

    :param codes: Validated operator-code rows.
    :type codes: np.ndarray[Any, Any]
    :param coefficient_indices: Static term-to-coefficient group indices.
    :type coefficient_indices: np.ndarray[Any, Any]
    :param site_tables: Canonical local operator table for every site.
    :type site_tables: Tuple[np.ndarray[Any, Any], ...]
    :param physical_indices: Physical sector indices defining the symmetry.
    :type physical_indices: Tuple[Any, ...]
    :raises ValueError: If a tied operator group breaks the declared symmetry.
    """
    if not isinstance(physical_indices[0], SectorIndex):
        return
    symmetry = physical_indices[0].symmetry
    tolerance = 1e-10
    groups: dict[Any, list[int]] = {}
    for term, (row, coefficient) in enumerate(zip(codes, coefficient_indices)):
        support = tuple(np.flatnonzero(row).tolist())
        groups.setdefault((int(coefficient), support), []).append(term)
    local_transfers = []
    for index in physical_indices:
        charges = tuple(
            charge
            for charge, degeneracy in zip(index.charges, index.degeneracies)
            for _ in range(degeneracy)
        )
        transfers: dict[Any, list[int]] = {}
        for output, output_charge in enumerate(charges):
            for input_, input_charge in enumerate(charges):
                transfer = symmetry.subtract(output_charge, input_charge)
                transfers.setdefault(transfer, []).append(
                    output * index.dimension + input_
                )
        local_transfers.append(transfers)
    zero = (0,) * symmetry.rank
    for (_, support), terms in groups.items():
        factors = {zero: np.ones((1, len(terms)), dtype=np.complex128)}
        for site in support:
            operators = site_tables[site][codes[terms, site]].reshape(len(terms), -1).T
            joined: dict[Any, list[Any]] = {}
            for transfer, positions in local_transfers[site].items():
                local = operators[positions]
                if not np.any(local):
                    continue
                for prefix, factor in factors.items():
                    charge = symmetry.add(prefix, transfer)
                    joined.setdefault(charge, []).append(
                        (factor[:, None, :] * local[None, :, :]).reshape(-1, len(terms))
                    )
            factors = {
                charge: np.linalg.qr(np.concatenate(chunks, axis=0), mode="r")  # type: ignore[misc]
                for charge, chunks in joined.items()
            }
            scale = max(
                (np.linalg.norm(factor) for factor in factors.values()), default=0
            )
            if scale:
                factors = {charge: factor / scale for charge, factor in factors.items()}
        if any(
            np.linalg.norm(np.sum(factor, axis=1)) > tolerance
            for charge, factor in factors.items()
            if charge != zero
        ):
            raise ValueError(
                "coefficient-tied operator group breaks the declared Abelian symmetry"
            )
