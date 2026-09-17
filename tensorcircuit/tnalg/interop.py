"""
Explicit host-side conversion between functional and legacy dense MPS APIs.
"""

from typing import Any

import jax.numpy as jnp
import tensornetwork as tn

from ..mps_base import FiniteMPS
from ..mpscircuit import MPSCircuit
from .mpo import MPOState, _dense_tensors as _dense_mpo_tensors
from .mps import MPSState, _dense_tensors


def _export_tensors(
    state: MPSState, *, allow_dense: bool, max_elements: int
) -> list[Any]:
    if state.spec.symmetry is not None and not allow_dense:
        raise ValueError("symmetric MPS export requires allow_dense=True")
    if (
        sum(
            left * physical * right
            for left, physical, right in zip(
                state.spec.bond_dims, state.spec.physical_dims, state.spec.bond_dims[1:]
            )
        )
        > max_elements
    ):
        raise ValueError("dense MPS export exceeds max_elements")
    tensors = _dense_tensors(state)
    if state.spec.physical_sector_indices is not None:
        tensors = tuple(
            (
                jnp.take(tensor, jnp.asarray(index.basis_to_canonical), axis=1)
                if index.basis_to_canonical
                else tensor
            )
            for tensor, index in zip(tensors, state.spec.physical_sector_indices)
        )
    return list(tensors)


def to_tn_mps(
    state: MPSState, *, allow_dense: bool = False, max_elements: int = 1_000_000
) -> FiniteMPS:
    """
    Create an independent legacy ``FiniteMPS`` at orthogonality center zero.

    :param state: TNALG MPS state to export.
    :type state: MPSState
    :param allow_dense: Whether a symmetric state may be materialized densely.
    :type allow_dense: bool
    :param max_elements: Maximum dense MPS elements to materialize.
    :type max_elements: int
    :return: An independent legacy finite MPS.
    :rtype: FiniteMPS
    """
    result = FiniteMPS(
        _export_tensors(state, allow_dense=allow_dense, max_elements=max_elements),
        canonicalize=False,
    )
    result.center_position = 0
    return result


def to_mpscircuit(
    state: MPSState, *, allow_dense: bool = False, max_elements: int = 1_000_000
) -> MPSCircuit:
    """
    Create an independent legacy ``MPSCircuit`` at orthogonality center zero.

    :param state: TNALG MPS state to export.
    :type state: MPSState
    :param allow_dense: Whether a symmetric state may be materialized densely.
    :type allow_dense: bool
    :param max_elements: Maximum dense MPS elements to materialize.
    :type max_elements: int
    :return: An independent legacy MPS circuit.
    :rtype: MPSCircuit
    """
    physical_dims = state.spec.physical_dims
    if len(set(physical_dims)) != 1:
        raise ValueError("MPSCircuit export requires a uniform physical dimension")
    return MPSCircuit(
        state.spec.nsites,
        center_position=0,
        tensors=_export_tensors(
            state, allow_dense=allow_dense, max_elements=max_elements
        ),
        dim=physical_dims[0],
    )


def to_tn_mpo(
    state: MPOState, *, allow_dense: bool = False, max_elements: int = 1_000_000
) -> object:
    """
    Export an MPO in the user physical basis with ``L,R,out,in`` axes.

    Symmetric exports require ``allow_dense=True``. ``max_elements`` bounds
    the total dense site storage before any tensors are allocated.

    :param state: TNALG MPO state to export.
    :type state: MPOState
    :param allow_dense: Whether a symmetric state may be materialized densely.
    :type allow_dense: bool
    :param max_elements: Maximum dense MPO elements to materialize.
    :type max_elements: int
    :return: A legacy finite MPO object.
    :rtype: object
    """
    if state.spec.symmetry is not None and not allow_dense:
        raise ValueError("symmetric MPO export requires allow_dense=True")
    if (
        sum(
            left * physical**2 * right
            for left, physical, right in zip(
                state.spec.bond_dims, state.spec.physical_dims, state.spec.bond_dims[1:]
            )
        )
        > max_elements
    ):
        raise ValueError("dense MPO export exceeds max_elements")
    buffers = _dense_mpo_tensors(state)
    if state.spec.physical_sector_indices is not None:
        buffers = tuple(
            (
                jnp.take(
                    jnp.take(buffer, jnp.asarray(index.basis_to_canonical), axis=1),
                    jnp.asarray(index.basis_to_canonical),
                    axis=2,
                )
                if index.basis_to_canonical
                else buffer
            )
            for buffer, index in zip(buffers, state.spec.physical_sector_indices)
        )
    tensors = [jnp.transpose(buffer, (0, 3, 1, 2)) for buffer in buffers]
    return tn.matrixproductstates.mpo.FiniteMPO(tensors)
