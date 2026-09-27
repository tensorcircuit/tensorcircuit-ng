"""
Portable host-array checkpointing for fixed-shape TNALG states.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np

from .layout import MPSSpec
from .mps import MPSState, _with_buffers
from .symmetry import AbelianSymmetry, SectorIndex

_SCHEMA_VERSION = 2


def _index_metadata(index: SectorIndex) -> Dict[str, Any]:
    return {
        "charges": [list(charge) for charge in index.charges],
        "degeneracies": list(index.degeneracies),
        "flow": index.flow,
        "basis_to_canonical": list(index.basis_to_canonical),
        "canonical_to_basis": list(index.canonical_to_basis),
    }


def _spec_metadata(spec: MPSSpec) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "physical_dims": list(spec.physical_dims),
        "bond_dims": list(spec.bond_dims),
    }
    if spec.symmetry is None:
        metadata["kind"] = "dense"
        return metadata
    if (
        spec.total_charge is None
        or spec.physical_sector_indices is None
        or spec.bond_sectors is None
    ):
        raise ValueError("symmetric MPS specification is incomplete")
    metadata.update(
        {
            "kind": "symmetric",
            "moduli": list(spec.symmetry.moduli),
            "total_charge": list(spec.total_charge),
            "physical_indices": [
                _index_metadata(index) for index in spec.physical_sector_indices
            ],
            "bond_sectors": [
                [[list(charge), degeneracy] for charge, degeneracy in sectors]
                for sectors in spec.bond_sectors
            ],
        }
    )
    return metadata


def _restore_index(symmetry: AbelianSymmetry, metadata: Dict[str, Any]) -> SectorIndex:
    return SectorIndex(
        symmetry=symmetry,
        charges=tuple(tuple(charge) for charge in metadata["charges"]),
        degeneracies=tuple(metadata["degeneracies"]),
        flow=metadata["flow"],
        basis_to_canonical=tuple(metadata["basis_to_canonical"]),
        canonical_to_basis=tuple(metadata["canonical_to_basis"]),
    )


def _restore_spec(metadata: Dict[str, Any]) -> MPSSpec:
    if metadata["kind"] == "dense":
        return MPSSpec.dense(
            physical_dims=tuple(metadata["physical_dims"]),
            bond_dims=tuple(metadata["bond_dims"]),
        )
    symmetry = AbelianSymmetry(tuple(metadata["moduli"]))
    physical_indices = tuple(
        _restore_index(symmetry, index) for index in metadata["physical_indices"]
    )
    bond_sectors = tuple(
        {tuple(charge): degeneracy for charge, degeneracy in sectors}
        for sectors in metadata["bond_sectors"]
    )
    return MPSSpec.from_sectors(
        physical_indices=physical_indices,
        total_charge=tuple(metadata["total_charge"]),
        bond_sectors=bond_sectors,
    )


def save_checkpoint(path: Union[str, Path], mps: MPSState) -> None:
    """
    Save MPS buffers and JSON-compatible static metadata in an NPZ archive.

    :param path: Destination NPZ checkpoint path.
    :type path: Union[str, Path]
    :param mps: MPS state to serialize.
    :type mps: MPSState
    """
    arrays: Dict[str, Any] = {
        f"buffer_{site}_{block}": np.asarray(buffer)
        for site, site_buffers in enumerate(mps.buffers)
        for block, buffer in enumerate(site_buffers)
    }
    metadata: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "spec": _spec_metadata(mps.spec),
        "buffer_counts": [len(site_buffers) for site_buffers in mps.buffers],
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(Path(path), **arrays)


def load_checkpoint(path: Union[str, Path]) -> MPSState:
    """
    Load a checkpoint without restoring executable, tracer, or callback state.

    :param path: Source NPZ checkpoint path.
    :type path: Union[str, Path]
    :return: The restored MPS state.
    :rtype: MPSState
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("unsupported TNALG checkpoint schema version")
        spec = _restore_spec(metadata["spec"])
        expected_counts = (
            tuple(len(layout.block_shapes) for layout in spec.site_layouts)
            if spec.site_layouts is not None
            else (1,) * spec.nsites
        )
        if tuple(metadata["buffer_counts"]) != expected_counts:
            raise ValueError("checkpoint buffer layout differs from its restored spec")
        buffers: Tuple[Tuple[Any, ...], ...] = tuple(
            tuple(
                jnp.asarray(archive[f"buffer_{site}_{block}"]) for block in range(count)
            )
            for site, count in enumerate(metadata["buffer_counts"])
        )
        mps = _with_buffers(spec, buffers)
        return mps
