"""
Functional dense MPS state data and contractions.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..backends.jax_ops import adaware_qr, adaware_svd
from .layout import MPSSpec, block_bucket_plan

Array = Any
NumpyArray = np.ndarray[Any, Any]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockBuffers:
    """
    Bucket-tensor storage with a read-only logical site/block view.

    :ivar spec: Static block layout used to interpret the buckets.
    :ivar buckets: Exact-shape JAX arrays containing the packed blocks.
    """

    spec: Any
    buckets: Tuple[Array, ...]

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], Any]:
        """
        Flatten packed buffers and retain the static layout as metadata.

        :return: Dynamic bucket leaves and the static layout.
        :rtype: Tuple[Tuple[Array, ...], Any]
        """
        return self.buckets, self.spec

    @classmethod
    def tree_unflatten(cls, spec: Any, buckets: Iterable[Array]) -> "BlockBuffers":
        """
        Reconstruct packed buffers from static layout metadata and leaves.

        :param spec: Static block layout.
        :type spec: Any
        :param buckets: Dynamic bucket leaves.
        :type buckets: Iterable[Array]
        :return: Reconstructed bucket storage.
        :rtype: BlockBuffers
        """
        return cls(spec, tuple(buckets))

    def __len__(self) -> int:
        return len(self.spec.physical_dims)

    def __getitem__(self, site: int) -> Tuple[Array, ...]:
        if self.spec.site_layouts is None:
            return (self.buckets[site],)
        plan = block_bucket_plan(self.spec.site_layouts)
        offset = sum(
            len(layout.block_shapes) for layout in self.spec.site_layouts[:site]
        )
        count = len(self.spec.site_layouts[site].block_shapes)
        positions = plan.positions[offset : offset + count]
        shapes = plan.true_shapes[offset : offset + count]
        return tuple(
            self.buckets[bucket][
                (slot,) + tuple(slice(0, dimension) for dimension in shape)
            ]
            for (bucket, slot), shape in zip(positions, shapes)
        )

    def __iter__(self) -> Iterator[Tuple[Array, ...]]:
        return (self[site] for site in range(len(self)))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MPSState:
    """
    A fixed-shape MPS whose numerical buffers are JAX PyTree leaves.

    :ivar spec: Static physical, virtual, and symmetry layout.
    :ivar buffers: Dense site buffers or symmetric block buckets.
    """

    spec: MPSSpec
    buffers: Any

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], MPSSpec]:
        """
        Flatten MPS buffers and retain the static specification as metadata.

        :return: Dynamic buffer leaves and the static MPS specification.
        :rtype: Tuple[Tuple[Array, ...], MPSSpec]
        """
        if isinstance(self.buffers, BlockBuffers):
            return self.buffers.buckets, self.spec
        leaves = tuple(buffer for site in self.buffers for buffer in site)
        return leaves, self.spec

    @classmethod
    def tree_unflatten(cls, spec: MPSSpec, leaves: Iterable[Array]) -> "MPSState":
        """
        Reconstruct an MPS state from static metadata and dynamic leaves.

        :param spec: Static MPS specification.
        :type spec: MPSSpec
        :param leaves: Flattened dense or block-bucket leaves.
        :type leaves: Iterable[Array]
        :return: Reconstructed MPS state.
        :rtype: MPSState
        """
        leaves = tuple(leaves)
        if spec.symmetry is not None and spec.site_layouts is not None:
            plan = block_bucket_plan(spec.site_layouts)
            if len(leaves) != len(plan.shapes):
                raise ValueError(
                    "MPS PyTree buckets do not match the static block layout"
                )
            return cls(spec, BlockBuffers(spec, leaves))
        counts = _buffer_counts(spec)
        if len(leaves) != sum(counts):
            raise ValueError("MPS PyTree leaves do not match the static block layout")
        offset = 0
        buffers: list[tuple[Array, ...]] = []
        for count in counts:
            buffers.append(leaves[offset : offset + count])
            offset += count
        return cls(spec, tuple(buffers))


def pack_block_buffers(
    spec: Any, buffers: Tuple[Tuple[Array, ...], ...]
) -> Tuple[Array, ...]:
    """
    Stack legal blocks into exact-shape tensor buckets.

    :param spec: Dense or block-sparse static layout.
    :type spec: Any
    :param buffers: Logical site/block buffers in layout order.
    :type buffers: Tuple[Tuple[Array, ...], ...]
    :return: Packed exact-shape buckets.
    :rtype: Tuple[Array, ...]
    """
    if spec.site_layouts is None:
        return tuple(buffer for site in buffers for buffer in site)
    plan = block_bucket_plan(spec.site_layouts)
    buckets: list[list[Array]] = [[] for _ in plan.shapes]
    for block, (bucket, _) in zip(
        (buffer for site in buffers for buffer in site),
        plan.positions,
    ):
        buckets[bucket].append(block)
    return tuple(jnp.stack(tuple(blocks)) for blocks in buckets)


def unpack_block_buffers(
    spec: Any, buckets: Tuple[Array, ...]
) -> Tuple[Tuple[Array, ...], ...]:
    """
    Recover logical blocks from exact-shape bucket tensors.

    :param spec: Dense or block-sparse static layout.
    :type spec: Any
    :param buckets: Packed exact-shape bucket tensors.
    :type buckets: Tuple[Array, ...]
    :return: Logical site/block buffers in layout order.
    :rtype: Tuple[Tuple[Array, ...], ...]
    """
    if spec.site_layouts is None:
        return tuple((buffer,) for buffer in buckets)
    plan = block_bucket_plan(spec.site_layouts)
    blocks = tuple(
        buckets[bucket][(slot,) + tuple(slice(0, dimension) for dimension in shape)]
        for (bucket, slot), shape in zip(plan.positions, plan.true_shapes)
    )
    offset = 0
    sites = []
    for layout in spec.site_layouts:
        count = len(layout.block_shapes)
        sites.append(blocks[offset : offset + count])
        offset += count
    return tuple(sites)


def _dense_tensors(mps: MPSState) -> Tuple[Array, ...]:
    """
    Return dense tensors, reconstructing symmetric blocks when necessary.
    """
    if mps.spec.symmetry is None:
        return tuple(site[0] for site in mps.buffers)
    if mps.spec.site_layouts is None:
        raise ValueError("symmetric MPS specifications require site layouts")
    return tuple(
        _blocks_to_dense(mps.spec, site, buffers)
        for site, buffers in enumerate(mps.buffers)
    )


def _with_dense_tensors(spec: MPSSpec, tensors: Tuple[Array, ...]) -> MPSState:
    """
    Build a dense MPS state from tensors already matching ``spec``.
    """
    if spec.symmetry is not None:
        raise ValueError("symmetric MPS states must be built from their block buffers")
    if len(tensors) != spec.nsites:
        raise ValueError("number of tensors does not match MPS specification")
    return MPSState(spec, tuple((tensor,) for tensor in tensors))


def _with_buffers(spec: MPSSpec, buffers: Tuple[Tuple[Array, ...], ...]) -> MPSState:
    """
    Build an MPS state after validating its static block-buffer counts.
    """
    if len(buffers) != spec.nsites or tuple(
        len(site) for site in buffers
    ) != _buffer_counts(spec):
        raise ValueError("MPS buffers do not match the specification layout")
    if spec.symmetry is not None:
        return MPSState(spec, BlockBuffers(spec, pack_block_buffers(spec, buffers)))
    return MPSState(spec, buffers)


def _buffer_counts(spec: MPSSpec) -> Tuple[int, ...]:
    if spec.site_layouts is None:
        return (1,) * spec.nsites
    return tuple(len(layout.block_coordinates) for layout in spec.site_layouts)


def _blocks_to_dense(spec: MPSSpec, site: int, buffers: Tuple[Array, ...]) -> Array:
    layout = spec.site_layouts[site]  # type: ignore[index]
    tensor = jnp.zeros(
        (spec.bond_dims[site], spec.physical_dims[site], spec.bond_dims[site + 1]),
        dtype=buffers[0].dtype,
    )
    offsets = tuple(index.offsets for index in layout.indices)
    for block, coordinate, shape in zip(
        buffers, layout.block_coordinates, layout.block_shapes
    ):
        starts = tuple(
            offsets[axis][position] for axis, position in enumerate(coordinate)
        )
        slices = tuple(slice(start, start + size) for start, size in zip(starts, shape))
        tensor = tensor.at[slices].set(block)
    return tensor


def _right_canonicalize(tensors: Tuple[Array, ...]) -> Tuple[Array, ...]:
    """
    Move the orthogonality center to site zero without changing the state.
    """
    mutable_tensors: list[Array] = list(tensors)
    for site in range(len(mutable_tensors) - 1, 0, -1):
        tensor = mutable_tensors[site]
        left_dim, physical_dim, right_dim = tensor.shape
        matrix = jnp.reshape(tensor, (left_dim, physical_dim * right_dim))
        q_transpose, r_transpose = adaware_qr(jnp.transpose(matrix))
        mutable_tensors[site] = jnp.reshape(
            jnp.transpose(q_transpose), (left_dim, physical_dim, right_dim)
        )
        mutable_tensors[site - 1] = jnp.einsum(
            "lpa,ab->lpb", mutable_tensors[site - 1], jnp.transpose(r_transpose)
        )
    return tuple(mutable_tensors)


def _right_canonicalize_host(tensors: Tuple[NumpyArray, ...]) -> Tuple[NumpyArray, ...]:
    """
    Canonicalize static dense product-state buffers before device transfer.
    """
    mutable_tensors = list(tensors)
    for site in range(len(mutable_tensors) - 1, 0, -1):
        tensor = mutable_tensors[site]
        left_dim, physical_dim, right_dim = tensor.shape
        matrix = np.reshape(tensor, (left_dim, physical_dim * right_dim))
        q_transpose, r_transpose = np.linalg.qr(np.transpose(matrix), mode="reduced")
        mutable_tensors[site] = np.reshape(
            np.transpose(q_transpose), (left_dim, physical_dim, right_dim)
        )
        mutable_tensors[site - 1] = np.einsum(
            "lpa,ab->lpb", mutable_tensors[site - 1], np.transpose(r_transpose)
        )
    return tuple(mutable_tensors)


def _right_canonicalize_symmetric(
    spec: MPSSpec, buffers: Tuple[Tuple[Array, ...], ...]
) -> Tuple[Tuple[Array, ...], ...]:
    """
    Right-canonicalize each Abelian left-charge sector independently.
    """
    if spec.site_layouts is None:
        raise ValueError("symmetric MPS specification is incomplete")
    mutable_buffers: list[list[Array]] = [
        list(site_buffers) for site_buffers in buffers
    ]
    for site in range(spec.nsites - 1, 0, -1):
        layout = spec.site_layouts[site]
        previous_layout = spec.site_layouts[site - 1]
        for left_sector, left_dimension in enumerate(layout.indices[0].degeneracies):
            positions = [
                position
                for position, coordinate in enumerate(layout.block_coordinates)
                if coordinate[0] == left_sector
            ]
            widths = [
                layout.block_shapes[position][1] * layout.block_shapes[position][2]
                for position in positions
            ]
            matrix = jnp.concatenate(
                [
                    jnp.reshape(
                        mutable_buffers[site][position], (left_dimension, width)
                    )
                    for position, width in zip(positions, widths)
                ],
                axis=1,
            )
            q_transpose, r_transpose = adaware_qr(jnp.transpose(matrix))
            q_matrix = jnp.transpose(q_transpose)
            transfer = jnp.transpose(r_transpose)
            offset = 0
            for position, width in zip(positions, widths):
                block_shape = layout.block_shapes[position]
                mutable_buffers[site][position] = jnp.reshape(
                    q_matrix[:, offset : offset + width], block_shape
                )
                offset += width
            for position, coordinate in enumerate(previous_layout.block_coordinates):
                if coordinate[2] == left_sector:
                    mutable_buffers[site - 1][position] = jnp.einsum(
                        "lpa,ab->lpb",
                        mutable_buffers[site - 1][position],
                        transfer,
                    )
    return tuple(tuple(site_buffers) for site_buffers in mutable_buffers)


def _right_canonicalize_symmetric_host(
    spec: MPSSpec, buffers: Tuple[Tuple[NumpyArray, ...], ...]
) -> Tuple[Tuple[NumpyArray, ...], ...]:
    """
    Canonicalize static symmetric product-state buffers before device transfer.
    """
    if spec.site_layouts is None:
        raise ValueError("symmetric MPS specification is incomplete")
    mutable_buffers: list[list[NumpyArray]] = [
        list(site_buffers) for site_buffers in buffers
    ]
    for site in range(spec.nsites - 1, 0, -1):
        layout = spec.site_layouts[site]
        previous_layout = spec.site_layouts[site - 1]
        for left_sector, left_dimension in enumerate(layout.indices[0].degeneracies):
            positions = [
                position
                for position, coordinate in enumerate(layout.block_coordinates)
                if coordinate[0] == left_sector
            ]
            widths = [
                layout.block_shapes[position][1] * layout.block_shapes[position][2]
                for position in positions
            ]
            matrix = np.concatenate(
                [
                    np.reshape(mutable_buffers[site][position], (left_dimension, width))
                    for position, width in zip(positions, widths)
                ],
                axis=1,
            )
            q_transpose, r_transpose = np.linalg.qr(np.transpose(matrix))
            q_matrix = np.transpose(q_transpose)
            transfer = np.transpose(r_transpose)
            offset = 0
            for position, width in zip(positions, widths):
                block_shape = layout.block_shapes[position]
                mutable_buffers[site][position] = np.reshape(
                    q_matrix[:, offset : offset + width], block_shape
                )
                offset += width
            for position, coordinate in enumerate(previous_layout.block_coordinates):
                if coordinate[2] == left_sector:
                    mutable_buffers[site - 1][position] = np.einsum(
                        "lpa,ab->lpb",
                        mutable_buffers[site - 1][position],
                        transfer,
                    )
    return tuple(tuple(site_buffers) for site_buffers in mutable_buffers)


def product_state(
    basis: Tuple[int, ...], *, spec: MPSSpec, dtype: Any
) -> Tuple[MPSState, dict[str, Array]]:
    """
    Embed a computational-basis product state in a fixed MPS specification.

    :param basis: One computational-basis index for every physical site.
    :type basis: Tuple[int, ...]
    :param spec: Target dense or symmetric MPS layout.
    :type spec: MPSSpec
    :param dtype: Dtype of the created tensor buffers.
    :type dtype: Any
    :return: The product-state MPS and norm/conversion diagnostics.
    :rtype: Tuple[MPSState, dict[str, Array]]
    """
    if len(basis) != spec.nsites:
        raise ValueError("basis must provide one local basis index per site")
    if spec.symmetry is not None:
        return _symmetric_product_state(basis, spec, dtype)
    tensors = []
    np_dtype = np.dtype(dtype)
    for site, value in enumerate(basis):
        physical_dim = spec.physical_dims[site]
        if value < 0 or value >= physical_dim:
            raise ValueError(f"basis index {value} is invalid at site {site}")
        tensor = np.zeros(
            (
                spec.bond_dims[site],
                physical_dim,
                spec.bond_dims[site + 1],
            ),
            dtype=np_dtype,
        )
        tensor[0, value, 0] = 1
        tensors.append(tensor)
    state = _with_dense_tensors(
        spec,
        tuple(
            jnp.asarray(tensor) for tensor in _right_canonicalize_host(tuple(tensors))
        ),
    )
    return state, {"input_norm": jnp.asarray(1.0), "output_norm": jnp.asarray(1.0)}


def _basis_sector_position(index: Any, basis_position: int) -> Tuple[int, int]:
    canonical_position = (
        index.basis_to_canonical[basis_position]
        if index.basis_to_canonical
        else basis_position
    )
    offset = 0
    for sector, degeneracy in enumerate(index.degeneracies):
        if canonical_position < offset + degeneracy:
            return sector, canonical_position - offset
        offset += degeneracy
    raise ValueError("basis position is outside the sector index")


def _symmetric_product_state(
    basis: Tuple[int, ...], spec: MPSSpec, dtype: Any
) -> Tuple[MPSState, dict[str, Array]]:
    if (
        spec.physical_sector_indices is None
        or spec.bond_indices is None
        or spec.site_layouts is None
        or spec.total_charge is None
        or spec.symmetry is None
    ):
        raise ValueError("symmetric MPS specification is incomplete")
    accumulated_charge = (0,) * spec.symmetry.rank
    buffers = []
    np_dtype = np.dtype(dtype)
    for site, basis_position in enumerate(basis):
        physical = spec.physical_sector_indices[site]
        if basis_position < 0 or basis_position >= physical.dimension:
            raise ValueError(f"basis index {basis_position} is invalid at site {site}")
        physical_sector, physical_offset = _basis_sector_position(
            physical, basis_position
        )
        left = spec.bond_indices[site]
        right = spec.bond_indices[site + 1]
        left_sector = left.charges.index(accumulated_charge)
        accumulated_charge = spec.symmetry.add(
            accumulated_charge, physical.charges[physical_sector]
        )
        if accumulated_charge not in right.charges:
            raise ValueError(
                f"product state leaves the allocated sector space at site {site}"
            )
        right_sector = right.charges.index(accumulated_charge)
        layout = spec.site_layouts[site]
        site_buffers = [
            np.zeros(shape, dtype=np_dtype) for shape in layout.block_shapes
        ]
        try:
            block_position = layout.block_coordinates.index(
                (left_sector, physical_sector, right_sector)
            )
        except ValueError as error:
            raise ValueError(
                f"product state is not allowed by the MPS layout at site {site}"
            ) from error
        site_buffers[block_position][0, physical_offset, 0] = 1
        buffers.append(tuple(site_buffers))
    if accumulated_charge != spec.total_charge:
        raise ValueError(
            "product-state total charge does not match the MPS specification"
        )
    state = _with_buffers(
        spec,
        tuple(
            tuple(jnp.asarray(block) for block in site_buffers)
            for site_buffers in _right_canonicalize_symmetric_host(spec, tuple(buffers))
        ),
    )
    return state, {"input_norm": jnp.asarray(1.0), "output_norm": jnp.asarray(1.0)}


def random_mps(
    key: Array, *, spec: MPSSpec, dtype: Any
) -> Tuple[MPSState, dict[str, Array]]:
    """
    Create a normalized random MPS in the specified fixed space.

    :param key: JAX random key used to generate the tensor buffers.
    :type key: Array
    :param spec: Target dense or symmetric MPS layout.
    :type spec: MPSSpec
    :param dtype: Dtype of the random tensor buffers.
    :type dtype: Any
    :return: The normalized random MPS and norm/conversion diagnostics.
    :rtype: Tuple[MPSState, dict[str, Array]]
    """
    if spec.symmetry is not None:
        if spec.site_layouts is None:
            raise ValueError("symmetric MPS specification is incomplete")
        keys = jax.random.split(key, sum(_buffer_counts(spec)))
        key_position = 0
        initial_buffers: list[tuple[Array, ...]] = []
        real_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
        is_complex = jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)
        for layout in spec.site_layouts:
            site_buffers = []
            scale = (
                sum(np.prod(shape) for shape in layout.block_shapes)
                / layout.indices[0].dimension
                * (2 if is_complex else 1)
            ) ** 0.5
            for shape in layout.block_shapes:
                site_key = keys[key_position]
                key_position += 1
                if is_complex:
                    real_key, imag_key = jax.random.split(site_key)
                    block = jax.random.normal(real_key, shape, dtype=real_dtype)
                    block = block + 1j * jax.random.normal(
                        imag_key, shape, dtype=real_dtype
                    )
                else:
                    block = jax.random.normal(site_key, shape, dtype=dtype)
                site_buffers.append(jnp.asarray(block / scale, dtype=dtype))
            initial_buffers.append(tuple(site_buffers))
        state = _with_buffers(
            spec, _right_canonicalize_symmetric(spec, tuple(initial_buffers))
        )
        initial_norm = norm(state)
        normalized_buffers: list[tuple[Array, ...]] = list(state.buffers)
        normalized_buffers[0] = tuple(
            buffer / initial_norm for buffer in normalized_buffers[0]
        )
        state = _with_buffers(spec, tuple(normalized_buffers))
        return state, {
            "input_norm": initial_norm,
            "output_norm": norm(state),
            "discarded_weight_abs": jnp.zeros(
                (spec.nsites - 1,), dtype=initial_norm.dtype
            ),
        }
    keys = jax.random.split(key, spec.nsites)
    is_complex = jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)
    tensors = []
    for site, site_key in enumerate(keys):
        shape = (
            spec.bond_dims[site],
            spec.physical_dims[site],
            spec.bond_dims[site + 1],
        )
        if is_complex:
            real_key, imag_key = jax.random.split(site_key)
            tensor = jax.random.normal(
                real_key, shape, dtype=jnp.real(jnp.zeros((), dtype=dtype)).dtype
            )
            tensor = tensor + 1j * jax.random.normal(
                imag_key, shape, dtype=jnp.real(jnp.zeros((), dtype=dtype)).dtype
            )
        else:
            tensor = jax.random.normal(site_key, shape, dtype=dtype)
        scale = (shape[1] * shape[2] * (2 if is_complex else 1)) ** 0.5
        tensors.append(jnp.asarray(tensor / scale, dtype=dtype))
    tensors = list(_right_canonicalize(tuple(tensors)))
    state = _with_dense_tensors(spec, tuple(tensors))
    initial_norm = norm(state)
    tensors[0] = tensors[0] / initial_norm
    state = _with_dense_tensors(spec, tuple(tensors))
    return state, {
        "input_norm": initial_norm,
        "output_norm": norm(state),
        "discarded_weight_abs": jnp.zeros((spec.nsites - 1,), dtype=initial_norm.dtype),
    }


def _embed_tensors(tensors: Sequence[Array], spec: MPSSpec) -> Tuple[Array, ...]:
    """
    Embed a lower-capacity dense MPS exactly in ``spec``.
    """
    embedded = []
    for site, tensor in enumerate(tensors):
        left_dim, physical_dim, right_dim = tensor.shape
        target_shape = (
            spec.bond_dims[site],
            spec.physical_dims[site],
            spec.bond_dims[site + 1],
        )
        if physical_dim != target_shape[1]:
            raise ValueError(f"physical dimension mismatch at site {site}")
        if left_dim > target_shape[0] or right_dim > target_shape[2]:
            raise ValueError("source MPS does not fit in the requested specification")
        target = jnp.zeros(target_shape, dtype=tensor.dtype)
        embedded.append(target.at[:left_dim, :, :right_dim].set(tensor))
    return tuple(embedded)


def _compress_tensors(
    tensors: Sequence[Array], spec: MPSSpec
) -> Tuple[Tuple[Array, ...], Array]:
    """
    Compress a canonical dense MPS with the fixed quota in ``spec``.
    """
    tensors = list(_right_canonicalize(tuple(tensors)))
    discarded = []
    for site in range(spec.nsites - 1):
        left = tensors[site]
        right = tensors[site + 1]
        left_dim, physical_left, _ = left.shape
        _, physical_right, right_dim = right.shape
        matrix = jnp.reshape(
            jnp.einsum("lpm,mqr->lpqr", left, right),
            (left_dim * physical_left, physical_right * right_dim),
        )
        keep = spec.bond_dims[site + 1]
        u, singular_values, vh = adaware_svd(matrix)
        if keep > singular_values.shape[0]:
            raise ValueError(
                "target bond dimension exceeds the available local capacity"
            )
        discarded.append(jnp.sum(singular_values[keep:] ** 2))
        tensors[site] = jnp.reshape(
            u[:, :keep],
            (left_dim, physical_left, keep),
        )
        tensors[site + 1] = jnp.reshape(
            singular_values[:keep, None] * vh[:keep],
            (keep, physical_right, right_dim),
        )
    return _right_canonicalize(tuple(tensors)), jnp.stack(discarded)


def as_mps(
    source: Any, *, spec: MPSSpec, truncate: bool = False
) -> Tuple[MPSState, dict[str, Array]]:
    """
    Import a dense MPS without mutating the source object.

    Tensor sequences, TensorCircuit ``MPSCircuit`` objects, and TensorNetwork
    finite MPS objects are accepted. Smaller compatible bonds are embedded
    exactly; larger bonds require explicit fixed-quota compression.

    :param source: MPSState, tensor sequence, MPSCircuit, or finite MPS source.
    :type source: Any
    :param spec: Target dense MPS layout.
    :type spec: MPSSpec
    :param truncate: Whether to compress oversized bonds to the target quotas.
    :type truncate: bool
    :return: The imported MPS and norm/truncation diagnostics.
    :rtype: Tuple[MPSState, dict[str, Array]]
    """
    if isinstance(source, MPSState) and source.spec == spec:
        source_norm = norm(source)
        return source, {
            "input_norm": source_norm,
            "output_norm": source_norm,
            "discarded_weight_abs": jnp.zeros(
                (spec.nsites - 1,), dtype=source_norm.dtype
            ),
        }
    if spec.symmetry is not None:
        raise ValueError("symmetric import requires sector-aware source metadata")
    if isinstance(source, MPSState):
        if source.spec.symmetry is not None:
            raise ValueError(
                "symmetric MPS import cannot implicitly densify; use to_tn_mps with allow_dense=True first"
            )
        source_tensors = _dense_tensors(source)
    elif hasattr(source, "get_tensors"):
        source_tensors = tuple(source.get_tensors())
    elif hasattr(source, "tensors"):
        source_tensors = tuple(source.tensors)
    else:
        source_tensors = tuple(source)
    if len(source_tensors) != spec.nsites:
        raise ValueError("source MPS has a different number of sites")
    tensors = tuple(jnp.asarray(tensor) for tensor in source_tensors)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("each source MPS tensor must have rank three")
    source_bonds = (tensors[0].shape[0],) + tuple(tensor.shape[2] for tensor in tensors)
    if source_bonds[0] != 1 or source_bonds[-1] != 1:
        raise ValueError("source MPS must use open boundary bonds of dimension one")
    source_state = _with_dense_tensors(
        MPSSpec.dense(
            tuple(tensor.shape[1] for tensor in tensors),
            tuple(int(dimension) for dimension in source_bonds),
        ),
        tensors,
    )
    source_norm = norm(source_state)
    if all(source <= target for source, target in zip(source_bonds, spec.bond_dims)):
        prepared_tensors = _right_canonicalize(_embed_tensors(tensors, spec))
        discarded = jnp.zeros((spec.nsites - 1,), dtype=source_norm.dtype)
    elif truncate and all(
        source >= target for source, target in zip(source_bonds, spec.bond_dims)
    ):
        prepared_tensors, discarded = _compress_tensors(tensors, spec)
    else:
        raise ValueError(
            "source bond dimensions are incompatible with spec; request truncate=True "
            "only for a smaller fixed-quota specification"
        )
    state = _with_dense_tensors(spec, prepared_tensors)
    return state, {
        "input_norm": source_norm,
        "output_norm": norm(state),
        "discarded_weight_abs": discarded,
    }


def _symmetric_overlap(left: MPSState, right: MPSState) -> Array:
    """
    Contract the charge-diagonal overlap environment without densification.
    """
    assert left.spec.site_layouts is not None
    dtype = jnp.result_type(left.buffers[0][0], right.buffers[0][0])
    environment: Tuple[Array, ...] = (jnp.ones((1, 1), dtype=dtype),)
    for site, layout in enumerate(left.spec.site_layouts):
        updated = [
            jnp.zeros((d, d), dtype=dtype) for d in layout.indices[2].degeneracies
        ]
        for coordinate, bra, ket in zip(
            layout.block_coordinates, left.buffers[site], right.buffers[site]
        ):
            updated[coordinate[2]] = updated[coordinate[2]] + jnp.einsum(
                "ab,apc,bpd->cd", environment[coordinate[0]], jnp.conj(bra), ket
            )
        environment = tuple(updated)
    return environment[0][0, 0]


def norm(mps: MPSState) -> Array:
    """
    Return the MPS two-norm using a full differentiable contraction.

    :param mps: MPS state whose norm is evaluated.
    :type mps: MPSState
    :return: The non-negative MPS two-norm.
    :rtype: Array
    """
    if mps.spec.symmetry is not None:
        return jnp.sqrt(jnp.real(_symmetric_overlap(mps, mps)))
    environment = jnp.ones((1, 1), dtype=_dense_tensors(mps)[0].dtype)
    for tensor in _dense_tensors(mps):
        environment = jnp.einsum(
            "ab,apc,bpd->cd", environment, jnp.conj(tensor), tensor
        )
    return jnp.sqrt(jnp.real(environment[0, 0]))


def overlap(left: MPSState, right: MPSState) -> Array:
    """
    Return the differentiable inner product ``<left|right>``.

    :param left: Bra MPS state.
    :type left: MPSState
    :param right: Ket MPS state with the same static specification.
    :type right: MPSState
    :return: The scalar MPS overlap.
    :rtype: Array
    """
    if left.spec != right.spec:
        raise ValueError("overlap requires identical MPS specifications")
    if left.spec.symmetry is not None:
        return _symmetric_overlap(left, right)
    environment = jnp.ones(
        (1, 1), dtype=jnp.result_type(_dense_tensors(left)[0], _dense_tensors(right)[0])
    )
    for left_tensor, right_tensor in zip(_dense_tensors(left), _dense_tensors(right)):
        environment = jnp.einsum(
            "ab,apc,bpd->cd",
            environment,
            jnp.conj(left_tensor),
            right_tensor,
        )
    return environment[0, 0]
