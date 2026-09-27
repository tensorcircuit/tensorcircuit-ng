"""
Shared spatial scans over equivalent local block layouts.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .layout import MPOSpec, MPSSpec, block_bucket_plan
from .symmetric_environment import (
    LegalEnvironment,
    _legal_environment_coordinates,
    environment_bucket_plan,
    mpo_left_boundary,
    mpo_right_boundary,
    mpo_update_right,
    prepare_mpo_site,
)

Array = Any
Buckets = Tuple[Array, ...]


@dataclass(frozen=True)
class BlockAccess:
    """
    Batched bucket reads/writes for one local layout at several sites.

    :ivar count: Number of logical blocks in the addressed site window.
    :ivar groups: Static bucket, shape, position, and slot metadata.
    """

    count: int
    groups: Tuple[Any, ...]

    def read(self, buckets: Buckets, row: Array) -> Tuple[Array, ...]:
        """
        Read one site-window row from bucket storage.

        :param buckets: Packed bucket tensors.
        :type buckets: Buckets
        :param row: Row index into the static site schedule.
        :type row: Array
        :return: Logical blocks for the selected row.
        :rtype: Tuple[Array, ...]
        """
        blocks: list[Array] = [None] * self.count
        for bucket, shape, positions, slots, contiguous in self.groups:
            if contiguous:
                values = jax.lax.dynamic_slice(
                    buckets[bucket],
                    (jnp.asarray(slots)[row, 0],) + (np.int32(0),) * len(shape),
                    (len(positions),) + shape,
                )
            else:
                values = buckets[bucket][
                    (jnp.asarray(slots)[row],) + tuple(slice(0, d) for d in shape)
                ]
            for i, position in enumerate(positions):
                blocks[position] = values[i]
        return tuple(blocks)

    def write(self, buckets: Buckets, row: Array, blocks: Tuple[Array, ...]) -> Buckets:
        """
        Write one logical site-window row into bucket storage.

        :param buckets: Packed bucket tensors.
        :type buckets: Buckets
        :param row: Row index into the static site schedule.
        :type row: Array
        :param blocks: Logical blocks to write.
        :type blocks: Tuple[Array, ...]
        :return: Updated packed bucket tensors.
        :rtype: Buckets
        """
        result = list(buckets)
        for bucket, shape, positions, slots, contiguous in self.groups:
            values = jnp.stack(tuple(blocks[position] for position in positions))
            if contiguous:
                result[bucket] = jax.lax.dynamic_update_slice(
                    result[bucket],
                    values,
                    (jnp.asarray(slots)[row, 0],) + (np.int32(0),) * len(shape),
                )
            else:
                index = (jnp.asarray(slots)[row],) + tuple(slice(0, d) for d in shape)
                result[bucket] = result[bucket].at[index].set(values)
        return tuple(result)


def _access(
    plan: Any, offsets: Tuple[int, ...], sites: Tuple[int, ...], count: int
) -> BlockAccess:
    groups: dict[Any, list[int]] = {}
    for block in range(count):
        position = offsets[sites[0]] + block
        bucket, _ = plan.positions[position]
        groups.setdefault((bucket, plan.true_shapes[position]), []).append(block)
    prepared = []
    for (bucket, shape), blocks in groups.items():
        slots = np.asarray(
            [
                [plan.positions[offsets[site] + block][1] for block in blocks]
                for site in sites
            ],
            dtype=np.int32,
        )
        prepared.append(
            (
                bucket,
                shape,
                tuple(blocks),
                slots,
                bool(np.all(np.diff(slots, axis=1) == 1)),
            )
        )
    return BlockAccess(count, tuple(prepared))


def site_access(spec: Any, sites: Tuple[int, ...], *, width: int = 1) -> BlockAccess:
    """
    Prepare addressing for equal-layout site windows, flattened in site order.

    :param spec: Static block layout specification.
    :type spec: Any
    :param sites: Site indices with equivalent local layouts.
    :type sites: Tuple[int, ...]
    :param width: Number of consecutive sites in each addressed window.
    :type width: int
    :return: Static bucket addressing metadata.
    :rtype: BlockAccess
    """
    layouts = spec.site_layouts
    offsets = tuple(
        np.cumsum([0] + [len(layout.block_shapes) for layout in layouts[:-1]])
    )
    return _access(
        block_bucket_plan(layouts),
        offsets,
        sites,
        sum(
            len(layout.block_shapes) for layout in layouts[sites[0] : sites[0] + width]
        ),
    )


def environment_access(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, cuts: Tuple[int, ...]
) -> BlockAccess:
    """
    Prepare local cut addressing.

    :param mps_spec: Static MPS block layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO block layout.
    :type mpo_spec: MPOSpec
    :param cuts: MPS cuts with equivalent environment layouts.
    :type cuts: Tuple[int, ...]
    :return: Static environment bucket addressing metadata.
    :rtype: BlockAccess
    """
    plan = environment_bucket_plan(mps_spec, mpo_spec)
    count = len(_legal_environment_coordinates(mps_spec, mpo_spec, cuts[0]))
    return _access(plan, plan.offsets, cuts, count)


def _layout_key(layout: Any) -> Any:
    return (
        layout.block_coordinates,
        layout.block_shapes,
        tuple(index.degeneracies for index in layout.indices),
    )


@dataclass(frozen=True)
class SiteGroup:
    """
    Sites with identical numerical kernels, irrespective of charge origins.

    :ivar site: Representative site of the group.
    :ivar mps: MPS block accessors for each site in the local window.
    :ivar mpo: MPO block accessors for each site in the local window.
    :ivar environments: Environment accessors for each local cut.
    :ivar sites: Actual site indices represented by the group.
    """

    site: int
    mps: Tuple[BlockAccess, ...]
    mpo: Tuple[BlockAccess, ...]
    environments: Tuple[BlockAccess, ...]
    sites: Tuple[int, ...]


@dataclass(frozen=True)
class SweepSchedule:
    """
    Shared local kernels with statically scheduled periodic bulk scans.

    :ivar groups: Site groups sharing numerical kernels.
    :ivar indices: Static site-to-group and row schedule.
    :ivar programs: Forward and reverse scan programs.
    """

    groups: Tuple[SiteGroup, ...]
    indices: Any
    programs: Tuple[Any, Any]

    def run(
        self,
        kernel: Callable[..., Any],
        carry: Any,
        context: Any = None,
        *,
        reverse: bool = False,
    ) -> Any:
        """
        Run a scheduled local kernel in the requested direction.

        :param kernel: Local kernel receiving a group, row, carry, and context.
        :type kernel: Callable[..., Any]
        :param carry: Initial scan carry.
        :type carry: Any
        :param context: Static or dynamic context passed to each kernel.
        :type context: Any
        :param reverse: Run the reverse spatial program when true.
        :type reverse: bool
        :return: Final scan carry.
        :rtype: Any
        """
        if not self.groups:
            return carry
        kernels = tuple(
            lambda value, record, group=group: kernel(group, record[0], value, context)
            for group in self.groups
        )
        return run_program(self.programs[int(reverse)], kernels, carry)


def prepare_program(indices: Any) -> Tuple[Any, ...]:
    """
    Extract repeating operation patterns, leaving boundary operations static.

    :param indices: Integer schedule with a group code in its first column.
    :type indices: Any
    :return: Static periodic run program.
    :rtype: Tuple[Any, ...]
    """
    runs = []
    start = 0
    while start < len(indices):
        best_period, best_count = 1, 1
        candidates = np.flatnonzero(indices[start + 1 :, 0] == indices[start, 0]) + 1
        for period in candidates[candidates <= (len(indices) - start) // 2]:
            period = int(period)
            count = 1
            pattern = indices[start : start + period, 0]
            while start + (count + 1) * period <= len(indices) and np.array_equal(
                pattern,
                indices[start + count * period : start + (count + 1) * period, 0],
            ):
                count += 1
            if count > 1 and count * period > best_count * best_period:
                best_period, best_count = period, count
        stop = start + best_period * best_count
        runs.append(
            (
                tuple(int(code) for code in indices[start : start + best_period, 0]),
                indices[start:stop, 1:].reshape(best_count, best_period, -1),
            )
        )
        start = stop
    return tuple(runs)


def run_program(
    program: Any, kernels: Tuple[Callable[..., Any], ...], carry: Any
) -> Any:
    """
    Scan periodic runs directly so local buffer updates can alias the carry.

    :param program: Static periodic run program.
    :type program: Any
    :param kernels: Kernel indexed by the program's group codes.
    :type kernels: Tuple[Callable[..., Any], ...]
    :param carry: Initial scan carry.
    :type carry: Any
    :return: Final scan carry.
    :rtype: Any
    """
    for pattern, records in program:

        def body(value: Any, record: Array) -> Tuple[Any, None]:
            for offset, code in enumerate(pattern):
                value = kernels[code](value, record[offset])
            return value, None

        if len(records) == 1:
            carry = body(carry, jnp.asarray(records[0]))[0]
        else:
            carry = jax.lax.scan(body, carry, jnp.asarray(records))[0]
    return carry


def prepare_schedule(
    mps_spec: MPSSpec, mpo_spec: Optional[MPOSpec] = None, *, width: int = 2
) -> SweepSchedule:
    """
    Group sites by shapes and relative block connectivity before tracing.

    :param mps_spec: Static MPS block layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Optional static MPO block layout.
    :type mpo_spec: Optional[MPOSpec]
    :param width: Number of consecutive sites in each local kernel window.
    :type width: int
    :return: Static forward and reverse sweep schedule.
    :rtype: SweepSchedule
    """
    assert mps_spec.site_layouts is not None
    groups: dict[Any, list[int]] = {}
    for site in range(mps_spec.nsites - width + 1):
        key = tuple(
            _layout_key(layout) for layout in mps_spec.site_layouts[site : site + width]
        )
        if mpo_spec is not None:
            assert mpo_spec.site_layouts is not None
            key += tuple(
                _layout_key(layout)
                for layout in mpo_spec.site_layouts[site : site + width]
            )
            key += tuple(
                _legal_environment_coordinates(mps_spec, mpo_spec, cut)
                for cut in range(site, site + width + 1)
            )
        groups.setdefault(key, []).append(site)
    prepared = []
    indices = np.empty((mps_spec.nsites - width + 1, 2), dtype=np.int32)
    for code, sites_list in enumerate(groups.values()):
        sites = tuple(sites_list)
        if mpo_spec is not None:
            for offset in range(width):
                prepare_mpo_site(mps_spec, mpo_spec, sites[0] + offset)
        prepared.append(
            SiteGroup(
                sites[0],
                tuple(
                    site_access(mps_spec, tuple(site + offset for site in sites))
                    for offset in range(width)
                ),
                (
                    tuple(
                        site_access(mpo_spec, tuple(site + offset for site in sites))
                        for offset in range(width)
                    )
                    if mpo_spec is not None
                    else ()
                ),
                (
                    tuple(
                        environment_access(
                            mps_spec, mpo_spec, tuple(site + offset for site in sites)
                        )
                        for offset in range(width + 1)
                    )
                    if mpo_spec is not None
                    else ()
                ),
                sites,
            )
        )
        indices[np.asarray(sites), 0] = code
        indices[np.asarray(sites), 1] = np.arange(len(sites))
    return SweepSchedule(
        tuple(prepared),
        indices,
        (prepare_program(indices), prepare_program(indices[::-1])),
    )


def empty_environments(
    mps_spec: MPSSpec, mpo_spec: MPOSpec, dtype: Any, *, left: bool
) -> Buckets:
    """
    Allocate bucket tensors directly, writing only the physical boundary.

    :param mps_spec: Static MPS block layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO block layout.
    :type mpo_spec: MPOSpec
    :param dtype: Dtype for the environment buckets.
    :type dtype: Any
    :param left: Allocate and initialize the left boundary when true.
    :type left: bool
    :return: Packed environment buckets.
    :rtype: Buckets
    """
    plan = environment_bucket_plan(mps_spec, mpo_spec)
    buckets = tuple(
        jnp.zeros((count,) + shape, dtype=dtype)
        for shape, count in zip(plan.shapes, plan.counts)
    )
    cut = 0 if left else mps_spec.nsites
    boundary = (
        mpo_left_boundary(mps_spec, mpo_spec, dtype)
        if left
        else mpo_right_boundary(mps_spec, mpo_spec, dtype)
    )
    return environment_access(mps_spec, mpo_spec, (cut,)).write(
        buckets, 0, boundary.blocks
    )


def read_environment(
    group: SiteGroup,
    offset: int,
    row: Array,
    buckets: Buckets,
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
) -> LegalEnvironment:
    """
    Read one local environment from scheduled bucket storage.

    :param group: Site group containing the environment accessor.
    :type group: SiteGroup
    :param offset: Offset of the requested local cut in the group window.
    :type offset: int
    :param row: Row index into the static schedule.
    :type row: Array
    :param buckets: Packed environment buckets.
    :type buckets: Buckets
    :param mps_spec: Static MPS block layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO block layout.
    :type mpo_spec: MPOSpec
    :return: Legal environment for the requested cut.
    :rtype: LegalEnvironment
    """
    return LegalEnvironment(
        mps_spec,
        mpo_spec,
        group.site + offset,
        group.environments[offset].read(buckets, row),
    )


def build_right(
    schedule: SweepSchedule,
    mps_spec: MPSSpec,
    mpo_spec: MPOSpec,
    mps: Buckets,
    mpo: Buckets,
) -> Buckets:
    """
    Initialize right environments with the same shared spatial kernels.

    :param schedule: Prepared spatial sweep schedule.
    :type schedule: SweepSchedule
    :param mps_spec: Static MPS block layout.
    :type mps_spec: MPSSpec
    :param mpo_spec: Static MPO block layout.
    :type mpo_spec: MPOSpec
    :param mps: Packed MPS block buckets.
    :type mps: Buckets
    :param mpo: Packed MPO block buckets.
    :type mpo: Buckets
    :return: Packed right-environment buckets.
    :rtype: Buckets
    """
    initial = empty_environments(
        mps_spec, mpo_spec, jnp.result_type(mps[0], mpo[0]), left=False
    )

    def update(group: SiteGroup, row: Array, right: Buckets, context: Any) -> Buckets:
        del context
        value = mpo_update_right(
            mps_spec,
            mpo_spec,
            group.site,
            read_environment(group, 1, row, right, mps_spec, mpo_spec),
            group.mps[0].read(mps, row),
            group.mpo[0].read(mpo, row),
        )
        return group.environments[0].write(right, row, value.blocks)

    return schedule.run(update, initial, reverse=True)  # type: ignore[no-any-return]
