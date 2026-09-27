"""
Static layout data for fixed-shape dense MPS algorithms.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Optional, Tuple

from .symmetry import AbelianSymmetry, Charge, SectorIndex


def _u1_auto_bond_sectors(
    nsites: int, total_charge: int, chi: int, cut: int
) -> dict[int, int]:
    """Allocate an empirical Gaussian-like U(1) sector profile."""
    charge_min = max(0, total_charge - (nsites - cut))
    charge_max = min(cut, total_charge)
    center = cut * total_charge / nsites
    radius = max(1, (chi.bit_length() - 1) // 2)
    sigma = 0.95 + 0.075 * max(0.0, math.log2(chi) - 4.0)
    lower = max(charge_min, math.ceil(center - radius))
    upper = min(charge_max, math.floor(center + radius))
    charges = list(range(lower, upper + 1))
    capacities = [
        min(math.comb(cut, charge), math.comb(nsites - cut, total_charge - charge))
        for charge in charges
    ]
    target = min(chi, sum(capacities))
    if target == 0:
        return {}
    weights = [
        math.exp(-((charge - center) ** 2) / (2.0 * sigma**2)) for charge in charges
    ]
    desired = [target * weight / sum(weights) for weight in weights]
    allocation = [0] * len(charges)
    charge_positions = {charge: index for index, charge in enumerate(charges)}
    symmetric = abs(2.0 * center - round(2.0 * center)) < 1e-12
    if symmetric:
        groups: list[tuple[int, ...]] = []
        visited = set()
        for index, charge in enumerate(charges):
            if index in visited:
                continue
            mirror = int(round(2.0 * center - charge))
            mirror_index = charge_positions.get(mirror)
            if mirror_index is None or capacities[index] != capacities[mirror_index]:
                symmetric = False
                break
            if mirror_index == index:
                groups.append((index,))
            else:
                groups.append((index, mirror_index))
                visited.add(mirror_index)
            visited.add(index)
    if symmetric:
        for group in groups:
            if len(group) == 1:
                index = group[0]
                allocation[index] = min(
                    capacities[index], int(math.floor(desired[index]))
                )
            else:
                value = min(
                    capacities[group[0]],
                    int(math.floor(min(desired[index] for index in group))),
                )
                for index in group:
                    allocation[index] = value
    else:
        allocation = [
            min(capacity, int(math.floor(value)))
            for capacity, value in zip(capacities, desired)
        ]
    while sum(allocation) < target:
        candidates = [
            index
            for index, capacity in enumerate(capacities)
            if allocation[index] < capacity
        ]
        if symmetric and sum(allocation) + 1 < target:
            pair_candidates = [
                group
                for group in groups
                if len(group) == 2
                and allocation[group[0]] < capacities[group[0]]
                and allocation[group[1]] < capacities[group[1]]
            ]
            center_candidates = [
                group
                for group in groups
                if len(group) == 1 and allocation[group[0]] + 1 < capacities[group[0]]
            ]
            double_candidates = pair_candidates + center_candidates
            if double_candidates and sum(allocation) + 2 <= target:
                group = max(
                    double_candidates,
                    key=lambda item: (
                        (desired[item[0]] - allocation[item[0]])
                        / (2 if len(item) == 1 else 1),
                        -abs(charges[item[0]] - center),
                        -charges[item[0]],
                    ),
                )
                for index in group:
                    allocation[index] += 2 if len(group) == 1 else 1
                continue
        if symmetric and sum(allocation) + 1 == target:
            single_candidates = [
                group[0]
                for group in groups
                if len(group) == 1 and allocation[group[0]] < capacities[group[0]]
            ]
            if single_candidates:
                allocation[single_candidates[0]] += 1
                continue
        index = max(
            candidates,
            key=lambda item: (
                desired[item] - allocation[item],
                -abs(charges[item] - center),
                -charges[item],
            ),
        )
        allocation[index] += 1
    return {
        charge: degeneracy
        for charge, degeneracy in zip(charges, allocation)
        if degeneracy > 0
    }


@dataclass(frozen=True)
class BlockLayout:
    """
    Static allowed block coordinates and real tensor shapes for one site.

    :ivar indices: Directed sector indices for the tensor legs.
    :ivar total_charge: Total charge carried by the tensor.
    :ivar block_coordinates: Sector coordinates of all legal blocks.
    :ivar block_shapes: Dense shapes corresponding to ``block_coordinates``.
    """

    indices: Tuple[SectorIndex, ...]
    total_charge: Charge
    block_coordinates: Tuple[Tuple[int, ...], ...]
    block_shapes: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class BlockBucketPlan:
    """
    Exact-shape buckets for scan-friendly sparse block storage.

    :ivar shapes: Padded bucket shapes, indexed by bucket number.
    :ivar positions: Bucket and slot assigned to each logical block.
    :ivar counts: Number of blocks stored in each bucket.
    :ivar true_shapes: Unpadded shape of each logical block.
    """

    shapes: Tuple[Tuple[int, ...], ...]
    positions: Tuple[Tuple[int, int], ...]
    counts: Tuple[int, ...]
    true_shapes: Tuple[Tuple[int, ...], ...]


@lru_cache(maxsize=128)
def block_bucket_plan(layouts: Tuple[BlockLayout, ...]) -> BlockBucketPlan:
    """
    Group legal blocks by their true shapes without extra storage padding.

    :param layouts: Site layouts whose blocks should be bucketed.
    :type layouts: Tuple[BlockLayout, ...]
    :return: The cached exact-shape bucket plan.
    :rtype: BlockBucketPlan
    """
    true_shapes = tuple(shape for layout in layouts for shape in layout.block_shapes)
    shapes = tuple(sorted(set(true_shapes)))
    shape_positions = {shape: position for position, shape in enumerate(shapes)}
    counts = [0] * len(shapes)
    positions = []
    for shape in true_shapes:
        bucket = shape_positions[shape]
        positions.append((bucket, counts[bucket]))
        counts[bucket] += 1
    return BlockBucketPlan(shapes, tuple(positions), tuple(counts), true_shapes)


@dataclass(frozen=True)
class MPSSpec:
    """
    Static physical and virtual dimensions of an open-boundary MPS.

    :ivar physical_dims: Physical dimension at each site.
    :ivar bond_dims: Virtual bond dimension at every MPS cut.
    :ivar symmetry: Abelian symmetry, or ``None`` for a dense MPS.
    :ivar physical_sector_indices: Physical sector indices for a symmetric MPS.
    :ivar total_charge: Total charge carried by the MPS.
    :ivar bond_indices: Virtual sector indices at every MPS cut.
    :ivar bond_sectors: Charge quotas at every MPS cut.
    :ivar site_layouts: Legal block layouts for every site.
    """

    physical_dims: Tuple[int, ...]
    bond_dims: Tuple[int, ...]
    symmetry: Optional[AbelianSymmetry] = None
    physical_sector_indices: Optional[Tuple[SectorIndex, ...]] = None
    total_charge: Optional[Charge] = None
    bond_indices: Optional[Tuple[SectorIndex, ...]] = None
    bond_sectors: Optional[Tuple[Tuple[Tuple[Charge, int], ...], ...]] = None
    site_layouts: Optional[Tuple[BlockLayout, ...]] = None

    @classmethod
    def dense(
        cls,
        physical_dims: Tuple[int, ...],
        bond_dims: Optional[Tuple[int, ...]] = None,
        *,
        chi: Optional[int] = None,
        symmetry: None = None,
    ) -> "MPSSpec":
        """
        Construct a dense open-boundary MPS specification.

        Exactly one of ``bond_dims`` and ``chi`` must be supplied. A ``chi``
        value is expanded to the largest physically reachable dimensions.

        :param physical_dims: Physical dimension at each site.
        :type physical_dims: Tuple[int, ...]
        :param bond_dims: Explicit dimensions at every MPS cut.
        :type bond_dims: Optional[Tuple[int, ...]]
        :param chi: Maximum bond dimension used to derive ``bond_dims``.
        :type chi: Optional[int]
        :param symmetry: Must be ``None`` for this dense constructor.
        :type symmetry: None
        :return: The dense open-boundary MPS specification.
        :rtype: MPSSpec
        """
        if symmetry is not None:
            raise ValueError("MPSSpec.dense requires symmetry=None")
        if not physical_dims or any(d <= 0 for d in physical_dims):
            raise ValueError(
                "physical_dims must be a non-empty tuple of positive integers"
            )
        if (bond_dims is None) == (chi is None):
            raise ValueError("specify exactly one of bond_dims and chi")
        nsites = len(physical_dims)
        if chi is not None:
            if chi <= 0:
                raise ValueError("chi must be positive")
            left_dim = 1
            right_dims = [1] * (nsites + 1)
            for site in range(nsites - 1, -1, -1):
                right_dims[site] = right_dims[site + 1] * physical_dims[site]
            dimensions = [1]
            for site in range(nsites - 1):
                left_dim *= physical_dims[site]
                dimensions.append(min(chi, left_dim, right_dims[site + 1]))
            dimensions.append(1)
            bond_dims = tuple(dimensions)
        assert bond_dims is not None
        if len(bond_dims) != nsites + 1:
            raise ValueError("bond_dims must have len(physical_dims) + 1 entries")
        if bond_dims[0] != 1 or bond_dims[-1] != 1:
            raise ValueError("open-boundary bond_dims must begin and end with 1")
        if any(d <= 0 for d in bond_dims):
            raise ValueError("bond_dims must contain positive integers")
        for site, physical_dim in enumerate(physical_dims):
            left_dim = bond_dims[site]
            right_dim = bond_dims[site + 1]
            if right_dim > left_dim * physical_dim:
                raise ValueError(
                    f"bond {site + 1} has dimension {right_dim}, exceeding "
                    f"left capacity {left_dim * physical_dim}"
                )
            if left_dim > physical_dim * right_dim:
                raise ValueError(
                    f"bond {site} has dimension {left_dim}, exceeding "
                    f"right capacity {physical_dim * right_dim}"
                )
        return cls(tuple(physical_dims), tuple(bond_dims))

    @classmethod
    def u1(
        cls,
        nsites: int,
        total_charge: int,
        chi: int,
        *,
        charge_sectors: Optional[int] = None,
        bond_sectors: Optional[Tuple[Mapping[int, int], ...]] = None,
    ) -> "MPSSpec":
        """
        Construct a spin-1/2 U(1)-symmetric MPS specification.

        The local ``|0>`` and ``|1>`` states carry charges zero and one. Bond
        quotas use an empirical center-enhanced profile when
        ``charge_sectors`` is omitted. An explicit ``charge_sectors`` value
        selects the legacy uniform quota over a centered charge window and can
        be used to check convergence with respect to the retained sectors.
        ``bond_sectors`` provides full control over the charge-to-dimension
        map at every bond.

        :param nsites: Number of physical sites.
        :type nsites: int
        :param total_charge: Target total U(1) charge of the MPS.
        :type total_charge: int
        :param chi: Maximum total virtual dimension at a cut.
        :type chi: int
        :param charge_sectors: Maximum number of retained charge sectors per
            interior cut, or ``None`` for an automatic choice.
        :type charge_sectors: Optional[int]
        :param bond_sectors: Explicit U(1) charge-to-dimension maps at every
            bond, using integer charge keys. Mutually exclusive with
            ``charge_sectors``.
        :type bond_sectors: Optional[Tuple[Mapping[int, int], ...]]
        :return: The U(1)-symmetric MPS specification.
        :rtype: MPSSpec
        """
        if nsites <= 0:
            raise ValueError("nsites must be positive")
        if total_charge < 0 or total_charge > nsites:
            raise ValueError("total_charge must be between zero and nsites")
        if chi <= 0:
            raise ValueError("chi must be positive")
        if bond_sectors is not None and charge_sectors is not None:
            raise ValueError("bond_sectors and charge_sectors are mutually exclusive")
        if bond_sectors is not None and len(bond_sectors) != nsites + 1:
            raise ValueError("bond_sectors must contain one entry per MPS cut")
        automatic_profile = bond_sectors is None and charge_sectors is None
        if bond_sectors is None:
            if automatic_profile:
                charge_sectors = min(chi, max(2, 1 << ((chi.bit_length() - 1) // 2)))
            assert charge_sectors is not None
            if charge_sectors <= 0 or charge_sectors > chi:
                raise ValueError("charge_sectors must be between one and chi")

        symmetry = AbelianSymmetry((0,))
        physical = SectorIndex.from_basis(
            symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
        )
        if bond_sectors is not None:
            if any(sum(sectors.values()) > chi for sectors in bond_sectors):
                raise ValueError("bond_sectors exceed chi")
            return cls.from_sectors(
                physical_indices=(physical,) * nsites,
                total_charge=(total_charge,),
                bond_sectors=tuple(
                    {(charge,): dimension for charge, dimension in sectors.items()}
                    for sectors in bond_sectors
                ),
            )
        assert charge_sectors is not None
        quota = chi // charge_sectors
        bonds: list[dict[int, int]] = []
        for cut in range(nsites + 1):
            if automatic_profile:
                bonds.append(_u1_auto_bond_sectors(nsites, total_charge, chi, cut))
                continue
            center = cut * total_charge // nsites
            lower = max(
                0,
                total_charge - (nsites - cut),
                center - (charge_sectors - 1) // 2,
            )
            upper = min(cut, total_charge, center + charge_sectors // 2)
            bonds.append(
                {
                    charge: min(
                        quota,
                        math.comb(cut, charge),
                        math.comb(nsites - cut, total_charge - charge),
                    )
                    for charge in range(lower, upper + 1)
                }
            )

        changed = True
        while changed:
            changed = False
            for cut in range(1, nsites):
                for charge, dimension in tuple(bonds[cut].items()):
                    capacity = min(
                        bonds[cut - 1].get(charge, 0)
                        + bonds[cut - 1].get(charge - 1, 0),
                        bonds[cut + 1].get(charge, 0)
                        + bonds[cut + 1].get(charge + 1, 0),
                    )
                    if dimension > capacity:
                        changed = True
                        if capacity:
                            bonds[cut][charge] = capacity
                        else:
                            del bonds[cut][charge]

        return cls.from_sectors(
            physical_indices=(physical,) * nsites,
            total_charge=(total_charge,),
            bond_sectors=tuple(
                {(charge,): dimension for charge, dimension in bond.items()}
                for bond in bonds
            ),
        )

    @classmethod
    def z2(
        cls,
        nsites: int,
        chi: int,
        *,
        parity: int = 0,
    ) -> "MPSSpec":
        """
        Construct a spin-1/2 Z2-symmetric MPS specification.

        The local ``|0>`` and ``|1>`` states carry parities zero and one. The
        total bond-dimension budget is split as evenly as possible between the
        two parity sectors and capped by the exact reachable dimensions.

        :param nsites: Number of physical sites.
        :type nsites: int
        :param chi: Maximum total virtual dimension at an interior cut.
        :type chi: int
        :param parity: Target total parity, either zero or one.
        :type parity: int
        :return: The Z2-symmetric MPS specification.
        :rtype: MPSSpec
        """
        if nsites <= 0:
            raise ValueError("nsites must be positive")
        if chi <= 0:
            raise ValueError("chi must be positive")
        if nsites > 1 and chi < 2:
            raise ValueError("chi must be at least two for a multi-site Z2 layout")
        if parity not in (0, 1):
            raise ValueError("parity must be zero or one")

        symmetry = AbelianSymmetry((2,))
        physical = SectorIndex.from_basis(
            symmetry=symmetry, basis_charges=((0,), (1,)), flow=1
        )
        bonds: list[dict[Charge, int]] = []
        for cut in range(nsites + 1):
            if cut == 0:
                bonds.append({(0,): 1})
            elif cut == nsites:
                bonds.append({(parity,): 1})
            else:
                capacity = min(2 ** (cut - 1), 2 ** (nsites - cut - 1))
                bonds.append(
                    {
                        (0,): min((chi + 1) // 2, capacity),
                        (1,): min(chi // 2, capacity),
                    }
                )
        return cls.from_sectors(
            physical_indices=(physical,) * nsites,
            total_charge=(parity,),
            bond_sectors=tuple(bonds),
        )

    @classmethod
    def from_sectors(
        cls,
        *,
        physical_indices: Tuple[SectorIndex, ...],
        total_charge: Iterable[int],
        bond_sectors: Tuple[Mapping[Charge, int], ...],
    ) -> "MPSSpec":
        """
        Create a fixed Abelian MPS layout from explicit bond quotas.

        :param physical_indices: Physical sector index at every site.
        :type physical_indices: Tuple[SectorIndex, ...]
        :param total_charge: Target total charge of the MPS.
        :type total_charge: Iterable[int]
        :param bond_sectors: Charge-to-degeneracy map at every MPS cut.
        :type bond_sectors: Tuple[Mapping[Charge, int], ...]
        :return: The validated fixed-layout symmetric MPS specification.
        :rtype: MPSSpec
        """
        if not physical_indices:
            raise ValueError("physical_indices must be non-empty")
        symmetry = physical_indices[0].symmetry
        if any(
            index.symmetry != symmetry or index.flow != 1 for index in physical_indices
        ):
            raise ValueError("physical indices must share symmetry and have flow +1")
        total_charge = symmetry.normalize(total_charge)
        if len(bond_sectors) != len(physical_indices) + 1:
            raise ValueError("bond_sectors must contain one entry per MPS cut")
        canonical_sectors = []
        bond_indices = []
        for bond, sectors in enumerate(bond_sectors):
            normalized = tuple(
                sorted(
                    (symmetry.normalize(charge), int(degeneracy))
                    for charge, degeneracy in sectors.items()
                )
            )
            if not normalized:
                raise ValueError(f"bond {bond} must contain at least one sector")
            if len({charge for charge, _ in normalized}) != len(normalized):
                raise ValueError(f"bond {bond} contains duplicate charge sectors")
            if any(degeneracy <= 0 for _, degeneracy in normalized):
                raise ValueError(f"bond {bond} contains a non-positive sector quota")
            canonical_sectors.append(normalized)
            bond_indices.append(
                SectorIndex.from_sectors(
                    symmetry=symmetry,
                    sectors=normalized,
                    flow=-1,
                    name=f"bond_{bond}",
                )
            )
        left_boundary = canonical_sectors[0]
        right_boundary = canonical_sectors[-1]
        if left_boundary != (((0,) * symmetry.rank, 1),):
            raise ValueError("left boundary must be the unique zero-charge sector")
        if right_boundary != ((total_charge, 1),):
            raise ValueError("right boundary must be the unique target-charge sector")
        for site, physical in enumerate(physical_indices):
            left = bond_indices[site]
            right = bond_indices[site + 1]
            for right_charge, right_degeneracy in zip(
                right.charges, right.degeneracies
            ):
                capacity = sum(
                    left.degeneracy(left_charge) * physical.degeneracy(physical_charge)
                    for left_charge in left.charges
                    for physical_charge in physical.charges
                    if symmetry.add(left_charge, physical_charge) == right_charge
                )
                if right_degeneracy > capacity:
                    raise ValueError(
                        f"bond {site + 1} sector {right_charge} exceeds left capacity"
                    )
            for left_charge, left_degeneracy in zip(left.charges, left.degeneracies):
                capacity = sum(
                    physical.degeneracy(physical_charge)
                    * right.degeneracy(right_charge)
                    for physical_charge in physical.charges
                    for right_charge in right.charges
                    if symmetry.add(left_charge, physical_charge) == right_charge
                )
                if left_degeneracy > capacity:
                    raise ValueError(
                        f"bond {site} sector {left_charge} exceeds right capacity"
                    )
        layouts = []
        for site, physical in enumerate(physical_indices):
            left = bond_indices[site]
            right = bond_indices[site + 1]
            coordinates = []
            shapes = []
            for left_position, left_charge in enumerate(left.charges):
                for physical_position, physical_charge in enumerate(physical.charges):
                    target_charge = symmetry.add(left_charge, physical_charge)
                    if target_charge not in right.charges:
                        continue
                    right_position = right.charges.index(target_charge)
                    coordinates.append(
                        (left_position, physical_position, right_position)
                    )
                    shapes.append(
                        (
                            left.degeneracies[left_position],
                            physical.degeneracies[physical_position],
                            right.degeneracies[right_position],
                        )
                    )
            if not coordinates:
                raise ValueError(f"site {site} has no allowed charge blocks")
            layouts.append(
                BlockLayout(
                    indices=(left, physical, right),
                    total_charge=(0,) * symmetry.rank,
                    block_coordinates=tuple(coordinates),
                    block_shapes=tuple(shapes),
                )
            )
        return cls(
            physical_dims=tuple(index.dimension for index in physical_indices),
            bond_dims=tuple(index.dimension for index in bond_indices),
            symmetry=symmetry,
            physical_sector_indices=physical_indices,
            total_charge=total_charge,
            bond_indices=tuple(bond_indices),
            bond_sectors=tuple(canonical_sectors),
            site_layouts=tuple(layouts),
        )

    @property
    def nsites(self) -> int:
        """
        Number of physical sites.

        :return: The number of physical sites.
        :rtype: int
        """
        return len(self.physical_dims)

    @property
    def physical_indices(self) -> Tuple[int, ...]:
        """
        Dense physical dimensions in the common index-accepting API form.

        :return: Physical dimensions or sector indices for every site.
        :rtype: Tuple[int, ...]
        """
        if self.physical_sector_indices is not None:
            return self.physical_sector_indices  # type: ignore[return-value]
        return self.physical_dims


@dataclass(frozen=True)
class GateSpec:
    """
    Static support and dimensions of a compiled nearest-neighbor TEBD model.

    :ivar physical_dims: Physical dimension at each site.
    """

    physical_dims: Tuple[int, ...]


@dataclass(frozen=True)
class MPOSpec:
    """
    Static physical and virtual dimensions of an open-boundary MPO.

    :ivar physical_dims: Physical dimension at each site.
    :ivar bond_dims: Virtual bond dimension at every MPO cut.
    :ivar symmetry: Abelian symmetry, or ``None`` for a dense MPO.
    :ivar physical_sector_indices: Physical sector indices for a symmetric MPO.
    :ivar bond_indices: Virtual sector indices at every MPO cut.
    :ivar site_layouts: Legal block layouts for every site.
    """

    physical_dims: Tuple[int, ...]
    bond_dims: Tuple[int, ...]
    symmetry: Optional[AbelianSymmetry] = None
    physical_sector_indices: Optional[Tuple[SectorIndex, ...]] = None
    bond_indices: Optional[Tuple[SectorIndex, ...]] = None
    site_layouts: Optional[Tuple[BlockLayout, ...]] = None


def validate_mps_mpo_specs(mps: MPSSpec, mpo: MPOSpec) -> None:
    """
    Reject incompatible physical dimensions, symmetries, or basis orders.

    :param mps: MPS specification to validate.
    :type mps: MPSSpec
    :param mpo: MPO specification to validate.
    :type mpo: MPOSpec
    :raises ValueError: If the physical layout or symmetry metadata differ.
    """
    if mps.physical_dims != mpo.physical_dims:
        raise ValueError("MPS and MPO physical dimensions must match")
    if mps.symmetry != mpo.symmetry:
        raise ValueError("MPS and MPO symmetries must match")
    if mps.physical_sector_indices != mpo.physical_sector_indices:
        raise ValueError(
            "MPS and MPO physical sector indices and basis order must match"
        )


@lru_cache(maxsize=128)
def block_mps_spec(spec: MPSSpec) -> MPSSpec:
    """
    Represent dense sites as one block of the trivial group for shared sweeps.

    :param spec: Dense or symmetric MPS specification.
    :type spec: MPSSpec
    :return: The original symmetric specification or its trivial-group view.
    :rtype: MPSSpec
    """
    if spec.symmetry is not None:
        return spec
    symmetry = AbelianSymmetry(())
    physical = tuple(
        SectorIndex.from_sectors(symmetry=symmetry, sectors=(((), d),), flow=1)
        for d in spec.physical_dims
    )
    return MPSSpec.from_sectors(
        physical_indices=physical,
        total_charge=(),
        bond_sectors=tuple({(): d} for d in spec.bond_dims),
    )


@lru_cache(maxsize=128)
def block_mpo_spec(spec: MPOSpec) -> MPOSpec:
    """
    Represent a dense MPO with one legal block per site, without truncation.

    :param spec: Dense or symmetric MPO specification.
    :type spec: MPOSpec
    :return: The original symmetric specification or its trivial-group view.
    :rtype: MPOSpec
    """
    if spec.symmetry is not None:
        return spec
    symmetry = AbelianSymmetry(())
    physical = tuple(
        SectorIndex.from_sectors(symmetry=symmetry, sectors=(((), d),), flow=1)
        for d in spec.physical_dims
    )
    bonds = tuple(
        SectorIndex.from_sectors(symmetry=symmetry, sectors=(((), d),), flow=1)
        for d in spec.bond_dims
    )
    layouts = tuple(
        BlockLayout(
            (bonds[site], index, index.dual(), bonds[site + 1].dual()),
            (),
            ((0, 0, 0, 0),),
            (
                (
                    spec.bond_dims[site],
                    index.dimension,
                    index.dimension,
                    spec.bond_dims[site + 1],
                ),
            ),
        )
        for site, index in enumerate(physical)
    )
    return MPOSpec(
        spec.physical_dims, spec.bond_dims, symmetry, physical, bonds, layouts
    )
