"""
Static Abelian charge and canonical sector-index primitives.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

Charge = Tuple[int, ...]


@dataclass(frozen=True)
class AbelianSymmetry:
    """
    A product of U(1) and finite cyclic Abelian charge groups.

    :ivar moduli: Modulus for each charge component; zero denotes U(1).
    """

    moduli: Tuple[int, ...]

    def __post_init__(self) -> None:
        if any(modulus == 1 or modulus < 0 for modulus in self.moduli):
            raise ValueError("symmetry moduli must be 0 or integers at least 2")

    @property
    def rank(self) -> int:
        """
        Number of commuting charge components.

        :return: The number of charge components.
        :rtype: int
        """
        return len(self.moduli)

    def normalize(self, charge: Iterable[int]) -> Charge:
        """
        Normalize a charge according to the finite cyclic components.

        :param charge: Charge values to normalize.
        :type charge: Iterable[int]
        :return: The normalized charge tuple.
        :rtype: Charge
        """
        charge = tuple(int(value) for value in charge)
        if len(charge) != self.rank:
            raise ValueError("charge rank does not match the symmetry")
        return tuple(
            value if modulus == 0 else value % modulus
            for value, modulus in zip(charge, self.moduli)
        )

    def add(self, left: Charge, right: Charge) -> Charge:
        """
        Add two charges.

        :param left: First charge.
        :type left: Charge
        :param right: Second charge.
        :type right: Charge
        :return: The normalized sum.
        :rtype: Charge
        """
        return self.normalize(
            tuple(
                left_value + right_value for left_value, right_value in zip(left, right)
            )
        )

    def negate(self, charge: Charge) -> Charge:
        """
        Return the group inverse of a charge.

        :param charge: Charge to invert.
        :type charge: Charge
        :return: The normalized inverse charge.
        :rtype: Charge
        """
        return self.normalize(tuple(-value for value in charge))

    def subtract(self, left: Charge, right: Charge) -> Charge:
        """
        Subtract two charges.

        :param left: Charge from which to subtract.
        :type left: Charge
        :param right: Charge to subtract.
        :type right: Charge
        :return: The normalized difference.
        :rtype: Charge
        """
        return self.add(left, self.negate(right))


@dataclass(frozen=True)
class SectorIndex:
    """
    A directed canonical basis partitioned into Abelian charge sectors.

    :ivar symmetry: Abelian charge arithmetic used by the index.
    :ivar charges: Canonically sorted charge label for each sector.
    :ivar degeneracies: Basis degeneracy of each sector.
    :ivar flow: Index direction, either ``1`` or ``-1``.
    :ivar basis_to_canonical: User-basis to canonical-basis permutation.
    :ivar canonical_to_basis: Canonical-basis to user-basis permutation.
    :ivar name: Optional non-semantic label for the index.
    """

    symmetry: AbelianSymmetry
    charges: Tuple[Charge, ...]
    degeneracies: Tuple[int, ...]
    flow: int = 1
    basis_to_canonical: Tuple[int, ...] = ()
    canonical_to_basis: Tuple[int, ...] = ()
    name: Optional[str] = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.flow not in (-1, 1):
            raise ValueError("SectorIndex flow must be +1 or -1")
        if len(self.charges) != len(self.degeneracies) or not self.charges:
            raise ValueError(
                "charges and positive degeneracies must be non-empty and aligned"
            )
        canonical_charges = tuple(
            self.symmetry.normalize(charge) for charge in self.charges
        )
        if (
            canonical_charges != self.charges
            or tuple(sorted(self.charges)) != self.charges
        ):
            raise ValueError(
                "SectorIndex charges must be normalized and canonically sorted"
            )
        if any(degeneracy <= 0 for degeneracy in self.degeneracies):
            raise ValueError("sector degeneracies must be positive")
        if len(set(self.charges)) != len(self.charges):
            raise ValueError("SectorIndex charges must be unique")
        if self.basis_to_canonical and len(self.basis_to_canonical) != self.dimension:
            raise ValueError("basis_to_canonical has the wrong length")
        if self.canonical_to_basis and len(self.canonical_to_basis) != self.dimension:
            raise ValueError("canonical_to_basis has the wrong length")

    @classmethod
    def from_basis(
        cls,
        *,
        symmetry: AbelianSymmetry,
        basis_charges: Iterable[Iterable[int]],
        flow: int = 1,
        name: Optional[str] = None,
    ) -> "SectorIndex":
        """
        Group a user basis by charge while preserving reversible ordering maps.

        :param symmetry: Abelian symmetry carried by the basis.
        :type symmetry: AbelianSymmetry
        :param basis_charges: Charge assigned to each user-basis vector.
        :type basis_charges: Iterable[Iterable[int]]
        :param flow: Index direction, either ``1`` or ``-1``.
        :type flow: int
        :param name: Optional non-semantic label.
        :type name: Optional[str]
        :return: The canonical sector index and basis permutations.
        :rtype: SectorIndex
        """
        normalized = tuple(symmetry.normalize(charge) for charge in basis_charges)
        if not normalized:
            raise ValueError("basis_charges must be non-empty")
        charges = tuple(sorted(set(normalized)))
        positions: dict[Charge, list[int]] = {charge: [] for charge in charges}
        for position, charge in enumerate(normalized):
            positions[charge].append(position)
        canonical_to_basis = tuple(
            position for charge in charges for position in positions[charge]
        )
        basis_to_canonical_list = [0] * len(normalized)
        for canonical_position, basis_position in enumerate(canonical_to_basis):
            basis_to_canonical_list[basis_position] = canonical_position
        return cls(
            symmetry=symmetry,
            charges=charges,
            degeneracies=tuple(len(positions[charge]) for charge in charges),
            flow=flow,
            basis_to_canonical=tuple(basis_to_canonical_list),
            canonical_to_basis=canonical_to_basis,
            name=name,
        )

    @classmethod
    def from_sectors(
        cls,
        *,
        symmetry: AbelianSymmetry,
        sectors: Iterable[Tuple[Iterable[int], int]],
        flow: int = 1,
        name: Optional[str] = None,
    ) -> "SectorIndex":
        """
        Construct a canonical index from explicit charge/degeneracy pairs.

        :param symmetry: Abelian symmetry carried by the index.
        :type symmetry: AbelianSymmetry
        :param sectors: Charge and positive degeneracy pairs.
        :type sectors: Iterable[Tuple[Iterable[int], int]]
        :param flow: Index direction, either ``1`` or ``-1``.
        :type flow: int
        :param name: Optional non-semantic label.
        :type name: Optional[str]
        :return: The canonical sector index.
        :rtype: SectorIndex
        """
        normalized = tuple(
            sorted(
                (symmetry.normalize(charge), int(degeneracy))
                for charge, degeneracy in sectors
            )
        )
        return cls(
            symmetry=symmetry,
            charges=tuple(charge for charge, _ in normalized),
            degeneracies=tuple(degeneracy for _, degeneracy in normalized),
            flow=flow,
            name=name,
        )

    @property
    def dimension(self) -> int:
        """
        Total vector-space dimension represented by this index.

        :return: The sum of all sector degeneracies.
        :rtype: int
        """
        return sum(self.degeneracies)

    @property
    def offsets(self) -> Tuple[int, ...]:
        """
        Starting canonical-basis offset of every charge sector.

        :return: The starting offset of each sector.
        :rtype: Tuple[int, ...]
        """
        total = 0
        offsets = []
        for degeneracy in self.degeneracies:
            offsets.append(total)
            total += degeneracy
        return tuple(offsets)

    def degeneracy(self, charge: Charge) -> int:
        """
        Return the degeneracy of one sector, or zero if it is absent.

        :param charge: Sector charge to query.
        :type charge: Charge
        :return: The sector degeneracy, or zero when absent.
        :rtype: int
        """
        charge = self.symmetry.normalize(charge)
        try:
            return self.degeneracies[self.charges.index(charge)]
        except ValueError:
            return 0

    def dual(self) -> "SectorIndex":
        """
        Reverse only the index direction without relabeling charges.

        :return: A sector index with the opposite flow.
        :rtype: SectorIndex
        """
        return SectorIndex(
            self.symmetry,
            self.charges,
            self.degeneracies,
            -self.flow,
            self.basis_to_canonical,
            self.canonical_to_basis,
            self.name,
        )
