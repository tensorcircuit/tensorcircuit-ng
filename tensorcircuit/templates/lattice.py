# -*- coding: utf-8 -*-
"""
The lattice module for defining and manipulating lattice geometries.
"""
import abc
from typing import (
    Any,
    Dict,
    Hashable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import matplotlib.pyplot as plt

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.axes

# --- Type Aliases for Readability ---
SiteIndex = int
SiteIdentifier = Hashable
Coordinates = np.ndarray[Any, Any]
NeighborMap = Dict[SiteIndex, List[SiteIndex]]


class AbstractLattice(abc.ABC):
    """Abstract base class for describing lattice systems.

    This class defines the common interface for all lattice structures,
    providing access to fundamental properties like site information
    (count, coordinates, identifiers) and neighbor relationships.
    Subclasses are responsible for implementing the specific logic for
    generating the lattice points and calculating neighbor connections.

    Args:
        dimensionality (int): The spatial dimension of the lattice (e.g., 1, 2, 3).
    """

    def __init__(self, dimensionality: int):
        """Initializes the base lattice class."""
        self._dimensionality = dimensionality

        # --- Internal Data Structures (to be populated by subclasses) ---
        self._indices: List[SiteIndex] = []
        self._identifiers: List[SiteIdentifier] = []
        self._coordinates: List[Coordinates] = []
        self._ident_to_idx: Dict[SiteIdentifier, SiteIndex] = {}
        self._neighbor_maps: Dict[int, NeighborMap] = {}

    @property
    def num_sites(self) -> int:
        """Returns the total number of sites (N) in the lattice."""
        return len(self._indices)

    @property
    def dimensionality(self) -> int:
        """Returns the spatial dimension of the lattice."""
        return self._dimensionality

    def __len__(self) -> int:
        """Returns the total number of sites, enabling `len(lattice)`."""
        return self.num_sites

    # --- Public API for Accessing Lattice Information ---

    def get_coordinates(self, index: SiteIndex) -> Coordinates:
        """Gets the spatial coordinates of a site by its integer index.

        Args:
            index (SiteIndex): The integer index of the site.

        Returns:
            Coordinates: The spatial coordinates as a NumPy array.

        Raises:
            IndexError: If the site index is out of range.
        """
        if 0 <= index < self.num_sites:
            return self._coordinates[index]
        raise IndexError(f"Site index {index} out of range (0-{self.num_sites-1})")

    def get_identifier(self, index: SiteIndex) -> SiteIdentifier:
        """Gets the abstract identifier of a site by its integer index.

        Args:
            index (SiteIndex): The integer index of the site.

        Returns:
            SiteIdentifier: The unique, hashable identifier of the site.

        Raises:
            IndexError: If the site index is out of range.
        """
        if 0 <= index < self.num_sites:
            return self._identifiers[index]
        raise IndexError(f"Site index {index} out of range (0-{self.num_sites-1})")

    def get_index(self, identifier: SiteIdentifier) -> SiteIndex:
        """Gets the integer index of a site by its unique identifier.

        Args:
            identifier (SiteIdentifier): The unique identifier of the site.

        Returns:
            SiteIndex: The corresponding integer index of the site.

        Raises:
            ValueError: If the identifier is not found in the lattice.
        """
        try:
            return self._ident_to_idx[identifier]
        except KeyError:
            # Re-raising as a ValueError can be more user-friendly,
            # as it's about the value of the identifier, not just a key lookup.
            raise ValueError(f"Identifier {identifier} not found in the lattice.")

    def get_site_info(
        self, index_or_identifier: Union[SiteIndex, SiteIdentifier]
    ) -> Tuple[SiteIndex, SiteIdentifier, Coordinates]:
        """Gets all information for a single site.

        This method provides a convenient way to retrieve all relevant data for a
        site (its index, identifier, and coordinates) by using either its
        integer index or its unique identifier.

        Args:
            index_or_identifier (Union[SiteIndex, SiteIdentifier]): The integer
                index or the unique identifier of the site to look up.

        Returns:
            Tuple[SiteIndex, SiteIdentifier, Coordinates]: A tuple containing:
                - The site's integer index.
                - The site's unique identifier.
                - The site's coordinates as a NumPy array.

        Raises:
            IndexError: If the given index is out of bounds.
            ValueError: If the given identifier is not found in the lattice.
        """
        if isinstance(index_or_identifier, int):  # SiteIndex is an int
            idx = index_or_identifier
            if not (0 <= idx < self.num_sites):
                raise IndexError(
                    f"Site index {idx} out of range (0-{self.num_sites - 1})"
                )
            return idx, self._identifiers[idx], self._coordinates[idx]
        else:  # Identifier
            ident = index_or_identifier
            # get_index() already raises a descriptive ValueError, so we let it.
            idx = self.get_index(ident)
            return idx, ident, self._coordinates[idx]

    def sites(self) -> Iterator[Tuple[SiteIndex, SiteIdentifier, Coordinates]]:
        """Returns an iterator over all sites in the lattice.

        This provides a convenient way to loop through all sites, for example:
        `for idx, ident, coords in my_lattice.sites(): ...`

        Yields:
            Iterator[Tuple[SiteIndex, SiteIdentifier, Coordinates]]: An iterator
                where each item is a tuple containing the site's index,
                identifier, and coordinates.
        """
        for i in range(self.num_sites):
            yield i, self._identifiers[i], self._coordinates[i]

    def get_neighbors(self, index: SiteIndex, k: int = 1) -> List[SiteIndex]:
        """Gets the list of k-th nearest neighbor indices for a given site.

        Args:
            index (SiteIndex): The integer index of the center site.
            k (int, optional): The order of the neighbors, where k=1 corresponds
                to nearest neighbors (NN), k=2 to next-nearest neighbors (NNN),
                and so on. Defaults to 1.

        Returns:
            List[SiteIndex]: A list of integer indices for the neighboring sites.
            Returns an empty list if neighbors for the given `k` have not been
            pre-calculated or if the site has no such neighbors.
        """
        if k not in self._neighbor_maps:
            print(
                f"Warning: {k}-th nearest neighbors not pre-calculated. "
                "Returning empty list."
            )
            return []
        return self._neighbor_maps[k].get(index, [])

    def get_neighbor_pairs(
        self, k: int = 1, unique: bool = True
    ) -> List[Tuple[SiteIndex, SiteIndex]]:
        """Gets all pairs of k-th nearest neighbors, representing bonds.

        Args:
            k (int, optional): The order of the neighbors to consider.
                Defaults to 1.
            unique (bool, optional): If True, returns only one representation
                for each pair (i, j) such that i < j, avoiding duplicates
                like (j, i). If False, returns all directed pairs.
                Defaults to True.

        Returns:
            List[Tuple[SiteIndex, SiteIndex]]: A list of tuples, where each
            tuple is a pair of neighbor indices.
        """
        pairs = []
        if k not in self._neighbor_maps:
            print(
                f"Warning: {k}-th nearest neighbors not pre-calculated. "
                "Returning empty list."
            )
            return []

        for i, neighbors in self._neighbor_maps[k].items():
            for j in neighbors:
                if unique:
                    if i < j:
                        pairs.append((i, j))
                else:
                    pairs.append((i, j))
        return sorted(pairs)

    # Sorting provides a deterministic output order
    # --- Abstract Methods for Subclass Implementation ---

    @abc.abstractmethod
    def _build_lattice(self, *args: Any, **kwargs: Any) -> None:
        """
        Abstract method for subclasses to generate the lattice data.

        A concrete implementation of this method in a subclass is responsible
        for populating the following internal attributes:
        - self._indices
        - self._identifiers
        - self._coordinates
        - self._ident_to_idx
        """
        pass

    @abc.abstractmethod
    def _build_neighbors(self, max_k: int = 1) -> None:
        """
        Abstract method for subclasses to calculate neighbor relationships.

        A concrete implementation of this method should calculate the neighbor
        relationships up to `max_k` and populate the `self._neighbor_maps`
        dictionary. The keys of the dictionary should be the neighbor order (k),
        and the values should be a dictionary mapping site indices to their
        list of k-th neighbors.
        """
        pass

    # --- Concrete, Inheritable Methods ---

    def show(
        self,
        show_indices: bool = False,
        show_identifiers: bool = False,
        show_bonds_k: Optional[int] = 1,
        ax: Optional["matplotlib.axes.Axes"] = None,
        **kwargs: Any,
    ) -> None:
        """Visualizes the lattice structure using Matplotlib.

        This method supports 1D, 2D, and 3D plotting. For 1D lattices, sites
        are plotted along the x-axis.

        Args:
            show_indices (bool, optional): If True, displays the integer index
                next to each site. Defaults to False.
            show_identifiers (bool, optional): If True, displays the unique
                identifier next to each site. Defaults to False.
            show_bonds_k (Optional[int], optional): Specifies which order of
                neighbor bonds to draw (e.g., 1 for NN, 2 for NNN). If None,
                no bonds are drawn. If the specified neighbors have not been
                calculated, a warning is printed. Defaults to 1.
            ax (Optional["matplotlib.axes.Axes"], optional): An existing
                Matplotlib Axes object to plot on. If None, a new Figure and
                Axes are created automatically. Defaults to None.
            **kwargs: Additional keyword arguments to be passed directly to the
                `matplotlib.pyplot.scatter` function for customizing site appearance.
        """
        # creat "fig_created_internally" as flag
        fig_created_internally = False

        if self.num_sites == 0:
            print("Lattice is empty, nothing to show.")
            return
        if self.dimensionality not in [1, 2, 3]:
            print(
                f"Warning: show() is not implemented for {self.dimensionality}D lattices."
            )
            return

        if ax is None:
            # when ax is none, make fig_created_internally true
            fig_created_internally = True
            if self.dimensionality == 3:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure  # type: ignore

        coords = np.array(self._coordinates)
        scatter_args = {"s": 100, "zorder": 2}
        scatter_args.update(kwargs)
        if self.dimensionality == 1:
            ax.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), **scatter_args)  # type: ignore
        elif self.dimensionality == 2:
            ax.scatter(coords[:, 0], coords[:, 1], **scatter_args)  # type: ignore
        elif self.dimensionality > 2:  # Safely handle 3D and future higher dimensions
            scatter_args["s"] = scatter_args.get("s", 100) // 2
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **scatter_args)  # type: ignore

        if show_indices or show_identifiers:
            for i in range(self.num_sites):
                label = str(self._identifiers[i]) if show_identifiers else str(i)
                offset = (
                    0.02 * np.max(np.ptp(coords, axis=0)) if coords.size > 0 else 0.1
                )
                if self.dimensionality == 1:
                    ax.text(coords[i, 0], offset, label, fontsize=9, ha="center")   # type: ignore[arg-type]
                    ax.text(
                        coords[i, 0] + offset,
                        coords[i, 1] + offset,
                        label,
                        fontsize=9,
                        zorder=3,
                    )                # type: ignore[arg-type]

                elif self.dimensionality > 2:
                    ax.text(
                        coords[i, 0],
                        coords[i, 1],
                        coords[i, 2] + offset,
                        label,
                        fontsize=9,
                        zorder=3,
                    )                    # type: ignore[arg-type]
        if show_bonds_k is not None and show_bonds_k in self._neighbor_maps:
            try:
                bonds = self.get_neighbor_pairs(k=show_bonds_k, unique=True)
                for i, j in bonds:
                    p1, p2 = self._coordinates[i], self._coordinates[j]
                    if self.dimensionality == 1:
                        ax.plot([p1[0], p2[0]], [0, 0], "k-", alpha=0.6, zorder=1)
                    elif self.dimensionality == 2:
                        ax.plot(
                            [p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.6, zorder=1
                        )
                    elif self.dimensionality > 2:
                        ax.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            [p1[2], p2[2]],
                            "k-",
                            alpha=0.6,
                            zorder=1,
                        )
            except ValueError as e:
                print(f"Could not draw bonds: {e}")
        elif show_bonds_k is not None:
            print(
                f"Warning: Cannot draw bonds. k={show_bonds_k} neighbors have not been calculated."
            )

        ax.set_title(f"{self.__class__.__name__} ({self.num_sites} sites)")
        if self.dimensionality == 2:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        if self.dimensionality > 1:
            ax.set_ylabel("y")
        if self.dimensionality > 2 and hasattr(ax, "set_zlabel"):
            ax.set_zlabel("z")
        ax.grid(True)

        # 3.  whether plt.show()
        if fig_created_internally:
            plt.show()

    def _identify_distance_shells(
        self,
        all_distances_sq: Union[Coordinates, List[float]],
        max_k: int,
        tol: float = 1e-6,
    ) -> List[float]:
        """Identifies unique distance shells from a list of squared distances.

        This helper function takes a flat list of squared distances, sorts them,
        and identifies the first `max_k` unique distance shells based on a
        numerical tolerance.

        Args:
            all_distances_sq (Union[np.ndarray, List[float]]): A list or array
                of all squared distances between pairs of sites.
            max_k (int): The maximum number of neighbor shells to identify.
            tol (float): The numerical tolerance to consider two distances equal.

        Returns:
            List[float]: A sorted list of squared distances representing the shells.
        """
        all_distances_sq = np.asarray(all_distances_sq)
        # Now, the .size call below is guaranteed to be safe.
        if all_distances_sq.size == 0:
            return []

        # Sort unique distances and filter out zero-distance (self-loops)
        unique_sorted_dist = sorted(
            [d for d in np.unique(all_distances_sq) if d > tol**2]
        )

        if not unique_sorted_dist:
            return []

        # Identify shells by checking if a new distance is significantly
        # larger than the last identified shell distance.
        dist_shells = [unique_sorted_dist[0]]
        for d_sq in unique_sorted_dist[1:]:
            if len(dist_shells) >= max_k:
                break
            # If the current distance is notably larger than the last shell's distance
            if d_sq > dist_shells[-1] + tol**2:
                dist_shells.append(d_sq)

        return dist_shells

    def _find_neighbors_by_distance(self, max_k: int = 2, tol: float = 1e-6) -> None:
        """A generic, distance-based neighbor finding method.

        This method calculates the full N x N distance matrix to find neighbor
        shells. It is computationally expensive for large N (O(N^2)) and is
        best suited for non-periodic or custom-defined lattices.

        Args:
            max_k (int, optional): The maximum number of neighbor shells to
                calculate. Defaults to 2.
            tol (float, optional): The numerical tolerance for distance
                comparisons. Defaults to 1e-6.
        """
        if self.num_sites < 2:
            return

        all_coords = np.array(self._coordinates)
        dist_matrix_sq = np.sum(
            (all_coords[:, np.newaxis, :] - all_coords[np.newaxis, :, :]) ** 2, axis=-1
        )

        all_distances_sq = dist_matrix_sq.flatten()
        dist_shells_sq = self._identify_distance_shells(all_distances_sq, max_k, tol)

        self._neighbor_maps = {k: {} for k in range(1, len(dist_shells_sq) + 1)}
        for k_idx, target_d_sq in enumerate(dist_shells_sq):
            k = k_idx + 1
            current_k_map: Dict[int, List[int]] = {}
            for i in range(self.num_sites):
                neighbor_indices = np.where(
                    np.isclose(dist_matrix_sq[i], target_d_sq, rtol=0, atol=tol**2)
                )[0]
                if len(neighbor_indices) > 0:
                    current_k_map[i] = sorted(neighbor_indices.tolist())
            self._neighbor_maps[k] = current_k_map


class TILattice(AbstractLattice):
    """Describes a periodic lattice with translational invariance.

    This class serves as a base for any lattice defined by a repeating unit
    cell. The geometry is specified by lattice vectors, the coordinates of
    basis sites within a unit cell, and the total size of the lattice in
    terms of unit cells.

    The site identifier for this class is a tuple in the format of
    `(uc_coord_1, ..., uc_coord_d, basis_index)`, where `uc_coord` represents
    the integer coordinate of the unit cell and `basis_index` is the index
    of the site within that unit cell's basis.

    Args:
        dimensionality (int): The spatial dimension of the lattice.
        lattice_vectors (np.ndarray): The lattice vectors defining the unit
            cell, given as row vectors. Shape: (dimensionality, dimensionality).
            For example, in 2D: `np.array([[ax, ay], [bx, by]])`.
        basis_coords (np.ndarray): The Cartesian coordinates of the basis sites
            within the unit cell. Shape: (num_basis_sites, dimensionality).
            For a simple Bravais lattice, this would be `np.array([[0, 0]])`.
        size (Tuple[int, ...]): A tuple specifying the number of unit cells
            to generate in each lattice vector direction (e.g., (Nx, Ny)).
        pbc (Union[bool, Tuple[bool, ...]], optional): Specifies whether
            periodic boundary conditions are applied. Can be a single boolean
            for all dimensions or a tuple of booleans for each dimension
            individually. Defaults to True.
    """

    def __init__(
        self,
        dimensionality: int,
        lattice_vectors: Coordinates,
        basis_coords: Coordinates,
        size: Tuple[int, ...],
        pbc: Union[bool, Tuple[bool, ...]] = True,
    ):
        """Initializes the Translationally Invariant Lattice."""
        super().__init__(dimensionality)
        assert lattice_vectors.shape == (
            dimensionality,
            dimensionality,
        ), "Lattice vectors shape mismatch"
        assert (
            basis_coords.shape[1] == dimensionality
        ), "Basis coordinates dimension mismatch"
        assert len(size) == dimensionality, "Size tuple length mismatch"

        self.lattice_vectors = lattice_vectors
        self.basis_coords = basis_coords
        self.num_basis = basis_coords.shape[0]
        self.size = size
        if isinstance(pbc, bool):
            self.pbc = tuple([pbc] * dimensionality)
        else:
            assert len(pbc) == dimensionality, "PBC tuple length mismatch"
            self.pbc = tuple(pbc)

        # Build the lattice sites and their neighbor relationships
        self._build_lattice()
        self._build_neighbors()  # Default: builds NN (k=1) and NNN (k=2)

    def _build_lattice(self) -> None:
        """Generates all site information for the periodic lattice.

        This method iterates through each unit cell defined by `self.size`,
        and for each unit cell, it iterates through all basis sites. It then
        calculates the real-space coordinates and creates a unique identifier
        for each site, populating the internal lattice data structures.
        """
        current_index = 0

        # Iterate over all unit cell coordinates elegantly using np.ndindex
        for cell_coord in np.ndindex(self.size):
            cell_coord_arr = np.array(cell_coord)
            # R = n1*a1 + n2*a2 + ...
            cell_vector = np.dot(cell_coord_arr, self.lattice_vectors)

            # Iterate over the basis sites within the unit cell
            for basis_index in range(self.num_basis):
                basis_vec = self.basis_coords[basis_index]

                # Calculate the real-space coordinate
                coord = cell_vector + basis_vec
                # Create a structured identifier
                identifier = cell_coord + (basis_index,)

                # Store site information
                self._indices.append(current_index)
                self._identifiers.append(identifier)
                self._coordinates.append(coord)
                self._ident_to_idx[identifier] = current_index
                current_index += 1

    def _pbc_wrap(self, cell_coord: Coordinates) -> Coordinates:
        """Applies periodic boundary conditions to a unit cell coordinate.

        Args:
            cell_coord (np.ndarray): The integer coordinate of a unit cell.

        Returns:
            np.ndarray: The wrapped unit cell coordinate according to the
            `self.pbc` settings.
        """
        wrapped_coord = np.array(cell_coord)
        for dim in range(self.dimensionality):
            if self.pbc[dim]:
                wrapped_coord[dim] %= self.size[dim]
        return wrapped_coord

    def _build_neighbors(self, max_k: int = 2, tol: float = 1e-6) -> None:
        """Calculates neighbor relationships for the periodic lattice.

        This method uses an optimized approach suitable for periodic systems.
        Instead of a full N^2 distance calculation, it considers each site
        and only computes distances to sites in its own and adjacent unit
        cells (a 3x3x... hypercube). This significantly reduces computational
        cost for large lattices. It correctly handles both periodic and open
        boundary conditions by calculating distances to unwrapped coordinates.

        Args:
            max_k (int, optional): The maximum number of neighbor shells to
                calculate. Defaults to 2.
            tol (float, optional): The numerical tolerance for distance
                comparisons. Defaults to 1e-6.
        """
        self._neighbor_maps = {k: {} for k in range(1, max_k + 1)}
        if self.num_sites < 2:
            return

        potential_neighbors: Dict[int, List[Tuple[float, int]]] = {}
        relative_cell_shifts = (
            np.array([p for p in np.ndindex(*((3,) * self.dimensionality))]) - 1
        )

        for i in range(self.num_sites):
            # Use a temporary set for each site to automatically handle duplicates
            neighbors_set: set[Tuple[float, int]] = set()

            ident_i = self._identifiers[i]
            coord_i = self._coordinates[i]
            assert isinstance(ident_i, tuple)
            cell_coord_i = np.array(ident_i[:-1])

            for cell_shift in relative_cell_shifts:
                target_cell_coord_unwrapped = cell_coord_i + cell_shift
                target_cell_coord_wrapped = self._pbc_wrap(target_cell_coord_unwrapped)

                is_valid = all(
                    self.pbc[dim]
                    or (0 <= target_cell_coord_unwrapped[dim] < self.size[dim])
                    for dim in range(self.dimensionality)
                )
                if not is_valid:
                    continue

                for basis_j in range(self.num_basis):
                    target_ident = tuple(target_cell_coord_wrapped) + (basis_j,)
                    if target_ident == ident_i:
                        continue

                    j = self._ident_to_idx[target_ident]

                    target_coord_unwrapped = (
                        np.dot(target_cell_coord_unwrapped, self.lattice_vectors)
                        + self.basis_coords[basis_j]
                    )
                    dist_sq = np.sum((target_coord_unwrapped - coord_i) ** 2)

                    if dist_sq > tol**2:
                        neighbors_set.add((dist_sq, j))

            # After checking all shifts, convert the set to a sorted list
            potential_neighbors[i] = sorted(list(neighbors_set))

        # --- The rest of the method for identifying shells remains the same ---

        # Collect all potential neighbor distances into a flat list
        all_dist_sq = [
            d_sq
            for i in range(self.num_sites)
            for d_sq, j in potential_neighbors.get(i, [])  # Use .get for safety
        ]

        # Use the helper method from the base class to identify distance shells
        dist_shells_sq = self._identify_distance_shells(all_dist_sq, max_k, tol)

        for k_idx, target_d_sq in enumerate(dist_shells_sq):
            k = k_idx + 1
            current_k_map: Dict[int, List[int]] = {}
            for i in range(self.num_sites):
                neighbors_k = [
                    j
                    for d_sq, j in potential_neighbors[i]
                    if np.isclose(d_sq, target_d_sq, rtol=0, atol=tol**2)
                ]
                if neighbors_k:
                    current_k_map[i] = sorted(neighbors_k)
            self._neighbor_maps[k] = current_k_map


class SquareLattice(TILattice):
    """A 2D square lattice.

    This is a concrete implementation of a translationally invariant lattice
    representing a simple square grid. It is a Bravais lattice with a
    single-site basis.

    Args:
        size (Tuple[int, int]): A tuple (Nx, Ny) specifying the number of
            unit cells (sites) in the x and y directions.
        lattice_constant (float, optional): The distance between two adjacent
            sites. Defaults to 1.0.
        pbc (Union[bool, Tuple[bool, bool]], optional): Specifies periodic
            boundary conditions for the x and y directions. Defaults to True.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
    ):
        """Initializes the SquareLattice."""
        dimensionality = 2

        # Define lattice vectors for a square lattice
        lattice_vectors = np.array([[lattice_constant, 0.0], [0.0, lattice_constant]])

        # A square lattice has a single site in its basis
        basis_coords = np.array([[0.0, 0.0]])

        # Call the parent TILattice constructor with these parameters
        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
        )


class HoneycombLattice(TILattice):
    """A 2D honeycomb lattice.

    This is a classic example of a composite lattice. It consists of a
    two-site basis (sublattices A and B) on an underlying triangular
    Bravais lattice.

    Args:
        size (Tuple[int, int]): A tuple (Nx, Ny) specifying the number of unit
            cells along the two lattice vector directions.
        lattice_constant (float, optional): The bond length, i.e., the distance
            between two nearest neighbor sites. Defaults to 1.0.
        pbc (Union[bool, Tuple[bool, bool]], optional): Specifies periodic
            boundary conditions. Defaults to True.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
    ):
        """Initializes the HoneycombLattice."""
        dimensionality = 2
        a = lattice_constant

        # Define the primitive lattice vectors for the underlying triangular lattice
        lattice_vectors = a * np.array([[1.5, np.sqrt(3) / 2], [1.5, -np.sqrt(3) / 2]])

        # Define the coordinates of the two basis sites (A and B)
        basis_coords = a * np.array([[0.0, 0.0], [1.0, 0.0]])  # Site A  # Site B

        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
        )


class TriangularLattice(TILattice):
    """A 2D triangular lattice.

    This is a Bravais lattice where each site has 6 nearest neighbors.

    Args:
        size (Tuple[int, int]): A tuple (Nx, Ny) specifying the number of
            unit cells along the two lattice vector directions.
        lattice_constant (float, optional): The bond length, i.e., the
            distance between two nearest neighbor sites. Defaults to 1.0.
        pbc (Union[bool, Tuple[bool, bool]], optional): Specifies periodic
            boundary conditions. Defaults to True.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
    ):
        """Initializes the TriangularLattice."""
        dimensionality = 2
        a = lattice_constant

        # Define the primitive lattice vectors for a triangular lattice
        lattice_vectors = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])

        # A triangular lattice is a Bravais lattice, with a single site in its basis
        basis_coords = np.array([[0.0, 0.0]])

        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
        )


class CustomizeLattice(AbstractLattice):
    """A general lattice built from an explicit list of sites and coordinates.

    This class is suitable for creating lattices with arbitrary geometries,
    such as finite clusters, disordered systems, or any custom structure
    that does not have translational symmetry. The lattice is defined simply
    by providing lists of identifiers and coordinates for each site.

    Args:
        dimensionality (int): The spatial dimension of the lattice.
        identifiers (List[SiteIdentifier]): A list of unique, hashable
            identifiers for the sites. The length must match `coordinates`.
        coordinates (List[Union[List[float], np.ndarray]]): A list of site
            coordinates. Each coordinate should be a list of floats or a
            NumPy array.

    Raises:
        ValueError: If the lengths of `identifiers` and `coordinates` lists
            do not match, or if a coordinate's dimension is incorrect.
    """

    def __init__(
        self,
        dimensionality: int,
        identifiers: List[SiteIdentifier],
        coordinates: List[Union[List[float], Coordinates]],
    ):
        """Initializes the CustomizeLattice."""
        super().__init__(dimensionality)
        if len(identifiers) != len(coordinates):
            raise ValueError(
                "Identifiers and coordinates lists must have the same length."
            )

        # The _build_lattice logic is simple enough to be in __init__
        self._identifiers = list(identifiers)
        self._coordinates = [np.array(c) for c in coordinates]
        self._indices = list(range(len(identifiers)))
        self._ident_to_idx = {ident: idx for idx, ident in enumerate(identifiers)}

        # Validate coordinate dimensions
        for i, coord in enumerate(self._coordinates):
            if coord.shape != (dimensionality,):
                raise ValueError(
                    f"Coordinate at index {i} has shape {coord.shape}, "
                    f"expected ({dimensionality},)"
                )

        # This print statement is useful for debugging but might be removed
        # in a final library version to avoid cluttering user output.
        print(f"CustomizeLattice with {self.num_sites} sites created.")

    def _build_lattice(self, *args: Any, **kwargs: Any) -> None:
        """For CustomizeLattice, lattice data is built during __init__."""
        pass

    def _build_neighbors(self, max_k: int = 1, tol: float = 1e-6) -> None:
        """Calculates neighbors by distance.

        This method delegates the neighbor calculation to the generic,
        distance-based `_find_neighbors_by_distance` method from the
        `AbstractLattice` base class.

        Args:
            max_k (int, optional): The maximum number of neighbor shells to
                calculate. Defaults to 1.
            tol (float, optional): The numerical tolerance for distance
                comparisons. Defaults to 1e-6.
        """
        print(f"Building neighbors for CustomizeLattice up to k={max_k}...")
        self._find_neighbors_by_distance(max_k=max_k, tol=tol)
        print("Neighbor building complete.")
