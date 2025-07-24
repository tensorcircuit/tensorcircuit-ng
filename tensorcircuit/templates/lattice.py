# -*- coding: utf-8 -*-
"""
The lattice module for defining and manipulating lattice geometries.
"""
import logging
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
    cast,
)

logger = logging.getLogger(__name__)
import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform


# This block resolves a name resolution issue for the static type checker (mypy).
# GOAL:
#   Keep `matplotlib` as an optional dependency, so it is only imported
#   inside the `show()` method, not at the module level.
# PROBLEM:
#   The type hint for the `ax` parameter in `show()`'s signature
#   (`ax: Optional["matplotlib.axes.Axes"]`) needs to know what `matplotlib` is.
#   Without this block, mypy would raise a "Name 'matplotlib' is not defined" error.
# SOLUTION:
#   The `if TYPE_CHECKING:` block is ignored at runtime but processed by mypy.
#   This makes the name `matplotlib` available to the type checker without
#   creating a hard dependency for the user.
if TYPE_CHECKING:
    import matplotlib.axes
    from mpl_toolkits.mplot3d import Axes3D

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

    :param dimensionality: The spatial dimension of the lattice (e.g., 1, 2, 3).
    :type dimensionality: int
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
        self._distance_matrix: Optional[Coordinates] = None

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
    @property
    def distance_matrix(self) -> Coordinates:
        """
        Returns the full N x N distance matrix.
        The matrix is computed on the first access and then cached for
        subsequent calls. This computation can be expensive for large lattices.
        """
        if self._distance_matrix is None:
            logger.info("Distance matrix not cached. Computing now...")
            self._distance_matrix = self._compute_distance_matrix()
        return self._distance_matrix

    def _validate_index(self, index: SiteIndex) -> None:
        """A private helper to check if a site index is within the valid range."""
        if not (0 <= index < self.num_sites):
            raise IndexError(
                f"Site index {index} out of range (0-{self.num_sites - 1})"
            )

    def get_coordinates(self, index: SiteIndex) -> Coordinates:
        """Gets the spatial coordinates of a site by its integer index.

        :param index: The integer index of the site.
        :type index: SiteIndex
        :raises IndexError: If the site index is out of range.
        :return: The spatial coordinates as a NumPy array.
        :rtype: Coordinates
        """
        self._validate_index(index)
        return self._coordinates[index]

    def get_identifier(self, index: SiteIndex) -> SiteIdentifier:
        """Gets the abstract identifier of a site by its integer index.

        :param index: The integer index of the site.
        :type index: SiteIndex
        :raises IndexError: If the site index is out of range.
        :return: The unique, hashable identifier of the site.
        :rtype: SiteIdentifier
        """
        self._validate_index(index)
        return self._identifiers[index]

    def get_index(self, identifier: SiteIdentifier) -> SiteIndex:
        """Gets the integer index of a site by its unique identifier.

        :param identifier: The unique identifier of the site.
        :type identifier: SiteIdentifier
        :raises ValueError: If the identifier is not found in the lattice.
        :return: The corresponding integer index of the site.
        :rtype: SiteIndex
        """
        try:
            return self._ident_to_idx[identifier]
        except KeyError as e:
            raise ValueError(
                f"Identifier {identifier} not found in the lattice."
            ) from e

    def get_site_info(
        self, index_or_identifier: Union[SiteIndex, SiteIdentifier]
    ) -> Tuple[SiteIndex, SiteIdentifier, Coordinates]:
        """Gets all information for a single site.

        This method provides a convenient way to retrieve all relevant data for a
        site (its index, identifier, and coordinates) by using either its
        integer index or its unique identifier.

        :param index_or_identifier: The integer
            index or the unique identifier of the site to look up.
        :type index_or_identifier: Union[SiteIndex, SiteIdentifier]
        :raises IndexError: If the given index is out of bounds.
        :raises ValueError: If the given identifier is not found in the lattice.
        :return: A tuple containing:
            - The site's integer index.
            - The site's unique identifier.
            - The site's coordinates as a NumPy array.
        :rtype: Tuple[SiteIndex, SiteIdentifier, Coordinates]
        """
        if isinstance(index_or_identifier, int):  # SiteIndex is an int
            idx = index_or_identifier
            self._validate_index(idx)
            return idx, self._identifiers[idx], self._coordinates[idx]
        else:  # Identifier
            ident = index_or_identifier
            idx = self.get_index(ident)
            return idx, ident, self._coordinates[idx]

    def sites(self) -> Iterator[Tuple[SiteIndex, SiteIdentifier, Coordinates]]:
        """Returns an iterator over all sites in the lattice.

        This provides a convenient way to loop through all sites, for example:
        `for idx, ident, coords in my_lattice.sites(): ...`

        :return: An iterator where each item is a tuple containing the site's
            index, identifier, and coordinates.
        :rtype: Iterator[Tuple[SiteIndex, SiteIdentifier, Coordinates]]
        """
        for i in range(self.num_sites):
            yield i, self._identifiers[i], self._coordinates[i]

    def get_neighbors(self, index: SiteIndex, k: int = 1) -> List[SiteIndex]:
        """Gets the list of k-th nearest neighbor indices for a given site.

        :param index: The integer index of the center site.
        :type index: SiteIndex
        :param k: The order of the neighbors, where k=1 corresponds
            to nearest neighbors (NN), k=2 to next-nearest neighbors (NNN),
            and so on. Defaults to 1.
        :type k: int, optional
        :return: A list of integer indices for the neighboring sites.
            Returns an empty list if neighbors for the given `k` have not been
            pre-calculated or if the site has no such neighbors.
        :rtype: List[SiteIndex]
        """
        if k not in self._neighbor_maps:
            logger.info(
                f"Neighbors for k={k} not pre-computed. Building now up to max_k={k}."
            )
            self._build_neighbors(max_k=k)

        if k not in self._neighbor_maps:
            return []

        return self._neighbor_maps[k].get(index, [])

    def get_neighbor_pairs(
        self, k: int = 1, unique: bool = True
    ) -> List[Tuple[SiteIndex, SiteIndex]]:
        """Gets all pairs of k-th nearest neighbors, representing bonds.

        :param k: The order of the neighbors to consider.
            Defaults to 1.
        :type k: int, optional
        :param unique: If True, returns only one representation
            for each pair (i, j) such that i < j, avoiding duplicates
            like (j, i). If False, returns all directed pairs.
            Defaults to True.
        :type unique: bool, optional
        :return: A list of tuples, where each
            tuple is a pair of neighbor indices.
        :rtype: List[Tuple[SiteIndex, SiteIndex]]
        """

        if k not in self._neighbor_maps:
            logger.info(
                f"Neighbor pairs for k={k} not pre-computed. Building now up to max_k={k}."
            )
            self._build_neighbors(max_k=k)

        # After attempting to build, check again. If still not found, return empty.
        if k not in self._neighbor_maps:
            return []

        pairs = []
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
    def _build_neighbors(self, max_k: int = 1, **kwargs: Any) -> None:
        """
        Abstract method for subclasses to calculate neighbor relationships.

        A concrete implementation of this method should calculate the neighbor
        relationships up to `max_k` and populate the `self._neighbor_maps`
        dictionary. The keys of the dictionary should be the neighbor order (k),
        and the values should be a dictionary mapping site indices to their
        list of k-th neighbors.
        """
        pass

    @abc.abstractmethod
    def _compute_distance_matrix(self) -> Coordinates:
        """
        Abstract method for subclasses to implement the actual matrix calculation.
        This method is called by the `distance_matrix` property when the matrix
        needs to be computed for the first time.
        """
        pass

    def show(
        self,
        show_indices: bool = False,
        show_identifiers: bool = False,
        show_bonds_k: Optional[int] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        bond_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Visualizes the lattice structure using Matplotlib.

        This method supports 1D, 2D, and 3D plotting. For 1D lattices, sites
        are plotted along the x-axis.

        :param show_indices: If True, displays the integer index
            next to each site. Defaults to False.
        :type show_indices: bool, optional
        :param show_identifiers: If True, displays the unique
            identifier next to each site. Defaults to False.
        :type show_identifiers: bool, optional
        :param show_bonds_k: Specifies which order of
            neighbor bonds to draw (e.g., 1 for NN, 2 for NNN). If None,
            no bonds are drawn. If the specified neighbors have not been
            calculated, a warning is printed. Defaults to None.
        :type show_bonds_k: Optional[int], optional
        :param ax: An existing Matplotlib Axes object to plot on.
            If None, a new Figure and Axes are created automatically. Defaults to None.
        :type ax: Optional["matplotlib.axes.Axes"], optional
        :param bond_kwargs: A dictionary of keyword arguments for customizing bond appearance,
            passed directly to the Matplotlib plot function. Defaults to None.
        :type bond_kwargs: Optional[Dict[str, Any]], optional

        :param kwargs: Additional keyword arguments to be passed directly to the
            `matplotlib.pyplot.scatter` function for customizing site appearance.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error(
                "Matplotlib is required for visualization. "
                "Please install it using 'pip install matplotlib'."
            )
            return

        # creat "fig_created_internally" as flag
        fig_created_internally = False

        if self.num_sites == 0:
            logger.info("Lattice is empty, nothing to show.")
            return
        if self.dimensionality not in [1, 2, 3]:
            logger.warning(
                f"show() is not implemented for {self.dimensionality}D lattices."
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

                # Robust Logic: Decide plotting strategy based on known dimensionality.

                if self.dimensionality == 1:
                    ax.text(coords[i, 0], offset, label, fontsize=9, ha="center")
                elif self.dimensionality == 2:
                    ax.text(
                        coords[i, 0] + offset,
                        coords[i, 1] + offset,
                        label,
                        fontsize=9,
                        zorder=3,
                    )
                elif self.dimensionality == 3:
                    ax_3d = cast("Axes3D", ax)
                    ax_3d.text(
                        coords[i, 0],
                        coords[i, 1],
                        coords[i, 2] + offset,
                        label,
                        fontsize=9,
                        zorder=3,
                    )

                # Note: No 'else' needed as we already check dimensionality at the start.

        if show_bonds_k is not None:
            if show_bonds_k not in self._neighbor_maps:
                logger.warning(
                    f"Cannot draw bonds. k={show_bonds_k} neighbors have not been calculated."
                )
            else:
                try:
                    bonds = self.get_neighbor_pairs(k=show_bonds_k, unique=True)
                    plot_bond_kwargs = {
                        "color": "k",
                        "linestyle": "-",
                        "alpha": 0.6,
                        "zorder": 1,
                    }
                    if bond_kwargs:
                        plot_bond_kwargs.update(bond_kwargs)

                    if self.dimensionality > 2:
                        ax_3d = cast("Axes3D", ax)
                        for i, j in bonds:
                            p1, p2 = self._coordinates[i], self._coordinates[j]
                            ax_3d.plot(
                                [p1[0], p2[0]],
                                [p1[1], p2[1]],
                                [p1[2], p2[2]],
                                **plot_bond_kwargs,
                            )
                    else:
                        for i, j in bonds:
                            p1, p2 = self._coordinates[i], self._coordinates[j]
                            if self.dimensionality == 1:  #  type: ignore

                                ax.plot([p1[0], p2[0]], [0, 0], **plot_bond_kwargs)  # type: ignore
                            else:  # dimensionality == 2
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **plot_bond_kwargs)  # type: ignore

                except ValueError as e:
                    logger.info(f"Could not draw bonds: {e}")

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

        :param all_distances_sq: A list or array
            of all squared distances between pairs of sites.
        :type all_distances_sq: Union[np.ndarray, List[float]]
        :param max_k: The maximum number of neighbor shells to identify.
        :type max_k: int
        :param tol: The numerical tolerance to consider two distances equal.
        :type tol: float
        :return: A sorted list of squared distances representing the shells.
        :rtype: List[float]
        """
        ZERO_THRESHOLD_SQ = 1e-12

        all_distances_sq = np.asarray(all_distances_sq)
        # Now, the .size call below is guaranteed to be safe.
        if all_distances_sq.size == 0:
            return []

        sorted_dist = np.sort(all_distances_sq[all_distances_sq > ZERO_THRESHOLD_SQ])

        if sorted_dist.size == 0:
            return []

        # Identify shells using the user-provided tolerance.
        dist_shells = [sorted_dist[0]]

        for d_sq in sorted_dist[1:]:
            if len(dist_shells) >= max_k:
                break
            # If the current distance is notably larger than the last shell's distance
            if d_sq > dist_shells[-1] + tol**2:
                dist_shells.append(d_sq)

        return dist_shells

    def _build_neighbors_by_distance_matrix(
        self, max_k: int = 2, tol: float = 1e-6
    ) -> None:
        """A generic, distance-based neighbor finding method.

        This method calculates the full N x N distance matrix to find neighbor
        shells. It is computationally expensive for large N (O(N^2)) and is
        best suited for non-periodic or custom-defined lattices.

        :param max_k: The maximum number of neighbor shells to
            calculate. Defaults to 2.
        :type max_k: int, optional
        :param tol: The numerical tolerance for distance
            comparisons. Defaults to 1e-6.
        :type tol: float, optional
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
        self._distance_matrix = np.sqrt(dist_matrix_sq)


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

    :param dimensionality: The spatial dimension of the lattice.
    :type dimensionality: int
    :param lattice_vectors: The lattice vectors defining the unit
        cell, given as row vectors. Shape: (dimensionality, dimensionality).
        For example, in 2D: `np.array([[ax, ay], [bx, by]])`.
    :type lattice_vectors: np.ndarray
    :param basis_coords: The Cartesian coordinates of the basis sites
        within the unit cell. Shape: (num_basis_sites, dimensionality).
        For a simple Bravais lattice, this would be `np.array([[0, 0]])`.
    :type basis_coords: np.ndarray
    :param size: A tuple specifying the number of unit cells
        to generate in each lattice vector direction (e.g., (Nx, Ny)).
    :type size: Tuple[int, ...]
    :param pbc: Specifies whether
        periodic boundary conditions are applied. Can be a single boolean
        for all dimensions or a tuple of booleans for each dimension
        individually. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, ...]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional

    """

    def __init__(
        self,
        dimensionality: int,
        lattice_vectors: Coordinates,
        basis_coords: Coordinates,
        size: Tuple[int, ...],
        pbc: Union[bool, Tuple[bool, ...]] = True,
        precompute_neighbors: Optional[int] = None,
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
        if precompute_neighbors is not None and precompute_neighbors > 0:
            logger.info(f"Pre-computing neighbors up to k={precompute_neighbors}...")
            self._build_neighbors(max_k=precompute_neighbors)

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

    def _get_distance_matrix_with_mic(self) -> Coordinates:
        """
        Computes the full N x N distance matrix, correctly applying the
        Minimum Image Convention (MIC) for all periodic dimensions.
        """
        all_coords = np.array(self._coordinates)
        size_arr = np.array(self.size)
        system_vectors = self.lattice_vectors * size_arr[:, np.newaxis]

        # Generate translation vectors ONLY for periodic dimensions
        pbc_dims = [d for d in range(self.dimensionality) if self.pbc[d]]
        translations = [np.zeros(self.dimensionality)]
        if pbc_dims:
            num_pbc_dims = len(pbc_dims)
            pbc_system_vectors = system_vectors[pbc_dims, :]

            # Create all 3^k - 1 non-zero shifts for k periodic dimensions
            shift_options = [np.array([-1, 0, 1])] * num_pbc_dims
            shifts_grid = np.meshgrid(*shift_options, indexing="ij")
            all_shifts = np.stack(shifts_grid, axis=-1).reshape(-1, num_pbc_dims)
            all_shifts = all_shifts[np.any(all_shifts != 0, axis=1)]

            pbc_translations = all_shifts @ pbc_system_vectors
            translations.extend(pbc_translations)

        translations_arr = np.array(translations, dtype=float)

        # Calculate the distance matrix applying MIC
        dist_matrix_sq = np.full((self.num_sites, self.num_sites), np.inf, dtype=float)
        for i in range(self.num_sites):
            displacements = all_coords - all_coords[i]
            image_displacements = (
                displacements[:, np.newaxis, :] - translations_arr[np.newaxis, :, :]
            )
            image_d_sq = np.sum(image_displacements**2, axis=2)
            dist_matrix_sq[i, :] = np.min(image_d_sq, axis=1)

        return cast(Coordinates, np.sqrt(dist_matrix_sq))

    def _build_neighbors(self, max_k: int = 2, **kwargs: Any) -> None:
        """Calculates neighbor relationships for the periodic lattice.

        This method calculates neighbor relationships by computing the full N x N
        distance matrix. It robustly handles all boundary conditions (fully
        periodic, open, or mixed) by applying the Minimum Image Convention
        (MIC) only to the periodic dimensions.

        From this distance matrix, it identifies unique neighbor shells up to
        the specified `max_k` and populates the neighbor maps. The computed
        distance matrix is then cached for future use.

        :param max_k: The maximum number of neighbor shells to
            calculate. Defaults to 2.
        :type max_k: int, optional
        :param tol: The numerical tolerance for distance
            comparisons. Defaults to 1e-6.
        :type tol: float, optional
        """
        tol = kwargs.get("tol", 1e-6)
        dist_matrix = self._get_distance_matrix_with_mic()
        dist_matrix_sq = dist_matrix**2
        self._distance_matrix = dist_matrix
        all_distances_sq = dist_matrix_sq.flatten()
        dist_shells_sq = self._identify_distance_shells(all_distances_sq, max_k, tol)

        self._neighbor_maps = {k: {} for k in range(1, len(dist_shells_sq) + 1)}
        for k_idx, target_d_sq in enumerate(dist_shells_sq):
            k = k_idx + 1
            current_k_map: Dict[int, List[int]] = {}
            match_indices = np.where(
                np.isclose(dist_matrix_sq, target_d_sq, rtol=0, atol=tol**2)
            )
            for i, j in zip(*match_indices):
                if i == j:
                    continue
                if i not in current_k_map:
                    current_k_map[i] = []
                current_k_map[i].append(j)

            for i in current_k_map:
                current_k_map[i].sort()

            self._neighbor_maps[k] = current_k_map

    def _compute_distance_matrix(self) -> Coordinates:
        """Computes the distance matrix using the Minimum Image Convention."""
        return self._get_distance_matrix_with_mic()


class SquareLattice(TILattice):
    """A 2D square lattice.

    This is a concrete implementation of a translationally invariant lattice
    representing a simple square grid. It is a Bravais lattice with a
    single-site basis.

    :param size: A tuple (Nx, Ny) specifying the number of
        unit cells (sites) in the x and y directions.
    :type size: Tuple[int, int]
    :param lattice_constant: The distance between two adjacent
        sites. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies periodic boundary conditions. Can be a single boolean
        for all dimensions or a tuple of booleans for each dimension
        individually. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
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
            precompute_neighbors=precompute_neighbors,
        )


class HoneycombLattice(TILattice):
    """A 2D honeycomb lattice.

    This is a classic example of a composite lattice. It consists of a
    two-site basis (sublattices A and B) on an underlying triangular
    Bravais lattice.

    :param size: A tuple (Nx, Ny) specifying the number of unit
        cells along the two lattice vector directions.
    :type size: Tuple[int, int]
    :param lattice_constant: The bond length, i.e., the distance
        between two nearest neighbor sites. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies periodic
        boundary conditions. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional

    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
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
            precompute_neighbors=precompute_neighbors,
        )


class TriangularLattice(TILattice):
    """A 2D triangular lattice.

    This is a Bravais lattice where each site has 6 nearest neighbors.

    :param size: A tuple (Nx, Ny) specifying the number of
        unit cells along the two lattice vector directions.
    :type size: Tuple[int, int]
    :param lattice_constant: The bond length, i.e., the
        distance between two nearest neighbor sites. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies periodic
        boundary conditions. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional

    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
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
            precompute_neighbors=precompute_neighbors,
        )


class ChainLattice(TILattice):
    """A 1D chain (simple Bravais lattice).

    :param size: A tuple `(N,)` specifying the number of sites in the chain.
    :type size: Tuple[int]
    :param lattice_constant: The distance between two adjacent sites. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies if periodic boundary conditions are applied. Defaults to True.
    :type pbc: bool, optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int],
        lattice_constant: float = 1.0,
        pbc: bool = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 1
        lattice_vectors = np.array([[lattice_constant]])
        basis_coords = np.array([[0.0]])
        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
            precompute_neighbors=precompute_neighbors,
        )


class DimerizedChainLattice(TILattice):
    """A 1D chain with an AB sublattice (dimerized chain).

    The unit cell contains two sites, A and B. The bond length is uniform.

    :param size: A tuple `(N,)` specifying the number of **unit cells**.
        The total number of sites in the chain will be `2 * N`, as each
        unit cell contains two sites.
    :type size: Tuple[int]
    :param lattice_constant: The distance between two adjacent sites (bond length). Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies if periodic boundary conditions are applied. Defaults to True.
    :type pbc: bool, optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int],
        lattice_constant: float = 1.0,
        pbc: bool = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 1
        # The unit cell vector connects two A sites, spanning length 2*a
        lattice_vectors = np.array([[2 * lattice_constant]])
        # Basis has site A at origin, site B at distance 'a'
        basis_coords = np.array([[0.0], [lattice_constant]])

        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
            precompute_neighbors=precompute_neighbors,
        )


class RectangularLattice(TILattice):
    """A 2D rectangular lattice.

    This is a generalization of the SquareLattice where the lattice constants
    in the x and y directions can be different.

    :param size: A tuple (Nx, Ny) specifying the number of sites in x and y.
    :type size: Tuple[int, int]
    :param lattice_constants: The distance between adjacent sites
        in the x and y directions, e.g., (ax, ay). Defaults to (1.0, 1.0).
    :type lattice_constants: Tuple[float, float], optional
    :param pbc: Specifies periodic boundary conditions. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constants: Tuple[float, float] = (1.0, 1.0),
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 2
        ax, ay = lattice_constants
        lattice_vectors = np.array([[ax, 0.0], [0.0, ay]])
        basis_coords = np.array([[0.0, 0.0]])

        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
            precompute_neighbors=precompute_neighbors,
        )


class CheckerboardLattice(TILattice):
    """A 2D checkerboard lattice (a square lattice with an AB sublattice).

    The unit cell is a square rotated by 45 degrees, containing two sites.

    :param size: A tuple (Nx, Ny) specifying the number of unit cells. Total sites will be 2*Nx*Ny.
    :type size: Tuple[int, int]
    :param lattice_constant: The bond length between nearest neighbors. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies periodic boundary conditions. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 2
        a = lattice_constant
        # Primitive vectors for a square lattice rotated by 45 degrees.
        lattice_vectors = a * np.array([[1.0, 1.0], [1.0, -1.0]])
        # Two-site basis
        basis_coords = a * np.array([[0.0, 0.0], [1.0, 0.0]])
        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
            precompute_neighbors=precompute_neighbors,
        )


class KagomeLattice(TILattice):
    """A 2D Kagome lattice.

    This is a lattice with a three-site basis on a triangular Bravais lattice.

    :param size: A tuple (Nx, Ny) specifying the number of unit cells. Total sites will be 3*Nx*Ny.
    :type size: Tuple[int, int]
    :param lattice_constant: The bond length. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies periodic boundary conditions. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 2
        a = lattice_constant
        # Using a rectangular unit cell definition for simplicity
        lattice_vectors = a * np.array([[2.0, 0.0], [1.0, np.sqrt(3)]])
        # Three-site basis
        basis_coords = a * np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])
        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
            precompute_neighbors=precompute_neighbors,
        )


class LiebLattice(TILattice):
    """A 2D Lieb lattice.

    This is a lattice with a three-site basis on a square Bravais lattice.
    It has sites at the corners and centers of the edges of a square.

    :param size: A tuple (Nx, Ny) specifying the number of unit cells. Total sites will be 3*Nx*Ny.
    :type size: Tuple[int, int]
    :param lattice_constant: The bond length. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies periodic boundary conditions. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        """Initializes the LiebLattice."""
        dimensionality = 2
        # Use a more descriptive name for clarity. In a Lieb lattice,
        # the lattice_constant is the bond length between nearest neighbors.
        bond_length = lattice_constant

        # The unit cell of a Lieb lattice is a square with side length
        # equal to twice the bond length.
        unit_cell_side = 2 * bond_length
        lattice_vectors = np.array([[unit_cell_side, 0.0], [0.0, unit_cell_side]])

        # The three-site basis consists of a corner site, a site on the
        # center of the horizontal edge, and a site on the center of the vertical edge.
        # Their coordinates are defined directly in terms of the physical bond length.
        basis_coords = np.array(
            [
                [0.0, 0.0],  # Corner site
                [bond_length, 0.0],  # Horizontal edge center
                [0.0, bond_length],  # Vertical edge center
            ]
        )

        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
            precompute_neighbors=precompute_neighbors,
        )


class CubicLattice(TILattice):
    """A 3D cubic lattice.

    This is a simple Bravais lattice, the 3D generalization of SquareLattice.

    :param size: A tuple (Nx, Ny, Nz) specifying the number of sites.
    :type size: Tuple[int, int, int]
    :param lattice_constant: The distance between adjacent sites. Defaults to 1.0.
    :type lattice_constant: float, optional
    :param pbc: Specifies periodic boundary conditions. Defaults to True.
    :type pbc: Union[bool, Tuple[bool, bool, bool]], optional
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional
    """

    def __init__(
        self,
        size: Tuple[int, int, int],
        lattice_constant: float = 1.0,
        pbc: Union[bool, Tuple[bool, bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 3
        a = lattice_constant
        lattice_vectors = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        basis_coords = np.array([[0.0, 0.0, 0.0]])
        super().__init__(
            dimensionality=dimensionality,
            lattice_vectors=lattice_vectors,
            basis_coords=basis_coords,
            size=size,
            pbc=pbc,
            precompute_neighbors=precompute_neighbors,
        )


class CustomizeLattice(AbstractLattice):
    """A general lattice built from an explicit list of sites and coordinates.

    This class is suitable for creating lattices with arbitrary geometries,
    such as finite clusters, disordered systems, or any custom structure
    that does not have translational symmetry. The lattice is defined simply
    by providing lists of identifiers and coordinates for each site.

    :param dimensionality: The spatial dimension of the lattice.
    :type dimensionality: int
    :param identifiers: A list of unique, hashable
        identifiers for the sites. The length must match `coordinates`.
    :type identifiers: List[SiteIdentifier]
    :param coordinates: A list of site
        coordinates. Each coordinate should be a list of floats or a
        NumPy array.
    :type coordinates: List[Union[List[float], Coordinates]]
    :raises ValueError: If the lengths of `identifiers` and `coordinates` lists
        do not match, or if a coordinate's dimension is incorrect.
    :param precompute_neighbors: If specified, pre-computes neighbor relationships
        up to the given order `k` upon initialization. Defaults to None.
    :type precompute_neighbors: Optional[int], optional

    """

    def __init__(
        self,
        dimensionality: int,
        identifiers: List[SiteIdentifier],
        coordinates: List[Union[List[float], Coordinates]],
        precompute_neighbors: Optional[int] = None,
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

        logger.info(f"CustomizeLattice with {self.num_sites} sites created.")

        if precompute_neighbors is not None and precompute_neighbors > 0:
            self._build_neighbors(max_k=precompute_neighbors)

    def _build_lattice(self, *args: Any, **kwargs: Any) -> None:
        """For CustomizeLattice, lattice data is built during __init__."""
        pass

    def _build_neighbors(self, max_k: int = 1, **kwargs: Any) -> None:
        """Calculates neighbors using a KDTree for efficiency.

        This method uses a memory-efficient approach to identify neighbors without
        initially computing the full N x N distance matrix. It leverages
        `scipy.spatial.distance.pdist` to find unique distance shells and then
        a `scipy.spatial.KDTree` for fast radius queries. This approach is
        significantly more memory-efficient during the neighbor identification phase.

        After the neighbors are identified, the full distance matrix is computed
        from the pairwise distances and cached for potential future use.

        :param max_k: The maximum number of neighbor shells to
            calculate. Defaults to 1.
        :type max_k: int, optional
        :param tol: The numerical tolerance for distance
            comparisons. Defaults to 1e-6.
        :type tol: float, optional
        """
        tol = kwargs.get("tol", 1e-6)
        logger.info(f"Building neighbors for CustomizeLattice up to k={max_k}...")
        if self.num_sites < 2:
            return

        all_coords = np.array(self._coordinates)

        # 1. Use pdist for memory-efficient calculation of pairwise distances
        #    to robustly identify the distance shells.
        all_distances_sq = pdist(all_coords, metric="sqeuclidean")
        dist_shells_sq = self._identify_distance_shells(all_distances_sq, max_k, tol)

        if not dist_shells_sq:
            logger.info("No distinct neighbor shells found.")
            return

        # 2. Build the KDTree for efficient querying.
        tree = KDTree(all_coords)
        self._neighbor_maps = {k: {} for k in range(1, len(dist_shells_sq) + 1)}

        # 3. Find neighbors by isolating shells using inclusion-exclusion.
        # `found_indices` will store all neighbors within a given radius.
        found_indices: List[set[int]] = []
        for k_idx, target_d_sq in enumerate(dist_shells_sq):
            radius = np.sqrt(target_d_sq) + tol
            # Query for all points within the new, larger radius.
            current_shell_indices = tree.query_ball_point(
                all_coords, r=radius, return_sorted=True
            )

            # Now, isolate the neighbors for the current shell k
            k = k_idx + 1
            current_k_map: Dict[int, List[int]] = {}
            for i in range(self.num_sites):

                if k_idx == 0:
                    co_located_indices = tree.query_ball_point(all_coords[i], r=1e-12)
                    prev_found = set(co_located_indices)
                else:
                    prev_found = found_indices[i]

                # The new neighbors are those in the current radius shell,
                # excluding those already found in smaller shells.
                new_neighbors = set(current_shell_indices[i]) - prev_found

                if new_neighbors:
                    current_k_map[i] = sorted(list(new_neighbors))

            self._neighbor_maps[k] = current_k_map
            found_indices = [
                set(l) for l in current_shell_indices
            ]  # Update for next iteration
        self._distance_matrix = np.sqrt(squareform(all_distances_sq))

        logger.info("Neighbor building complete using KDTree.")

    def _compute_distance_matrix(self) -> Coordinates:
        """Computes the distance matrix from the stored coordinates.

        This implementation uses scipy.pdist for a memory-efficient
        calculation of pairwise distances, which is then converted to a
        full square matrix.
        """
        if self.num_sites < 2:
            return cast(Coordinates, np.empty((self.num_sites, self.num_sites)))

        all_coords = np.array(self._coordinates)
        # Use pdist for memory-efficiency, then build the full matrix.
        all_distances_sq = pdist(all_coords, metric="sqeuclidean")
        dist_matrix_sq = squareform(all_distances_sq)
        return cast(Coordinates, np.sqrt(dist_matrix_sq))

    def _reset_computations(self) -> None:
        """Resets all cached data that depends on the lattice structure."""
        self._neighbor_maps = {}
        self._distance_matrix = None

    @classmethod
    def from_lattice(cls, lattice: "AbstractLattice") -> "CustomizeLattice":
        """Creates a CustomizeLattice instance from any existing lattice object.

        This is useful for 'detaching' a procedurally generated lattice (like
        a SquareLattice) into a customizable one for further modifications,
        such as adding defects or extra sites.

        :param lattice: An instance of any AbstractLattice subclass.
        :type lattice: AbstractLattice
        :return: A new CustomizeLattice instance with the same sites.
        :rtype: CustomizeLattice
        """
        all_sites_info = list(lattice.sites())

        if not all_sites_info:
            return cls(
                dimensionality=lattice.dimensionality, identifiers=[], coordinates=[]
            )

        # Unzip the list of tuples into separate lists of identifiers and coordinates
        _, identifiers, coordinates = zip(*all_sites_info)

        return cls(
            dimensionality=lattice.dimensionality,
            identifiers=list(identifiers),
            coordinates=list(coordinates),
        )

    def add_sites(
        self,
        identifiers: List[SiteIdentifier],
        coordinates: List[Union[List[float], Coordinates]],
    ) -> None:
        """Adds new sites to the lattice.

        This operation modifies the lattice in-place. After adding sites, any
        previously computed neighbor information is cleared and must be
        recalculated.

        :param identifiers: A list of unique, hashable identifiers for the new sites.
        :type identifiers: List[SiteIdentifier]
        :param coordinates: A list of coordinates for the new sites.
        :type coordinates: List[Union[List[float], np.ndarray]]
        :raises ValueError: If input lists have mismatched lengths, or if any new
            identifier already exists in the lattice.
        """
        if len(identifiers) != len(coordinates):
            raise ValueError(
                "Identifiers and coordinates lists must have the same length."
            )
        if not identifiers:
            return  # Nothing to add

        # Check for duplicate identifiers before making any changes
        existing_ids = set(self._identifiers)
        new_ids = set(identifiers)
        if not new_ids.isdisjoint(existing_ids):
            raise ValueError(
                f"Duplicate identifiers found: {new_ids.intersection(existing_ids)}"
            )

        for i, coord in enumerate(coordinates):
            coord_arr = np.asarray(coord)
            if coord_arr.shape != (self.dimensionality,):
                raise ValueError(
                    f"New coordinate at index {i} has shape {coord_arr.shape}, "
                    f"expected ({self.dimensionality},)"
                )
            self._coordinates.append(coord_arr)
            self._identifiers.append(identifiers[i])

        # Rebuild index mappings from scratch
        self._indices = list(range(len(self._identifiers)))
        self._ident_to_idx = {ident: idx for idx, ident in enumerate(self._identifiers)}

        # Invalidate any previously computed neighbors or distance matrices
        self._reset_computations()
        logger.info(
            f"{len(identifiers)} sites added. Lattice now has {self.num_sites} sites."
        )

    def remove_sites(self, identifiers: List[SiteIdentifier]) -> None:
        """Removes specified sites from the lattice.

        This operation modifies the lattice in-place. After removing sites,
        all site indices are re-calculated, and any previously computed
        neighbor information is cleared.

        :param identifiers: A list of identifiers for the sites to be removed.
        :type identifiers: List[SiteIdentifier]
        :raises ValueError: If any of the specified identifiers do not exist.
        """
        if not identifiers:
            return  # Nothing to remove

        ids_to_remove = set(identifiers)
        current_ids = set(self._identifiers)
        if not ids_to_remove.issubset(current_ids):
            raise ValueError(
                f"Non-existent identifiers provided for removal: {ids_to_remove - current_ids}"
            )

        # Create new lists containing only the sites to keep
        new_identifiers: List[SiteIdentifier] = []
        new_coordinates: List[Coordinates] = []
        for ident, coord in zip(self._identifiers, self._coordinates):
            if ident not in ids_to_remove:
                new_identifiers.append(ident)
                new_coordinates.append(coord)

        # Replace old data with the new, filtered data
        self._identifiers = new_identifiers
        self._coordinates = new_coordinates

        # Rebuild index mappings
        self._indices = list(range(len(self._identifiers)))
        self._ident_to_idx = {ident: idx for idx, ident in enumerate(self._identifiers)}

        # Invalidate caches
        self._reset_computations()
        logger.info(
            f"{len(ids_to_remove)} sites removed. Lattice now has {self.num_sites} sites."
        )
