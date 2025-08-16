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
    Set,
)

import itertools
import math
import numpy as np
from scipy.spatial import KDTree

from .. import backend


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

logger = logging.getLogger(__name__)

Tensor = Any
SiteIndex = int
SiteIdentifier = Hashable
Coordinates = Tensor

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

        # Core data structures for storing site information.
        self._indices: List[SiteIndex] = []  # List of integer indices [0, 1, ..., N-1]
        self._identifiers: List[SiteIdentifier] = (
            []
        )  # List of unique, hashable site identifiers
        # Always initialize to an empty coordinate tensor with correct dimensionality
        # so that type checkers know this is indexable and not Optional.
        self._coordinates: Coordinates = backend.zeros((0, dimensionality))

        # Mappings for efficient lookups.
        self._ident_to_idx: Dict[SiteIdentifier, SiteIndex] = (
            {}
        )  # Maps identifiers to indices

        # Cached properties, computed on demand.
        self._neighbor_maps: Dict[int, NeighborMap] = (
            {}
        )  # Caches neighbor info for different k
        self._distance_matrix: Optional[Coordinates] = (
            None  # Caches the full N x N distance matrix
        )

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
        coords = self._coordinates[index]
        return coords

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
            index = self._ident_to_idx[identifier]
            return index
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
        else:
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

    def get_all_pairs(self) -> List[Tuple[SiteIndex, SiteIndex]]:
        """
        Returns a list of all unique pairs of site indices (i, j) where i < j.

        This method provides all-to-all connectivity, useful for Hamiltonians
        where every site interacts with every other site.

        Note on Differentiability:
        This method provides a static list of index pairs and is not differentiable
        itself. However, it is designed to be used in combination with the fully
        differentiable ``distance_matrix`` property. By using the pairs from this
        method to index into the ``distance_matrix``, one can construct differentiable
        objective functions based on all-pair interactions, effectively bypassing the
        non-differentiable ``get_neighbor_pairs`` method for geometry optimization tasks.

        :return: A list of tuples, where each tuple is a unique pair of site indices.
        :rtype: List[Tuple[SiteIndex, SiteIndex]]
        """
        if self.num_sites < 2:
            return []
        # Use itertools.combinations to efficiently generate all unique pairs (i, j) with i < j.
        return sorted(list(itertools.combinations(range(self.num_sites), 2)))

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

    def _compute_distance_matrix(self) -> Coordinates:
        """
        Default generic distance matrix computation (no periodic images).

        Subclasses can override this when a specialized rule is required
        (e.g., applying Minimum Image Convention for PBC in TILattice).
        """
        # Handle empty lattices and trivial 1-site lattices
        if self.num_sites == 0:
            return backend.zeros((0, 0))

        # Vectorized pairwise Euclidean distances
        all_coords = self._coordinates
        displacements = backend.expand_dims(all_coords, 1) - backend.expand_dims(
            all_coords, 0
        )
        dist_matrix_sq = backend.sum(displacements**2, axis=-1)
        return backend.sqrt(dist_matrix_sq)

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
            logger.warning(
                "Matplotlib is required for visualization. "
                "Please install it using 'pip install matplotlib'."
            )
            return

        # Flag to track if the Matplotlib figure was created by this method.
        # This prevents calling plt.show() if the user provided their own Axes.
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
            # If no Axes object is provided, create a new figure and axes.
            fig_created_internally = True
            if self.dimensionality == 3:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure  # type: ignore

        coords = np.array(self._coordinates)
        # Prepare arguments for the scatter plot, allowing user overrides.
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
                # Calculate a small offset for placing text labels to avoid overlap with sites.
                offset = (
                    0.02 * np.max(np.ptp(coords, axis=0)) if coords.size > 0 else 0.1
                )

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
                            else:
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

        # Display the plot only if the figure was created within this function.
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
        # A small threshold to filter out zero distances (site to itself).
        ZERO_THRESHOLD_SQ = 1e-12

        all_distances_sq = backend.convert_to_tensor(all_distances_sq)
        # Now, the .size call below is guaranteed to be safe.
        if backend.sizen(all_distances_sq) == 0:
            return []

        # Filter out self-distances and sort the remaining squared distances.
        sorted_dist = backend.sort(
            all_distances_sq[all_distances_sq > ZERO_THRESHOLD_SQ]
        )

        if backend.sizen(sorted_dist) == 0:
            return []

        dist_shells = [sorted_dist[0]]

        for d_sq in sorted_dist[1:]:
            if len(dist_shells) >= max_k:
                break
            if backend.sqrt(d_sq) - backend.sqrt(dist_shells[-1]) > tol:
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

        all_coords = self._coordinates
        # Vectorized computation of the squared distance matrix:
        # (N, 1, D) - (1, N, D) -> (N, N, D) -> (N, N)
        displacements = backend.expand_dims(all_coords, 1) - backend.expand_dims(
            all_coords, 0
        )
        dist_matrix_sq = backend.sum(displacements**2, axis=-1)

        # Flatten the matrix to a list of all squared distances to identify shells.
        all_distances_sq = backend.reshape(dist_matrix_sq, [-1])
        dist_shells_sq = self._identify_distance_shells(all_distances_sq, max_k, tol)

        self._neighbor_maps = self._build_neighbor_map_from_distances(
            dist_matrix_sq, dist_shells_sq, tol
        )
        self._distance_matrix = backend.sqrt(dist_matrix_sq)

    def _build_neighbor_map_from_distances(
        self,
        dist_matrix_sq: Coordinates,
        dist_shells_sq: List[float],
        tol: float = 1e-6,
    ) -> Dict[int, NeighborMap]:
        """
        Builds a neighbor map from a squared distance matrix and identified shells.
        This is a generic helper function to reduce code duplication.
        """
        neighbor_maps: Dict[int, NeighborMap] = {
            k: {} for k in range(1, len(dist_shells_sq) + 1)
        }
        for k_idx, target_d_sq in enumerate(dist_shells_sq):
            k = k_idx + 1
            current_k_map: Dict[int, List[int]] = {}
            # For each shell, find all pairs of sites (i, j) with that distance.
            is_close_matrix = backend.abs(dist_matrix_sq - target_d_sq) < tol
            rows, cols = backend.where(is_close_matrix)

            for i, j in zip(backend.numpy(rows), backend.numpy(cols)):
                if i == j:
                    continue
                if i not in current_k_map:
                    current_k_map[i] = []
                current_k_map[i].append(j)

            for i in current_k_map:
                current_k_map[i].sort()

            neighbor_maps[k] = current_k_map
        return neighbor_maps


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

        self.lattice_vectors = backend.convert_to_tensor(lattice_vectors)
        self.basis_coords = backend.convert_to_tensor(basis_coords)

        if self.lattice_vectors.shape != (dimensionality, dimensionality):
            raise ValueError(
                f"Lattice vectors shape {self.lattice_vectors.shape} does not match "
                f"expected ({dimensionality}, {dimensionality})"
            )
        if self.basis_coords.shape[1] != dimensionality:
            raise ValueError(
                f"Basis coordinates dimension {self.basis_coords.shape[1]} does not "
                f"match lattice dimensionality {dimensionality}"
            )
        if len(size) != dimensionality:
            raise ValueError(
                f"Size tuple length {len(size)} does not match dimensionality {dimensionality}"
            )

        self.num_basis = self.basis_coords.shape[0]
        self.size = size
        if isinstance(pbc, bool):
            self.pbc = tuple([pbc] * dimensionality)
        else:
            if len(pbc) != dimensionality:
                raise ValueError(
                    f"PBC tuple length {len(pbc)} does not match dimensionality {dimensionality}"
                )
            self.pbc = tuple(pbc)

        self._build_lattice()
        if precompute_neighbors is not None and precompute_neighbors > 0:
            logger.info(f"Pre-computing neighbors up to k={precompute_neighbors}...")
            self._build_neighbors(max_k=precompute_neighbors)

    def _build_lattice(self) -> None:
        """
        Generates all site information for the periodic lattice in a vectorized manner.
        """
        ranges = [backend.arange(s) for s in self.size]

        # Generate a grid of all integer unit cell coordinates.
        grid = backend.meshgrid(*ranges, indexing="ij")
        all_cell_coords = backend.reshape(
            backend.stack(grid, axis=-1), (-1, self.dimensionality)
        )

        all_cell_coords = backend.cast(all_cell_coords, self.lattice_vectors.dtype)

        cell_vectors = backend.tensordot(
            all_cell_coords, self.lattice_vectors, axes=[[1], [0]]
        )

        cell_vectors = backend.cast(cell_vectors, self.basis_coords.dtype)

        # Combine cell vectors with basis coordinates to get all site positions
        # via broadcasting: (num_cells, 1, D) + (1, num_basis, D) -> (num_cells, num_basis, D)
        all_coords = backend.expand_dims(cell_vectors, 1) + backend.expand_dims(
            self.basis_coords, 0
        )

        self._coordinates = backend.reshape(all_coords, (-1, self.dimensionality))

        self._indices = []
        self._identifiers = []
        self._ident_to_idx = {}
        current_index = 0

        # Generate integer indices and tuple-based identifiers for all sites.
        # e.g., identifier = (uc_x, uc_y, basis_idx)
        size_ranges = [range(s) for s in self.size]
        for cell_coord_tuple in itertools.product(*size_ranges):
            for basis_index in range(self.num_basis):
                identifier = cell_coord_tuple + (basis_index,)
                self._indices.append(current_index)
                self._identifiers.append(identifier)
                self._ident_to_idx[identifier] = current_index
                current_index += 1

    def _get_distance_matrix_with_mic_vectorized(self) -> Coordinates:
        """
        Computes the full N x N distance matrix using a fully vectorized approach
        that correctly applies the Minimum Image Convention (MIC) for periodic
        boundary conditions.

        This method uses full vectorization for optimal performance and compatibility
        with JIT compilation frameworks like JAX. The implementation processes all
        site pairs simultaneously rather than iterating row-by-row, which provides:

        - Better performance through vectorized operations
        - Full compatibility with automatic differentiation
        - JIT compilation support (e.g., JAX, TensorFlow)
        - Consistent tensor operations throughout

        The trade-off is higher memory usage compared to iterative approaches,
        as it computes all pairwise distances simultaneously. For very large
        lattices (N > 10^4 sites), memory usage scales as O(N^2).

        :return: Distance matrix with shape (N, N) where entry (i,j) is the
            minimum distance between sites i and j under periodic boundary conditions.
        :rtype: Coordinates
        """
        # Ensure dtype consistency across backends (especially torch) by explicitly
        # casting size and lattice_vectors to the same floating dtype used internally.
        # Strategy: prefer existing lattice_vectors dtype; if it's an unusual dtype,
        # fall back to float32 to avoid mixed-precision issues in vectorized ops.
        # Note: `self.lattice_vectors` is always created via `backend.convert_to_tensor`
        # in __init__, so `backend.dtype(...)` is reliable here and doesn't need try/except.
        target_dt = str(backend.dtype(self.lattice_vectors))  # type: ignore
        if target_dt not in ("float32", "float64"):
            # fallback for unusual dtypes
            target_dt = "float32"

        size_arr = backend.cast(backend.convert_to_tensor(self.size), target_dt)
        lattice_vecs = backend.cast(
            backend.convert_to_tensor(self.lattice_vectors), target_dt
        )
        system_vectors = lattice_vecs * backend.expand_dims(size_arr, axis=1)

        pbc_mask = backend.convert_to_tensor(self.pbc)

        # Generate all 3^d possible image shifts (-1, 0, 1) for all dimensions
        shift_options = [
            backend.convert_to_tensor([-1.0, 0.0, 1.0])
        ] * self.dimensionality
        shifts_grid = backend.meshgrid(*shift_options, indexing="ij")
        all_shifts = backend.reshape(
            backend.stack(shifts_grid, axis=-1), (-1, self.dimensionality)
        )

        # Only apply shifts to periodic dimensions
        masked_shifts = all_shifts * backend.cast(pbc_mask, all_shifts.dtype)

        # Calculate all translation vectors due to PBC
        translations_arr = backend.tensordot(
            masked_shifts, system_vectors, axes=[[1], [0]]
        )

        # Vectorized computation of all displacements between any two sites
        # Shape: (N, 1, D) - (1, N, D) -> (N, N, D)
        displacements = backend.expand_dims(self._coordinates, 1) - backend.expand_dims(
            self._coordinates, 0
        )

        # Consider all periodic images for each displacement
        # Shape: (N, N, 1, D) - (1, 1, num_translations, D) -> (N, N, num_translations, D)
        image_displacements = backend.expand_dims(
            displacements, 2
        ) - backend.expand_dims(backend.expand_dims(translations_arr, 0), 0)

        # Sum of squares for distances
        image_d_sq = backend.sum(image_displacements**2, axis=3)

        # Find the minimum distance among all images (Minimum Image Convention)
        min_dist_sq = backend.min(image_d_sq, axis=2)

        safe_dist_matrix_sq = backend.where(min_dist_sq > 0, min_dist_sq, 0.0)
        return backend.sqrt(safe_dist_matrix_sq)

    def _build_neighbors(self, max_k: int = 2, **kwargs: Any) -> None:
        """Calculates neighbor relationships for the periodic lattice.

        This method computes neighbor information by first calculating the full
        distance matrix using the Minimum Image Convention (MIC) to correctly
        handle periodic boundary conditions. It then identifies unique distance
        shells (e.g., nearest, next-nearest) and populates the neighbor maps
        accordingly. This approach is general and works for any periodic lattice
        geometry defined by the TILattice class.

        :param max_k: The maximum order of neighbors to compute (e.g., k=1 for
            nearest neighbors, k=2 for next-nearest, etc.). Defaults to 2.
        :type max_k: int, optional
        :param kwargs: Additional keyword arguments. May include:
            - ``tol`` (float): The numerical tolerance used to determine if two
              distances are equal when identifying shells. Defaults to 1e-6.
        """
        tol = kwargs.get("tol", 1e-6)
        dist_matrix = self._get_distance_matrix_with_mic_vectorized()
        dist_matrix_sq = dist_matrix**2
        self._distance_matrix = dist_matrix
        all_distances_sq = backend.reshape(dist_matrix_sq, [-1])
        dist_shells_sq = self._identify_distance_shells(all_distances_sq, max_k, tol)
        self._neighbor_maps = self._build_neighbor_map_from_distances(
            dist_matrix_sq, dist_shells_sq, tol
        )

    def _compute_distance_matrix(self) -> Coordinates:
        """Computes the distance matrix using the Minimum Image Convention."""
        if self.num_sites == 0:
            return backend.zeros((0, 0))
        return self._get_distance_matrix_with_mic_vectorized()


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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        """Initializes the SquareLattice."""
        dimensionality = 2
        # Define orthogonal lattice vectors for a square.
        # Avoid mixing Python floats with backend Tensors (TF would error),
        # so first convert inputs to tensors of a unified dtype, then stack.
        lc = backend.convert_to_tensor(lattice_constant)
        dt = backend.dtype(lc)
        z = backend.cast(backend.convert_to_tensor(0.0), dt)
        row1 = backend.stack([lc, z])
        row2 = backend.stack([z, lc])
        lattice_vectors = backend.stack([row1, row2])
        # A square lattice is a Bravais lattice, so it has a single-site basis.
        basis_coords = backend.stack([backend.stack([z, z])])

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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        """Initializes the HoneycombLattice."""
        dimensionality = 2
        a = lattice_constant
        a_t = backend.convert_to_tensor(a)
        zero = a_t * 0.0

        # Define the two primitive lattice vectors for the underlying triangular Bravais lattice.
        rt3_over_2 = math.sqrt(3.0) / 2.0
        lattice_vectors = backend.stack(
            [
                backend.stack([a_t * 1.5, a_t * rt3_over_2]),
                backend.stack([a_t * 1.5, -a_t * rt3_over_2]),
            ]
        )
        # Define the two basis sites (A and B) within the unit cell.
        basis_coords = backend.stack(
            [backend.stack([zero, zero]), backend.stack([a_t * 1.0, zero])]
        )

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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        """Initializes the TriangularLattice."""
        dimensionality = 2
        a = lattice_constant
        a_t = backend.convert_to_tensor(a)
        zero = a_t * 0.0

        # Define the primitive lattice vectors for a triangular lattice.
        lattice_vectors = backend.stack(
            [
                backend.stack([a_t * 1.0, zero]),
                backend.stack(
                    [
                        a_t * 0.5,
                        a_t * backend.sqrt(backend.convert_to_tensor(3.0)) / 2.0,
                    ]
                ),
            ]
        )
        # A triangular lattice is a Bravais lattice with a single-site basis.
        basis_coords = backend.stack([backend.stack([zero, zero])])

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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: bool = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 1
        # The lattice vector is just the lattice constant along one dimension.
        lc = backend.convert_to_tensor(lattice_constant)
        lattice_vectors = backend.stack([backend.stack([lc])])
        # A simple chain is a Bravais lattice with a single-site basis.
        zero = lc * 0.0
        basis_coords = backend.stack([backend.stack([zero])])

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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: bool = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 1
        # The unit cell is twice the bond length, as it contains two sites.
        lc = backend.convert_to_tensor(lattice_constant)
        lattice_vectors = backend.stack([backend.stack([2 * lc])])
        # Two basis sites (A and B) separated by the bond length.
        zero = lc * 0.0
        basis_coords = backend.stack([backend.stack([zero]), backend.stack([lc])])

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
        lattice_constants: Union[Tuple[float, float], Any] = (1.0, 1.0),
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 2
        ax, ay = lattice_constants
        ax_t = backend.convert_to_tensor(ax)
        dt = backend.dtype(ax_t)
        ay_t = backend.cast(backend.convert_to_tensor(ay), dt)
        z = backend.cast(backend.convert_to_tensor(0.0), dt)
        # Orthogonal lattice vectors with potentially different lengths.
        row1 = backend.stack([ax_t, z])
        row2 = backend.stack([z, ay_t])
        lattice_vectors = backend.stack([row1, row2])
        # A rectangular lattice is a Bravais lattice with a single-site basis.
        basis_coords = backend.stack([backend.stack([z, z])])

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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 2
        a = lattice_constant
        a_t = backend.convert_to_tensor(a)
        # The unit cell is a square rotated by 45 degrees.
        lattice_vectors = backend.stack(
            [
                backend.stack([a_t * 1.0, a_t * 1.0]),
                backend.stack([a_t * 1.0, a_t * -1.0]),
            ]
        )
        # Two basis sites (A and B) within the unit cell.
        zero = a_t * 0.0
        basis_coords = backend.stack(
            [backend.stack([zero, zero]), backend.stack([a_t * 1.0, zero])]
        )

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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 2
        a = lattice_constant
        a_t = backend.convert_to_tensor(a)
        # The Kagome lattice is based on a triangular Bravais lattice.
        lattice_vectors = backend.stack(
            [
                backend.stack([a_t * 2.0, a_t * 0.0]),
                backend.stack([a_t * 1.0, a_t * backend.sqrt(3.0)]),
            ]
        )
        # It has a three-site basis, forming the corners of the triangles.
        zero = a_t * 0.0
        basis_coords = backend.stack(
            [
                backend.stack([zero, zero]),
                backend.stack([a_t * 1.0, zero]),
                backend.stack([a_t * 0.5, a_t * backend.sqrt(3.0) / 2.0]),
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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: Union[bool, Tuple[bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        """Initializes the LiebLattice."""
        dimensionality = 2
        bond_length = lattice_constant
        bl_t = backend.convert_to_tensor(bond_length)
        unit_cell_side_t = 2 * bl_t
        # The Lieb lattice is based on a square Bravais lattice.
        z = bl_t * 0.0
        lattice_vectors = backend.stack(
            [backend.stack([unit_cell_side_t, z]), backend.stack([z, unit_cell_side_t])]
        )
        # It has a three-site basis: one corner and two edge-centers.
        basis_coords = backend.stack(
            [
                backend.stack([z, z]),  # Corner site
                backend.stack([bl_t, z]),  # x-edge center
                backend.stack([z, bl_t]),  # y-edge center
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
        lattice_constant: Union[float, Any] = 1.0,
        pbc: Union[bool, Tuple[bool, bool, bool]] = True,
        precompute_neighbors: Optional[int] = None,
    ):
        dimensionality = 3
        a = lattice_constant
        a_t = backend.convert_to_tensor(a)
        # Orthogonal lattice vectors of equal length in 3D.
        z = a_t * 0.0
        lattice_vectors = backend.stack(
            [
                backend.stack([a_t, z, z]),
                backend.stack([z, a_t, z]),
                backend.stack([z, z, a_t]),
            ]
        )
        # A simple cubic lattice is a Bravais lattice with a single-site basis.
        basis_coords = backend.stack([backend.stack([z, z, z])])
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
        coordinates: Any,
        precompute_neighbors: Optional[int] = None,
    ):
        """Initializes the CustomizeLattice."""
        super().__init__(dimensionality)

        self._coordinates = backend.convert_to_tensor(coordinates)
        if len(identifiers) == 0:
            self._coordinates = backend.reshape(
                self._coordinates, (0, self.dimensionality)
            )

        if len(identifiers) != backend.shape_tuple(self._coordinates)[0]:
            raise ValueError(
                "The number of identifiers must match the number of coordinates. "
                f"Got {len(identifiers)} identifiers and "
                f"{backend.shape_tuple(self._coordinates)[0]} coordinates."
            )

        self._identifiers = list(identifiers)
        self._indices = list(range(len(identifiers)))
        self._ident_to_idx = {ident: idx for idx, ident in enumerate(identifiers)}

        if (
            self.num_sites > 0
            and backend.shape_tuple(self._coordinates)[1] != dimensionality
        ):
            raise ValueError(
                f"Coordinates tensor has dimension {backend.shape_tuple(self._coordinates)[1]}, "
                f"but expected dimensionality is {dimensionality}."
            )

        logger.info(f"CustomizeLattice with {self.num_sites} sites created.")

        if precompute_neighbors is not None and precompute_neighbors > 0:
            self._build_neighbors(max_k=precompute_neighbors)

    def _build_lattice(self, *args: Any, **kwargs: Any) -> None:
        """For CustomizeLattice, lattice data is built during __init__."""
        pass

    def _build_neighbors(self, max_k: int = 1, **kwargs: Any) -> None:
        """
        Calculates neighbor relationships using either KDTree or distance matrix methods.

        This method supports two modes:
        1. KDTree mode (use_kdtree=True): Fast, O(N log N) performance for large lattices
           but breaks differentiability due to scipy dependency
        2. Distance matrix mode (use_kdtree=False): Slower O(N²) but fully differentiable
           and backend-agnostic

        :param max_k: Maximum number of neighbor shells to compute
        :type max_k: int
        :param kwargs: Additional arguments including:
            - use_kdtree (bool): Whether to use KDTree optimization. Defaults to False.
            - tol (float): Distance tolerance for neighbor identification. Defaults to 1e-6.
        """
        tol = kwargs.get("tol", 1e-6)
        # Reviewer suggestion: prefer differentiable method by default
        use_kdtree = kwargs.get("use_kdtree", False)

        if self.num_sites < 2:
            return

        # Choose algorithm based on user preference
        if use_kdtree:
            logger.info(
                f"Using KDTree method for {self.num_sites} sites up to k={max_k}"
            )
            self._build_neighbors_kdtree(max_k, tol)
        else:
            logger.info(
                f"Using differentiable distance matrix method for {self.num_sites} sites up to k={max_k}"
            )

            # Use the existing distance matrix method
            self._build_neighbors_by_distance_matrix(max_k, tol)

    def _build_neighbors_kdtree(self, max_k: int, tol: float) -> None:
        """
        Build neighbors using KDTree for optimal performance.

        This method provides O(N log N) performance for neighbor finding but breaks
        differentiability due to scipy dependency. Use this method when:
        - Performance is critical
        - Differentiability is not required
        - Large lattices (N > 1000)

        Note: This method uses numpy arrays directly and may not be compatible
        with all backend types (JAX, TensorFlow, etc.).
        """

        # For small lattices or cases with potential duplicate coordinates,
        # fall back to distance matrix method for robustness
        if self.num_sites < 200:
            logger.info(
                "Small lattice detected, falling back to distance matrix method for robustness"
            )
            self._build_neighbors_by_distance_matrix(max_k, tol)
            return

        # Convert coordinates to numpy for KDTree
        coords_np = backend.numpy(self._coordinates)

        # Build KDTree
        logger.info("Building KDTree...")
        tree = KDTree(coords_np)
        # Find all distances for shell identification - use comprehensive sampling
        logger.info("Identifying distance shells...")
        distances_for_shells: List[float] = []

        # For robust shell identification, query all pairwise distances for smaller lattices
        # or use dense sampling for larger ones
        if self.num_sites <= 100:
            # For small lattices, compute all pairwise distances for accuracy
            for i in range(self.num_sites):
                query_k = min(self.num_sites - 1, max_k * 20)
                if query_k > 0:
                    dists, _ = tree.query(
                        coords_np[i], k=query_k + 1
                    )  # +1 to exclude self
                    if isinstance(dists, np.ndarray):
                        distances_for_shells.extend(dists[1:])  # Skip distance to self
                    else:
                        distances_for_shells.append(dists)  # Single distance
        else:
            # For larger lattices, use adaptive sampling but ensure we capture all shells
            sample_size = min(1000, self.num_sites // 2)  # More conservative sampling
            for i in range(0, self.num_sites, max(1, self.num_sites // sample_size)):
                query_k = min(max_k * 20 + 50, self.num_sites - 1)
                if query_k > 0:
                    dists, _ = tree.query(
                        coords_np[i], k=query_k + 1
                    )  # +1 to exclude self
                    if isinstance(dists, np.ndarray):
                        distances_for_shells.extend(dists[1:])  # Skip distance to self
                    else:
                        distances_for_shells.append(dists)  # Single distance

        # Filter out zero distances (duplicate coordinates) before shell identification
        ZERO_THRESHOLD = 1e-12
        distances_for_shells = [d for d in distances_for_shells if d > ZERO_THRESHOLD]

        if not distances_for_shells:
            logger.warning("No valid distances found for shell identification")
            self._neighbor_maps = {}
            return

        # Use the same shell identification logic as distance matrix method
        distances_for_shells_sq = [d * d for d in distances_for_shells]
        dist_shells_sq = self._identify_distance_shells(
            distances_for_shells_sq, max_k, tol
        )
        dist_shells = [np.sqrt(d_sq) for d_sq in dist_shells_sq]

        logger.info(f"Found {len(dist_shells)} distance shells: {dist_shells[:5]}...")

        # Initialize neighbor maps
        self._neighbor_maps = {k: {} for k in range(1, len(dist_shells) + 1)}

        # Build neighbor lists for each site
        for i in range(self.num_sites):
            # Query enough neighbors to capture all shells
            query_k = min(max_k * 20 + 50, self.num_sites - 1)
            if query_k > 0:
                distances, indices = tree.query(
                    coords_np[i], k=query_k + 1
                )  # +1 for self

                # Skip the first entry (distance to self)
                # Handle both single value and array cases
                if isinstance(distances, np.ndarray) and len(distances) > 1:
                    distances_slice = distances[1:]
                    indices_slice = (
                        indices[1:]
                        if isinstance(indices, np.ndarray)
                        else np.array([], dtype=int)
                    )
                else:
                    # Single value or empty case - no neighbors to process
                    distances_slice = np.array([])
                    indices_slice = np.array([], dtype=int)

                # Filter out zero distances (duplicate coordinates)
                valid_pairs = [
                    (d, idx)
                    for d, idx in zip(distances_slice, indices_slice)
                    if d > ZERO_THRESHOLD
                ]

                # Assign neighbors to shells
                for shell_idx, shell_dist in enumerate(dist_shells):
                    k = shell_idx + 1
                    shell_neighbors = []

                    for dist, neighbor_idx in valid_pairs:
                        if abs(dist - shell_dist) <= tol:
                            shell_neighbors.append(int(neighbor_idx))
                        elif dist > shell_dist + tol:
                            break  # Distances are sorted, no more matches

                    if shell_neighbors:
                        self._neighbor_maps[k][i] = sorted(shell_neighbors)

        # Set distance matrix to None - will compute on demand
        self._distance_matrix = None

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
        _, identifiers, _ = zip(*all_sites_info)

        # Detach-and-copy coordinates while remaining in tensor form to avoid
        # host roundtrips and device/dtype changes; this keeps CustomizeLattice
        # decoupled from the original graph but backend-friendly.
        # Some backends (e.g., NumPy) don't implement stop_gradient; fall back.
        try:
            coords_detached = backend.stop_gradient(lattice._coordinates)
        except NotImplementedError:
            coords_detached = lattice._coordinates
        coords_tensor = backend.copy(coords_detached)

        return cls(
            dimensionality=lattice.dimensionality,
            identifiers=list(identifiers),
            coordinates=coords_tensor,
        )

    def add_sites(
        self,
        identifiers: List[SiteIdentifier],
        coordinates: Any,
    ) -> None:
        """Adds new sites to the lattice.

        This operation modifies the lattice in-place. After adding sites, any
        previously computed neighbor information is cleared and must be
        recalculated.

        :param identifiers: A list of unique identifiers for the new sites.
        :type identifiers: List[SiteIdentifier]
        :param coordinates: The coordinates for the new sites. Can be a list of lists,
            a NumPy array, or a backend-compatible tensor (e.g., jax.numpy.ndarray).
        :type coordinates: Any
        """
        if not identifiers:
            return

        new_coords_tensor = backend.convert_to_tensor(coordinates)

        if len(identifiers) != backend.shape_tuple(new_coords_tensor)[0]:
            raise ValueError(
                "Identifiers and coordinates lists must have the same length."
            )

        if backend.shape_tuple(new_coords_tensor)[1] != self.dimensionality:
            raise ValueError(
                f"New coordinate tensor has dimension {backend.shape_tuple(new_coords_tensor)[1]}, "
                f"but expected dimensionality is {self.dimensionality}."
            )

        # Ensure that the new identifiers are unique and do not already exist.
        existing_ids = set(self._identifiers)
        new_ids = set(identifiers)
        if not new_ids.isdisjoint(existing_ids):
            raise ValueError(
                f"Duplicate identifiers found: {new_ids.intersection(existing_ids)}"
            )

        self._coordinates = backend.concat(
            [self._coordinates, new_coords_tensor], axis=0
        )
        self._identifiers.extend(identifiers)

        self._indices = list(range(len(self._identifiers)))
        self._ident_to_idx = {ident: idx for idx, ident in enumerate(self._identifiers)}

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
        """
        if not identifiers:
            return

        ids_to_remove = set(identifiers)
        current_ids = set(self._identifiers)
        if not ids_to_remove.issubset(current_ids):
            raise ValueError(
                f"Non-existent identifiers provided for removal: {ids_to_remove - current_ids}"
            )

        # Find the indices of the sites that we want to keep.
        indices_to_keep = [
            idx
            for idx, ident in enumerate(self._identifiers)
            if ident not in ids_to_remove
        ]

        new_identifiers = [self._identifiers[i] for i in indices_to_keep]

        self._coordinates = backend.gather1d(
            self._coordinates,
            backend.cast(backend.convert_to_tensor(indices_to_keep), "int32"),
        )

        self._identifiers = new_identifiers

        self._indices = list(range(len(self._identifiers)))
        self._ident_to_idx = {ident: idx for idx, ident in enumerate(self._identifiers)}

        self._reset_computations()
        logger.info(
            f"{len(ids_to_remove)} sites removed. Lattice now has {self.num_sites} sites."
        )


def get_compatible_layers(bonds: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """
    Partitions a list of pairs (bonds) into compatible layers for parallel
    gate application using a greedy edge-coloring algorithm.

    This function takes a list of pairs, representing connections like
    nearest-neighbor (NN) or next-nearest-neighbor (NNN) bonds, and
    partitions them into the minimum number of sets ("layers") where no two
    pairs in a set share an index. This is a general utility for scheduling
    non-overlapping operations.

    :Example:

    >>> from tensorcircuit.templates.lattice import SquareLattice
    >>> sq_lattice = SquareLattice(size=(2, 2), pbc=False)
    >>> nn_bonds = sq_lattice.get_neighbor_pairs(k=1, unique=True)

    >>> gate_layers = get_compatible_layers(nn_bonds)
    >>> print(gate_layers)
    [[[0, 1], [2, 3]], [[0, 2], [1, 3]]]

    :param bonds: A list of tuples, where each tuple represents a bond (i, j)
        of site indices to be scheduled.
    :type bonds: List[Tuple[int, int]]
    :return: A list of layers. Each layer is a list of tuples, where each
        tuple represents a bond. All bonds within a layer are non-overlapping.
    :rtype: List[List[Tuple[int, int]]]
    """
    uncolored_edges: Set[Tuple[int, int]] = {(min(bond), max(bond)) for bond in bonds}

    layers: List[List[Tuple[int, int]]] = []

    while uncolored_edges:
        current_layer: List[Tuple[int, int]] = []
        qubits_in_this_layer: Set[int] = set()

        edges_to_process = sorted(list(uncolored_edges))

        for edge in edges_to_process:
            i, j = edge
            if i not in qubits_in_this_layer and j not in qubits_in_this_layer:
                current_layer.append(edge)
                qubits_in_this_layer.add(i)
                qubits_in_this_layer.add(j)

        uncolored_edges -= set(current_layer)
        layers.append(sorted(current_layer))

    return layers
