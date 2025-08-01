from unittest.mock import patch
import logging

# import time

import matplotlib

matplotlib.use("Agg")


import pytest
import numpy as np

from tensorcircuit.templates.lattice import (
    ChainLattice,
    CheckerboardLattice,
    CubicLattice,
    CustomizeLattice,
    DimerizedChainLattice,
    HoneycombLattice,
    KagomeLattice,
    LiebLattice,
    RectangularLattice,
    SquareLattice,
    TriangularLattice,
    AbstractLattice,
    get_compatible_layers,
)


@pytest.fixture
def simple_square_lattice() -> CustomizeLattice:
    """
    Provides a simple 2x2 square CustomizeLattice instance for neighbor tests.
    The sites are indexed as follows:
    2--3
    |  |
    0--1
    """
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    ids = list(range(len(coords)))
    lattice = CustomizeLattice(dimensionality=2, identifiers=ids, coordinates=coords)
    # Pre-calculate neighbors up to the 2nd shell for use in tests.
    lattice._build_neighbors(max_k=2)
    return lattice


@pytest.fixture
def kagome_lattice_fragment() -> CustomizeLattice:
    """
    Pytest fixture to provide a standard CustomizeLattice instance.
    This represents the Kagome fragment from the project requirements,
    making it a reusable object for multiple tests.
    """
    kag_coords = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],  # Triangle 1
        [2, 0],
        [1.5, np.sqrt(3) / 2],  # Triangle 2 (shifted basis)
        [1.0, np.sqrt(3)],  # Top site
    ]
    kag_ids = list(range(len(kag_coords)))
    return CustomizeLattice(
        dimensionality=2, identifiers=kag_ids, coordinates=kag_coords
    )


class TestCustomizeLattice:
    """
    A test class to group all tests related to the CustomizeLattice.
    This helps in organizing the test suite.
    """

    def test_initialization_and_properties(self, kagome_lattice_fragment):
        """
        Test case for successful initialization and verification of basic properties.
        This test function receives the 'kagome_lattice_fragment' fixture as an argument.
        """
        # Arrange: The fixture has already prepared the 'lattice' object for us.
        lattice = kagome_lattice_fragment

        # Assert: Check if the object's properties match our expectations.
        assert lattice.dimensionality == 2
        assert lattice.num_sites == 6
        assert len(lattice) == 6  # This also tests the __len__ dunder method

        # Verify that coordinates are correctly stored as numpy arrays.
        # It's important to use np.testing.assert_array_equal for numpy array comparison.
        expected_coord = np.array([0.5, np.sqrt(3) / 2])
        np.testing.assert_array_equal(lattice.get_coordinates(2), expected_coord)

        # Verify that the mapping between identifiers and indices is correct.
        assert lattice.get_identifier(4) == 4
        assert lattice.get_index(4) == 4

    def test_input_validation_mismatched_lengths(self):
        """
        Tests that a ValueError is raised if identifiers and coordinates
        lists have mismatched lengths.
        """
        # Arrange: Prepare invalid inputs.
        coords = [[0.0, 0.0], [1.0, 0.0]]  # 2 coordinates
        ids = [0, 1, 2]  # 3 identifiers

        # Act & Assert: Use pytest.raises as a context manager to ensure
        # the specified exception is raised within the 'with' block.
        with pytest.raises(
            ValueError,
            match="Identifiers and coordinates lists must have the same length.",
        ):
            CustomizeLattice(dimensionality=2, identifiers=ids, coordinates=coords)

    def test_input_validation_wrong_dimension(self):
        """
        Tests that a ValueError is raised if a coordinate's dimension
        does not match the lattice's specified dimensionality.
        """
        # Arrange: Prepare coordinates with mixed dimensions for a 2D lattice.
        coords_wrong_dim = [[0.0, 0.0], [1.0, 0.0, 0.0]]  # A mix of 2D and 3D
        ids_ok = [0, 1]

        # Act & Assert: Check for the specific error message. The 'r' before the string
        # indicates a raw string, which is good practice for regex patterns.
        with pytest.raises(
            ValueError, match=r"Coordinate at index 1 has shape \(3,\), expected \(2,\)"
        ):
            CustomizeLattice(
                dimensionality=2, identifiers=ids_ok, coordinates=coords_wrong_dim
            )

    def test_neighbor_finding(self, simple_square_lattice):
        """
        Tests the k-th nearest neighbor finding functionality (_build_neighbors
        and get_neighbors).
        """
        # Arrange: The fixture provides the lattice with pre-built neighbors.
        lattice = simple_square_lattice

        # --- Assertions for k=1 (Nearest Neighbors) ---
        # We use set() for comparison to ignore the order of neighbors.
        assert set(lattice.get_neighbors(0, k=1)) == {1, 2}
        assert set(lattice.get_neighbors(1, k=1)) == {0, 3}
        assert set(lattice.get_neighbors(2, k=1)) == {0, 3}
        assert set(lattice.get_neighbors(3, k=1)) == {1, 2}

        # --- Assertions for k=2 (Next-Nearest Neighbors) ---
        # These should be the diagonal sites.
        assert set(lattice.get_neighbors(0, k=2)) == {3}
        assert set(lattice.get_neighbors(1, k=2)) == {2}

    def test_neighbor_pairs(self, simple_square_lattice):
        """
        Tests the retrieval of unique neighbor pairs (bonds) using
        get_neighbor_pairs.
        """
        # Arrange: Use the same fixture.
        lattice = simple_square_lattice

        # --- Test for k=1 (Nearest Neighbor bonds) ---
        # Act: Get unique nearest neighbor pairs.
        nn_pairs = lattice.get_neighbor_pairs(k=1, unique=True)

        # Assert: The set of pairs should match the expected bonds.
        # We convert the list of pairs to a set of tuples for order-independent comparison.
        expected_nn_pairs = {(0, 1), (0, 2), (1, 3), (2, 3)}
        assert set(map(tuple, nn_pairs)) == expected_nn_pairs

        # --- Test for k=2 (Next-Nearest Neighbor bonds) ---
        # Act: Get unique next-nearest neighbor pairs.
        nnn_pairs = lattice.get_neighbor_pairs(k=2, unique=True)

        # Assert:
        expected_nnn_pairs = {(0, 3), (1, 2)}
        assert set(map(tuple, nnn_pairs)) == expected_nnn_pairs

    def test_neighbor_pairs_non_unique(self, simple_square_lattice):
        """
        Tests get_neighbor_pairs with unique=False to ensure all
        directed pairs (bonds) are returned.
        """
        # Arrange: Use the same 2x2 square lattice fixture.
        # 2--3
        # |  |
        # 0--1
        lattice = simple_square_lattice

        # Act: Get NON-unique nearest neighbor pairs.
        nn_pairs = lattice.get_neighbor_pairs(k=1, unique=False)

        # Assert:
        # There are 4 bonds, so we expect 4 * 2 = 8 directed pairs.
        assert len(nn_pairs) == 8

        # Your source code sorts the output, so we can compare against a
        # sorted list for a precise match.
        expected_pairs = sorted(
            [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)]
        )

        assert nn_pairs == expected_pairs

    @patch("matplotlib.pyplot.show")
    def test_show_method_runs_and_calls_plt_show(
        self, mock_show, simple_square_lattice
    ):
        """
        Smoke test for the .show() method.
        It verifies that the method runs without raising an exception and that it
        triggers a call to matplotlib's show() function.
        We use @patch to "mock" the show function, preventing a plot window
        from actually appearing during tests.
        """
        # Arrange: Get the lattice instance from the fixture
        lattice = simple_square_lattice

        # Act: Call the .show() method.
        # We wrap it in a try...except block to give a more specific error
        # if the method fails for any reason.
        try:
            lattice.show()
        except Exception as e:
            pytest.fail(f".show() method raised an unexpected exception: {e}")

        # Assert: Check that our mocked matplotlib.pyplot.show was called exactly once.
        mock_show.assert_called_once()

    def test_sites_iterator(self, simple_square_lattice):
        """
        Tests the sites() iterator to ensure it yields all sites correctly.
        """
        # Arrange
        lattice = simple_square_lattice
        expected_num_sites = 4

        # Act
        # The sites() method returns an iterator, we convert it to a list to check its length.
        all_sites = list(lattice.sites())

        # Assert
        assert len(all_sites) == expected_num_sites

        # For a more thorough check, verify the content of one of the yielded tuples.
        # For the simple_square_lattice fixture, site 3 has identifier 3 and coords [1, 1].
        idx, ident, coords = all_sites[3]
        assert idx == 3
        assert ident == 3
        np.testing.assert_array_equal(coords, np.array([1, 1]))

    def test_get_site_info_with_identifier(self, simple_square_lattice):
        """
        Tests the get_site_info() method using a site identifier instead of an index.
        This covers the 'else' branch of the type check in the method.
        """
        # Arrange
        lattice = simple_square_lattice
        # In this fixture, the identifier for the site at index 2 is also the integer 2.
        identifier_to_test = 2
        expected_index = 2
        expected_coords = np.array([0, 1])

        # Act
        idx, ident, coords = lattice.get_site_info(identifier_to_test)

        # Assert
        assert idx == expected_index
        assert ident == identifier_to_test
        np.testing.assert_array_equal(coords, expected_coords)

    @patch("matplotlib.pyplot.show")
    def test_show_method_with_labels(self, mock_show, simple_square_lattice):
        """
        Tests that the .show() method runs without error when label-related
        options are enabled. This covers the logic inside the
        'if show_indices or show_identifiers:' block.
        """
        # Arrange
        lattice = simple_square_lattice

        # Act & Assert
        try:
            # Call .show() with options to display indices and identifiers.
            lattice.show(show_indices=True, show_identifiers=True)
        except Exception as e:
            pytest.fail(
                f".show() with label options raised an unexpected exception: {e}"
            )

        # Ensure the plotting function is still called.
        mock_show.assert_called_once()

    def test_get_neighbors_logs_info_for_uncached_k(
        self, simple_square_lattice, caplog
    ):
        """
        Tests that an INFO message is logged when get_neighbors is called for a 'k'
        that has not been pre-calculated, triggering on-demand computation.
        """
        # Arrange
        lattice = simple_square_lattice  # This fixture builds neighbors up to k=2
        k_to_test = 99  # A value that is clearly not cached
        caplog.set_level(logging.INFO)  # Ensure INFO logs are captured

        # Act
        # This will now trigger the on-demand computation
        _ = lattice.get_neighbors(0, k=k_to_test)

        # Assert
        # Check that the correct INFO message about on-demand building was logged.
        expected_log = (
            f"Neighbors for k={k_to_test} not pre-computed. "
            f"Building now up to max_k={k_to_test}."
        )
        assert expected_log in caplog.text

    @patch("matplotlib.pyplot.show")
    def test_show_prints_warning_for_uncached_bonds(
        self, mock_show, simple_square_lattice, caplog
    ):
        """
        Tests that a warning is printed when .show() is asked to draw a bond layer 'k'
        that has not been pre-calculated.
        """
        # Arrange
        lattice = simple_square_lattice  # This fixture builds neighbors up to k=2
        k_to_test = 99  # A value that is clearly not cached

        # Act
        lattice.show(show_bonds_k=k_to_test)

        # Assert
        assert (
            f"Cannot draw bonds. k={k_to_test} neighbors have not been calculated"
            in caplog.text
        )

    @patch("matplotlib.pyplot.show")
    def test_show_method_for_3d_lattice(self, mock_show):
        """
        Tests that the .show() method can handle a 3D lattice without
        crashing. This covers the 'if self.dimensionality == 3:' branches.
        """
        # Arrange: Create a simple 2-site lattice in 3D space.
        coords_3d = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        ids_3d = [0, 1]
        lattice_3d = CustomizeLattice(
            dimensionality=3, identifiers=ids_3d, coordinates=coords_3d
        )

        # Assert basic property
        assert lattice_3d.dimensionality == 3

        # Act & Assert
        # We just need to ensure that calling .show() on a 3D object
        # executes the 3D plotting logic without raising an exception.
        try:
            lattice_3d.show(show_indices=True, show_bonds_k=None)
        except Exception as e:
            pytest.fail(f".show() for 3D lattice raised an unexpected exception: {e}")

        # Verify that the plotting pipeline was completed.
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_show_method_actually_draws_2d_labels(
        self, mock_subplots, simple_square_lattice
    ):
        """
        Tests if ax.text is actually called for a 2D lattice when labels are enabled.
        """
        # Arrange:
        # 1. Prepare mock Figure and Axes objects that `matplotlib.pyplot.subplots` will return.
        # This allows us to inspect calls to the `ax` object.
        mock_fig = matplotlib.figure.Figure()
        mock_ax = matplotlib.axes.Axes(mock_fig, [0.0, 0.0, 1.0, 1.0])
        mock_subplots.return_value = (mock_fig, mock_ax)

        # 2. Mock the text method on our mock Axes object to monitor its calls.
        with patch.object(mock_ax, "text") as mock_text_method:
            lattice = simple_square_lattice

            # Act:
            # Call the show method. It will now operate on our mock_ax object.
            lattice.show(show_indices=True)

            # Assert:
            # Check if the ax.text method was called. For a 4-site lattice, it should be called 4 times.
            assert mock_text_method.call_count == lattice.num_sites

    def test_custom_irregular_geometry_neighbors(self):
        """
        Tests neighbor finding on a more complex, non-grid-like custom geometry
        to stress-test the distance shell and KDTree logic.
        """
        # Arrange: A "star-shaped" lattice with a central point,
        # an inner shell, and an outer shell.
        coords = [
            [0.0, 0.0],  # Site 0: Center
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],  # Sites 1-4: Inner shell (dist=1)
            [2.0, 0.0],
            [0.0, 2.0],
            [-2.0, 0.0],
            [0.0, -2.0],  # Sites 5-8: Outer shell (dist=2)
        ]
        ids = list(range(len(coords)))
        lattice = CustomizeLattice(
            dimensionality=2, identifiers=ids, coordinates=coords
        )
        lattice._build_neighbors(max_k=3)

        # Assert 1: Neighbors of the central point (0) should be the distinct shells.
        assert set(lattice.get_neighbors(0, k=1)) == {1, 2, 3, 4}
        # The shell at dist=2.0 (d_sq=4.0) is the 3rd global shell, so we check k=3.
        assert set(lattice.get_neighbors(0, k=3)) == {5, 6, 7, 8}

        assert lattice.get_neighbors(0, k=2) == []

        # Assert 2: Neighbors of a point on the inner shell, e.g., site 1 ([1.0, 0.0]).
        # Its nearest neighbors (k=1) are the center (0) and the closest point on the outer shell (5).
        # Both are at distance 1.0.
        assert set(lattice.get_neighbors(1, k=1)) == {0, 5}

        # Its next-nearest neighbors (k=2) are the other two points on the inner shell (2 and 4),
        # both at distance sqrt(2).
        assert set(lattice.get_neighbors(1, k=2)) == {2, 4}

    def test_customizelattice_max_k_precomputation_and_ondemand(self):
        """
        A robust test to verify `precompute_neighbors` (max_k) for CustomizeLattice.
        This test is designed to FAIL on the buggy code.
        """
        coords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [-2.0, 0.0],
            [0.0, -2.0],
        ]
        ids = list(range(len(coords)))
        k_precompute = 2

        lattice = CustomizeLattice(
            dimensionality=2,
            identifiers=ids,
            coordinates=coords,
            precompute_neighbors=k_precompute,
        )

        computed_shells = sorted(list(lattice._neighbor_maps.keys()))
        expected_shells = list(range(1, k_precompute + 1))

        assert computed_shells == expected_shells, (
            f"TEST FAILED for CustomizeLattice with k={k_precompute}. "
            f"Expected shells {expected_shells}, but found {computed_shells}."
        )

        k_ondemand = 3
        _ = lattice.get_neighbors(0, k=k_ondemand)

        computed_shells_after = sorted(list(lattice._neighbor_maps.keys()))
        expected_shells_after = list(range(1, k_ondemand + 1))

        assert computed_shells_after == expected_shells_after, (
            f"ON-DEMAND TEST FAILED for CustomizeLattice. "
            f"Expected shells {expected_shells_after} after demanding k={k_ondemand}, "
            f"but found {computed_shells_after}."
        )


@pytest.fixture
def obc_square_lattice() -> SquareLattice:
    """Provides a 3x3 SquareLattice with Open Boundary Conditions."""
    return SquareLattice(size=(3, 3), pbc=False)


@pytest.fixture
def pbc_square_lattice() -> SquareLattice:
    """Provides a 3x3 SquareLattice with Periodic Boundary Conditions."""
    return SquareLattice(size=(3, 3), pbc=True)


class TestSquareLattice:
    """
    Groups all tests for the SquareLattice class, which implicitly tests
    the core functionality of its parent, TILattice.
    """

    def test_initialization_and_properties(self, obc_square_lattice):
        """
        Tests the basic properties of a SquareLattice instance.
        """
        lattice = obc_square_lattice
        assert lattice.dimensionality == 2
        assert lattice.num_sites == 9  # A 3x3 lattice should have 9 sites.
        assert len(lattice) == 9

    def test_site_info_and_identifiers(self, obc_square_lattice):
        """
        Tests that site information (coordinates, identifiers) is correct.
        """
        lattice = obc_square_lattice
        center_idx = lattice.get_index((1, 1, 0))
        assert center_idx == 4

        _, ident, coords = lattice.get_site_info(center_idx)
        assert ident == (1, 1, 0)
        np.testing.assert_array_equal(coords, np.array([1.0, 1.0]))

        corner_idx = 0
        _, ident, coords = lattice.get_site_info(corner_idx)
        assert ident == (0, 0, 0)
        np.testing.assert_array_equal(coords, np.array([0.0, 0.0]))

    def test_neighbors_with_open_boundaries(self, obc_square_lattice):
        """
        Tests neighbor finding with Open Boundary Conditions (OBC) using specific
        neighbor identities.
        """
        lattice = obc_square_lattice
        # Site indices for a 3x3 grid (row-major order):
        # 0 1 2
        # 3 4 5
        # 6 7 8
        center_idx = 4  # (1, 1, 0)
        corner_idx = 0  # (0, 0, 0)
        edge_idx = 3  # (1, 0, 0)

        # Assert center site (4) has neighbors 1, 3, 5, 7
        assert set(lattice.get_neighbors(center_idx, k=1)) == {1, 3, 5, 7}
        # Assert corner site (0) has neighbors 1, 3
        assert set(lattice.get_neighbors(corner_idx, k=1)) == {1, 3}
        # Assert edge site (3) has neighbors 0, 4, 6
        assert set(lattice.get_neighbors(edge_idx, k=1)) == {0, 4, 6}

    def test_neighbors_with_periodic_boundaries(self, pbc_square_lattice):
        """
        Tests neighbor finding with Periodic Boundary Conditions (PBC).
        """
        lattice = pbc_square_lattice
        corner_idx = lattice.get_index((0, 0, 0))

        neighbors = lattice.get_neighbors(corner_idx, k=1)
        neighbor_idents = {lattice.get_identifier(i) for i in neighbors}
        expected_neighbor_idents = {(1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 2, 0)}
        assert neighbor_idents == expected_neighbor_idents

        nnn_neighbors = lattice.get_neighbors(corner_idx, k=2)
        nnn_neighbor_idents = {lattice.get_identifier(i) for i in nnn_neighbors}
        expected_nnn_idents = {(1, 1, 0), (2, 1, 0), (1, 2, 0), (2, 2, 0)}
        assert nnn_neighbor_idents == expected_nnn_idents


# --- Tests for HoneycombLattice ---


@pytest.fixture
def pbc_honeycomb_lattice() -> HoneycombLattice:
    """Provides a 2x2 HoneycombLattice with Periodic Boundary Conditions."""
    return HoneycombLattice(size=(2, 2), pbc=True)


class TestHoneycombLattice:
    """
    Tests the HoneycombLattice class, focusing on its two-site basis.
    """

    def test_initialization_and_properties(self, pbc_honeycomb_lattice):
        """
        Tests that the total number of sites is correct for a composite lattice.
        """
        lattice = pbc_honeycomb_lattice
        assert lattice.num_sites == 8
        assert lattice.num_basis == 2

    def test_honeycomb_neighbors(self, pbc_honeycomb_lattice):
        """
        Tests that every site in a honeycomb lattice has 3 nearest neighbors.
        """
        lattice = pbc_honeycomb_lattice
        site_a_idx = lattice.get_index((0, 0, 0))
        assert len(lattice.get_neighbors(site_a_idx, k=1)) == 3

        site_b_idx = lattice.get_index((0, 0, 1))
        assert len(lattice.get_neighbors(site_b_idx, k=1)) == 3


# --- Tests for TriangularLattice ---


@pytest.fixture
def pbc_triangular_lattice() -> TriangularLattice:
    """
    Provides a 3x3 TriangularLattice with Periodic Boundary Conditions.
    A 3x3 size is used to ensure all 6 nearest neighbors are unique sites.
    """
    return TriangularLattice(size=(3, 3), pbc=True)


class TestTriangularLattice:
    """
    Tests the TriangularLattice class, focusing on its coordination number.
    """

    def test_initialization_and_properties(self, pbc_triangular_lattice):
        """
        Tests the basic properties of the triangular lattice.
        """
        lattice = pbc_triangular_lattice
        assert lattice.num_sites == 9  # 3 * 3 = 9 sites for a 3x3 grid

    def test_triangular_neighbors(self, pbc_triangular_lattice):
        """
        Tests that every site in a triangular lattice has 6 nearest neighbors.
        """
        lattice = pbc_triangular_lattice
        site_idx = 0
        assert len(lattice.get_neighbors(site_idx, k=1)) == 6


# --- Tests for New TILattice Implementations ---


class TestRectangularLattice:
    """Tests for the 2D RectangularLattice."""

    def test_rectangular_properties_and_neighbors(self):
        """Tests neighbor counts for an OBC rectangular lattice."""
        lattice = RectangularLattice(size=(3, 4), pbc=False)
        assert lattice.num_sites == 12
        assert lattice.dimensionality == 2

        # Test neighbor counts for different site types
        center_idx = lattice.get_index((1, 1, 0))
        corner_idx = lattice.get_index((0, 0, 0))
        edge_idx = lattice.get_index((0, 1, 0))

        assert len(lattice.get_neighbors(center_idx, k=1)) == 4
        assert len(lattice.get_neighbors(corner_idx, k=1)) == 2
        assert len(lattice.get_neighbors(edge_idx, k=1)) == 3


class TestTILatticeEdgeCases:
    """
    A dedicated class for testing the behavior of TILattice and its
    subclasses under less common, "edge-case" conditions.
    """

    @pytest.fixture
    def obc_1d_chain(self) -> ChainLattice:
        """
        Provides a 5-site 1D chain with Open Boundary Conditions.
        """
        # 0--1--2--3--4
        return ChainLattice(size=(5,), pbc=False)

    def test_1d_chain_properties_and_neighbors(self, obc_1d_chain):
        # Arrange
        lattice = obc_1d_chain

        # Assert basic properties
        assert lattice.num_sites == 5
        assert lattice.dimensionality == 1

        # Assert neighbor counts for different positions
        # Endpoint (site 0) should have 1 neighbor (site 1)
        endpoint_idx = lattice.get_index((0, 0))
        assert lattice.get_neighbors(endpoint_idx, k=1) == [1]

        # Middle point (site 2) should have 2 neighbors (sites 1 and 3)
        middle_idx = lattice.get_index((2, 0))
        assert len(lattice.get_neighbors(middle_idx, k=1)) == 2
        assert set(lattice.get_neighbors(middle_idx, k=1)) == {1, 3}

    @pytest.fixture
    def nonsquare_lattice(self) -> SquareLattice:
        """Provides a non-square 2x3 lattice to test indexing."""
        return SquareLattice(size=(2, 3), pbc=False)

    def test_nonsquare_lattice_indexing(self, nonsquare_lattice):
        """
        Tests site indexing and coordinate generation on a non-square (2x3) lattice.
        This ensures the logic correctly handles different dimension lengths.
        The lattice sites are indexed row by row:
        (0,0) (0,1) (0,2)  -> indices 0, 1, 2
        (1,0) (1,1) (1,2)  -> indices 3, 4, 5
        """
        # Arrange
        lattice = nonsquare_lattice

        # Assert properties
        assert lattice.num_sites == 6  # 2 * 3 = 6

        # Act & Assert: Check a non-trivial site, e.g., the last one.
        # The identifier for the site in the last row and last column.
        ident = (1, 2, 0)
        expected_idx = 5
        expected_coords = np.array([1.0, 2.0])

        # Get index from identifier
        idx = lattice.get_index(ident)
        assert idx == expected_idx

        # Get info from index
        _, _, coords = lattice.get_site_info(idx)
        np.testing.assert_array_equal(coords, expected_coords)

    @patch("matplotlib.pyplot.show")
    def test_show_method_for_1d_lattice(self, mock_show, obc_1d_chain):
        """
        Tests that the .show() method can handle a 1D lattice (chain)
        without crashing. This covers the 'if self.dimensionality == 1:' branches.
        """
        # Arrange
        lattice_1d = obc_1d_chain

        # Assert basic property
        assert lattice_1d.num_sites == 5

        # Act & Assert
        try:
            # Call .show() on the 1D lattice to execute the 1D plotting logic.
            lattice_1d.show(show_indices=True)
        except Exception as e:
            pytest.fail(f".show() for 1D lattice raised an unexpected exception: {e}")

        # Verify that the plotting pipeline was completed.
        mock_show.assert_called_once()


# --- Tests for API Robustness / Negative Cases ---


class TestApiRobustness:
    """
    Groups tests that verify the API's behavior with invalid inputs.
    This ensures the lattice classes fail gracefully and predictably.
    """

    def test_access_with_out_of_bounds_index(self, simple_square_lattice):
        """
        Tests that an IndexError is raised when accessing a site index
        that is out of the valid range (0 to num_sites-1).
        """
        # Arrange
        lattice = simple_square_lattice  # This lattice has 4 sites (indices 0, 1, 2, 3)
        invalid_index = 999

        # Act & Assert
        # We use pytest.raises to confirm that the expected exception is thrown.
        with pytest.raises(IndexError):
            lattice.get_coordinates(invalid_index)

        with pytest.raises(IndexError):
            lattice.get_identifier(invalid_index)

        with pytest.raises(IndexError):
            # get_site_info should also raise IndexError for an invalid index
            lattice.get_site_info(invalid_index)

    def test_empty_lattice_handles_gracefully(self, caplog):
        """
        Tests that an empty lattice initializes correctly and that methods
        like .show() and ._build_neighbors() handle the zero-site case
        gracefully without crashing.
        """
        # Arrange: Create an empty CustomizeLattice instance.
        empty_lattice = CustomizeLattice(
            dimensionality=2, identifiers=[], coordinates=[]
        )

        # Assert: Verify basic properties.
        assert empty_lattice.num_sites == 0
        assert len(empty_lattice) == 0

        # Act & Assert for .show(): Verify it prints the expected message without crashing.
        caplog.set_level(logging.INFO)

        empty_lattice.show()
        assert "Lattice is empty, nothing to show." in caplog.text

        # Act & Assert for neighbor finding: Verify these calls run without errors.
        empty_lattice._build_neighbors()
        assert empty_lattice.get_neighbor_pairs(k=1) == []

    def test_single_site_lattice_handles_gracefully(self):
        """
        Tests that a lattice with a single site correctly handles neighbor
        finding (i.e., returns no neighbors).
        """
        # Arrange: Create a CustomizeLattice with a single site.
        single_site_lattice = CustomizeLattice(
            dimensionality=2, identifiers=[0], coordinates=[[0.0, 0.0]]
        )

        # Assert: Verify basic properties.
        assert single_site_lattice.num_sites == 1

        # Act: Attempt to build neighbor relationships.
        single_site_lattice._build_neighbors(max_k=1)

        # Assert: The single site should have no neighbors.
        assert single_site_lattice.get_neighbors(0, k=1) == []

    def test_access_with_non_existent_identifier(self, simple_square_lattice):
        """
        Tests that a ValueError is raised when accessing a site
        with an identifier that does not exist in the lattice.
        """
        # Arrange
        lattice = simple_square_lattice
        invalid_identifier = "non_existent_site"

        # Act & Assert
        # Your code raises a ValueError with a specific message. We can even
        # use the 'match' parameter to check if the error message is correct.
        with pytest.raises(ValueError, match="not found in the lattice"):
            lattice.get_index(invalid_identifier)

        with pytest.raises(ValueError, match="not found in the lattice"):
            lattice.get_site_info(invalid_identifier)

    def test_show_warning_for_unsupported_dimension(self, caplog):
        """
        Tests that .show() prints a warning when called on a lattice with a
        dimensionality that it does not support for plotting (e.g., 4D).
        """
        # Arrange: Create a simple lattice with an unsupported dimension.
        lattice_4d = CustomizeLattice(
            dimensionality=4, identifiers=[0], coordinates=[[0, 0, 0, 0]]
        )

        # Act
        lattice_4d.show()

        # Assert: Check that the appropriate warning was printed to stdout.
        assert "show() is not implemented for 4D lattices." in caplog.text

    def test_disconnected_lattice_neighbor_finding(self):
        """
        Tests that neighbor finding algorithms work correctly for a lattice
        composed of multiple, physically disconnected components.
        """
        # Arrange: Create a lattice with two disconnected 2x2 squares,
        # separated by a large distance.
        # Component 1: sites with indices 0, 1, 2, 3
        # Component 2: sites with indices 4, 5, 6, 7
        coords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],  # Square 1
            [100.0, 0.0],
            [101.0, 0.0],
            [100.0, 1.0],
            [101.0, 1.0],  # Square 2
        ]
        ids = list(range(len(coords)))
        lattice = CustomizeLattice(
            dimensionality=2, identifiers=ids, coordinates=coords
        )
        lattice._build_neighbors(max_k=1)  # Explicitly build neighbors

        # --- Test 1: get_neighbors() ---
        # Act: Get neighbors for a site in the first component.
        neighbors_of_site_0 = lattice.get_neighbors(0, k=1)

        # Assert: Its neighbors must only be within the first component.
        assert set(neighbors_of_site_0) == {1, 2}

        # --- Test 2: get_neighbor_pairs() ---
        # Act: Get all unique bonds for the entire lattice.
        all_bonds = lattice.get_neighbor_pairs(k=1, unique=True)

        # Assert: No bond should connect a site from Component 1 to Component 2.
        for i, j in all_bonds:
            # A bond is valid only if both its sites are in the same component.
            # We check this by seeing if their indices fall in the same range.
            is_in_first_component = i < 4 and j < 4
            is_in_second_component = i >= 4 and j >= 4

            assert is_in_first_component or is_in_second_component, (
                f"Found an invalid bond { (i,j) } that incorrectly connects "
                "two separate components of the lattice."
            )

    def test_lattice_with_duplicate_coordinates(self):
        """
        Tests a pathological case where multiple sites share the exact same coordinates.
        The neighbor-finding logic must still treat them as distinct sites and
        correctly identify neighbors based on other non-overlapping sites.
        """
        # Arrange
        # Site 'A' and 'B' are at the same position (0,0).
        # Site 'C' is at (1,0), which should be a neighbor to both 'A' and 'B'.
        ids = ["A", "B", "C"]
        coords = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]

        lattice = CustomizeLattice(
            dimensionality=2, identifiers=ids, coordinates=coords
        )
        lattice._build_neighbors(max_k=1)  # Build nearest neighbors

        # Act
        idx_A = lattice.get_index("A")
        idx_B = lattice.get_index("B")
        idx_C = lattice.get_index("C")

        neighbors_A = lattice.get_neighbors(idx_A, k=1)
        neighbors_B = lattice.get_neighbors(idx_B, k=1)

        # Assert
        # 1. The distance between the overlapping points 'A' and 'B' is 0,
        #    so they should NOT be considered neighbors of each other.
        assert (
            idx_B not in neighbors_A
        ), "Overlapping sites should not be their own neighbors."
        assert (
            idx_A not in neighbors_B
        ), "Overlapping sites should not be their own neighbors."

        # 2. Both 'A' and 'B' should correctly identify 'C' as their neighbor.
        #    This is the key test of robustness.
        assert neighbors_A == [
            idx_C
        ], "Site 'A' failed to find its correct neighbor 'C'."
        assert neighbors_B == [
            idx_C
        ], "Site 'B' failed to find its correct neighbor 'C'."

        # 3. Conversely, 'C' should identify both 'A' and 'B' as its neighbors.
        neighbors_C = lattice.get_neighbors(idx_C, k=1)
        assert set(neighbors_C) == {
            idx_A,
            idx_B,
        }, "Site 'C' failed to find both overlapping neighbors."

    def test_neighbor_shells_with_tiny_separation(self):
        """
        Tests the numerical stability of neighbor shell identification.
        Creates a lattice where the k=1 and k=2 shells are separated by a
        distance much smaller than the default tolerance, and verifies that they
        are still correctly identified as distinct shells.
        """
        # Arrange
        # Let d1 be the distance to the first neighbor shell.
        d1 = 1.0
        # Let d2 be the distance to the second shell, which is extremely close to d1.
        epsilon = 1e-8  # A tiny separation
        d2 = d1 + epsilon

        # Create a 1D lattice with these specific distances.
        # Site 0 is origin. Site 1 is at d1. Site 2 is at d2.
        ids = [0, 1, 2]
        coords = [[0.0], [d1], [d2]]

        # We explicitly use a tolerance LARGER than the separation,
        # which SHOULD cause the shells to merge.
        lattice_merged = CustomizeLattice(
            dimensionality=1, identifiers=ids, coordinates=coords
        )
        # Use a tolerance that cannot distinguish d1 and d2.
        lattice_merged._build_neighbors(max_k=2, tol=1e-7)

        # Now, use a tolerance SMALLER than the separation,
        # which SHOULD correctly distinguish the shells.
        lattice_distinct = CustomizeLattice(
            dimensionality=1, identifiers=ids, coordinates=coords
        )
        lattice_distinct._build_neighbors(max_k=2, tol=1e-9)

        # Assert for the merged case
        # With a large tolerance, site 1 and 2 should both be in the k=1 shell.
        merged_neighbors_k1 = lattice_merged.get_neighbors(0, k=1)
        assert set(merged_neighbors_k1) == {
            1,
            2,
        }, "Shells were not merged with a large tolerance."
        # There should be no k=2 shell.
        merged_neighbors_k2 = lattice_merged.get_neighbors(0, k=2)
        assert (
            merged_neighbors_k2 == []
        ), "A k=2 shell should not exist when shells are merged."

        # Assert for the distinct case
        # With a small tolerance, only site 1 should be in the k=1 shell.
        distinct_neighbors_k1 = lattice_distinct.get_neighbors(0, k=1)
        assert distinct_neighbors_k1 == [
            1
        ], "k=1 shell is incorrect with a small tolerance."
        # Site 2 should now be in its own k=2 shell.
        distinct_neighbors_k2 = lattice_distinct.get_neighbors(0, k=2)
        assert distinct_neighbors_k2 == [
            2
        ], "k=2 shell is incorrect with a small tolerance."


class TestTILattice:
    """
    A dedicated class for testing the Translationally Invariant Lattice (TILattice)
    and its subclasses like SquareLattice.
    """

    def test_init_with_mismatched_shapes_raises_error(self):
        """
        Tests that TILattice raises AssertionError if the 'size' parameter's
        length does not match the dimensionality.
        """
        # Act & Assert:
        # Pass a 'size' tuple with 3 elements to a 2D SquareLattice.
        # This should trigger the AssertionError from the parent TILattice class.
        with pytest.raises(AssertionError, match="Size tuple length mismatch"):
            SquareLattice(size=(2, 2, 2))

    def test_init_with_tuple_pbc(self):
        """
        Tests that TILattice correctly handles a tuple input for the 'pbc'
        (periodic boundary conditions) parameter. This covers the 'else' branch.
        """
        # Arrange
        pbc_tuple = (True, False)

        # Act
        # Initialize a lattice with a tuple for pbc.
        lattice = SquareLattice(size=(3, 3), pbc=pbc_tuple)

        # Assert
        # The public 'pbc' attribute should be identical to the tuple we passed.
        assert lattice.pbc == pbc_tuple

    @pytest.mark.parametrize(
        "LatticeClass, init_args, k_precompute",
        [
            (HoneycombLattice, {"size": (4, 5), "pbc": True}, 1),
            (SquareLattice, {"size": (5, 5), "pbc": True}, 2),
            (SquareLattice, {"size": (5, 5), "pbc": False}, 1),
            (KagomeLattice, {"size": (3, 3), "pbc": True}, 1),
        ],
    )
    def test_tilattice_max_k_precomputation_and_ondemand(
        self, LatticeClass, init_args, k_precompute
    ):
        """
        A robust, parameterized test to verify that `precompute_neighbors` (max_k)
        works correctly across various TILattice types and conditions.
        This test is designed to FAIL on the buggy code.
        """
        lattice = LatticeClass(**init_args, precompute_neighbors=k_precompute)

        computed_shells = sorted(list(lattice._neighbor_maps.keys()))
        expected_shells = list(range(1, k_precompute + 1))

        assert computed_shells == expected_shells, (
            f"TEST FAILED for {LatticeClass.__name__} with k={k_precompute}. "
            f"Expected shells {expected_shells}, but found {computed_shells}."
        )

        k_ondemand = k_precompute + 1

        _ = lattice.get_neighbors(0, k=k_ondemand)

        computed_shells_after = sorted(list(lattice._neighbor_maps.keys()))
        expected_shells_after = list(range(1, k_ondemand + 1))

        assert computed_shells_after == expected_shells_after, (
            f"ON-DEMAND TEST FAILED for {LatticeClass.__name__}. "
            f"Expected shells {expected_shells_after} after demanding k={k_ondemand}, "
            f"but found {computed_shells_after}."
        )


class TestLongRangeNeighborFinding:
    """
    Tests neighbor finding on larger lattices and for longer-range interactions (large k),
    addressing suggestions from code review.
    """

    @pytest.fixture(scope="class")
    def large_pbc_square_lattice(self) -> SquareLattice:
        """
        Provides a single 6x8 SquareLattice with PBC for all tests in this class.
        Using scope="class" makes it more efficient as it's created only once.
        """
        # We choose a non-square size to catch potential bugs with non-uniform dimensions.
        return SquareLattice(size=(7, 9), pbc=True)

    def test_neighbor_shell_structure_on_large_lattice(self, large_pbc_square_lattice):
        """
        Tests the coordination number of various neighbor shells (k) on a large
        periodic lattice. In a PBC square lattice, every site is identical, so
        the number of neighbors for each shell k should be the same for all sites.

        Shell distances squared and their coordination numbers for a 2D square lattice:
        - k=1: dist_sq=1  (e.g., (1,0)) -> 4 neighbors
        - k=2: dist_sq=2  (e.g., (1,1)) -> 4 neighbors
        - k=3: dist_sq=4  (e.g., (2,0)) -> 4 neighbors
        - k=4: dist_sq=5  (e.g., (2,1)) -> 8 neighbors
        - k=5: dist_sq=8  (e.g., (2,2)) -> 4 neighbors
        - k=6: dist_sq=9  (e.g., (3,0)) -> 4 neighbors
        - k=7: dist_sq=10 (e.g., (3,1)) -> 8 neighbors
        """
        lattice = large_pbc_square_lattice
        # Pick an arbitrary site, e.g., index 0.
        site_idx = 0

        # Expected coordination numbers for the first few shells.
        expected_coordinations = {1: 4, 2: 4, 3: 4, 4: 8, 5: 4, 6: 4, 7: 8}

        for k, expected_count in expected_coordinations.items():
            neighbors = lattice.get_neighbors(site_idx, k=k)
            assert (
                len(neighbors) == expected_count
            ), f"Failed for k={k}. Expected {expected_count}, got {len(neighbors)}"

    def test_requesting_k_beyond_max_possible_shell(self, large_pbc_square_lattice):
        """
        Tests that requesting a neighbor shell 'k' that is larger than any
        possible shell in the finite lattice returns an empty list, and does
        not raise an error.
        """
        lattice = large_pbc_square_lattice
        site_idx = 0

        # 1. First, find out the maximum number of shells that *do* exist.
        # We do this by calling _build_neighbors with a very large max_k.
        # This is a bit of "white-box" testing but necessary to find the true max k.
        lattice._build_neighbors(max_k=100)
        max_k_found = len(lattice._neighbor_maps)

        # 2. Assert that the last valid shell is not empty.
        last_shell_neighbors = lattice.get_neighbors(site_idx, k=max_k_found)
        assert len(last_shell_neighbors) > 0

        # 3. Assert that requesting a shell just beyond the last valid one returns empty.
        # This is the core of the test.
        non_existent_shell_neighbors = lattice.get_neighbors(
            site_idx, k=max_k_found + 1
        )
        assert non_existent_shell_neighbors == []

    @patch("matplotlib.pyplot.subplots")
    def test_show_method_with_custom_bond_kwargs(
        self, mock_subplots, simple_square_lattice
    ):
        """
        Tests that .show() correctly uses the `bond_kwargs` parameter
        to customize the appearance of neighbor bonds.
        """
        # Arrange:
        # 1. Set up mock Figure and Axes objects, similar to other show() tests.
        mock_fig = matplotlib.figure.Figure()
        mock_ax = matplotlib.axes.Axes(mock_fig, [0.0, 0.0, 1.0, 1.0])
        mock_subplots.return_value = (mock_fig, mock_ax)

        # 2. Define our custom styles and the expected final styles.
        lattice = simple_square_lattice
        custom_bond_kwargs = {"color": "red", "linestyle": ":", "linewidth": 2}

        # The final dictionary should contain the defaults updated by our custom arguments.
        expected_plot_kwargs = {
            "color": "red",  # Overridden
            "linestyle": ":",  # Overridden
            "linewidth": 2,  # A new key
            "alpha": 0.6,  # From default
            "zorder": 1,  # From default
        }

        # 3. We specifically mock the `plot` method on our mock `ax` object.
        with patch.object(mock_ax, "plot") as mock_plot_method:
            # Act:
            # Call the show method with our custom bond styles.
            lattice.show(show_bonds_k=1, bond_kwargs=custom_bond_kwargs)

            # Assert:
            # Check that the plot method was called. For a 2x2 square, there are 4 NN bonds.
            assert mock_plot_method.call_count == 4

            # Get the keyword arguments from the very first call to plot().
            # Note: call_args is a tuple (positional_args, keyword_args). We need the second element.
            actual_kwargs = mock_plot_method.call_args[1]

            # Verify that the keyword arguments used for plotting match our expectations.
            assert actual_kwargs == expected_plot_kwargs

    def test_mixed_boundary_conditions(self):
        """
        Tests neighbor finding with mixed PBC (periodic in x, open in y).
        This verifies that the neighbor finding logic correctly handles
        anisotropy in periodic boundary conditions and returns sorted indices.
        """
        # Arrange: Create a 3x3 square lattice, periodic in x, open in y.
        lattice = SquareLattice(size=(3, 3), pbc=(True, False))

        # We will test a site on the corner of the open boundary: (0, 0)
        corner_site_idx = lattice.get_index((0, 0, 0))

        # --- Test corner site (0, 0, 0), which is index 0 ---
        # Act
        corner_neighbors = lattice.get_neighbors(corner_site_idx, k=1)

        # Assert: The expected neighbors are (1,0,0), (2,0,0) [periodic], and (0,1,0)
        # We get their indices and sort them to create the expected output.
        expected_indices = sorted(
            [
                lattice.get_index((1, 0, 0)),  # Right neighbor
                lattice.get_index((2, 0, 0)),  # "Left" neighbor (wraps around)
                lattice.get_index((0, 1, 0)),  # "Up" neighbor
            ]
        )

        # The list returned by get_neighbors should be identical to our sorted list.
        assert (
            corner_neighbors == expected_indices
        ), "Failed for corner site with mixed BC."

        # --- Test middle site on the edge (1, 0, 0), which is index 1 ---
        edge_site_idx = lattice.get_index((1, 0, 0))

        # Act
        edge_neighbors = lattice.get_neighbors(edge_site_idx, k=1)

        # Assert
        expected_edge_indices = sorted(
            [
                lattice.get_index((0, 0, 0)),  # Left neighbor
                lattice.get_index((2, 0, 0)),  # Right neighbor
                lattice.get_index((1, 1, 0)),  # "Up" neighbor
            ]
        )
        assert (
            edge_neighbors == expected_edge_indices
        ), "Failed for edge site with mixed BC."


class TestAllTILattices:
    """
    A parameterized test class to verify the basic properties and coordination
    numbers for all implemented TILattice subclasses. This avoids code duplication.
    """

    # --- Test data in a structured and readable format ---
    # Format:
    # (
    #     LatticeClass,           # The lattice class to test
    #     {"size": ..., ...},     # Arguments for the constructor
    #     expected_num_sites,     # Expected total number of sites
    #     expected_num_basis,     # Expected number of sites in the basis
    #     {site_repr: count}      # Dict of {representative_site: neighbor_count}
    # )
    # For `site_repr`:
    #   - For simple lattices (basis=1), it's the integer index of the site.
    #   - For composite lattices (basis>1), it's the *basis index* to test.
    lattice_test_cases = [
        # 1D Lattices
        (ChainLattice, {"size": (5,), "pbc": True}, 5, 1, {0: 2, 2: 2}),
        (ChainLattice, {"size": (5,), "pbc": False}, 5, 1, {0: 1, 2: 2}),
        (DimerizedChainLattice, {"size": (3,), "pbc": True}, 6, 2, {0: 2, 1: 2}),
        # 2D Lattices
        (
            RectangularLattice,
            {"size": (3, 4), "pbc": False},
            12,
            1,
            {5: 4, 0: 2, 4: 3},
        ),  # center, corner, edge
        (HoneycombLattice, {"size": (2, 2), "pbc": True}, 8, 2, {0: 3, 1: 3}),
        (TriangularLattice, {"size": (3, 3), "pbc": True}, 9, 1, {0: 6}),
        (CheckerboardLattice, {"size": (2, 2), "pbc": True}, 8, 2, {0: 4, 1: 4}),
        (KagomeLattice, {"size": (2, 2), "pbc": True}, 12, 3, {0: 4, 1: 4, 2: 4}),
        (LiebLattice, {"size": (2, 2), "pbc": True}, 12, 3, {0: 4, 1: 2, 2: 2}),
        # 3D Lattices
        (CubicLattice, {"size": (3, 3, 3), "pbc": True}, 27, 1, {0: 6, 13: 6}),
    ]

    @pytest.mark.parametrize(
        "LatticeClass, init_args, num_sites, num_basis, coordination_numbers",
        lattice_test_cases,
    )
    def test_lattice_properties_and_coordination(
        self,
        LatticeClass,
        init_args,
        num_sites,
        num_basis,
        coordination_numbers,
    ):
        """
        A single, parameterized test to validate all TILattice types.
        """
        # --- Arrange ---
        # Create the lattice instance dynamically from the test data.
        lattice = LatticeClass(**init_args)

        # --- Assert: Basic properties ---
        assert lattice.num_sites == num_sites
        assert lattice.num_basis == num_basis
        assert lattice.dimensionality == len(init_args["size"])

        # --- Assert: Coordination numbers (nearest neighbors, k=1) ---
        for site_repr, expected_count in coordination_numbers.items():
            # This logic correctly gets the site index to test,
            # whether it's a simple or composite lattice.
            if lattice.num_basis > 1:
                # For composite lattices, site_repr is the basis_index.
                # We find the index of this basis site in the first unit cell.
                uc_coord = (0,) * lattice.dimensionality
                test_site_idx = lattice.get_index(uc_coord + (site_repr,))
            else:
                # For simple lattices, site_repr is the absolute site index.
                test_site_idx = site_repr

            neighbors = lattice.get_neighbors(test_site_idx, k=1)
            assert len(neighbors) == expected_count
        if isinstance(LatticeClass, ChainLattice) and not init_args.get("pbc"):
            if test_site_idx == 0:
                assert 1 in neighbors


class TestCustomizeLatticeDynamic:
    """Tests the dynamic modification capabilities of CustomizeLattice."""

    @pytest.fixture
    def initial_lattice(self) -> CustomizeLattice:
        """Provides a basic 3-site lattice for modification tests."""
        return CustomizeLattice(
            dimensionality=2,
            identifiers=["A", "B", "C"],
            coordinates=[[0, 0], [1, 0], [0, 1]],
        )

    def test_from_lattice_conversion(self):
        """Tests creating a CustomizeLattice from a TILattice."""
        # Arrange
        sq_lattice = SquareLattice(size=(2, 2), pbc=False)

        # Act
        custom_lattice = CustomizeLattice.from_lattice(sq_lattice)

        # Assert
        assert isinstance(custom_lattice, CustomizeLattice)
        assert custom_lattice.num_sites == sq_lattice.num_sites
        assert custom_lattice.dimensionality == sq_lattice.dimensionality
        # Verify a site to be sure
        np.testing.assert_array_equal(
            custom_lattice.get_coordinates(3), sq_lattice.get_coordinates(3)
        )
        assert custom_lattice.get_identifier(3) == sq_lattice.get_identifier(3)

    def test_add_sites_successfully(self, initial_lattice):
        """Tests adding new, valid sites to the lattice."""
        # Arrange
        lat = initial_lattice
        assert lat.num_sites == 3

        # Act
        lat.add_sites(identifiers=["D", "E"], coordinates=[[1, 1], [2, 2]])

        # Assert
        assert lat.num_sites == 5
        assert lat.get_identifier(4) == "E"
        np.testing.assert_array_equal(lat.get_coordinates(3), np.array([1, 1]))
        assert "E" in lat._ident_to_idx

    def test_remove_sites_successfully(self, initial_lattice):
        """Tests removing existing sites from the lattice."""
        # Arrange
        lat = initial_lattice
        assert lat.num_sites == 3

        # Act
        lat.remove_sites(identifiers=["A", "C"])

        # Assert
        assert lat.num_sites == 1
        assert lat.get_identifier(0) == "B"  # Site 'B' is now at index 0
        assert "A" not in lat._ident_to_idx
        np.testing.assert_array_equal(lat.get_coordinates(0), np.array([1, 0]))

    def test_add_duplicate_identifier_raises_error(self, initial_lattice):
        """Tests that adding a site with an existing identifier fails."""
        with pytest.raises(ValueError, match="Duplicate identifiers found"):
            initial_lattice.add_sites(identifiers=["A"], coordinates=[[9, 9]])

    def test_remove_nonexistent_identifier_raises_error(self, initial_lattice):
        """Tests that removing a non-existent site fails."""
        with pytest.raises(ValueError, match="Non-existent identifiers provided"):
            initial_lattice.remove_sites(identifiers=["Z"])

    def test_modification_clears_neighbor_cache(self, initial_lattice):
        """
        Tests that add_sites and remove_sites correctly invalidate the
        pre-computed neighbor map.
        """
        # Arrange: Pre-compute neighbors on the initial lattice
        initial_lattice._build_neighbors(max_k=1)
        assert 0 in initial_lattice._neighbor_maps[1]  # Check that neighbors exist

        # Act 1: Add a site
        initial_lattice.add_sites(identifiers=["D"], coordinates=[[5, 5]])

        # Assert 1: The neighbor map should now be empty
        assert not initial_lattice._neighbor_maps

        # Arrange 2: Re-compute neighbors and then remove a site
        initial_lattice._build_neighbors(max_k=1)
        assert 0 in initial_lattice._neighbor_maps[1]

        # Act 2: Remove a site
        initial_lattice.remove_sites(identifiers=["A"])

        # Assert 2: The neighbor map should be empty again
        assert not initial_lattice._neighbor_maps

    def test_modification_clears_distance_matrix_cache(self, initial_lattice):
        """
        Tests that add_sites and remove_sites correctly invalidate the
        cached distance matrix and that the recomputed matrix is correct.
        """
        # Arrange 1: Compute, cache, and perform a meaningful check on the original matrix.
        lat = initial_lattice
        original_matrix = lat.distance_matrix
        assert lat._distance_matrix is not None
        assert original_matrix.shape == (3, 3)
        # Meaningful check: distance from 'A'(idx 0) to 'B'(idx 1) should be 1.0
        np.testing.assert_allclose(original_matrix[0, 1], 1.0)

        # Act 1: Add a site. This should invalidate the cache.
        lat.add_sites(identifiers=["D"], coordinates=[[1, 1]])

        # Assert 1: Check cache is cleared and the new matrix is correct.
        assert lat._distance_matrix is None  # Verify cache invalidation
        new_matrix_added = lat.distance_matrix
        assert new_matrix_added.shape == (4, 4)
        # Meaningful check: distance from 'B'(idx 1) to new site 'D'(idx 3) should be 1.0
        # Coords: B=[1,0], D=[1,1]
        np.testing.assert_allclose(new_matrix_added[1, 3], 1.0)

        # Act 2: Remove a site. This should also invalidate the cache.
        lat.remove_sites(identifiers=["A"])

        # Assert 2: Check cache is cleared again and the final matrix is correct.
        assert lat._distance_matrix is None  # Verify cache invalidation
        final_matrix = lat.distance_matrix
        assert final_matrix.shape == (3, 3)  # Now has 3 sites again
        # Meaningful check: After removing 'A', the sites are B, C, D.
        # 'B' is now at index 0 (coords [1,0])
        # 'C' is now at index 1 (coords [0,1])
        # 'D' is now at index 2 (coords [1,1])
        # Distance from new 'B' (idx 0) to new 'D' (idx 2) should be 1.0
        np.testing.assert_allclose(final_matrix[0, 2], 1.0)

    def test_neighbor_finding_returns_sorted_list(self, simple_square_lattice):
        """
        Ensures that the list of neighbors returned by get_neighbors is always sorted.
        This provides a stricter check than set-based comparisons.
        """
        # Arrange
        lattice = simple_square_lattice

        # Act
        # Get neighbors for the central site (index 1 in a 2x2 grid)
        # Expected neighbors are 0, 3.
        neighbors = lattice.get_neighbors(1, k=1)

        # Assert
        # We compare directly against a pre-sorted list, not a set.
        # This will fail if the implementation returns [3, 0] instead of [0, 3].
        assert neighbors == [
            0,
            3,
        ], "The neighbor list should be sorted in ascending order."


class TestDistanceMatrix:

    # This is the upgraded, parameterized test.
    @pytest.mark.parametrize(
        # We define test scenarios as tuples:
        # (build_k, check_site_identifier, expected_dist_sq)
        # build_k: The number of neighbor shells to pre-build.
        # check_site_identifier: The identifier of a site whose distance from the origin we will check.
        # expected_dist_sq: The expected squared distance to that site.
        "build_k, check_site_identifier, expected_dist_sq",
        [
            # Scenario 1: The most common case. Build only NN (k=1), but check a NNN (k=2) distance.
            # A buggy cache would fail this.
            (1, (1, 1, 0), 2.0),
            # Scenario 2: Build up to k=2, but check a k=3 distance.
            (2, (2, 0, 0), 4.0),
            # Scenario 3: Build up to k=3, but check a k=4 distance.
            (3, (2, 1, 0), 5.0),
            # Scenario 4: A more complex, higher-order neighbor.
            (5, (3, 1, 0), 10.0),
        ],
    )
    def test_tilattice_full_pbc_distance_matrix_is_correct_regardless_of_build_k(
        self, build_k, check_site_identifier, expected_dist_sq
    ):
        """
        Tests that the distance matrix for a fully periodic TILattice is
        always fully correct, no matter how many neighbor shells were pre-calculated.
        This is a high-strength test designed to catch subtle caching bugs where
        the cached matrix might only contain partial information.
        """
        # Arrange
        # Using a larger, non-square lattice to avoid accidental symmetries
        lat = SquareLattice(size=(7, 9), pbc=True)

        # Act
        # Step 1: Pre-build neighbors. This is where a faulty caching
        # mechanism in the source code might be triggered.
        lat._build_neighbors(max_k=build_k)

        # Step 2: Access the distance_matrix property. A correct implementation
        # will return a fully valid matrix.
        dist_matrix = lat.distance_matrix

        # Assert
        # Find the indices for the sites we want to check.
        origin_idx = lat.get_index((0, 0, 0))
        check_site_idx = lat.get_index(check_site_identifier)

        # The core assertion: check the distance.
        actual_dist_sq = dist_matrix[origin_idx, check_site_idx] ** 2

        error_message = (
            f"Distance matrix failed when building k={build_k}. "
            f"Checking distance to site {check_site_identifier} (expected sq={expected_dist_sq}) "
            f"but got sq={actual_dist_sq} instead."
        )

        np.testing.assert_allclose(
            actual_dist_sq, expected_dist_sq, err_msg=error_message
        )

    def test_tilattice_mixed_bc_distance_matrix_is_correct(self):
        """
        Tests that the distance matrix is correctly calculated for a TILattice
        with mixed boundary conditions (e.g., periodic in x, open in y).
        """
        # Arrange
        # pbc=(True, False) means periodic along x-axis, open along y-axis.
        lat = SquareLattice(size=(5, 5), pbc=(True, False))

        # Pre-build neighbors to engage the caching logic.
        lat._build_neighbors(max_k=2)
        dist_matrix = lat.distance_matrix

        # Assert
        origin_idx = lat.get_index((0, 0, 0))

        # 1. Test a distance affected by the periodic boundary (x-direction)
        # The distance between (0,0) and (4,0) should be 1.0 due to PBC wrap-around.
        pbc_neighbor_idx = lat.get_index((4, 0, 0))
        np.testing.assert_allclose(dist_matrix[origin_idx, pbc_neighbor_idx], 1.0)

        # 2. Test a distance affected by the open boundary (y-direction)
        # The distance between (0,0) and (0,4) should be 4.0 as there's no wrap-around.
        obc_neighbor_idx = lat.get_index((0, 4, 0))
        np.testing.assert_allclose(dist_matrix[origin_idx, obc_neighbor_idx], 4.0)

        # 3. Test a general, off-axis point.
        # Distance from (0,0) to (3,3) with x-pbc. The x-distance is min(3, 5-3=2) = 2.
        # The y-distance is 3. So total distance is sqrt(2^2 + 3^2) = sqrt(13).
        general_neighbor_idx = lat.get_index((3, 3, 0))
        np.testing.assert_allclose(
            dist_matrix[origin_idx, general_neighbor_idx], np.sqrt(13)
        )

    # --- This list and the following test are now at the correct indentation level ---
    lattice_instances_for_invariant_test = [
        SquareLattice(size=(4, 4), pbc=True),
        SquareLattice(size=(4, 3), pbc=(True, False)),  # Mixed BC, non-square
        HoneycombLattice(size=(3, 3), pbc=True),
        TriangularLattice(size=(4, 4), pbc=False),
        CustomizeLattice(
            dimensionality=2,
            identifiers=list(range(4)),
            coordinates=[[0, 0], [1, 1], [0, 1], [1, 0]],
        ),
    ]

    @pytest.mark.parametrize("lattice", lattice_instances_for_invariant_test)
    def test_distance_matrix_invariants_for_all_lattice_types(self, lattice):
        """
        Tests that the distance matrix for any lattice type adheres to
        fundamental mathematical properties (invariants): symmetry, zero diagonal,
        and positive off-diagonal elements.
        """
        # Arrange
        n = lattice.num_sites
        if n < 2:
            pytest.skip("Invariant test requires at least 2 sites.")

        # Act
        # We call the property directly, without building neighbors first,
        # to test the on-demand computation path.
        matrix = lattice.distance_matrix

        # Assert
        # 1. Symmetry: The matrix must be equal to its transpose.
        np.testing.assert_allclose(
            matrix,
            matrix.T,
            err_msg=f"Distance matrix for {type(lattice).__name__} is not symmetric.",
        )

        # 2. Zero Diagonal: All diagonal elements must be zero.
        np.testing.assert_allclose(
            np.diag(matrix),
            np.zeros(n),
            err_msg=f"Diagonal of distance matrix for {type(lattice).__name__} is not zero.",
        )

        # 3. Positive Off-diagonal: All non-diagonal elements must be > 0.
        # We create a boolean mask for the off-diagonal elements.
        off_diagonal_mask = ~np.eye(n, dtype=bool)
        assert np.all(
            matrix[off_diagonal_mask] > 1e-9
        ), f"Found non-positive off-diagonal elements in distance matrix for {type(lattice).__name__}."


# @pytest.mark.slow
# class TestPerformance:
#     def test_pbc_implementation_is_not_significantly_slower_than_obc(self):
#         """
#         A performance regression test.
#         It ensures that the specialized implementation for fully periodic
#         lattices (pbc=True) is not substantially slower than the general
#         implementation used for open boundaries (pbc=False).
#         This test will FAIL with the current code, exposing the performance bug.
#         """
#         # Arrange: Use a large-enough lattice to make performance differences apparent
#         size = (30, 30)
#         k = 1

#         # Act 1: Measure the execution time of the general (OBC) implementation
#         start_time_obc = time.time()
#         _ = SquareLattice(size=size, pbc=False, precompute_neighbors=k)
#         duration_obc = time.time() - start_time_obc

#         # Act 2: Measure the execution time of the specialized (PBC) implementation
#         start_time_pbc = time.time()
#         _ = SquareLattice(size=size, pbc=True, precompute_neighbors=k)
#         duration_pbc = time.time() - start_time_pbc

#         print(
#             f"\n[Performance] OBC ({size}): {duration_obc:.4f}s | PBC ({size}): {duration_pbc:.4f}s"
#         )

#         # Assert: The PBC implementation should not be drastically slower.
#         # We allow it to be up to 3 times slower to account for minor overheads,
#         # but this will catch the current 10x+ regression.
#         # THIS ASSERTION WILL FAIL with the current buggy code.
#         assert duration_pbc < duration_obc * 5, (
#             "The specialized PBC implementation is significantly slower "
#             "than the general-purpose implementation."
#         )


def _validate_layers(bonds, layers) -> None:
    """
    A helper function to scientifically validate the output of get_compatible_layers.
    """
    # MODIFICATION: This function now takes the original bonds list for comparison.
    expected_edges = set(tuple(sorted(b)) for b in bonds)
    actual_edges = set(tuple(sorted(edge)) for layer in layers for edge in layer)

    assert (
        expected_edges == actual_edges
    ), "Completeness check failed: The set of all edges in the layers must "
    "exactly match the input bonds."

    for i, layer in enumerate(layers):
        qubits_in_layer: set[int] = set()
        for edge in layer:
            q1, q2 = edge
            assert (
                q1 not in qubits_in_layer
            ), f"Compatibility check failed: Qubit {q1} is reused in layer {i}."
            qubits_in_layer.add(q1)
            assert (
                q2 not in qubits_in_layer
            ), f"Compatibility check failed: Qubit {q2} is reused in layer {i}."
            qubits_in_layer.add(q2)


@pytest.mark.parametrize(
    "lattice_instance",
    [
        SquareLattice(size=(3, 2), pbc=False),
        SquareLattice(size=(3, 3), pbc=True),
        HoneycombLattice(size=(2, 2), pbc=False),
    ],
    ids=[
        "SquareLattice_3x2_OBC",
        "SquareLattice_3x3_PBC",
        "HoneycombLattice_2x2_OBC",
    ],
)
def test_layering_on_various_lattices(lattice_instance: AbstractLattice):
    """Tests gate layering for various standard lattice types."""
    bonds = lattice_instance.get_neighbor_pairs(k=1, unique=True)
    layers = get_compatible_layers(bonds)

    assert len(layers) > 0, "Layers should not be empty for non-trivial lattices."
    _validate_layers(bonds, layers)


def test_layering_on_1d_chain_pbc():
    """Test layering on a 1D chain with periodic boundaries (a cycle graph)."""
    lattice_even = ChainLattice(size=(6,), pbc=True)
    bonds_even = lattice_even.get_neighbor_pairs(k=1, unique=True)
    layers_even = get_compatible_layers(bonds_even)
    _validate_layers(bonds_even, layers_even)

    lattice_odd = ChainLattice(size=(5,), pbc=True)
    bonds_odd = lattice_odd.get_neighbor_pairs(k=1, unique=True)
    layers_odd = get_compatible_layers(bonds_odd)
    assert len(layers_odd) == 3, "A 5-site cycle graph should be 3-colorable."
    _validate_layers(bonds_odd, layers_odd)


def test_layering_on_custom_star_graph():
    """Test layering on a custom lattice forming a star graph."""
    star_edges = [(0, 1), (0, 2), (0, 3)]
    layers = get_compatible_layers(star_edges)
    assert len(layers) == 3, "A star graph S_4 requires 3 layers."
    _validate_layers(star_edges, layers)


def test_layering_on_edge_cases():
    """Test various edge cases: empty, single-site, and no-edge lattices."""
    layers_empty = get_compatible_layers([])
    assert layers_empty == [], "Layers should be empty for an empty set of bonds."

    single_edge = [(0, 1)]
    layers_single = get_compatible_layers(single_edge)
    assert layers_single == [[(0, 1)]]
    _validate_layers(single_edge, layers_single)
