import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch

import pytest
import numpy as np

from tensorcircuit.templates.lattice import (
    CustomizeLattice,
    SquareLattice,
    HoneycombLattice,
    TriangularLattice,
)


@pytest.fixture
def simple_square_lattice():
    """
    Provides a simple 2x2 square CustomizeLattice instance for neighbor tests.
    The sites are indexed as follows:
    2--3
    |  |
    0--1
    """
    coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    ids = list(range(len(coords)))
    lattice = CustomizeLattice(dimensionality=2, identifiers=ids, coordinates=coords)
    # Pre-calculate neighbors up to the 2nd shell for use in tests.
    lattice._build_neighbors(max_k=2)
    return lattice


@pytest.fixture
def kagome_lattice_fragment():
    """
    Pytest fixture to provide a standard CustomizeLattice instance.
    This represents the Kagome fragment from the project requirements,
    making it a reusable object for multiple tests.
    """
    kag_coords = [
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3) / 2],  # Triangle 1
        [2, 0],
        [1.5, np.sqrt(3) / 2],  # Triangle 2 (shifted basis)
        [1, np.sqrt(3)],  # Top site
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
        coords = [[0, 0], [1, 0]]  # 2 coordinates
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
        coords_wrong_dim = [[0, 0], [1, 0, 0]]  # A mix of 2D and 3D
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

        # --- Assertion for non-calculated neighbors ---
        # The neighbor map was only built up to k=2, so k=3 should be empty.
        assert lattice.get_neighbors(0, k=3) == []

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

    @patch("tensorcircuit.templates.lattice.plt.show")
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

    def test_get_neighbors_prints_warning_for_uncached_k(
        self, simple_square_lattice, capsys
    ):
        """
        Tests that a warning is printed when get_neighbors is called for a 'k'
        that has not been pre-calculated. This covers the warning branch.
        """
        # Arrange
        lattice = simple_square_lattice  # This fixture builds neighbors up to k=2
        k_to_test = 99  # A value that is clearly not cached

        # Act
        result = lattice.get_neighbors(0, k=k_to_test)

        # Assert
        # First, ensure the function returns an empty list as documented.
        assert result == []

        # Second, use the 'capsys' fixture to check what was printed to stdout.
        captured = capsys.readouterr()
        assert (
            f"Warning: {k_to_test}-th nearest neighbors not pre-calculated"
            in captured.out
        )

    @patch("matplotlib.pyplot.show")
    def test_show_prints_warning_for_uncached_bonds(
        self, mock_show, simple_square_lattice, capsys
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
        captured = capsys.readouterr()
        assert (
            f"Warning: Cannot draw bonds. k={k_to_test} neighbors have not been calculated"
            in captured.out
        )

    @patch("matplotlib.pyplot.show")
    def test_show_method_for_3d_lattice(self, mock_show):
        """
        Tests that the .show() method can handle a 3D lattice without
        crashing. This covers the 'if self.dimensionality == 3:' branches.
        """
        # Arrange: Create a simple 2-site lattice in 3D space.
        coords_3d = [[0, 0, 0], [1, 1, 1]]
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


# --- Tests for TILattice using SquareLattice ---


@pytest.fixture
def obc_square_lattice():
    """Provides a 3x3 SquareLattice with Open Boundary Conditions."""
    return SquareLattice(size=(3, 3), pbc=False)


@pytest.fixture
def pbc_square_lattice():
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

        idx, ident, coords = lattice.get_site_info(center_idx)
        assert ident == (1, 1, 0)
        np.testing.assert_array_equal(coords, np.array([1.0, 1.0]))

        corner_idx = 0
        idx, ident, coords = lattice.get_site_info(corner_idx)
        assert ident == (0, 0, 0)
        np.testing.assert_array_equal(coords, np.array([0.0, 0.0]))

    def test_neighbors_with_open_boundaries(self, obc_square_lattice):
        """
        Tests neighbor finding with Open Boundary Conditions (OBC).
        """
        lattice = obc_square_lattice
        center_idx = lattice.get_index((1, 1, 0))
        corner_idx = lattice.get_index((0, 0, 0))
        edge_idx = lattice.get_index((1, 0, 0))

        assert len(lattice.get_neighbors(center_idx, k=1)) == 4
        assert len(lattice.get_neighbors(corner_idx, k=1)) == 2
        assert len(lattice.get_neighbors(edge_idx, k=1)) == 3

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
def pbc_honeycomb_lattice():
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
def pbc_triangular_lattice():
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


class TestTILatticeEdgeCases:
    """
    A dedicated class for testing the behavior of TILattice and its
    subclasses under less common, "edge-case" conditions.
    """

    @pytest.fixture
    def obc_1d_chain(self):
        """
        Provides a 5-site 1D chain using SquareLattice with Open Boundary Conditions.
        This is created by setting one dimension of the size to 1.
        """
        # 0--1--2--3--4
        return SquareLattice(size=(5, 1), pbc=False)

    def test_1d_chain_properties_and_neighbors(self, obc_1d_chain):
        """
        Tests neighbor finding on a 1D chain.
        """
        # Arrange
        lattice = obc_1d_chain

        # Assert basic properties
        assert lattice.num_sites == 5
        # Note: SquareLattice is hardcoded to 2D, but its behavior should be 1D.
        assert lattice.dimensionality == 2

        # Assert neighbor counts for different positions
        # Endpoint (site 0) should have 1 neighbor (site 1)
        endpoint_idx = lattice.get_index((0, 0, 0))
        assert len(lattice.get_neighbors(endpoint_idx, k=1)) == 1
        assert lattice.get_neighbors(endpoint_idx, k=1) == [1]

        # Middle point (site 2) should have 2 neighbors (sites 1 and 3)
        middle_idx = lattice.get_index((2, 0, 0))
        assert len(lattice.get_neighbors(middle_idx, k=1)) == 2
        assert set(lattice.get_neighbors(middle_idx, k=1)) == {1, 3}

    @pytest.fixture
    def nonsquare_lattice(self):
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
        # The obc_1d_chain fixture is defined in the TestTILatticeEdgeCases class,
        # but can be used here if this test is in a class that also has access to it,
        # or you can move the fixture to a more general scope (like conftest.py) if needed.
        # For simplicity, we assume it's accessible.
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

    def test_empty_lattice_handles_gracefully(self, capsys):
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
        # capsys is a pytest fixture that captures stdout/stderr.
        empty_lattice.show()
        captured = capsys.readouterr()
        assert "Lattice is empty, nothing to show." in captured.out

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
            dimensionality=2, identifiers=[0], coordinates=[[0, 0]]
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

    def test_show_warning_for_unsupported_dimension(self, capsys):
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
        captured = capsys.readouterr()
        assert "Warning: show() is not implemented for 4D lattices." in captured.out


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
        # We should not pass low-level arguments that SquareLattice doesn't accept.
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
