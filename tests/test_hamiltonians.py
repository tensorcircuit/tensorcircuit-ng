import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture as lf

import tensorcircuit as tc

from tensorcircuit.templates.lattice import (
    ChainLattice,
    SquareLattice,
    CustomizeLattice,
)
from tensorcircuit.templates.hamiltonians import (
    heisenberg_hamiltonian,
    rydberg_hamiltonian,
)

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_I = np.eye(2, dtype=complex)


class TestHeisenbergHamiltonian:
    """
    Test suite for the heisenberg_hamiltonian function.
    """

    def test_empty_lattice(self):
        """
        Test that an empty lattice produces a 0x0 matrix.
        """
        empty_lattice = CustomizeLattice(
            dimensionality=2, identifiers=[], coordinates=[]
        )
        h = heisenberg_hamiltonian(empty_lattice)
        assert h.shape == (1, 1)
        assert h.nnz == 0

    def test_single_site(self):
        """
        Test that a single-site lattice (no bonds) produces a 2x2 zero matrix.
        """
        single_site_lattice = ChainLattice(size=(1,), pbc=False)
        h = heisenberg_hamiltonian(single_site_lattice)
        assert h.shape == (2, 2)
        assert h.nnz == 0

    @pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
    def test_two_sites_chain(self, backend):
        """
        Test a two-site chain against a manually calculated Hamiltonian.
        This is the most critical test for scientific correctness.
        """
        lattice = ChainLattice(size=(2,), pbc=False)
        j_coupling = -1.5  # Test with a non-trivial coupling constant
        h_generated = heisenberg_hamiltonian(lattice, j_coupling=j_coupling)

        # Manually construct the expected Hamiltonian: H = J * (X_0X_1 + Y_0Y_1 + Z_0Z_1)
        xx = np.kron(PAULI_X, PAULI_X)
        yy = np.kron(PAULI_Y, PAULI_Y)
        zz = np.kron(PAULI_Z, PAULI_Z)
        h_expected = j_coupling * (xx + yy + zz)

        assert h_generated.shape == (4, 4)
        print(tc.backend.to_dense(h_generated))
        assert np.allclose(tc.backend.to_dense(h_generated), h_expected, atol=1e-5)

    def test_square_lattice_properties(self):
        """
        Test properties of a larger lattice (2x2 square) without full matrix comparison.
        """
        lattice = SquareLattice(size=(2, 2), pbc=True)  # 4 sites, 8 bonds with PBC
        h = heisenberg_hamiltonian(lattice, j_coupling=1.0)

        assert h.shape == (16, 16)
        assert h.nnz > 0
        h_dense = tc.backend.to_dense(h)
        assert np.allclose(h_dense, h_dense.conj().T)


class TestRydbergHamiltonian:
    """
    Test suite for the rydberg_hamiltonian function.
    """

    def test_single_site_rydberg(self):
        """
        Test a single atom, which should only have driving and detuning terms.
        """
        lattice = ChainLattice(size=(1,), pbc=False)
        omega, delta, c6 = 2.0, 0.5, 100.0
        h_generated = rydberg_hamiltonian(lattice, omega, delta, c6)

        h_expected = (omega / 2.0) * PAULI_X + (delta / 2.0) * PAULI_Z

        assert h_generated.shape == (2, 2)
        assert np.allclose(tc.backend.to_dense(h_generated), h_expected)

    def test_two_sites_rydberg(self):
        """
        Test a two-site chain for Rydberg Hamiltonian, including interaction.
        """
        lattice = ChainLattice(size=(2,), pbc=False, lattice_constant=1.5)
        omega, delta, c6 = 1.0, -0.5, 10.0
        h_generated = rydberg_hamiltonian(lattice, omega, delta, c6)

        v_ij = c6 / (1.5**6)

        h1 = (omega / 2.0) * (np.kron(PAULI_X, PAULI_I) + np.kron(PAULI_I, PAULI_X))
        z0_coeff = delta / 2.0 - v_ij / 4.0
        z1_coeff = delta / 2.0 - v_ij / 4.0
        h2 = z0_coeff * np.kron(PAULI_Z, PAULI_I) + z1_coeff * np.kron(PAULI_I, PAULI_Z)
        h3 = (v_ij / 4.0) * np.kron(PAULI_Z, PAULI_Z)

        h_expected = h1 + h2 + h3

        assert h_generated.shape == (4, 4)
        h_generated_dense = tc.backend.to_dense(h_generated)

        assert np.allclose(h_generated_dense, h_expected)

    @pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
    def test_anisotropic_heisenberg(self, backend):
        """
        Test the anisotropic Heisenberg model with different Jx, Jy, Jz.
        """
        lattice = ChainLattice(size=(2,), pbc=False)
        j_coupling = [-1.0, 0.5, 2.0]  # Jx, Jy, Jz
        h_generated = heisenberg_hamiltonian(lattice, j_coupling=j_coupling)

        # Manually construct the expected Hamiltonian
        jx, jy, jz = j_coupling
        xx = np.kron(PAULI_X, PAULI_X)
        yy = np.kron(PAULI_Y, PAULI_Y)
        zz = np.kron(PAULI_Z, PAULI_Z)
        h_expected = jx * xx + jy * yy + jz * zz

        h_generated_dense = tc.backend.to_dense(h_generated)
        assert h_generated_dense.shape == (4, 4)
        assert np.allclose(h_generated_dense, h_expected)

    def test_heisenberg_hamiltonian_all_interactions(self):
        """
        Test the Heisenberg Hamiltonian with 'all' interaction scope.
        For a 3-site chain, this should include interactions (0,1), (0,2), and (1,2).
        """
        lattice = ChainLattice(size=(3,), pbc=False)
        j_coupling = 1.0
        h_generated = heisenberg_hamiltonian(
            lattice, j_coupling=j_coupling, interaction_scope="all"
        )

        # Manually construct the expected Hamiltonian for all-to-all interaction
        # H = J * [ (X0X1 + Y0Y1 + Z0Z1) + (X0X2 + Y0Y2 + Z0Z2) + (X1X2 + Y1Y2 + Z1Z2) ]
        xx_01 = np.kron(PAULI_X, np.kron(PAULI_X, PAULI_I))
        yy_01 = np.kron(PAULI_Y, np.kron(PAULI_Y, PAULI_I))
        zz_01 = np.kron(PAULI_Z, np.kron(PAULI_Z, PAULI_I))

        xx_02 = np.kron(PAULI_X, np.kron(PAULI_I, PAULI_X))
        yy_02 = np.kron(PAULI_Y, np.kron(PAULI_I, PAULI_Y))
        zz_02 = np.kron(PAULI_Z, np.kron(PAULI_I, PAULI_Z))

        xx_12 = np.kron(PAULI_I, np.kron(PAULI_X, PAULI_X))
        yy_12 = np.kron(PAULI_I, np.kron(PAULI_Y, PAULI_Y))
        zz_12 = np.kron(PAULI_I, np.kron(PAULI_Z, PAULI_Z))

        h_expected = j_coupling * (
            (xx_01 + yy_01 + zz_01) + (xx_02 + yy_02 + zz_02) + (xx_12 + yy_12 + zz_12)
        )

        assert h_generated.shape == (8, 8)
        assert np.allclose(tc.backend.to_dense(h_generated), h_expected)
