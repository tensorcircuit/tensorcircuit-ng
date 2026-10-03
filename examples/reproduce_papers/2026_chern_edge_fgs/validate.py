"""Validate Figure 2(a) with independent single-particle and Fock-space dynamics."""

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, expm
import tensorcircuit as tc

if __package__:
    from .main import fixed_number_alpha, hamiltonian, simulate
else:
    from main import fixed_number_alpha, hamiltonian, simulate


def single_particle_hamiltonian(length, potential=0.0):
    """Independent single-particle construction, without any FGS helper."""
    h = np.zeros((length**2, length**2), dtype=complex)
    for site in range(length**2):
        y, x = divmod(site, length)
        if x < length - 1:
            h[site, site + 1] = -1.0
        if y < length - 1:
            h[site, site + length] = -np.exp(-2j * np.pi * x / 3)
    h += h.conj().T
    h[length * (length - 1), length * (length - 1)] = potential
    return h


def check_full_size(output):
    """Compare all correlations, conserved quantities, and saved plotted arrays."""
    times = (0.0, 4.0, 8.0, 12.0)
    h_fgs, alpha, reference, correlations = [
        np.asarray(tc.backend.numpy(a)) for a in simulate(times=times)
    ]
    modes, particles = 144, 96
    h = single_particle_hamiltonian(12)
    initial_h = single_particle_hamiltonian(12, potential=-1.0)
    _, orbitals = eigh(initial_h)
    rho0 = orbitals[:, :particles] @ orbitals[:, :particles].conj().T
    _, gs_orbitals = eigh(h)
    rho_gs = gs_orbitals[:, :particles] @ gs_orbitals[:, :particles].conj().T
    expected_h = np.zeros((2 * modes, 2 * modes), dtype=complex)
    expected_h[:modes, :modes] = h / 2
    expected_h[modes:, modes:] = -h.T / 2
    np.testing.assert_allclose(h_fgs, expected_h, atol=1e-14, rtol=0)
    np.testing.assert_allclose(reference[modes:, modes:], rho_gs.T, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        alpha.conj().T @ alpha, np.eye(modes), atol=2e-12, rtol=0
    )
    expected = []
    for t in times:
        u = expm(-1j * t * h)
        rho = u @ rho0 @ u.conj().T
        cm = np.zeros_like(expected_h)
        cm[:modes, :modes] = np.eye(modes) - rho
        cm[modes:, modes:] = rho.T
        expected.append(cm)
    expected = np.array(expected)
    np.testing.assert_allclose(correlations, expected, atol=2e-11, rtol=0)
    np.testing.assert_allclose(
        correlations @ correlations, correlations, atol=2e-11, rtol=0
    )
    np.testing.assert_allclose(
        correlations, correlations.conj().transpose(0, 2, 1), atol=2e-12, rtol=0
    )
    occupied = correlations[:, modes:, modes:]
    numbers = np.trace(occupied, axis1=1, axis2=2).real
    energies = np.einsum("ij,tij->t", h, occupied).real
    np.testing.assert_allclose(numbers, particles, atol=1e-9, rtol=0)
    np.testing.assert_allclose(energies, energies[0], atol=1e-9, rtol=0)
    delta = np.diagonal(occupied, axis1=1, axis2=2).real - np.diag(rho_gs).real
    np.testing.assert_allclose(delta.sum(axis=1), 0.0, atol=1e-9, rtol=0)
    with np.load(output / "densities.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["times"], times)
        np.testing.assert_allclose(
            saved["delta_density"], delta.reshape(4, 12, 12), atol=2e-11, rtol=0
        )
    # The coefficient product along right/up/left/down fixes the flux sign.
    loop = h[0, 1] * h[1, 13] * h[13, 12] * h[12, 0]
    np.testing.assert_allclose(loop, np.exp(-2j * np.pi / 3), atol=1e-14, rtol=0)


def check_fock_space():
    """Check complex hopping, particle filling, and physical time on four modes."""
    length, modes, particles = 2, 4, 2
    h = single_particle_hamiltonian(length)
    initial_h = single_particle_hamiltonian(length, potential=-1.0)
    annihilators = []
    for site in range(modes):
        a = np.zeros((2**modes, 2**modes), dtype=complex)
        for state in range(2**modes):
            mask = 1 << (modes - 1 - site)
            if state & mask:
                parity = (state >> (modes - site)).bit_count()
                a[state ^ mask, state] = (-1) ** parity
        annihilators.append(a)
    bilinears = np.array([[a.conj().T @ b for b in annihilators] for a in annihilators])
    many_body_h = np.einsum("ij,ijab->ab", h, bilinears)
    many_body_initial = np.einsum("ij,ijab->ab", initial_h, bilinears)
    sector = [i for i in range(2**modes) if i.bit_count() == particles]
    _, states = eigh(many_body_initial[np.ix_(sector, sector)])
    psi0 = np.zeros(2**modes, dtype=complex)
    psi0[sector] = states[:, 0]
    h_fgs = hamiltonian(length)
    alpha = fixed_number_alpha(hamiltonian(length, potential=-1.0), particles)
    for t in (0.0, 0.37, 1.2):
        circuit = tc.Circuit(modes, inputs=psi0)
        circuit.any(*range(modes), unitary=expm(-1j * t * many_body_h))
        psi = np.asarray(tc.backend.numpy(circuit.state()))
        expected = np.einsum("a,ijab,b->ij", psi.conj(), bilinears, psi)
        sim = tc.FGSSimulator(modes, alpha=alpha)
        sim.evol_hamiltonian(2 * t * h_fgs)
        actual = np.asarray(tc.backend.numpy(sim.get_cmatrix()))[modes:, modes:]
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    # Analytic two-mode oscillation makes the factor-of-two convention explicit.
    sim = tc.FGSSimulator(2, filled=[0])
    sim.evol_hamiltonian(2 * 0.37 * tc.FGSSimulator.hopping(1j, 0, 1, 2))
    density = np.diag(np.asarray(tc.backend.numpy(sim.get_cmatrix())))[2:].real
    np.testing.assert_allclose(
        density, [math.cos(0.37) ** 2, math.sin(0.37) ** 2], atol=2e-12, rtol=0
    )


def main():
    """Run fail-fast numerical checks after main.py has produced densities.npz."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    output = parser.parse_args().output_dir
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    check_full_size(output)
    check_fock_space()
    print("All single-particle, conservation, and Fock-space checks passed.")


if __name__ == "__main__":
    main()
