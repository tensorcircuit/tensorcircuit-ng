"""
Reproduction of "Real-Time Dynamics in Two Dimensions with Tensor Network States
via Time-Dependent Variational Monte Carlo Method"
Link: https://arxiv.org/abs/2512.06768
Description:
This script reproduces published Figure 2(a) using TensorCircuit-NG's
FGSSimulator for exact free-fermion dynamics, at the original 12 x 12 size.
"""

import argparse
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorcircuit as tc


def hamiltonian(length, flux=2 * math.pi / 3, potential=0.0):
    """
    Build the open Hofstadter model in FGS's half-sized Nambu convention.

    Sites have index y * length + x, with x rightward and y upward.
    The published phase is exp(-i * flux * x) on c[x,y]^dagger c[x,y+1].
    The pinning site (0, length - 1) is the upper-left corner.
    """
    modes = length**2
    h = tc.backend.zeros((2 * modes, 2 * modes), dtype="complex128")
    for y in range(length):
        for x in range(length):
            site = y * length + x
            if x + 1 < length:
                h += tc.FGSSimulator.hopping(-1.0, site, site + 1, modes)
            if y + 1 < length:
                phase = tc.backend.exp(tc.backend.cast(-1j * flux * x, "complex128"))
                h += tc.FGSSimulator.hopping(-phase, site, site + length, modes)
    h += tc.FGSSimulator.chemical_potential(potential, length * (length - 1), modes)
    return h


def fixed_number_alpha(h, particles):
    """
    Explicitly occupy the lowest `particles` single-particle orbitals.

    C = alpha @ alpha^dagger = diag(I - rho, rho.T), where
    rho[i,j] = <c[j]^dagger c[i]>. A zero-chemical-potential FGS ground
    state would not, in general, have the requested particle number.
    """
    K = tc.backend
    modes = h.shape[0] // 2
    _, orbitals = K.eigh(2 * h[:modes, :modes])
    empty = orbitals[:, particles:]
    occupied = K.conj(orbitals[:, :particles])
    top = K.concat([empty, K.zeros((modes, particles), dtype="complex128")], axis=1)
    bottom = K.concat(
        [K.zeros((modes, modes - particles), dtype="complex128"), occupied], axis=1
    )
    return K.concat([top, bottom], axis=0)


def correlation_at_time(alpha, h, t):
    """Evolve through physical time t; hopping()/chemical_potential() include 1/2."""
    sim = tc.FGSSimulator(h.shape[0] // 2, alpha=alpha)
    sim.evol_hamiltonian(2 * t * h)
    return sim.get_cmatrix()


def simulate(length=12, particles=96, times=(0.0, 4.0, 8.0, 12.0)):
    """Return FGS correlations and the unpinned, fixed-number reference state."""
    h = hamiltonian(length)
    initial_h = h + tc.FGSSimulator.chemical_potential(
        -1.0, length * (length - 1), length**2
    )
    alpha = fixed_number_alpha(initial_h, particles)
    reference = tc.FGSSimulator(
        length**2, alpha=fixed_number_alpha(h, particles)
    ).get_cmatrix()
    evolve = tc.backend.jit(correlation_at_time)
    correlations = tc.backend.stack(
        [evolve(alpha, h, tc.backend.cast(t, "float64")) for t in times]
    )
    return h, alpha, reference, correlations


def plot_densities(times, delta_density, output):
    """Use the paper's shared 0..0.04 color scale; save unclipped data separately."""
    length = delta_density.shape[-1]
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), layout="constrained")
    for ax, t, density in zip(axes, times, delta_density):
        mesh = ax.pcolormesh(
            np.arange(length + 1),
            np.arange(length + 1),
            density,
            cmap="viridis_r",
            vmin=0.0,
            vmax=0.04,
            edgecolors=(0, 0, 0, 0.15),
            linewidth=0.35,
        )
        ax.set(aspect="equal", title=f"$t = {t:g}$", xlabel="$x$")
        ax.set_xticks(range(0, length + 1, 2))
        ax.set_yticks(range(0, length + 1, 2))
    axes[0].set_ylabel("$y$")
    fig.colorbar(mesh, ax=axes, shrink=0.8, label=r"$\delta n_i(t)$")
    fig.suptitle("Figure 2(a) · Chern-insulator edge dynamics with FGSSimulator")
    fig.savefig(output / "result.png", dpi=180)
    plt.close(fig)


def main():
    """Run the full-size benchmark and save densities with numerical diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    times = np.array([0.0, 4.0, 8.0, 12.0])
    started = time.perf_counter()
    h, _, reference, correlations = simulate(times=times)
    K = tc.backend
    modes = 144
    occupied = correlations[:, modes:, modes:]
    density = K.reshape(K.real(K.einsum("tii->ti", occupied)), (-1, 12, 12))
    reference_density = K.reshape(
        K.real(K.einsum("ii->i", reference[modes:, modes:])), (12, 12)
    )
    energies = K.real(K.einsum("ij,tij->t", 2 * h[:modes, :modes], occupied))
    purity_error = K.max(K.abs(correlations @ correlations - correlations))
    density, reference_density, energies, purity_error = [
        np.asarray(K.numpy(a))
        for a in (density, reference_density, energies, purity_error)
    ]
    elapsed = time.perf_counter() - started
    delta = density - reference_density
    np.savez_compressed(
        output / "densities.npz",
        times=times,
        density=density,
        reference_density=reference_density,
        delta_density=delta,
    )
    report = {
        "length": 12,
        "particles": 96,
        "times": times.tolist(),
        "flux": 2 * math.pi / 3,
        "pinning_potential": -1.0,
        "particle_number": density.sum(axis=(1, 2)).tolist(),
        "energy": energies.tolist(),
        "max_energy_drift": float(np.max(np.abs(energies - energies[0]))),
        "max_purity_error": float(purity_error),
        "delta_density_min": delta.min(axis=(1, 2)).tolist(),
        "delta_density_max": delta.max(axis=(1, 2)).tolist(),
        "calculation_seconds_including_jit": elapsed,
        "tensorcircuit_version": tc.__version__,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_densities(times, delta, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
