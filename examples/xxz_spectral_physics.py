"""
Finite-size XXZ spectral observables with the matrix-function API.

The scan is deliberately small enough for exact diagonalization, so every
matrix-function observable is also compared with a Lehmann reference. The
exact diagonalization is used only for this validation, while KPM, SLQ, and
Krylov methods produce the plotted approximate results.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import tensorcircuit as tc

plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
)


def xxz_hamiltonian(num_sites: int, delta: float) -> np.ndarray:
    """
    Build an open-boundary XXZ Hamiltonian with ``Jxy=1`` and ``Jz=delta``.
    """
    graph = tc.templates.graphs.Line1D(num_sites, pbc=False)
    hamiltonian = tc.quantum.heisenberg_hamiltonian(
        graph,
        hxx=1.0,
        hyy=1.0,
        hzz=delta,
        sparse=False,
        numpy=True,
    )
    return np.asarray(hamiltonian, dtype=np.complex128)


def local_z_operator(num_sites: int, site: int) -> np.ndarray:
    """
    Return the dense Pauli-Z operator acting on one site.
    """
    pauli_string = [0] * num_sites
    pauli_string[site] = 3
    operator = tc.quantum.PauliStringSum2Dense([pauli_string], [1.0], numpy=True)
    return np.asarray(operator, dtype=np.complex128)


def broadened_dos(
    energies: np.ndarray, queries: np.ndarray, width: float
) -> np.ndarray:
    """
    Evaluate the exact Lorentzian-broadened finite-size DOS.
    """
    difference = queries[:, None] - energies[None, :]
    return np.sum(
        width / (np.pi * (difference**2 + width**2)),
        axis=1,
    )


def run_xxz_scan(
    num_sites: int = 6,
    deltas: Tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5),
    output_path: Optional[str] = None,
) -> None:
    """
    Compute and plot XXZ observables with the matrix-function API.

    :param num_sites: Number of sites in the open chain used for the main scan.
    :param deltas: XXZ anisotropies shown in the observable panels.
    :param output_path: Optional PNG path for the six-panel figure.
    """
    tc.set_dtype("complex128")
    dimension = 2**num_sites
    # Use a complete, rescaled computational basis here. This keeps the
    # finite-size example deterministic: the stochastic interfaces reduce to
    # exact traces, so the printed errors test the matrix-function truncation
    # rather than typicality noise from a small random probe ensemble.
    probes = np.sqrt(dimension) * np.eye(dimension, dtype=np.complex128)
    # For this six-site chain, 20 vectors cover every magnetization sector
    # (the largest sector has dimension 20) while keeping the example quick.
    krylov = tc.matrixfunc.KrylovConfig(max_dim=min(20, dimension))
    beta = np.linspace(0.05, 2.5, 24)
    frequency = np.linspace(-4.0, 4.0, 400)
    dos_width = 0.12
    green_width = 0.10

    hamiltonians: List[np.ndarray] = []
    eigensystems = []
    for delta in deltas:
        hamiltonian = xxz_hamiltonian(num_sites, delta)
        hamiltonians.append(hamiltonian)
        eigensystems.append(np.linalg.eigh(hamiltonian))

    lowest = min(float(values[0][0]) for values in eigensystems)
    highest = max(float(values[0][-1]) for values in eigensystems)
    padding = 0.08 * max(1.0, highest - lowest)
    bounds = (lowest - padding, highest + padding)
    energy = np.linspace(bounds[0] + 1.0e-5, bounds[1] - 1.0e-5, 500)
    kpm = tc.matrixfunc.ChebyshevConfig(
        order=128,
        bounds=bounds,
        kernel="jackson",
    )

    dos_kpm_values = []
    dos_slq_values = []
    exact_dos_values = []
    thermal_values = []
    exact_thermal_values = []
    heat_values = []
    exact_heat_values = []
    spectral_values = []
    exact_spectral_values = []

    for delta, hamiltonian, (eigenvalues, eigenvectors) in zip(
        deltas, hamiltonians, eigensystems
    ):
        measure = tc.matrixfunc.slq_measure(
            hamiltonian,
            probes,
            krylov,
            probe_batch_size=4,
        )
        dos_kpm = tc.spectral.density_of_states(
            hamiltonian,
            energy,
            method=kpm,
            probes=probes,
            normalization="states",
            probe_batch_size=4,
        )
        dos_kpm_values.append(np.asarray(tc.backend.numpy(dos_kpm)))
        dos_slq = tc.spectral.density_of_states(
            hamiltonian,
            energy,
            method=krylov,
            probes=probes,
            broadening=dos_width,
            normalization="states",
            probe_batch_size=4,
        )
        dos_slq_values.append(np.asarray(tc.backend.numpy(dos_slq)))
        exact_dos_values.append(broadened_dos(eigenvalues, energy, dos_width))

        thermal = tc.spectral.thermal_energy(measure, beta)
        heat = tc.spectral.heat_capacity(measure, beta)
        thermal_values.append(np.asarray(tc.backend.numpy(thermal)))
        heat_values.append(np.asarray(tc.backend.numpy(heat)))
        boltzmann = np.exp(-beta[:, None] * eigenvalues[None, :])
        exact_thermal = np.sum(boltzmann * eigenvalues[None, :], axis=1) / np.sum(
            boltzmann, axis=1
        )
        exact_heat = beta**2 * (
            np.sum(boltzmann * eigenvalues[None, :] ** 2, axis=1)
            / np.sum(boltzmann, axis=1)
            - exact_thermal**2
        )
        exact_thermal_values.append(exact_thermal)
        exact_heat_values.append(exact_heat)

        response = local_z_operator(num_sites, num_sites // 2) @ eigenvectors[:, 0]
        green = tc.spectral.zero_temperature_greens_function(
            hamiltonian,
            frequency,
            ground_energy=eigenvalues[0],
            particle_right=response,
            particle_left=response,
            broadening=green_width,
            method=krylov,
        )
        spectral_values.append(
            np.asarray(tc.backend.numpy(tc.spectral.spectral_function(green)))
        )
        response_weights = np.abs(eigenvectors.conj().T @ response) ** 2
        exact_green = np.sum(
            response_weights[None, :]
            / (
                frequency[:, None]
                + eigenvalues[0]
                + 1.0j * green_width
                - eigenvalues[None, :]
            ),
            axis=1,
        )
        exact_spectral_values.append(-np.imag(exact_green) / np.pi)
        thermal_error = np.max(np.abs(thermal_values[-1] - exact_thermal))
        heat_error = np.max(np.abs(heat_values[-1] - exact_heat))
        dos_error = np.max(np.abs(dos_slq_values[-1] - exact_dos_values[-1]))
        print(
            f"Delta={delta:+.2f}: max thermal error={thermal_error:.3e}, "
            f"max heat-capacity error={heat_error:.3e}, "
            f"max broadened-DOS error={dos_error:.3e}"
        )

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for index, delta in enumerate(deltas):
        color = colors[index % len(colors)]
        labels = ("KPM", "SLQ", "Lehmann") if index == 0 else (None,) * 3
        for values, style, label, alpha in (
            (dos_kpm_values[index], "-", labels[0], 1.0),
            (dos_slq_values[index], ":", labels[1], 0.8),
            (exact_dos_values[index], "--", labels[2], 0.55),
        ):
            axes[0, 0].plot(
                energy, values, style, color=color, alpha=alpha, label=label
            )
        for axis, x_values, values, exact, label in (
            (
                axes[0, 1],
                frequency,
                spectral_values[index],
                exact_spectral_values[index],
                rf"$\Delta={delta:g}$",
            ),
            (
                axes[1, 0],
                beta,
                thermal_values[index],
                exact_thermal_values[index],
                None,
            ),
            (axes[1, 1], beta, heat_values[index], exact_heat_values[index], None),
        ):
            axis.plot(x_values, values, color=color, label=label)
            axis.plot(x_values, exact, "--", color=color, alpha=0.55)

    for axis, xlabel, ylabel in (
        (axes[0, 0], "Energy", r"$\rho(E)$"),
        (axes[0, 1], "Frequency", r"$-\operatorname{Im}G^R/\pi$"),
        (axes[1, 0], r"$\beta$", r"$\langle H\rangle_\beta$"),
        (axes[1, 1], r"$\beta$", r"$C_\beta$"),
    ):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[0, 1].legend(frameon=False, loc="upper left")

    panel_labels = "abcdef"
    for axis, label in zip(axes.flat, panel_labels):
        axis.text(
            -0.16,
            1.04,
            f"({label})",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        axis.tick_params(direction="in", which="both")

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved figure to {output_path}")
    plt.show()


if __name__ == "__main__":
    run_xxz_scan(output_path=str(Path(__file__).with_name("xxz_spectral_physics.png")))
