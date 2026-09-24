"""XXZ density of states, thermodynamics, and local response from MVPs."""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import tensorcircuit as tc

_DOS_BROADENING = 0.15
_GREEN_BROADENING = 0.10
_CHEBYSHEV_ORDER = 256
_NUM_PROBES = 256
_KRYLOV_DIM = 128


def _xxz_terms(num_sites: int, delta: Any) -> Tuple[List[List[int]], List[Any]]:
    """Return Pauli strings and weights of an open XXZ chain."""
    structures, weights = [], []
    for site in range(num_sites - 1):
        for pauli, weight in ((1, 1.0), (2, 1.0), (3, delta)):
            term = [0] * num_sites
            term[site] = term[site + 1] = pauli
            structures.append(term)
            weights.append(weight)
    return structures, weights


def _central_z(num_sites: int) -> List[List[int]]:
    """Return the Pauli string of Z on the central site."""
    term = [0] * num_sites
    term[num_sites // 2] = 3
    return [term]


def compute_with_spectral_api(
    num_sites: int = 8, deltas: Tuple[float, ...] = (0.5, 1.0, 1.5)
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run the jitted matrix-free spectral calculations for an XXZ scan."""
    dimension = 2**num_sites
    energy_bound = (num_sites - 1) * (2.0 + max(abs(delta) for delta in deltas))
    grid = {
        "num_sites": num_sites,
        "energy_bound": energy_bound,
        "energies": np.linspace(-energy_bound + 1e-5, energy_bound - 1e-5, 500),
        "beta": np.linspace(0.05, 2.5, 32),
        "frequencies": np.linspace(-4.0, 4.0, 400),
    }
    energies, beta, frequencies = (
        tc.backend.convert_to_tensor(grid[key], dtype=tc.rdtypestr)
        for key in ("energies", "beta", "frequencies")
    )
    probes = tc.spectral.random_trace_probes(
        num_probes=_NUM_PROBES, dimension=dimension
    )
    seed = tc.spectral.random_trace_probes(num_probes=1, dimension=dimension)[0]
    krylov = tc.matrixfunc.KrylovConfig(max_dim=min(_KRYLOV_DIM, dimension))
    chebyshev = tc.matrixfunc.ChebyshevConfig(
        order=_CHEBYSHEV_ORDER, bounds=(-energy_bound, energy_bound), kernel="jackson"
    )
    central_z = tc.quantum.PauliStringSum2MVP(_central_z(num_sites), [1.0])

    @tc.backend.jit
    def observables(delta: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        hamiltonian = tc.quantum.PauliStringSum2MVP(*_xxz_terms(num_sites, delta))
        measure = tc.matrixfunc.slq_measure(
            hamiltonian, probes, krylov, probe_batch_size=4
        )
        estimates = {
            "dos_slq": tc.spectral.density_of_states_from_slq(
                measure,
                energies,
                broadening=_DOS_BROADENING,
                normalization="states",
                with_std=True,
            ),
            "dos_kpm": tc.spectral.density_of_states(
                hamiltonian,
                energies,
                method=chebyshev,
                probes=probes,
                normalization="states",
                probe_batch_size=4,
                with_std=True,
            ),
            "thermal_energy": tc.spectral.thermal_energy(measure, beta, with_std=True),
            "heat_capacity": tc.spectral.heat_capacity(measure, beta, with_std=True),
        }
        ground_energy, ground_state = tc.matrixfunc.lanczos_lowest_eigenpair(
            hamiltonian, seed, krylov
        )
        residual = hamiltonian(ground_state) - ground_energy * ground_state
        response_state = central_z(ground_state)
        green = tc.spectral.zero_temperature_greens_function(
            hamiltonian,
            frequencies,
            ground_energy=ground_energy,
            particle_right=response_state,
            particle_left=response_state,
            broadening=_GREEN_BROADENING,
            method=krylov,
        )
        values = {key: mean for key, (mean, _) in estimates.items()}
        values.update(
            ground_energy=ground_energy,
            ground_residual=tc.backend.norm(residual),
            spectral_response=tc.spectral.spectral_function(green),
        )
        return values, {key: stderr for key, (_, stderr) in estimates.items()}

    results = []
    for delta in deltas:
        values, stderr = tc.backend.tree_map(
            np.asarray,
            observables(tc.backend.convert_to_tensor(delta, dtype=tc.rdtypestr)),
        )
        results.append({"delta": delta, "stderr": stderr, **values})
    return grid, results


def validate_with_ed(
    grid: Dict[str, Any], results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Check the estimates against exact diagonalization and return ED curves."""
    num_sites, bound = grid["num_sites"], grid["energy_bound"]
    energies, beta = grid["energies"], grid["beta"]
    orders = np.arange(_CHEBYSHEV_ORDER)
    scaled = energies[:, None] / bound
    kpm_coefficients = (
        np.where(orders == 0, 1.0, 2.0)
        * np.cos(np.arccos(scaled) * orders)
        / (np.pi * bound * np.sqrt(1.0 - scaled**2))
    )
    angle = np.pi / _CHEBYSHEV_ORDER
    jackson = (
        (_CHEBYSHEV_ORDER - orders) * np.cos(angle * orders)
        + np.sin(angle * orders) / np.tan(angle)
    ) / _CHEBYSHEV_ORDER
    central_z = np.asarray(
        tc.quantum.PauliStringSum2Dense(_central_z(num_sites), [1.0], numpy=True)
    )

    references = []
    for result in results:
        hamiltonian = np.asarray(
            tc.quantum.PauliStringSum2Dense(
                *_xxz_terms(num_sites, result["delta"]), numpy=True
            )
        )
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        moments = np.cos(np.arccos(eigenvalues[:, None] / bound) * orders).sum(axis=0)
        boltzmann = np.exp(-np.outer(beta, eigenvalues))
        boltzmann /= boltzmann.sum(axis=1, keepdims=True)
        thermal = boltzmann @ eigenvalues
        overlaps = eigenvectors.conj().T @ central_z @ eigenvectors[:, 0]
        green = np.abs(overlaps) ** 2 / (
            grid["frequencies"][:, None]
            + eigenvalues[0]
            + 1.0j * _GREEN_BROADENING
            - eigenvalues
        )
        reference = {
            "dos_slq": np.sum(
                _DOS_BROADENING
                / (
                    np.pi
                    * ((energies[:, None] - eigenvalues) ** 2 + _DOS_BROADENING**2)
                ),
                axis=1,
            ),
            "dos_kpm": kpm_coefficients @ (jackson * moments),
            "thermal_energy": thermal,
            "heat_capacity": beta**2 * (boltzmann @ eigenvalues**2 - thermal**2),
            "ground_energy": eigenvalues[0],
            "spectral_response": -np.imag(green.sum(axis=1)) / np.pi,
        }
        for key, stderr in result["stderr"].items():
            np.testing.assert_array_less(
                np.abs(result[key] - reference[key]), 4.0 * (stderr + 1e-8)
            )
        for key in ("ground_energy", "spectral_response"):
            np.testing.assert_allclose(
                result[key], reference[key], rtol=1e-8, atol=1e-8
            )
        np.testing.assert_allclose(result["ground_residual"], 0.0, atol=1e-8)
        references.append(reference)
    return references


def plot_results(
    grid: Dict[str, Any],
    results: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> None:
    """Plot the API estimates beside the exact finite-size references."""
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    beta_label = r"Inverse temperature $\beta$"
    panels = (
        (axes[0, 0], "energies", (("dos_kpm", "KPM"), ("dos_slq", "SLQ"))),
        (axes[0, 1], "beta", (("heat_capacity", ""),)),
        (axes[1, 0], "frequencies", (("spectral_response", ""),)),
        (axes[1, 1], "beta", (("thermal_energy", ""),)),
    )
    labels = (
        ("Energy", "Density of states"),
        (beta_label, "Heat capacity"),
        ("Frequency", r"$-\mathrm{Im}\,G^R/\pi$"),
        (beta_label, "Thermal energy"),
    )
    styles = (("-", "--"), (":", "-."))
    for (axis, grid_key, curves), (xlabel, ylabel) in zip(panels, labels):
        x = grid[grid_key]
        for (key, method), (style, ed_style) in zip(curves, styles):
            for index, (result, reference) in enumerate(zip(results, references)):
                color = f"C{index}"
                label = rf"{method} $\Delta={result['delta']:g}$".strip()
                axis.plot(x, result[key], style, color=color, label=label)
                axis.plot(
                    x,
                    reference[key],
                    ed_style,
                    color=color,
                    alpha=0.65,
                    label=f"ED {method}".strip() if index == 0 else None,
                )
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.legend(frameon=False, fontsize=7)
    figure.suptitle(f"Open XXZ chain, L={grid['num_sites']}")
    plt.show()


def main() -> None:
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.backend.set_random_state(2026)
    grid, results = compute_with_spectral_api()
    plot_results(grid, results, validate_with_ed(grid, results))


if __name__ == "__main__":
    main()
