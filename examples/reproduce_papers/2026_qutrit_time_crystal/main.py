"""Reproduction of "A Qutrit Time Crystal Stabilized with Native Chiral Interactions"
Link: https://arxiv.org/abs/2605.14293
Description:
This script reproduces Figure 2(b) from the paper using TensorCircuit-NG.
It performs a noiseless JAX simulation of the driven qutrit Floquet model,
scaled from 15 qutrits to 9 qutrits for local execution while preserving the
period-tripling versus thermal crossover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

DIM = 3
N_SITES = 9
ORIGINAL_N_SITES = 15
NUM_RANDOM_STATES = 10
NUM_FLOQUET_CYCLES = 40
G_DTC = 0.98
G_THERMAL = 0.60
G_VALUES = np.linspace(0.55, 1.05, 25)
PAPER_CROSSOVER = 0.86
KICK_PHASES = jnp.array([0.0, 2 * jnp.pi / 3, -2 * jnp.pi / 3], dtype=jnp.float64)
OMEGA3 = np.exp(2j * np.pi / 3)
Z_DIAG = jnp.array([1.0, OMEGA3, OMEGA3**2], dtype=jnp.complex128)
M_DIAG = jnp.array([1.0, 0.0, -1.0], dtype=jnp.complex128)


@dataclass(frozen=True)
class ModelParameters:
    """Disorder parameters of the driven qutrit Floquet model."""

    j: jnp.ndarray
    theta: jnp.ndarray
    j_prime: jnp.ndarray
    theta_prime: jnp.ndarray
    h: jnp.ndarray
    phi: jnp.ndarray


def wrap_to_pi(values: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to the interval (-pi, pi]."""
    return (values + jnp.pi) % (2 * jnp.pi) - jnp.pi


def qutrit_dft() -> jnp.ndarray:
    """Return the qutrit discrete Fourier transform."""
    matrix = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, OMEGA3, OMEGA3**2],
            [1.0, OMEGA3**2, OMEGA3],
        ],
        dtype=jnp.complex128,
    )
    return matrix / jnp.sqrt(3.0)


def fractional_shift_unitary(g: jnp.ndarray) -> jnp.ndarray:
    """Smoothly interpolate the qutrit shift operator X^g via its spectrum."""
    fourier = qutrit_dft()
    eigenvalues = jnp.exp(1j * KICK_PHASES * g)
    return fourier @ jnp.diag(eigenvalues) @ jnp.conj(fourier.T)


def local_field_unitary(h_value: float, phi_value: float) -> jnp.ndarray:
    """Single-qutrit unitary exp(-i h e^{i phi} Z + h.c.)."""
    energy = 2.0 * h_value * jnp.real(jnp.exp(1j * phi_value) * Z_DIAG)
    return jnp.diag(jnp.exp(-1j * energy))


def pair_interaction_tensor(
    j_value: float,
    theta_value: float,
    j_prime_value: float,
    theta_prime_value: float,
) -> jnp.ndarray:
    """Two-qutrit diagonal tensor for exp(-i H_int bond)."""
    z_left = Z_DIAG[:, None]
    z_right = Z_DIAG[None, :]
    energy = (
        2.0 * j_value * jnp.real(jnp.exp(1j * theta_value) * z_left * jnp.conj(z_right))
    )
    energy += (
        2.0
        * j_prime_value
        * jnp.real(jnp.exp(1j * theta_prime_value) * z_left * z_right)
    )
    values = jnp.exp(-1j * energy).reshape(-1)
    return jnp.diag(values).reshape(DIM, DIM, DIM, DIM)


def build_basis_digits(n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Return base-3 digits and powers for all computational basis states."""
    powers = (DIM ** np.arange(n_sites - 1, -1, -1)).astype(int)
    indices = np.arange(DIM**n_sites)
    digits = np.stack([(indices // power) % DIM for power in powers], axis=1)
    return digits, powers


def sample_model_parameters(n_sites: int, seed: int = 260514293) -> ModelParameters:
    """Sample a single disorder realization within the manuscript parameter windows."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)

    j = jax.random.uniform(keys[0], (n_sites - 1,), minval=0.08, maxval=0.25)
    theta_mod = jax.random.uniform(keys[1], (n_sites - 1,), minval=0.125, maxval=0.9)
    theta_sector = jax.random.randint(keys[2], (n_sites - 1,), 0, 6)
    theta = wrap_to_pi(theta_mod + theta_sector * (jnp.pi / 3))

    j_prime = jax.random.uniform(keys[3], (n_sites - 1,), minval=0.10, maxval=0.30)
    theta_prime = jax.random.uniform(
        keys[4], (n_sites - 1,), minval=-jnp.pi, maxval=jnp.pi
    )
    h = jax.random.uniform(keys[5], (n_sites,), minval=0.10, maxval=0.30)
    phi = jax.random.uniform(keys[6], (n_sites,), minval=-jnp.pi, maxval=jnp.pi)

    return ModelParameters(
        j=j,
        theta=theta,
        j_prime=j_prime,
        theta_prime=theta_prime,
        h=h,
        phi=phi,
    )


def build_phase_vector(
    parameters: ModelParameters,
    basis_digits: np.ndarray,
) -> jnp.ndarray:
    """Build the diagonal phase exp(-i(H0 + Hint)) in the computational basis."""
    digits = jnp.asarray(basis_digits)
    z_by_site = Z_DIAG[digits]

    onsite = 2.0 * jnp.sum(
        parameters.h[None, :]
        * jnp.real(jnp.exp(1j * parameters.phi)[None, :] * z_by_site),
        axis=1,
    )
    zz_chiral = z_by_site[:, :-1] * jnp.conj(z_by_site[:, 1:])
    zz_symmetry_breaking = z_by_site[:, :-1] * z_by_site[:, 1:]

    interaction = 2.0 * jnp.sum(
        parameters.j[None, :]
        * jnp.real(jnp.exp(1j * parameters.theta)[None, :] * zz_chiral),
        axis=1,
    )
    interaction += 2.0 * jnp.sum(
        parameters.j_prime[None, :]
        * jnp.real(
            jnp.exp(1j * parameters.theta_prime)[None, :] * zz_symmetry_breaking
        ),
        axis=1,
    )
    return jnp.exp(-1j * (onsite + interaction))


def state_index(trits: np.ndarray, powers: np.ndarray) -> int:
    """Map a product-state trit string to its flattened basis index."""
    return int(np.dot(trits, powers))


def build_initial_state_batch(
    n_sites: int,
    powers: np.ndarray,
    num_random_states: int = NUM_RANDOM_STATES,
    seed: int = 7,
) -> tuple[list[str], jnp.ndarray]:
    """Return 3 ferromagnetic and 10 random product states as a dense batch."""
    dim = DIM**n_sites
    labels: list[str] = []
    vectors: list[np.ndarray] = []

    for state_value in range(DIM):
        trits = np.full(n_sites, state_value, dtype=int)
        label = f"FM {state_value}"
        vector = np.zeros(dim, dtype=np.complex128)
        vector[state_index(trits, powers)] = 1.0
        labels.append(label)
        vectors.append(vector)

    rng = np.random.default_rng(seed)
    seen = {tuple(np.full(n_sites, value, dtype=int)) for value in range(DIM)}
    while len(vectors) < DIM + num_random_states:
        trits = tuple(rng.integers(0, DIM, size=n_sites).tolist())
        if trits in seen:
            continue
        seen.add(trits)
        vector = np.zeros(dim, dtype=np.complex128)
        vector[state_index(np.asarray(trits), powers)] = 1.0
        labels.append("".join(str(x) for x in trits))
        vectors.append(vector)

    return labels, jnp.asarray(np.stack(vectors))


def apply_local_unitary_batch(
    psi_batch: jnp.ndarray,
    unitary: jnp.ndarray,
    site: int,
    n_sites: int,
) -> jnp.ndarray:
    """Apply a one-qutrit unitary to a batch of dense qutrit statevectors."""
    reshaped = psi_batch.reshape((psi_batch.shape[0],) + (DIM,) * n_sites)
    reshaped = jnp.moveaxis(reshaped, site + 1, 1)
    reshaped = jnp.tensordot(unitary, reshaped, axes=[[1], [1]])
    reshaped = jnp.moveaxis(reshaped, 0, 1)
    reshaped = jnp.moveaxis(reshaped, 1, site + 1)
    return reshaped.reshape(psi_batch.shape[0], -1)


def apply_global_kick_batch(
    psi_batch: jnp.ndarray,
    kick_unitary: jnp.ndarray,
    n_sites: int,
) -> jnp.ndarray:
    """Apply the global kick X^g to every site."""
    for site in range(n_sites):
        psi_batch = apply_local_unitary_batch(psi_batch, kick_unitary, site, n_sites)
    return psi_batch


def one_cycle_batch(
    psi_batch: jnp.ndarray,
    g: jnp.ndarray,
    phase_vector: jnp.ndarray,
    n_sites: int,
) -> jnp.ndarray:
    """Apply a single Floquet cycle to a batch of states."""
    kicked = apply_global_kick_batch(psi_batch, fractional_shift_unitary(g), n_sites)
    return kicked * phase_vector[None, :]


def build_evolution_fn(n_sites: int, n_steps: int):
    """Create a JAX-compiled Floquet evolution function."""

    @jax.jit
    def evolve_batch(
        g: jnp.ndarray,
        psi_batch: jnp.ndarray,
        phase_vector: jnp.ndarray,
        magnetization_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        def body(carry: jnp.ndarray, _: None) -> tuple[jnp.ndarray, jnp.ndarray]:
            state = one_cycle_batch(carry, g, phase_vector, n_sites)
            magnetization = jnp.real(
                jnp.sum(
                    jnp.conj(state) * state * magnetization_vector[None, :],
                    axis=1,
                )
            )
            return state, magnetization

        _, magnetizations = jax.lax.scan(body, psi_batch, xs=None, length=n_steps)
        return magnetizations

    return evolve_batch


def verify_one_cycle_against_qudit_circuit(
    parameters: ModelParameters,
    g: float = 0.97,
) -> float:
    """Cross-check one cycle of the dense update against tc.QuditCircuit."""
    verify_n = 4
    verify_parameters = ModelParameters(
        j=parameters.j[: verify_n - 1],
        theta=parameters.theta[: verify_n - 1],
        j_prime=parameters.j_prime[: verify_n - 1],
        theta_prime=parameters.theta_prime[: verify_n - 1],
        h=parameters.h[:verify_n],
        phi=parameters.phi[:verify_n],
    )
    digits, powers = build_basis_digits(verify_n)
    phase_vector = build_phase_vector(verify_parameters, digits)
    initial_trits = np.asarray([0, 2, 1, 0], dtype=int)
    psi0 = np.zeros(DIM**verify_n, dtype=np.complex128)
    psi0[state_index(initial_trits, powers)] = 1.0

    dense_result = np.asarray(
        one_cycle_batch(
            jnp.asarray(psi0[None, :]), jnp.asarray(g), phase_vector, verify_n
        )
    )[0]

    circuit = tc.QuditCircuit(verify_n, dim=DIM, inputs=psi0)
    kick_matrix = np.asarray(fractional_shift_unitary(jnp.asarray(g)))
    for site in range(verify_n):
        circuit.unitary(site, unitary=tc.gates.Gate(kick_matrix))
    for site in range(verify_n - 1):
        circuit.unitary(
            site,
            site + 1,
            unitary=tc.gates.Gate(
                pair_interaction_tensor(
                    float(verify_parameters.j[site]),
                    float(verify_parameters.theta[site]),
                    float(verify_parameters.j_prime[site]),
                    float(verify_parameters.theta_prime[site]),
                )
            ),
        )
    for site in range(verify_n):
        circuit.unitary(
            site,
            unitary=tc.gates.Gate(
                local_field_unitary(
                    float(verify_parameters.h[site]),
                    float(verify_parameters.phi[site]),
                )
            ),
        )

    reference = np.asarray(tc.backend.numpy(circuit.wavefunction()))
    return float(np.max(np.abs(dense_result - reference)))


def compute_spectroscopy(
    evolve_batch,
    g_values: np.ndarray,
    psi0_batch: jnp.ndarray,
    phase_vector: jnp.ndarray,
    magnetization_vector: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FFT spectra and the 2pi/3 peak across a kick-strength sweep."""
    spectra = []
    peak_line = []
    frequencies = 2 * np.pi * np.fft.rfftfreq(NUM_FLOQUET_CYCLES, d=1.0)
    target_index = int(np.argmin(np.abs(frequencies - 2 * np.pi / 3)))

    for g_value in g_values:
        traces = np.asarray(
            evolve_batch(
                jnp.asarray(g_value),
                psi0_batch,
                phase_vector,
                magnetization_vector,
            )
        )
        fft_values = np.abs(np.fft.rfft(traces, axis=0)) / NUM_FLOQUET_CYCLES
        spectrum = fft_values.mean(axis=1)
        spectra.append(spectrum)
        peak_line.append(spectrum[target_index])

    return frequencies, np.stack(spectra, axis=1), np.asarray(peak_line)


def plot_result(
    dtc_traces: np.ndarray,
    thermal_traces: np.ndarray,
    frequencies: np.ndarray,
    spectra: np.ndarray,
    peak_line: np.ndarray,
) -> None:
    """Render a composite figure matching the structure of Fig. 2(b)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.5, 5.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.15, 1.0],
        height_ratios=[0.33, 0.67],
        wspace=0.30,
        hspace=0.14,
    )
    ax_left = fig.add_subplot(grid[:, 0])
    ax_peak = fig.add_subplot(grid[0, 1])
    ax_fft = fig.add_subplot(grid[1, 1], sharex=ax_peak)

    cycles = np.arange(1, NUM_FLOQUET_CYCLES + 1)
    for trace in dtc_traces.T:
        ax_left.plot(cycles, trace, color="#2468B1", alpha=0.22, linewidth=1.0)
    for trace in thermal_traces.T:
        ax_left.plot(cycles, trace, color="#C73E1D", alpha=0.22, linewidth=1.0)
    ax_left.plot(
        cycles,
        dtc_traces.mean(axis=1),
        color="#114D8A",
        linewidth=2.2,
        label=f"DTC mean, g={G_DTC:.2f}",
    )
    ax_left.plot(
        cycles,
        thermal_traces.mean(axis=1),
        color="#8E2510",
        linewidth=2.2,
        label=f"Thermal mean, g={G_THERMAL:.2f}",
    )
    ax_left.set_xlabel("Floquet cycle")
    ax_left.set_ylabel(r"$\langle M_{\mathrm{mid}}(t) \rangle$")
    ax_left.set_title(
        f"Scaled Noiseless Traces ({N_SITES} qutrits, original {ORIGINAL_N_SITES})"
    )
    ax_left.legend(frameon=False, loc="upper right")
    ax_left.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    ax_peak.plot(G_VALUES, peak_line, color="#111111", linewidth=2.0)
    ax_peak.set_xlabel("Kick strength g")
    ax_peak.set_ylabel(r"FFT at $2\pi/3$")
    ax_peak.set_title(r"$\omega = 2\pi/3$ Peak")
    ax_peak.grid(alpha=0.20, linestyle="--", linewidth=0.6)

    image = ax_fft.imshow(
        spectra,
        origin="lower",
        aspect="auto",
        extent=(G_VALUES[0], G_VALUES[-1], frequencies[0], frequencies[-1]),
        cmap="magma",
    )
    ax_fft.axhline(2 * np.pi / 3, color="white", linestyle="--", linewidth=1.0)
    ax_fft.set_xlabel("Kick strength g")
    ax_fft.set_ylabel(r"Frequency $\omega$")
    colorbar = fig.colorbar(image, ax=ax_fft, pad=0.02)
    colorbar.set_label("Mean FFT amplitude")

    fig.suptitle(
        "Fig. 2(b) Reproduction: Noiseless Driven Chiral Clock Model",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.94, top=0.88, bottom=0.12)
    fig.savefig(OUTPUT_DIR / "result.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the scaled no-noise Fig. 2(b) reproduction."""
    parameters = sample_model_parameters(N_SITES)
    verification_error = verify_one_cycle_against_qudit_circuit(parameters)
    if verification_error > 1e-10:
        raise ValueError(
            f"One-cycle verification against tc.QuditCircuit failed: {verification_error}"
        )

    basis_digits, powers = build_basis_digits(N_SITES)
    phase_vector = build_phase_vector(parameters, basis_digits)
    _, psi0_batch = build_initial_state_batch(N_SITES, powers)
    magnetization_vector = M_DIAG[jnp.asarray(basis_digits)[:, N_SITES // 2]]
    evolve_batch = build_evolution_fn(N_SITES, NUM_FLOQUET_CYCLES)

    dtc_traces = np.asarray(
        evolve_batch(jnp.asarray(G_DTC), psi0_batch, phase_vector, magnetization_vector)
    )
    thermal_traces = np.asarray(
        evolve_batch(
            jnp.asarray(G_THERMAL),
            psi0_batch,
            phase_vector,
            magnetization_vector,
        )
    )
    frequencies, spectra, peak_line = compute_spectroscopy(
        evolve_batch,
        G_VALUES,
        psi0_batch,
        phase_vector,
        magnetization_vector,
    )
    plot_result(dtc_traces, thermal_traces, frequencies, spectra, peak_line)

    np.savez(
        OUTPUT_DIR / "raw_data.npz",
        g_values=G_VALUES,
        frequencies=frequencies,
        spectra=spectra,
        peak_line=peak_line,
        dtc_traces=dtc_traces,
        thermal_traces=thermal_traces,
    )
    print(f"Verification max error against tc.QuditCircuit: {verification_error:.3e}")
    print(f"Saved figure to {OUTPUT_DIR / 'result.png'}")
    print(
        "Simplifications: 15 qutrits -> 9 qutrits, noiseless dynamics, and the kick "
        "X^g is modeled by a smooth spectral interpolation of the qutrit shift."
    )


if __name__ == "__main__":
    main()
