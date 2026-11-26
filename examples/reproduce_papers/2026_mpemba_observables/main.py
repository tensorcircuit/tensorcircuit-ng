"""
Reproduction of "Detection of Mpemba Effect through Observables in Open Quantum Systems"
Link: https://arxiv.org/abs/2512.02709

Description:
This script reproduces Figure 1(a–d) from the paper using TensorCircuit-NG.
"""

from functools import reduce
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div

import tensorcircuit as tc

SITES = 9
HOPPING = 1.0
DEPHASING = 5.0
OUTPUT = Path(__file__).resolve().parent / "outputs"


def initial_populations():
    localized = np.eye(SITES)[SITES // 2]
    spread = np.ones(SITES) / (SITES - 1)
    spread[-1] = 0
    return np.stack([localized, spread])


def pauli_generator():
    """Encode the site index on four qubits and build the Lindblad MVP once."""
    qubits = (SITES - 1).bit_length()
    dimension = 2**qubits
    hopping = HOPPING * np.diag(np.ones(SITES - 1), 1)
    hamiltonian = np.pad(hopping + hopping.T, (0, dimension - SITES))
    paulis = [np.asarray(gate.tensor) for gate in tc.gates.pauli_gates]
    words, weights = [], []
    for word in product(range(4), repeat=qubits):
        matrix = reduce(np.kron, (paulis[i] for i in word))
        coefficient = np.vdot(matrix, hamiltonian).real / dimension
        if coefficient:
            words.append(word)
            weights.append(coefficient)

    # Averaging Z strings removes off-diagonal entries in the site basis.
    jumps = [
        ([word], [np.sqrt(DEPHASING / dimension)])
        for word in product((0, 3), repeat=qubits)
        if any(word)
    ]
    strings, coefficients = tc.quantum.Lindblad2PauliStringSum((words, weights), jumps)
    return tc.quantum.PauliStringSum2MVP(strings, coefficients), dimension


def evolve(times, initial):
    mvp, dimension = pauli_generator()

    @tc.backend.jit
    def solve(rho0):
        return tc.timeevol.mesolve(
            lambda vector, _time: mvp(vector),
            rho0,
            times,
            mode="raw",
            rtol=1e-11,
            atol=1e-13,
        )

    padded = np.pad(initial, ((0, 0), (0, dimension - SITES)))
    return np.stack([np.asarray(solve(np.diag(pop))) for pop in padded])


def effective_populations(times, initial):
    """Eq. (7) is a Lindblad walk with jumps in both directions on each bond."""
    basis = np.eye(SITES)
    jumps = []
    for left, right in zip(basis[:-1], basis[1:]):
        jump = np.sqrt(2 * HOPPING**2 / DEPHASING) * np.outer(left, right)
        jumps.extend((jump, jump.T))
    states = [
        tc.timeevol.mesolve(
            np.zeros((SITES, SITES)), np.diag(pop), times, jumps, rtol=1e-11, atol=1e-13
        )
        for pop in initial
    ]
    return np.diagonal(np.asarray(states), axis1=-2, axis2=-1).real


def entropy_deficit(spectrum):
    """Relative entropy to I/L, including the zero-eigenvalue limit."""
    probabilities = np.maximum(spectrum, 0.0)
    return np.sum(kl_div(probabilities, 1 / SITES), axis=-1)


def plot(times, populations, walk, exact, effective):
    colors = ("#245a9a", "#ba4b28")
    labels = ("Center-localized", "Spread over sites 1–8")
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), layout="constrained")
    entropy_ax = axes[0, 0]
    for i, (color, label) in enumerate(zip(colors, labels)):
        entropy_ax.semilogy(times, exact[i], color=color, label=label)
        entropy_ax.semilogy(times, effective[i], "--", color=color, linewidth=1.1)
    entropy_ax.plot([], [], color="0.25", label="Lindblad")
    entropy_ax.plot([], [], "--", color="0.25", label="Effective walk")
    entropy_ax.legend(fontsize=8, frameon=False, loc="lower left")
    entropy_ax.set(title="(a)  Distance to the steady state", ylabel=r"$\ln 9-S(\rho)$")

    for ax, site, panel in zip(axes.flat[1:], (1, 3, 5), "bcd"):
        twin = ax.twinx() if site != 1 else ax
        inset = twin.inset_axes([0.52, 0.32, 0.44, 0.42])
        for curves, style in ((populations, "-"), (walk, "--")):
            for pop, color, scale in zip(curves, colors, (ax, twin)):
                occupation = pop[:, site - 1]
                scale.plot(times, occupation, style, color=color, linewidth=1.3)
                inset.semilogy(
                    times,
                    np.abs(occupation - 1 / SITES),
                    style,
                    color=color,
                    linewidth=1,
                )
        ax.axhline(1 / SITES, color="0.45", linewidth=0.7, linestyle=":")
        ax.set(title=f"({panel})  Site {site}", ylabel=rf"$\langle n_{site}\rangle$")
        if twin is not ax:
            ax.tick_params(axis="y", colors=colors[0])
            twin.tick_params(axis="y", colors=colors[1])
        inset.set(title=rf"$|\langle n_{site}\rangle-1/9|$", xticks=[0, 50])
        inset.title.set_fontsize(8)
        inset.tick_params(labelsize=7)
        inset.set_xlim(times[0], times[-1])
    for ax in axes.flat:
        ax.set(xlim=(times[0], times[-1]), xticks=[0, 15, 30, 45, 60], xlabel=r"$Jt$")
    fig.savefig(OUTPUT / "result.png", dpi=200)
    plt.close(fig)


def main():
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    times = np.linspace(0, 70, 1401)
    initial = initial_populations()
    states = evolve(times, initial)[:, :, :SITES, :SITES]
    populations = np.diagonal(states, axis1=-2, axis2=-1).real
    exact = entropy_deficit(np.linalg.eigvalsh(states))
    walk = effective_populations(times, initial)
    effective = entropy_deficit(walk)
    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT / "curves.npz",
        times=times,
        populations=populations,
        effective_populations=walk,
        entropy_deficit=exact,
        effective_entropy_deficit=effective,
    )
    plot(times, populations, walk, exact, effective)


if __name__ == "__main__":
    main()
