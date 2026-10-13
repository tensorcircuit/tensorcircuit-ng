"""
Reproduction of "Entanglement Growth from Entangled States:
A Unified Perspective on Entanglement Generation and Transport"
Link: https://arxiv.org/abs/2510.08344

Description:
Reproduction of Figures 1(d), 1(e), and 3 at L=12 with 72 disorder samples.
Run main.py to generate outputs/summary.npz and outputs/result.png;
use --plot-only to redraw the generated data.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations
from math import comb
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma

import tensorcircuit as tc

L = 12
SAMPLES = 72
SEED = 20260917
TAU = np.r_[
    np.arange(0, 3.25, 0.25),
    [3.3, 3.6, 3.9, 4.2, 4.5],
    np.arange(5, 10.5, 0.5),
    [11, 12.2, 13.7, 15.7, 19, 24, 32, 500],
]
DEPTH = 2000
REALIZATIONS = 5
BAEE_TIMES = np.linspace(0, 50, 101)
OUTPUT = Path(__file__).resolve().parent / "outputs"
PROTOCOLS = {
    "thermal": (np.pi / 2, np.pi),
    "A": (0.0, np.pi),
    "B": (np.pi / 2, 0.0),
    "C": (np.pi, 0.0),
    "D": (np.pi, np.pi / 2),
}
COLORS = ["#e31a1c", "#33a02c", "#1f78b4", "#fdbf6f", "#ff7f00", "#6a3d9a"]


def sector_indices(L):
    if L < 2 or L % 2:
        raise ValueError("L must be positive, even, and at least 2")
    return np.array([i for i in range(2**L) if bin(i).count("1") == L // 2])


def hamiltonian(L, fields, jz=0.5, jxy=1.0):
    """Eq. (1): S=Pauli/2, OBC, restricted to total Sz=0."""
    terms, weights = [], []
    for i in range(L - 1):
        for code, weight in [(1, jxy / 4), (2, jxy / 4), (3, jz / 4)]:
            if weight:
                p = [0] * L
                p[i] = p[i + 1] = code
                terms.append(p)
                weights.append(weight)
    for i, field in enumerate(fields):
        p = [0] * L
        p[i] = 3
        terms.append(p)
        weights.append(field / 2)
    full = tc.quantum.PauliStringSum2COO(terms, weights, numpy=True).tocsr()
    idx = sector_indices(L)
    return tc.backend.convert_to_tensor(full[idx][:, idx].real.toarray())


class Evolution:
    def __init__(self, H):
        self.e, self.v = tc.backend.eigh(H)

    def at(self, psi, times):
        """Yield states without retaining all time snapshots in memory."""
        K = tc.backend
        coeff = K.transpose(K.conj(self.v)) @ psi
        for t in times:
            # Phase reduction cannot remove double-precision eigenvalue error.
            angle = K.mod(
                K.cast(self.e, "longdouble") * K.cast(t, "longdouble"),
                2 * K.acos(K.cast(-1, "longdouble")),
            )
            phase = K.exp(-1j * K.cast(angle, "float64"))
            state = self.v @ (
                phase[:, None] * coeff if coeff.ndim == 2 else phase * coeff
            )
            norm = K.sqrt(K.sum(K.abs(state) ** 2, axis=0))
            yield state / norm


class Floquet:
    def __init__(self, L, fields, xy_evolution=None):
        xy = xy_evolution or Evolution(hamiltonian(L, np.zeros(L), jz=0))
        Uxy = (xy.v * tc.backend.exp(-0.4j * xy.e)) @ tc.backend.transpose(xy.v)
        diagonal = tc.backend.diagonal(hamiltonian(L, fields, jz=1, jxy=0))
        F = tc.backend.exp(-1j * diagonal)[:, None] * Uxy
        triangular, self.v = tc.backend.schur(F, output="complex")
        diagonal = tc.backend.diagonal(triangular)
        self.e = -tc.backend.atan2(tc.backend.imag(diagonal), tc.backend.real(diagonal))

    at = Evolution.at


@lru_cache(maxsize=32)
def gate(alpha, beta):
    """Eq. (2) using TC rotations exp(-i theta Pauli-product / 2)."""
    c = tc.Circuit(2)
    c.rxx(0, 1, theta=alpha / 2)
    c.ryy(0, 1, theta=alpha / 2)
    c.rzz(0, 1, theta=beta / 2)
    return c.matrix()


class JaxKernels:
    """Compile entropy batches and whole random-circuit trajectories once per L."""

    def __init__(self, length):
        self.K = tc.set_backend("jax", set_global=False)
        K = self.K
        self.length = length
        self.idx = sector_indices(length)
        self.cuts = [(0,) + x for x in combinations(range(1, length), length // 2 - 1)]
        half = self.cut_maps(tuple(range(length // 2)))
        self.half = K.jit(K.vmap(self.entropy_at, vectorized_argnums=0))
        self.half_maps = half

        inverse = np.full(2**length, -1, dtype=int)
        inverse[self.idx] = np.arange(len(self.idx))
        maps = []
        for bond in range(length - 1):
            p, q = length - 1 - bond, length - 2 - bond
            code = ((self.idx >> p) & 1) * 2 + ((self.idx >> q) & 1)
            opposite = (code == 1) | (code == 2)
            partner = np.arange(len(self.idx))
            partner[opposite] = inverse[self.idx[opposite] ^ (1 << p) ^ (1 << q)]
            maps.append((code, partner, opposite))
        code, partner, opposite = zip(*maps)
        self.code = K.convert_to_tensor(np.array(code))
        self.partner = K.convert_to_tensor(np.array(partner))
        self.opposite = K.convert_to_tensor(np.array(opposite))

        def apply(state, unitary, bond):
            codes = self.code[bond]
            diagonal = unitary[codes, codes]
            off = unitary[codes, 3 - codes] * self.opposite[bond]
            return diagonal * state + off * state[self.partner[bond]]

        apply_many = K.vmap(apply, vectorized_argnums=0)
        entropy_many = K.vmap(self.entropy_at, vectorized_argnums=0)

        def circuit(states, unitary, bonds):
            late = min(100, bonds.shape[0])
            states = K.scan(
                lambda carry, bond: apply_many(carry, unitary, bond),
                bonds[:-late],
                states,
            )

            def measure(carry, bond):
                state, total = carry
                state = apply_many(state, unitary, bond)
                return state, total + entropy_many(state, half)

            state, total = K.scan(
                measure, bonds[-late:], (states, K.zeros((states.shape[0],), "float64"))
            )
            return state, total / late

        self.circuit = K.jit(K.vmap(circuit, vectorized_argnums=2))

        def sum_cuts(states, maps):
            return K.scan(
                lambda total, cut: total + entropy_many(states, cut),
                maps,
                K.zeros((states.shape[0],), "float64"),
            )

        self.sum_cuts = K.jit(sum_cuts)

    def cut_maps(self, cut):
        """Static amplitude indices, grouped by subsystem charge."""
        rest = tuple(i for i in range(self.length) if i not in cut)

        def labels(sites):
            out = np.zeros(len(self.idx), dtype=int)
            for i in sites:
                out = out * 2 + ((self.idx >> (self.length - 1 - i)) & 1)
            return out

        a, b = labels(cut), labels(rest)
        counts = np.array([bin(int(x)).count("1") for x in a])
        result = []
        for charge in range(len(cut) + 1):
            mask = np.flatnonzero(counts == charge)
            aa, rows = np.unique(a[mask], return_inverse=True)
            bb, columns = np.unique(b[mask], return_inverse=True)
            indices = np.empty((len(aa), len(bb)), dtype=np.int32)
            indices[rows, columns] = mask
            result.append(indices)
        return tuple(result)

    def entropy_at(self, state, maps):
        """Charge-block form of TC entanglement entropy, without regularization."""
        K = self.K
        norm = K.sum(K.abs(state) ** 2)
        entropy = K.zeros((), "float64")
        for indices in maps:
            _, singular, _, _ = K.svd(state[indices], pivot_axis=1)
            probabilities = K.abs(singular) ** 2 / norm
            safe = K.where(probabilities > 0, probabilities, K.ones_like(probabilities))
            entropy -= K.sum(probabilities * K.log(safe)) / np.log(2)
        return entropy

    def hcee(self, states):
        return self.K.numpy(
            self.half(self.K.convert_to_tensor(states.T), self.half_maps)
        )

    def baee(self, states):
        cuts = self.cuts
        K = self.K
        states = K.convert_to_tensor(states.T)
        total = K.zeros((states.shape[0],), "float64")
        # Stream cut chunks instead of materializing all cuts and all SVDs.
        for start in range(0, len(cuts), 32):
            chunk = [self.cut_maps(cut) for cut in cuts[start : start + 32]]
            maps = tuple(K.convert_to_tensor(np.stack(x)) for x in zip(*chunk))
            total = total + self.sum_cuts(states, maps)
        return K.numpy(total / len(cuts))

    def rqc(self, states, unitary, bonds):
        K = self.K
        _, means = self.circuit(
            K.convert_to_tensor(states.T),
            K.convert_to_tensor(unitary),
            K.convert_to_tensor(bonds),
        )
        return K.numpy(K.mean(means, axis=0))


@lru_cache(maxsize=1)
def simulation_kernels():
    """Initialize the CPU preprocessing and JAX kernels once per worker."""
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy")
    return JaxKernels(L)


def sample(number):
    """One disorder sample; average circuit realizations before computing SEM."""
    kernels = simulation_kernels()
    K = tc.backend
    rng = np.random.default_rng(np.random.SeedSequence([SEED, number]))
    initial_index = int(rng.integers(len(kernels.idx)))
    fields = {
        name: rng.uniform(-W, W, L)
        for name, W in [
            ("prep", 0.5),
            ("thermal", 0.5),
            ("MBL", 5.0),
            ("AL", 5.0),
            ("Floquet", 5.0),
        ]
    }
    psi0 = K.scatter(K.zeros((len(kernels.idx),)), [[initial_index]], [1.0 + 0j])
    preparation = Evolution(hamiltonian(L, fields["prep"]))
    states = K.stack(list(preparation.at(psi0, TAU)), axis=1)
    fig3_states = K.stack(list(preparation.at(psi0, BAEE_TIMES)), axis=1)
    del preparation
    out = {
        "initial_hcee": kernels.hcee(states),
        "initial_baee": kernels.baee(states),
        "fig3_hcee": kernels.hcee(fig3_states),
        "fig3_baee": kernels.baee(fig3_states),
    }
    for name, jz in [("thermal", 0.5), ("MBL", 0.5), ("AL", 0.0)]:
        ev = Evolution(hamiltonian(L, fields[name], jz))
        out[f"sat_{name}"] = kernels.hcee(next(ev.at(states, [1e12])))
    # H_XY is both the free-fermion Hamiltonian and the Floquet hopping step.
    xy = Evolution(hamiltonian(L, np.zeros(L), jz=0))
    out["sat_free"] = np.mean(
        [kernels.hcee(s) for s in xy.at(states, np.arange(201, 301))], axis=0
    )
    floquet = Floquet(L, fields["Floquet"], xy)
    out["sat_Floquet"] = kernels.hcee(next(floquet.at(states, [300_000_000_000])))
    del floquet, ev, xy
    bonds = rng.integers(0, L - 1, (REALIZATIONS, DEPTH))
    for name, angles in PROTOCOLS.items():
        out[f"rqc_{name}"] = kernels.rqc(states, gate(*angles), bonds)
    return out


def aggregate(records):
    """Keep disorder samples paired when estimating growth and reservoir SEM."""
    summary = dict(
        L=L,
        samples=SAMPLES,
        seed=SEED,
        depth=DEPTH,
        circuit_realizations=REALIZATIONS,
        tau=TAU,
        baee_times=BAEE_TIMES,
    )
    initial = np.array([r["initial_hcee"] for r in records])
    for key in records[0]:
        values = np.array([record[key] for record in records])
        save_statistics(summary, key, values)
        if key.startswith(("sat_", "rqc_")) or key == "initial_baee":
            save_statistics(summary, key + "_growth", values - initial)
    save_statistics(
        summary,
        "fig3_reservoir",
        np.array([r["fig3_baee"] - r["fig3_hcee"] for r in records]),
    )
    return summary


def save_statistics(summary, key, values):
    """Means and standard errors over independent disorder samples."""
    summary[key] = values.mean(axis=0)
    summary[key + "_sem"] = values.std(axis=0, ddof=1) / np.sqrt(len(values))


def haar_half_filling(length):
    """Mean complex-Haar half-chain entropy in the half-filled sector, in bits."""
    dimension = comb(length, length // 2)
    answer = 0.0
    for charge in range(length // 2 + 1):
        block_size = comb(length // 2, charge)
        weight = block_size**2 / dimension
        answer += weight * (
            digamma(dimension + 1)
            - digamma(block_size + 1)
            - (block_size - 1) / (2 * block_size)
        )
    return float(answer / np.log(2))


def plot(summary):
    """Plot the three target panels with one-SEM bars and the Haar reference."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.19, top=0.91, wspace=0.25)

    def line(ax, x, key, name, color, xerror=None):
        ax.errorbar(
            x,
            summary[key],
            yerr=summary[key + "_sem"],
            xerr=xerror,
            label=name,
            color=color,
            linewidth=1.5,
            elinewidth=0.8,
            capsize=2,
            marker=".",
            markersize=3,
        )

    for name, display, color in zip(
        ("thermal", "AL", "free", "MBL", "Floquet"),
        ("thermal", "AL", "free fermion", "Hamiltonian MBL", "Floquet MBL"),
        COLORS,
    ):
        line(
            axes[0],
            summary["initial_hcee"],
            f"sat_{name}_growth",
            display,
            color,
            summary["initial_hcee_sem"],
        )
    circuit_labels = (
        r"RQC thermal $\alpha=\pi/2,\ \beta=\pi$",
        r"RQC(A) $\alpha=0,\ \beta=\pi$",
        r"RQC(B) $\alpha=\pi/2,\ \beta=0$",
        r"RQC(C) $\alpha=\pi,\ \beta=0$",
        r"RQC(D) $\alpha=\pi,\ \beta=\pi/2$",
        "random SWAP circuit",
    )
    for name, display, color in zip(
        ("thermal", "A", "B", "C", "D", "SWAP"), circuit_labels, COLORS
    ):
        key = "initial_baee" if name == "SWAP" else f"rqc_{name}"
        line(
            axes[1],
            summary["initial_hcee"],
            f"{key}_growth",
            display,
            color,
            summary["initial_hcee_sem"],
        )
    for key, name, color in zip(
        ("hcee", "baee", "reservoir"),
        (r"$S$", r"$\bar{S}$", r"$\bar{S}-S$"),
        (COLORS[2], COLORS[4], COLORS[1]),
    ):
        line(axes[2], summary["baee_times"], f"fig3_{key}", name, color)
    length = int(summary["L"])
    haar = haar_half_filling(length)
    for index, (ax, title) in enumerate(
        zip(axes, ("Fig. 1(d)", "Fig. 1(e)", "Fig. 3"))
    ):
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, length / 2)
        ax.set_ylabel(r"$\Delta S$" if index < 2 else "EE", fontsize=15)
        ax.set_xlabel(
            r"$S_{\mathrm{initial}}$" if index < 2 else r"$\tau$", fontsize=15
        )
        ax.axhline(haar, color="black", linestyle="--", linewidth=1.5, zorder=0)
        if index < 2:
            ax.set_xlim(0, length / 2)
            ax.axvline(haar, color="black", linestyle="--", linewidth=1.5, zorder=0)
        else:
            ax.set_xlim(0, 50)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(
            fontsize=9 if index < 2 else 13,
            loc="upper right" if index < 2 else "center right",
        )
    fig.text(
        0.5,
        0.035,
        f"Entanglement growth  |  L={length}, "
        f"{int(summary['samples'])} disorder samples per group  |  "
        "Error bars: one SEM; dashed lines: fixed-charge Haar mean",
        ha="center",
        fontsize=10,
    )
    fig.savefig(OUTPUT / "result.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent disorder workers; each needs about 2 GB RAM",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with np.load(OUTPUT / "summary.npz", allow_pickle=False) as summary:
            plot(summary)
        return
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as pool:
        records = []
        for number, record in enumerate(pool.map(sample, range(SAMPLES)), start=1):
            records.append(record)
            print(f"Completed disorder sample {number}/{SAMPLES}", flush=True)
    summary = aggregate(records)
    np.savez_compressed(OUTPUT / "summary.npz", **summary)
    plot(summary)
    print(f"Saved {OUTPUT / 'result.png'}")


if __name__ == "__main__":
    main()
