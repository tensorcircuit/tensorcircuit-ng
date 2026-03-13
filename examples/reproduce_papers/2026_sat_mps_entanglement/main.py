"""Reproduction of "Entanglement Barriers from Computational Complexity:
Matrix-Product-State Approach to Satisfiability"
Link: https://arxiv.org/abs/2602.20299
Description:
This script reproduces Figure 2 from the paper using TensorCircuit-NG.
It computes the maximal half-chain entanglement bump under imaginary-time evolution
for Weigt hard-satisfiable 3-SAT instances and plots Figure 2(a,b)-style trends.
The simulation is scaled down (n=4-15, 200 instances, Panel B n=8-10) for laptop-friendly
runtimes while preserving the core physics. Use command-line arguments to run with
the full paper settings (n=4-20, 1000 instances, Panel B n=8-13).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import tensorcircuit as tc

PAPER_COLORS = {
    "deep_purple": "#502080",
    "mid_purple": "#7060B0",
    "light_purple": "#B0B0D0",
    "very_light_purple": "#D0D0E0",
}


def set_backend() -> tuple:
    """Set TensorCircuit backend with graceful fallback.

    Returns (backend_object, backend_name).
    """
    try:
        backend = tc.set_backend("jax")
        backend_name = "jax"
    except ImportError:
        backend = tc.set_backend("numpy")
        backend_name = "numpy"
    tc.set_dtype("complex128")
    return backend, backend_name


@lru_cache(maxsize=None)
def all_bitstrings(n_variables: int) -> np.ndarray:
    """Return all binary assignments as an array of shape (2**n, n)."""
    states = np.arange(1 << n_variables, dtype=np.uint32)
    bits = (states[:, None] >> np.arange(n_variables, dtype=np.uint32)[None, :]) & 1
    return bits.astype(np.int8)


def generate_weigt_satisfiable_instance_signed(
    n_variables: int, n_clauses: int, rng: np.random.Generator, p0: float
) -> np.ndarray:
    """Generate a hard satisfiable random 3-SAT instance via Weigt protocol.

    Clause literals are signed integers in +/-{1,...,n}. Signs are sampled by
    the p0/p1/p2 rule; variables are a random permutation and the first 3 are
    used. Default p0=0.08 corresponds to the hard satisfiable ensemble.
    """
    p1 = (1.0 - 4.0 * p0) / 6.0
    p2 = (1.0 + 2.0 * p0) / 6.0
    clauses = np.empty((n_clauses, 3), dtype=np.int16)

    for clause_id in range(n_clauses):
        p = rng.random()
        terms = rng.permutation(np.arange(1, n_variables + 1, dtype=np.int16))
        signs = np.array([1, 1, 1], dtype=np.int8)

        if p < p0:
            pass
        elif p < p0 + 3.0 * p1:
            idx = int(np.floor((p - p0) / p1))
            signs[idx] = -1
        else:
            signs[:] = -1
            idx = int(np.floor((p - p0 - 3.0 * p1) / p2))
            signs[idx] = 1

        clauses[clause_id] = (signs * terms[:3]).astype(np.int16)
    return clauses


def violation_vector(
    n_variables: int, signed_clauses: np.ndarray, bit_table: np.ndarray
) -> np.ndarray:
    """Count clause violations for every computational-basis state."""
    violations = np.zeros(1 << n_variables, dtype=np.int16)
    clause_vars = np.abs(signed_clauses).astype(np.int16) - 1
    clause_negs = (signed_clauses < 0).astype(np.int8)
    selected_bits = bit_table[:, clause_vars]
    violated = np.all(selected_bits == clause_negs[None, :, :], axis=-1)
    violations += violated.sum(axis=-1).astype(np.int16)
    return violations


def build_init_state(n_variables: int):
    """Prepare the uniform superposition |+...+> via tc.Circuit."""
    c = tc.Circuit(n_variables)
    for i in range(n_variables):
        c.h(i)
    return c.state()


def half_chain_entropy(state_vector, n_variables: int) -> float:
    """Compute half-chain von Neumann entropy S = -Tr(rho log rho)."""
    psi_np = np.asarray(state_vector)
    norm_val = np.linalg.norm(psi_np)
    if norm_val < 1e-15:
        return 0.0
    n_left = n_variables // 2
    n_right = n_variables - n_left
    psi_matrix = (psi_np / norm_val).reshape(1 << n_left, 1 << n_right)
    singular_values = np.linalg.svd(psi_matrix, compute_uv=False, full_matrices=False)
    probabilities = np.abs(singular_values) ** 2
    probabilities = probabilities[probabilities > 1e-14]
    return float(-(probabilities * np.log(probabilities)).sum())


def maximal_bump_height(
    energies: np.ndarray, n_variables: int, init_state, backend
) -> tuple[float, float]:
    """Find the maximal entanglement bump height over imaginary time.

    Constructs |psi(tau)> = e^{-tau H}|+...+> using tc.Circuit for the initial
    state and tc.backend for the Boltzmann weights, then optimises tau in
    [0, 7.5].  Rejects samples with entropy still increasing at tau=7.5
    (checked via comparison with tau=10.0).
    """
    e_tensor = backend.convert_to_tensor(energies.astype(np.float64))

    def neg_entropy(x: np.ndarray) -> float:
        tau = float(np.ravel(x)[0])
        boltzmann = backend.exp(-tau * e_tensor)
        psi = init_state * boltzmann
        return -half_chain_entropy(psi, n_variables)

    if neg_entropy(np.array([7.5])) > neg_entropy(np.array([10.0])):
        return -1.0, -1.0

    result = optimize.minimize_scalar(neg_entropy, bounds=(0.0, 7.5), method="bounded")
    return float(-result.fun), float(result.x)


def sample_bump_heights(
    n_variables: int,
    alpha: float,
    n_instances: int,
    rng: np.random.Generator,
    p0: float,
    backend,
) -> np.ndarray:
    """Sample maximal entanglement bump heights for fixed n and alpha."""
    n_clauses = int(round(alpha * n_variables))
    bit_table = all_bitstrings(n_variables)
    init_state = build_init_state(n_variables)
    heights = np.empty(n_instances, dtype=np.float64)

    for sample_id in range(n_instances):
        max_entropy = -1.0
        while max_entropy < 0.0:
            clauses = generate_weigt_satisfiable_instance_signed(
                n_variables=n_variables, n_clauses=n_clauses, rng=rng, p0=p0
            )
            energies = violation_vector(n_variables, clauses, bit_table)
            max_entropy, _ = maximal_bump_height(
                energies, n_variables, init_state, backend
            )
        heights[sample_id] = max_entropy
    return heights


def collect_panel_a(
    n_values: Iterable[int],
    alpha_c: float,
    n_instances: int,
    rng: np.random.Generator,
    p0: float,
    backend,
) -> dict[int, np.ndarray]:
    """Collect data for Figure 2(a)-style scaling at alpha=alpha_c."""
    panel_a: dict[int, np.ndarray] = {}
    for n_value in n_values:
        print(f"[Panel A] n={n_value}, alpha={alpha_c}, instances={n_instances}")
        panel_a[n_value] = sample_bump_heights(
            n_value, alpha_c, n_instances, rng, p0, backend
        )
    return panel_a


def collect_panel_b(
    n_values: Iterable[int],
    alpha_values: np.ndarray,
    n_instances: int,
    rng: np.random.Generator,
    p0: float,
    backend,
) -> dict[int, np.ndarray]:
    """Collect data for Figure 2(b)-style universal curve of <S_hat>/n."""
    panel_b: dict[int, np.ndarray] = {}
    for n_value in n_values:
        normalized_curve = np.empty(alpha_values.shape[0], dtype=np.float64)
        for idx, alpha in enumerate(alpha_values):
            print(f"[Panel B] n={n_value}, alpha={alpha:.2f}, instances={n_instances}")
            heights = sample_bump_heights(n_value, alpha, n_instances, rng, p0, backend)
            normalized_curve[idx] = heights.mean() / n_value
        panel_b[n_value] = normalized_curve
    return panel_b


def plot_results(
    panel_a: dict[int, np.ndarray],
    panel_b: dict[int, np.ndarray],
    alpha_values: np.ndarray,
    alpha_c: float,
    output_file: Path,
) -> None:
    """Generate and save a Figure-2-style two-panel plot."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    n_keys = sorted(panel_a.keys())
    n_values_a = np.array(n_keys, dtype=float)
    means_a = np.array([panel_a[n].mean() for n in n_keys], dtype=float)
    stds_a = np.array([panel_a[n].std(ddof=1) for n in n_keys], dtype=float)

    ax_a.errorbar(
        n_values_a,
        means_a,
        yerr=stds_a,
        fmt="o-",
        capsize=3.0,
        lw=1.8,
        color=PAPER_COLORS["deep_purple"],
        markerfacecolor=PAPER_COLORS["deep_purple"],
        markeredgecolor=PAPER_COLORS["deep_purple"],
        label=r"$\mu=\langle\hat{S}\rangle$",
    )
    s_max = 0.5 * np.log(2.0) * n_values_a
    s_page = s_max - 0.5
    y_top = 3.0
    ax_a.fill_between(
        n_values_a,
        s_max,
        y_top,
        facecolor="none",
        hatch="////",
        edgecolor="#9A9A9A",
        linewidth=0.0,
        label=r"$S_{\max}=n\ln2/2$",
    )
    ax_a.plot(
        n_values_a,
        s_page,
        linestyle="-.",
        color="black",
        linewidth=1.5,
        label=r"Page law: $S_{\mathrm{typ}}\approx S_{\max}-1/2$",
    )
    ax_a.set_title(rf"(a) $\alpha=\alpha_c={alpha_c}$")
    ax_a.set_xlabel("number of variables n")
    ax_a.set_ylabel(r"$\hat{S}$")
    ax_a.set_ylim(0.0, y_top)
    ax_a.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax_a.legend(frameon=False)

    inset = ax_a.inset_axes([0.70, 0.05, 0.24, 0.26])
    sample = panel_a[n_keys[-1]]
    mean_sample = float(sample.mean())
    std_sample = float(sample.std(ddof=1))
    hist_min = max(0.0, mean_sample - 3.0 * std_sample)
    hist_max = mean_sample + 3.0 * std_sample
    x_grid = np.linspace(hist_min, hist_max, 200)
    gaussian = np.exp(-0.5 * ((x_grid - mean_sample) / std_sample) ** 2) / (
        np.sqrt(2.0 * np.pi) * std_sample
    )
    inset.hist(
        sample,
        bins=14,
        density=True,
        alpha=0.8,
        color=PAPER_COLORS["very_light_purple"],
        label="data",
    )
    inset.plot(
        x_grid,
        gaussian,
        color=PAPER_COLORS["deep_purple"],
        lw=1.5,
        label=r"$f_{\mu,\sigma}$",
    )
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
    inset.tick_params(labelsize=6, length=2)
    inset.set_xticks([mean_sample - std_sample, mean_sample, mean_sample + std_sample])
    inset.set_xticklabels([r"$\mu{-}\sigma$", r"$\mu$", r"$\mu{+}\sigma$"], fontsize=6)
    inset.set_yticks([])
    inset.set_ylabel("pdf", fontsize=7, labelpad=1)
    inset.legend(fontsize=5, frameon=False, loc="upper right")

    curve_palette = [
        PAPER_COLORS["deep_purple"],
        PAPER_COLORS["mid_purple"],
        "#8078B8",
        "#928FC2",
        PAPER_COLORS["light_purple"],
        PAPER_COLORS["very_light_purple"],
    ]
    for curve_id, n_value in enumerate(sorted(panel_b.keys())):
        ax_b.plot(
            alpha_values,
            panel_b[n_value],
            marker="o",
            markersize=3.2,
            linewidth=1.4,
            color=curve_palette[curve_id % len(curve_palette)],
            label=f"n={n_value}",
        )
    ax_b.axvline(alpha_c, color="black", linestyle="--", linewidth=1.2)
    ax_b.set_title(r"(b) $\langle\hat{S}\rangle/n$ vs clause-variable ratio")
    ax_b.set_xlabel(r"clause-variable ratio $\alpha$")
    ax_b.set_ylabel(r"$\langle\hat{S}\rangle/n$")
    ax_b.set_xlim(alpha_values[0], alpha_values[-1])
    ax_b.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 2 of MPS2SAT with the authors' Zenodo settings."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha-c", type=float, default=4.27)
    parser.add_argument("--instances-a", type=int, default=200)
    parser.add_argument("--instances-b", type=int, default=200)
    parser.add_argument(
        "--n-values-a",
        type=int,
        nargs="+",
        default=list(range(4, 16)),
    )
    parser.add_argument("--n-values-b", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument("--alpha-min", type=float, default=2.0)
    parser.add_argument("--alpha-max", type=float, default=7.0)
    parser.add_argument("--alpha-points", type=int, default=30)
    parser.add_argument("--weigt-p0", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    """Run data generation and plot Figure 2 reproduction."""
    args = parse_args()
    backend, backend_name = set_backend()
    print(f"TensorCircuit backend: {backend_name}")

    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    rng = np.random.default_rng(args.seed)

    panel_a = collect_panel_a(
        n_values=args.n_values_a,
        alpha_c=args.alpha_c,
        n_instances=args.instances_a,
        rng=rng,
        p0=args.weigt_p0,
        backend=backend,
    )
    panel_b = collect_panel_b(
        n_values=args.n_values_b,
        alpha_values=alpha_values,
        n_instances=args.instances_b,
        rng=rng,
        p0=args.weigt_p0,
        backend=backend,
    )

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / "result.png"

    plot_results(panel_a, panel_b, alpha_values, args.alpha_c, output_image)
    print(f"Saved figure: {output_image}")


if __name__ == "__main__":
    main()
