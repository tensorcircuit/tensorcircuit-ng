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
import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorcircuit as tc

PAPER_COLORS = {
    "deep_purple": "#502080",
    "mid_purple": "#7060B0",
    "light_purple": "#B0B0D0",
    "very_light_purple": "#D0D0E0",
}

K = tc.set_backend("jax")
tc.set_dtype("complex128")


@lru_cache(maxsize=None)
def all_bitstrings(n_variables: int) -> np.ndarray:
    """Return all binary assignments as an array of shape (2**n, n)."""
    states = np.arange(1 << n_variables, dtype=np.uint32)
    bits = (states[:, None] >> np.arange(n_variables, dtype=np.uint32)[None, :]) & 1
    return bits.astype(np.int8)


@lru_cache(maxsize=None)
def _cached_bit_table_jax(n_variables: int):
    """Cache the JAX device-side bit table per n."""
    return K.convert_to_tensor(all_bitstrings(n_variables))


@lru_cache(maxsize=None)
def _cached_init_state(n_variables: int):
    """Cache the uniform superposition |+...+> per n (immutable for same n)."""
    c = tc.Circuit(n_variables)
    for i in range(n_variables):
        c.h(i)
    return c.state()


def _generate_batch_clauses(key, n_variables, n_clauses, batch_size, p0):
    """Generate a batch of Weigt hard-satisfiable 3-SAT instances (JAX-vectorized).

    Replaces the per-clause Python loop with fully vectorized JAX operations.
    Returns int16 array of shape (batch_size, n_clauses, 3).
    """
    p1 = (1.0 - 4.0 * p0) / 6.0
    p2 = (1.0 + 2.0 * p0) / 6.0

    k1, k2 = jax.random.split(key)
    p_vals = jax.random.uniform(k1, (batch_size, n_clauses))

    # variable selection: argsort of uniform noise → random permutation, take first 3
    noise = jax.random.uniform(k2, (batch_size, n_clauses, n_variables))
    terms = K.argsort(noise, axis=-1)[:, :, :3] + 1

    eye3 = K.eye(3, dtype="int16")

    in_c2 = (p_vals >= p0) & (p_vals < p0 + 3.0 * p1)
    idx2 = K.cast(K.clip(K.floor((p_vals - p0) / p1), 0, 2), "int32")
    signs = K.where(in_c2[:, :, None], 1 - 2 * eye3[idx2], 1)

    in_c3 = p_vals >= p0 + 3.0 * p1
    idx3 = K.cast(K.clip(K.floor((p_vals - p0 - 3.0 * p1) / p2), 0, 2), "int32")
    signs = K.where(in_c3[:, :, None], -1 + 2 * eye3[idx3], signs)

    return K.cast(signs * terms, "int16")


_jit_generate_clauses = K.jit(_generate_batch_clauses, static_argnums=(1, 2, 3, 4))


def _batch_violation_jax(all_clauses, bit_table):
    """Batch violation vectors for B instances at once."""
    clause_vars = K.cast(K.abs(all_clauses), "int32") - 1
    clause_negs = K.cast(all_clauses < 0, "int8")
    selected = bit_table[:, clause_vars].transpose(1, 0, 2, 3)
    violated = K.all(selected == clause_negs[:, None, :, :], axis=-1)
    return K.cast(violated.sum(axis=-1), "float64")


_jit_batch_violation = K.jit(_batch_violation_jax)


_N_COARSE = 16
_N_FINE = 8
_MAX_BATCH_BYTES = 1 << 30


def _svd_entropy(psi_batch, n_left):
    """Batched half-chain von Neumann entropy via SVD."""
    dim_left = 1 << n_left
    mat = psi_batch.reshape(-1, dim_left, psi_batch.shape[-1] // dim_left)
    s = jnp.linalg.svd(mat, compute_uv=False)
    p = K.clip(s**2, 1e-30, 1e30)
    return -K.sum(p * K.log(p), axis=-1)


def _batch_grid_search_core(all_energies, init_state, n_left):
    """Jittable: batch coarse+fine grid search over tau in [0, 7.5]."""
    b_size = all_energies.shape[0]

    tau_coarse = jnp.linspace(0.0, 7.5, _N_COARSE)
    tau_scan = K.concat([tau_coarse, K.convert_to_tensor([10.0])])
    m_scan = tau_scan.shape[0]

    boltz = K.exp(-tau_scan[None, :, None] * all_energies[:, None, :])
    psi = init_state[None, None, :] * boltz
    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)
    scan_ent = _svd_entropy(psi.reshape(b_size * m_scan, -1), n_left).reshape(
        b_size, m_scan
    )

    coarse_ent = scan_ent[:, :-1]
    is_valid = coarse_ent[:, -1] >= scan_ent[:, -1]
    best_idx = K.argmax(coarse_ent, axis=-1)
    best_tau = tau_coarse[best_idx]
    delta = (7.5 / (_N_COARSE - 1)) * 2.0

    t_lo = jnp.maximum(best_tau - delta, 0.0)
    t_hi = jnp.minimum(best_tau + delta, 7.5)
    t = jnp.linspace(0.0, 1.0, _N_FINE)
    tau_fine = t_lo[:, None] + t[None, :] * (t_hi - t_lo)[:, None]

    boltz_f = K.exp(-tau_fine[:, :, None] * all_energies[:, None, :])
    psi_f = init_state[None, None, :] * boltz_f
    psi_f = psi_f / jnp.linalg.norm(psi_f, axis=-1, keepdims=True)
    fine_ent = _svd_entropy(psi_f.reshape(b_size * _N_FINE, -1), n_left).reshape(
        b_size, _N_FINE
    )

    fine_best_idx = K.argmax(fine_ent, axis=-1)
    max_ent = fine_ent[K.arange(0, b_size, 1), fine_best_idx]
    return max_ent, is_valid


_jit_batch_grid_search = K.jit(_batch_grid_search_core, static_argnums=(2,))


def sample_bump_heights(
    n_variables: int,
    alpha: float,
    n_instances: int,
    rng: np.random.Generator,
    p0: float,
) -> np.ndarray:
    """Sample maximal entanglement bump heights for fixed n and alpha."""
    n_clauses = int(alpha * n_variables)
    bit_table_jax = _cached_bit_table_jax(n_variables)
    init_state = _cached_init_state(n_variables)
    n_left = n_variables // 2

    bytes_per_inst = (_N_COARSE + _N_FINE + 1) * (1 << n_variables) * 16
    mem_limit = max(1, _MAX_BATCH_BYTES // bytes_per_inst)
    batch = min(mem_limit, max(n_instances, 256))

    key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))
    collected = []
    filled = 0

    while filled < n_instances:
        key, subkey = jax.random.split(key)
        all_clauses = _jit_generate_clauses(subkey, n_variables, n_clauses, batch, p0)
        all_energies = _jit_batch_violation(all_clauses, bit_table_jax)
        max_ent, is_valid = _jit_batch_grid_search(all_energies, init_state, n_left)

        valid_heights = np.asarray(max_ent)[np.asarray(is_valid)]
        collected.append(valid_heights)
        filled += len(valid_heights)

    return np.concatenate(collected)[:n_instances]


def collect_panel_a(
    n_values: Iterable[int],
    alpha_c: float,
    n_instances: int,
    rng: np.random.Generator,
    p0: float,
) -> dict[int, np.ndarray]:
    """Collect data for Figure 2(a)-style scaling at alpha=alpha_c."""
    panel_a: dict[int, np.ndarray] = {}
    for n_value in n_values:
        print(f"[Panel A] n={n_value}, alpha={alpha_c}, instances={n_instances}")
        panel_a[n_value] = sample_bump_heights(n_value, alpha_c, n_instances, rng, p0)
    return panel_a


def collect_panel_b(
    n_values: Iterable[int],
    alpha_values: np.ndarray,
    n_instances: int,
    rng: np.random.Generator,
    p0: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Collect data for Figure 2(b)-style universal curve of <S_hat>/n.

    Following the Julia reference: m = floor(alpha*n), actual_alpha = m/n.
    Returns {n: (alpha_actual, S_over_n)} with per-n discretized x-axis.
    """
    panel_b: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n_value in n_values:
        m_values = np.floor(alpha_values * n_value).astype(int)
        alpha_actual = m_values / n_value
        normalized_curve = np.empty(len(alpha_values), dtype=np.float64)
        for idx, alpha in enumerate(alpha_values):
            print(
                f"[Panel B] n={n_value}, m={m_values[idx]}, "
                f"alpha_actual={alpha_actual[idx]:.4f}, instances={n_instances}"
            )
            heights = sample_bump_heights(n_value, alpha, n_instances, rng, p0)
            normalized_curve[idx] = heights.mean() / n_value
        panel_b[n_value] = (alpha_actual, normalized_curve)
    return panel_b


def plot_results(
    panel_a: dict[int, np.ndarray],
    panel_b: dict[int, tuple[np.ndarray, np.ndarray]],
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
        np.minimum(s_max, y_top),
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
        alpha_actual, s_over_n = panel_b[n_value]
        ax_b.plot(
            alpha_actual,
            s_over_n,
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
    all_alphas = np.concatenate([panel_b[n][0] for n in panel_b])
    ax_b.set_xlim(all_alphas.min(), all_alphas.max())
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

    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    rng = np.random.default_rng(args.seed)

    panel_a = collect_panel_a(
        n_values=args.n_values_a,
        alpha_c=args.alpha_c,
        n_instances=args.instances_a,
        rng=rng,
        p0=args.weigt_p0,
    )
    panel_b = collect_panel_b(
        n_values=args.n_values_b,
        alpha_values=alpha_values,
        n_instances=args.instances_b,
        rng=rng,
        p0=args.weigt_p0,
    )

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / "result.png"

    plot_results(panel_a, panel_b, args.alpha_c, output_image)


if __name__ == "__main__":
    main()
