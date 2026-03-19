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

import jax
import jax.numpy as jnp
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
    """Uniform superposition |+...+> in complex128."""
    dim = 1 << n_variables
    return jnp.ones(dim, dtype=jnp.complex128) / jnp.sqrt(float(dim))


def _generate_batch_clauses(key, n_variables, n_clauses, batch_size, p0):
    """Generate a batch of Weigt hard-satisfiable 3-SAT instances"""
    p1 = (1.0 - 4.0 * p0) / 6.0
    p2 = (1.0 + 2.0 * p0) / 6.0

    k1, k2 = jax.random.split(key)
    p_vals = jax.random.uniform(k1, (batch_size, n_clauses))

    noise = jax.random.uniform(k2, (batch_size, n_clauses, n_variables))
    _, terms = jax.lax.top_k(noise, 3)
    terms = terms + 1

    eye3 = K.eye(3, dtype="int16")

    in_c2 = (p_vals >= p0) & (p_vals < p0 + 3.0 * p1)
    idx2 = K.cast(K.clip(K.floor((p_vals - p0) / p1), 0, 2), "int32")
    signs = K.where(in_c2[:, :, None], 1 - 2 * eye3[idx2], 1)

    in_c3 = p_vals >= p0 + 3.0 * p1
    idx3 = K.cast(K.clip(K.floor((p_vals - p0 - 3.0 * p1) / p2), 0, 2), "int32")
    signs = K.where(in_c3[:, :, None], -1 + 2 * eye3[idx3], signs)

    return K.cast(signs * terms, "int16")


def _single_violation(instance_clauses, bit_table):
    """Violation count for each bitstring of one instance."""
    clause_vars = K.cast(K.abs(instance_clauses), "int32") - 1
    clause_negs = K.cast(instance_clauses < 0, "int8")
    selected = bit_table[:, clause_vars]
    violated = K.all(selected == clause_negs[None, :, :], axis=-1)
    return K.cast(violated.sum(axis=-1), "float64")


# Same setting as author's Julia reference
_TAU_UPPER = 7.5
_TAU_VALID = 10.0
_CHECK_BATCH = 2048
_MAX_BATCH_BYTES = 1 << 30


def _entropy_from_singular_values(s, xp):
    """Shared entropy helper from singular values."""
    s2 = s * s
    p = xp.where(s2 < 1e-10, 0.0, s2)
    p = p / xp.sum(p)
    p = xp.clip(p, 1e-30, None)
    return -xp.sum(p * xp.log(p))


def _entropy_at_tau_jax(tau, energies, init_state, n_left):
    """JAX entropy used by stage1 validity filtering."""
    psi = init_state * K.exp(-tau * energies)
    dim_left = 1 << n_left
    s = jnp.linalg.svd(psi.reshape(dim_left, -1), compute_uv=False)
    return _entropy_from_singular_values(s, jnp)


def _entropy_at_tau(tau, energies_np, init_state_np, n_left):
    """NumPy entropy for host-side SciPy peak search."""
    tau = float(tau)
    psi = init_state_np * np.exp(-tau * energies_np)
    dim_left = 1 << n_left
    s = np.linalg.svd(psi.reshape(dim_left, -1), compute_uv=False)
    return float(_entropy_from_singular_values(s, np))


def _check_valid_single(energies, init_state, n_left):
    """Validity check per Julia reference: S(7.5) >= S(10.0)."""
    return _entropy_at_tau_jax(
        _TAU_UPPER, energies, init_state, n_left
    ) >= _entropy_at_tau_jax(_TAU_VALID, energies, init_state, n_left)


def _stage1_generate_and_check(
    key, bit_table, init_state, n_variables, n_clauses, batch_size, p0
):
    """generate instances, compute energies, check validity."""
    n_left = n_variables // 2
    clauses = _generate_batch_clauses(key, n_variables, n_clauses, batch_size, p0)
    energies = K.vmap(lambda c: _single_violation(c, bit_table))(clauses)
    is_valid = K.vmap(lambda e: _check_valid_single(e, init_state, n_left))(energies)
    return energies, is_valid


def _find_peak_single(energies_np, init_state_np, n_left):
    """Find peak entropy via coarse scan + bounded scalar minimize."""
    tau_grid = np.array([0.0, 1.5, 3.0, 4.5, 6.0, _TAU_UPPER], dtype=np.float64)
    coarse_max = max(
        _entropy_at_tau(t, energies_np, init_state_np, n_left) for t in tau_grid
    )

    def neg_entropy(tau):
        return -_entropy_at_tau(tau, energies_np, init_state_np, n_left)

    result = optimize.minimize_scalar(
        neg_entropy,
        bounds=(0.0, _TAU_UPPER),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 64},
    )
    return max(coarse_max, -float(result.fun))


_jit_stage1 = K.jit(_stage1_generate_and_check, static_argnums=(3, 4, 5))


def sample_bump_heights(
    n_variables: int,
    n_clauses: int,
    n_instances: int,
    rng: np.random.Generator,
    p0: float,
) -> np.ndarray:
    """Sample maximal entanglement bump heights for fixed n and m (clause count)."""
    bit_table_jax = _cached_bit_table_jax(n_variables)
    init_state = _cached_init_state(n_variables)
    init_state_np = np.asarray(jax.device_get(init_state), dtype=np.complex128)
    n_left = n_variables // 2

    bytes_per_inst = (1 << n_variables) * n_clauses * 4
    mem_limit = max(1, _MAX_BATCH_BYTES // bytes_per_inst)
    check_batch = min(mem_limit, _CHECK_BATCH)

    key = jax.random.key(int(rng.integers(0, 2**31)))

    valid_energies: list[np.ndarray] = []
    n_valid = 0
    while n_valid < n_instances:
        key, subkey = jax.random.split(key)
        energies, is_valid = _jit_stage1(
            subkey,
            bit_table_jax,
            init_state,
            n_variables,
            n_clauses,
            check_batch,
            p0,
        )
        e_np, v_np = jax.device_get((energies, is_valid))
        valid_energies.append(e_np[v_np])
        n_valid += int(v_np.sum())

    all_valid = np.concatenate(valid_energies)[:n_instances]
    peaks = np.array(
        [float(_find_peak_single(row, init_state_np, n_left)) for row in all_valid]
    )
    return peaks


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
        n_clauses = int(np.floor(alpha_c * n_value))
        print(f"[Panel A] n={n_value}, m={n_clauses}, instances={n_instances}")
        panel_a[n_value] = sample_bump_heights(n_value, n_clauses, n_instances, rng, p0)
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
        unique_m, inv = np.unique(m_values, return_inverse=True)
        unique_vals = np.empty(len(unique_m), dtype=np.float64)
        for i, m in enumerate(unique_m):
            print(
                f"[Panel B] n={n_value}, m={m}, "
                f"alpha_actual={m / n_value:.4f}, instances={n_instances}"
            )
            heights = sample_bump_heights(n_value, int(m), n_instances, rng, p0)
            unique_vals[i] = heights.mean() / n_value
        panel_b[n_value] = (alpha_actual, unique_vals[inv])
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
        PAPER_COLORS["very_light_purple"],
        PAPER_COLORS["light_purple"],
        "#928FC2",
        "#8078B8",
        PAPER_COLORS["mid_purple"],
        PAPER_COLORS["deep_purple"],
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
