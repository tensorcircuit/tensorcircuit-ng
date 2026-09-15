"""
Reproduction of "Real-Time Dynamics in Two Dimensions with Tensor Network States
via Time-Dependent Variational Monte Carlo Method"
Link: https://arxiv.org/abs/2512.06768
Description:
This script studies the Figure 2(b,c) quench with number-projected fermionic
PEPS-tVMC and an independent TensorCircuit-NG FGS reference.
"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tencirpauli as tcp
import tensorcircuit as tc

from main import correlation_at_time, fixed_number_alpha
from peps import FermionPEPS, Hofstadter
from tvmc import MonteCarlo


def peak_memory_mib():
    """Normalize the platform-dependent peak RSS units."""
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def fgs_reference(problem, times):
    """Reuse the validated FGS example for the identical rectangular Hamiltonian."""
    K = tc.backend
    h = problem.one_body()
    initial = problem.one_body(-1.0)

    def nambu(matrix):
        zero = K.zeros(matrix.shape, dtype="complex128")
        return (
            K.concat(
                [
                    K.concat([matrix, zero], axis=1),
                    K.concat([zero, -K.transpose(matrix)], axis=1),
                ],
                axis=0,
            )
            / 2
        )

    hf, initial_hf = nambu(h), nambu(initial)
    alpha = fixed_number_alpha(initial_hf, problem.particles)
    reference = tc.FGSSimulator(
        problem.nsites, alpha=fixed_number_alpha(hf, problem.particles)
    ).get_cmatrix()
    evolve = K.jit(K.vmap(lambda t: correlation_at_time(alpha, hf, t)))
    correlations = evolve(K.convert_to_tensor(times))
    densities = K.real(
        K.einsum("tii->ti", correlations[:, problem.nsites :, problem.nsites :])
    )
    background = K.real(
        K.einsum("ii->i", reference[problem.nsites :, problem.nsites :])
    )
    ground_energy = K.sum(K.eigh(initial)[0][: problem.particles])
    return tuple(np.asarray(K.numpy(x)) for x in (densities, background, ground_energy))


def run_chunks(sampler, theta, chains, key, dt, steps, potential, imaginary):
    """Keep fixed trajectories on device; transfer diagnostics only per chunk."""
    K = tc.backend
    chunk_size = min(20, steps)
    run = K.jit(
        lambda p, s, k: sampler.trajectory(
            p, s, k, dt, chunk_size, potential, imaginary
        )
    )
    histories = []
    started = time.perf_counter()
    for offset in range(0, steps, chunk_size):
        count = min(chunk_size, steps - offset)
        if count != chunk_size:
            run = K.jit(
                lambda p, s, k: sampler.trajectory(
                    p, s, k, dt, count, potential, imaginary
                )
            )
        theta, chains, key, history = run(theta, chains, key)
        history = np.asarray(K.numpy(history))
        if not np.all(np.isfinite(history)) or not np.all(np.isfinite(K.numpy(theta))):
            raise FloatingPointError(
                "Non-finite PEPS trajectory; inspect contraction and SR conditioning."
            )
        histories.append(history)
        print(
            json.dumps(
                {
                    "stage": "ground_state" if imaginary else "real_time",
                    "steps": offset + count,
                    "energy": float(history[-1, 0]),
                    "variance": float(history[-1, 1]),
                    "sr_residual": float(history[-1, 2]),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    return theta, chains, key, np.concatenate(histories), time.perf_counter() - started


def plot_result(times, density, errors, exact, background, peps, output):
    """Show raw signed density changes and independent edge-site time traces."""
    fig = plt.figure(figsize=(12.4, 8.0), layout="constrained")
    layout = fig.add_gridspec(3, 4, height_ratios=(1, 1, 1.15))
    selected = np.linspace(0, len(times) - 1, 4, dtype=int)
    delta, reference_delta = density - background, exact - background
    limit = float(
        np.max(np.abs(np.concatenate([delta[selected], reference_delta[selected]])))
    )
    for row, values in enumerate((reference_delta, delta)):
        for column, index in enumerate(selected):
            ax = fig.add_subplot(layout[row, column])
            mesh = ax.imshow(
                values[index].reshape(peps.rows, peps.columns),
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_title(f"{'FGS' if row == 0 else 'PEPS-tVMC'}  t={times[index]:g}")
            ax.set_xlabel("x")
            ax.set_xticks(range(peps.columns))
            ax.set_yticks(range(peps.rows))
            if column == 0:
                ax.set_ylabel("y")
    fig.colorbar(mesh, ax=fig.axes[:8], label=r"$\delta n_i$")
    sites = ((peps.rows - 1) * peps.columns, peps.nsites - 1, peps.columns - 1, 0)
    lower = min(np.min(exact[:, sites]), np.min(density[:, sites] - errors[:, sites]))
    upper = max(np.max(exact[:, sites]), np.max(density[:, sites] + errors[:, sites]))
    for column, site in enumerate(sites):
        ax = fig.add_subplot(layout[2, column])
        ax.plot(
            times, exact[:, site], color="black", linewidth=1.2, zorder=3, label="FGS"
        )
        ax.plot(
            times,
            density[:, site],
            color="tab:blue",
            linewidth=0.7,
            alpha=0.65,
            zorder=2,
            label="PEPS-tVMC",
        )
        ax.fill_between(
            times,
            density[:, site] - errors[:, site],
            density[:, site] + errors[:, site],
            alpha=0.25,
            color="tab:blue",
        )
        y, x = divmod(site, peps.columns)
        ax.set(title=f"Site ({x}, {y})", xlabel="t", ylabel=r"$\langle n_i\rangle$")
        ax.set_ylim(lower - 0.01, upper + 0.01)
        if column == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Fermionic PEPS-tVMC · {peps.rows} × {peps.columns}, D={peps.bond_dim}"
    )
    fig.savefig(output / "peps_result.png", dpi=160)
    plt.close(fig)


def profile(problem, sampler, theta, chains, key, output):
    """Measure contraction, score, local energy, and one MC proposal separately."""
    K = tc.backend
    results = {}
    functions = {
        "amplitude": (K.jit(problem.peps.batch_amplitude), (theta, chains)),
        "scores": (K.jit(problem.peps.batch_scores), (theta, chains)),
        "local_energy": (K.jit(problem.local_batch), (theta, chains)),
        "mc_proposal": (
            K.jit(lambda p, s, k: sampler.advance(p, s, k, 1)),
            (theta, chains, key),
        ),
    }
    for name, (function, arguments) in functions.items():
        elapsed = []
        for _ in range(2):
            start = time.perf_counter()
            values = function(*arguments)
            leaves = K.tree_map(lambda x: np.asarray(K.numpy(x)), values)
            if not all(np.all(np.isfinite(x)) for x in K.tree_flatten(leaves)[0]):
                raise FloatingPointError(f"Non-finite {name} output.")
            elapsed.append(time.perf_counter() - start)
        results[name] = {"first_seconds": elapsed[0], "steady_seconds": elapsed[1]}
        print(json.dumps({name: results[name]}), flush=True)
    report = {
        "rows": problem.peps.rows,
        "columns": problem.peps.columns,
        "D": problem.peps.bond_dim,
        "boundary_dim": problem.peps.boundary_dim,
        "parameters": problem.peps.nparams,
        "batch_size": sampler.chains,
        "timings": results,
        "max_rss_mib": peak_memory_mib(),
        "tensorcircuit": tc.__version__,
        "tencirpauli": tcp.__version__,
    }
    (output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--boundary-dim", type=int)
    parser.add_argument("--chains", type=int, default=256)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--prepare-steps", type=int, default=400)
    parser.add_argument("--prepare-dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--solver", choices=("sr", "minsr"), default="sr")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    args = parser.parse_args()
    if args.dt <= 0 or args.time <= 0 or args.prepare_steps < 1 or args.prepare_dt <= 0:
        parser.error("Positive dt, time, and preparation steps are required.")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    K = tc.backend
    peps = FermionPEPS(args.rows, args.columns, args.bond_dim, args.boundary_dim)
    problem = Hofstadter(peps, 2 * peps.nsites // 3)
    sampler = MonteCarlo(problem, args.chains, args.draws, args.sweeps, args.solver)
    key, draw = K.random_split(K.get_random_state(args.seed))
    theta = peps.random_parameters(draw)
    key, draw = K.random_split(key)
    chains = sampler.initial_chains(draw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.profile:
        print(
            json.dumps(
                profile(problem, sampler, theta, chains, key, args.output_dir), indent=2
            )
        )
        return
    steps = round(args.time / args.dt)
    if steps < 1 or not np.isclose(steps * args.dt, args.time):
        parser.error("time must be an integer multiple of dt")
    times = np.arange(steps + 1) * args.dt
    exact, background, ground_energy = fgs_reference(problem, times)
    if args.initial_state is not None:
        saved = np.load(args.initial_state)
        theta = K.convert_to_tensor(saved["initial_theta"])
        if (
            theta.shape != (peps.nparams,)
            or int(saved["rows"]) != args.rows
            or int(saved["columns"]) != args.columns
            or int(saved["bond_dim"]) != args.bond_dim
        ):
            parser.error(
                "Initial PEPS does not match the chosen lattice and bond dimension."
            )
    chains, key, _ = K.jit(lambda p, s, k: sampler.advance(p, s, k, 512))(
        theta, chains, key
    )
    if args.initial_state is None:
        preparation_config = {
            "dt": args.prepare_dt,
            "steps": args.prepare_steps,
            "chains": args.chains,
            "draws": args.draws,
            "sweeps": args.sweeps,
            "seed": args.seed,
            "solver": args.solver,
            "regulator": 1e-4,
        }
        preparation = MonteCarlo(
            problem, args.chains, args.draws, args.sweeps, args.solver, regulator=1e-4
        )
        theta, chains, key, prep_history, prep_seconds = run_chunks(
            preparation,
            theta,
            chains,
            key,
            args.prepare_dt,
            args.prepare_steps,
            -1.0,
            True,
        )
    else:
        prep_history = saved["preparation"]
        preparation_config = json.loads(str(saved["preparation_config"]))
        prep_seconds = 0.0
    initial_theta = np.asarray(K.numpy(theta))
    np.savez_compressed(
        args.output_dir / "peps_initial_state.npz",
        initial_theta=initial_theta,
        preparation=prep_history,
        preparation_config=json.dumps(preparation_config),
        rows=args.rows,
        columns=args.columns,
        bond_dim=args.bond_dim,
    )
    theta, chains, key, history, evolve_seconds = run_chunks(
        sampler, theta, chains, key, args.dt, steps, 0.0, False
    )
    _, final, _, _ = K.jit(sampler.estimate)(theta, chains, key)
    history = np.concatenate([history, np.asarray(K.numpy(final))[None]])
    density, error = history[:, 4 : 4 + peps.nsites], history[:, 4 + peps.nsites :]
    report = {
        "rows": args.rows,
        "columns": args.columns,
        "particles": problem.particles,
        "D": args.bond_dim,
        "boundary_dim": args.boundary_dim,
        "dt": args.dt,
        "time": args.time,
        "seed": args.seed,
        "samples_per_stage": args.chains * args.draws,
        "chains": args.chains,
        "draws": args.draws,
        "sweeps": args.sweeps,
        "solver": args.solver,
        "regulator": sampler.regulator,
        "preparation_config": preparation_config,
        "ground_state_exact_energy": float(ground_energy),
        "last_preparation_energy": float(prep_history[-1, 0]),
        "last_preparation_variance": float(prep_history[-1, 1]),
        "max_sampled_density_error_vs_fgs": float(np.max(np.abs(density - exact))),
        "max_sampled_energy_drift": float(
            np.max(np.abs(history[:, 0] - history[0, 0]))
        ),
        "max_sr_residual": float(np.max(history[:, 2])),
        "preparation_seconds_including_jit": prep_seconds,
        "evolution_seconds_including_jit": evolve_seconds,
        "max_rss_mib": peak_memory_mib(),
        "tensorcircuit": tc.__version__,
        "tencirpauli": tcp.__version__,
    }
    np.savez_compressed(
        args.output_dir / "peps_results.npz",
        times=times,
        density=density,
        density_stderr=error,
        exact_density=exact,
        background=background,
        diagnostics=history,
        preparation=prep_history,
        initial_theta=initial_theta,
        final_theta=np.asarray(K.numpy(theta)),
    )
    (args.output_dir / "peps_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    plot_result(times, density, error, exact, background, peps, args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
