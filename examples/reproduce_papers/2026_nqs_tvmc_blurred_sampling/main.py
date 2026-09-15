"""
Reproduction of "Removing Nodal and Support-Mismatch Pathologies in Variational Monte Carlo via Blurred Sampling"
Link: https://arxiv.org/abs/2603.18148
Description:
This script reproduces formal Figure 5(a,b) from the paper using TensorCircuit-NG.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import platform
import resource
from time import perf_counter
from importlib.metadata import version

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorcircuit as tc

if __package__:
    from .physics import SpinProblem, schmitt_velocity
else:
    from physics import SpinProblem, schmitt_velocity

matplotlib.use("Agg")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def initial_parameters(problem):
    """Load the independently regenerated author SR state shared by both samplers."""
    if problem.nsites == 1:
        return tc.backend.convert_to_tensor([1.0, 1.0, 0.0, 0.0])
    with np.load(Path(__file__).resolve().parent / "initial_state.npz") as data:
        return tc.backend.convert_to_tensor(data["parameters"])


def exact_operators(problem):
    """Independent Pauli construction, used only for validation and reference."""
    K = tc.backend
    rows = list(product([1, -1], repeat=problem.nsites))
    allowed = [i for i, x in enumerate(rows) if problem.nsites == 1 or sum(x) == 0]
    indices = K.convert_to_tensor(allowed)
    basis = K.convert_to_tensor([rows[i] for i in allowed])
    if problem.nsites == 1:
        hfull = tc.quantum.PauliStringSum2Dense([[2]])
        ofull = tc.quantum.PauliStringSum2Dense([[3]])
        gauge = K.ones([2], dtype="float64")
    else:
        terms, weights = [], []
        for (i, j), coupling in zip(problem.bonds, K.numpy(problem.couplings)):
            for axis in [1, 2, 3]:
                term = [0] * 4
                term[i], term[j] = axis, axis
                terms.append(term)
                weights.append(float(coupling) / 4)
        hfull = tc.quantum.PauliStringSum2Dense(terms, weights)
        ofull = tc.quantum.PauliStringSum2Dense(terms[:3], [0.25] * 3)
        gauge = basis[:, 0] * basis[:, 3]
    h = hfull[indices[:, None], indices[None, :]]
    observable = ofull[indices[:, None], indices[None, :]]
    return (
        basis,
        gauge[:, None] * h * gauge[None, :],
        (gauge[:, None] * observable * gauge[None, :]),
        hfull,
        indices,
        gauge,
    )


def exact_trajectory(problem, theta, times):
    """Unitary reference starting from the same actual variational state."""
    K = tc.backend
    basis, h, observable, _, _, _ = exact_operators(problem)
    psi = problem.amplitudes(theta, basis)
    psi = psi / K.norm(psi)
    ev, vectors = K.eigh(h)
    coefficients = K.conj(K.transpose(vectors)) @ psi
    states = K.vmap(lambda t: vectors @ (K.exp(-1j * ev * t) * coefficients))(
        K.convert_to_tensor(times)
    )
    values = K.real(K.sum(K.conj(states) * (states @ K.transpose(observable)), axis=1))
    return K.numpy(values), K.numpy(states)


def make_rhs(problem, q):
    """Return the TC backend Monte Carlo TDVP right-hand side."""

    def rhs(theta, chains, key):
        s, f, rows, weights, counts, chains, key = problem.estimate(
            theta, chains, key, q
        )
        # Author drivers use the standard deviation of force rows scaled by
        # 1/N but their sum as the signal: their SNR is N times mean-based SNR.
        cutoff = 2.0 / tc.backend.sum(counts)
        velocity = schmitt_velocity(s, f, rows, cutoff, counts)
        mass = counts / tc.backend.sum(counts)
        ess = tc.backend.sum(mass * weights) ** 2 / tc.backend.sum(mass * weights**2)
        return velocity, chains, key, ess

    return rhs


def make_heun(rhs):
    """Fixed-step explicit trapezoidal rule (Table II, four-spin panel)."""

    def step(theta, chains, key, dt):
        first, chains, key, ess = rhs(theta, chains, key)
        second, chains, key, _ = rhs(theta + dt * first, chains, key)
        return theta + dt * (first + second) / 2, chains, key, ess

    return tc.backend.jit(step)


def make_rk45(rhs):
    """Fixed-step Dormand-Prince RK5, without reusing stochastic stages."""
    tableau = (
        (),
        (1 / 5,),
        (3 / 40, 9 / 40),
        (44 / 45, -56 / 15, 32 / 9),
        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    )
    weights = (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)

    def step(theta, chains, key, dt):
        stages = []
        for coefficients in tableau:
            trial = theta + dt * sum(c * s for c, s in zip(coefficients, stages))
            velocity, chains, key, ess = rhs(trial, chains, key)
            stages.append(velocity)
        candidate = theta + dt * sum(c * s for c, s in zip(weights, stages))
        return candidate, chains, key, ess

    return tc.backend.jit(step)


def make_trajectory(step, step_sizes):
    """Stage all fixed steps, carrying parameters, chains, RNG and an output buffer."""
    K = tc.backend
    step_sizes = K.convert_to_tensor(step_sizes)

    def evolve(theta, chains, key):
        trajectory = K.zeros(
            [step_sizes.shape[0] + 1, theta.shape[0] + 1], dtype="float64"
        )
        first = K.concat([theta, K.ones([1], dtype="float64")])
        trajectory = K.scatter(trajectory, K.convert_to_tensor([[0]]), first[None, :])

        def body(carry, index):
            parameters, current, rng, buffer = carry
            parameters, current, rng, ess = step(
                parameters, current, rng, step_sizes[index]
            )
            row = K.concat([parameters, K.reshape(ess, [1])])
            buffer = K.scatter(buffer, K.reshape(index + 1, [1, 1]), row[None, :])
            return parameters, current, rng, buffer

        return K.scan(
            body, K.arange(step_sizes.shape[0]), (theta, chains, key, trajectory)
        )

    return K.jit(evolve)


def run_dynamics(
    problem,
    theta0,
    q,
    samples,
    seed,
    dt,
    final_time,
):
    """Run one trajectory, preserving every stochastic outcome and failure."""
    K = tc.backend
    key = K.set_random_state(seed, get_only=True)
    chains = problem.initial_chains(samples)
    chains, key = K.jit(lambda p, c, r: problem.metropolis(p, c, r, sweeps=128))(
        theta0, chains, key
    )
    rhs = make_rhs(problem, q)
    step = make_rk45(rhs) if problem.nsites == 1 else make_heun(rhs)
    nsteps = int(np.ceil(final_time / dt))
    times = np.minimum(dt * np.arange(nsteps + 1), final_time)
    step_sizes = np.full(nsteps, dt)
    step_sizes[-1] = final_time - times[-2]
    evolve = make_trajectory(step, step_sizes)
    before = perf_counter()
    K.numpy(evolve(theta0, chains, key)[-1])
    compilation = perf_counter() - before
    before = perf_counter()
    trajectory = K.numpy(evolve(theta0, chains, key)[-1])
    elapsed = perf_counter() - before
    if not np.isfinite(trajectory).all():
        failed = np.flatnonzero(~np.isfinite(trajectory).all(axis=1))[0]
        raise FloatingPointError(
            f"Nonfinite trajectory: n={problem.nsites}, q={q}, seed={seed}, t={times[failed]}"
        )
    parameters, effective_samples = trajectory[:, :-1], trajectory[:, -1]
    basis, _, observable, _, _, _ = exact_operators(problem)
    params = K.convert_to_tensor(parameters)
    states = K.vmap(lambda p: problem.amplitudes(p, basis))(params)
    states = states / K.sqrt(K.sum(K.abs(states) ** 2, axis=1))[:, None]
    observable_values = K.real(
        K.sum(K.conj(states) * (states @ K.transpose(observable)), axis=1)
    )
    exact_values, exact_states = exact_trajectory(problem, theta0, times)
    infidelity = (
        1 - np.abs(np.sum(np.conj(exact_states) * K.numpy(states), axis=1)) ** 2
    )
    result = {
        "times": times,
        "parameters": parameters,
        "observable": K.numpy(observable_values),
        "exact": exact_values,
        "infidelity": infidelity,
        "ess_fraction": effective_samples,
    }
    squared_error = (result["observable"] - exact_values) ** 2
    integrated_error = np.sum(
        np.diff(result["times"]) * (squared_error[1:] + squared_error[:-1]) / 2
    )
    record = {
        "nsites": problem.nsites,
        "q": q,
        "samples": samples,
        "seed": seed,
        "snr_cutoff": 2.0,
        "snr_convention": "author force-row normalization",
        "integrator": "fixed RK45" if problem.nsites == 1 else "Heun",
        "dt": dt,
        "final_time": final_time,
        "steps": len(times) - 1,
        "compile_and_first_trajectory_seconds": compilation,
        "trajectory_seconds": elapsed,
        "max_observable_error": float(
            np.max(np.abs(result["observable"] - exact_values))
        ),
        "rms_observable_error": float(np.sqrt(integrated_error / final_time)),
        "max_infidelity": float(np.max(infidelity)),
    }
    print(json.dumps(record), flush=True)
    return result, record


def plot_results(data, seeds):
    """Compare independently evolved standard and blurred states with exact dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = {0.0: "#2878b5", 0.5: "#d85c27"}
    for panel, nsites in enumerate([1, 4]):
        ax = axes[panel]
        seed = seeds[nsites]
        time_factor = 1 if nsites == 1 else 4
        grid = np.linspace(0, 2 if nsites == 1 else 6, 601)
        for q in [0.0, 0.5]:
            prefix = f"n{nsites}_q{q}_seed{seed}"
            t, y = data[prefix + "_times"], data[prefix + "_observable"]
            label = "Standard tVMC" if q == 0 else "Blurred tVMC (q=0.5)"
            ax.plot(t / time_factor, y, color=colors[q], label=label, lw=1.6)
        exact, _ = exact_trajectory(
            SpinProblem(nsites),
            initial_parameters(SpinProblem(nsites)),
            grid,
        )
        ax.plot(grid / time_factor, exact, "k--", lw=1.4, label="Exact")
        ax.set_title(
            (
                "(a) Single spin"
                if nsites == 1
                else "(b) Four spins · one-hidden-unit RBM"
            ),
            loc="left",
        )
        ax.set_xlabel(
            r"Time $t$" if nsites == 1 else r"Figure time $t=t_{\mathrm{physical}}/4$"
        )
        ax.set_ylabel(
            r"$\langle Z\rangle$"
            if nsites == 1
            else r"$\langle\mathbf{S}_0\cdot\mathbf{S}_1\rangle$"
        )
        ax.set_xlim(0, grid[-1] / time_factor)
        if nsites == 4:
            ax.set_ylim(-0.53, 0.23)
        ax.grid(alpha=0.17)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(OUTPUT_DIR / "result.png", dpi=180)
    plt.close(fig)


def main():
    """Generate both published panels using the author's respective seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Override both panels' sampling seeds")
    args = parser.parse_args()
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    OUTPUT_DIR.mkdir(exist_ok=True)
    data, records = {}, []
    seeds = {1: 100, 4: 1500} if args.seed is None else {1: args.seed, 4: args.seed}
    for nsites, samples, dt, end in [(1, 8192, 0.001, 2.0), (4, 16384, 0.001, 6.0)]:
        problem = SpinProblem(nsites)
        theta0 = initial_parameters(problem)
        for q in [0.0, 0.5]:
            result, record = run_dynamics(
                problem, theta0, q, samples, seeds[nsites], dt, end
            )
            prefix = f"n{nsites}_q{q}_seed{seeds[nsites]}"
            data.update({prefix + "_" + name: value for name, value in result.items()})
            records.append(record)
            np.savez_compressed(OUTPUT_DIR / "results.npz", **data)
    metadata = {
        "seeds": seeds,
        "initial_state": "independently regenerated author SR state",
        "precision": "complex128 / float64",
        "platform": platform.system(),
        "python": platform.python_version(),
        "backend": "jax",
        "tensorcircuit": tc.__version__,
        "device": tc.backend.device(theta0),
        "versions": {
            name: version(name)
            for name in ["jax", "jaxlib", "numpy", "scipy", "matplotlib"]
        },
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024 if platform.system() == "Darwin" else 1024),
        "runs": records,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(data, seeds)


if __name__ == "__main__":
    main()
