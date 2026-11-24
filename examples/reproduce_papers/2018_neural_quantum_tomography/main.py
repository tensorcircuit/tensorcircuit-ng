"""Reproduction of "Neural-network quantum state tomography"
Link: https://arxiv.org/abs/1703.05334
Description:
This script reproduces the W-state experiments in Figure 1 from the paper using
TensorCircuit-NG, scaled to 8, 10, and 12 qubits. Unrestricted amplitude and phase
RBMs maximize the empirical measurement likelihood. Exact enumeration replaces
Monte Carlo normalization, and L-BFGS replaces SGD/natural-gradient optimization.
Phase probability pretraining and likelihood-based amplitude restarts stabilize
the optimization; both use measured frequencies only.
The original Z/adjacent-XX/adjacent-XY measurement schedule is retained. No target
amplitudes, phases, support constraints, or fidelities enter the training loss.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize
import tensorcircuit as tc

SIZES = (8, 10, 12)
SEEDS = (0, 1, 2)
SHOTS = (20, 100, 500, 2000, 10000)
INITIAL_SCALES = (0.1, 0.5, 1.0)
PHASE_SHOTS = 6400
PROBABILITY_FLOOR = 1e-12


def configurations(n):
    """Enumerate all visible configurations in TensorCircuit's bit order."""
    return ((np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(
        np.float64
    )


def log_rbm(parameters, visible):
    """Log unnormalized RBM marginal after summing binary hidden units (A3)."""
    k = tc.backend
    n = visible.shape[-1]
    bias, hidden = parameters[:n], parameters[n : 2 * n]
    weights = k.reshape(parameters[2 * n :], (n, n))
    fields = k.matmul(visible, k.transpose(weights)) + hidden
    softplus = (fields + k.abs(fields)) / 2 + k.log(1 + k.exp(-k.abs(fields)))
    return k.sum(visible * bias, axis=-1) + k.sum(softplus, axis=-1)


def normalized_log_probability(parameters, visible):
    """Exact normalization over the full Hilbert space, without a support mask."""
    k = tc.backend
    values = log_rbm(parameters, visible)
    maximum = k.max(values)
    log_partition = maximum + k.log(k.sum(k.exp(values - maximum)))
    return values - log_partition


def rbm_state(amplitude, phase, visible):
    """Amplitude/phase RBM wavefunction from Eq. A4; phase may be absent."""
    k = tc.backend
    magnitudes = k.exp(normalized_log_probability(amplitude, visible) / 2)
    state = k.cast(magnitudes, tc.dtypestr)
    if phase is not None:
        angles = log_rbm(phase, visible) / 2
        state = state * k.exp(1j * k.cast(angles - angles[0], tc.dtypestr))
    return state


def pair_rotation(state, n, pair, second_axis):
    """Apply H tensor H or H tensor (H S-dagger) as in Eq. B8."""
    k = tc.backend
    h = k.convert_to_tensor([[1, 1], [1, -1]]) / np.sqrt(2.0)
    second = h
    if second_axis == "Y":
        second = k.convert_to_tensor([[1, -1j], [1, 1j]]) / np.sqrt(2.0)
    rotation = k.kron(k.cast(h, tc.dtypestr), k.cast(second, tc.dtypestr))
    block = k.reshape(state, (2**pair, 4, 2 ** (n - pair - 2)))
    return k.reshape(k.einsum("ab,lbr->lar", rotation, block), (-1,))


def basis_probabilities(state, n):
    """Probabilities in Z, XX_0, XY_0, ..., XX_(N-2), XY_(N-2)."""
    k = tc.backend
    states = [state]
    for pair in range(n - 1):
        states.extend(pair_rotation(state, n, pair, axis) for axis in ("X", "Y"))
    rotated = k.stack(states)
    return k.real(rotated * k.conj(rotated))


def w_circuit(n, phases=None):
    """Prepare W by excitation sharing, followed by optional local RZ gates."""
    circuit = tc.Circuit(n)
    circuit.x(0)
    for site in range(n - 1):
        angle = 2 * np.arccos(1 / np.sqrt(n - site))
        circuit.cry(site, site + 1, theta=angle)
        circuit.cnot(site + 1, site)
    if phases is not None:
        for site, angle in enumerate(phases):
            circuit.rz(site, theta=angle)
    return circuit


def analytic_w(n, phases=None):
    """Independent reference, including the global phase from local RZ gates."""
    angles = np.zeros(n) if phases is None else np.asarray(phases)
    indices = 1 << np.arange(n - 1, -1, -1)
    state = np.zeros(2**n, dtype=np.complex128)
    state[indices] = np.exp(1j * (angles - angles.sum() / 2)) / np.sqrt(n)
    return state


def measurement_streams(state, n, shots, rng, all_bases=True):
    """Generate actual Circuit.sample outcomes using explicit seeded randomness."""
    streams = []
    labels = [None]
    if all_bases:
        labels.extend((pair, axis) for pair in range(n - 1) for axis in ("X", "Y"))
    for label in labels:
        circuit = tc.Circuit(n, inputs=state)
        if label is not None:
            pair, axis = label
            circuit.h(pair)
            if axis == "Y":
                circuit.rz(pair + 1, theta=-np.pi / 2)
            circuit.h(pair + 1)
        outcomes = circuit.sample(
            batch=shots,
            allow_state=True,
            format="sample_int",
            status=tc.backend.convert_to_tensor(rng.random(shots)),
        )
        streams.append(np.asarray(tc.backend.numpy(outcomes), dtype=np.uint16))
    return np.stack(streams)


def measurement_counts(streams, n):
    """Lossless sufficient statistics for the independent-shot likelihood."""
    return np.stack([np.bincount(row, minlength=2**n) for row in streams])


class RBMTomography:
    """Small-system exact-likelihood kernels, compiled once per system size."""

    def __init__(self, n):
        self.n = n
        self.parameter_count = n * (n + 2)
        self.visible = tc.backend.convert_to_tensor(configurations(n))
        k = tc.backend
        self.amplitude_value_grad = k.jit(k.value_and_grad(self.amplitude_loss))
        self.phase_warmup_value_grad = k.jit(k.value_and_grad(self.phase_warmup_loss))
        self.phase_value_grad = k.jit(k.value_and_grad(self.phase_loss))
        self.joint_value_grad = k.jit(k.value_and_grad(self.joint_loss))
        self.state = k.jit(lambda a, p: rbm_state(a, p, self.visible))

    def amplitude_loss(self, parameters, frequency):
        """Z-basis NLL with exact partition function."""
        return -tc.backend.sum(
            frequency * normalized_log_probability(parameters, self.visible)
        )

    def phase_loss(self, phase, amplitude, frequency):
        """Multi-basis NLL with pretrained amplitude parameters held fixed."""
        return self.joint_loss(tc.backend.concat([amplitude, phase]), frequency)

    def phase_warmup_loss(self, phase, amplitude, frequency):
        """Empirical-probability pretraining avoids zero-probability NLL barriers."""
        k = tc.backend
        probabilities = basis_probabilities(
            rbm_state(amplitude, phase, self.visible), self.n
        )
        return k.sum((probabilities - frequency) ** 2) / frequency.shape[0]

    def joint_loss(self, parameters, frequency):
        """Equal-basis empirical NLL; only empirical frequencies enter the loss."""
        k = tc.backend
        a = parameters[: self.parameter_count]
        p = parameters[self.parameter_count :]
        probabilities = basis_probabilities(rbm_state(a, p, self.visible), self.n)
        log_probabilities = k.log(k.clip(probabilities, PROBABILITY_FLOOR, 1.0))
        return -k.sum(frequency * log_probabilities) / frequency.shape[0]


def fit(value_grad, initial, *arguments, maxiter=2000):
    """Optimize likelihood only, never using exact-state evaluation to stop."""
    tensors = tuple(tc.backend.convert_to_tensor(x) for x in arguments)

    def objective(parameters):
        loss, gradient = value_grad(tc.backend.convert_to_tensor(parameters), *tensors)
        loss = float(tc.backend.numpy(loss))
        gradient = np.asarray(tc.backend.numpy(gradient), dtype=np.float64)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Non-finite likelihood or gradient")
        return loss, gradient

    initial_loss = objective(initial)[0]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
    )
    diagnostics = {
        "initial_loss": initial_loss,
        "final_loss": float(result.fun),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "gradient_max": float(np.max(np.abs(result.jac))),
        "converged": bool(result.success),
        "message": str(result.message),
    }
    return result.x, diagnostics


def fit_amplitude(model, initializations, frequency):
    """Restart only on an empirical-likelihood gap, selecting by NLL alone."""
    positive = frequency[frequency > 0]
    empirical_entropy = float(-np.sum(positive * np.log(positive)))
    candidates = []
    diagnostics = []
    for initial in initializations:
        parameters, diagnostic = fit(model.amplitude_value_grad, initial, frequency)
        candidates.append(parameters)
        diagnostics.append(diagnostic)
        if diagnostic["final_loss"] - empirical_entropy <= 1e-5:
            break
    selected = int(np.argmin([d["final_loss"] for d in diagnostics]))
    return candidates[selected], {
        "objective": "Z-basis empirical NLL",
        "empirical_entropy": empirical_entropy,
        "selected_attempt": selected,
        "empirical_kl": diagnostics[selected]["final_loss"] - empirical_entropy,
        "attempts": diagnostics,
    }


def amplitude_initializations(rng, parameter_count):
    """Fixed generic starts, shared by all sample-size prefixes within a seed."""
    return [rng.normal(0, scale, parameter_count) for scale in INITIAL_SCALES]


def reconstruction_metrics(target, learned, n, complex_state=False):
    """Evaluate full-space fidelity, leakage, phase error, and Pauli observables."""
    target = np.asarray(tc.backend.numpy(target))
    learned = np.asarray(tc.backend.numpy(learned))
    probability = np.abs(learned) ** 2
    indices = 1 << np.arange(n - 1, -1, -1)
    fidelity = float(np.abs(np.vdot(target, learned)) ** 2)
    metric = {
        "fidelity": fidelity,
        "overlap": float(np.sqrt(fidelity)),
        "norm_error": float(abs(probability.sum() - 1)),
        "leakage": float(1 - probability[indices].sum()),
        "z_tv": float(np.abs(probability - np.abs(target) ** 2).sum() / 2),
    }
    if complex_state:
        alignment = np.angle(np.vdot(target, learned))
        errors = np.angle(
            learned[indices] * target[indices].conj() * np.exp(-1j * alignment)
        )
        metric["phase_rmse"] = float(np.sqrt(np.mean(errors**2)))
        metric["relative_phases"] = np.angle(
            learned[indices] / learned[indices[0]]
        ).tolist()
    exact_circuit = tc.Circuit(n, inputs=tc.backend.convert_to_tensor(target))
    learned_circuit = tc.Circuit(n, inputs=tc.backend.convert_to_tensor(learned))
    operators = [{"z": [site]} for site in range(n)]
    operators.extend({"z": [site, site + 1]} for site in range(n - 1))
    if complex_state:
        operators.extend({"x": [site, site + 1]} for site in range(n - 1))
        operators.extend({"x": [site], "y": [site + 1]} for site in range(n - 1))
    errors = []
    for operator in operators:
        expected = exact_circuit.expectation_ps(**operator)
        actual = learned_circuit.expectation_ps(**operator)
        errors.append(float(np.abs(tc.backend.numpy(expected - actual))))
    metric["observable_max_error"] = max(errors)
    return metric


def positive_experiments(model, records, raw):
    """Nested sample-size sweep, with independently initialized fits per prefix."""
    n = model.n
    target = w_circuit(n).state()
    np.testing.assert_allclose(tc.backend.numpy(target), analytic_w(n), atol=1e-10)
    for seed in SEEDS:
        data_rng = np.random.default_rng(np.random.SeedSequence([110, n, seed]))
        streams = measurement_streams(target, n, SHOTS[-1], data_rng, all_bases=False)
        raw[f"positive_shots_n{n}_seed{seed}"] = streams
        initializations = amplitude_initializations(
            np.random.default_rng(np.random.SeedSequence([210, n, seed])),
            model.parameter_count,
        )
        for shots in SHOTS:
            counts = measurement_counts(streams[:, :shots], n)[0]
            fitted, diagnostics = fit_amplitude(model, initializations, counts / shots)
            learned = rbm_state(
                tc.backend.convert_to_tensor(fitted), None, model.visible
            )
            metric = reconstruction_metrics(target, learned, n)
            records["positive"].append(
                dict(n=n, seed=seed, shots=shots, **metric, optimization=diagnostics)
            )
            raw[f"positive_parameters_n{n}_seed{seed}_shots{shots}"] = fitted
            print(
                f"Positive N={n} seed={seed} shots={shots}: "
                f"F={metric['fidelity']:.6f}, leakage={metric['leakage']:.3g}",
                flush=True,
            )


def complex_experiments(model, records, raw):
    """Multi-basis amplitude/phase learning and the corresponding Z-only controls."""
    n = model.n
    for seed in SEEDS:
        target_rng = np.random.default_rng(np.random.SeedSequence([310, n, seed]))
        phases = target_rng.uniform(-np.pi, np.pi, n)
        target = w_circuit(n, phases).state()
        np.testing.assert_allclose(
            tc.backend.numpy(target), analytic_w(n, phases), atol=1e-10
        )
        np.testing.assert_allclose(
            np.abs(tc.backend.numpy(target)) ** 2,
            np.abs(analytic_w(n)) ** 2,
            atol=1e-10,
        )
        data_rng = np.random.default_rng(np.random.SeedSequence([410, n, seed]))
        streams = measurement_streams(target, n, PHASE_SHOTS, data_rng)
        counts = measurement_counts(streams, n)
        frequency = counts / PHASE_SHOTS
        initial_rng = np.random.default_rng(np.random.SeedSequence([510, n, seed]))
        initializations = amplitude_initializations(initial_rng, model.parameter_count)
        amplitude, amplitude_diagnostics = fit_amplitude(
            model, initializations, frequency[0]
        )
        z_state = rbm_state(
            tc.backend.convert_to_tensor(amplitude), None, model.visible
        )
        control = reconstruction_metrics(target, z_state, n, complex_state=True)
        initial_phase = initial_rng.normal(0, 0.1, model.parameter_count)
        warm_phase, warmup_diagnostics = fit(
            model.phase_warmup_value_grad, initial_phase, amplitude, frequency
        )
        phase, phase_diagnostics = fit(
            model.phase_value_grad, warm_phase, amplitude, frequency
        )
        joint, joint_diagnostics = fit(
            model.joint_value_grad, np.concatenate([amplitude, phase]), frequency
        )
        learned = model.state(
            joint[: model.parameter_count], joint[model.parameter_count :]
        )
        metric = reconstruction_metrics(target, learned, n, complex_state=True)
        records["complex"].append(
            dict(
                n=n,
                seed=seed,
                shots_per_basis=PHASE_SHOTS,
                basis_count=2 * n - 1,
                **metric,
                z_only_fidelity=control["fidelity"],
                target_relative_phases=np.angle(
                    np.exp(1j * (phases - phases[0]))
                ).tolist(),
                optimization={
                    "amplitude": amplitude_diagnostics,
                    "phase_warmup": warmup_diagnostics,
                    "phase": phase_diagnostics,
                    "joint": joint_diagnostics,
                },
            )
        )
        raw[f"complex_shots_n{n}_seed{seed}"] = streams
        raw[f"complex_target_phases_n{n}_seed{seed}"] = phases
        raw[f"complex_z_parameters_n{n}_seed{seed}"] = amplitude
        raw[f"complex_parameters_n{n}_seed{seed}"] = joint
        print(
            f"Complex N={n} seed={seed}: F={metric['fidelity']:.6f}, "
            f"Z-only F={control['fidelity']:.6f}, phase RMSE={metric['phase_rmse']:.4f}",
            flush=True,
        )


def acceptance(records):
    """Apply the fixed quality criteria across all seeds and sizes."""
    checks = {}
    for n in SIZES:
        high = [
            r for r in records["positive"] if r["n"] == n and r["shots"] == SHOTS[-1]
        ]
        low = [r for r in records["positive"] if r["n"] == n and r["shots"] == SHOTS[0]]
        phase = [r for r in records["complex"] if r["n"] == n]
        checks[f"n{n}_complete"] = (
            len(high) == len(low) == len(phase) == len(SEEDS)
            and {r["seed"] for r in high} == set(SEEDS)
            and {r["seed"] for r in low} == set(SEEDS)
            and {r["seed"] for r in phase} == set(SEEDS)
        )
        checks[f"n{n}_positive_mean_fidelity"] = (
            np.mean([r["fidelity"] for r in high]) >= 0.98
        )
        checks[f"n{n}_positive_min_fidelity"] = min(r["fidelity"] for r in high) >= 0.95
        checks[f"n{n}_sample_size_gain"] = (
            np.mean([r["fidelity"] for r in high])
            - np.mean([r["fidelity"] for r in low])
            >= 0.03
        )
        checks[f"n{n}_leakage"] = max(r["leakage"] for r in high) <= 0.02
        checks[f"n{n}_z_tv"] = max(r["z_tv"] for r in high) <= 0.05
        checks[f"n{n}_phase_mean_fidelity"] = (
            np.mean([r["fidelity"] for r in phase]) >= 0.97
        )
        checks[f"n{n}_phase_min_fidelity"] = min(r["fidelity"] for r in phase) >= 0.95
        checks[f"n{n}_phase_rmse"] = np.mean([r["phase_rmse"] for r in phase]) <= 0.25
        checks[f"n{n}_multibasis_gain"] = (
            np.mean([r["fidelity"] - r["z_only_fidelity"] for r in phase]) >= 0.20
        )
        checks[f"n{n}_observables"] = (
            max(r["observable_max_error"] for r in high + phase) <= 0.05
        )
    all_records = records["positive"] + records["complex"]
    checks["all_states_normalized"] = max(r["norm_error"] for r in all_records) <= 1e-10
    checks["all_experiments_present"] = (
        len(records["positive"]) == 45
        and len(records["complex"]) == 9
        and {(r["n"], r["seed"], r["shots"]) for r in records["positive"]}
        == {(n, seed, shots) for n in SIZES for seed in SEEDS for shots in SHOTS}
    )
    return {name: bool(value) for name, value in checks.items()}


def plot_results(records, raw, output):
    """Plot all seeds' statistics and explicitly label fidelity versus overlap."""
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = ["#2878b5", "#df8f44", "#3c9865"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for n, color in zip(SIZES, colors):
        groups = [
            [r for r in records["positive"] if r["n"] == n and r["shots"] == shots]
            for shots in SHOTS
        ]
        values = np.array([[r["overlap"] for r in group] for group in groups])
        axes[0].errorbar(
            SHOTS,
            values.mean(axis=1),
            yerr=values.std(axis=1, ddof=1),
            marker="o",
            color=color,
            label=f"N = {n}",
            capsize=3,
        )
    axes[0].set(
        xscale="log",
        xlabel="Z-basis training shots",
        ylabel=r"Overlap $O=|\langle\Psi|\psi\rangle|$",
        title="Positive W: sample-size dependence",
        ylim=(0.6, 1.01),
    )
    axes[0].legend()
    n = SIZES[-1]
    indices = 1 << np.arange(n - 1, -1, -1)
    model = RBMTomography(n)
    for shots, color in zip((SHOTS[0], SHOTS[2], SHOTS[-1]), colors):
        parameters = tc.backend.convert_to_tensor(
            raw[f"positive_parameters_n{n}_seed0_shots{shots}"]
        )
        probability = (
            np.abs(tc.backend.numpy(rbm_state(parameters, None, model.visible))) ** 2
        )
        axes[1].plot(
            np.arange(1, n + 1),
            probability[indices],
            marker="o",
            color=color,
            label=f"{shots} shots",
        )
    axes[1].axhline(1 / n, color="black", linestyle="--", label="Exact 1/N")
    axes[1].set(
        xlabel="Excited site",
        ylabel="Unconditioned probability",
        title="N = 12, seed 0: learned populations",
    )
    axes[1].legend(fontsize=8)
    positions = np.arange(len(SIZES))
    for offset, key, label, color in [
        (-0.18, "z_only_fidelity", "Z only", "#999999"),
        (0.18, "fidelity", "Z + XX + XY", "#2878b5"),
    ]:
        groups = [[r[key] for r in records["complex"] if r["n"] == n] for n in SIZES]
        means = [np.mean(group) for group in groups]
        deviations = [np.std(group, ddof=1) for group in groups]
        axes[2].bar(
            positions + offset,
            means,
            width=0.34,
            yerr=deviations,
            capsize=3,
            label=label,
            color=color,
        )
    axes[2].set(
        xticks=positions,
        xticklabels=[str(n) for n in SIZES],
        xlabel="Qubits",
        ylabel=r"Fidelity $F=O^2$",
        title="Random phases require more than Z data",
        ylim=(0, 1.06),
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Torlai et al., Figure 1 mechanisms at N = 8, 10, 12\n"
        "Unrestricted RBMs; exact likelihood; three seeds; error bars = sample SD",
        fontsize=12,
    )
    fig.savefig(output / "result.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, n in zip(axes, SIZES):
        selected = next(r for r in records["complex"] if r["n"] == n and r["seed"] == 0)
        exact = np.array(selected["target_relative_phases"])
        learned = np.array(selected["relative_phases"])
        ax.plot(np.arange(1, n + 1), exact, "o", color="black", label="Exact")
        ax.plot(np.arange(1, n + 1), learned, "x", color="#c45568", label="Learned")
        ax.set(
            xlabel="Excited site",
            ylabel="Phase relative to site 1 (rad)",
            title=f"N = {n}, seed 0; F = {selected['fidelity']:.4f}",
            ylim=(-3.4, 3.4),
        )
        ax.legend()
    fig.suptitle(
        "Phase-augmented W: relative-phase reconstruction\n6400 shots per basis; Z plus adjacent XX and XY",
        fontsize=12,
    )
    fig.savefig(output / "phases.png", dpi=160)
    plt.close(fig)


def smoke():
    """Exercise both optimization paths before the production sweep."""
    n = 4
    model = RBMTomography(n)
    rng = np.random.default_rng(700)
    phases = rng.uniform(-np.pi, np.pi, n)
    target = w_circuit(n, phases).state()
    streams = measurement_streams(target, n, PHASE_SHOTS, rng)
    frequency = measurement_counts(streams, n) / PHASE_SHOTS
    initial = rng.normal(0, 0.1, model.parameter_count)
    amplitude, _ = fit(model.amplitude_value_grad, initial, frequency[0])
    warm_phase, _ = fit(model.phase_warmup_value_grad, initial, amplitude, frequency)
    phase, _ = fit(model.phase_value_grad, warm_phase, amplitude, frequency)
    joint, diagnostics = fit(
        model.joint_value_grad, np.concatenate([amplitude, phase]), frequency
    )
    metric = reconstruction_metrics(
        target,
        model.state(joint[: model.parameter_count], joint[model.parameter_count :]),
        n,
        complex_state=True,
    )
    print(
        json.dumps(dict(metrics=metric, optimization=diagnostics), indent=2), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a four-qubit check without saving artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Results directory relative to this script (default: outputs)",
    )
    arguments = parser.parse_args()
    output = Path(__file__).resolve().parent / arguments.output
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    if arguments.smoke:
        smoke()
        return
    start = time.perf_counter()
    records = {
        "paper": "Torlai et al., Nature Physics 14, 447-450",
        "arxiv_id": "1703.05334",
        "config": {
            "sizes": SIZES,
            "seeds": SEEDS,
            "positive_shots": SHOTS,
            "phase_shots_per_basis": PHASE_SHOTS,
            "alpha": 1,
            "probability_floor": PROBABILITY_FLOOR,
            "optimizer": "L-BFGS-B",
            "maxiter_per_fit": 2000,
            "amplitude_initial_scales": INITIAL_SCALES,
            "amplitude_restart_kl_threshold": 1e-5,
            "phase_pretraining": "Squared error in empirical basis probabilities",
        },
        "software": {
            "tensorcircuit": tc.__version__,
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "devices": [device.platform for device in jax.devices()],
        },
        "positive": [],
        "complex": [],
    }
    raw = {}
    output.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        model = RBMTomography(n)
        positive_experiments(model, records, raw)
        complex_experiments(model, records, raw)
    records["elapsed_seconds"] = time.perf_counter() - start
    checks = acceptance(records)
    records["acceptance"] = {"passed": all(checks.values()), "checks": checks}
    np.savez_compressed(output / "raw_data.npz", **raw)
    (output / "metrics.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError("Scientific acceptance failed: " + ", ".join(failures))
    plot_results(records, raw, output)
    print(
        f"Scientific acceptance passed ({len(checks)} checks); elapsed {records['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
