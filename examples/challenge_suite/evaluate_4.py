"""
Evaluation for Challenge Suite Problem 4.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np

DEFAULT_CONFIG = {
    "n_qubits": 12,
    "entangler_angle": 0.31,
    "true_p01": 0.034,
    "true_p10": 0.011,
    "initial_p01": 0.070,
    "initial_p10": 0.040,
    "max_steps": 120,
    "learning_rate": 0.04,
}

PROBE_NAMES = ("zeros", "ones", "neel", "plus")


def asymmetric_bitflip_kraus_numpy(p01, p10):
    return [
        np.array(
            [[np.sqrt(1.0 - p01), 0.0], [0.0, np.sqrt(1.0 - p10)]],
            dtype=np.complex128,
        ),
        np.array([[0.0, np.sqrt(p10)], [0.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [np.sqrt(p01), 0.0]], dtype=np.complex128),
    ]


def trace_preserving_error(p01, p10):
    kraus = asymmetric_bitflip_kraus_numpy(p01, p10)
    identity = np.eye(2, dtype=np.complex128)
    contraction = sum(k.conj().T @ k for k in kraus)
    return float(np.max(np.abs(contraction - identity)))


def print_expectation_table(targets, fitted):
    observable_names = [f"Z{i}" for i in range(targets.shape[1] - 1)] + ["parity"]
    print("Expectation comparison:")
    for probe_index, probe_name in enumerate(PROBE_NAMES):
        for obs_index, obs_name in enumerate(observable_names):
            target = targets[probe_index, obs_index]
            value = fitted[probe_index, obs_index]
            print(
                f"  {probe_name:15s} {obs_name:6s} "
                f"target={target: .8f} fitted={value: .8f} "
                f"abs_error={abs(target - value):.3e}"
            )


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    targets = np.asarray(results["target_expectations"], dtype=float)
    fitted = np.asarray(results["fitted_expectations"], dtype=float)
    expectation_errors = np.abs(fitted - targets)
    final_p01 = float(results["final_p01"])
    final_p10 = float(results["final_p10"])
    absolute_error_p01 = abs(final_p01 - config["true_p01"])
    absolute_error_p10 = abs(final_p10 - config["true_p10"])
    tp_error = trace_preserving_error(final_p01, final_p10)

    criteria = {
        "loss history length": len(results["loss_history"]) == config["max_steps"],
        "loss improves": float(results["final_loss"]) < float(results["initial_loss"]),
        "target shape": targets.shape == (len(PROBE_NAMES), config["n_qubits"] + 1),
        "fitted shape": fitted.shape == (len(PROBE_NAMES), config["n_qubits"] + 1),
        "p01 absolute error <= 1e-4": absolute_error_p01 <= 1e-4,
        "p10 absolute error <= 1e-4": absolute_error_p10 <= 1e-4,
        "trace preserving error <= 1e-8": tp_error <= 1e-8,
    }

    print("Challenge 4 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"True p01, p10: {config['true_p01']:.8f}, {config['true_p10']:.8f}")
    print(
        "Initial p01, p10: "
        f"{float(results['initial_p01']):.8f}, {float(results['initial_p10']):.8f}"
    )
    print(f"Fitted p01, p10: {final_p01:.8f}, {final_p10:.8f}")
    print(f"Absolute p01 error: {absolute_error_p01:.8e}")
    print(f"Absolute p10 error: {absolute_error_p10:.8e}")
    print(f"Initial loss: {float(results['initial_loss']):.8e}")
    print(f"Final loss: {float(results['final_loss']):.8e}")
    print(f"Max expectation abs error: {float(np.max(expectation_errors)):.8e}")
    print(f"Mean expectation abs error: {float(np.mean(expectation_errors)):.8e}")
    print(f"Trace-preserving error: {tp_error:.8e}")
    print(f"Loss history length: {len(results['loss_history'])}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print_expectation_table(targets, fitted)
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_4")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
