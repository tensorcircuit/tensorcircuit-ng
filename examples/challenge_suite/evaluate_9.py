"""
Evaluation for Challenge Suite Problem 9.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np

DEFAULT_CONFIG = {
    "n_qubits": 200,
    "observable_index": 100,
    "n_layers": 8,
    "max_steps": 100,
    "n_restarts": 200,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.02,
    "seed": 2035,
    "success_threshold": 0.9,
}


def parameter_count(config):
    n_qubits = config["n_qubits"]
    count = 0
    for layer in range(config["n_layers"]):
        count += 2 * n_qubits
        count += len(range(layer % 2, n_qubits - 1, 2))
    return count


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    history = np.asarray(results["observable_history"], dtype=float)
    initial = history[:, 0]
    final = history[:, -1]
    grad_norms = np.asarray(results["final_grad_norms"], dtype=float)
    n_restarts = config["n_restarts"]
    steps_run = config["max_steps"]
    mean_initial = float(np.mean(initial))
    mean_final = float(np.mean(final))
    variance_final = float(np.var(final))
    std_final = float(np.std(final))
    best_final = float(np.max(final))
    success_fraction = float(np.mean(final >= config["success_threshold"]))

    criteria = {
        "history shape": history.shape == (config["n_restarts"], config["max_steps"]),
        "initial/final restart count": initial.shape == final.shape == (n_restarts,),
        "initial values finite": np.all(np.isfinite(initial)),
        "final values finite": np.all(np.isfinite(final)),
        "history values finite": np.all(np.isfinite(history)),
        "mean expectation improves": mean_final > mean_initial,
        "best final reaches threshold": best_final >= config["success_threshold"],
        "final grad norm shape": grad_norms.shape == (n_restarts,),
        "final grad norms finite": np.all(np.isfinite(grad_norms)),
        "final variance finite": np.isfinite(variance_final) and variance_final >= 0.0,
    }

    print("Challenge 9 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Qubits: {config['n_qubits']}")
    print(f"Observable: Z{config['observable_index']}")
    print(f"Layers: {config['n_layers']}")
    print(f"Restarts: {n_restarts}")
    print(f"Steps per restart: {steps_run}")
    print(f"Parameter count: {parameter_count(config)}")
    print(f"Observable history shape: {history.shape}")
    print(f"Mean initial <Z100>: {mean_initial:.10f}")
    print(f"Mean final <Z100>: {mean_final:.10f}")
    print(f"Variance final <Z100>: {variance_final:.10e}")
    print(f"Std final <Z100>: {std_final:.10e}")
    print(f"Best final <Z100>: {best_final:.10f}")
    print(f"Success fraction: {success_fraction:.6f}")
    print(f"Final grad norm mean: {float(np.mean(grad_norms)):.8e}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_9")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--n-restarts", type=int, default=DEFAULT_CONFIG["n_restarts"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    config["n_restarts"] = args.n_restarts
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
