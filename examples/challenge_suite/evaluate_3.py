"""
Evaluation for Challenge Suite Problem 3.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, computes reference metrics, and
prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np
from scipy.sparse.linalg import eigsh

import tensorcircuit as tc

DEFAULT_CONFIG = {
    "n_qubits": 12,
    "transverse_field": 0.9,
    "n_steps": 10,
    "log_probability_weight": 0.05,
    "max_steps": 300,
    "learning_rate": 0.01,
    "maximum_energy_density_gap": 1.0,
}


def build_sparse_tfim(n_qubits, transverse_field):
    patterns = []
    weights = []

    for i in range(n_qubits - 1):
        pattern = [0] * n_qubits
        pattern[i] = 3
        pattern[i + 1] = 3
        patterns.append(pattern)
        weights.append(-1.0)

    for i in range(n_qubits):
        pattern = [0] * n_qubits
        pattern[i] = 1
        patterns.append(pattern)
        weights.append(-transverse_field)

    return tc.quantum.PauliStringSum2COO(
        np.array(patterns, dtype=np.int32),
        np.array(weights, dtype=np.float32),
        numpy=True,
    )


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    hamiltonian = build_sparse_tfim(config["n_qubits"], config["transverse_field"])
    exact_energy = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0]
    exact_energy_density = exact_energy / config["n_qubits"]

    energy_history = np.asarray(results["energy_density_history"], dtype=float)
    success_history = np.asarray(results["success_probability_history"], dtype=float)
    mean_log_history = np.asarray(results["mean_log_probability_history"], dtype=float)
    loss_history = np.asarray(results["loss_history"], dtype=float)
    success_probability = float(success_history[-1])
    mean_log_probability = float(mean_log_history[-1])
    n_events = config["n_steps"] * len(range(0, config["n_qubits"], 2))
    recomputed_success = float(np.exp(n_events * mean_log_probability))

    criteria = {
        "energy history length": len(energy_history) == config["max_steps"],
        "success history length": len(success_history) == config["max_steps"],
        "mean log probability history length": len(mean_log_history)
        == config["max_steps"],
        "loss history length": len(loss_history) == config["max_steps"],
        "loss improves": float(loss_history[-1]) < float(loss_history[0]),
        "energy density improves": float(energy_history[-1]) < float(energy_history[0]),
        "loose exact energy-density upper bound": float(energy_history[-1])
        <= exact_energy_density + config["maximum_energy_density_gap"],
        "success probability is valid": 0.0 < success_probability <= 1.0,
        "success matches mean log probability": np.isclose(
            success_probability, recomputed_success, rtol=1e-4, atol=1e-30
        ),
    }

    print("Challenge 3 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Exact sparse ground energy density: {exact_energy_density:.8f}")
    print(f"Initial energy density: {float(energy_history[0]):.8f}")
    print(f"Final energy density: {float(energy_history[-1]):.8f}")
    print(f"Final success probability: {success_probability:.8e}")
    print(f"Final mean log event probability: {mean_log_probability:.8e}")
    print(f"Initial loss: {float(loss_history[0]):.8f}")
    print(f"Final loss: {float(loss_history[-1]):.8f}")
    print(f"Energy history length: {len(energy_history)}")
    print(f"Success history length: {len(success_history)}")
    print("Mean log probability history length: " f"{len(mean_log_history)}")
    print(f"Loss history length: {len(loss_history)}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_3")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
