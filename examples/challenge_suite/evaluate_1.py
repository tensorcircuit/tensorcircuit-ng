"""
Evaluation for Challenge Suite Problem 1.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and computes reference metrics.
"""

import argparse
import importlib
import time

import numpy as np
from scipy.sparse.linalg import eigsh

import tensorcircuit as tc

DEFAULT_CONFIG = {
    "n_qubits": 20,
    "field": 1.05,
    "dmrg_chi": 8,
    "dmrg_sweeps": 2,
    "n_layers": 4,
    "max_steps": 500,
    "learning_rate": 0.005,
}


def tfim_measurement_data(n_qubits, field):
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
        weights.append(-field)

    return np.array(patterns, dtype=np.int32), np.array(weights, dtype=np.float32)


def build_sparse_tfim(n_qubits, field):
    patterns, weights = tfim_measurement_data(n_qubits, field)
    return tc.quantum.PauliStringSum2COO(patterns, weights, numpy=True)


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    hamiltonian = build_sparse_tfim(config["n_qubits"], config["field"])
    exact_energy = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0]

    dmrg_error = float(results["dmrg_energy"] - exact_energy)
    initial_error = float(results["initial_energy"] - exact_energy)
    final_error = float(results["final_energy"] - exact_energy)
    energy_gain = float(results["initial_energy"] - results["final_energy"])
    criteria = {
        "energy history length": len(results["energy_history"]) == config["max_steps"],
        "gradient history length": len(results["grad_norm_history"])
        == config["max_steps"],
        "refinement improves energy": energy_gain > 0.0,
        "refinement beats dmrg": final_error < dmrg_error,
        "final error <= 1.5e-3": final_error <= 1.5e-3,
        "dmrg error <= 2.5e-3": abs(dmrg_error) <= 2.5e-3,
    }

    print("Challenge 1 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Exact sparse energy: {exact_energy:.8f}")
    print(f"DMRG energy error: {dmrg_error:.8e}")
    print(f"Initial variational error: {initial_error:.8e}")
    print(f"Final variational error: {final_error:.8e}")
    print(f"Energy improvement from circuit refinement: {energy_gain:.8e}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_1")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
