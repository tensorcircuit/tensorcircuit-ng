"""
Evaluation for Challenge Suite Problem 5.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, computes a sparse exact reference,
and prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np
from scipy.sparse.linalg import eigsh

import tensorcircuit as tc

DEFAULT_CONFIG = {
    "n_qubits": 18,
    "transverse_field": 1.10,
    "n_layers": 10,
    "initial_filter_strength": 0.01,
    "max_steps": 600,
    "learning_rate": 0.02,
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
        np.array(weights, dtype=np.float64),
        numpy=True,
    )


def format_array(values):
    return np.array2string(np.asarray(values, dtype=float), precision=6, separator=", ")


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    hamiltonian = build_sparse_tfim(config["n_qubits"], config["transverse_field"])
    exact_energy = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0]
    exact_energy_density = exact_energy / config["n_qubits"]

    final_a = np.asarray(results["final_a"], dtype=float)
    final_b = np.asarray(results["final_b"], dtype=float)
    final_energy_density = float(results["final_energy_density"])
    strength_shape = (config["n_layers"] // 2, 2)

    criteria = {
        "energy history length": len(results["energy_density_history"])
        == config["max_steps"],
        "final a shape": final_a.shape == strength_shape,
        "final b shape": final_b.shape == strength_shape,
        "energy density improves": final_energy_density
        < float(results["initial_energy_density"]),
        "energy respects exact lower bound": final_energy_density
        >= exact_energy_density - 1e-8,
    }

    print("Challenge 5 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Exact sparse ground energy density: {exact_energy_density:.10f}")
    print(f"Initial energy density: {float(results['initial_energy_density']):.10f}")
    print(f"Final energy density: {final_energy_density:.10f}")
    print(f"Learned a strengths by block [even, odd]: {format_array(final_a)}")
    print(f"Learned b strengths by block [even, odd]: {format_array(final_b)}")
    print(f"Energy history length: {len(results['energy_density_history'])}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_5")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
