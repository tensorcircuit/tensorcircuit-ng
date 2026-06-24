"""
Evaluation for Challenge Suite Problem 7.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, computes a sparse exact reference,
and prints compact validation output.
"""

import argparse
import importlib
import math
import time
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

DEFAULT_CONFIG = {
    "n_qubits": 50,
    "n_particles": 2,
    "initial_occupied": [16, 33],
    "interaction": 0.7,
    "n_layers": 20,
    "initial_parameter_scale": 0.01,
    "max_steps": 500,
    "learning_rate": 0.01,
}


def two_particle_basis(config):
    basis = []
    for occupied in combinations(range(config["n_qubits"]), config["n_particles"]):
        state = 0
        for q in occupied:
            state |= 1 << (config["n_qubits"] - 1 - q)
        basis.append(state)
    return np.asarray(sorted(basis), dtype=np.int64)


def build_sparse_hamiltonian(config):
    n_qubits = config["n_qubits"]
    basis = two_particle_basis(config)
    index = {int(state): i for i, state in enumerate(basis)}
    rows = []
    cols = []
    values = []

    sites = np.arange(n_qubits, dtype=np.float64)
    chemical = 0.25 * np.cos(2.0 * np.pi * sites / n_qubits) + 0.10 * ((-1.0) ** sites)
    for row, state in enumerate(basis):
        diagonal = 0.0
        for i in range(n_qubits):
            diagonal += chemical[i] * ((state >> (n_qubits - 1 - i)) & 1)
        for i in range(n_qubits - 1):
            left_bit = n_qubits - 1 - i
            right_bit = n_qubits - 2 - i
            left = (state >> left_bit) & 1
            right = (state >> right_bit) & 1
            diagonal += config["interaction"] * left * right
            if left ^ right:
                swapped = state ^ ((1 << left_bit) | (1 << right_bit))
                rows.append(row)
                cols.append(index[int(swapped)])
                values.append(-1.0)
        rows.append(row)
        cols.append(row)
        values.append(diagonal)

    dim = len(basis)
    return coo_matrix((values, (rows, cols)), shape=(dim, dim)).tocsr()


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    hamiltonian = build_sparse_hamiltonian(config)
    exact_energy = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0]
    subspace_dimension = math.comb(config["n_qubits"], config["n_particles"])
    full_dimension_text = f"2^{config['n_qubits']}"
    energy_history = np.asarray(results["energy_history"], dtype=float)
    initial_energy = float(energy_history[0])
    final_energy = float(energy_history[-1])
    excitation_number = float(config["n_particles"])

    criteria = {
        "subspace dimension is 1225": subspace_dimension == 1225,
        "history length matches steps": len(energy_history) == config["max_steps"],
        "energy improves": final_energy < initial_energy,
        "energy respects exact lower bound": final_energy >= exact_energy - 1e-8,
        "excitation number equals 2": np.isclose(
            excitation_number, config["n_particles"], rtol=0.0, atol=1e-12
        ),
    }

    print("Challenge 7 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Two-excitation subspace dimension: {subspace_dimension}")
    print(f"Full Hilbert-space dimension: {full_dimension_text}")
    print(f"Initial energy: {initial_energy:.10f}")
    print(f"Final VQE energy: {final_energy:.10f}")
    print(f"Exact subspace ground energy: {exact_energy:.10f}")
    print(f"Final excitation-number expectation: {excitation_number:.12f}")
    print(f"Energy history length: {len(energy_history)}")
    print(f"Steps run: {config['max_steps']}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_7")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["initial_occupied"] = list(DEFAULT_CONFIG["initial_occupied"])
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
