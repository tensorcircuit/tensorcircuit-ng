"""
Evaluation for Challenge Suite Problem 10.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and computes a strict sparse-Lanczos
ground-state reference for the ordinary TFIM Hamiltonian.
"""

import argparse
import importlib
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

DEFAULT_CONFIG = {
    "n_qubits": 22,
    "selected_qubits": [
        0,
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
    ],
    "initial_ones": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    "n_layers": 2,
    "max_steps": 200,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.08,
    "zz_strength": 1.0,
    "x_strength": 1.05,
    "seed": 2040,
    "minimum_energy_improvement": 1e-3,
    "exact_tol": 1e-7,
    "exact_maxiter": 400,
    "exact_ncv": 24,
    "exact_lower_bound_tolerance": 1e-5,
    "maximum_energy_density_gap": 0.25,
}


def exact_tfim_ground_energy(config):
    n_qubits = config["n_qubits"]
    dim = 1 << n_qubits
    basis = np.arange(dim, dtype=np.uint32)
    diagonal = np.zeros(dim, dtype=np.float64)

    for q in range(n_qubits - 1):
        left = 1.0 - 2.0 * ((basis >> (n_qubits - 1 - q)) & 1).astype(np.float64)
        right = 1.0 - 2.0 * ((basis >> (n_qubits - 2 - q)) & 1).astype(np.float64)
        diagonal -= config["zz_strength"] * left * right

    masks = [np.uint32(1 << (n_qubits - 1 - q)) for q in range(n_qubits)]
    x_strength = float(config["x_strength"])

    def matvec(vector):
        result = diagonal * vector
        for mask in masks:
            result -= x_strength * vector[np.bitwise_xor(basis, mask)]
        return result

    operator = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    value = eigsh(
        operator,
        k=1,
        which="SA",
        tol=config["exact_tol"],
        maxiter=config["exact_maxiter"],
        ncv=config["exact_ncv"],
        return_eigenvectors=False,
    )[0]
    return float(value)


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    solution_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    exact_energy = exact_tfim_ground_energy(config)
    exact_elapsed = time.perf_counter() - start

    exact_density = exact_energy / config["n_qubits"]
    energy_history = np.asarray(results["energy_history"], dtype=float)
    initial_energy = float(energy_history[0])
    final_energy = float(energy_history[-1])
    final_parameters = np.asarray(results["final_parameters"], dtype=float)
    largest_gate_qubits = len(config["selected_qubits"])
    improvement = initial_energy - final_energy
    vqe_gap = final_energy - exact_density

    expected_param_shape = (
        config["n_layers"],
        config["n_qubits"],
        3,
    )
    criteria = {
        "18-qubit largest gate": largest_gate_qubits
        == len(config["selected_qubits"])
        == 18,
        "history length matches steps": energy_history.shape == (config["max_steps"],),
        "parameter shape": final_parameters.shape == expected_param_shape,
        "energy improves": improvement >= config["minimum_energy_improvement"],
        "history finite": np.all(np.isfinite(energy_history)),
        "above exact ground energy": final_energy
        >= exact_density - config["exact_lower_bound_tolerance"],
        "loose VQE gap threshold": vqe_gap <= config["maximum_energy_density_gap"],
    }

    print("Challenge 10 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"Solution time: {solution_elapsed:.2f}s")
    print(f"Exact reference time: {exact_elapsed:.2f}s")
    print(f"Qubits: {config['n_qubits']}")
    print(f"Selected CMZ qubits: {len(config['selected_qubits'])}")
    print(f"Largest ansatz gate qubits: {largest_gate_qubits}")
    print(f"Layers: {config['n_layers']}")
    print(f"Steps run: {config['max_steps']}")
    print(f"Initial energy density: {initial_energy:.10f}")
    print(f"Final energy density: {final_energy:.10f}")
    print(f"Exact ground energy density: {exact_density:.10f}")
    print(f"VQE energy-density gap: {vqe_gap:.10f}")
    print(f"Energy improvement: {improvement:.10f}")
    print(f"Energy history length: {len(energy_history)}")
    print(f"Final parameter shape: {final_parameters.shape}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_10")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--n-layers", type=int, default=DEFAULT_CONFIG["n_layers"])
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
    )
    parser.add_argument("--exact-tol", type=float, default=DEFAULT_CONFIG["exact_tol"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["selected_qubits"] = list(DEFAULT_CONFIG["selected_qubits"])
    config["initial_ones"] = list(DEFAULT_CONFIG["initial_ones"])
    config["max_steps"] = args.max_steps
    config["n_layers"] = args.n_layers
    config["learning_rate"] = args.learning_rate
    config["exact_tol"] = args.exact_tol
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
