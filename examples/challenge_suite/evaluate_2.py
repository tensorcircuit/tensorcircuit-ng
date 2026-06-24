"""
Evaluation for Challenge Suite Problem 2.

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
    "zz_anisotropy": 1.2,
    "staggered_field": 0.35,
    "n_layers": 6,
    "subsystem_size": 6,
    "target_entropies": np.array([0.30, 0.60, 0.80], dtype=np.float32),
    "entropy_weight": 0.25,
    "max_steps": 500,
    "learning_rate": 0.015,
}


def build_sparse_xxz(n_qubits, zz_anisotropy, staggered_field):
    patterns = []
    weights = []

    for i in range(n_qubits - 1):
        for pauli, weight in ((1, 1.0), (2, 1.0), (3, zz_anisotropy)):
            pattern = [0] * n_qubits
            pattern[i] = pauli
            pattern[i + 1] = pauli
            patterns.append(pattern)
            weights.append(weight)

    for i in range(n_qubits):
        pattern = [0] * n_qubits
        pattern[i] = 3
        patterns.append(pattern)
        weights.append(staggered_field * ((-1.0) ** i))

    return tc.quantum.PauliStringSum2COO(
        np.array(patterns, dtype=np.int32),
        np.array(weights, dtype=np.float32),
        numpy=True,
    )


def format_array(values):
    return np.array2string(np.asarray(values, dtype=float), precision=6, separator=", ")


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    n_qubits = config["n_qubits"]
    zz_anisotropy = config["zz_anisotropy"]
    staggered_field = config["staggered_field"]
    hamiltonian = build_sparse_xxz(n_qubits, zz_anisotropy, staggered_field)
    exact_energy = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0]
    exact_energy_density = exact_energy / n_qubits

    energy_history = np.asarray(results["energy_density_history"], dtype=float)
    loss_history = np.asarray(results["loss_history"], dtype=float)
    entropy_mse_history = np.asarray(results["entropy_mse_history"], dtype=float)
    entropy_history = np.asarray(results["entropy_history"], dtype=float)
    final_entropies = entropy_history[-1]
    target_entropies = np.asarray(config["target_entropies"], dtype=float)
    recomputed_entropy_mse = float(np.mean((final_entropies - target_entropies) ** 2))
    entropy_rmse = float(np.sqrt(recomputed_entropy_mse))
    reported_entropy_mse = float(entropy_mse_history[-1])

    criteria = {
        "energy history length": len(energy_history) == config["max_steps"],
        "loss history length": len(loss_history) == config["max_steps"],
        "entropy mse history length": len(entropy_mse_history) == config["max_steps"],
        "entropy history shape": entropy_history.shape
        == (config["max_steps"], len(config["target_entropies"])),
        "loss improves": float(loss_history[-1]) < float(loss_history[0]),
        "energy density improves": float(energy_history[-1]) < float(energy_history[0]),
        "reported entropy mse matches": np.isclose(
            recomputed_entropy_mse, reported_entropy_mse, rtol=1e-5, atol=1e-7
        ),
    }

    print("Challenge 2 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Exact sparse ground energy density: {exact_energy_density:.8f}")
    print(f"Initial energy density: {float(energy_history[0]):.8f}")
    print(f"Final energy density: {float(energy_history[-1]):.8f}")
    print(f"Final block entropies: {format_array(final_entropies)}")
    print(f"Target entropies: {format_array(target_entropies)}")
    print(f"Entropy-profile MSE: {reported_entropy_mse:.8e}")
    print(f"Entropy-profile RMSE: {entropy_rmse:.8e}")
    print(f"Initial loss: {float(loss_history[0]):.8f}")
    print(f"Final loss: {float(loss_history[-1]):.8f}")
    print(f"Energy history length: {len(energy_history)}")
    print(f"Loss history length: {len(loss_history)}")
    print(f"Entropy history shape: {entropy_history.shape}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_2")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["target_entropies"] = np.asarray(
        DEFAULT_CONFIG["target_entropies"], dtype=np.float32
    )
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
