"""
Evaluation for Challenge Suite Problem 7.
"""

import argparse
import importlib
import time

import numpy as np

DEFAULT_CONFIG = {
    "n_data_qubits": 8,
    "n_ancilla_qubits": 8,
    "n_qubits": 16,
    "n_layers": 2,
    "n_trajectories": 128,
    "initial_parameter_scale": 0.1,
    "max_steps": 100,
    "learning_rate": 0.02,
    "seed": 2047,
    "transverse_field": 1.05,
    "minimum_improvement": 0.3,
    "target_final_energy": -8.3,
}


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    energy_history = np.asarray(results["energy_history"])
    final_trajectory_energies = np.asarray(results["final_trajectory_energies"])
    initial_energy = float(energy_history[0])
    final_energy = float(energy_history[-1])
    improvement = initial_energy - final_energy

    criteria = {
        "energy history length": len(energy_history) == config["max_steps"],
        "trajectory energy shape": final_trajectory_energies.shape
        == (config["n_trajectories"],),
        "energy decreases": final_energy < initial_energy,
        "minimum improvement": improvement >= config["minimum_improvement"],
        "target final energy": final_energy <= config["target_final_energy"],
    }

    print("Challenge 7 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Initial trajectory-averaged energy: {initial_energy:.10f}")
    print(f"Final trajectory-averaged energy: {final_energy:.10f}")
    print(f"Energy improvement: {improvement:.10f}")
    print(
        "Final trajectory energy mean/std: "
        f"{np.mean(final_trajectory_energies):.10f} / {np.std(final_trajectory_energies):.10f}"
    )
    print(f"Energy history length: {len(energy_history)}")
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
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
