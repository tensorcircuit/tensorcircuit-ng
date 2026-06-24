"""
Evaluation for Challenge Suite Problem 6.

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
    "n_qubits": 14,
    "n_blocks": 4,
    "t_min": 0.05,
    "t_max": 0.50,
    "ode_rtol": 1e-6,
    "ode_atol": 1e-6,
    "ode_max_steps": 16,
    "max_steps": 100,
    "learning_rate": 0.12,
}


def build_sparse_target(n_qubits):
    ls, w = [], []
    for i in range(n_qubits - 1):
        for p, c in ((1, 0.7), (2, 0.7), (3, 1.1)):
            s = [0] * n_qubits
            s[i] = p
            s[i + 1] = p
            ls.append(s)
            w.append(c)
    for i in range(n_qubits):
        s = [0] * n_qubits
        s[i] = 3
        ls.append(s)
        w.append(0.25 * ((-1.0) ** i))
    return tc.quantum.PauliStringSum2COO(
        np.array(ls, dtype=np.int32),
        np.array(w, dtype=np.float64),
        numpy=True,
    )


def format_array(values, precision=6):
    return np.array2string(
        np.asarray(values, dtype=float), precision=precision, separator=", "
    )


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    hamiltonian = build_sparse_target(config["n_qubits"])
    exact_energy = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0]
    exact_energy_density = exact_energy / config["n_qubits"]

    n_blocks = config["n_blocks"]
    t_min = config["t_min"]
    t_max = config["t_max"]

    final_times = np.asarray(results["final_analog_times"], dtype=float)
    final_couplings = np.asarray(results["final_analog_couplings"], dtype=float)
    final_detunings = np.asarray(results["final_analog_detunings"], dtype=float)
    energy_history = np.asarray(results["energy_density_history"], dtype=float)
    final_energy_density = float(energy_history[-1])

    criteria = {
        "energy history length": len(energy_history) == config["max_steps"],
        "analog times shape": final_times.shape == (n_blocks,),
        "analog times in bounds": bool(
            np.all(final_times > t_min) and np.all(final_times < t_max)
        ),
        "couplings in bounds": bool(np.all(np.abs(final_couplings) < 1.0)),
        "detunings in bounds": bool(np.all(np.abs(final_detunings) < 1.0)),
        "energy density improves": final_energy_density < float(energy_history[0]),
        "energy respects exact lower bound": final_energy_density
        >= exact_energy_density - 1e-6,
    }

    print("Challenge 6 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Exact sparse ground energy density: {exact_energy_density:.10f}")
    print(f"Initial energy density: {float(energy_history[0]):.10f}")
    print(f"Final energy density: {final_energy_density:.10f}")
    print(f"Learned analog times: {format_array(final_times)}")
    print(f"Learned analog couplings (J): {format_array(final_couplings)}")
    print(f"Learned analog detunings (Delta): {format_array(final_detunings)}")
    print(f"Energy history length: {len(energy_history)}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_6")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
