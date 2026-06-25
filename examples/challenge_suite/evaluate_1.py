"""
Evaluation for Challenge Suite Problem 1.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and computes reference metrics.
"""

import argparse
import importlib
import time

import numpy as np
import quimb.tensor as qtn

DEFAULT_CONFIG = {
    "n_qubits": 32,
    "field": 1.05,
    "dmrg_chi": 8,
    "dmrg_sweeps": 2,
    "n_layers": 4,
    "max_steps": 500,
    "learning_rate": 0.005,
}
DMRG_ENERGY_WINDOW = 2.5e-3
ALLOWED_DMRG_REGRESSION = 5.0e-4


def build_tfim_mpo(config):
    hamiltonian = qtn.SpinHam1D(S=0.5)
    hamiltonian += -4.0, "Z", "Z"
    hamiltonian += -2.0 * config["field"], "X"
    return hamiltonian.build_mpo(config["n_qubits"])


def dmrg_initial_state(config):
    mpo = build_tfim_mpo(config)
    dmrg = qtn.DMRG2(mpo, bond_dims=[config["dmrg_chi"]], cutoffs=1e-8)
    dmrg.solve(tol=1e-7, max_sweeps=config["dmrg_sweeps"], verbosity=0)
    dmrg.state.normalize()
    return dmrg.state, float(dmrg.energy)


def evaluate(solution_module, config):
    dmrg_state, dmrg_energy = dmrg_initial_state(config)
    solution_config = dict(config)
    solution_config["dmrg_state"] = dmrg_state
    solution_config["dmrg_energy"] = dmrg_energy

    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(solution_config)
    elapsed = time.perf_counter() - start

    energy_history = np.asarray(results["energy_history"], dtype=float)
    initial_energy = float(energy_history[0])
    final_energy = float(energy_history[-1])
    initial_shift = initial_energy - dmrg_energy
    final_shift = final_energy - dmrg_energy
    energy_gain = initial_energy - final_energy
    criteria = {
        "energy history length": len(energy_history) == config["max_steps"],
        "history finite": np.all(np.isfinite(energy_history)),
        "initial energy stays near DMRG": abs(initial_shift) <= DMRG_ENERGY_WINDOW,
        "final energy stays near DMRG": abs(final_shift) <= DMRG_ENERGY_WINDOW,
        "refinement does not regress too far": final_shift <= ALLOWED_DMRG_REGRESSION,
    }

    print("Challenge 1 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"DMRG reference energy: {dmrg_energy:.8f}")
    print(f"Initial variational energy: {initial_energy:.8f}")
    print(f"Final variational energy: {final_energy:.8f}")
    print(f"Initial minus DMRG reference: {initial_shift:.8e}")
    print(f"Final minus DMRG reference: {final_shift:.8e}")
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
