"""
Evaluation for Challenge Suite Problem 12.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np
import quimb.tensor as qtn

DEFAULT_CONFIG = {
    "n_qubits": 32,
    "zz_anisotropy": 1.4,
    "staggered_field": 0.2,
    "dmrg_chi": 8,
    "dmrg_sweeps": 4,
    "dmrg_tolerance": 1e-7,
    "n_layers": 2,
    "max_steps": 5000,
    "learning_rate": 0.02,
    "initial_parameter_scale": 0.02,
    "seed": 2039,
    "fidelity_threshold": 0.85,
}


def parameter_count(config):
    n_qubits = config["n_qubits"]
    count = 0
    for layer in range(config["n_layers"]):
        count += 15 * len(range(layer % 2, n_qubits - 1, 2))
    return count


def build_xxz_mpo(config):
    n_qubits = config["n_qubits"]
    anisotropy = config["zz_anisotropy"]
    field = config["staggered_field"]
    hamiltonian = qtn.SpinHam1D(S=0.5)

    for i in range(n_qubits - 1):
        hamiltonian[i, i + 1] += 4.0, "X", "X"
        hamiltonian[i, i + 1] += 4.0, "Y", "Y"
        hamiltonian[i, i + 1] += 4.0 * anisotropy, "Z", "Z"

    for i in range(n_qubits):
        hamiltonian[i] += 2.0 * field * ((-1) ** i), "Z"

    return hamiltonian.build_mpo(n_qubits)


def dmrg_target_state(config):
    mpo = build_xxz_mpo(config)
    dmrg = qtn.DMRG2(mpo, bond_dims=[config["dmrg_chi"]], cutoffs=1e-8)
    dmrg.solve(
        tol=config["dmrg_tolerance"],
        max_sweeps=config["dmrg_sweeps"],
        verbosity=0,
    )
    dmrg.state.normalize()
    max_bond = int(max(dmrg.state.bond_sizes()))
    return dmrg.state, max_bond


def evaluate(solution_module, config):
    target_state, expected_target_max_bond = dmrg_target_state(config)
    solution_config = dict(config)
    solution_config["dmrg_state"] = target_state

    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(solution_config)
    elapsed = time.perf_counter() - start

    loss_history = np.asarray(results["loss_history"], dtype=float)
    fidelity_history = np.asarray(results["fidelity_history"], dtype=float)
    final_parameters = np.asarray(results["final_parameters"], dtype=float)
    final_fidelity = float(fidelity_history[-1])
    final_loss = float(loss_history[-1])
    final_phase = float(results["final_overlap_phase"])
    final_grad_norm = float(results["final_grad_norm"])

    finite_arrays = [loss_history, fidelity_history, final_parameters]
    criteria = {
        "loss history shape": loss_history.shape == (config["max_steps"],),
        "fidelity history shape": fidelity_history.shape == (config["max_steps"],),
        "final parameter shape": final_parameters.shape == (parameter_count(config),),
        "fidelity values finite": np.all(np.isfinite(fidelity_history)),
        "loss values finite": np.all(np.isfinite(loss_history)),
        "final values finite": all(np.all(np.isfinite(a)) for a in finite_arrays)
        and np.isfinite(final_fidelity)
        and np.isfinite(final_loss)
        and np.isfinite(final_phase)
        and np.isfinite(final_grad_norm),
        "fidelity improves": final_fidelity > float(fidelity_history[0]),
        "loss improves": final_loss < float(loss_history[0]),
        "final fidelity reaches threshold": final_fidelity
        >= config["fidelity_threshold"],
        "fidelity bounded": 0.0 <= final_fidelity <= 1.0 + 1e-5,
        "target bond dimension valid": 0
        < expected_target_max_bond
        <= config["dmrg_chi"],
    }

    print("Challenge 12 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Qubits: {config['n_qubits']}")
    print(f"DMRG chi: {config['dmrg_chi']}")
    print(f"DMRG sweeps: {config['dmrg_sweeps']}")
    print(f"Layers: {config['n_layers']}")
    print(f"Steps: {config['max_steps']}")
    print(f"Trainable circuit parameters: {parameter_count(config)}")
    print(f"Target MPS max bond dimension: {expected_target_max_bond}")
    print(f"Initial fidelity: {float(fidelity_history[0]):.8e}")
    print(f"Final fidelity: {final_fidelity:.8e}")
    print(f"Initial loss: {float(loss_history[0]):.8e}")
    print(f"Final loss: {final_loss:.8e}")
    print(f"Final overlap phase: {final_phase:.8e}")
    print(f"Final gradient norm: {final_grad_norm:.8e}")
    print(f"Fidelity history shape: {fidelity_history.shape}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_12")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
