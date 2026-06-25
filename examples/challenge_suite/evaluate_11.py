"""
Evaluation for Challenge Suite Problem 11.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np

DEFAULT_CONFIG = {
    "n_qubits": 20,
    "n_field_params": 2,
    "n_layers": 6,
    "max_steps": 300,
    "learning_rate": 0.01,
    "initial_parameter_scale": 0.02,
    "readout_penalty_weight": 0.05,
    "seed": 2037,
    "final_response_mse_tolerance": 1e-9,
}


def sensor_positions(config):
    return np.linspace(-1.0, 1.0, config["n_qubits"], dtype=np.float64)


def target_response_matrix(config):
    x_sites = sensor_positions(config)
    return np.stack([np.ones(config["n_qubits"]), x_sites], axis=1)


def parameter_count(config):
    n_qubits = config["n_qubits"]
    count = 0
    for layer in range(config["n_layers"]):
        count += 2 * n_qubits
        count += len(range(layer % 2, n_qubits - 1, 2))
    return count


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    loss_history = np.asarray(results["loss_history"], dtype=float)
    response_mse_history = np.asarray(results["response_mse_history"], dtype=float)
    readout_penalty_history = np.asarray(
        results["readout_penalty_history"], dtype=float
    )
    response = np.asarray(results["final_response_matrix"], dtype=float)
    exact_target = target_response_matrix(config)
    zero_readouts = np.asarray(results["final_zero_field_readouts"], dtype=float)

    final_response_mse = float(np.mean((response - exact_target) ** 2))
    final_readout_penalty = float(np.mean(zero_readouts**2))
    final_loss = (
        final_response_mse + config["readout_penalty_weight"] * final_readout_penalty
    )

    finite_arrays = [
        loss_history,
        response_mse_history,
        readout_penalty_history,
        response,
        zero_readouts,
    ]
    criteria = {
        "loss history shape": loss_history.shape == (config["max_steps"],),
        "response mse history shape": response_mse_history.shape
        == (config["max_steps"],),
        "readout penalty history shape": readout_penalty_history.shape
        == (config["max_steps"],),
        "response shape": response.shape
        == (config["n_qubits"], config["n_field_params"]),
        "zero readout shape": zero_readouts.shape == (config["n_qubits"],),
        "response mse improves": final_response_mse < float(response_mse_history[0]),
        "final response mse <= tolerance": final_response_mse
        <= config["final_response_mse_tolerance"],
        "loss improves": final_loss < float(loss_history[0]),
        "returned arrays finite": all(np.all(np.isfinite(a)) for a in finite_arrays),
    }

    print("Challenge 11 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Qubits: {config['n_qubits']}")
    print(f"Layers: {config['n_layers']}")
    print(f"Physical field parameters: {config['n_field_params']}")
    print(f"Readout observables: {config['n_qubits']}")
    print(f"Trainable circuit parameters: {parameter_count(config)}")
    print(f"Initial response MSE: {float(response_mse_history[0]):.8e}")
    print(f"Final response MSE: {final_response_mse:.8e}")
    print(f"Final zero-field readout penalty: {final_readout_penalty:.8e}")
    print(f"Initial total loss: {float(loss_history[0]):.8e}")
    print(f"Final total loss: {final_loss:.8e}")
    print(f"Loss history shape: {loss_history.shape}")
    print(f"Final response matrix shape: {response.shape}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_11")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
