"""
Evaluation for Challenge Suite Problem 9.

The evaluator dynamically imports a solution module, passes a fully materialized
random local gate tape and Pauli-term list in config, consumes only the NumPy
result dictionary returned by run_solution, and prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np

DEFAULT_CONFIG = {
    "n_qubits": 512,
    "n_layers": 6,
    "max_steps": 100,
    "n_restarts": 200,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.02,
    "seed": 2035,
    "edge_keep_prob": 0.24,
    "success_threshold": 1.0,
}

PAULI_TERMS = (
    (0.5645931361768194, (("x", 388), ("z", 390))),
    (1.0, (("x", 16), ("y", 19))),
)


def ladder_edges(n_qubits):
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even for the two-leg ladder")
    n_columns = n_qubits // 2
    edges = []
    for column in range(n_columns):
        edges.append((2 * column, 2 * column + 1))
        if column + 1 < n_columns:
            edges.append((2 * column, 2 * (column + 1)))
            edges.append((2 * column + 1, 2 * (column + 1) + 1))
            if column % 2 == 0:
                edges.append((2 * column, 2 * (column + 1) + 1))
            else:
                edges.append((2 * column + 1, 2 * (column + 1)))
    return edges


def generate_gate_tape(config):
    rng = np.random.default_rng(config["seed"])
    single_axes = ("rx", "ry", "rz")
    two_axes = ("rxx", "ryy", "rzz")
    graph_edges = ladder_edges(config["n_qubits"])
    gate_tape = []
    parameter_index = 0

    for _ in range(config["n_layers"]):
        for qubit in range(config["n_qubits"]):
            axis = single_axes[int(rng.integers(len(single_axes)))]
            gate_tape.append((axis, qubit, parameter_index))
            parameter_index += 1

        edges = list(graph_edges)
        rng.shuffle(edges)
        used = set()
        for qubit_a, qubit_b in edges:
            if qubit_a in used or qubit_b in used:
                continue
            if rng.random() > config["edge_keep_prob"]:
                continue
            axis = two_axes[int(rng.integers(len(two_axes)))]
            gate_tape.append((axis, qubit_a, qubit_b, parameter_index))
            parameter_index += 1
            used.add(qubit_a)
            used.add(qubit_b)

    return tuple(gate_tape), parameter_index


def cone_size(gate_tape, pauli_term):
    support = {qubit for _, qubit in pauli_term}
    for gate in reversed(gate_tape):
        if len(gate) == 4 and (gate[1] in support or gate[2] in support):
            support.add(gate[1])
            support.add(gate[2])
    return len(support)


def finalize_config(config):
    config = dict(config)
    gate_tape, parameter_count = generate_gate_tape(config)
    config["gate_tape"] = gate_tape
    config["parameter_count"] = parameter_count
    config["pauli_terms"] = PAULI_TERMS
    config["pauli_cone_sizes"] = tuple(
        cone_size(gate_tape, term) for _, term in PAULI_TERMS
    )
    return config


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    history = np.asarray(results["observable_history"], dtype=float)
    initial = history[:, 0]
    final = history[:, -1]
    n_restarts = config["n_restarts"]
    steps_run = config["max_steps"]
    mean_initial = float(np.mean(initial))
    mean_final = float(np.mean(final))
    variance_final = float(np.var(final))
    std_final = float(np.std(final))
    best_final = float(np.max(final))
    success_fraction = float(np.mean(final >= config["success_threshold"]))

    criteria = {
        "history shape": history.shape == (config["n_restarts"], config["max_steps"]),
        "initial/final restart count": initial.shape == final.shape == (n_restarts,),
        "initial values finite": np.all(np.isfinite(initial)),
        "final values finite": np.all(np.isfinite(final)),
        "history values finite": np.all(np.isfinite(history)),
        "mean expectation improves": mean_final > mean_initial,
        "best final reaches threshold": best_final >= config["success_threshold"],
        "final variance finite": np.isfinite(variance_final) and variance_final >= 0.0,
    }

    print("Challenge 9 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Qubits: {config['n_qubits']}")
    print(f"Layers: {config['n_layers']}")
    print(f"Gate tape length: {len(config['gate_tape'])}")
    print(f"Parameter count: {config['parameter_count']}")
    print(f"Pauli terms: {config['pauli_terms']}")
    print(f"Pauli cone sizes: {config['pauli_cone_sizes']}")
    print(f"Restarts: {n_restarts}")
    print(f"Steps per restart: {steps_run}")
    print(f"Observable history shape: {history.shape}")
    print(f"Mean initial local objective: {mean_initial:.10f}")
    print(f"Mean final local objective: {mean_final:.10f}")
    print(f"Variance final local objective: {variance_final:.10e}")
    print(f"Std final local objective: {std_final:.10e}")
    print(f"Best final local objective: {best_final:.10f}")
    print(f"Success fraction: {success_fraction:.6f}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_9")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--n-restarts", type=int, default=DEFAULT_CONFIG["n_restarts"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    config["n_restarts"] = args.n_restarts
    evaluate(args.solution, finalize_config(config))


if __name__ == "__main__":
    main()
