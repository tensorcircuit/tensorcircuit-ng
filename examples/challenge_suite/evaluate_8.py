"""
Evaluation for Challenge Suite Problem 8.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, computes exact IQP moment
references, and prints compact validation output.
"""

import argparse
import importlib
import time

import numpy as np

DEFAULT_CONFIG = {
    "grid_side": 7,
    "n_qubits": 49,
    "n_samples": 8192,
    "theta_offset": 0.43,
    "theta_sin_scale": 0.17,
    "theta_sin_frequency": 0.37,
    "theta_cos_scale": 0.11,
    "theta_cos_frequency": 0.19,
    "single_z_tolerance": 0.1,
    "edge_zz_tolerance": 0.1,
    "parity_tolerance": 1e-6,
}


def grid_sublayers(config):
    n_side = config["grid_side"]
    layers = []

    for parity in (0, 1):
        layers.append(
            [
                (row * n_side + col, row * n_side + col + 1)
                for row in range(n_side)
                for col in range(parity, n_side - 1, 2)
            ]
        )

    for parity in (0, 1):
        layers.append(
            [
                (row * n_side + col, (row + 1) * n_side + col)
                for row in range(parity, n_side - 1, 2)
                for col in range(n_side)
            ]
        )

    return layers


def edge_angle(index, left, right, config):
    return (
        config["theta_offset"]
        + config["theta_sin_scale"]
        * np.sin(config["theta_sin_frequency"] * (index + 1))
        + config["theta_cos_scale"]
        * np.cos(config["theta_cos_frequency"] * (left + 2 * right + 1))
    )


def grid_edges_and_angles(config):
    edges = []
    angles = []
    for layer in grid_sublayers(config):
        for left, right in layer:
            edges.append((left, right))
            angles.append(edge_angle(len(angles), left, right, config))
    return edges, np.asarray(angles, dtype=np.float64)


def exact_observables(config):
    edges, angles = grid_edges_and_angles(config)
    incident_cosines = [[] for _ in range(config["n_qubits"])]
    edge_angle_by_pair = {}
    for (left, right), theta in zip(edges, angles):
        cosine = np.cos(theta)
        incident_cosines[left].append(cosine)
        incident_cosines[right].append(cosine)
        edge_angle_by_pair[(left, right)] = cosine

    single_z = np.asarray([np.prod(values) for values in incident_cosines])
    edge_zz = []
    for left, right in edges:
        value = np.prod(incident_cosines[left]) * np.prod(incident_cosines[right])
        value /= edge_angle_by_pair[(left, right)] ** 2
        edge_zz.append(value)
    return single_z, np.asarray(edge_zz), 1.0


def empirical_observables(samples, config):
    edges, _ = grid_edges_and_angles(config)
    z_samples = 1.0 - 2.0 * samples.astype(np.float64)
    single_z = np.mean(z_samples, axis=0)
    edge_zz = np.asarray(
        [np.mean(z_samples[:, left] * z_samples[:, right]) for left, right in edges]
    )
    parity = np.mean(np.prod(z_samples, axis=1))
    return single_z, edge_zz, parity


def bitstrings(samples):
    return ["".join(str(int(bit)) for bit in row) for row in samples]


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    samples = np.asarray(results["samples"])
    exact_single_z, exact_edge_zz, exact_parity = exact_observables(config)
    empirical_single_z, empirical_edge_zz, empirical_parity = empirical_observables(
        samples, config
    )
    single_z_error = np.max(np.abs(empirical_single_z - exact_single_z))
    edge_zz_error = np.max(np.abs(empirical_edge_zz - exact_edge_zz))
    parity_error = abs(empirical_parity - exact_parity)

    criteria = {
        "sample shape": samples.shape == (config["n_samples"], config["n_qubits"]),
        "samples are binary": np.all((samples == 0) | (samples == 1)),
        "single-Z finite-sample error": single_z_error <= config["single_z_tolerance"],
        "grid-edge ZZ finite-sample error": edge_zz_error
        <= config["edge_zz_tolerance"],
        "parity symmetry": parity_error <= config["parity_tolerance"],
    }

    print("Challenge 8 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Grid side: {config['grid_side']}")
    print(f"Sample shape: {samples.shape}")
    print("First sampled bitstrings:")
    for value in bitstrings(samples[: min(10, len(samples))]):
        print(f"  {value}")
    print(f"Empirical full-grid parity: {empirical_parity:.10f}")
    print(f"Exact full-grid parity: {exact_parity:.10f}")
    print(f"Full-grid parity absolute error: {parity_error:.10f}")
    print(f"Max single-site Z absolute error: {single_z_error:.10f}")
    print(f"Max grid-edge ZZ absolute error: {edge_zz_error:.10f}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_8")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_CONFIG["n_samples"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["n_samples"] = args.n_samples
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
