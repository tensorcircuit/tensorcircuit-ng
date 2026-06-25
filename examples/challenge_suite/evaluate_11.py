"""
Evaluation for Challenge Suite Problem 11.

The evaluator dynamically imports a solution module, consumes only the NumPy
result dictionary returned by run_solution, and computes an exact sparse
ground-state reference for the same spin-1 chain outside the timed solution run.
"""

import argparse
import importlib
import time
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh

DIM = 3
SQRT2 = np.sqrt(2.0)

SX = (
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    / SQRT2
)
SY = (
    np.array(
        [[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]],
        dtype=np.complex128,
    )
    / SQRT2
)
SZ = np.diag([1.0, 0.0, -1.0]).astype(np.complex128)
SZ2 = np.diag([1.0, 0.0, 1.0]).astype(np.complex128)
DOT_BOND = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
DOT_BOND_SQUARED = DOT_BOND @ DOT_BOND
MS = np.array([1.0, 0.0, -1.0], dtype=np.float64)
MIDDLE_VALUES = np.array([-1.0, 1.0, -1.0], dtype=np.float64)

DEFAULT_CONFIG = {
    "n_sites": 12,
    "n_layers": 5,
    "beta": 0.20,
    "single_ion_anisotropy": 0.15,
    "max_steps": 500,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.05,
    "seed": 2041,
    "minimum_energy_improvement": 5e-3,
    "maximum_energy_density_gap": 0.12,
    "maximum_string_order_mae": 0.12,
}


def string_pairs(config):
    n_sites = config["n_sites"]
    return tuple((i, n_sites - 1 - i) for i in range(3))


@lru_cache(maxsize=None)
def identity_shell(n_sites):
    return eye(DIM**n_sites, dtype=np.complex128, format="csr")


def embed_one_site_operator(op, site, n_sites):
    return kron(
        kron(identity_shell(site), csr_matrix(op), format="csr"),
        identity_shell(n_sites - site - 1),
        format="csr",
    )


def embed_two_site_operator(op, left, n_sites):
    return kron(
        kron(identity_shell(left), csr_matrix(op), format="csr"),
        identity_shell(n_sites - left - 2),
        format="csr",
    )


def build_hamiltonian(config):
    n_sites = config["n_sites"]
    dim = DIM**n_sites
    hamiltonian = csr_matrix((dim, dim), dtype=np.complex128)
    bond_term = DOT_BOND + config["beta"] * DOT_BOND_SQUARED
    for left in range(n_sites - 1):
        hamiltonian += embed_two_site_operator(bond_term, left, n_sites)
    onsite_prefactor = config["single_ion_anisotropy"]
    for site in range(n_sites):
        hamiltonian += onsite_prefactor * embed_one_site_operator(SZ2, site, n_sites)
    return hamiltonian


@lru_cache(maxsize=None)
def basis_digits(n_sites):
    dim = DIM**n_sites
    digits = np.empty((dim, n_sites), dtype=np.int8)
    values = np.arange(dim, dtype=np.int64)
    for site in range(n_sites - 1, -1, -1):
        digits[:, site] = values % DIM
        values //= DIM
    return digits


def string_order_from_state(state, config, pair):
    i, j = pair
    digits = basis_digits(config["n_sites"])
    weights = MS[digits[:, i]] * MS[digits[:, j]]
    for site in range(i + 1, j):
        weights *= MIDDLE_VALUES[digits[:, site]]
    probabilities = np.abs(state) ** 2
    return float(np.sum(probabilities * weights))


def exact_reference(config):
    hamiltonian = build_hamiltonian(config)
    value, vector = eigsh(hamiltonian, k=1, which="SA", tol=1e-8, maxiter=400)
    ground = vector[:, 0]
    energy_density = float(np.real(value[0])) / config["n_sites"]
    strings = [
        string_order_from_state(ground, config, pair) for pair in string_pairs(config)
    ]
    return energy_density, np.asarray(strings, dtype=np.float64)


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    exact_energy_density, exact_strings = exact_reference(config)
    exact_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    pairs = string_pairs(config)
    energy_history = np.asarray(results["energy_density_history"], dtype=float)
    final_energy_density = float(
        np.asarray(results["final_energy_density"], dtype=float)
    )
    final_strings = np.asarray(results["final_string_orders"], dtype=float)
    string_mae = float(np.mean(np.abs(final_strings - exact_strings)))
    improvement = float(energy_history[0] - final_energy_density)
    gap = float(final_energy_density - exact_energy_density)

    criteria = {
        "energy history shape": energy_history.shape == (config["max_steps"],),
        "final string shape": final_strings.shape == (len(pairs),),
        "energy improves": improvement >= config["minimum_energy_improvement"],
        "returned arrays finite": np.all(np.isfinite(energy_history))
        and np.isfinite(final_energy_density)
        and np.all(np.isfinite(final_strings)),
        "energy gap threshold": gap <= config["maximum_energy_density_gap"],
        "string-order mae threshold": string_mae <= config["maximum_string_order_mae"],
    }

    print("Challenge 11 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Exact reference time: {exact_elapsed:.2f}s")
    print(f"Sites: {config['n_sites']}")
    print(f"Layers: {config['n_layers']}")
    print(f"Steps: {config['max_steps']}")
    print(f"Initial energy density: {float(energy_history[0]):.10f}")
    print(f"Final history energy density: {float(energy_history[-1]):.10f}")
    print(f"Final returned energy density: {final_energy_density:.10f}")
    print(f"Exact ground-state density: {exact_energy_density:.10f}")
    print(f"Energy-density gap: {gap:.10f}")
    print(f"Energy improvement: {improvement:.10f}")
    print("String correlators (final / exact):")
    for pair, value, ref in zip(pairs, final_strings, exact_strings):
        print(f"  O_string^z{pair}: {value:.10f} / {ref:.10f}")
    print(f"String-order MAE: {string_mae:.10f}")
    print(f"Energy history shape: {energy_history.shape}")
    print(f"Final string-order shape: {final_strings.shape}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_11")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--n-layers", type=int, default=DEFAULT_CONFIG["n_layers"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["max_steps"] = args.max_steps
    config["n_layers"] = args.n_layers
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
