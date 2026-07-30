"""
Benchmark of fixed-schedule Taylor evolution against SciPy expm_multiply.
"""

import argparse
import time
from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.sparse.linalg import expm_multiply as scipy_expm_multiply

import tensorcircuit as tc


def heisenberg_terms(n_qubits: int) -> Tuple[List[List[int]], List[float]]:
    """Return open-boundary Heisenberg-chain Pauli terms."""
    structures = []
    weights = []
    for i in range(n_qubits - 1):
        for pauli in (1, 2, 3):
            term = [0] * n_qubits
            term[i] = pauli
            term[i + 1] = pauli
            structures.append(term)
            weights.append(1.0)
    return structures, weights


def sync(value: Any) -> None:
    """Synchronize an asynchronous JAX result when supported."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def median_runtime(fn: Callable[[], Any], repeats: int) -> Tuple[Any, float]:
    """Return the last result and median synchronized wall-clock time."""
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        sync(result)
        timings.append(time.perf_counter() - start)
    return result, float(np.median(timings))


def benchmark_case(n_qubits: int, t: float, repeats: int, operator_mode: str) -> None:
    """Benchmark one Hilbert-space size and evolution time."""
    structures, weights = heisenberg_terms(n_qubits)
    h_sparse = tc.quantum.PauliStringSum2COO(structures, weights)
    h_mvp = tc.quantum.PauliStringSum2MVP(structures, weights)
    h_scipy = tc.quantum.PauliStringSum2COO(structures, weights, numpy=True).tocsr()
    dimension = 2**n_qubits
    psi_numpy = np.cos(np.arange(dimension)) + 1.0j * np.sin(
        0.37 * np.arange(dimension)
    )
    psi_numpy = psi_numpy / np.linalg.norm(psi_numpy)
    psi = tc.backend.convert_to_tensor(psi_numpy)

    # Every Pauli string has 1-norm one. The triangle inequality gives a
    # static, conservative bound on the operator 1-norm.
    m, s = tc.timeevol.estimate_expm_multiply_parameters(
        t, sum(abs(w) for w in weights)
    )
    scipy_result, scipy_seconds = median_runtime(
        lambda: scipy_expm_multiply(-1.0j * t * h_scipy, psi_numpy), repeats
    )
    operators = {"sparse": h_sparse, "mvp": h_mvp}
    if operator_mode != "both":
        operators = {operator_mode: operators[operator_mode]}

    for name, operator in operators.items():

        def evolve(state: Any, op: Any = operator) -> Any:
            return tc.timeevol.expm_multiply_evol(op, state, t, m=m, s=s)

        runner = tc.backend.jit(evolve)
        start = time.perf_counter()
        first_result = runner(psi)
        sync(first_result)
        first_seconds = time.perf_counter() - start
        result, steady_seconds = median_runtime(lambda: runner(psi), repeats)
        error = np.linalg.norm(np.asarray(result) - scipy_result)
        print(
            f"n={n_qubits:2d}, dim={dimension:6d}, t={t:5.2f}, {name:6s}, "
            f"m={m:2d}, s={s:2d}, JAX first={first_seconds:.5f}s, "
            f"JAX steady={steady_seconds:.5f}s, SciPy={scipy_seconds:.5f}s, "
            f"|delta|={error:.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sites", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--times", type=float, nargs="+", default=[0.2, 1.0, 5.0])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--operator-mode", choices=["sparse", "mvp", "both"], default="both"
    )
    args = parser.parse_args()

    tc.set_backend("jax")
    tc.set_dtype("complex128")
    print(
        "Fixed-schedule Taylor expm benchmark on JAX. "
        "JAX first includes compilation; JAX steady and SciPy are median execution times."
    )
    for n_qubits in args.num_sites:
        for t in args.times:
            benchmark_case(n_qubits, t, args.repeats, args.operator_mode)


if __name__ == "__main__":
    main()
