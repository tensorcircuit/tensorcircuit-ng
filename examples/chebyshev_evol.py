"""Chebyshev time-evolution benchmark with sparse-matrix and MVP Hamiltonians."""

import argparse
import time as timer
from typing import Any, List, Tuple

import numpy as np
from scipy.linalg import expm

import tensorcircuit as tc

tc.set_dtype("complex128")


def heisenberg_terms(n_qubits: int) -> Tuple[List[List[int]], List[float]]:
    structures = []
    weights = []
    for i in range(n_qubits - 1):
        for pauli in [3, 1, 2]:
            term = [0] * n_qubits
            term[i] = pauli
            term[i + 1] = pauli
            structures.append(term)
            weights.append(1.0)
    return structures, weights


def sync(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def bound_probe(dim: int) -> Any:
    probe = np.cos(np.arange(dim)) + 0.5 * np.sin(0.37 * np.arange(dim))
    return tc.backend.convert_to_tensor(probe)


def fidelity(left: Any, right: Any) -> float:
    left_np = np.asarray(tc.backend.numpy(left))
    right_np = np.asarray(tc.backend.numpy(right))
    left_np = left_np / np.linalg.norm(left_np)
    right_np = right_np / np.linalg.norm(right_np)
    return float(abs(np.vdot(left_np, right_np)) ** 2)


def exact_evolution(hamiltonian: Any, state: Any, t: float) -> Any:
    h_dense = (
        tc.backend.to_dense(hamiltonian)
        if tc.backend.is_sparse(hamiltonian)
        else hamiltonian
    )
    evolved = expm(-1j * np.asarray(tc.backend.numpy(h_dense)) * t) @ np.asarray(
        tc.backend.numpy(state)
    )
    return tc.backend.convert_to_tensor(evolved)


def run_benchmark(args: argparse.Namespace) -> None:
    tc.set_backend(args.backend)
    tc.set_dtype("complex128")

    structures, weights = heisenberg_terms(args.num_sites)
    h_matrix = tc.quantum.PauliStringSum2COO(structures, weights)
    h_mvp = tc.quantum.PauliStringSum2MVP(structures, weights)
    operators = {"matrix": h_matrix, "mvp": h_mvp}
    if args.operator_mode != "both":
        operators = {args.operator_mode: operators[args.operator_mode]}

    dim = 2**args.num_sites
    state0 = tc.backend.ones([dim])
    state0 = state0 / tc.backend.norm(state0)
    probe = bound_probe(dim)
    exact_state = (
        exact_evolution(h_matrix, state0, args.time) if dim <= args.exact_dim else None
    )

    print(
        f"Chebyshev benchmark: n={args.num_sites}, dim={dim}, terms={len(structures)}, "
        f"backend={args.backend}, jit={args.jit}"
    )
    if exact_state is None:
        print(f"Exact dense expm baseline skipped for dim > {args.exact_dim}.")
    else:
        print(f"Exact dense expm baseline enabled for dim <= {args.exact_dim}.")

    results = {}
    for name, operator in operators.items():
        start = timer.perf_counter()
        estimated_bounds = tc.timeevol.estimate_spectral_bounds(
            operator,
            n_iter=args.bound_iter,
            psi0=probe,
        )
        sync(estimated_bounds[0])
        sync(estimated_bounds[1])
        bounds = (
            float(estimated_bounds[0]) + args.bound_padding,
            float(estimated_bounds[1]) - args.bound_padding,
        )
        bound_time = timer.perf_counter() - start
        k = tc.timeevol.estimate_k(args.time, bounds)
        m = tc.timeevol.estimate_M(args.time, bounds, k=k)

        def run(state: Any, op: Any = operator) -> Any:
            return tc.timeevol.chebyshev_evol(op, state, args.time, bounds, k, m)

        runner = tc.backend.jit(run) if args.jit else run
        start = timer.perf_counter()
        evolved = runner(state0)
        sync(evolved)
        evol_time = timer.perf_counter() - start
        results[name] = {
            "state": evolved,
            "bound_time": bound_time,
            "evol_time": evol_time,
        }
        fields = [
            f"{name}",
            f"bounds=({bounds[0]:.6f}, {bounds[1]:.6f})",
            f"bound_time={bound_time:.4f}s",
            f"k={k}",
            f"M={m}",
            f"evol_time={evol_time:.4f}s",
            f"norm={float(tc.backend.numpy(tc.backend.norm(evolved))):.8f}",
        ]
        if exact_state is not None:
            fields.append(f"F_exact={fidelity(exact_state, evolved):.8f}")
        print("  " + ", ".join(fields))

    if args.operator_mode == "both":
        matrix_state = results["matrix"]["state"]
        mvp_state = results["mvp"]["state"]
        diff = np.linalg.norm(np.asarray(tc.backend.numpy(matrix_state - mvp_state)))
        print(
            "  comparison, "
            f"bound_speedup={results['matrix']['bound_time'] / results['mvp']['bound_time']:.2f}x, "
            f"evol_speedup={results['matrix']['evol_time'] / results['mvp']['evol_time']:.2f}x, "
            f"|delta psi|={diff:.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_sites", type=int, default=8)
    parser.add_argument("--time", type=float, default=500.0)
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "jax", "tensorflow", "pytorch"],
    )
    parser.add_argument("--jit", action="store_true")
    parser.add_argument(
        "--operator_mode",
        type=str,
        default="matrix",
        choices=["matrix", "mvp", "both"],
    )
    parser.add_argument("--bound_iter", type=int, default=40)
    parser.add_argument("--bound_padding", type=float, default=0.1)
    parser.add_argument("--exact_dim", type=int, default=1024)
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
