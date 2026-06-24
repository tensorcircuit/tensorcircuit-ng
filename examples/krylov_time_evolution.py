"""Krylov time-evolution benchmark with sparse-matrix and MVP Hamiltonians."""

import argparse
import time as timer
from typing import Any, List, Tuple

import numpy as np
from scipy.linalg import expm

import tensorcircuit as tc

tc.set_dtype("complex128")


def parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.replace(" ", "").split(",") if x]


def parse_ints(text: str) -> List[int]:
    return [int(x) for x in parse_floats(text)]


def sync(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def heisenberg_terms(
    n_qubits: int, hz: float, hx: float, hy: float
) -> Tuple[List[List[int]], List[float]]:
    structures = []
    weights = []
    for i in range(n_qubits - 1):
        for pauli in [3, 1, 2]:
            term = [0] * n_qubits
            term[i] = pauli
            term[i + 1] = pauli
            structures.append(term)
            weights.append(1.0)
    for i in range(n_qubits):
        for pauli, coeff in [(3, hz), (1, hx), (2, hy)]:
            if coeff != 0:
                term = [0] * n_qubits
                term[i] = pauli
                structures.append(term)
                weights.append(coeff)
    return structures, weights


def neel_state(n_qubits: int) -> Any:
    circuit = tc.Circuit(n_qubits)
    circuit.x([2 * i for i in range(n_qubits // 2)])
    return circuit.state()


def magnetization(state: Any) -> float:
    n_qubits = int(np.log2(state.shape[0]))
    circuit = tc.Circuit(n_qubits, inputs=state)
    total = sum(((-1) ** i) * circuit.expectation_ps(z=[i]) for i in range(n_qubits))
    return float(np.real(tc.backend.numpy(total)))


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
    tc.set_dtype("complex128")
    tc.set_backend(args.backend)

    structures, weights = heisenberg_terms(args.num_sites, args.hz, args.hx, args.hy)
    h_matrix = tc.quantum.PauliStringSum2COO(structures, weights)
    h_mvp = tc.quantum.PauliStringSum2MVP(structures, weights)
    operators = {"matrix": h_matrix, "mvp": h_mvp}
    if args.operator_mode != "both":
        operators = {args.operator_mode: operators[args.operator_mode]}

    runners = {}
    for name, operator in operators.items():

        def run(state: Any, times: Any, subspace_dim: int, op: Any = operator) -> Any:
            return tc.timeevol.krylov_evol(
                op, state, times, subspace_dim, scan_impl=args.scan_impl
            )

        runners[name] = tc.backend.jit(run, static_argnums=(2,)) if args.jit else run

    state0 = neel_state(args.num_sites)
    dim = 2**args.num_sites

    print(
        f"Krylov benchmark: n={args.num_sites}, dim={dim}, terms={len(structures)}, "
        f"backend={args.backend}, jit={args.jit}, scan={args.scan_impl}"
    )
    if dim <= args.exact_dim:
        print(f"Exact dense expm baseline enabled for dim <= {args.exact_dim}.")
    else:
        print(f"Exact dense expm baseline skipped for dim > {args.exact_dim}.")

    for t in args.time_points:
        exact_state = (
            exact_evolution(h_matrix, state0, t) if dim <= args.exact_dim else None
        )
        print(f"\nt = {t}")
        for m in args.subspace_dims:
            row = {}
            for name, runner in runners.items():
                start = timer.perf_counter()
                result = runner(state0, tc.backend.convert_to_tensor([t]), m)
                sync(result)
                row[name] = (result[0], timer.perf_counter() - start)

            if args.operator_mode == "both":
                matrix_state, matrix_time = row["matrix"]
                mvp_state, mvp_time = row["mvp"]
                diff = np.linalg.norm(
                    np.asarray(tc.backend.numpy(matrix_state - mvp_state))
                )
                fields = [
                    f"m={m}",
                    f"matrix={matrix_time:.4f}s",
                    f"mvp={mvp_time:.4f}s",
                    f"speedup={matrix_time / mvp_time:.2f}x",
                    f"|delta psi|={diff:.2e}",
                    f"mz={magnetization(mvp_state):.6f}",
                ]
                if exact_state is not None:
                    fields.append(f"F_exact={fidelity(exact_state, mvp_state):.8f}")
                print("  " + ", ".join(fields))
            else:
                name = args.operator_mode
                evolved, elapsed = row[name]
                fields = [
                    f"m={m}",
                    f"{name}={elapsed:.4f}s",
                    f"mz={magnetization(evolved):.6f}",
                ]
                if exact_state is not None:
                    fields.append(f"F_exact={fidelity(exact_state, evolved):.8f}")
                print("  " + ", ".join(fields))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_sites", type=int, default=10)
    parser.add_argument("--time_points", type=parse_floats, default="1,2,5")
    parser.add_argument("--subspace_dims", type=parse_ints, default="20,50,100")
    parser.add_argument("--hz", type=float, default=0.5)
    parser.add_argument("--hx", type=float, default=0.3)
    parser.add_argument("--hy", type=float, default=0.2)
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "jax", "tensorflow", "pytorch"],
    )
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--scan_impl", action="store_true")
    parser.add_argument(
        "--operator_mode",
        type=str,
        default="matrix",
        choices=["matrix", "mvp", "both"],
    )
    parser.add_argument("--exact_dim", type=int, default=1024)
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
