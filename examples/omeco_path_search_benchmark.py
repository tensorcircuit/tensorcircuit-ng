"""
Search-only benchmark for TensorCircuit contraction paths.

The benchmark reports realized path quality and search time on the same
TensorCircuit topology.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import time
from typing import Any

import cotengra as ctg
import numpy as np
import omeco
import tensorcircuit as tc

tc.set_backend("numpy")
tc.set_dtype("complex64")

DEFAULT_WORKLOADS = ["amplitude_ladder", "amplitude_brickwork", "vqe_expectation_1d"]
DEFAULT_STRATEGIES = [
    "cotengra_combo_trials",
    "cotengra_sa_trials",
    "omeco_treesa_trials",
]
DEFAULT_CASES = ["12x8"]


def package_version(name: str, fallback: str = "missing") -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def parse_case(text: str) -> tuple[int, int]:
    n_text, layers_text = text.split("x", maxsplit=1)
    return int(n_text), int(layers_text)


def parse_parallel(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "auto":
        return "auto"
    raise ValueError("cotengra parallel must be one of: auto, true, false")


def get_combo_trials(args: argparse.Namespace) -> int:
    if args.combo_trials > 0:
        return args.combo_trials
    return 8 * args.trials


def get_sa_temperatures(args: argparse.Namespace) -> list[float]:
    return [
        float(x) for x in np.geomspace(args.sa_tstart, args.sa_tfinal, args.sa_iters)
    ]


def get_omeco_betas(args: argparse.Namespace) -> list[float]:
    return [1.0 / t for t in get_sa_temperatures(args)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(DEFAULT_WORKLOADS),
        choices=list(DEFAULT_WORKLOADS),
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=[
            "cotengra_combo_trials",
            "cotengra_sa_trials",
            "omeco_treesa_trials",
            "omeco_greedy",
        ],
    )
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--combo-trials", type=int, default=0)
    parser.add_argument("--sa-iters", type=int, default=64)
    parser.add_argument("--sa-tstart", type=float, default=2.0)
    parser.add_argument("--sa-tfinal", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cotengra-parallel", default="auto")
    parser.add_argument("--cotengra-max-time", type=float, default=3600.0)
    parser.add_argument("--omeco-tc-weight", type=float, default=1.0)
    parser.add_argument("--omeco-sc-weight", type=float, default=0.0)
    parser.add_argument("--omeco-rw-weight", type=float, default=64.0)
    parser.add_argument("--omeco-sc-target", type=float, default=20.0)
    return parser.parse_args()


def print_environment(args: argparse.Namespace) -> None:
    print("Environment:")
    print(f"  backend={tc.backend.name}")
    print(f"  tensorcircuit={tc.__version__}")
    print(f"  cotengra={package_version('cotengra')}")
    print(f"  cmaes={package_version('cmaes')}")
    print(f"  omeco={package_version('omeco')}")
    print(
        f"  cotengrust_importable={importlib.util.find_spec('cotengrust') is not None}"
    )
    print(f"  cotengra_parallel={args.cotengra_parallel}")
    print(
        f"  trials={args.trials}, combo_trials={get_combo_trials(args)}, "
        f"sa_iters={args.sa_iters}, repeats={args.repeats}"
    )
    print(
        f"  sa_schedule=tstart:{args.sa_tstart}, tfinal:{args.sa_tfinal}, "
        f"beta_steps:{len(get_omeco_betas(args))}"
    )
    print(
        "  omeco_score="
        f"tc:{args.omeco_tc_weight}, "
        f"sc:{args.omeco_sc_weight}, "
        f"rw:{args.omeco_rw_weight}, "
        f"sc_target:{args.omeco_sc_target}"
    )


def make_params(count: int) -> np.ndarray:
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.linspace(0.05, 0.95, count, dtype=np.float32)


def build_ladder_circuit(n: int, layers: int) -> tc.Circuit:
    circuit = tc.Circuit(n)
    circuit.h(range(n))
    params = make_params(layers * (n - 1))
    index = 0
    for _ in range(layers):
        for i in range(n - 1):
            circuit.rzz(i, i + 1, theta=float(params[index]))
            index += 1
    return circuit


def build_brickwork_circuit(n: int, layers: int) -> tc.Circuit:
    circuit = tc.Circuit(n)
    circuit.h(range(n))
    nparams = layers * (((n - 1) // 2) + (n // 2))
    params = make_params(nparams)
    index = 0
    for _ in range(layers):
        for i in range(0, n - 1, 2):
            circuit.rzz(i, i + 1, theta=float(params[index]))
            index += 1
        for i in range(1, n - 1, 2):
            circuit.rzz(i, i + 1, theta=float(params[index]))
            index += 1
    return circuit


def build_vqe_circuit(n: int, layers: int) -> tc.Circuit:
    circuit = tc.Circuit(n)
    params = make_params(layers * (3 * n - 1))
    index = 0
    for _ in range(layers):
        for i in range(n):
            circuit.rx(i, theta=float(params[index]))
            index += 1
            circuit.ry(i, theta=float(params[index]))
            index += 1
        for i in range(n - 1):
            circuit.rzz(i, i + 1, theta=float(params[index]))
            index += 1
    return circuit


def workload_evaluator(workload: str, n: int, layers: int) -> Any:
    if workload == "amplitude_ladder":
        bitstring = "0" * n

        def evaluate_ladder() -> Any:
            return build_ladder_circuit(n, layers).amplitude(bitstring)

        return evaluate_ladder

    if workload == "amplitude_brickwork":
        bitstring = "0" * n

        def evaluate_brickwork() -> Any:
            return build_brickwork_circuit(n, layers).amplitude(bitstring)

        return evaluate_brickwork

    if workload == "vqe_expectation_1d":

        def evaluate_vqe_expectation() -> Any:
            circuit = build_vqe_circuit(n, layers)
            return circuit.expectation(
                [tc.gates.z(), [0]],
                [tc.gates.z(), [1]],
                reuse=False,
            )

        return evaluate_vqe_expectation

    raise ValueError(f"Unknown workload: {workload}")


def build_optimizer(strategy: str, args: argparse.Namespace) -> Any:
    if strategy == "cotengra_combo_trials":
        return ctg.ReusableHyperOptimizer(
            methods=["greedy", "kahypar"],
            optlib="cmaes",
            parallel=parse_parallel(args.cotengra_parallel),
            minimize="combo",
            max_time=args.cotengra_max_time,
            max_repeats=get_combo_trials(args),
            progbar=False,
        )

    if strategy == "cotengra_sa_trials":
        return ctg.ReusableHyperOptimizer(
            methods=["greedy", "kahypar"],
            optlib="cmaes",
            parallel=parse_parallel(args.cotengra_parallel),
            minimize="combo",
            max_time=args.cotengra_max_time,
            max_repeats=args.trials,
            progbar=False,
            seed=11,
            simulated_annealing_opts={
                "tstart": args.sa_tstart,
                "tfinal": args.sa_tfinal,
                "tsteps": args.sa_iters,
                "numiter": args.sa_iters,
                "minimize": "combo",
                "seed": 11,
                "progbar": False,
            },
        )

    if strategy == "omeco_treesa_trials":
        score = omeco.ScoreFunction(
            tc_weight=args.omeco_tc_weight,
            sc_weight=args.omeco_sc_weight,
            rw_weight=args.omeco_rw_weight,
            sc_target=args.omeco_sc_target,
        )
        optimizer = omeco.TreeSA(
            ntrials=args.trials,
            niters=args.sa_iters,
            betas=get_omeco_betas(args),
            score=score,
        )
        return tc.cons.OMEOptimizer(optimizer)

    if strategy == "omeco_greedy":
        return tc.cons.OMEOptimizer(omeco.GreedyMethod())

    raise ValueError(f"Unknown strategy: {strategy}")


class SearchRecorder:
    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.metrics: dict[str, Any] = {}
        self.path: list[tuple[int, int]] | None = None

    def __call__(
        self,
        inputs: Any,
        output: Any,
        size_dict: Any,
        memory_limit: Any = None,
        **kws: Any,
    ) -> list[tuple[int, int]]:
        started = time.perf_counter()
        path = self.optimizer(
            inputs, output, size_dict, memory_limit=memory_limit, **kws
        )
        search_time_s = time.perf_counter() - started
        path = [tuple(pair) for pair in path]
        tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
        flops = float(tree.total_flops())
        size = float(tree.max_size())
        write = float(tree.total_write())
        self.path = path
        self.metrics = {
            "search_time_s": search_time_s,
            "flops": flops,
            "size": size,
            "write": write,
            "log10_flops": math.log10(flops),
            "log2_size": math.log2(size),
            "log2_write": math.log2(write),
            "path_length": len(path),
        }
        return path

    def close(self) -> None:
        if hasattr(self.optimizer, "close"):
            self.optimizer.close()


def run_search(
    strategy: str, evaluate: Any, args: argparse.Namespace
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    recorder = SearchRecorder(build_optimizer(strategy, args))
    try:
        with tc.runtime_contractor(
            "custom",
            optimizer=recorder,
            preprocessing=True,
            debug_level=2,
        ):
            evaluate()
    finally:
        recorder.close()
    if recorder.path is None:
        raise RuntimeError("path search did not run")
    return recorder.metrics, recorder.path


def quality_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["log10_flops"]),
        float(row["log2_size"]),
        float(row["log2_write"]),
        float(row["search_time_s"]),
    )


def print_quality_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        print("No successful rows.")
        return []

    best_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = {}
    for row in ok_rows:
        key = (row["workload"], row["n"], row["layers"], row["strategy"])
        grouped.setdefault(key, []).append(row)

    for key in sorted(grouped):
        best_rows.append(min(grouped[key], key=quality_key))

    print("\nBest quality row for each strategy:")
    print("=" * 150)
    print(
        f"{'workload':<22} {'case':<8} {'strategy':<22} {'search_s':>10} "
        f"{'log10_flops':>12} {'log2_size':>11} {'log2_write':>12}"
    )
    print("-" * 150)
    for row in best_rows:
        case = f"{row['n']}x{row['layers']}"
        print(
            f"{row['workload']:<22} {case:<8} "
            f"{row['strategy']:<22} {row['search_time_s']:>10.6f} "
            f"{row['log10_flops']:>12.3f} {row['log2_size']:>11.3f} "
            f"{row['log2_write']:>12.3f}"
        )
    print("=" * 150)
    return best_rows


def main() -> None:
    args = parse_args()
    cases = [parse_case(text) for text in args.cases]
    print_environment(args)

    rows: list[dict[str, Any]] = []

    for n, layers in cases:
        for workload in args.workloads:
            print(f"\nRunning search for {workload} {n}x{layers}")
            evaluate = workload_evaluator(workload, n, layers)
            for strategy in args.strategies:
                for repeat in range(args.repeats):
                    row = {
                        "workload": workload,
                        "n": n,
                        "layers": layers,
                        "strategy": strategy,
                        "repeat": repeat,
                        "search_time_s": None,
                        "flops": None,
                        "size": None,
                        "write": None,
                        "log10_flops": None,
                        "log2_size": None,
                        "log2_write": None,
                        "path_length": None,
                        "status": "error",
                        "error": "",
                    }
                    try:
                        metrics, _ = run_search(strategy, evaluate, args)
                        row.update(metrics)
                        row["status"] = "ok"
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)

    print_quality_summary(rows)


if __name__ == "__main__":
    main()
