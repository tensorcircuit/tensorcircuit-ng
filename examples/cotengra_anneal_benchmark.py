"""
Benchmark JAX staging and running time for two cotengra combo strategies.

This example compares:

- the common TensorCircuit combo ReusableHyperOptimizer pattern
- an annealed combo ReusableHyperOptimizer tuned from the path-search study

The timed workload is amplitude evaluation of the 1D ``ladder`` and
``brickwork`` circuit families used in the path-search study. Here one
brickwork layer means a full even-plus-odd nearest-neighbor round.
"""

import argparse
import math
import time


import cotengra as ctg
import jax
import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

DEFAULT_SEED = 11


def parse_parallel(value):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "auto":
        return "auto"
    raise ValueError("parallel must be one of: true, false, auto")


def build_bonds(family, n, layers):
    bonds = []
    for _ in range(layers):
        if family == "ladder":
            bonds.append([(i, i + 1) for i in range(n - 1)])
        elif family == "brickwork":
            bonds.append([(i, i + 1) for i in range(0, n - 1, 2)])
            bonds.append([(i, i + 1) for i in range(1, n - 1, 2)])
        else:
            raise ValueError(f"Unsupported family: {family}")
    return bonds


def make_default_combo_optimizer(parallel):
    return ctg.ReusableHyperOptimizer(
        methods=["greedy", "kahypar"],
        parallel=parallel,
        minimize="combo",
        max_time=30,
        max_repeats=1024,
        progbar=False,
    )


def make_anneal_combo_optimizer(parallel):
    return ctg.ReusableHyperOptimizer(
        methods=["greedy", "kahypar"],
        parallel=parallel,
        minimize="combo",
        max_time=15,
        max_repeats=1024,
        progbar=False,
        seed=DEFAULT_SEED,
        simulated_annealing_opts={
            "tstart": 2.0,
            "tfinal": 0.05,
            "tsteps": 64,
            "numiter": 64,
            "minimize": "combo",
            "seed": DEFAULT_SEED,
            "progbar": False,
        },
    )


def make_amplitude_function(n, layers, family, bitstring):
    bonds = build_bonds(family, n, layers)

    def amplitude_fn(params):
        circuit = tc.Circuit(n)
        circuit.h(range(n))
        index = 0
        for layer_bonds in bonds:
            for i, j in layer_bonds:
                circuit.rzz(i, j, theta=params[index])
                index += 1
        return jnp.real(circuit.amplitude(bitstring))

    nparams = sum(len(layer_bonds) for layer_bonds in bonds)
    return amplitude_fn, nparams


def search_contraction_tree(optimizer_factory, family, n, layers, parallel):
    optimizer = optimizer_factory(parallel)
    try:
        bitstring = "0" * n
        _, nparams = make_amplitude_function(n, layers, family, bitstring)
        params = jnp.zeros((nparams,), dtype=jnp.float32)
        circuit = tc.Circuit(n)
        circuit.h(range(n))
        bonds = build_bonds(family, n, layers)
        index = 0
        for layer_bonds in bonds:
            for i, j in layer_bonds:
                circuit.rzz(i, j, theta=params[index])
                index += 1
        nodes = circuit.amplitude_before(bitstring)
        merged_nodes, _ = tc.cons._merge_single_gates(nodes)
        info, _ = tc.cons.get_tn_info(merged_nodes)
        return optimizer.search(*info)
    finally:
        if hasattr(optimizer, "close"):
            optimizer.close()


def contraction_costs(tree):
    flops = float(tree.total_flops())
    write = float(tree.total_write())
    return {
        "flops": flops,
        "log10_flops": math.log10(flops),
        "write": write,
        "log2_write": math.log2(write),
        "width": int(tree.contraction_width()),
    }


def time_compiled_function(compiled_fn, params, later_runs):
    started = time.perf_counter()
    first_value = jax.block_until_ready(compiled_fn(params))
    first_run_time = time.perf_counter() - started

    started = time.perf_counter()
    last_value = first_value
    for _ in range(later_runs):
        last_value = jax.block_until_ready(compiled_fn(params))
    later_avg_time = (time.perf_counter() - started) / later_runs

    return {
        "first_value": float(first_value),
        "last_value": float(last_value),
        "first_run_time_s": first_run_time,
        "later_avg_time_s": later_avg_time,
    }


def benchmark_strategy(
    strategy_name, optimizer_factory, family, n, layers, later_runs, parallel
):
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()

    tree = search_contraction_tree(optimizer_factory, family, n, layers, parallel)
    costs = contraction_costs(tree)

    def cached_path(inputs, output, size_dict, **kwargs):
        return tree.get_path()

    tc.set_contractor("custom", optimizer=cached_path, preprocessing=True)

    bitstring = "0" * n
    amplitude_fn, nparams = make_amplitude_function(n, layers, family, bitstring)
    compiled_fn = jax.jit(amplitude_fn)
    params = jax.random.normal(jax.random.PRNGKey(DEFAULT_SEED), (nparams,))

    timing = time_compiled_function(compiled_fn, params, later_runs)

    return {
        "family": family,
        "strategy": strategy_name,
        "n": n,
        "layers": layers,
        "nparams": nparams,
        **costs,
        **timing,
    }


def print_results(rows):
    print("=" * 154)
    print(
        f"{'family':<10} {'strategy':<20} {'params':>8} "
        f"{'log10_flops':>12} {'log2_write':>12} {'width':>7} "
        f"{'first_run_s':>14} {'later_avg_s':>14} {'speedup':>10} {'value':>18}"
    )
    print("-" * 154)
    for row in rows:
        speedup = row["first_run_time_s"] / row["later_avg_time_s"]
        print(
            f"{row['family']:<10} {row['strategy']:<20} {row['nparams']:>8d} "
            f"{row['log10_flops']:>12.3f} {row['log2_write']:>12.3f} {row['width']:>7d} "
            f"{row['first_run_time_s']:>14.6f} {row['later_avg_time_s']:>14.6f} "
            f"{speedup:>10.3f} {row['last_value']:>18.10f}"
        )
    print("=" * 154)


def print_comparison(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["family"], {})[row["strategy"]] = row

    print("\nPer-family comparison:")
    for family, family_rows in grouped.items():
        default_row = family_rows["tc_combo_default"]
        anneal_row = family_rows["anneal_combo"]
        first_ratio = anneal_row["first_run_time_s"] / default_row["first_run_time_s"]
        later_ratio = anneal_row["later_avg_time_s"] / default_row["later_avg_time_s"]
        print(
            f"- {family}: anneal/default first-run={first_ratio:.3f}, "
            f"later-run={later_ratio:.3f}, "
            f"log10_flops={anneal_row['log10_flops']:.3f}/{default_row['log10_flops']:.3f}, "
            f"log2_write={anneal_row['log2_write']:.3f}/{default_row['log2_write']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--later-runs", type=int, default=5)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["ladder", "brickwork"],
        choices=["ladder", "brickwork"],
    )
    parser.add_argument(
        "--parallel",
        default="auto",
        help="cotengra parallel mode: auto, true, or false",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parallel = parse_parallel(args.parallel)
    print(
        f"Backend={tc.backend.name}, n={args.n}, layers={args.layers}, "
        f"later_runs={args.later_runs}, parallel={parallel}"
    )

    rows = []
    for family in args.families:
        print(f"\nBenchmarking family={family}")
        rows.append(
            benchmark_strategy(
                "tc_combo_default",
                make_default_combo_optimizer,
                family,
                args.n,
                args.layers,
                args.later_runs,
                parallel,
            )
        )
        rows.append(
            benchmark_strategy(
                "anneal_combo",
                make_anneal_combo_optimizer,
                family,
                args.n,
                args.layers,
                args.later_runs,
                parallel,
            )
        )

    print_results(rows)
    print_comparison(rows)


if __name__ == "__main__":
    main()
