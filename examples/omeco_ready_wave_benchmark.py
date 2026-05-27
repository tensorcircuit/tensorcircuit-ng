"""
Benchmark explicit ready-wave batching on one fixed OMECO contraction tree.

The script builds a circuit expectation tensor network with ``reuse=False``,
searches a single contraction tree with ``omeco.TreeSA``, and then replays the
same path in two ways:

- plain sequential pairwise contractions
- ready-wave batching for same-shape contractions

The main metric is ``jax.jit(jax.value_and_grad(...))`` runtime, since that is
where the compile/runtime tradeoff is easiest to see.

Example:

    python examples/omeco_ready_wave_benchmark.py \\
        --n 24 --layers 12 --m-values 2 4 8 16 --remat
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import time
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import cotengra as ctg
import jax
import jax.numpy as jnp
import omeco
import tensornetwork as tn
import tensorcircuit as tc

Signature = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


class ContractStep:
    def __init__(
        self,
        node_id: int,
        left_id: int,
        right_id: int,
        axes_left: Tuple[int, ...],
        axes_right: Tuple[int, ...],
        left_perm: Tuple[int, ...],
        right_perm: Tuple[int, ...],
        contract_ndim: int,
        signature: Signature,
    ):
        self.node_id = node_id
        self.left_id = left_id
        self.right_id = right_id
        self.axes_left = axes_left
        self.axes_right = axes_right
        self.left_perm = left_perm
        self.right_perm = right_perm
        self.contract_ndim = contract_ndim
        self.signature = signature


class ContractGroup:
    def __init__(self, signature: Signature, steps: Tuple[ContractStep, ...]):
        self.signature = signature
        self.steps = steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--m-values", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--max-batched-waves", type=int, default=None)
    parser.add_argument("--grad-iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--omeco-trials", type=int, default=32)
    parser.add_argument("--sa-iters", type=int, default=32)
    parser.add_argument("--sa-tstart", type=float, default=2.0)
    parser.add_argument("--sa-tfinal", type=float, default=0.05)
    parser.add_argument("--omeco-tc-weight", type=float, default=1.0)
    parser.add_argument("--omeco-sc-weight", type=float, default=1.0)
    parser.add_argument("--omeco-rw-weight", type=float, default=64.0)
    parser.add_argument("--omeco-sc-target", type=float, default=24.0)
    parser.add_argument("--remat", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in args.m_values):
        parser.error("--m-values must be >= 1.")
    if args.max_batched_waves is not None and args.max_batched_waves < 0:
        parser.error("--max-batched-waves must be >= 0.")
    return args


def build_two_qubit_bonds(n: int, layers: int) -> List[List[Tuple[int, int]]]:
    bonds = []
    for _ in range(layers):
        bonds.append([(i, i + 1) for i in range(n - 1)])
    return bonds


def build_circuit_nodes(n: int, layers: int, params: jax.Array) -> List[Any]:
    circuit = tc.Circuit(n)
    circuit.h(range(n))
    index = 0
    for layer_bonds in build_two_qubit_bonds(n, layers):
        for i, j in layer_bonds:
            circuit.rzz(i, j, theta=params[index])
            index += 1
    return circuit.expectation_before((tc.gates.z(), [n // 2]), reuse=False)


def get_omeco_betas(args: argparse.Namespace) -> List[float]:
    temperatures = jnp.geomspace(args.sa_tstart, args.sa_tfinal, args.sa_iters)
    return [float(1.0 / value) for value in temperatures]


def search_tree(
    input_sets: Sequence[Sequence[Any]],
    output_set: Sequence[Any],
    size_dict: Dict[Any, int],
    args: argparse.Namespace,
) -> Tuple[Any, float]:
    score = omeco.ScoreFunction(
        tc_weight=args.omeco_tc_weight,
        sc_weight=args.omeco_sc_weight,
        rw_weight=args.omeco_rw_weight,
        sc_target=args.omeco_sc_target,
    )
    optimizer = tc.cons.OMEOptimizer(
        omeco.TreeSA(
            ntrials=args.omeco_trials,
            niters=args.sa_iters,
            betas=get_omeco_betas(args),
            score=score,
        )
    )
    start = time.perf_counter()
    path = optimizer(input_sets, output_set, size_dict)
    tree = ctg.ContractionTree.from_path(input_sets, output_set, size_dict, path=path)
    return tree, time.perf_counter() - start


def build_ready_wave_groups(
    steps: Sequence[ContractStep], nleaves: int
) -> Tuple[Tuple[int, Tuple[ContractGroup, ...]], ...]:
    produced_ids = {step.node_id for step in steps}
    step_by_id = {step.node_id: step for step in steps}
    consumers: DefaultDict[int, List[int]] = defaultdict(list)
    pending_inputs: Dict[int, int] = {}

    for step in steps:
        pending = 0
        if step.left_id in produced_ids:
            pending += 1
            consumers[step.left_id].append(step.node_id)
        elif step.left_id >= nleaves:
            raise ValueError(f"Missing producer for node {step.left_id}")
        if step.right_id in produced_ids:
            pending += 1
            consumers[step.right_id].append(step.node_id)
        elif step.right_id >= nleaves:
            raise ValueError(f"Missing producer for node {step.right_id}")
        pending_inputs[step.node_id] = pending

    ready_ids = sorted(
        [node_id for node_id, pending in pending_inputs.items() if pending == 0]
    )
    waves = []
    wave_index = 0
    emitted = 0

    while ready_ids:
        grouped: DefaultDict[Signature, List[ContractStep]] = defaultdict(list)
        for node_id in ready_ids:
            grouped[step_by_id[node_id].signature].append(step_by_id[node_id])

        wave_groups = []
        for signature, grouped_steps in grouped.items():
            grouped_steps.sort(key=lambda step: step.node_id)
            wave_groups.append(ContractGroup(signature, tuple(grouped_steps)))
        wave_groups.sort(key=lambda group: (len(group.steps), group.signature))
        waves.append((wave_index, tuple(wave_groups)))

        next_ready = []
        for node_id in ready_ids:
            emitted += 1
            for consumer_id in consumers.get(node_id, []):
                pending_inputs[consumer_id] -= 1
                if pending_inputs[consumer_id] == 0:
                    next_ready.append(consumer_id)
        ready_ids = sorted(next_ready)
        wave_index += 1

    if emitted != len(steps):
        raise ValueError("Ready-wave schedule did not emit all contraction steps.")

    return tuple(waves)


def build_plan(
    leaves: Sequence[jax.Array],
    input_sets: Sequence[Sequence[Any]],
    path: Sequence[Tuple[int, int]],
) -> Tuple[
    Tuple[ContractStep, ...],
    Tuple[Tuple[int, Tuple[ContractGroup, ...]], ...],
    int,
    int,
]:
    queue = list(range(len(leaves)))
    node_info: Dict[int, Tuple[Tuple[Any, ...], Tuple[int, ...]]] = {
        i: (tuple(input_sets[i]), tuple(leaves[i].shape)) for i in range(len(leaves))
    }
    steps = []
    next_node_id = len(leaves)

    for pair in path:
        left_id = queue[pair[0]]
        right_id = queue[pair[1]]
        left_labels, left_shape = node_info[left_id]
        right_labels, right_shape = node_info[right_id]
        right_label_set = set(right_labels)
        left_label_set = set(left_labels)
        common_labels = tuple(
            label for label in left_labels if label in right_label_set
        )
        axes_left = tuple(left_labels.index(label) for label in common_labels)
        axes_right = tuple(right_labels.index(label) for label in common_labels)
        left_keep_axes = tuple(
            i for i, label in enumerate(left_labels) if label not in right_label_set
        )
        right_keep_axes = tuple(
            i for i, label in enumerate(right_labels) if label not in left_label_set
        )
        left_keep = tuple(left_shape[i] for i in left_keep_axes)
        right_keep = tuple(right_shape[i] for i in right_keep_axes)
        contract_shape = tuple(left_shape[i] for i in axes_left)
        steps.append(
            ContractStep(
                node_id=next_node_id,
                left_id=left_id,
                right_id=right_id,
                axes_left=axes_left,
                axes_right=axes_right,
                left_perm=left_keep_axes + axes_left,
                right_perm=axes_right + right_keep_axes,
                contract_ndim=len(axes_left),
                signature=(left_keep, contract_shape, right_keep),
            )
        )
        output_labels = tuple(
            [label for label in left_labels if label not in right_label_set]
            + [label for label in right_labels if label not in left_label_set]
        )
        node_info[next_node_id] = (output_labels, left_keep + right_keep)
        for index in sorted(pair, reverse=True):
            queue.pop(index)
        queue.append(next_node_id)
        next_node_id += 1

    if len(queue) != 1:
        raise ValueError("Contraction path did not reduce to one output.")

    steps = tuple(steps)
    return steps, build_ready_wave_groups(steps, len(leaves)), queue[0], next_node_id


def make_problem(args: argparse.Namespace) -> Dict[str, Any]:
    bonds = build_two_qubit_bonds(args.n, args.layers)
    nparams = sum(len(layer_bonds) for layer_bonds in bonds)
    params = jax.random.normal(
        jax.random.PRNGKey(args.seed), (nparams,), dtype=jnp.float32
    )
    nodes = build_circuit_nodes(args.n, args.layers, params)
    merged_nodes, _ = tc.cons._merge_single_gates(nodes)
    info, ordered_nodes = tc.cons.get_tn_info(merged_nodes)
    if any(isinstance(node, tn.CopyNode) for node in ordered_nodes):
        raise ValueError(
            "This example assumes a plain tensor network without CopyNode."
        )
    input_sets, output_set, size_dict = info
    leaves = tuple(node.tensor for node in ordered_nodes)
    tree, path_search_time_s = search_tree(input_sets, output_set, size_dict, args)
    steps, ready_wave_groups, final_id, total_nodes = build_plan(
        leaves, input_sets, tree.get_path()
    )
    return {
        "n": args.n,
        "layers": args.layers,
        "leaves": leaves,
        "tree": tree,
        "steps": steps,
        "ready_wave_groups": ready_wave_groups,
        "final_id": final_id,
        "total_nodes": total_nodes,
        "path_search_time_s": path_search_time_s,
    }


def contract_pair(
    left: jax.Array,
    right: jax.Array,
    axes_left: Tuple[int, ...],
    axes_right: Tuple[int, ...],
) -> jax.Array:
    return jnp.tensordot(left, right, axes=(axes_left, axes_right))


def batched_contract(
    left: jax.Array, right: jax.Array, contract_ndim: int
) -> jax.Array:
    return jax.vmap(
        lambda left_operand, right_operand: contract_pair(
            left_operand,
            right_operand,
            tuple(range(left_operand.ndim - contract_ndim, left_operand.ndim)),
            tuple(range(contract_ndim)),
        )
    )(left, right)


def prepare_group_batches(
    values: Sequence[Any], group: ContractGroup
) -> Tuple[jax.Array, jax.Array]:
    left_batch = jnp.stack(
        [jnp.transpose(values[step.left_id], step.left_perm) for step in group.steps]
    )
    right_batch = jnp.stack(
        [jnp.transpose(values[step.right_id], step.right_perm) for step in group.steps]
    )
    return left_batch, right_batch


class ReplayExecutor:
    def __init__(
        self,
        problem: Dict[str, Any],
        min_group_size: Optional[int] = None,
        max_batched_waves: Optional[int] = None,
    ):
        self.problem = problem
        self.min_group_size = min_group_size
        self.max_batched_waves = max_batched_waves

    def __call__(self, leaves: Sequence[jax.Array]) -> jax.Array:
        values: List[Any] = list(leaves) + [None] * (
            self.problem["total_nodes"] - len(leaves)
        )
        if self.min_group_size is None:
            for step in self.problem["steps"]:
                values[step.node_id] = contract_pair(
                    values[step.left_id],
                    values[step.right_id],
                    step.axes_left,
                    step.axes_right,
                )
            return jnp.real(values[self.problem["final_id"]])
        for wave_index, groups in self.problem["ready_wave_groups"]:
            can_batch_this_wave = self.max_batched_waves is None or (
                wave_index < self.max_batched_waves
            )
            for group in groups:
                if (not can_batch_this_wave) or len(group.steps) < self.min_group_size:
                    for step in group.steps:
                        values[step.node_id] = contract_pair(
                            values[step.left_id],
                            values[step.right_id],
                            step.axes_left,
                            step.axes_right,
                        )
                    continue
                left_batch, right_batch = prepare_group_batches(values, group)
                outputs = batched_contract(
                    left_batch, right_batch, group.steps[0].contract_ndim
                )
                for index, step in enumerate(group.steps):
                    values[step.node_id] = outputs[index]
        return jnp.real(values[self.problem["final_id"]])


def block_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def tree_max_abs_diff(left: Any, right: Any) -> float:
    max_diff = 0.0
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    ):
        max_diff = max(max_diff, float(jnp.max(jnp.abs(left_leaf - right_leaf))))
    return max_diff


def benchmark(fun: Any, *args: Any, iters: int) -> Tuple[float, float, Any]:
    start = time.perf_counter()
    first_result = fun(*args)
    block_tree(first_result)
    after_first = time.perf_counter()
    for _ in range(iters):
        block_tree(fun(*args))
    after_loop = time.perf_counter()
    return (
        after_first - start,
        (after_loop - after_first) / max(1, iters),
        first_result,
    )


def print_problem(
    problem: Dict[str, Any], remat: bool, max_batched_waves: Optional[int]
) -> None:
    repeated = [
        group
        for _, groups in problem["ready_wave_groups"]
        for group in groups
        if len(group.steps) > 1
    ]
    repeated_levels = sum(
        1
        for _, groups in problem["ready_wave_groups"]
        if any(len(group.steps) > 1 for group in groups)
    )
    print("=== Problem ===")
    print("topology: ladder")
    print("n:", problem["n"])
    print("layers:", problem["layers"])
    print("remat:", remat)
    print(
        "max_batched_waves:",
        "all" if max_batched_waves is None else max_batched_waves,
    )
    print("path_search_time_s:", f"{problem['path_search_time_s']:.3f}")
    print("leaf_tensors:", len(problem["leaves"]))
    print("path_length:", len(problem["steps"]))
    print("tree_width:", int(problem["tree"].contraction_width()))
    print("log10_flops:", f"{jnp.log10(float(problem['tree'].total_flops())):.3f}")
    print("log2_write:", f"{jnp.log2(float(problem['tree'].total_write())):.3f}")
    print(
        "ready_wave_repeat_stats:",
        f"levels={repeated_levels},",
        f"groups={len(repeated)},",
        f"ops={sum(len(group.steps) for group in repeated)},",
        f"max_group={max([len(group.steps) for group in repeated] or [0])}",
    )
    print()


def print_timing_row(label: str, timing: Tuple[float, float, Any]) -> None:
    compile_and_first_run_s, running_time_s, _ = timing
    print(
        f"{label:<28} running time: {1000.0 * running_time_s:.3f} ms | "
        f"compile+run: {compile_and_first_run_s:.3f} s"
    )


def main() -> None:
    args = parse_args()
    tc.set_backend("jax")
    tc.set_dtype("complex64")

    problem = make_problem(args)
    print_problem(problem, args.remat, args.max_batched_waves)

    variants: List[Tuple[str, Any]] = [("sequential_vag", ReplayExecutor(problem))]
    for m in sorted(set(args.m_values)):
        variants.append(
            (
                f"ready_wave_m{m}_vag",
                ReplayExecutor(problem, m, args.max_batched_waves),
            )
        )

    results: Dict[str, Tuple[float, float, Any]] = {}
    for label, executor in variants:
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
        differentiated = jax.checkpoint(executor) if args.remat else executor
        vag_fun = jax.jit(jax.value_and_grad(differentiated))
        results[label] = benchmark(
            vag_fun,
            problem["leaves"],
            iters=args.grad_iters,
        )

    _, _, (base_value, base_grad) = results["sequential_vag"]
    for label, timing in results.items():
        if label == "sequential_vag":
            continue
        _, _, (value, grad) = timing
        value_diff = float(jnp.max(jnp.abs(base_value - value)))
        grad_diff = tree_max_abs_diff(base_grad, grad)
        if value_diff > 1e-5 or grad_diff > 1e-5:
            raise ValueError(
                f"{label} failed correctness check: value_diff={value_diff}, grad_diff={grad_diff}"
            )

    print("=== Value And Grad Timings ===")
    baseline = results["sequential_vag"]
    baseline_compile, baseline_running_time, _ = baseline
    for label, timing in results.items():
        print_timing_row(label, timing)
    for label, timing in results.items():
        if label == "sequential_vag":
            continue
        compile_and_first_run_s, running_time_s, _ = timing
        print(
            f"{label} speedup over sequential_vag:",
            f"{baseline_running_time / running_time_s:.3f}",
        )
        print(
            f"{label} compile slowdown vs sequential_vag:",
            f"{compile_and_first_run_s / baseline_compile:.3f}",
        )


if __name__ == "__main__":
    main()

# Empirically, ready-wave batching gives only marginal running-time gains on the
# tested ladder benchmarks once compile overhead is included in the picture.
# Without stronger repeated tensor structure or other special cases, this is not
# a robust default engineering optimization and should be treated as exploratory.
