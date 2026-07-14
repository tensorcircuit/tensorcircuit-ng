"""
Best-practice OMECO slicing workflow for a large 1D amplitude contraction.

This example builds a 100-qubit, 28-layer nearest-neighbor RZZ circuit and
searches a sliced contraction using a two-stage OMECO workflow:

1. Find a strong unsliced seed tree with ``omeco.TreeSA``.
2. Refine and slice it with ``omeco.TreeSASlicer``.

The default target is ``log2(max_intermediate_size) <= 30``. This target
requires slicing for the default seed budget on this topology. Metrics are
reported through a cotengra ``ContractionTree`` so total FLOPs/write include
the multiplicity of all slices.
"""

import importlib.metadata as md
import math
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import cotengra as ctg
import numpy as np
import omeco

import tensorcircuit as tc

tc.set_backend("numpy")
tc.set_dtype("complex64")

COMBO_FACTOR = 64.0
NQUBITS = 100
LAYERS = 28
TARGET_LOG2_SIZE = 30.0
TREE_CONFIG = "24x64"
SLICER_CONFIG = "2x48"
SLICER_SC_WEIGHT = 64.0
SLICER_RW_WEIGHT = 128.0


def parse_config(text: str) -> Tuple[int, int]:
    trials, iters = text.split("x", maxsplit=1)
    return int(trials), int(iters)


def betas(iters: int) -> List[float]:
    return [float(1.0 / x) for x in np.geomspace(2.0, 0.05, iters)]


def build_topology(
    nqubits: int, layers: int
) -> Tuple[Sequence[Sequence[Any]], Sequence[Any], Dict[Any, int]]:
    circuit = tc.Circuit(nqubits)
    circuit.h(range(nqubits))
    params = np.linspace(0.05, 0.95, layers * (nqubits - 1), dtype=np.float32)
    index = 0
    for _ in range(layers):
        for i in range(nqubits - 1):
            circuit.rzz(i, i + 1, theta=float(params[index]))
            index += 1
    nodes = circuit.amplitude_before("0" * nqubits)
    merged_nodes, _ = tc.cons._merge_single_gates(nodes)
    return tc.cons.get_tn_info(merged_nodes)[0]


def normalize_omeco_topology(
    input_sets: Sequence[Sequence[Any]],
    output_set: Sequence[Any],
    size_dict: Dict[Any, int],
) -> Tuple[List[List[int]], List[int], Dict[int, int], Dict[int, Any]]:
    mapping: Dict[Any, int] = {}

    def relabel(symbol: Any) -> int:
        if symbol not in mapping:
            mapping[symbol] = len(mapping)
        return mapping[symbol]

    new_inputs = [[relabel(symbol) for symbol in term] for term in input_sets]
    new_output = [relabel(symbol) for symbol in output_set]
    new_sizes = {relabel(symbol): int(size) for symbol, size in size_dict.items()}
    inverse = {value: key for key, value in mapping.items()}
    return new_inputs, new_output, new_sizes, inverse


def combo(flops: float, write: float, factor: float) -> float:
    return float(flops) + factor * float(write)


def path_from_omeco(code: Any, ninputs: int) -> List[Tuple[int, int]]:
    return tc.cons._omeco_tree_to_path(code.to_dict(), ninputs)


def tree_metrics(tree: ctg.ContractionTree, combo_factor: float) -> Dict[str, Any]:
    flops = float(tree.total_flops())
    write = float(tree.total_write())
    return {
        "log10_flops": round(math.log10(flops), 6),
        "log2_size": round(math.log2(float(tree.max_size())), 6),
        "log2_write": round(math.log2(write), 6),
        "nslices": int(tree.nslices),
        "nsliced_indices": len(tree.sliced_inds),
        "combo_log2": round(math.log2(combo(flops, write, combo_factor)), 6),
    }


def omeco_code_to_cotengra_tree(
    code: Any,
    input_sets: Sequence[Sequence[Any]],
    output_set: Sequence[Any],
    size_dict: Dict[Any, int],
) -> ctg.ContractionTree:
    return ctg.ContractionTree.from_path(
        input_sets,
        output_set,
        size_dict,
        path=path_from_omeco(code, len(input_sets)),
    )


def sliced_code_to_cotengra_tree(
    sliced: Any,
    input_sets: Sequence[Sequence[Any]],
    output_set: Sequence[Any],
    size_dict: Dict[Any, int],
    inverse_labels: Dict[int, Any],
) -> ctg.ContractionTree:
    tree = omeco_code_to_cotengra_tree(sliced.tree(), input_sets, output_set, size_dict)
    for ix in sliced.slicing():
        tree.remove_ind_(inverse_labels[int(ix)])
    return tree


def search_seed_tree(
    omeco_inputs: Sequence[Sequence[int]],
    omeco_output: Sequence[int],
    omeco_sizes: Dict[int, int],
    tree_config: str,
    combo_factor: float,
) -> Tuple[Any, float]:
    ntrials, niters = parse_config(tree_config)
    start = time.perf_counter()
    tree = omeco.optimize_code(
        omeco_inputs,
        omeco_output,
        omeco_sizes,
        omeco.TreeSA(
            ntrials=ntrials,
            niters=niters,
            betas=betas(niters),
            score=omeco.ScoreFunction(
                tc_weight=1.0,
                sc_weight=0.0,
                rw_weight=combo_factor,
                sc_target=64.0,
            ),
        ),
    )
    return tree, time.perf_counter() - start


def slice_seed_tree(
    seed_tree: Any,
    omeco_inputs: Sequence[Sequence[int]],
    omeco_sizes: Dict[int, int],
    target_log2_size: float,
    slicer_config: str,
    sc_weight: float,
    rw_weight: float,
) -> Tuple[Any, float]:
    ntrials, niters = parse_config(slicer_config)
    start = time.perf_counter()
    sliced = omeco.slice_code(
        seed_tree,
        omeco_inputs,
        omeco_sizes,
        omeco.TreeSASlicer(
            ntrials=ntrials,
            niters=niters,
            betas=betas(niters),
            optimization_ratio=2.0,
            score=omeco.ScoreFunction(
                tc_weight=1.0,
                sc_weight=sc_weight,
                rw_weight=rw_weight,
                sc_target=target_log2_size,
            ),
        ),
    )
    return sliced, time.perf_counter() - start


def print_row(name: str, row: Dict[str, Any]) -> None:
    print(name, row, flush=True)


def main() -> None:
    sys.setrecursionlimit(20000)

    start = time.perf_counter()
    input_sets, output_set, size_dict = build_topology(NQUBITS, LAYERS)
    omeco_inputs, omeco_output, omeco_sizes, inverse_labels = normalize_omeco_topology(
        input_sets, output_set, size_dict
    )
    print_row(
        "environment",
        {
            "omeco": md.version("omeco"),
            "cotengra": md.version("cotengra"),
            "tensorcircuit": tc.__version__,
            "nqubits": NQUBITS,
            "layers": LAYERS,
            "ntensors": len(input_sets),
            "nindices": len(size_dict),
            "target_log2_size": TARGET_LOG2_SIZE,
            "tree_config": TREE_CONFIG,
            "slicer_config": SLICER_CONFIG,
            "slicer_sc_weight": SLICER_SC_WEIGHT,
            "slicer_rw_weight": SLICER_RW_WEIGHT,
            "build_s": round(time.perf_counter() - start, 6),
        },
    )

    seed_tree, seed_s = search_seed_tree(
        omeco_inputs,
        omeco_output,
        omeco_sizes,
        TREE_CONFIG,
        COMBO_FACTOR,
    )
    seed_ctg_tree = omeco_code_to_cotengra_tree(
        seed_tree, input_sets, output_set, size_dict
    )
    print_row(
        "omeco_seed",
        {
            "seed_s": round(seed_s, 6),
            **tree_metrics(seed_ctg_tree, COMBO_FACTOR),
        },
    )

    sliced, slice_s = slice_seed_tree(
        seed_tree,
        omeco_inputs,
        omeco_sizes,
        TARGET_LOG2_SIZE,
        SLICER_CONFIG,
        SLICER_SC_WEIGHT,
        SLICER_RW_WEIGHT,
    )
    sliced_ctg_tree = sliced_code_to_cotengra_tree(
        sliced, input_sets, output_set, size_dict, inverse_labels
    )
    print_row(
        "omeco_sliced",
        {
            "target_log2_size": TARGET_LOG2_SIZE,
            "slicer_config": SLICER_CONFIG,
            "slice_s": round(slice_s, 6),
            "total_s": round(seed_s + slice_s, 6),
            "sliced_indices": tuple(int(ix) for ix in sliced.slicing()),
            **tree_metrics(sliced_ctg_tree, COMBO_FACTOR),
        },
    )


if __name__ == "__main__":
    main()
