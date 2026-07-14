---
name: contraction-tuner
description: Tune tensor-network contraction path search and slicing for TensorCircuit-NG workloads, especially OMECO and cotengra hyperparameters, memory targets, total FLOPs/write, slice counts, and large-circuit amplitude or expectation contractions.
allowed-tools: Bash, Read, Grep, Glob, Write
---

# Contraction Tuner

Use this skill when a task asks to optimize, compare, or benchmark tensor-network contraction paths, slicing targets, OMECO, cotengra, contraction width, FLOPs, write, or slice count for TensorCircuit-NG.

## Core Principle

Treat contraction tuning as an empirical search problem. Do not infer quality from width alone. Always report at least:

- path/search time and slicer/reconfiguration time separately
- `log10(total FLOPs)`
- `log2(max intermediate size)`
- `log2(total write)`
- number of sliced indices and total slice tasks
- package versions and topology size (`ntensors`, `nindices`)

For dim-2 circuit indices, `k` sliced indices means `2**k` slice tasks. A path with more slices can still have lower total FLOPs if per-slice contractions are much cheaper.

## Metric Hygiene

- cotengra `target_size` is a linear tensor element count. For target width `W`, pass `target_size=2**W`, not `W`.
- cotengra `ContractionTree.total_flops()` and `total_write()` include slicing multiplicity.
- OMECO `SlicedEinsum.complexity()` is per-slice. For fair comparison, convert the OMECO sliced tree to a cotengra `ContractionTree` and apply `remove_ind_()` for each `sliced.slicing()` label, or multiply per-slice FLOPs/write by `prod(size_dict[ix] for ix in sliced.slicing())`.
- OMECO `SlicedEinsum.num_slices()` is the number of sliced labels, not total slice tasks.
- A result where `sliced.slicing()` is empty but width improves means OMECO slicer reoptimized the tree enough to meet the target without actual slicing.

## OMECO Workflow

Use OMECO as the default first pass for large/deep tensor networks because it often finds strong seed trees much faster than cotengra.

Recommended seed search grid:

```python
seed_configs = ["8x48", "12x64", "24x64"]
score = omeco.ScoreFunction(
    tc_weight=1.0,
    sc_weight=0.0,
    rw_weight=64.0,
    sc_target=64.0,
)
```

Parse `AxB` as `ntrials=A`, `niters=B`, and use `len(betas)=B`, for example:

```python
betas = [float(1.0 / x) for x in np.geomspace(2.0, 0.05, niters)]
tree = omeco.optimize_code(
    inputs,
    output,
    sizes,
    omeco.TreeSA(ntrials=ntrials, niters=niters, betas=betas, score=score),
)
```

## OMECO Slicer Workflow

OMECO `TreeSASlicer` is not a pure fixed-tree post-slicer. It greedily adds near-peak labels as sliced indices and runs TreeSA again with sliced labels assigned log2 size 0. This makes it conservative and stable on large networks, but its slice-label choice is still greedy.

Default starting point for large networks:

```python
slicer_configs = ["2x24"]
score = omeco.ScoreFunction(
    tc_weight=1.0,
    sc_weight=64.0,
    rw_weight=64.0,
    sc_target=target_width,
)
```

If the first result has too many slices or high write, try:

```python
slicer_configs = ["2x32", "2x48"]
score_grid = [
    {"sc_weight": 64.0, "rw_weight": 128.0},
    {"sc_weight": 32.0, "rw_weight": 64.0},
    {"sc_weight": 128.0, "rw_weight": 128.0},
]
```

For 100q x 28-layer 1D amplitude with target width 30, a tuned starting point was:

```text
TreeSA seed: 24x64
TreeSASlicer: 2x48
sc_weight=64
rw_weight=128
```

This reduced the observed result from hundreds of slices to a two-slice plan in that benchmark. Treat this as a strong candidate, not a universal optimum.

## cotengra Workflow

Use cotengra for small/medium networks or when integrated tree+slicing search can be afforded.

Fixed post-slicing is a fast diagnostic:

```python
post = tree.slice(
    target_size=int(2**target_width),
    minimize="combo",
    max_repeats=4,
    allow_outer=False,
    reslice=True,
    inplace=False,
)
```

Integrated slicing-aware SA:

```python
resliced = tree.simulated_anneal(
    tstart=2.0,
    tfinal=0.05,
    tsteps=24,
    numiter=24,
    minimize="combo",
    target_size=int(2**target_width),
    slice_mode="reslice",
    inplace=False,
)
```

Sweep `slice_mode` only when needed:

- `reslice`: best quality on many small/medium networks, but can over-slice badly on large networks.
- `drift`: often faster and less rigid; useful if `reslice` over-slices.
- `basic`: can overslice and should mainly be a baseline.

Abort cotengra integrated reslice early if sliced indices jump to dozens on a large network; this is usually a bad basin rather than a useful result.

## Search Protocol

1. Build or extract topology once: `input_sets`, `output_set`, `size_dict`.
2. Print topology size and package versions.
3. Run OMECO seed grid first. Pick 1-2 seeds by `combo_log2`, not width alone.
4. If `seed_width <= target`, record that no slicing is required; do not automatically run cotengra reslice.
5. Run OMECO slicer starting with `2x24`; if quality or slice count is poor, try `2x32`, `2x48`, and a small `sc_weight/rw_weight` grid.
6. For small/medium networks, compare cotengra fixed post-slice and integrated `reslice`.
7. Repeat promising configurations several times. OMECO and cotengra searches can have large stochastic variation even with identical hyperparameters, and a repeated run can find a much better slice set. Report best and median metrics rather than a single run when the budget permits.
8. If the user's goal is to actually run the contraction later, persist the best path or sliced tree immediately. Do not rely on rerunning stochastic search to reproduce it.
9. Rank by `(target violation, combo_log2, log10_flops, log2_write, nslices, time)`.
10. Preserve the exact topology, seed config, slicer config, target, repeat count, and version info in the report.

## Rules of Thumb

- OMECO seed plus OMECO slicer is a robust default for large/deep 1D amplitude networks.
- cotengra integrated `reslice` often wins on small/medium 2D grid-like networks where it does not over-slice.
- More slicer iterations are not monotonic. `2x32` can beat `2x24`, but can also be worse. Always compare the actual metrics.
- Repeating the same configuration is itself a useful tuning axis. If a single configuration sometimes returns 2 slices and sometimes 16 slices, run it multiple times and keep the best valid result; also report the variation so the user knows the robustness.
- When a good stochastic result is found, save the actual path/tree and sliced labels, not just the hyperparameters. Hyperparameters describe a distribution, while the saved path is the reproducible artifact.
- If target is only 1-3 width bits below a good seed, slicing is usually tractable.
- If a target needs 10+ dim-2 sliced indices, execution may still be valid but is entering deep slicing; consider stronger seed search, a less aggressive target, or parallel slice execution.
- If a run produces thousands to millions of slices, report it honestly. It may be necessary under a hard memory target, but it should not be called a lightweight best-practice path.

## Implementation Pattern

When writing benchmark scripts, include helpers to:

- normalize labels for OMECO while keeping an inverse label map
- convert OMECO trees to cotengra trees via TensorCircuit's `_omeco_tree_to_path`
- apply OMECO sliced labels to cotengra trees with `remove_ind_`
- set `sys.setrecursionlimit(20000)` before converting very deep OMECO trees
- print one JSON-like row per result for easy comparison
- support a `--repeats` or equivalent option and include `repeat` in each result row
- support a `--save-best` or equivalent option that writes the best tree/path, sliced labels, metrics, target, topology metadata, and package versions to disk
