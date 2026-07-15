# Contraction and Benchmarking

Use this file for algebraic contraction invariants, cotengra/OMECO path search and slicing, and interpretation of contraction runtime and memory.

## Algebraic contraction invariants

- Route networks with hyperedges through the algebraic contractor so optimization does not materialize diagonal copy tensors. In particular, do not run `_merge_single_gates`-style preprocessing while `CopyNode`s remain.
- Sort nodes deterministically before building topology strings or einsum expressions; stable ordering improves path reuse and compiled-cache hits.
- After algebraically contracting a subgraph, reattach its original dangling edges to the replacement node before continuing.
- `Circuit.matrix()` creates identity-input `CopyNode`s and therefore uses the algebraic hyperedge path even under the default global contractor; unless cotengra is explicitly selected, path search still uses the default opt-einsum greedy/optimal logic.
- Adapt third-party optimizers with integer-only labels at the path-finder boundary, reconstruct a standard pairwise path, and reuse the shared algebraic contractor. Normalize non-callable optional optimizer state once in `set_contractor("custom", optimizer=...)` and keep its import lazy.

## Search and path reuse

- For large contractions, start with `tc.set_contractor("cotengra")` and a reusable optimizer. Search once and use the exact resulting path or tree for both reported metrics and timed execution; make timed runs cache-only so they cannot silently search a different stochastic path.
- Keep preprocessing enabled for OMECO shortcut contractors unless measurement shows otherwise. Disabling it can make path search and tracing dominate large perfect-sampling workloads without improving path quality.
- Keep the contraction tree fixed when testing execution-only variants such as ready-wave batching. Rerunning path search injects enough quality noise to hide small runtime effects.
- JAX/XLA usually does not overlap independent early contractions automatically, but batching them is not a default win: same-shape leaf or ready-wave batching often gives only modest steady-state gains while greatly increasing compile time.
- Compare search methods at similar budgets unless budget scaling is itself the experiment. Repeat promising stochastic settings; more trials or annealing steps are not monotonic improvements.

## Slicing semantics and strategy

- TensorCircuit's OMECO shortcut wraps `omeco.TreeSA` and returns an unsliced pairwise path. Exercise `TreeSASlicer` directly with `omeco.slice_code(tree, ...)`; cotengra exposes both post-search slicing and integrated SA-time slicing.
- OMECO `TreeSASlicer` couples greedy slice-label selection with renewed TreeSA optimization, so it can rewrite the tree rather than merely post-slice a fixed tree.
- For `SlicedEinsum`, `num_slices()` is the number of sliced labels, not the number of tasks. The task count is `prod(size_dict[ix] for ix in slicing())`, and per-slice FLOPs/write from `complexity()` must be multiplied by that count before comparison with cotengra totals.
- Cotengra `target_size` is a linear element count: use `2**target_log2`, not the logarithmic width itself.
- Sweep cotengra `slice_mode`. `drift` is often a better fast choice than basic post-slicing, while integrated `reslice` can find better small/medium trees but may pathologically explode the slice count on large networks; abort that basin rather than interpreting it as an intrinsic target requirement.
- If an OMECO seed already meets the target width, do not reslice it. When only a few width bits must be removed, a moderately strong TreeSA seed followed by `TreeSASlicer` is a robust starting point; `ntrials=2`, 24 beta points, and 24 iterations is a useful initial budget to measure, not a universal optimum.
- Slicing `k` binary indices creates `2**k` tasks. If thousands of slices are needed, improve the unsliced seed, relax the target, exploit a lightcone or network reformulation, or plan explicit parallel execution instead of merely increasing slicer effort.

## Benchmark interpretation and tuning

- If cotengra imports fail because `autoray.get_namespace` is unavailable, diagnose the cotengra/autoray dependency mismatch before TensorCircuit contraction code. Process-pool failures during hyper-optimization can also be execution-environment artifacts rather than search bugs.
- Use `parallel="auto"` for realistic cotengra end-to-end baselines, but report search separately from compile and steady execution.
- Judge paths with measured execution, peak memory, FLOPs, total write, and maximum intermediate width. FLOPs alone is insufficient: write can dominate steady runtime, while width and allocator behavior often dominate forward peak memory.
- Tune OMECO `rw_weight` empirically. Zero can produce acceptable FLOPs but poor write/width and OOM-prone execution; excessive write penalties can increase width or FLOPs. Rank repeated candidates by actual post-compile runtime and memory, not a single stochastic metric result.
- Treat topology as part of the workload definition. For brickwork comparisons, define a layer as a complete even-plus-odd nearest-neighbor round; even gate-count-normalized brickwork can remain harder than ladder-like networks.
- In tested complex64 amplitude contractions, a useful first-pass forward-memory estimate was about `9 * 2**(log2_size - 27)` GiB, with `11x` logical-intermediate bytes as a conservative planning factor. For full forward-plus-reverse gradients, total write was a better proxy, with roughly `2-3 * 2**(log2_write - 27)` GiB observed and `4x` as a conservative factor. These are workload-specific priors; always replace them with fresh-process measured peaks.
- Verify apparent gradient OOMs with a fresh-process, single-variant `value_and_grad` run. If isolation does not remove the failure, treat it as a real graph-size limit rather than accumulated compilation cache.
