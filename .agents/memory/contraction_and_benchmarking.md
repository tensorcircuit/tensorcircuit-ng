# Contraction and Benchmarking

Use this file for algebraic contraction invariants, cotengra/OMECO path search and slicing, and interpretation of contraction runtime and memory.

## Algebraic contraction invariants

- Route networks with hyperedges through the algebraic contractor so optimization does not materialize diagonal copy tensors. In particular, do not run `_merge_single_gates`-style preprocessing while `CopyNode`s remain.
- Sort nodes deterministically before building topology strings or einsum expressions; stable ordering improves path reuse and compiled-cache hits.
- After algebraically contracting a subgraph, reattach its original dangling edges to the replacement node before continuing.
- `Circuit.matrix()` creates identity-input `CopyNode`s and therefore uses the algebraic hyperedge path even under the default global contractor; unless cotengra is explicitly selected, path search still uses the default opt-einsum greedy/optimal logic.
- Adapt third-party optimizers with integer-only labels at the path-finder boundary, reconstruct a standard pairwise path, and reuse the shared algebraic contractor. Normalize non-callable optional optimizer state once in `set_contractor("custom", optimizer=...)` and keep its import lazy.
- For parameterized two-site gates with low operator-Schmidt rank, benchmark a direct analytic factorization before accepting the dense rank-4 gate topology. Put the parameter-dependent coefficients on one factor and keep the other factor constant; this can reduce forward FLOPs, reverse total write, and the exact spatial-cut carry even though it increases tensor count. RZZ has rank 2, whereas a generic two-qubit gate can require rank 4. Validate full-network values and elementwise gradients, then run a fresh path search because the old path is not reusable.
- Do not assume that precontracting adjacent Schmidt factors into a layer-MPO site tensor preserves the path benefit. It is algebraically exact but changes the topology and can sharply worsen width, FLOPs, and write; keep or revert it only after a matched search-budget comparison.
- For exact spatial block-scan contractions, optimize left cap, repeated bulk block, and right cap separately, persist all three paths, and account for total work as `left + nblocks * bulk + right`. A large block is not automatically better: the cut tensor is fixed by circuit depth, while small regular blocks can admit much stronger paths. Rank block sizes with actual JIT compile time, steady gradient runtime, and peak memory in addition to summed FLOPs/write.

## Search and path reuse

- For large contractions, start with `tc.set_contractor("cotengra")` and a reusable optimizer. Search once and use the exact resulting path or tree for both reported metrics and timed execution; make timed runs cache-only so they cannot silently search a different stochastic path.
- Keep preprocessing enabled for OMECO shortcut contractors unless measurement shows otherwise. Disabling it can make path search and tracing dominate large perfect-sampling workloads without improving path quality.
- Keep the contraction tree fixed when testing execution-only variants such as ready-wave batching. Rerunning path search injects enough quality noise to hide small runtime effects.
- Keep the tree fixed for large complex64 numerical regressions as well. Algebraically equivalent stochastic paths can change accumulation order enough to visibly move extensive observables and gradients; use a persisted path for reproducibility, then use complex128 and small full-network value-plus-elementwise-gradient checks to distinguish rounding from topology errors.
- JAX/XLA usually does not overlap independent early contractions automatically, but batching them is not a default win: same-shape leaf or ready-wave batching often gives only modest steady-state gains while greatly increasing compile time.
- Compare search methods at similar budgets unless budget scaling is itself the experiment. Repeat promising stochastic settings; more trials or annealing steps are not monotonic improvements.
- Persist a strong unsliced OMECO seed before invoking a native slicer. On deep trees, process-level `TreeSASlicer` crashes can come from Rust/Rayon worker-stack overflow in recursive native traversals; set `RUST_MIN_STACK` before the Rayon global pool initializes and control `RAYON_NUM_THREADS`. Raising Python's recursion limit does not fix this, and fixing stack stability says nothing about slice quality.
- Convert very large OMECO binary trees with an iterative postorder traversal and an order-statistics structure. Repeated `list.index` and `pop` make path reconstruction quadratic, and recursive conversion can fail on deep trees; validate binary structure and accept only complete zero- or one-based leaf index sets.

## Slicing semantics and strategy

- TensorCircuit's OMECO shortcut wraps `omeco.TreeSA` and returns an unsliced pairwise path. Exercise `TreeSASlicer` directly with `omeco.slice_code(tree, ...)`; cotengra exposes both post-search slicing and integrated SA-time slicing.
- OMECO `TreeSASlicer` couples greedy near-peak slice-label selection with renewed TreeSA optimization, so it can meet a target with no sliced labels through tree improvement, or chase a moving peak and accumulate an impractical label set. Report the final sliced labels and target violation rather than assuming that invoking the slicer produced useful slicing.
- For `SlicedEinsum`, `num_slices()` is the number of sliced labels, not the number of tasks. The task count is `prod(size_dict[ix] for ix in slicing())`, and per-slice FLOPs/write from `complexity()` must be multiplied by that count before comparison with cotengra totals.
- Cotengra `target_size` is a linear element count: use `2**target_log2`, not the logarithmic width itself.
- Sweep cotengra `slice_mode`. `drift` is often a better fast choice than basic post-slicing, while integrated `reslice` can find better small/medium trees but may pathologically explode the slice count on large networks; abort that basin rather than interpreting it as an intrinsic target requirement.
- Drive slicing from the actual `target_size`, not from a desired number of tasks. Treat sliced-label and task counts as outputs; converting “at least N tasks” into an extra width reduction can force pathological over-slicing. If an OMECO seed already meets the target, keep it unsliced unless tree reoptimization is itself the explicit experiment.
- Slicing `k` binary indices creates `2**k` tasks. If thousands of slices are needed, improve the unsliced seed, relax the target, exploit a lightcone or network reformulation, or plan explicit parallel execution instead of merely increasing slicer effort.
- Cotengra `tree.slice()` is a stochastic fixed-tree post-slicer with candidate scoring, temperature, and repeats, not a purely naive label removal. Compare it with OMECO at the same seed and `target_size`; treat OMECO tree-plus-slice optimization and cotengra integrated `reslice` as separate algorithms, and report width, task count, total/per-task write, and FLOPs instead of declaring a winner from one metric.
- In cotengra 0.8.x `SliceFinder`, a trial can stop when either `target_size` or `target_slices` is reached while final selection requires both. Supplying both targets at once can therefore yield no valid candidate; apply the size target first, then add slices in a second call when a minimum task count is still needed.
- Track per-slice write as `total_write / nslices` for sequential slice-gradient execution. A width-compliant plan can multiply total FLOPs/write yet barely reduce per-slice write, so it may solve forward peak memory without solving reverse-mode memory.

## Benchmark interpretation and tuning

- If cotengra imports fail because `autoray.get_namespace` is unavailable, diagnose the cotengra/autoray dependency mismatch before TensorCircuit contraction code. Process-pool failures during hyper-optimization can also be execution-environment artifacts rather than search bugs.
- Use `parallel="auto"` for realistic cotengra end-to-end baselines, but report search separately from compile and steady execution.
- Judge paths with measured execution, peak memory, FLOPs, total write, and maximum intermediate width. FLOPs alone is insufficient: write can dominate steady runtime, while width and allocator behavior often dominate forward peak memory.
- Tune OMECO `rw_weight` empirically. Zero can produce acceptable FLOPs but poor write/width and OOM-prone execution; excessive write penalties can increase width or FLOPs. Rank repeated candidates by actual post-compile runtime and memory, not a single stochastic metric result.
- Treat topology as part of the workload definition. For brickwork comparisons, define a layer as a complete even-plus-odd nearest-neighbor round; even gate-count-normalized brickwork can remain harder than ladder-like networks.
- In tested complex64 amplitude contractions, a useful first-pass forward-memory estimate was about `9 * 2**(log2_size - 27)` GiB, with `11x` logical-intermediate bytes as a conservative planning factor. For full forward-plus-reverse gradients, total write was a better proxy, with roughly `2-3 * 2**(log2_write - 27)` GiB observed and `4x` as a conservative factor. These are workload-specific priors; always replace them with fresh-process measured peaks.
- Verify apparent gradient OOMs with a fresh-process, single-variant `value_and_grad` run. If isolation does not remove the failure, treat it as a real graph-size limit rather than accumulated compilation cache.
