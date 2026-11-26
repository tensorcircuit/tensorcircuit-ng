# Contraction and Benchmarking

Use this file for algebraic contraction invariants, cotengra/OMECO path search and slicing, and interpretation of contraction runtime and memory.

## Algebraic contraction invariants

- Route networks with hyperedges through the algebraic contractor so optimization does not materialize diagonal copy tensors. In particular, do not run `_merge_single_gates`-style preprocessing while `CopyNode`s remain.
- Sort nodes deterministically before building topology strings or einsum expressions; stable ordering improves path reuse and compiled-cache hits.
- After algebraically contracting a subgraph, reattach its original dangling edges to the replacement node before continuing.
- `Circuit.matrix()` creates identity-input `CopyNode`s and therefore uses the algebraic hyperedge path even under the default global contractor; unless cotengra is explicitly selected, path search still uses the default opt-einsum greedy/optimal logic.
- Adapt third-party optimizers with integer-only labels at the path-finder boundary, reconstruct a standard pairwise path, and reuse the shared algebraic contractor. Normalize non-callable optional optimizer state once in `set_contractor("custom", optimizer=...)` and keep its import lazy.
- For parameterized two-site gates with low operator-Schmidt rank such as RZZ, benchmark a direct analytic factorization before accepting the dense rank-4 gate topology. Validate values and gradients, then run a fresh path search because the old path is not reusable.

## Search and path reuse

- For large contractions, start with `tc.set_contractor("cotengra")` and a reusable optimizer. Search once and use the exact resulting path or tree for both reported metrics and timed execution; make timed runs cache-only so they cannot silently search a different stochastic path.
- Keep preprocessing enabled for OMECO shortcut contractors unless measurement shows otherwise. Disabling it can make path search and tracing dominate large perfect-sampling workloads without improving path quality.
- Keep the contraction tree fixed when testing execution-only variants or running complex64 numerical regressions. Rerunning path search injects quality noise, and algebraically equivalent paths can visibly move extensive observables through accumulation order.
- Compare search methods at similar budgets unless budget scaling is itself the experiment. Repeat promising stochastic settings; more trials or annealing steps are not monotonic improvements.

## Slicing semantics and strategy

- TensorCircuit's OMECO shortcut wraps `omeco.TreeSA` and returns an unsliced pairwise path. Exercise `TreeSASlicer` directly with `omeco.slice_code(tree, ...)`; cotengra exposes both post-search slicing and integrated SA-time slicing.
- For `SlicedEinsum`, `num_slices()` is the number of sliced labels, not the number of tasks. The task count is `prod(size_dict[ix] for ix in slicing())`, and per-slice FLOPs/write from `complexity()` must be multiplied by that count before comparison with cotengra totals.
- Cotengra `target_size` is a linear element count: use `2**target_log2`, not the logarithmic width itself.
- Drive slicing from the actual `target_size`, not from a desired number of tasks. Treat sliced-label and task counts as outputs. If an OMECO seed already meets the target, keep it unsliced.
- Slicing `k` binary indices creates `2**k` tasks. If thousands of slices are needed, improve the unsliced seed, relax the target, exploit a lightcone or network reformulation, or plan explicit parallel execution instead of merely increasing slicer effort.
- Track per-slice write as `total_write / nslices` for sequential slice-gradient execution. A width-compliant plan can solve forward peak memory without solving reverse-mode memory.

## Built-in contractor methods

- `set_contractor` dispatches TC's own contractors (`plain`, `plain-experimental`) plus the `opt_einsum`/cotengra/omeco family via `_base`. `plain` is a fixed-order reverse linear contraction (naive state-vector path); `plain-experimental` runs `_merge_single_gates` + dq-fusion preprocessing then the same reverse linear tail. Both bypass path search entirely.
- Unknown method names fall through to the generic `opt_einsum.paths` lookup and raise `AttributeError`, which is the intended fail-fast behavior. The removed `"tng"` method should not be reintroduced: it depended on tensornetwork internals and was empirically worse than `plain`.
- `_merge_single_gates` has two entry points: `experimental_contractor` (always, for >5 nodes) and `_base` (only when `preprocessing=True` and no hyperedges). Tests must exercise both; the default `greedy` contractor without `preprocessing` does not call it.

## Benchmark interpretation and tuning

- If cotengra imports fail because `autoray.get_namespace` is unavailable, diagnose the cotengra/autoray dependency mismatch before TensorCircuit contraction code. Process-pool failures during hyper-optimization can also be execution-environment artifacts rather than search bugs.
- Compare supported public workflows on the same scientific task. Match circuit, algorithm, dtype, device, and measurement method, verify values and gradients before interpreting timings, and state explicitly when one path uses sparse-Hamiltonian expectation rather than a Pauli-term loop. Describe unsupported capabilities directly instead of replacing them with extensive custom code. Keep benchmark scripts minimal and reproducible.
- Use `parallel="auto"` for realistic cotengra end-to-end baselines, but report search separately from compile and steady execution.
- Judge paths with measured execution, peak memory, FLOPs, total write, and maximum intermediate width. FLOPs alone is insufficient: write can dominate steady runtime, while width and allocator behavior often dominate forward peak memory.
- Tune OMECO `rw_weight` empirically. Zero can produce acceptable FLOPs but poor write/width and OOM-prone execution; excessive write penalties can increase width or FLOPs.
- Treat topology as part of the workload definition. For brickwork comparisons, define a layer as a complete even-plus-odd nearest-neighbor round.
- Verify apparent gradient OOMs with a fresh-process, single-variant `value_and_grad` run. If isolation does not remove the failure, treat it as a real graph-size limit rather than accumulated compilation cache.
- Straight-line `jax.checkpoint` over contraction-tree segments is not a reliable memory saver because XLA can undo it. Prefer a control-flow boundary such as `lax.scan` over repeated blocks, and confirm with `compile().memory_analysis()`.
