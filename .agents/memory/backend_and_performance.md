# Backend and Performance

Use this file for backend-wrapper behavior, JIT/vmap issues, and contraction performance.

## Backend wrapper rules

- Treat `tc.backend` as the source of truth for tensor operations. Avoid native `numpy` on backend tensors inside core logic.
- If a core algorithm needs a new primitive, add it to the backend abstraction and cover it with backend-fixture tests instead of dropping a raw backend-specific call into one code path.
- `tc.backend.vmap` uses `vectorized_argnums`, not JAX's `in_axes`. When multiple arguments are batched, pass them explicitly, for example `vectorized_argnums=(0, 1)`.
- If a backend helper is missing, build the operation from existing backend-safe primitives instead of dropping into backend-specific APIs.
- `lexsort(keys)` follows the NumPy convention: the last key is the primary key. Test custom backend implementations against that behavior.

## JIT, shapes, and benchmarking

- Keep shapes static inside compiled code. For dynamically populated buffers, prefer masked or functional updates over shape-changing slice logic.
- Put JIT around the outer computation step rather than around tiny inner helpers.
- Warm up JIT with the same shapes used in the real run, especially for vmapped workloads, or compilation will leak into the timed path.
- Benchmark JAX only after synchronizing the result with `.block_until_ready()`.
- JAX/XLA compilation behavior can be tuned with `XLA_FLAGS=--xla_backend_optimization_level=<0..3>` set before Python/JAX startup. Current XLA defaults to level 3; lower levels may reduce some optimization work but can also change compile/runtime tradeoffs in non-monotonic ways.
- For GPU JAX/XLA workloads where first-call compile time dominates, first try `XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=true`; it can reduce backend compilation without necessarily hurting steady-state runtime. `XLA_FLAGS=--xla_disable_hlo_passes=fusion` is a more aggressive fast-compile diagnostic or exploratory mode that may further reduce fusion work but usually slows steady-state execution, so report these flags explicitly and evaluate break-even against repeated calls.
- `Circuit.matrix()` can hit the cotengra-backed algebraic hyperedge path even under the default global contractor. The identity-input construction introduces `CopyNode`s, so execution goes through algebraic contraction while path search still comes from the default `opt_einsum` greedy/optimal logic rather than `set_contractor("cotengra")` hyper-optimization.
- For VQE value-and-gradient benchmarks, report compile/warmup time separately from the first post-JIT value+grad step. Prebuild static objects such as contraction paths and sparse Hamiltonians outside the timed callable.
- When comparing quantum simulators or backends, keep ansatz, dtype, device, and measurement algorithm identical, and verify energies/gradients before interpreting runtime. A sparse-Hamiltonian expectation path can be much fairer for high-qubit Hamiltonian sums than looping over Pauli strings, but the timed algorithm must be stated explicitly.
- `expectation_ps` defaults to `reuse=True`. When summing many Pauli terms on the same circuit state, TensorCircuit already reuses the contracted state unless the caller explicitly disables it, so benchmark or refactor plans should not assume every term recomputes the state from scratch.
- In contraction benchmarks, separate one-time path-search and compile costs from steady-state contraction time. For a fixed traced network, path search is usually a staging cost unless changing static structure or retracing forces it to rerun.
- For very large reductions, prefer `scan` or another streaming pattern over `vmap` if `vmap` would materialize all intermediates at once.
- When large-system APIs expose flags that avoid dense-state materialization, prefer those settings for high-qubit workloads.
- For layered VQE circuits, scanning over layers can cut JAX compile time by keeping the traced program small, but it can also increase steady-state runtime versus an unrolled circuit. Treat `scan` as a fast-compile mode, not automatically as peak-runtime mode.
- When reporting VQE performance, it can be useful to present two TensorCircuit-NG modes explicitly: a scan-based fast-compile mode for low first-call cost, and an unrolled peak-runtime mode for maximum post-compilation throughput. Do not mix the two in a single speedup claim.
- For repeated VQE workloads, compute break-even in total value-and-gradient calls using `warmup + (N - 1) * run`. This is often more informative than reporting compile time and runtime independently.
- For large explicit-status perfect-sampling workloads, `Circuit.sample(batch=..., allow_state=False)` JITs a single-shot sampler and dispatches it once per shot. If many shots share a fixed circuit, benchmark `backend.jit(backend.vmap(lambda seed: circuit.perfect_sampling(seed)[0]))` over the status batch or fixed-size chunks; this can cut both first-call and end-to-end time by avoiding per-shot Python dispatch. Pair this with contractor-budget sweeps, because high OMECO budgets can dominate tracing without improving the batched sampler enough to justify them.

## Hamiltonian MVPs and ODE hot paths

- In JAX/Diffrax ODE vector fields, prefer matrix-free Pauli-sum MVP callables such as `H_mvp(y)` over sparse COO matmul such as `H @ y` when the Hamiltonian is a Pauli-string sum. Nested adaptive-solve tracing and reverse-mode AD can otherwise drag large COO index constants through `jit` / `grad` / `eval_shape`, causing very slow compilation.
- Benchmark this as an ODE-specific optimization, not just an observable optimization. In the challenge-suite digital-analog VQE pattern, replacing analog `COO @ y` calls with `PauliStringSum2MVP(y)` kept the final energy essentially unchanged and reduced end-to-end runtime by about an order of magnitude.
- Keep MVP closures pure under JAX transformations. Do not store backend tensors created during a trace in mutable Python caches inside the MVP closure, because cached tracers can leak across nested transforms such as Diffrax's shape evaluation.

## Contraction patterns

- For networks with hyperedges, prefer the algebraic contraction path so `cotengra` can optimize the whole einsum without constructing large diagonal copy tensors.
- Do not run `_merge_single_gates`-style preprocessing on networks that still contain `CopyNode`s. Route those networks straight to the algebraic contractor; early preprocessing can contract into copy tensors and trigger backend dtype/device mismatches.
- Sort nodes deterministically before building topology strings or einsum expressions. Stable ordering improves both contraction-path reuse and JIT cache hits.
- After partially contracting a subgraph algebraically, reattach the original dangling edges to the replacement node before continuing.
- For large contractions, `tc.set_contractor("cotengra")` plus a reusable optimizer is the default high-performance path.
- When integrating a third-party path finder that only accepts integer labels, adapt the algebraic `input_sets`/`output_set`/`size_dict` into integer-labeled topology, reconstruct a standard pairwise contraction path, and reuse the existing algebraic contractor so hyperedge support stays shared.
- Optional contraction optimizers with non-callable state should be normalized once at the `set_contractor("custom", optimizer=...)` boundary into a callable path finder. Keep the optional import lazy and avoid probing that dependency for ordinary callable optimizers or explicit path lists.
- Keep preprocessing enabled for OMECO shortcut contractors (`tc.set_contractor("omeco...")`) unless a benchmark proves otherwise. On large perfect-sampling tensor networks, disabling preprocessing can make first-call path/JIT staging dominate and cause multi-fold end-to-end slowdowns, while restoring preprocessing can recover the expected runtime without increasing the OMECO search budget.
- When benchmarking explicit contraction-tree batching, keep the contraction tree fixed across variants. If each variant reruns path search, path-quality noise can hide or fake batching gains.
- In the current JAX `jit` / XLA contraction-tree workflow, early independent small contractions are not automatically overlapped. In large circuit simulations this usually has limited end-to-end impact anyway, because runtime is dominated by the late large contractions.
- On large JAX reverse-mode contraction workloads, batching same-shape leaf or ready-wave contractions often yields only modest steady-state gains while inflating compile time much more. Treat these batching schemes as optional tuning knobs, not default contractor behavior.

## Contraction benchmarking protocol

- If `cotengra` fails at import time with `ImportError: cannot import name 'get_namespace' from autoray`, treat it as an environment dependency mismatch and fix `autoray` before debugging TensorCircuit contraction logic.
- Cotengra hyper-optimization can fail inside the sandbox when `joblib` or `loky` needs process-pool primitives such as `SC_SEM_NSEMS_MAX`; for end-to-end benchmark validation, rerun the exact command with escalated permissions instead of changing TensorCircuit logic.
- For cotengra-heavy benchmarks on this repo, prefer direct escalated runs over sandboxed dry-runs; sandbox failures can distort conclusions about search quality or correctness.
- When comparing contraction-search strategies, keep the search budget comparable across methods unless the benchmark explicitly studies budget scaling.
- As of `omeco 0.2.6`, TensorCircuit's OMECO contractor shortcut wraps `omeco.TreeSA` only and returns a normal unsliced pairwise path; `omeco.TreeSASlicer` must be exercised directly with `omeco.slice_code(tree, ...)` and is not a sliced TensorCircuit contractor path. Cotengra supports both post-trial slicing via `slicing_opts` and SA-time slicing via `simulated_annealing_opts={"target_size": ...}`, so distinguish these modes in comparisons.
- In direct cotengra `ContractionTree.simulated_anneal(..., target_size=...)` slicing comparisons, sweep `slice_mode`; `basic` can overslice badly, `drift` is usually faster and better than post-slicing at deeper targets, and `reslice` often gives the best integrated sliced quality but can cost about 2x more search time.
- OMECO `TreeSASlicer` is not a pure fixed-tree post-slicer. In `omeco 0.2.6` it repeatedly finds near-peak intermediates, greedily adds the most frequent unsliced label when above `sc_target`, then runs TreeSA again with sliced labels given log2 size 0. This makes it a conservative coupled slicer: it can improve/rewrite the tree during slicing, but its slice-label selection is still greedy rather than a global slice-cost search.
- For OMECO `SlicedEinsum`, `slicing()` returns the sliced labels and `num_slices()` is the number of sliced labels, not the total number of slice tasks. The total task count is `prod(size_dict[ix] for ix in sliced.slicing())`; for dim-2 circuit indices this is `2 ** len(slicing())`. `SlicedEinsum.complexity()` reports per-slice complexity, so multiply FLOPs/write by the total slice task count before comparing with cotengra `ContractionTree.total_flops()` / `total_write()`.
- Cotengra `target_size` is a linear largest-intermediate element count, not a log2 width. Pass `target_size=2**target_log2`, not `target_log2`.
- Cotengra integrated `slice_mode="reslice"` is often the best quality mode on small/medium networks because slice set and tree rotations co-evolve, but on large networks it can fall into a pathological basin where local subtree SA and repeated reslicing force dozens of sliced indices. Treat this as a failure mode, not as evidence the target is intrinsically impossible; inspect `len(tree.sliced_inds)` and abort early if it explodes.
- OMECO `TreeSASlicer` is usually more stable on large 1D amplitude networks, especially when a stronger OMECO seed tree is already within a few width bits of the target. If `seed_width <= target`, do not run cotengra reslice just because slicing is enabled; it can degrade an already valid OMECO tree.
- For larger amplitude-network slicing benchmarks, a practical robust starting point is: first run a moderately strong OMECO `TreeSA` seed, then run OMECO `TreeSASlicer` with `ntrials=2`, 24 beta points, and `niters=24` (`2×24×24`). This OMECO-seed-plus-slicer workflow was consistently faster and more stable than cotengra integrated `reslice` on large/deep 1D cases such as 200q×25 and 30q×80, where cotengra often over-sliced into tens of indices. Treat `2×24×24` as a default starting point, not a universal optimum; still inspect total slices, total FLOPs, and total write.
- If the target requires slicing many dim-2 indices, both algorithms become unreliable or impractical: `k` sliced indices means `2**k` slice tasks. Use slicing mainly for “a few width bits over memory” cases; when results require thousands or more slices, prioritize a better unsliced seed, a less aggressive target, reformulating the network/lightcone, or explicit parallel execution rather than just increasing slicer budget.
- Do not assume larger OMECo TreeSA budgets will materially improve runtime for a fixed scanned VQE topology. Moderate budgets can already find the useful path, and heavier searches should be justified by measured runtime gains, not only by path-search intuition.
- Prefer `parallel="auto"` as the default cotengra benchmark mode for realistic end-to-end comparisons.
- Search once and reuse the exact path or tree for both reported FLOPs-write metrics and timed execution. Make timed runs cache-only so they cannot silently re-search a different tree.
- `tc_combo_default`-style unseeded combo searches are fine for realism, but any logged metrics must be tied to the exact reused path because repeated combo searches are not deterministic.
- In internal 1D amplitude-network studies, larger-budget annealed searches materially improved reusable paths over the default combo shortcut, but the best heuristic is topology-sensitive: ladder-like networks can benefit more from combo-seeded reconfiguration, while brickwork-like networks often favor different FLOPs-write tradeoffs.
- When comparing ladder and brickwork ansaetze, define one brickwork layer as a full even-plus-odd nearest-neighbor round. Even after gate-count normalization that way, brickwork remains substantially harder, so topology still dominates.
- Judge contraction quality by timed execution together with FLOPs, write, and width. Lower write can beat slightly better FLOPs, so FLOPs alone is not a reliable runtime proxy.

## Dtype and long-lived constants

- If a registered object may survive across tracing boundaries, normalize raw NumPy inputs to the current TensorCircuit dtype before they become closed-over constants.
- Do not assume a late cast inside `__call__` is enough once JAX has already captured the underlying array during lowering.
- For GPU complex64 quantum-simulation benchmarks, disable TF32 or equivalent reduced-precision modes before trusting cross-backend numerical comparisons. Treat any speed result with mismatched energies or gradients as invalid until the precision mode is understood.
- JAX disables `int64` by default under TensorCircuit's complex64 mode. Sparse bit-packed algorithms that explicitly require `int64` bitwise operations should either run under the repo's high-precision dtype fixture or clearly fail/guard when x64 is unavailable; do not infer coefficient precision from this requirement.
