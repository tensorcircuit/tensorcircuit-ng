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
- When benchmarking explicit contraction-tree batching, keep the contraction tree fixed across variants. If each variant reruns path search, path-quality noise can hide or fake batching gains.
- In the current JAX `jit` / XLA contraction-tree workflow, early independent small contractions are not automatically overlapped. In large circuit simulations this usually has limited end-to-end impact anyway, because runtime is dominated by the late large contractions.
- On large JAX reverse-mode contraction workloads, batching same-shape leaf or ready-wave contractions often yields only modest steady-state gains while inflating compile time much more. Treat these batching schemes as optional tuning knobs, not default contractor behavior.

## Contraction benchmarking protocol

- If `cotengra` fails at import time with `ImportError: cannot import name 'get_namespace' from autoray`, treat it as an environment dependency mismatch and fix `autoray` before debugging TensorCircuit contraction logic.
- Cotengra hyper-optimization can fail inside the sandbox when `joblib` or `loky` needs process-pool primitives such as `SC_SEM_NSEMS_MAX`; for end-to-end benchmark validation, rerun the exact command with escalated permissions instead of changing TensorCircuit logic.
- For cotengra-heavy benchmarks on this repo, prefer direct escalated runs over sandboxed dry-runs; sandbox failures can distort conclusions about search quality or correctness.
- When comparing contraction-search strategies, keep the search budget comparable across methods unless the benchmark explicitly studies budget scaling.
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
