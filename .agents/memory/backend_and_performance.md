# Backend and Performance

Use this file for backend-wrapper behavior, JIT/vmap issues, and contraction performance.

## Backend wrapper rules

- Treat `tc.backend` as the source of truth for tensor operations. Avoid native `numpy` on backend tensors inside core logic.
- `tc.backend.vmap` uses `vectorized_argnums`, not JAX's `in_axes`. When multiple arguments are batched, pass them explicitly, for example `vectorized_argnums=(0, 1)`.
- If a backend helper is missing, build the operation from existing backend-safe primitives instead of dropping into backend-specific APIs.
- `lexsort(keys)` follows the NumPy convention: the last key is the primary key. Test custom backend implementations against that behavior.

## JIT, shapes, and benchmarking

- Keep shapes static inside compiled code. For dynamically populated buffers, prefer masked or functional updates over shape-changing slice logic.
- Put JIT around the outer computation step rather than around tiny inner helpers.
- Warm up JIT with the same shapes used in the real run, especially for vmapped workloads, or compilation will leak into the timed path.
- Benchmark JAX only after synchronizing the result with `.block_until_ready()`.
- For very large reductions, prefer `scan` or another streaming pattern over `vmap` if `vmap` would materialize all intermediates at once.
- When large-system APIs expose flags that avoid dense-state materialization, prefer those settings for high-qubit workloads.

## Contraction patterns

- For networks with hyperedges, prefer the algebraic contraction path so `cotengra` can optimize the whole einsum without constructing large diagonal copy tensors.
- Sort nodes deterministically before building topology strings or einsum expressions. Stable ordering improves both contraction-path reuse and JIT cache hits.
- After partially contracting a subgraph algebraically, reattach the original dangling edges to the replacement node before continuing.
- For large contractions, `tc.set_contractor("cotengra")` plus a reusable optimizer is the default high-performance path.

## Dtype and long-lived constants

- If a registered object may survive across tracing boundaries, normalize raw NumPy inputs to the current TensorCircuit dtype before they become closed-over constants.
- Do not assume a late cast inside `__call__` is enough once JAX has already captured the underlying array during lowering.
