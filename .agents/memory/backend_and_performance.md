# Backend and Performance

Use this file for backend-wrapper semantics, tracing and vectorization, XLA execution, hot-path structure, and dtype behavior. Use `contraction_and_benchmarking.md` for tensor-network path search and slicing.

## Backend wrapper semantics

- Add a missing tensor primitive to the backend abstraction and cover it with backend-fixture tests instead of introducing a backend-specific call in core logic.
- `tc.backend.vmap` selects batched arguments with `vectorized_argnums`, not JAX-style `in_axes`; use a tuple such as `vectorized_argnums=(0, 1)` for multiple batched arguments.
- `lexsort(keys)` follows NumPy semantics: the last key is primary. Test every custom backend implementation against that ordering.
- When a primitive can be expressed cleanly from existing backend operations, prefer that composition over a backend-specific escape hatch.

## Tracing, vectorization, and timing

- Keep compiled shapes static. Represent dynamically populated buffers with masks and functional updates rather than tensor-dependent shape changes or in-place mutation.
- JIT the outer computation step, warm it with production shapes, and synchronize asynchronous JAX results with `.block_until_ready()` before timing.
- Use `scan` or chunked streaming when `vmap` would materialize all intermediates. For layered VQE, `scan` is often a fast-compile mode while unrolling gives better steady-state throughput; report them as separate modes and compare total-call break-even with `warmup + (N - 1) * run`.
- For many perfect-sampling shots on a fixed circuit, benchmark a jitted, vmapped `perfect_sampling` over status seeds or fixed-size chunks. This avoids the per-shot Python dispatch used by `Circuit.sample(batch=..., allow_state=False)`; tune the contraction-search budget separately because search can dominate staging.
- Prebuild static contraction paths, sparse Hamiltonians, and other structural objects outside timed value-and-gradient callables. Separate search, compile/warmup, first execution, and steady-state timing.
- Simulator comparisons must match circuit, dtype, device, and measurement algorithm and must verify values and gradients first. State explicitly when one path uses sparse-Hamiltonian expectation rather than a Pauli-term loop.
- `expectation_ps` defaults to `reuse=True`; repeated terms on one circuit do not each reconstruct the state unless reuse is explicitly disabled.

## XLA and GPU execution

- Set `XLA_FLAGS` before Python initializes JAX and report every nondefault flag. Optimization levels are workload-dependent; lowering `--xla_backend_optimization_level` is not a monotonic compile-time improvement.
- For GPU contractions with expensive first execution, A/B test `--xla_gpu_autotune_level=0`. TensorCircuit's `gpu_memory_share(True)` already sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so disabling autotune is usually the independent variable; unsetting only the shell variable does not restore preallocation after TensorCircuit configures JAX.
- `--xla_gpu_enable_llvm_module_compilation_parallelism=true` can reduce backend compilation when the CUDA/XLA installation supports it, but may fail when the required provider or linker support is absent. Treat `--xla_disable_hlo_passes=fusion` as a diagnostic flag: tested contraction workloads compiled or ran worse without a memory benefit.
- XLA logical buffer analysis and contraction width underpredict actual GPU peak memory because execution also needs transposes, workspaces, caches, and contiguous allocator blocks. Measure peak device memory in a fresh process for serious capacity decisions; `jax.clear_caches()` need not release autotune or device-library state.

## Hamiltonian MVP and ODE hot paths

- In JAX/Diffrax vector fields for Pauli sums, prefer a matrix-free callable such as `H_mvp(y)` over sparse COO matmul. Nested solve tracing and reverse-mode AD can otherwise carry large COO index constants through `jit`, `grad`, and shape evaluation.
- Keep MVP closures pure under JAX transformations. Do not cache tensors created during tracing in mutable Python state, because tracers can leak across Diffrax's nested transforms.
- Validate the matrix-free and sparse paths numerically before claiming the ODE-specific compile or runtime gain.

## Dtype and long-lived constants

- If a registered object can outlive one trace, normalize raw NumPy inputs to the active TensorCircuit dtype before they become closed-over constants; a late cast in `__call__` may occur after JAX has captured the array.
- Disable TF32 or equivalent reduced-precision GPU modes before trusting complex64 cross-backend comparisons.
- JAX disables `int64` by default under TensorCircuit complex64 mode. Bit-packed sparse algorithms that require `int64` should use the high-precision test fixture or fail clearly when x64 is unavailable; coefficient precision is a separate concern.
