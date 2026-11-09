# Backend and Performance

Use this file for backend-wrapper semantics, tracing and vectorization, XLA execution, hot-path structure, and dtype behavior. Use `contraction_and_benchmarking.md` for tensor-network path search and slicing.

## Backend wrapper semantics

- JAX backend `is_tensor` should use `jnp.ndarray`, which also recognizes tracers, and exclude host NumPy arrays. Checking private storage such as `_value` rejects tracers; an explicit `jax.core.Tracer` check is unnecessary.
- PyTorch `index_put` accepts int32 indices eagerly but its vmap path can require int64. Normalize int32 scatter indices in the backend, preserving int64 inputs, and check batched values and update gradients.
- Add a missing tensor primitive to the backend abstraction and cover it with backend-fixture tests instead of introducing a backend-specific call in core logic. When it can be expressed cleanly from existing backend operations, prefer that composition over a backend-specific escape hatch.
- `tc.backend.vmap` selects batched arguments with `vectorized_argnums`, not JAX-style `in_axes`; use a tuple such as `vectorized_argnums=(0, 1)` for multiple batched arguments.
- `lexsort(keys)` follows NumPy semantics: the last key is primary. Test every custom backend implementation against that ordering.
- TensorFlow `tc.backend.to_dense` runs under `jit_compile=True`, where XLA skips index-order validation, so tests built on it cannot catch a non-canonical `SparseTensor`. Host sparse builders should emit row-major canonical COO and regression-test ordering with eager TF.
- Sparse matvec format preference is backend- and device-dependent: on JAX>=0.10 BCSR is faster on CPU while BCOO is faster on GPU, and sorting BCOO indices slows GPU matvec. Torch sparse matmul only takes its fast path for COO tensors flagged coalesced, and CSR beats COO on CPU and GPU. Re-benchmark when bumping framework versions.
- A `lax.scan` (or other jitted region) whose step function is a fresh closure on every call recompiles on every eager call. Put the kernel at module level and wrap it with `backend.jit(..., static_argnums=...)` so eager callers reuse the compiled executable.

## Tracing, vectorization, and timing

- Keep compiled shapes static. Represent dynamically populated buffers with masks and functional updates rather than tensor-dependent shape changes or in-place mutation.
- JIT the outer computation step, warm it with production shapes, and synchronize asynchronous JAX results with `.block_until_ready()` before timing.
- Eager JAX runs each op as a single-primitive jit cached per input shape and dtype, so setup code executed outside `jit` over many distinct shapes pays about 1–2 ms per new shape. A common trap is `convert_to_tensor(np_array, dtype=...)` with a dtype different from the array's; cast on the host with `arr.astype(npdtype)` before conversion. When eager setup is unexpectedly slow, count compiles with `jax.log_compiles()`.
- Enable x64 around the complete transformed invocation, then restore the previous setting through the context manager. A local x64 scope inside a mixed-precision function can work for its forward JIT yet fail when the enclosing reverse transform runs with x64 disabled.
- Use `scan` or chunked streaming when `vmap` would materialize all intermediates. For layered VQE, `scan` is often a fast-compile mode while unrolling gives better steady-state throughput; report them as separate modes and compare total-call break-even with `warmup + (N - 1) * run`.
- For an exact finite-depth 1D bra–MPO–ket contraction, consider scanning fixed-shape spatial blocks with the complete cut tensor as the carry and wrapping the block body in `jax.checkpoint`, as in `examples/omeco_spatial_scan_mpo.py`. The carry grows exponentially with depth and gate operator-Schmidt rank, so this suits low-rank gates such as RZZ.
- For many perfect-sampling shots on a fixed circuit, benchmark a jitted, vmapped `perfect_sampling` over status seeds or fixed-size chunks. This avoids the per-shot Python dispatch used by `Circuit.sample(batch=..., allow_state=False)`.
- Prebuild static contraction paths, sparse Hamiltonians, and other structural objects outside timed value-and-gradient callables. Separate search, compile/warmup, first execution, and steady-state timing.
- `expectation_ps` defaults to `reuse=True`; repeated terms on one circuit do not each reconstruct the state unless reuse is explicitly disabled.

## XLA and GPU execution

- Set `XLA_FLAGS` before Python initializes JAX and report every nondefault flag. Optimization levels are workload-dependent; lowering `--xla_backend_optimization_level` is not a monotonic compile-time improvement.
- For GPU contractions with expensive first execution, A/B test `--xla_gpu_autotune_level=0`. TensorCircuit's `gpu_memory_share(True)` already sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so disabling autotune is usually the independent variable.
- XLA logical buffer analysis and contraction width underpredict actual GPU peak memory. Measure peak device memory in a fresh process for serious capacity decisions.
- A multi-GPU run finishing is not evidence that it is correct. Check every addressable shard of a small `pmap` or sharded result on a new machine before trusting larger runs.

## Hamiltonian MVP and ODE hot paths

- In JAX/Diffrax vector fields for Pauli sums, prefer a matrix-free callable such as `H_mvp(y)` over sparse COO matmul. Nested solve tracing and reverse-mode AD can otherwise carry large COO index constants through `jit`, `grad`, and shape evaluation.
- Keep MVP closures pure under JAX transformations. Do not cache tensors created during tracing in mutable Python state, because tracers can leak across Diffrax's nested transforms.
- Validate the matrix-free and sparse paths numerically before claiming the ODE-specific compile or runtime gain.

## Dtype and long-lived constants

- If a registered object can outlive one trace, normalize raw NumPy inputs to the active TensorCircuit dtype before they become closed-over constants; a late cast in `__call__` may occur after JAX has captured the array.
- Disable TF32 before trusting complex64 results on NVIDIA GPUs by setting `NVIDIA_TF32_OVERRIDE=0`; `jax_default_matmul_precision="highest"` alone is not enough. Verify state norms or a CPU reference.
- In host NumPy hot loops, `int32_array * complex64` promotes to `complex128`, and `(-1j) ** np.int64(k)` is a `complex128` scalar. Precompute phase tables and coefficients in the target dtype. Likewise create tensors from Python scalars with `convert_to_tensor(x, dtype=...)`: converting first and casting later goes through float32 on torch and TensorFlow.
- JAX disables `int64` by default under TensorCircuit complex64 mode. Bit-packed sparse algorithms that require `int64` should use the high-precision test fixture or fail clearly when x64 is unavailable; coefficient precision is a separate concern.

## JAX GPU SVD

- JAX's default GPU `jnp.linalg.svd` can return NaNs on finite rank-deficient complex matrices, for example in MPS two-site splits. The JAX backend therefore uses `jax.lax.linalg.svd(..., algorithm=SvdAlgorithm.QR)` on GPU; keep it when touching SVD and do not swap in another algorithm based on one matrix.
- Pass the LAX `algorithm` argument only on GPU and only when `SvdAlgorithm.QR` exists, since older JAX lacks it and some CPU versions reject it. Check the capability rather than `jax.__version__`.
- A NaN at the first observable of an MPS evolution can come from earlier preparation SVDs. Verify norm and observables after preparation before interpreting later steps.
