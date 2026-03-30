# JAX, Compilation, and Performance

1.  **Benchmarking JAX Code**:
    - JAX execution is asynchronous. To measure execution time accurately, always call `.block_until_ready()` on the result's data buffer (e.g., `result.data.block_until_ready()`).
    - Failing to block will only measure the dispatch time, leading to confusing results (e.g., "first run faster than second run").

2.  **Memory Management (OOM Prevention)**:
    - For operations over large batches (e.g., summing $2^{22}$ Pauli strings), `vmap` materializes all intermediate results in memory.
    - **Protocol**: Use `jax.lax.scan` or sequential loops for reductions over large inputs to keep peak memory usage constant ($O(1)$) rather than linear ($O(N)$). When using `scan`, ensure the qubits' state can actually reside in memory.
    - **Large Qubit Safeguard**: For large qubit counts, ensure switches like `reuse=False` or `allow_state=False` in methods like `expectation`, `expectation_ps`, or `sample` are turned off. This prevents the formation of the full dense state, avoiding catastrophic OOM.

3.  **Vmap Broadcasting with Optimizers (Optax/Custom)**:
    - **Pitfall**: `tc.backend.vmap(func)` implicitly sets `vectorized_argnums=0`. If `func` takes multiple arguments that are _all_ batched (e.g. `update_step(params, opt_state)`), you MUST specify `vectorized_argnums=(0, 1)`.
    - **Symptom**: Dimension mismatch errors (e.g. `broadcast_shapes got incompatible shapes`) where one argument is treated as a scalar/unbatched while the other is batched.
    - **Protocol**: Explicitly define `vectorized_argnums` when vmapping functions with optimizer states or multiple batched inputs.

4.  **JIT Granularity and Placement**:
    - **Protocol**: Place JIT at the most outside part of the computation loop possible (e.g., wrapping the entire optimization step including the loop itself via `jax.lax.scan`).
    - **Trade-off**: Avoid JIT-ing small functions inside a Python loop; the dispatch and staging overhead can exceed the execution gain.
5.  **Batched Execution (Vmap) and JIT Re-triggering**:
    - **Lesson**: JAX JIT caches specialized functions based on input shapes. If you warm up a JIT-ed vmap function with a small batch (e.g., `batch=2`) but run it with a large batch (e.g., `batch=100`), JAX will re-trigger compilation during the "execution" phase, leading to heavily distorted benchmarks.
    - **Protocol**: Always warm up the JIT compiler with the **exact** same batch dimension shape as the production/benchmark run.
