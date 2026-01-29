# TensorCircuit Development Experience & Protocols

This document records specific technical protocols, lessons learned, and advanced practices for developing TensorCircuit. Refer to this when implementing complex features or debugging performance issues.

## JAX, Compilation, and Performance

1.  **Benchmarking JAX Code**:
    *   JAX execution is asynchronous. To measure execution time accurately, always call `.block_until_ready()` on the result's data buffer (e.g., `result.data.block_until_ready()`).
    *   Failing to block will only measure the dispatch time, leading to confusing results (e.g., "first run faster than second run").

2.  **Memory Management (OOM Prevention)**:
    *   For operations over large batches (e.g., summing $2^{22}$ Pauli strings), `vmap` materializes all intermediate results in memory.
    *   **Protocol**: Use `jax.lax.scan` or sequential loops for reductions over large inputs to keep peak memory usage constant ($O(1)$) rather than linear ($O(N)$).

3.  **Vmap Broadcasting with Optimizers (Optax/Custom)**:
    *   **Pitfall**: `tc.backend.vmap(func)` implicitly sets `vectorized_argnums=0`. If `func` takes multiple arguments that are *all* batched (e.g. `update_step(params, opt_state)`), you MUST specify `vectorized_argnums=(0, 1)`.
    *   **Symptom**: Dimension mismatch errors (e.g. `broadcast_shapes got incompatible shapes`) where one argument is treated as a scalar/unbatched while the other is batched.
    *   **Protocol**: Explicitly define `vectorized_argnums` when vmapping functions with optimizer states or multiple batched inputs.

## Qudit Simulation & Advanced Models

1.  **Ansatz Expressibility**:
    *   For $d > 2$ (Qudits), simple "Hardware Efficient" ansätze (single layer of rotations) are often insufficient to reach ground states of complex Hamiltonians (e.g. Potts model).
    *   **Protocol**: Ensure high expressibility by parameterizing *all* $d$ diagonal phases (`rz` on levels $0 \dots d-1$) and *all* off-diagonal mixing angles (`ry` on pairs $(j, k)$).
    *   **Dimension Agnostic Code**: Write code using variables `d` and loops `range(d)` instead of hardcoding `d=3`. This allows the same script to simulate qutrits, ququarts, etc. seamlessly.

2.  **Sparse Matrix Hamiltonian**:
    *   For larger Hilbert spaces ($d^N \gg 10^3$), dense matrix construction explodes in memory.
    *   **Protocol**: Construct Hamiltonians using `scipy.sparse` (COO format), but first prefer to use `PauliStringSum2COO` if available.
    *   **Integration**: Convert to JAX Sparse via `tc.backend.coo_sparse_matrix(indices, values, shape)` and use `tc.backend.sparse_dense_matmul(H_sparse, ket)` for expectation values. This provides massive speedups and enables simulation of larger $N$ or $d$.

## Testing and Robustness

1.  **Sparse Matrix Compatibility**:
    *   TensorCircuit supports multiple sparse formats (Scipy CSR/COO, JAX BCOO, TensorFlow SparseTensor).
    *   **Protocol**: When handling sparse outputs, do NOT assume specific attributes like `.row` or `.col` exist (missing in JAX BCOO or Scipy CSR).
    *   **Check**: Use `hasattr(obj, "tocoo")` to detect Scipy sparse matrices and convert them if a standard COO interface is needed.
    *   **Comparison**: Compare sparse matrices by subtraction (`abs(A - B).max() ≈ 0`) rather than element-wise attribute checks, to be robust against format differences.

## Visualization

1.  **Non-Blocking Plots**:
    *   Library plotting functions (e.g., `Lattice.show`) must accept an optional `ax` (Matplotlib Axis) argument.
    *   **Protocol**: Draw on the provided `ax` and avoid calling `plt.show()` inside library functions. This allows users to embed plots in subplots and manage the figure lifecycle (saving/closing) themselves.

## Benchmarking External Libraries

1.  **Environment Isolation**:
    *   When benchmarking against external libraries (e.g., QuSpin, Quimb, Qiskit) that may have conflicting dependencies, create a dedicated Conda environment (e.g., `bench_quspin`).
    *   Install the external library there using pip from the specific conda environment and run the benchmark script using that specific interpreter.

2.  **Fair Comparison**:
    *   Ensure "apples-to-apples" settings (e.g., same precision `complex128`) to support valid performance claims.

## Noise Modeling and Mitigation

1.  **Readout Error Handling**:
    *   The `circuit_with_noise(c, noise_conf)` function internally applies general quantum noise (Kraus channels specified in `nc`) to gates, but it does **not** automatically apply readout error configuration (`readout_error`) encoded in `NoiseConf`.
    *   **Protocol**: You must **explicitly** pass the readout error when calling `circuit.sample()`. Example: 
        ```python
        c_noisy.sample(..., readout_error=noise_conf.readout_error)
        ```
    *   Failing to do so will result in noiseless measurements even if `readout_error` is present in `NoiseConf`.

2.  **Readout Error Format**:
    *   When adding readout noise via `NoiseConf.add_noise("readout", errors)`, the expected format for each qubit's error is `[p(0|0), p(1|1)]` (a list of two probabilities), **not** the full $2 \times 2$ confusion matrix.
    *   **Pitfall**: Passing a full matrix like `[[0.9, 0.1], [0.1, 0.9]]` can cause unexpected `TypeError` during internal matrix construction (e.g., `1 - list` error).
    *   **Protocol**: Specify readout error as `[0.9, 0.9]` which implies $p(0|0)=0.9$ and $p(1|1)=0.9$.

3.  **Circuit Expectation with Noise**:
    *   The `Circuit.expectation` method supports `noise_conf` as a keyword argument (e.g., `c.expectation(..., noise_conf=conf, nmc=1000)`). This is often cleaner than calling `tc.noisemodel.expectation_noisfy(c, ...)` directly.

4.  **Multi-Qubit Thermal Noise**:
    *   The `thermalrelaxationchannel` returns single-qubit Kraus operators. To apply thermal noise to multi-qubit gates (like CNOT), you generally cannot simply pass the single-qubit channel to `add_noise("cnot", ...)` because of dimension mismatch.

## Backend API Wrapper Quirks

1.  **JAX Vmap in TensorCircuit Wrapper**:
    *   When using `tc.backend.vmap` (alias `K.vmap`) with the JAX backend, do **not** use JAX-native arguments like `in_axes`. The TC wrapper unifies behavior and exposes `vectorized_argnums` (like TensorFlow) instead.
    *   **Protocol**: Always use `vectorized_argnums=(0, 1, ...)` to specify batched arguments, regardless of the backend (JAX/TF/Torch). Passing `in_axes` will raise a `TypeError` because the wrapper function definition doesn't accept it.
