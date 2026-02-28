# TensorCircuit Development Experience & Protocols

This document records specific technical protocols, lessons learned, and advanced practices for developing TensorCircuit. Refer to this when implementing complex features or debugging performance issues.

## JAX, Compilation, and Performance

1.  **Benchmarking JAX Code**:
    *   JAX execution is asynchronous. To measure execution time accurately, always call `.block_until_ready()` on the result's data buffer (e.g., `result.data.block_until_ready()`).
    *   Failing to block will only measure the dispatch time, leading to confusing results (e.g., "first run faster than second run").

2.  **Memory Management (OOM Prevention)**:
    *   For operations over large batches (e.g., summing $2^{22}$ Pauli strings), `vmap` materializes all intermediate results in memory.
    *   **Protocol**: Use `jax.lax.scan` or sequential loops for reductions over large inputs to keep peak memory usage constant ($O(1)$) rather than linear ($O(N)$). When using `scan`, ensure the qubits' state can actually reside in memory.
    *   **Large Qubit Safeguard**: For large qubit counts, ensure switches like `reuse=False` or `allow_state=False` in methods like `expectation`, `expectation_ps`, or `sample` are turned off. This prevents the formation of the full dense state, avoiding catastrophic OOM.

3.  **Vmap Broadcasting with Optimizers (Optax/Custom)**:
    *   **Pitfall**: `tc.backend.vmap(func)` implicitly sets `vectorized_argnums=0`. If `func` takes multiple arguments that are *all* batched (e.g. `update_step(params, opt_state)`), you MUST specify `vectorized_argnums=(0, 1)`.
    *   **Symptom**: Dimension mismatch errors (e.g. `broadcast_shapes got incompatible shapes`) where one argument is treated as a scalar/unbatched while the other is batched.
    *   **Protocol**: Explicitly define `vectorized_argnums` when vmapping functions with optimizer states or multiple batched inputs.

4.  **JIT Granularity and Placement**:
    *   **Protocol**: Place JIT at the most outside part of the computation loop possible (e.g., wrapping the entire optimization step including the loop itself via `jax.lax.scan`).
    *   **Trade-off**: Avoid JIT-ing small functions inside a Python loop; the dispatch and staging overhead can exceed the execution gain.

## Qudit Simulation & Advanced Models

1.  **Ansatz Expressibility**:
    *   For $d > 2$ (Qudits), simple "Hardware Efficient" ansätze (single layer of rotations) are often insufficient to reach ground states of complex Hamiltonians (e.g. Potts model).
    *   **Protocol**: Ensure high expressibility by parameterizing *all* $d$ diagonal phases (`rz` on levels $0 \dots d-1$) and *all* off-diagonal mixing angles (`ry` on pairs $(j, k)$).
    *   **Dimension Agnostic Code**: Write code using variables `d` and loops `range(d)` instead of hardcoding `d=3`. This allows the same script to simulate qutrits, ququarts, etc. seamlessly.

2.  **Sparse Matrix Hamiltonian**:
    *   For larger Hilbert spaces ($d^N \gg 10^3$), dense matrix construction explodes in memory.
    *   **Protocol**: Construct Hamiltonians using `scipy.sparse` (COO format), but **strictly prefer** `tc.quantum.PauliStringSum2COO` for Pauli sums as it is significantly faster than manual `kron` construction.
    *   **Integration**: Convert to JAX Sparse via `tc.backend.coo_sparse_matrix(indices, values, shape)` and use `tc.templates.measurements.sparse_expectation` or `tc.templates.measurements.mpo_expectation` for large system evaluations. This provides massive speedups and enables simulation of larger $N$ or $d$.

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

## Backend Quirks

1.  **Common Backend APIs**:
    *   The `tc.backend` interface provides unified methods like `K.zeros_like`, `K.ones_like`, and `K.scatter`.
    *   **Protocols for Missing Methods**:
        *   `K.any` -> `K.sum(probs) > 0`
        *   `K.minimum(a,b)` -> `K.where(a < b, a, b)`
        *   `K.outer(a, b)` -> `a[:, None] * b[None, :]` (Broadcasting)

2.  **JIT Buffer Management**:
    *   Updating a fixed-size buffer with a dynamic number of new terms in JAX JIT is challenging due to static shape requirements.
    *   **Pitfall**: `jax.lax.dynamic_update_slice` requires static slice shapes and can behave unexpectedly if logic implies dynamic sizes.
    *   **Protocol**: Use **masked updates** (`K.where`). Create a mask for valid insertion indices and map source indices to destination indices using modular arithmetic or standard indexing, masked by the valid region. This maintains static graph shapes.

3.  **Lexsort Priority Consistency**:
    *   **Protocol**: In TensorCircuit, `lexsort(keys)` consistently treats the **last** key in the sequence as the **primary** sort key, following the NumPy convention.
    *   **Pitfall**: Different backend libraries might have different default priorities (e.g. TensorFlow's `argsort` doesn't natively support lexsort, and custom implementations must be careful). Always verify and test against NumPy's behavior.

4.  **Migrating from PyTorch/TF to JAX**:
    *   **Protocol**: Always prefer the JAX backend for high-performance quantum kernels. 
    *   **Hybrid Workflow**: If the project requires PyTorch (e.g., for specific NN layers or legacy optimizers), use the `jax` backend for the quantum part and bridge it using `tc.interfaces.torch_interface`. This maintains an end-to-end differentiable graph while leveraging JAX's JIT and vectorization for the quantum bottle-necks.

## Pauli Propagation & Operator Evolution (Heisenberg Picture)

1.  **Heisenberg Picture Reverse Order**:
    *   Pauli Propagation evolves the *observable* rather than the state. 
    *   **Protocol**: Circuit operations must be applied in **reversed** chronological order (from the measurement gate back to the initial layer). This corresponds to applying the adjoint gate to the operator: $O \to U^\dagger O U$.

2.  **JAX Tracer Accumulation**:
    *   When initializing an operator state from a Hamiltonian where weights are JAX Tracers (e.g., in VQE optimization), direct indexing and assignment (`state[idx] = w`) will fail with `TracerArrayConversionError`.
    *   **Protocol**: Use the `at[].add()` or `at[].set()` functional update syntax for compatibility with JIT and AD.
        ```python
        state = state.at[target_idx, flat_idx].add(w)
        ```

3.  **Real-Valued Expectations for Gradients**:
    *   JAX gradients of loss functions often require the output to be a real-valued scalar. Even if the physics dictates a real expectation value, numerical complex types (even with zero imaginary part) can trigger `TypeError`.
    *   **Protocol**: Always explicitly take the real part using `K.real()` or `.real` before returning the expectation value from a loss function or engine method.


## Module Integration Protocols

1.  **Exporting and Aliasing**:
    *   Export new modules in `tensorcircuit/__init__.py`. 
    *   Provide both the module export (e.g. `from . import pauliprop`) and the primary helper function (e.g. `from .pauliprop import pauli_propagation`). 
    *   Use concise aliases for frequently used functions (e.g., `PauliProp = pauli_propagation`).

2.  **Backwards Compatibility in Helpers**:
    *   High-level wrapper functions should support multiple input formats (e.g., both raw arrays and convenient list-of-tuples for observables) to be user-friendly while maintaining internal efficiency.

3.  **Standardized Testing Patterns**:
    *   **Backend Isolation**: Avoid global `tc.set_backend()` in test files. Use the standard fixtures (`npb`, `tfb`, `jaxb`) from `conftest.py` as function arguments.
    *   **Test Levels**:
        *   **Unit**: Test initialization, state mapping, and single-gate kernels.
        *   **Correctness**: Compare results against `tc.Circuit.expectation` or `expectation_ps` for small $N$.
        *   **AD/Gradients**: Verify that `jax.grad` (or backend equivalent) works on the module's primary interfaces.
        *   **Scanning**: Verify that loop-optimization interfaces (e.g. `compute_expectation_scan`) match manual application results.

4.  **Documentation & Linting**:
    *   Achieve **10/10 pylint score** and pass `mypy` before finalizing a module. 
    *   Follow Google-style docstrings with reStructuredText markers. This is critical for automated documentation generation.

5.  **User Verification (Walkthroughs)**:
    *   Always provide a production-ready example in `examples/` (e.g., `pauli_propagation_vqe.py`) that showcases a real-world use case (optimization, dynamics, etc.) and demonstrates performance features like JAX JIT and Scanning.

## Circuit Class Architecture & API Consistency

1.  **Inheritance Hierarchy**:
    *   The circuit classes follow: `AbstractCircuit` → `BaseCircuit` → `Circuit` / `DMCircuit`. Side classes (`MPSCircuit`, `U1Circuit`, `StabilizerCircuit`) extend `AbstractCircuit` directly. `QuditCircuit` and `AnalogCircuit` are wrappers that compose an internal `Circuit`.
    *   **Protocol**: When adding a method to `AbstractCircuit`, verify it works for all subclasses. Not all share the same `__init__` signature — `MPSCircuit` uses `mps_inputs`, `DMCircuit` uses `dminputs`, etc.

2.  **`inverse()` Filters `inputs` from `circuit_params`**:
    *   `AbstractCircuit.inverse()` explicitly excludes `inputs`, `mps_inputs`, `dminputs`, `tableau_inputs`, `tensors`, `wavefunction` from the reconstructed circuit's params (line ~440 of `abstractcircuit.py`). This means inverted circuits always start from `|0⟩`.
    *   **Pitfall**: If you need the inverted circuit to support `replace_inputs()` (e.g., in `AnalogCircuit.state()` pipeline), you must rebuild it via `Circuit.from_qir(qir, circuit_params={"nqubits": n, "inputs": placeholder})` to ensure the circuit has a single input tensor node.
    *   **Root Cause**: `replace_inputs` requires `self.inputs is not None`. When `inputs` is filtered out, `Circuit.__init__` uses `all_zero_nodes()` which creates separate nodes with no single input tensor to overwrite.

3.  **`AnalogCircuit` Inverse — Negate Hamiltonian, Not Time**:
    *   The physical inverse of $e^{-iHT}$ is $e^{+iHT} = e^{-i(-H)T}$, achieved by negating the Hamiltonian.
    *   **Pitfall**: Negating the time array (e.g., `[0, 0.5]` → `[-0, -0.5]`) makes the ODE solver receive backward-running time (`times[0] > times[1]`), causing NaN divergence in both `jax.experimental.ode.odeint` and `diffrax`.
    *   **Protocol**: Create a wrapper `neg_ham = lambda t, _orig=block.hamiltonian_func: -_orig(t)` and pass it with the original (positive) time array.

4.  **QIR Roundtrip Consistency**:
    *   `to_qir()` / `from_qir()` is the canonical serialization path. All circuit classes should produce compatible QIR entries with consistent `gatef` and `parameters` fields.
    *   **Protocol**: When adding a new gate to a circuit class, ensure `apply_general_gate` records a proper QIR entry with `gatef` pointing to the gate factory and `parameters` containing all reconstruction-necessary kwargs.
    *   **Cross-class conversion**: `ClassA.from_qir(ClassB.to_qir())` works if both classes support the gates in the QIR. Use `circuit_params` dict to pass class-specific init args (e.g., `{"nqubits": n, "k": k}` for `U1Circuit`).

## Contraction Infrastructure & Hyperedges
 
 1.  **Algebraic Contraction Path for Hyperedges**:
     *   Legacy `tensornetwork` contraction loops (`tn.contract_between`) do not natively support hyperedges (shared edges between >2 tensors) without creating intermediate diagonal `CopyNode` tensors, which frequently leads to Out-Of-Memory (OOM) errors.
     *   **Protocol**: For networks with hyperedges, use the "algebraic" contraction path (e.g., `use_primitives=True`). This maps the entire node collection to a list of bare tensors and a single einsum expression, allowing `cotengra` to optimize the contraction tree globally without materializing massive diagonal tensors.
 
 2.  **Deterministic Topology for JIT Cache**:
     *   Backends like JAX and optimizers like `cotengra` rely on consistent string/topology representations for caching.
     *   **Protocol**: Before extracting einsum symbols or topology, **always sort the input nodes** (e.g., using `_stable_id_`). This ensures that structurally identical circuits always yield the same einsum string, maximizing path-search cache hits and JIT compilation reuse.
 
 
 3.  **Node Consistency in Partial Contraction**:
     *   **Pitfall**: An algebraic contraction result tensor is just a bare array. If the initial nodes were part of a larger network (partial contraction), simply wrapping the array in a new `tn.Node` breaks the connectivity of the remaining dangling edges.
     *   **Protocol**: After an algebraic contraction, manually re-attach the subgraph's original dangling edges to the new `final_node`:
         1.  Iterate through the dangling edges in the deterministic order used by the topology extractor.
         2.  Update each `edge.node1/2` and `edge.axis1/2` to point to the new `final_node`.
         3.  Assign the list of edges to `final_node.edges`.
         4.  Finally, call `final_node.reorder_edges(output_edge_order)` to ensure the tensor's axes match the expected edge sequence.

4.  **Cotengra Performance Tuning**:
    *   **Protocol**: For large-scale tensor network contractions, use `tc.set_contractor("cotengra")`.
    *   **Advanced Tuning**: For complex circuits, manually set a `ctg.ReusableHyperOptimizer` with specific `max_time` and `max_repeats` to find better paths. Use `preprocessing=True` to cache the path.

## Agentic Skills & Specialized Workflows

To maintain high development standards, use the following specialized skills:

1.  **arxiv-reproduce**: Autonomously reproduces ArXiv papers. Focuses on adaptive scaling, standardized meta.yaml generation, and strictly validated code quality.
2.  **performance-optimize**: Scientific bottleneck diagnosis and refactoring. Enforces the "Hypothesis -> Benchmark -> Refactor" workflow using JAX primitives.
3.  **tc-rosetta**: End-to-end intent-based translation from Qiskit/PennyLane to TC-NG. Prioritizes functional rewrites over syntax mapping to achieve maximum speedup.
