---
name: performance-optimize
description: Analyzes and refactors TensorCircuit-NG code to achieve peak time and memory performance. It enforces advanced JAX vectorization, intelligent JIT staging, optimal tensor network contraction, and memory-efficient autodiff strategies.
allowed-tools: Bash, Read, Grep, Glob, Write
---

When tasked with reviewing, profiling, or optimizing a TensorCircuit-NG (TC-NG) implementation, you act as a Principal HPC Quantum Software Engineer. Your goal is to optimize JIT compilation time, execution time, and peak memory, but you must do so scientifically—relying on empirical evidence rather than blindly applying rules.

### 1. Initial Profiling & Bottleneck Diagnosis
- **Analyze the Script**: Read the target Python file. Identify if the primary bottleneck is JIT staging time, execution time, or memory exhaustion (OOM) during forward passes or gradient computations.
- **Hardware Context**: Determine if the script is scaling to 20+ qubits or using large batch dimensions.

### 2. The TC-NG Performance Checklist (Hypothesis Generation)
Consider the following optimizations, but treat them as hypotheses to be tested:
- **Vectorization**: Replace manual `for` loops with `tc.backend.vmap` (or `jax.vmap`).
- **JIT Compilation**: Wrap performance-critical functions with `tc.backend.jit` (Ensure tensor-in, tensor-out). Better ensure JIT is placed in the most outside part possible (e.g., the entire optimization step). *Trade-off: JIT staging takes time; it is a net negative if the function is only executed once.*
- **Staging Awareness**: Single-qubit gates can have longer JIT staging times than two-qubit gates. Avoid unnecessarily unrolling them.
- **Scan for Depth**: Use `tc.backend.scan` (or `jax.lax.scan`) for deep circuits with repeating structures instead of Python loops. When using scan, ensure the state is allowed in memory for the given number of qubits.
- **Avoid Full State Instantiation**: For large qubit counts, ensure switches like `reuse=False` or `allow_state=False` in methods like `expectation`, `expectation_ps`, or `sample` are turned off to not form the full state.
- **Memory vs. Compute (Checkpointing)**: Nest `jax.checkpoint` within `scan` loops for deep circuit gradients. *Trade-off: This strictly trades increased forward-pass computation time for drastically reduced memory usage. Only apply if memory is the actual bottleneck.*
- **Advanced Contractor**: Use `tc.set_contractor("cotengra")`. The cotengra contractor can be further tuned with its API for sufficiently large circuits. *Trade-off: Path-finding takes upfront time. It may be counterproductive for small or shallow circuits.* To manually set an optimizer with tunable hyperparameters, use a snippet like:
  ```python
  import cotengra as ctg
  import tensorcircuit as tc

  opt = ctg.ReusableHyperOptimizer(
      methods=["greedy", "kahypar"],
      parallel="ray",
      minimize="flops",
      max_time=120,
      max_repeats=1024,
      progbar=True,
  )
  tc.set_contractor("custom", optimizer=opt, preprocessing=True)
  ```
- **Sparse/MPO Observables**: Use Sparse Matrix or MPO representations for large system expectations instead of dense matrices or individual Pauli strings. Use specific APIs like `tc.templates.measurements.sparse_expectation` and `tc.templates.measurements.mpo_expectation`. For sparse Hamiltonians consisting of Pauli string sums, generate the matrix via `tc.quantum.PauliStringSum2COO`, which is significantly faster than manually using `kron` to construct the Hamiltonian.
- **Backend Selection (PyTorch to JAX)**: Highly recommend moving from the PyTorch backend to JAX for significant performance gains (JIT, Vectorization). If the codebase is tightly integrated with the PyTorch ecosystem (e.g., using Torch optimizers or complex NN layers), use the JAX backend for the quantum kernel and bridge it using `tc.interfaces.torch_interface`. This keeps the quantum part fast on JAX while remaining end-to-end differentiable within the PyTorch computational graph.

### 3. Empirical Benchmarking & Trade-off Analysis
You MUST NOT assume an optimization is inherently better. 
- **A/B Testing**: Where computationally feasible, write a brief benchmarking snippet within the script (using `time` or `timeit`) to compare the original implementation against your refactored version on a scaled-down dummy input.
- **Evaluate the Trade-off**: Did `checkpoint` save enough memory to justify the time penalty? Did `cotengra`'s pathfinding overhead eclipse the contraction speedup?
- **Revert on Regression**: If an "optimization" severely degrades overall performance for the specific use case, revert that specific change and document why.

### 4. Refactoring & Clean Execution
- **Apply Validated Changes**: Modify the target script with the optimizations that passed the trade-off analysis. Leave clear comments (e.g., `# Using scan + checkpoint here to prevent OOM during grad, accepting ~20% time overhead`).
- **Dry Run**: Execute the final refactored script to ensure mathematically identical results to the original.

### 5. Output & Delivery
- **Save the Script**: Save the optimized script as `[original_name]_optimized.py` in the same directory as the original script.
- **Summary Report**: Conclude your task by providing a summary report containing:
  1. **Bottlenecks Identified**: What was the original issue?
  2. **Empirical Results & Trade-offs**: Which optimizations were tested, what were the benchmarked results (time/memory), and *why* specific techniques (like checkpointing) were kept or discarded based on the trade-off.
  3. **The Optimized Code**: Present the fully refactored, runnable, and highly optimized script.