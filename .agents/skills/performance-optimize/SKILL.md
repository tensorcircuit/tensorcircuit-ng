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
- **JIT Compilation**: Wrap performance-critical functions with `tc.backend.jit` (Ensure tensor-in, tensor-out). *Trade-off: JIT staging takes time; it is a net negative if the function is only executed once.*
- **Staging Awareness**: Single-qubit gates can have longer JIT staging times than two-qubit gates. Avoid unnecessarily unrolling them.
- **Scan for Depth**: Use `tc.backend.scan` (or `jax.lax.scan`) for deep circuits with repeating structures instead of Python loops.
- **Memory vs. Compute (Checkpointing)**: Nest `jax.checkpoint` within `scan` loops for deep circuit gradients. *Trade-off: This strictly trades increased forward-pass computation time for drastically reduced memory usage. Only apply if memory is the actual bottleneck.*
- **Advanced Contractor**: Use `tc.set_contractor("cotengra")`. *Trade-off: Path-finding takes upfront time. It may be counterproductive for small or shallow circuits.*
- **Sparse/MPO Observables**: Use Sparse Matrix or MPO representations for large system expectations instead of dense matrices or individual Pauli strings.

### 3. Empirical Benchmarking & Trade-off Analysis
You MUST NOT assume an optimization is inherently better. 
- **A/B Testing**: Where computationally feasible, write a brief benchmarking snippet within the script (using `time` or `timeit`) to compare the original implementation against your refactored version on a scaled-down dummy input.
- **Evaluate the Trade-off**: Did `checkpoint` save enough memory to justify the time penalty? Did `cotengra`'s pathfinding overhead eclipse the contraction speedup?
- **Revert on Regression**: If an "optimization" severely degrades overall performance for the specific use case, revert that specific change and document why.

### 4. Refactoring & Clean Execution
- **Apply Validated Changes**: Modify the target script with the optimizations that passed the trade-off analysis. Leave clear comments (e.g., `# Using scan + checkpoint here to prevent OOM during grad, accepting ~20% time overhead`).
- **Dry Run**: Execute the final refactored script to ensure mathematically identical results to the original.

### 5. Output Generation
Conclude your task by providing a summary report containing:
1. **Bottlenecks Identified**: What was the original issue?
2. **Empirical Results & Trade-offs**: Which optimizations were tested, what were the benchmarked results (time/memory), and *why* specific techniques (like checkpointing) were kept or discarded based on the trade-off.
3. **Refactored Code Path**: The location of the optimized script.