# Skill: Performance Optimize

## Description
The `performance-optimize` skill is designed to analyze and refactor TensorCircuit-NG (TC-NG) code to achieve maximum execution speed and minimal memory footprint. It focuses on scientific optimization using empirical benchmarking and High-Performance Computing (HPC) best practices.

## Optimization Checklist
- **Vectorization**: Replaces manual loops with `tc.backend.vmap` (JAX `vmap`).
- **JIT Compilation**: Wraps performance-critical code in `tc.backend.jit` with a focus on outermost loop placement.
- **Scan for Depth**: Replaces deep repetitive structures with `tc.backend.scan` (JAX `lax.scan`) to reduce compilation time.
- **Memory Management**: Nests `jax.checkpoint` within `scan` loops for deep-circuit gradient computation when memory is the bottleneck.
- **Advanced Contractions**: Configures the `cotengra` contractor for optimal tensor network paths.
- **Sparse & MPO Representations**: Utilizes `Sparse Matrix` or `MPO` observables (`sparse_expectation`, `mpo_expectation`) for large systems.
- **Efficient Hamiltonian Construction**: Uses `PauliStringSum2COO` for fast sparse operator generation.
- **Backend Migration**: Recommends moving from PyTorch/TensorFlow to the JAX backend (with `torch_interface` or `tf_interface` as bridges) for production-grade differentiable performance.

## When to Use
Use this skill when:
- A script is hitting Out-of-Memory (OOM) errors during the forward or backward pass.
- JIT compilation or execution time is excessively long for a large number of qubits (20+).
- You need to optimize an existing VQE, QAOA, or tensor network simulation for GPU/TPU deployment.

## Scientific Benchmarking Workflow
1. **Initial Profiling**: Identifies if the bottleneck is compilation, execution, or memory.
2. **Hypothesis Generation**: Formulates a set of potential optimizations (vmap, jit, checkpoint, etc.).
3. **Empirical A/B Testing**: Compares original vs. refactored code using scaled-down dummy inputs.
4. **Refactoring**: Applies only validated optimizations with documented trade-offs.
5. **Dry Run**: Ensures the final code produces mathematically identical results to the original.
