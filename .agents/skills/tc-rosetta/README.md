# Skill: TC-Rosetta

## Description
The `tc-rosetta` skill is designed to autonomously translate quantum computing code from other frameworks (Qiskit, PennyLane, Cirq) into performance-optimized TensorCircuit-NG (TC-NG) code. It's a "Principal Quantum Software Migration Expert" that performs end-to-end intent understanding rather than line-by-line syntax translation.

## Feature Highlights
- **Mathematical Intent Extraction**: Identifies Hamiltonian structures, Ansatz architectures, and optimization objectives from the source script.
- **Idiomatic Synthesis**: Rewrites the entire simulation from scratch in the JAX-native, differentiable, and functional paradigm of TC-NG.
- **Legacy Paradox Discarding**: Explicitly avoids the constraints of other frameworks (e.g., Qiskit's slow parameter-binding loops).
- **Execution & Strict Benchmarking**: Automatically runs the original script and the translated version to verify correctness and quantify the speedup.
- **Skill Chaining**: Automatically invokes the `performance-optimize` skill to squeeze maximum performance from the refactored code (using `vmap`, `jit`, and `lax.scan`).

## When to Use
Use this skill when you have:
- A Qiskit/PennyLane/Cirq VQE, QAOA, or hybrid simulation that is too slow.
- A research script from another framework that you want to integrate into the high-performance TC-NG ecosystem.
- A need to migrate legacy quantum code into a differentiable, GPU-accelerated environment.

## Execution & Benchmark Workflow
1. **Source Exploration**: Reads the source script and identifies the mathematical core.
2. **Translation**: Synthesizes a fresh TC-NG script using JAX-native operations.
3. **Validation**: Compares the results of the translated script with the original for numerical equivalence.
4. **Performance Check**: Measures and generates a speedup report (original time vs. TC-NG time).
5. **Optimization**: Further refines the TC-NG code using advanced HPC techniques (Skill Chaining).
6. **Delivery**: Provides a comprehensive migration report including the final, runnable, and highly optimized script.
