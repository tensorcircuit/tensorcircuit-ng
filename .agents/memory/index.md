# TensorCircuit Development Experience Memory Index

This directory contains progressive disclosure documents of TensorCircuit development experience, split into modular, non-redundant topics. Agents can grep or read this index to locate specific details quickly.

> [!IMPORTANT]
> **Contribute Back**: If you encounter a new technical bottleneck, a non-obvious backend quirk, or a significant optimization pattern during development, **record it here**. Update the relevant file or create a new one and link it in this index to ensure future agents don't repeat the same research.

## Execution, Performance & Backends
- `jax_compilation_performance.md`: Best practices for JAX compilation, benchmarking, and memory management (OOM prevention).
- `backend_quirks.md`: Backend APIs, JAX vectorization arguments (`vectorized_argnums`), lexsort, JIT buffer updates, and TF/PyTorch to JAX migration.
- `contraction_infrastructure.md`: Handling TensorNetwork contractions, `cotengra` integration, hyperedges, and partial contraction topologies.
- `gradients_and_ad.md`: Complex AD (Wirtinger calculus), stochastic gradients via parameter shift, variance reduction (CRN), and Wide Matrix QR/SVD AD instability.

## Simulator Models & Physics
- `qudit_advanced_models.md`: Specialized simulation tips for qudits ($d>2$) and sparse Hamiltonian formulations.
- `symmetry_aware_simulation.md`: Details on $U(1)$ subspace optimizations, Schmidt advantage for entropy, and large qubit bitwise protections.
- `pauli_operator_evolution.md`: Advice for the Heisenberg picture, backward circuit evolution, tracer accumulation, and real-valued exceptions.
- `noise_modeling_mitigation.md`: Readout errors, formatting requirements, and thermal relaxation quirks.
- `stabilizer_zx_qec_simulation.md`: Vectorized QEC simulation via `StabilizerTCircuit`: noise channels vs. explicit Pauli gates, DETECTOR/OBSERVABLE annotation indexing (0-based absolute, not Stim-style negative), `sample_detectors` output structure, and post-selection pattern.

## Symbolic Circuits & Fourier Analysis
- `symbolcircuit_lambdify.md`: SymbolCircuit gotchas: construct before `set_backend`, `real=True` on symbols, bind all free symbols in `to_circuit`; lambdify to JAX (`modules=[jnp,"math"]`, use `expr.free_symbols`); `to_qiskit` vs `qir2qiskit` incompatibility.

## Architecture, Testing & API
- `circuit_architecture_api.md`: Inheritance hierarchy (U1Circuit, MPSCircuit, AnalogCircuit), the `.inverse()` method's effect on parameters, and QIR roundtripping.
- `module_integration_protocols.md`: Export guidelines, backwards compatibility layers, and strict standards for testing and documentation.
- `testing_and_benchmarking.md`: Pytest acceleration, sparse matrix compatibility, robust testing practices, and guidelines for benchmarking external packages safely.
- `visualization.md`: Standardized protocols for non-blocking subplots embedded during visualization.
- `coding_paradigms.md`: Guidelines for error handling, avoiding defensive over-programming, and early feedback.

## Agent System & Environment
- `agentic_skills_workflows.md`: Available specialized skills (e.g., arxiv-reproduce, code-reviewer, performance-optimize, meta-explorer).
- `agentic_sandbox_cli.md`: Agent sandbox permissions handling, environment redirection (`NUMBA_CACHE_DIR`, `MPLCONFIGDIR`), and log paging.

## AI-Native Documentation Resources
- [Devin Deepwiki](https://deepwiki.com/tensorcircuit/tensorcircuit-ng)
- [Google Code Wiki](https://codewiki.google/github.com/tensorcircuit/tensorcircuit-ng)
- [Context7 MCP](https://context7.com/tensorcircuit/tensorcircuit-ng)
