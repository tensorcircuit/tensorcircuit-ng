# TensorCircuit Memory Index

This directory stores short, durable notes for future development. Read this index first, then open only the file closest to the task.

## What belongs here

- Keep repo-specific lessons that are easy to forget and expensive to rediscover.
- Prefer stable design rules, API invariants, and recurring backend pitfalls over one-off debugging logs.
- Update an existing topic file when possible. Create a new file only if the lesson does not fit the current taxonomy.

## Memory map

- `workflow_and_quality.md`: Runtime-state invariants, public-feature integration, example design, and high-signal validation guidance.
- `backend_and_performance.md`: `tc.backend` wrapper behavior, JIT/vmap/scan rules, XLA execution, hot-path design, and dtype pitfalls.
- `contraction_and_benchmarking.md`: Algebraic contraction invariants, path search and slicing semantics, and contraction benchmark interpretation.
- `autodiff_and_stochastic.md`: Complex AD conventions, gradient debugging workflow, stochastic-gradient rules, and real-loss requirements.
- `circuit_models_and_api.md`: Circuit hierarchy, inverse semantics, analog evolution inversion, and QIR reconstruction rules.
- `simulation_patterns.md`: U(1) subspace work, qudit/sparse-model guidance, and Heisenberg-style operator evolution.
- `symbolic_and_translation.md`: `SymbolCircuit` construction rules, SymPy-to-JAX lambdify usage, and translation/interoperability boundaries.
- `noise_and_qec.md`: Noise API behavior, readout-error conventions, detector sampling, and QEC post-selection patterns.

## External references

- [Devin Deepwiki](https://deepwiki.com/tensorcircuit/tensorcircuit-ng)
- [Google Code Wiki](https://codewiki.google/github.com/tensorcircuit/tensorcircuit-ng)
- [Context7 MCP](https://context7.com/tensorcircuit/tensorcircuit-ng)
