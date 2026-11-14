# TensorCircuit-NG Vision

This document states the scope and priorities of TensorCircuit-NG, so that maintainers, contributors, and AI agents can decide whether and how an issue or pull request should be handled. Technical correctness is necessary for a change to be merged, but not sufficient: the change should also fit the direction described here.

TensorCircuit-NG does not aim to be all-encompassing. Every feature carries a long-term cost in maintenance and user confusion, and we prefer a smaller framework that is reliable and coherent over a larger one that is uneven.

## Focus

TensorCircuit-NG is built around, and will keep developing, the following strengths:

1. **Machine-learning-native simulation:** automatic differentiation, just-in-time compilation, vectorized parallelism, and composition with neural networks.
2. **Large-scale simulation** on GPUs and supercomputers, with tensor-network contraction and distributed execution.
3. **A rich family of circuit simulators** behind one consistent interface.
4. **AI-native research:** context, rules, skills, and execution backends for AI agents.
5. **Unified hardware access** and hybrid computing across QPUs, CPUs, and GPUs.

Our users are researchers, enterprise users, and AI agents; when their needs conflict, human users come first.

The main line is computing expectation values and their gradients for parameterized circuits, as in variational algorithms. Its behavior must not change.

## Priorities

1. Stability and robustness.
2. Compatibility and consistency: across versions, backends, and circuit classes, and with the original TensorCircuit as a drop-in replacement.
3. Continuity of existing APIs and behavior.
4. New features.

In practice:

- High-frequency APIs do not receive breaking changes; they evolve by adding optional arguments whose defaults preserve existing behavior. Rare breaking changes are limited to rarely used features and documented under **Migration** in `CHANGELOG.md`.
- Bug fixes come before new features, and fixes on the main line and other high-frequency paths come first.
- Behavior on invalid input has low priority. Exhaustive validation makes the code harder to maintain for little benefit, so failing with a natural error is acceptable. Simple improvements may still be accepted.

## Scope

A new core feature should serve one of the strengths above, reuse existing abstractions instead of adding parallel ones, and be maintainable by the core team over the long term. Please open an issue before writing a large pull request.

Not planned in the framework:

- New families of tensor-network and numerical algorithms, such as PEPS and other two-dimensional tensor networks.
- Neural quantum states.
- A full-featured quantum compiler; compilation here is limited to what hardware workflows need, reusing tools such as Qiskit.
- Graphical user interfaces, which can be built on top of the library instead.

Work in these areas, however complex, is welcome in `examples/` as demonstrations or paper reproductions. A separate repository is considered only for work that is self-contained, complete, and of substantial value to users, since each repository adds maintenance and version-pinning burden. Everything else belongs in downstream projects.

`tensorcircuit.tnalg` is kept for its practical uses but will not grow into a general tensor-network algorithm library. Its symmetric block-sparse support fit poorly with the fixed-shape JIT model and cost more to maintain than it gave, a lesson behind the scope above. Likewise, marking a feature as experimental is not a route into the framework: a feature is either maintained properly or not added.

## Backends and Dependencies

- **JAX** is the primary backend, and new performance work targets it first.
- **TensorFlow** receives minimal fixes and keeps up with upstream versions, without large refactors.
- **NumPy** and **PyTorch** remain supported.
- **CuPy** is no longer actively maintained.
- **A complete new backend** is a low priority without a strong user or enterprise use case.
- **Required dependencies** stay minimal: NumPy, SciPy, TensorNetwork-NG, NetworkX, and SymPy. A new one is added only when truly necessary; packages needed by a single feature stay optional and are imported where used.

## Hardware

We are open to new hardware providers, but an integration cannot be verified without hardware, so its contributor should help the maintainers obtain access. Providers the maintainers cannot access are kept as they are, and will be actively completed once access is available. Vendor SDKs remain optional dependencies that users install themselves. We continue to improve the quality of results on real hardware and to deepen hardware support across the stack.

## Reviewing Issues and Pull Requests

1. Is it a bug fix, and how frequently is the affected path used?
2. Does it change existing high-frequency behavior rather than only add to it?
3. Which strength does it serve, and can the core team maintain it?
4. Does it add a parallel abstraction or a new required dependency?
5. Would it fit better in `examples/` or a downstream project?
6. For hardware support, can the maintainers verify it?
7. Is it the smallest correct change?

AI agents reviewing a pull request should report this scope assessment alongside the technical analysis. Coding and testing rules are in [AGENTS.md](AGENTS.md).
