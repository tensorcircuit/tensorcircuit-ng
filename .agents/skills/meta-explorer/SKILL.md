---
name: meta-explorer
description: Autonomously explores the space of quantum circuits and optimization strategies to solve specific physical problems (VQE, QAOA, QML, etc.). It uses a budget-constrained, multi-frontier search to discover high-performance solutions and records all experiments in a reproducible registry.
allowed-tools: Bash, Read, Grep, Glob, Write, Web
---

When acting as a **Meta-Explorer**, you are an autonomous researcher tasked with discovering the optimal quantum circuit or optimization strategy for a given objective. This objective can span multiple domains:
- **VQE**: Ground state energy minimization for physics models.
- **QAOA**: Combinatorial optimization on graphs.
- **QML**: Classification, regression, or generative modeling using Variational Quantum Classifiers (VQC).
- **Non-VQA**: Quantum state tomography, circuit compression and compilation, or error mitigation strategy discovery, etc.

### 1. Workspace Initialization
- **Create Directories**: Initialize the research folder: `examples/meta_exploration/<YYYYMMDD>_<short_objective>/`.
- **Create Subfolders**: Create `.snapshots/` to store code for every single successful experiment.
- **Define `objective.py`**: Create a fixed script that contains:
  - The problem core (Hamiltonian, Dataset, or Target State).
  - A `evaluate(circuit_fn, params)` method that returns the core metric (e.g., energy, fidelity, loss, or accuracy).
- **Initialize `ledger.json`**: Create a file to track metadata, results, and "Agent Thoughts" for every experiment.

### 2. Multi-Frontier Exploration Loop
You must maintain a **Top-K Frontier** (default K=3) of the most promising but diverse approaches.

**Exploration Intensity**: Push very hard to explore at least **30+ different experiments** (counting variants in hyperparameters, topologies, and initializations) before declaring a winner. If you can still observe evident progress after these experiments, don't stop, continue to push and explore.

**Creative Search**:
- **Beyond Depth & Optimizer**: Do NOT limit your search to simply increasing layers or swapping optimizers. Rethink the problem's fundamental structure.
- **Internal Brilliance**: Be creative! Invent new gate patterns, explore non-native lattice connectivities, or use ancilla-assisted measurement schemes, try su(4) two qubit gates which are the most expressive.
- **Literature Review**: Use search tools to find promising ideas in recent quantum computing literature and port them to TensorCircuit.
- **Reference Examples**: Examine existing scripts in the repository for implementation patterns and best practices.

**Tunable Gadgets**:
When designing experiments, consider varying the following:
- **Topology & Connectivity**: Grid, all-to-all, long-range bridges, random graphs, or "multiscale" connectivity.
- **Circuit Structure**: Ansatz type (HEA, HVA, UCC, Alt-HEA), gate types (RZZ, RXX, fSim, SU(4) universal 2-qubit gates), and non-standard gate orderings.
- **Physics Prior**: Incorporating known symmetries, specialized initial states, or adiabatic segments.
- **Hybrid Patterns**: Classical pre-processing, mid-circuit measurements, and feed-forward logic.
- **Depth & Width**: Circuit depth and the use of ancilla qubits.
- **Initialization**: Parameter initialization strategies.
- **Optimization**: Optimizer types and their hyperparameters.
- **And more**: dont limit your self to the above gadgets.

### 3. Scientific Integrity & Fairness
Each experiment must be a valid, fair test of the hypothesis:
- **No Cheating**: NEVER train on the test set for QML problems. Ensure the validation/test metrics are isolated from the training process.
- **Fair Comparison Protocol**: Keep non-target hyperparameters (steps, layers, LR) consistent when comparing patches, or explicitly treat them as a combined tuning frontier.
- **No Manual Fitting**: The metric MUST come from the actual simulation; do not hardcode or adjust reported values.

### 4. Implementation Quality
- **Code Quality Enforcement**: Every script generated during the exploration loop MUST be formatted with `black` and checked with `pylint` before execution.
- **Performance Optimization**: Use the `performance-optimize` skill to refactor Top-K candidates (e.g., using `K.jit`, `K.vmap`, or custom JAX ops).
- **Snapshot & Run**: Copy the exact script to `.snapshots/` AFTER successfully running. Stop experiments that exceed the time budget (by default should be at least 10 minutes).
- **Register**: Update `ledger.json` with metrics, duration, and a note on the research insight.

### 5. Synthesis & Final Report
1. **Identify the Winner**: Select the best implementation from the frontier.
2. **Mandatory Discovery Visualization**: You MUST generate a high-quality visualization (e.g., `discovery_curve.png`) that tracks the entire exploration progress. This figure is the "heart" of the report and must include:
   - **Discovery Curve**: A plot of the core metric (e.g., Accuracy, Energy, Inaccuracy in log scale) vs. the sequence of experiments. 
   - **Progress Tracking**: Clear visual markers showing how the "Frontier" moved from baseline to optimal.
3. **Research Summary Content**:
   - Write a `research_summary.md` explaining what worked, what failed, and non-intuitive discoveries.
   - Comparative table of the Top-K candidates.
   - Analysis of how the creative strategy outperforms standard baselines.
4. **Reproducibility**: Verify that the winning snapshot can be re-run to produce the reported result.
