# Skill: Meta-Explorer

The `meta-explorer` skill is an autonomous research agent designed to discover optimal quantum circuits and optimization strategies for a wide range of quantum computing tasks. 

## 🚀 Capabilities

- **Autonomous Research**: Formulates hypotheses, designs experiments, and analyzes results without human intervention.
- **Multi-Domain Support**: 
  - **VQE**: High-precision ground state search.
  - **QAOA**: Strategy discovery for combinatorial optimization.
  - **QML**: Architecture search for Variational Quantum Classifiers (VQC).
  - **Non-VQA**: Compilation, state tomography, and error mitigation.
- **Intensive Exploration**: Default target of **30+ experiments** to ensure global landscape coverage.
- **Fair Comparison Protocol**: Guarantees scientific integrity through controlled hyperparameter testing.
- **JAX-Native Performance**: Leverages `jit`, `vmap`, and `optax` for high-throughput exploration.

## 🛠️ Tunable Gadgets

The agent creatively explores:
- **Topology**: Grid, 3D, long-range bridges, and all-to-all connectivity.
- **Gate Sets**: Universal SU(4) gates, RZZ/RXX/RYY combinations, and custom gate ordering.
- **Optimization**: Hybrid classical/quantum preprocessing, physics-informed priors (symmetries), and multi-seed initialization stability checks.

## 📂 Research Structure

Every exploration creates a structured repository:
- `objective.py`: Problem definition and evaluation protocol.
- `ledger.json`: Continuous record of every experimental frontier.
- `.snapshots/`: Verifiable code for every successful experiment.
- `research_summary.md`: Synthesis of the "Synergistic Winning Strategy".

## 📜 Usage

When triggered, the agent initializes in `examples/meta_exploration/` and proceeds through a multi-frontier search, referencing existing library examples and using the `performance-optimize` skill for refining top-tier candidates.
