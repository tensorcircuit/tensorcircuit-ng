# Reproduce Papers with TensorCircuit-NG

This directory contains high-fidelity, reproducible simulations of key quantum computing and quantum information papers using **TensorCircuit-NG**. Each subfolder represents a specific paper reproduction.

## Directory Structure

For any new paper reproduction, follow the standardized naming and folder structure:
```
examples/reproduce_papers/
├── taxonomy.yaml                 # Controlled vocabulary for meta.yaml tags and tc_features
└── <YYYY>_<keywords>/            # e.g., 2026_diff_qec_surface/
    ├── meta.yaml                 # Standard metadata describing the reproduction strategy
    ├── main.py                   # Main executable reproduction script (JAX-native, JIT-friendly)
    └── outputs/                  # Directory containing generated results and plots
        └── result.png            # Reproduced figure/plot
```

## Metadata

`meta.yaml` is the single data source for the public reproduction gallery, which is generated
mechanically from the git-tracked folders here. Its `tags` and `tc_features` must come from the
controlled vocabulary in `taxonomy.yaml`, `card_title` and `summary` carry the gallery display text,
and every image under `outputs/` must be declared and git-tracked. Extend `taxonomy.yaml` in the same
pull request when an existing key genuinely does not fit.

Each gallery entry credits a contributor, taken from the earliest non-bot author in the folder's git
history so that the credit belongs to the person who created or first reviewed the reproduction. Add an
explicit `contributor` to `meta.yaml` only when the whole history is bot-authored.

## Guidance for AI Agents

Refer to the complete instruction set in the `arxiv-reproduce` skill (`.agents/skills/arxiv-reproduce/SKILL.md`) for detailed specifications.
