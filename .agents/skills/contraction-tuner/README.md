# Skill: Contraction Tuner

Tune tensor-network contraction path search and slicing for TensorCircuit-NG workloads, especially OMECO and cotengra comparisons under realistic memory targets.

Use this skill when a task involves contraction width, FLOPs, total write, slice counts, OMECO `TreeSA` / `TreeSASlicer`, cotengra post-slicing, cotengra integrated `reslice`, or saving a best contraction path for later execution.

The main instructions live in [SKILL.md](SKILL.md). They cover metric hygiene, recommended OMECO and cotengra starting points, stochastic repeat strategy, and when to persist the best found path or sliced tree.
