# Skill: Code Reviewer

## Description
The `code-reviewer` skill is an autonomous agentic process that analyzes and refactors TensorCircuit-NG (TC-NG) code. It focuses on maintaining the highest standards of mathematical correctness, HPC performance (via JAX JIT/vmap/scan), and software engineering best practices (DRY, no dead code, strict linting).

## Core Audit Pillars
- **Mathematical Correctness**: Verifies gate logic, wire connectivity, and expectation value calculations.
- **HPC Performance**: Optimizes code for JAX compilation. Prioritizes `tc.backend.jit`, `tc.backend.vmap`, and `tc.backend.scan` over Python-side loops and mutations.
- **Backend flexibility**: Ensures code is backend-agnostic using `tc.backend` abstractions instead of platform-specific libraries.
- **Static Quality**: Enforces strict formatting (`black`), type-safety (`mypy`), and high-quality linting (`pylint`).
- **DRY & Minimalism**: Eliminates code duplication and removes dead code/comments while maintaining a concise documentation style.

## When to Use
Use this skill when:
- You need to refactor a research script into a production-grade module.
- You want to ensure a complex quantum algorithm is mathematically correct and numerically stable.
- You want to optimize the JIT compilation time or execution speed of a differentiable circuit.
- You need to clean up a "messy" script before submitting a Pull Request.

## Workflow
1. **Analyze**: Read the logic and identify mathematical or performance bottlenecks.
2. **Refactor**: Apply functional programming patterns (JIT, vmap) and backend-agnostic abstractions.
3. **Validate**: Perform "equivalence testing" (A/B comparisons) to ensure the refactored code produces identical results to the original.
4. **Static Verification**: Run `black`, `mypy`, and `pylint` to reach 100% compliance.
5. **Report**: Provide a concise summary of optimizations and correctness improvements.
