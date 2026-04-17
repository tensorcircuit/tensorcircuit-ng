# Skill: Sanity Checker

## Description
The `sanity-checker` skill is a comprehensive codebase auditor designed to reduce entropy and improve long-term maintainability in the TensorCircuit-NG project. It systematically identifies technical debt across imports, documentation, code structure, and test coverage.

## Core Audit Pillars
- **Dependency Isolation**: Ensures optional dependencies (like Qiskit or CuPy) are imported lazily within function scopes, preventing unnecessary package requirements for the base library.
- **Minimalist Documentation**: Removes "noisy" comments, AI chain-of-thought, and outdated notes while ensuring public APIs have high-quality, up-to-date docstrings.
- **Code Integrity**: Identifies dead code, broad exception handling, and hardcoded secrets or magic numbers.
- **Architectural DRY**: Flags copy-pasted logic and duplicate implementations for consolidation.
- **Algorithmic Verification**: Evaluates tests for value-level correctness and physics-based validation, moving beyond simple syntax checks.

## When to Use
Use this skill when:
- You are cleaning up a feature branch before a pull request.
- You suspect technical debt or "entropy" is increasing in a specific module.
- You want to ensure the library remains lightweight with properly isolated optional dependencies.
- You want to audit a large section of the codebase for consistency and documentation quality.

## Workflow
1. **Audit (Read-Only)**: The skill scans the specified path (or the entire codebase) and generates a structured **Sanity Check Report**.
2. **Review**: The user reviews the report findings and recommendations.
3. **Approval**: The user explicitly approves some or all of the proposed changes.
4. **Refactor**: Only after approval, the skill applies surgical fixes, verifies them with tests, and ensures they meet linting standards.

## Report Categories
- **Critical/Security**: Immediate risks like hardcoded secrets or incorrect physics.
- **Structural/DRY**: Duplication and overly broad exception handling.
- **Maintainability**: Import isolation, docstring quality, and dead code.
- **Documentation**: Removal of verbose or outdated comments.
- **Testing**: Gaps in algorithmic and value-level verification.
