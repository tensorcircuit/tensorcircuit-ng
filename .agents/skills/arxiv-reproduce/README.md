# Skill: ArXiv Reproduce

## Description
The `arxiv-reproduce` skill is designed to autonomously reproduce quantum computing research papers from ArXiv using the TensorCircuit-NG (TC-NG) framework. It handles the entire lifecycle of research reproduction, from initial paper analysis and adaptive problem scaling to automated code synthesis, execution, and quality enforcement.

## Key Features
- **Adaptive Scaling**: Automatically identifies the core phenomenon of a paper and scales down the problem size (qubit count, circuit depth, etc.) to ensure it is computationally feasible on the local machine while preserving the physics.
- **Standardized Workspace**: Enforces a strict directory structure (`examples/reproduce_papers/<YYYY>_<keywords>/`) and metadata generation (`meta.yaml`).
- **High-Performance Implementation**: Synthesizes JAX-accelerated Python code (`main.py`) following TC-NG best practices (JIT, vmap, grad).
- **Autonomous Debugging**: Capable of self-correcting and retrying execution upon encountering resource limits (OOM) or algorithmic errors.
- **Code Quality Enforcement**: Automatically runs `black` for formatting and `pylint` for linting before final delivery.
- **Scientific Integrity**: Strictly prohibits "cheating" via fake data or hardcoded results; requires explicit transparency and documentation for any implementation simplifications or problem scaling.

## When to Use
Use this skill when you need to:
- Verify the results of a new quantum computing paper.
- Build a baseline implementation for a specific quantum algorithm (VQE, QAOA, DMRG, etc.).
- Create high-quality, documented examples of complex tensor network or quantum circuit operations.

## How it Works
1. **Analyze**: Extracts math, algorithms, and target figures from a paper URL or description.
2. **Initialize**: Sets up the standardized folder and `meta.yaml`.
3. **Synthesize**: Writes the `main.py` simulation script.
4. **Execute**: Runs the script and saves results (figures/data) to the `outputs/` folder.
5. **Review & Refactor**: Performs an internal code review to ensure HPC performance and logic correctness.
6. **Verify**: Formats and lints the code to meet production standards.
