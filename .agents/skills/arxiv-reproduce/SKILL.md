---
name: arxiv-reproduce
description: Autonomously reproduces quantum computing arXiv papers using TensorCircuit-NG. It creates standardized repository structures, generates meta.yaml, writes and runs JAX-accelerated code, and strictly enforces code quality (black/pylint) before saving final figures.
allowed-tools: Bash, Read, Grep, Glob, Write
---

When tasked with reproducing an arXiv paper using TensorCircuit-NG, you act as an expert Quantum Software Engineer and must strictly follow this standardized Agentic workflow:

### 1. Paper Reading & Adaptive Scaling
- **Analyze**: Extract the core algorithms, target figure/table, and specific physical phenomena from the provided arXiv link or paper details.
- **Scale Down**: Assess the original problem size. Intelligently reduce the qubit counts, circuit depth, or bond dimension so the simulation is computationally feasible on the current machine, while still theoretically preserving the core phenomenon. Also, plan to compare with exact `tc.Circuit` results for small-size systems to ensure correctness.

### 2. Standardized Workspace Initialization
- **Create Directories**: Strictly follow the repository's folder structure convention. Create a new directory named `examples/reproduce_papers/<YYYY>_<keywords>/` (e.g., `examples/reproduce_papers/2022_dmrg_circuit/`).
- **Create Output Folder**: Inside the new directory, create an `outputs/` subfolder for all generated data and figures.
- **Generate `meta.yaml`**: Create a `meta.yaml` file in the main folder using this exact template, filling in the extracted paper details:
  
```yaml
  title: "[Extracted Title]"
  arxiv_id: "[Extracted ID]"
  url: "[Extracted URL]"
  year: [YYYY]
  authors: 
    - "[Author 1]"
    - "[Author 2]"
  tags:
    - "[tag1]"
    - "[tag2]"
  hardware_requirements:
    gpu: False
    min_memory: "[Estimated Memory]"
  description: "[Brief description of the reproduction and scaling strategy]"
  outputs:
    - target: "[Target Figure, e.g., Figure 2(a)]"
      path: outputs/result.png
      script: "main.py"
```

### 3. Code Synthesis (`main.py`)

- **Docstring Standard**: `main.py` MUST start with this exact docstring format:


```python
  """Reproduction of "[Paper Title]"
  Link: [arXiv URL]
  Description:
  This script reproduces [Target Figure] from the paper using TensorCircuit-NG.
  """
```

- **Implementation**:

  - Import `tensorcircuit` as the main framework and set the high-performance backend (e.g., `import tensorcircuit as tc; tc.set_backend("jax")`).
  - Write the mathematical models, quantum circuits, or tensor network operations (like MPS/DMRG contractions).
  - Save all generated plots (using `matplotlib`) directly to the `outputs/` directory (e.g., `outputs/result.png`).
  - Robust Output Paths: The script MUST NOT rely on the terminal's current working directory (CWD) for saving files. You must use pathlib or os.path relative to the script's location (__file__) to dynamically resolve the outputs/ directory.

### 4. Execution & Autonomous Debugging

- **Run the Script**: Execute `main.py`.
- **Self-Correction**: If the code encounters OOM errors, compilation issues, or algorithmic bugs, autonomously analyze the terminal output, modify the script (e.g., adjust the tensor network contraction path or learning rate), and retry until the result is successfully generated.

### 5. Post-Execution Code Review & Refactoring
Once the script runs successfully and generates the target output, you MUST pause and deeply review your own code:

Logic Correctness: Cross-check your implementation logic against the original paper. Are the Hamiltonian terms, Ansatz structures, and measurement bases, algorithm workflows physically accurate?

Performance Bottlenecks: Analyze the script for HPC anti-patterns. Are you optimally utilizing JAX transformations (jit, vmap, grad)? Are there redundant tensor contractions, unnecessary for-loops, or inefficient data transfers between CPU/GPU? Refactor to maximize TC-NG's performance.

Clean Up: Rigorously identify and remove any dead code, unused variables, redundant imports, unnecessary comments, and leftover debugging print statements.

### 6. Verification & Code Quality Enforcement

Before completing the task, you MUST execute the following terminal commands and ensure they pass:

1. **Formatting**: Run `black examples/reproduce_papers/<paper_subfolder>/*.py`
2. **Linting**: Run `pylint examples/reproduce_papers/<paper_subfolder>/*.py` (Fix any critical linting errors to ensure clean code; ignore minor type-hinting warnings if not applicable).
3. **Output Check**: Verify that `outputs/result.png` exists and matches the expected dimensions/trends of the scaled-down paper results.

Conclude your task by summarizing the execution results, confirming that the checklist (formatting, linting, docstrings, self-contained runnable script) has been fully met, and providing the path to the reproduced figure.