---
name: arxiv-reproduce
description: Autonomously reproduces quantum computing arXiv papers using TensorCircuit-NG. It creates standardized repository structures, generates meta.yaml, writes and runs JAX-accelerated code, and strictly enforces code quality (black/pylint) before saving final figures.
allowed-tools: Bash, Read, Grep, Glob, Write
---

When tasked with reproducing an arXiv paper using TensorCircuit-NG, you act as an expert Quantum Software Engineer and must strictly follow this standardized Agentic workflow:

### 1. Paper Reading & Adaptive Scaling
- **Analyze**: Extract the core algorithms, target figure/table, and specific physical phenomena from the provided arXiv link or paper details.
- **Scale Down**: Assess the original problem size. Intelligently reduce the qubit counts, circuit depth, or bond dimension so the simulation is computationally feasible on the current machine, while still theoretically preserving the core phenomenon. Also, plan to compare with exact `tc.Circuit` results for small-size systems to ensure correctness.
- **Result Alignment Goal**: The final goal is to ensure the reproduced results (even if scaled down) qualitatively and quantitatively align with the trends and values in the manuscript.

### 2. Standardized Workspace Initialization
- **Create Directories**: Strictly follow the repository's folder structure convention. Create a new directory named `examples/reproduce_papers/<YYYY>_<keywords>/` (e.g., `examples/reproduce_papers/2022_dmrg_circuit/`).
- **Create Output Folder**: Inside the new directory, create an `outputs/` subfolder for all generated data and figures.
- **Generate `meta.yaml`**: Create a `meta.yaml` file in the main folder using this exact template, filling in the extracted paper details:

```yaml
  title: "[Extracted Title, in Title Case]"
  arxiv_id: "[Extracted ID]"
  url: "https://arxiv.org/abs/[Extracted ID]"
  year: [YYYY]
  authors:
    - "[Author 1]"
    - "[Author 2]"
  tags:
    - "[canonical tag from taxonomy.yaml]"
    - "[canonical tag from taxonomy.yaml]"
  tc_features:
    - "[canonical feature from taxonomy.yaml]"
  backend: "[jax | tensorflow | pytorch | numpy | cupy]"
  hardware_requirements:
    gpu: false
    min_memory: "[Estimated Memory]"
  card_title: "[Short gallery label, <= 60 chars, e.g. 'Figure 2(a) · Phase diagram']"
  summary: "[One sentence, <= 140 chars, what the reproduction shows]"
  description: "[Full description of the reproduction, scaling strategy, and simplifications]"
  outputs:
    - target: "[Target Figure, e.g., Figure 2(a)]"
      path: outputs/result.png
      script: "main.py"
  # thumbnail: outputs/other.png   # optional, only when the first image is a poor card image
```

`meta.yaml` is not a private note: it is the sole data source for the public reproduction gallery,
so a purely mechanical generator must be able to render it without any further judgment. Obey these
rules or the gallery build fails fast on your entry:

- **Controlled vocabulary**: every entry in `tags` and `tc_features` MUST be a key defined in
  `examples/reproduce_papers/taxonomy.yaml`. Read that file first and reuse an existing key, matching
  through its `aliases` where possible. Only if the reproduction genuinely does not fit, add a new key
  to `taxonomy.yaml` in the same change and explain why. Never invent a free-form tag in `meta.yaml`.
- **Honest `tc_features`**: claim a feature only if the corresponding TensorCircuit API actually appears
  in the script. If the script only uses `tc.backend.*` array operations and no circuit object, the
  single correct value is `backend-api`.
- **`card_title` and `summary` are display text**: `card_title` names the reproduced target in at most
  60 characters; `summary` is one plain sentence of at most 140 characters describing what the figure
  shows. Keep the long-form narrative, scaling factors, and simplifications in `description`.
- **Paths are folder-relative**: write `outputs/result.png`, never a repository-root path.
- **Outputs must match disk**: every image written to `outputs/` MUST be declared in `outputs`, and every
  declared path must exist and be git-tracked. Non-image artifacts such as `.log` or `.npz` may also be
  declared. Do not leave an undeclared figure behind.
- **Canonical `url`**: always the `/abs/` form, never `/pdf/`.
- **Only git-tracked reproductions are published**: `git add` the new folder, including `meta.yaml`,
  `main.py`, and everything under `outputs/` that `meta.yaml` declares.

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
  - **Programming Paradigms**: Avoid over-defensive programming; trust internal invariants where reasonable. Use `try...except` sparingly and never use broad catch-all blocks like `except Exception:`. Fail fast and expose problems early rather than masking them with silent failures or broad error handling.
  - Write the mathematical models, quantum circuits, or tensor network operations (like MPS/DMRG contractions).
  - Save all generated plots (using `matplotlib`) directly to the `outputs/` directory (e.g., `outputs/result.png`).
  - Robust Output Paths: The script MUST NOT rely on the terminal's current working directory (CWD) for saving files. You must use pathlib or os.path relative to the script's location (__file__) to dynamically resolve the outputs/ directory.

### 4. Execution & Autonomous Debugging

- **Run the Script**: Execute `main.py`.
- **Self-Correction**: If the code encounters OOM errors, compilation issues, or algorithmic bugs, autonomously analyze the terminal output, modify the script (e.g., adjust the tensor network contraction path or learning rate), and retry until the result is successfully generated.

### 5. Research Integrity & Transparency
- **No Cheating**: You MUST NOT "cheat" on key steps by using fake data, hardcoding expected results, or implementing meaningless workarounds just to get a plot to look right. The physics must be genuine.
- **No Laziness**: Do not skip foundational derivations or critical algorithmic steps in the original paper. Every line of code should have a physical or mathematical basis referenced from the paper.
- **Explicit Simplifications & Discrepancies**: If you take any numerical shortcuts or simplify the implementation (e.g., reducing lattice size, skipping noise channels per section 1.1), you MUST explicitly state exactly **which part** was simplified, **how** it was simplified, and **why** the core physics is still preserved. This information should be documented in both the `meta.yaml` and the final report to the user.
- **Result Analysis**: If the problem scale or hyperparameters are adapted for fast running, you must analyze the results to ensure they remain reasonable within the manuscript's theoretical framework.
- **Identify Non-Identical Results**: If the reproduction is not identical to the original manuscript (e.g., due to scaling, different optimization paths, or hardware constraints), you MUST explicitly mention this to the user in the final report.

### 6. Post-Execution Code Review & Refactoring
Once the script runs successfully and generates the target output, you MUST pause and deeply review your own code:

- **Logic Correctness**: Cross-check your implementation logic against the original paper. Are the Hamiltonian terms, Ansatz structures, measurement bases, and algorithm workflows physically accurate?
- **Scientific Honesty**: Re-verify that the generated results (e.g., phase transition curves, fidelity plateaus) are emerging from the physics logic and not from ad-hoc data manipulation.
- **Performance Bottlenecks**: Analyze the script for HPC anti-patterns. Are you optimally utilizing JAX transformations (jit, vmap, grad)? Refactor to maximize TC-NG's performance.
- **Clean Up**: Rigorously remove any dead code, unused variables, redundant imports, and leftover debugging print statements.

### 7. Verification & Code Quality Enforcement
Before completing the task, you MUST execute the following terminal commands and ensure they pass:

1. **Formatting**: Run `black examples/reproduce_papers/<paper_subfolder>/*.py`
2. **Linting**: Run `pylint examples/reproduce_papers/<paper_subfolder>/*.py`
3. **Output Check**: Verify that `outputs/result.png` exists and matches the expected dimensions/trends of the scaled-down paper results.
4. **Metadata Check**: Re-read `meta.yaml` against the rules in step 2 and against the finished script. Confirm every `tags` and `tc_features` value exists in `examples/reproduce_papers/taxonomy.yaml`, every claimed feature is actually used in the code, every image in `outputs/` is declared, every declared path exists and is git-tracked, and `card_title` and `summary` are within their length limits.
5. **Manuscript Comparison**: Rigorously compare the final numerical and visualized results with the manuscript's figures and tables. Document any discrepancies or confirm the alignment in your final summary.

Conclude your task by summarizing the execution results, confirming that the checklist has been fully met, and providing the path to the reproduced figure. Explicitly list any implementation simplifications made for computational feasibility and clearly state if the final results differ from the original manuscript.