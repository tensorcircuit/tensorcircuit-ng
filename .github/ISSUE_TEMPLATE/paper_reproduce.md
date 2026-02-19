---
name: Paper Reproduction Request
about: Suggest a paper to reproduce using TensorCircuit-NG
title: "Reproduce <Paper Title>"
labels: "good first issue"
assignees: ""
---

## 1. Paper Information

* **Title:** 

* **Link:**

* **Tags:** 

## 2. Reproduction Goal

Which specific figure or table from the paper needs to be reproduced?

* **Target:** 

* **Expected Metric/Result:** 

* **Allow to reduce the problem size compared to the paper:** Yes

## 3. Directory Structure Plan

To keep the repository organized, please strictly follow this folder structure:

* **Proposed Folder Name:** `examples/reproduce_papers/<YYYY>_<keywords>/`
    * *Naming Convention:* `year_keywords`  
    * *Example:* `examples/reproduce_papers/2023_quantum_transformer/`

## 4. Metadata Definition (`meta.yaml`)

Please provide the content for the `meta.yaml` file that will be placed in the folder. 

**Contributors/AI:** Copy and fill this block directly into the `meta.yaml` file.

```yaml
title: ""
arxiv_id: ""
url: ""
year: 
authors: 
  - ""
  - ""
tags:
  - ""
  - ""
hardware_requirements:
  gpu: False
  min_memory: ""
description: ""
outputs:
  - target: "Figure 3(a)"
    path: result.png
    script: "main.py"
  - target: "Figure 3(b)"
    path: comparison.csv
    script: "main.py"
```

## 5. Implementation Requirements

> **For AI Assistants / Contributors:** Please strictly follow the rules below.

* **Docstring Standard:**
    * The file **MUST** start with a docstring containing the paper title and the link.
    *Format example*:

        ```python
        """
        Reproduction of "Paper Title"
        Link: https://arxiv.org/abs/xxxx.xxxxx

        Description:
        This script reproduces Figure 3 from the paper using TensorCircuit-NG.
        """
        ```

* **Outputs:**
  The reproduced data and figures should be in outputs/ subfolder.

* **Subfolder structure:**

  ```
  examples/reproduce_papers/<YYYY>_<Keywords>/
  ├── meta.yaml
  ├── main.py
  ├── ...       # models, data, etc.
  └── outputs/
      ├── result.png    # used for gallery
      └── ...           # other outputs if necessary
  ```

* **Code Quality & Linter:**
    * [ ] **Formatter:** The code must be formatted using `black`.
    * [ ] **Linter:** The code must pass `pylint` checks (clean code, handle errors).
    * [ ] **Type Hints:** Standard type is **NOT REQUIRED**.

* **Dependencies:**
    * Use `tensorcircuit` as the main framework.
    * Use `matplotlib` for plotting if necessary.

## 6. Verification

To be considered complete, please provide:
1.  A screenshot of the generated plot or the terminal output matching the paper's result.
2.  Confirmation that `black` and `pylint` have been run.

```bash
# Example verification commands
black examples/reproduce_papers/<paper_subfolder>/*.py
pylint examples/reproduce_papers/<paper_subfolder>/*.py
```

## 7. Checklist

- [ ] I have checked that this paper hasn't been reproduced in the repo yet.

- [ ] The script is self-contained and runnable.

- [ ] The docstring includes the correct arXiv/DOI link.

- [ ] black formatting applied.

- [ ] pylint check passed.