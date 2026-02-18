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

## 3. Implementation Requirements

> **For AI Assistants / Contributors:** Please strictly follow the rules below.

* **File Location:** Create a new file in: `examples/reproduce_papers/<paper_related_name>.py`
    *Example:* `examples/reproduce_papers/data_reuploading.py`

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

* **Code Quality & Linter:**
    * [ ] **Formatter:** The code must be formatted using `black`.
    * [ ] **Linter:** The code must pass `pylint` checks (clean code, handle errors).
    * [ ] **Type Hints:** Standard type is **NOT REQUIRED**.

* **Dependencies:**
    * Use `tensorcircuit` as the main framework.
    * Use `matplotlib` for plotting if necessary.

## 4. Verification

To be considered complete, please provide:
1.  A screenshot of the generated plot or the terminal output matching the paper's result.
2.  Confirmation that `black` and `pylint` have been run.

```bash
# Example verification commands
black examples/reproduce_papers/your_script.py
pylint examples/reproduce_papers/your_script.py
```

## 5. Checklist

- [ ] I have checked that this paper hasn't been reproduced in the repo yet.

- [ ] The script is self-contained and runnable.

- [ ] The docstring includes the correct arXiv/DOI link.

- [ ] black formatting applied.

- [ ] pylint check passed.