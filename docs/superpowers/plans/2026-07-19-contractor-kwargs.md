# Replace `prefer_einsum` with `get_contractor_kwargs()` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `prefer_einsum: bool` class attribute on `ContractionAlgebra` with a `get_contractor_kwargs()` method, keeping the same behavior while decoupling the ABC from cotengra internals.

**Architecture:** A single new method on the ABC returns a dict of kwargs to forward to `ctg.core.make_contractor`. The default returns `{}`. `ComplexPairAlgebra` overrides to return `{"prefer_einsum": True}`. `cons.py` calls the method instead of reading the attribute.

**Tech Stack:** Pure Python — no new dependencies.

## Global Constraints

- No behavior change for any algebra
- All existing tests must pass
- Three files touched: `base.py`, `bcomplex32_algebra.py`, `cons.py`

---

### Task 1: Add `get_contractor_kwargs()` to ABC, remove `prefer_einsum` attribute

**Files:**
- Modify: `tensorcircuit/contraction_algebra/base.py:59-69`

**Interfaces:**
- Produces: `ContractionAlgebra.get_contractor_kwargs() -> dict` (default returns `{}`)

- [ ] **Step 1: Replace `prefer_einsum` attribute with `get_contractor_kwargs()` method**

Replace lines 59-69:

```python
    name: str = "abstract"
    representation: Representation = IdentityRepresentation()
    # When operands carry a trailing non-physical storage axis (e.g. the
    # complex<bf16> pair), cotengra's tensordot mode post-transposes results via
    # autoray and mishandles that extra axis (ValueError: axes don't match array)
    # -- set True to force einsum-only execution, which forwards operands verbatim
    # to ``einsum`` and skips the autoray transpose. Default False keeps tensordot
    # mode: tropical config-recovery backtracking depends on the tensordot
    # intermediate layout and would break under forced einsum.
    prefer_einsum: bool = False
```

With:

```python
    name: str = "abstract"
    representation: Representation = IdentityRepresentation()

    def get_contractor_kwargs(self) -> dict:
        """Extra kwargs forwarded to cotengra's ``make_contractor``.

        Override to return ``{'prefer_einsum': True}`` when your algebra's
        ``tensordot`` kernel carries non-physical storage axes (e.g. the
        complex<bf16> pair axis) that cotengra's post-tensordot autoray
        transpose would mishandle (ValueError: axes don't match array).
        ``prefer_einsum=True`` forces einsum-only execution, which skips
        the transpose entirely.

        Default ``{}`` keeps the standard tensordot+einsum mix — required
        by tropical config-recovery backtracking, which depends on the
        tensordot intermediate layout.
        """
        return {}
```

- [ ] **Step 2: Verify the file is valid Python**

Run: `D:/Software/miniconda3/envs/tcng/python.exe -c "from tensorcircuit.contraction_algebra.base import ContractionAlgebra; print(ContractionAlgebra().get_contractor_kwargs())"`
Expected: `{}`

- [ ] **Step 3: Commit**

```bash
git add tensorcircuit/contraction_algebra/base.py
git commit -m "refactor: replace prefer_einsum attr with get_contractor_kwargs() method on ContractionAlgebra"
```

---

### Task 2: Override `get_contractor_kwargs()` in `ComplexPairAlgebra`

**Files:**
- Modify: `applications/bcomplex32_algebra.py:175-178`

**Interfaces:**
- Consumes: `ContractionAlgebra.get_contractor_kwargs() -> dict` (from Task 1)

- [ ] **Step 1: Replace `prefer_einsum` class attribute with method override**

Replace line 178:

```python
    prefer_einsum = True  # pair operands carry a trailing storage axis
```

With:

```python
    def get_contractor_kwargs(self) -> dict:
        return {"prefer_einsum": True}
```

The class now looks like:

```python
class ComplexPairAlgebra(ContractionAlgebra):
    name = "bcomplex32_pair"
    representation = PairBf16Representation()

    def get_contractor_kwargs(self) -> dict:
        return {"prefer_einsum": True}

    def tensordot(self, be: Backend, a: Tensor, b: Tensor, axes: Any) -> Tensor:
        ...
```

- [ ] **Step 2: Verify**

Run: `D:/Software/miniconda3/envs/tcng/python.exe -c "from applications.bcomplex32_algebra import ComplexPairAlgebra; print(ComplexPairAlgebra().get_contractor_kwargs())"`
Expected: `{'prefer_einsum': True}`

- [ ] **Step 3: Commit**

```bash
git add applications/bcomplex32_algebra.py
git commit -m "refactor: use get_contractor_kwargs() override in ComplexPairAlgebra"
```

---

### Task 3: Update `cons.py` to call `get_contractor_kwargs()`

**Files:**
- Modify: `tensorcircuit/cons.py:784-785`

**Interfaces:**
- Consumes: `ContractionAlgebra.get_contractor_kwargs() -> dict` (from Task 1)

- [ ] **Step 1: Replace `alg.prefer_einsum` with `**alg.get_contractor_kwargs()`**

Replace lines 784-785:

```python
            contractor = ctg.core.make_contractor(
                tree, implementation=impl, prefer_einsum=alg.prefer_einsum
            )
```

With:

```python
            contractor = ctg.core.make_contractor(
                tree, implementation=impl, **alg.get_contractor_kwargs()
            )
```

- [ ] **Step 2: Verify no remaining references to `prefer_einsum` on algebra**

Run: `cd "e:\Study\.AShare\OneDrive\OneDriveSync\session\tc\tensorcircuit-ng" && grep -rn "prefer_einsum" tensorcircuit/ applications/ --include="*.py" | grep -v test | grep -v __pycache__`
Expected: only `bcomplex32_algebra.py` (inside the new method returning the dict), and possibly `base.py` docstring mentioning it as an example

- [ ] **Step 3: Run existing tests**

Run: `D:/Software/miniconda3/envs/tcng/python.exe -m pytest tests/test_bcomplex32_algebra.py tests/test_contraction_algebra.py tests/test_tropical.py -v`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tensorcircuit/cons.py
git commit -m "refactor: consume alg.get_contractor_kwargs() in _algebraic_base_contraction"
```