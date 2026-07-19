# Extract `prefer_einsum` from ContractionAlgebra ABC into `get_contractor_kwargs()`

Date: 2026-07-19

## Motivation

`ContractionAlgebra` currently carries a `prefer_einsum: bool` class attribute whose
sole purpose is to be forwarded to `ctg.core.make_contractor(tree, prefer_einsum=...)`.
It exists because cotengra's tensordot mode applies a post-contraction transpose whose
permutation axes do not account for non-physical storage axes (e.g. the trailing
complex<bf16> pair axis), causing `ValueError: axes don't match array`.

This flag is an implementation detail of *how* cotengra executes contractions and
does not belong in the algebra ABC.  The reviewer correctly observed that since the
algebra already customizes `tensordot`, a second escape-hatch flag seems redundant
— but the actual problem is in cotengra's internal transpose, not in our tensordot.

## Design

Replace the hard attribute with a method:

```python
class ContractionAlgebra(ABC):
    def get_contractor_kwargs(self) -> Dict[str, Any]:
        """Extra keyword arguments forwarded to ``ctg.core.make_contractor``.

        Override to return ``{"prefer_einsum": True}`` when your contraction
        kernels produce tensors with non-physical storage axes (e.g. the
        complex<bf16> pair axis) that cotengra's post-tensordot transpose
        cannot handle.

        The default empty dict preserves standard tensordot mode, which
        tropical config-recovery backtracking depends on.
        """
        return {}
```

`ComplexPairAlgebra` overrides it:

```python
class ComplexPairAlgebra(ContractionAlgebra):
    def get_contractor_kwargs(self) -> Dict[str, Any]:
        return {"prefer_einsum": True}
```

In `cons.py` the dispatch changes from:

```python
contractor = ctg.core.make_contractor(
    tree, implementation=impl, prefer_einsum=alg.prefer_einsum
)
```

to:

```python
contractor = ctg.core.make_contractor(
    tree, implementation=impl, **alg.get_contractor_kwargs()
)
```

## Trade-offs

- **Pro:** `ContractionAlgebra` ABC no longer exposes cotengra's `prefer_einsum`
  parameter as a first-class attribute.
- **Pro:** The dict return type is extensible — future cotengra kwargs
  (e.g. `strip_exponent`) can be supplied without changing the ABC again.
- **Pro:** Default `{}` keeps standard algebras unchanged; tropical contracts
  through the same path as before.
- **Con:** The method signature still implies that the algebra knows about
  cotengra's constructor interface.  This coupling is acceptable because the
  algebra's *only* consumer is `_algebraic_base_contraction`, which builds a
  cotengra contractor.

## Affected files

| File | Change |
|------|--------|
| `tensorcircuit/contraction_algebra/base.py` | Replace `prefer_einsum: bool` attribute with `get_contractor_kwargs()` method |
| `applications/bcomplex32_algebra.py` | Replace `prefer_einsum = True` with `get_contractor_kwargs()` override |
| `tensorcircuit/cons.py` | Call `alg.get_contractor_kwargs()` instead of reading `alg.prefer_einsum` |

Tests do not need changes — the runtime behavior is identical.