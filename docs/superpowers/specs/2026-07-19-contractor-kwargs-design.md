# ContractionAlgebra: replace `prefer_einsum` class attribute with `get_contractor_kwargs()` method

**Date:** 2026-07-19
**Status:** draft
**PR:** feat/contraction-algebra-tropical

## Motivation

`ContractionAlgebra.prefer_einsum` is a cotengra-specific escape hatch that leaks
implementation details into the algebra ABC. The reviewer asked: "but you can
customize the tensordot, can you avoid this?" — the flag exists **not** because
`alg.tensordot` itself is broken, but because cotengra's `Contractor` applies a
post-`tensordot` autoray transpose whose `perm` doesn't know about non-physical
storage axes (e.g., the bf16 pair axis). The custom `tensordot` cannot prevent
this transpose, so `prefer_einsum=True` is the only way to skip it.

However, `prefer_einsum` should not be a first-class ABC attribute. Instead, let
the algebra declare **contractor-level options** through a method whose return
value is transparently forwarded.

## Design

### ABC: `get_contractor_kwargs() -> dict`

```python
class ContractionAlgebra(ABC):
    def get_contractor_kwargs(self) -> dict:
        """Extra kwargs forwarded to cotengra's make_contractor.

        Override to return ``{'prefer_einsum': True}`` when your kernel
        carries non-physical storage axes that cotengra's post-tensordot
        autoray transpose cannot handle.
        """
        return {}
```

- Remove `prefer_einsum: bool = False`.
- Keep the explanatory comment, moved into the method docstring.

### `ComplexPairAlgebra`: override

```python
class ComplexPairAlgebra(ContractionAlgebra):
    def get_contractor_kwargs(self) -> dict:
        return {"prefer_einsum": True}
```

Remove the `prefer_einsum = True` class attribute.

### `cons.py`: consume the method

```python
# Before:
contractor = ctg.core.make_contractor(
    tree, implementation=impl, prefer_einsum=alg.prefer_einsum
)

# After:
contractor = ctg.core.make_contractor(
    tree, implementation=impl, **alg.get_contractor_kwargs()
)
```

## Files changed

| File | Change |
|------|--------|
| `tensorcircuit/contraction_algebra/base.py` | Remove `prefer_einsum` attr; add `get_contractor_kwargs()` method |
| `applications/bcomplex32_algebra.py` | Replace `prefer_einsum = True` with `get_contractor_kwargs()` override |
| `tensorcircuit/cons.py` | Call `alg.get_contractor_kwargs()` instead of `alg.prefer_einsum` |

## Non-goals

- Does not change cotengra's internal behavior
- Does not add new escape hatches — only re-packages the existing one
- No behavior change for `StandardAlgebra` or `TropicalAlgebra`

## Reviewer reply

> Good catch — `prefer_einsum` is now hidden behind a `get_contractor_kwargs()` method on the ABC so the algebra itself doesn't expose cotengra-specific flags as class attributes.