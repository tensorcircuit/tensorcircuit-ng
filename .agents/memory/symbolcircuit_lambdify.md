---
name: SymbolCircuit and Lambdify to JAX
description: TC-specific gotchas for SymbolCircuit construction, lambdify, and JAX integration
type: feedback
---

## SymbolCircuit pitfalls

**Construct before `tc.set_backend`.**  SymbolCircuit uses numpy object-dtype arrays internally; creating one after `tc.set_backend("jax")` raises `TypeError: Dtype object is not a valid JAX array type`.

**Declare gate-parameter symbols with `real=True`.**  `expectation_before()` conjugates the bra side via `tn.copy(nodes, conjugate=True)`, which calls `sympy.conjugate()` on every gate-matrix entry.  Without `real=True`, `sympy.conjugate(sin(theta))` stays unevaluated and breaks lambdify with JAX (`conjugate` has no `jnp` equivalent).
```python
theta = sympy.Symbol("theta", real=True)   # correct
theta = sympy.Symbol("theta")              # breaks lambdify to JAX
```

**`to_circuit` requires ALL free symbols bound — including data inputs.**  If a gate uses `x` (a data symbol), it must appear in the `param_dict`:
```python
sc.to_circuit({**trainable_dict, x_sym: x_value})
```

## Lambdify to JAX

**`modules=[jnp, "math"]`, not `"jax.numpy"`.**  Sympy only recognises predefined strings; `"jax.numpy"` raises `NameError`.  Pass the module object directly:
```python
sympy.lambdify(syms, expr, modules=[jnp, "math"])
```
Sympy calls `vars(jnp)` to map `sin`/`cos`/`exp`/… to `jnp.*`.  The result is fully JAX-native: `jit`, `grad`, and `vmap` all work.

**Use `expr.free_symbols`, not `sc.free_symbols()`.**  The circuit-level set is a superset; a specific expectation may analytically eliminate symbols.  Using stale symbols gives wrong argument counts.
```python
syms = sorted(expr.free_symbols, key=lambda s: s.name)
f = sympy.lambdify(syms, expr, modules=[jnp, "math"])
```

## to_qiskit vs translation.qir2qiskit

They cannot share gate-dispatch code because they live in different parameter worlds: `to_qiskit` converts sympy Symbols → Qiskit `Parameter` objects via `_sym_expr_to_qk`; `qir2qiskit` extracts numeric floats via `_get_float`.  Additional divergences: gate-name key format, handled gate set (qir2qiskit adds exp/measure/reset/barrier with backend-coupled ops).  `_translate_qiskit_params` in `translation.py` is the reverse direction (Qiskit → tc) and unrelated.
