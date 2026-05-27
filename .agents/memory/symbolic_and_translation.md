# Symbolic and Translation

Use this file for `SymbolCircuit`, SymPy-to-JAX, and translation or interoperability boundaries.

## SymbolCircuit rules

- Build `SymbolCircuit` objects before switching to the JAX backend. They rely on object-dtype symbolic storage that JAX arrays do not accept.
- Declare symbolic gate parameters with `real=True` when the expressions are meant to remain JAX-lambdifiable after conjugation.
- `to_circuit(...)` must bind every free symbol used by the concrete expression, including data-input symbols and not just trainable parameters.

## Lambdify to JAX

- Use `sympy.lambdify(..., modules=[jnp, "math"])`; passing the string `"jax.numpy"` does not work.
- Build the argument list from `expr.free_symbols`, not from a larger circuit-level symbol set, or the generated callable may have the wrong signature.
- The resulting function is JAX-native and can be composed with `jit`, `grad`, and `vmap` once the symbol set is correct.

## Qiskit translation boundaries

- `to_qiskit` and `translation.qir2qiskit` solve different problems and should not be forced into one dispatch path.
- `to_qiskit` preserves symbolic parameters, while `qir2qiskit` reconstructs numeric gates from serialized QIR.
- `_translate_qiskit_params` is the reverse-direction helper for Qiskit-to-TensorCircuit translation.

## Stim translation

- Keep `stim` imports optional and gate translation tests with `pytest.importorskip("stim")`.
- Implement basis-changed Stim measurements and resets by explicit basis transforms around the existing Z-basis measurement or reset path instead of inventing separate semantics for each variant.
- Preserve instruction arity when mapping noisy channels; multi-qubit noise should stay multi-qubit during translation instead of falling back to one-qubit helpers.
