# Pauli Propagation & Operator Evolution (Heisenberg Picture)

1.  **Heisenberg Picture Reverse Order**:
    - Pauli Propagation evolves the _observable_ rather than the state.
    - **Protocol**: Circuit operations must be applied in **reversed** chronological order (from the measurement gate back to the initial layer). This corresponds to applying the adjoint gate to the operator: $O \to U^\dagger O U$.

2.  **JAX Tracer Accumulation**:
    - When initializing an operator state from a Hamiltonian where weights are JAX Tracers (e.g., in VQE optimization), direct indexing and assignment (`state[idx] = w`) will fail with `TracerArrayConversionError`.
    - **Protocol**: Use the `at[].add()` or `at[].set()` functional update syntax for compatibility with JIT and AD.
      ```python
      state = state.at[target_idx, flat_idx].add(w)
      ```

3.  **Real-Valued Expectations for Gradients**:
    - JAX gradients of loss functions often require the output to be a real-valued scalar. Even if the physics dictates a real expectation value, numerical complex types (even with zero imaginary part) can trigger `TypeError`.
    - **Protocol**: Always explicitly take the real part using `K.real()` or `.real` before returning the expectation value from a loss function or engine method.
