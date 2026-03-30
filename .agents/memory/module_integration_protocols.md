# Module Integration Protocols

1.  **Exporting and Aliasing**:
    - Export new modules in `tensorcircuit/__init__.py`.
    - Provide both the module export (e.g. `from . import pauliprop`) and the primary helper function (e.g. `from .pauliprop import pauli_propagation`).
    - Use concise aliases for frequently used functions (e.g., `PauliProp = pauli_propagation`).

2.  **Backwards Compatibility in Helpers**:
    - High-level wrapper functions should support multiple input formats (e.g., both raw arrays and convenient list-of-tuples for observables) to be user-friendly while maintaining internal efficiency.

3.  **Standardized Testing Patterns**:
    - **Backend Isolation**: Avoid global `tc.set_backend()` in test files. Use the standard fixtures (`npb`, `tfb`, `jaxb`) from `conftest.py` as function arguments.
    - **Test Levels**:
      - **Unit**: Test initialization, state mapping, and single-gate kernels.
      - **Correctness**: Compare results against `tc.Circuit.expectation` or `expectation_ps` for small $N$.
      - **AD/Gradients**: Verify that `jax.grad` (or backend equivalent) works on the module's primary interfaces.
      - **Scanning**: Verify that loop-optimization interfaces (e.g. `compute_expectation_scan`) match manual application results.

4.  **Documentation & Linting**:
    - Achieve **10/10 pylint score** and pass `mypy` before finalizing a module.
    - Follow Google-style docstrings with reStructuredText markers. This is critical for automated documentation generation.

5.  **User Verification (Walkthroughs)**:
    - Always provide a production-ready example in `examples/` (e.g., `pauli_propagation_vqe.py`) that showcases a real-world use case (optimization, dynamics, etc.) and demonstrates performance features like JAX JIT and Scanning.
