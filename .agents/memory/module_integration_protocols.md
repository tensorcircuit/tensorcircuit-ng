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
    - **Docstring Standard**: Use reStructuredText format with explicitly typed parameters and return values:
      - `:param <name>: Description`
      - `:type <name>: <type_hint>`
      - `:return: Description`
      - `:rtype: <type_hint>`
    - **Import Placement**: All imports — internal (`tensorcircuit.*`) and any third-party package listed in `pyproject.toml` — MUST appear at the top of the file. Only third-party packages that are **not** declared as dependencies in `pyproject.toml` (e.g. `stim`, `qiskit`) may be imported inside a function body (lazy/optional import pattern).
    - **Comment Cleanup**: Remove all "think-out-loud" development comments, debug print leftovers, and redundant Phase/Part header comments that do not add value to the end user.

5.  **User Verification (Walkthroughs)**:
    - Always provide a production-ready example in `examples/` (e.g., `pauli_propagation_vqe.py`) that showcases a real-world use case (optimization, dynamics, etc.) and demonstrates performance features like JAX JIT and Scanning.

6.  **File Add Procedure** — when adding a new module file (alongside its test file and example file), also:
    - Run `python generate_rst.py` inside `docs/source/` to regenerate the API reference `.rst` files.
    - Update `changelog.md` with a brief entry describing the new module.
