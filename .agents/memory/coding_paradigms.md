# Programming Paradigms & Error Handling

TensorCircuit values **clarity**, **predictability**, and **directness** in its implementation. Avoid patterns that obscure bugs or make the execution flow hard to trace.

## 1. Avoid Defensive Over-Programming

- **Trust Internal Invariants**: Avoid re-validating the same state in every internal function. Use assertions only for truly critical, non-obvious invariants during development.
- **Data Flow Clarity**: Rely on clear, documented API contracts rather than extensive "just-in-case" checks. If a function is called incorrectly, the resulting error should be natural (e.g., `AttributeError` or `ValueError`) rather than hidden by defensive wrappers.
- **Fail Fast**: Let the program crash or raise an exception at the point of failure. This ensures that errors are caught during development and testing, rather than propagating silently into incorrect physical results.

## 2. Minimalist Exception Handling

- **Use `try...except` Sparingly**: Only catch exceptions that you can **meaningfully handle** or provide clear, actionable context for. 
- **No Broad Catching**: Never use `except Exception:` unless for very specific top-level logging or cleanup logic.
- **Explicit Reraising**: If you must catch an exception, ensure it is either handled or reraised immediately with better context (using `raise ... from e`).
- **Control Flow**: Never use exceptions for normal control flow. Use conditional statements or pattern matching instead.

## 3. Early Error Exposure

- **Direct Feedback**: If an operation is invalid (e.g., mismatched gate dimensions), raise an error immediately. 
- **Transparent Backends**: Because TensorCircuit is multi-backend, some errors may propagate differently. Structure code so that backend-specific errors (e.g., JAX tracer errors) are exposed clearly to the user, aiding in debugging JIT/AD issues.
- **Locate and Fix**: The goal is to make the root cause obvious. Masking errors with defaults or silent failures leads to "Heisenbugs" that are much harder to resolve later.
