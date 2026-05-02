# Workflow and Quality

Use this file for changes that touch public API, tests, docs, or code quality.

## Design defaults

- Prefer direct implementations over defensive wrappers. Fail fast when an invariant is violated instead of masking the root cause.
- Catch exceptions only when adding real context or recovery. Do not use broad exception handling as control flow.
- Validate unsupported structured inputs before tracing or JIT so invalid patterns fail early and do not poison later compiled runs.
- For type-checking fixes, prefer the smallest behavior-preserving change. Use targeted `# type: ignore[...]` or `Any` when the alternative would distort the runtime design.

## Public module changes

- Export new public modules from `tensorcircuit/__init__.py`, including the main convenience alias when the repo already follows that pattern.
- Keep public docstrings and examples aligned with the actual API. When a new module is user-facing, add or update a realistic example in `examples/`.
- When adding a new tracked module, also refresh the docs index from `docs/source/` and add a short changelog entry.
- Keep imports at the top of the file unless the dependency is intentionally optional and not part of the core declared requirements.

## Testing defaults

- In tests, use backend fixtures from `tests/conftest.py`; do not switch backend or dtype globally inside test bodies.
- Cover at least the core kernel, a correctness path against an existing reference API, and AD behavior when the feature is meant to be differentiable.
- For sparse outputs, compare results numerically instead of assuming a specific sparse container layout.
- Use `pytest -n auto` when available for broader runs, and use `bash check_all.sh` before landing substantial changes.

## Benchmarking and tooling

- Benchmark external libraries in isolated environments and match precision/settings before making performance claims.
- Library plotting helpers should accept an optional `ax` and should not call `plt.show()` internally.
- In read-only or sandboxed environments, redirect caches such as `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`, and mypy's cache directory to writable temporary locations.
