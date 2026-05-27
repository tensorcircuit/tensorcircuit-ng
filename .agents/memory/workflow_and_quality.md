# Workflow and Quality

Use this file for changes that touch public API, tests, docs, or code quality.

## Design defaults

- Prefer direct implementations over defensive wrappers. Fail fast when an invariant is violated instead of masking the root cause.
- Catch exceptions only when adding real context or recovery. Do not use broad exception handling as control flow.
- `set_backend`, `set_dtype`, and `set_contractor` mutate package-wide runtime state. Any temporary switch inside decorators, wrappers, or helpers must restore the previous state on every exit path, including exceptions.
- Validate unsupported structured inputs before tracing or JIT so invalid patterns fail early and do not poison later compiled runs.
- For type-checking fixes, prefer the smallest behavior-preserving change. Use targeted `# type: ignore[...]` or `Any` when the alternative would distort the runtime design.
- For thin wrappers over third-party APIs, guard known method-vs-attribute compatibility differences at the wrapper boundary and keep the fallback explicit.

## Public module changes

- Export new public modules from `tensorcircuit/__init__.py`, including the main convenience alias when the repo already follows that pattern.
- Keep public docstrings and examples aligned with the actual API. When a new module is user-facing, add or update a realistic example in `examples/`.
- For optional contractor integrations, keep user-facing examples on the actual public entrypoint (`tc.set_contractor("custom", optimizer=...)`) and keep search-only benchmark scripts separate from end-to-end contraction demos.
- When adding a new tracked module, also refresh the docs index from `docs/source/` and add a short changelog entry.
- Keep imports at the top of the file unless the dependency is intentionally optional and not part of the core declared requirements.
- Keep top-level package imports resilient when optional extras are missing. Prefer lazy package-boundary imports for optional subsystems, and gate optional-dependency tests with `pytest.importorskip(...)`.

## Example-script style

- Treat `examples/` as public reference material for both users and coding agents. Optimize for top-to-bottom readability and reuse of the main pattern, not for exhaustive configurability.
- Prefer one concrete workflow per example. If a helper class, dataclass, or abstraction is only used once inside the script and does not clarify the main idea, inline it.
- Expose only the flags that materially change the scientific or algorithmic point of the example. Hardcode incidental benchmarking knobs, validation harness details, and local execution scaffolding.
- Keep examples on the real public API surface rather than private compatibility hacks, unless the example is explicitly about internal infrastructure.
- In examples, fail fast on correctness mismatches instead of printing large debugging reports. The output should focus on the comparison or usage pattern the example is meant to teach.
- Split exploratory benchmarking scripts from clean pedagogical examples. Once a conclusion is known, distill the stable pattern into a shorter example rather than preserving the full exploration scaffold.

## Testing defaults

- In tests, use backend fixtures from `tests/conftest.py`; do not switch backend or dtype globally inside test bodies.
- Cover at least the core kernel, a correctness path against an existing reference API, and AD behavior when the feature is meant to be differentiable.
- For sparse outputs, compare results numerically instead of assuming a specific sparse container layout.
- Do not hide wrapper smoke tests behind `except Exception: pass`; assert the wrapped behavior directly against the underlying backend API so adapter bugs surface.
- Use `pytest -n auto` when available for broader runs, and use `bash check_all.sh` before landing substantial changes.

## Tooling and validation

- Benchmark external libraries in isolated environments and match precision/settings before making performance claims.
- Library plotting helpers should accept an optional `ax` and should not call `plt.show()` internally.
- In read-only or sandboxed environments, redirect caches such as `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`, and mypy's cache directory in the shell command or local harness used for validation, not in tracked source files, unless the repo explicitly needs that behavior.
- If a test or import failure plausibly comes from sandbox restrictions or third-party runtime cache/process-pool setup, rerun the exact command with escalated permissions before attributing the failure to TensorCircuit logic; sandbox artifacts can mask the real failure mode.
- On macOS sandboxed runs, `conda run -n <env> ...` can silently resolve to the base interpreter; verify `sys.executable` or call the env's Python directly when reproducing environment-sensitive issues.
- Sanity-check automation should stay high-confidence and repo-aware: scan only git-tracked Python files, reserve subtle duplication/comment-quality judgments for LLM review, keep test-specific rules separate from library import-hygiene rules to avoid noise from test bootstrap patterns, and only flag optional-import hygiene when package import failures actually point to an optional dependency rather than a missing required package.
