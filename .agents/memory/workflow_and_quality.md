# Workflow and Quality

Use this file for public-feature integration, examples, tests, and validation details that are not already covered by `AGENTS.md`.

## Runtime and wrapper invariants

- `set_backend`, `set_dtype`, and `set_contractor` mutate package-wide runtime state. Any temporary switch inside a decorator, wrapper, or helper must restore the previous state on every exit path, including exceptions.
- Validate unsupported structured inputs before tracing or JIT so invalid patterns fail before a compiled specialization is cached.
- For type-checking fixes, prefer the smallest behavior-preserving change. A targeted `# type: ignore[...]` or `Any` is preferable when a more elaborate annotation would distort the runtime design.
- At thin third-party wrapper boundaries, handle known method-versus-attribute compatibility differences explicitly rather than spreading version checks through core logic.

## Public features and optional integrations

- A new top-level peer API normally needs its package export and convenience alias, a realistic example, the docs index under `docs/source/`, and a changelog entry.
- Keep optional subsystems lazy at the package boundary so importing `tensorcircuit` does not require their extras; gate their tests with `pytest.importorskip(...)`.
- Demonstrate optional contractor integrations through the real public entrypoint, such as `tc.set_contractor("custom", optimizer=...)`; keep search-only benchmarks separate from end-to-end contraction examples.
- Library plotting helpers should accept an optional `ax` and should not call `plt.show()` internally.

## Example design

- Treat `examples/` as public reference implementations: present one concrete workflow, expose only parameters that change its scientific or algorithmic point, and inline one-use scaffolding that obscures the main pattern.
- Keep examples on the public API and fail fast on correctness mismatches. Move exploratory sweeps, verbose diagnostics, and incidental benchmark controls out of the pedagogical script.
- After exploration establishes a stable pattern, distill it into a short example instead of preserving the full experiment harness.

## Testing and validation refinements

- For a differentiable feature, cover the core kernel, a correctness comparison against an existing reference path, and its AD behavior.
- Compare sparse results numerically rather than depending on a particular sparse container layout.
- Wrapper smoke tests should assert behavior against the underlying backend API; never suppress adapter failures with `except Exception: pass`.
- Match algorithms, precision, and settings before comparing external libraries, and verify numerical outputs before interpreting timings.
- Sanity-check automation should favor high-confidence, repo-aware rules: scan git-tracked Python, separate test bootstrap conventions from library import hygiene, and reserve subjective duplication or comment-quality judgments for review.
