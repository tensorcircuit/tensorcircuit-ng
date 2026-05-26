# Workflow and Quality

Use this file for changes that touch public API, tests, docs, or code quality.

## Design defaults

- Prefer direct implementations over defensive wrappers. Fail fast when an invariant is violated instead of masking the root cause.
- Catch exceptions only when adding real context or recovery. Do not use broad exception handling as control flow.
- Validate unsupported structured inputs before tracing or JIT so invalid patterns fail early and do not poison later compiled runs.
- For type-checking fixes, prefer the smallest behavior-preserving change. Use targeted `# type: ignore[...]` or `Any` when the alternative would distort the runtime design.
- For thin wrappers over third-party APIs, guard known method-vs-attribute compatibility differences at the wrapper boundary and keep the fallback explicit.

## Public module changes

- Export new public modules from `tensorcircuit/__init__.py`, including the main convenience alias when the repo already follows that pattern.
- Keep public docstrings and examples aligned with the actual API. When a new module is user-facing, add or update a realistic example in `examples/`.
- For optional contractor integrations, keep user-facing examples on the actual public entrypoint (`tc.set_contractor("custom", optimizer=...)`) and keep search-only benchmark scripts separate from end-to-end contraction demos.
- When adding a new tracked module, also refresh the docs index from `docs/source/` and add a short changelog entry.
- Keep imports at the top of the file unless the dependency is intentionally optional and not part of the core declared requirements.

## Testing defaults

- In tests, use backend fixtures from `tests/conftest.py`; do not switch backend or dtype globally inside test bodies.
- Cover at least the core kernel, a correctness path against an existing reference API, and AD behavior when the feature is meant to be differentiable.
- For sparse outputs, compare results numerically instead of assuming a specific sparse container layout.
- Do not hide wrapper smoke tests behind `except Exception: pass`; assert the wrapped behavior directly against the underlying backend API so adapter bugs surface.
- Use `pytest -n auto` when available for broader runs, and use `bash check_all.sh` before landing substantial changes.

## Benchmarking and tooling

- Benchmark external libraries in isolated environments and match precision/settings before making performance claims.
- Library plotting helpers should accept an optional `ax` and should not call `plt.show()` internally.
- In read-only or sandboxed environments, redirect caches such as `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`, and mypy's cache directory in the shell command or local harness used for validation, not in tracked source files, unless the repo explicitly needs that behavior.
- If `cotengra` fails at import time with `ImportError: cannot import name 'get_namespace' from autoray`, treat it as an environment dependency mismatch and upgrade `autoray` before debugging TensorCircuit's contraction code.
- On macOS sandboxed runs, `conda run -n <env> ...` can silently resolve to the base interpreter; verify `sys.executable` or call the env's Python directly when reproducing environment-sensitive issues.
- Cotengra hyper-optimization that reaches `joblib`/`loky` process-pool setup can fail inside the sandbox with `PermissionError` from `os.sysconf("SC_SEM_NSEMS_MAX")`; for end-to-end benchmark validation, rerun the exact benchmark command with escalated permissions rather than changing TensorCircuit logic.
- For cotengra-heavy benchmarks on this repo, prefer running directly with escalated permissions rather than sandboxed dry-runs; sandbox failures can distort conclusions about search performance or correctness.
- When comparing contraction-search strategies, keep the search budget comparable across methods unless the benchmark explicitly studies budget scaling; mismatched `max_repeats` can make runtime and quality comparisons misleading.
- Prefer `parallel="auto"` as the default cotengra benchmark mode for realistic end-to-end comparisons; forcing serial search can make healthy runs look unexpectedly slow.
- When benchmarking a searched contraction path, search once and reuse that exact tree/path for both reported FLOPs-write metrics and timed execution; do not run a second optimizer search during contraction timing.
- For cotengra path benchmarking, force reused timed runs to be cache-backed only (for example `ReusableHyperOptimizer(directory=True, cache_only=True)` plus an explicit cached path) so the timed contraction cannot silently re-search a different tree.
- `tc_combo_default`-style unseeded combo searches are fine for realism, but any logged cost metrics must be tied to the exact reused path because repeated combo searches are not deterministic.
- In internal 1D amplitude-network path-search studies, explicit simulated annealing with larger repeat budgets materially improved the reusable search frontier over the default TensorCircuit combo shortcut; use that as heuristic guidance, not as a guarantee of universal runtime wins.
- Current cotengra heuristics appear family-sensitive: ladder-like 1D amplitude networks can benefit strongly from combo-seeded reconfiguration, while brickwork-like networks often improve more modestly and may favor annealed combo-style FLOPs-write compromises.
- In benchmark and example scripts, define one brickwork layer as a full even-plus-odd nearest-neighbor round when comparing against ladder layers; counting a single parity sweep as one layer makes cross-family parameter counts and runtimes misleading.
- After normalizing brickwork to full even-plus-odd rounds, `n=30, layers=25` gives matched parameter counts (`725`) for ladder and brickwork, but brickwork contraction remains substantially harder, so topology still dominates after gate-count normalization.
- In TensorCircuit amplitude benchmarks, lower write can correlate with better runtime even when FLOPs are slightly worse, so contraction quality should be judged by timed execution together with FLOPs/write/width rather than FLOPs alone.
- Do not assume cotengra FLOP estimates alone predict runtime ranking for TensorCircuit amplitude benchmarks; inspect write/width and actual timed contractions together, especially when two strategies land on similar FLOPs.
- Sanity-check automation should stay high-confidence and repo-aware: scan only git-tracked Python files, reserve subtle duplication/comment-quality judgments for LLM review, keep test-specific rules separate from library import-hygiene rules to avoid noise from test bootstrap patterns, and only flag optional-import hygiene when package import failures actually point to an optional dependency rather than a missing required package.
