# TensorCircuit-NG Repository Guide for AI Agents

## Mission

TensorCircuit is a tensor-network-first, multi-backend quantum computing framework. Optimize for backend-agnostic, differentiable, JIT-friendly code changes that match existing repository style.

## Non-Negotiable Rules

- Use `tc.backend` for core tensor operations. Do not call backend-specific APIs such as NumPy, JAX, or TensorFlow directly in core logic.
- Prefer backend-native abstractions already used in the repo, including `tc.backend.jit`, `tc.backend.grad`, `tc.backend.vmap`, and backend control-flow helpers when applicable.
- Preserve differentiability. Avoid graph-breaking conversions, in-place tensor mutation, and other patterns that block autodiff.
- Preserve JIT compatibility. Avoid Python control flow that depends on tensor values; prefer backend control-flow helpers or static structure.
- Keep changes minimal and consistent with existing architecture.
- Prefer simple, direct implementations. Avoid defensive complexity and broad `try...except` blocks.
- Fail fast. Expose real problems early instead of masking them with silent fallbacks or workaround-heavy logic.
- Do not cheat around repository invariants, tests, or framework behavior just to make a local change appear to pass.
- Focus only on git-tracked files. Do not search for or modify files in `.gitignore` unless explicitly specified by the user.

## Environment Rules

- Never install packages into system or user Python unless the user explicitly asks.
- If a command fails because of missing Python packages or `ModuleNotFoundError`, ask the user which environment to use.
- Once the environment is known, run Python tooling through that environment, for example `conda run -n <env> ...`.
- Dependency and tool configuration lives in `requirements/`, `pyproject.toml`, and `.pylintrc`.

## Where To Look First

- Search before guessing file locations.
- Treat `tests/test_*.py` as the source of truth for intended behavior.
- Core library code lives in `tensorcircuit/`.
- Examples in `examples/` are useful reference implementations.
- Documentation sources live in `docs/`.

## Coding Rules

- Import Placement: All imports — internal (`tensorcircuit.*`) and any third-party package listed in `pyproject.toml` — MUST appear at the top of the file. Only third-party packages that are NOT declared as must dependencies in `pyproject.toml` (e.g. `stim`, `qiskit`) should be imported inside a function body (lazy/optional import pattern). In `examples/`, every dependecy should be imported at the top of file`
- Match existing naming, structure, and API patterns.
- Keep comments minimal and useful; avoid explanatory debugging commentary.
- Use type hints and static analysis in the `tensorcircuit/` module.
- Write clear public docstrings when changing public APIs.
- Backend-agnostic, autodiff-friendly, and JIT-friendly patterns are preferred throughout the codebase.

## Testing Rules

- Use fixtures from `tests/conftest.py`.
- In tests, never call `tc.set_backend()` or `tc.set_dtype()` directly.
- Use backend fixtures such as `npb`, `tfb`, `jaxb`, `torchb`, and `cpb` instead of manual backend switching.
- Use the `highp` fixture when a test requires `complex128` precision.
- Prefer `np.testing.assert_allclose` for numeric comparisons instead of manual `abs(a - b) < tol` assertions. Use exact array equality helpers only when exact equality is the real intent.
- Prefer targeted tests first, then broader validation as needed.
- Use `pytest -n auto` to accelerate when broader test execution is needed and the environment supports it.
- For code quality, follow existing `black` and `pylint` expectations in `check_all.sh`.
- `.pylintrc` is the source of truth for linter behavior.
- Run `bash check_all.sh` before submitting substantial code changes when the environment is available.

## Known Issue

- `tests/test_circuit.py::test_qiskit2tc` can fail intermittently because of non-deterministic behavior in Qiskit's `UnitaryGate.control()` path. If this is the only failure, treat it as a likely upstream flake rather than a TensorCircuit regression.

## Further Reading

- Progressive Memory Disclosure: review `.agents/memory/index.md` first, then load only the relevant memory files.
- If you discover a durable, non-obvious lesson for tensorcircuit-ng project, record it in the appropriate memory file.
