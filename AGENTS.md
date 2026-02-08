# TensorCircuit-NG Repository Guide for AI Agents

## Core Philosophy

TensorCircuit is a **Tensor Network-first**, **Multi-Backend** quantum computing framework. When contributing to this codebase, you must adhere to the following architectural principles:

### Unified Backend Interface

- **Rule**: Never use backend-specific libraries (numpy, tensorflow, jax) directly in core logic.
- **Pattern**: Use the `tc.backend` abstraction for all tensor operations.
- **Example**: Use `tc.backend.sin(x)` instead of `np.sin(x)` or `tf.math.sin(x)`. This ensures code runs seamlessly on TensorFlow, JAX, PyTorch, and NumPy.

### Differentiable Programming (AD)

- **Rule**: All core components must be differentiable.
- **Pattern**: Avoid operations that break the computation graph (e.g., converting to numpy inside a differentiable function, using in-place assignments on tensors).
- **Goal**: End-to-end differentiability allows variational algorithms and gradient-based optimization to work out of the box.

### JIT-First Optimization

- **Rule**: Write code that is JIT-compilable (Just-In-Time).
- **Pattern**: Avoid Python control flow (if/else) that depends on tensor values. Use `tc.backend.cond` or `tc.backend.switch` if necessary, or structure code to be statically analyzable.
- **Benefit**: This enables massive speedups on JAX and TensorFlow backends.

## Repository Structure

- `tensorcircuit/`: Core package source code.
  - `backends/`: Backend implementations (avoid modifying unless necessary).
  - `templates/`: High-level modules (ansatzes, Hamiltonians, graphs).
- `examples/`: Usage demos and benchmarks. Use these as reference implementations.
- `tests/`: Comprehensive test suite. Check these for expected behavior.
- `docs/`: Sphinx documentation source.

## Configuration and Dependencies

### Core Dependencies

- numpy
- scipy
- tensornetwork-ng
- networkx

### Backend Dependencies (Optional)

- tensorflow
- jax
- jaxlib
- torch
- qiskit
- sympy
- symengine
- mthree

### Development Dependencies

- mypy
- pytest
- black (with jupyter support)
- pylint
- sphinx (>=4.0)

### Configuration Files

1. `requirements/` - Contains various requirement files:
   - [requirements.txt](requirements/requirements.txt) - Core dependencies
   - [requirements-dev.txt](requirements/requirements-dev.txt) - Development tools
   - [requirements-extra.txt](requirements/requirements-extra.txt) - Optional dependencies
   - [requirements-types.txt](requirements/requirements-types.txt) - Type checking dependencies

2. [pyproject.toml](pyproject.toml) - Build system configuration with mypy and pytest settings

3. `.pylintrc` - Code style enforcement with specific rules enabled

## AI Agent Best Practices

### Code Navigation

- **Search First**: The codebase is extensive. Search for class definitions (e.g., `class Hamiltonian`) rather than guessing file paths.
- **Check Tests**: `tests/test_*.py` files are the ultimate source of truth for how APIs are intended to be used.

### Coding Standards

- **Linting**: We enforce strict **Pylint** and **Black** formatting.
  - Run `bash check_all.sh` before submitting changes.
  - Target Pylint score: 10.0/10.
- **Type Hinting**: Use type hints liberally to aid static analysis.
- **Documentation**: Write clear docstrings (reStructuredText format) for all public APIs.

### Coding Style Suggestions

Follow these rules for all code changes in this repository:

- Minimize comments; be concise; code should be self-explanatory and self-documenting.
- Comments should be useful, for example, comments that remind the reader about some global context that is non-obvious and can't be inferred locally.
- Match existing code style and architectural patterns of the codebase.
- If uncertain, choose the simpler, more concise implementation.

### Common Workflows

#### 1. Running Tests

```bash
# Run all tests (auto-parallelized)
pytest -n auto

# Run specific test file (useful during debugging)
pytest tests/test_hamiltonians.py
```

#### 2. Checking Code Quality

```bash
# Check formatting
black . --check

# Run linter on specific file
pylint tensorcircuit/quantum.py
```

### Common Patterns in the Codebase

- Backend-agnostic operations through the tc.backend interface
- JIT compilation support via tc.backend.jit(), JIT is prefered for performance
- Automatic differentiation support via tc.backend.grad()
- Vectorized operations using tc.backend.vmap patterns
- Context managers for temporary configuration changes

### Branch Strategy

- master branch for stable releases
- beta branch for nightly builds (as seen in nightly_release.yml)
- pull requests for feature development

### Test Structure

- Tests located in /tests/ directory
- Example scripts in /examples/ directory also serve as integration tests
- CI runs example demos to ensure functionality

### Documentation

- Documentation built with Sphinx
- Both English and Chinese versions generated
- Located in /docs/ directory

### Package Distribution

- Distributed as tensorcircuit-ng package in PyPI
- Supports extra dependencies for specific backends (tensorflow, jax, torch, qiskit, cloud)

## Further Reading

- **Specific Protocols**: See `llm_experience.md` for detailed protocols on development, profiling, and performance tuning.

- **Official Docs**: https://tensorcircuit-ng.readthedocs.io/

### AI-Native Documentation Services

- Devin Deepwiki: https://deepwiki.com/tensorcircuit/tensorcircuit-ng
- Context7 MCP: https://context7.com/tensorcircuit/tensorcircuit-ng
