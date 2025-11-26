# TensorCircuit-NG Repository Guide for AI Agents

## Repository Overview

TensorCircuit is a high-performance unified quantum computing framework designed for the NISQ (Noisy Intermediate-Scale Quantum) era. It provides a comprehensive set of tools for quantum circuit simulation with support for multiple backends including Numpy, TensorFlow, JAX, and PyTorch.

## Documentation and AI Native Services

### Official Documentation

- Main Documentation: https://tensorcircuit-ng.readthedocs.io/
- Quick Start Guide: https://tensorcircuit-ng.readthedocs.io/en/latest/quickstart.html
- Tutorials: https://tensorcircuit-ng.readthedocs.io/en/latest/tutorial.html
- API Reference: Available in docstrings throughout the codebase

### AI-Native Documentation Services

- Devin Deepwiki: https://deepwiki.com/tensorcircuit/tensorcircuit-ng
- Context7 MCP: https://context7.com/tensorcircuit/tensorcircuit-ng

### Educational Resources

- Quantum Computing Lectures with TC-NG: https://github.com/sxzgroup/qc_lecture

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

## Common Bash Commands

### Development Checks

```bash
# Run all checks (black, mypy, pylint, pytest, sphinx)
bash check_all.sh

# Equivalent to the following individual checks:
black . --check                    # Code formatting check
mypy tensorcircuit                 # Type checking
pylint tensorcircuit tests examples/*.py  # Code linting
pytest -n auto --cov=tensorcircuit -vv -W ignore::DeprecationWarning  # Run tests

# Run all tests with coverage report
pytest --cov=tensorcircuit --cov-report=xml -svv --benchmark-skip

# Run specific test file
pytest tests/test_circuit.py

# Install dependencies
pip install --upgrade pip
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
pip install -r requirements/requirements-extra.txt
pip install -r requirements/requirements-types.txt
```

## AI Agent Best Practices

### Efficient Code Navigation

- Use the search function to find specific classes and functions rather than browsing files
- Look for example usage in the /examples/ directory when learning new features
- Check tests in /tests/ directory for detailed usage examples
- Refer to docstrings for API documentation

### Common Patterns in the Codebase

- Backend-agnostic operations through the tc.backend interface
- JIT compilation support via tc.backend.jit(), JIT is prefered for performance
- Automatic differentiation support via tc.backend.grad()
- Vectorized operations using tc.backend.vmap patterns
- Context managers for temporary configuration changes

### Working with Quantum Concepts

- Familiarity with quantum computing basics (qubits, gates, measurements)
- Understanding of tensor network concepts for advanced features
- Knowledge of different quantum computing paradigms (digital, analog, noisy, etc.)

## Additional Information

### Test Structure

- Tests located in /tests/ directory
- Example scripts in /examples/ directory also serve as integration tests
- CI runs example demos to ensure functionality

### Documentation

- Documentation built with Sphinx
- Both English and Chinese versions generated
- Located in /docs/ directory

### Package Distribution

- Distributed as tensorcircuit-ng package
- Supports extra dependencies for specific backends (tensorflow, jax, torch, qiskit, cloud)

### Core Design Principles

- Unified interface across multiple backends
- High performance through tensor network optimizations
- Extensible architecture for quantum computing research
- Compatibility with major quantum computing frameworks

### Branch Strategy

- master branch for stable releases
- beta branch for nightly builds (as seen in nightly_release.yml)
- pull requests for feature development
