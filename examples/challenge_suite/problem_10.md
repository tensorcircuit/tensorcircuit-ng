# Problem 10: 22-Qubit VQE With an 18-Qubit Controlled-Z Hyperedge

## Goal

Optimize a variational state whose ansatz contains a fixed 18-qubit controlled-Z gate acting on non-adjacent qubits. The task tests whether a framework can express and differentiate a large high-arity diagonal gate directly, without materializing a dense `2^18 x 2^18` matrix or decomposing the gate into a long sequence of small controlled gates.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 22,
    "selected_qubits": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    "initial_ones": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    "n_layers": 2,
    "max_steps": 200,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.08,
    "zz_strength": 1.0,
    "x_strength": 1.05,
    "seed": 2040,
    "minimum_energy_improvement": 1e-3,
    "exact_tol": 1e-7,
    "exact_maxiter": 400,
    "exact_ncv": 24,
    "exact_lower_bound_tolerance": 1e-5,
    "maximum_energy_density_gap": 0.25,
}
```

Use the ordinary open-boundary transverse-field Ising Hamiltonian

```text
H = -zz_strength sum_{i=0}^{20} Z_i Z_{i+1}
    -x_strength sum_{i=0}^{21} X_i .
```

The selected-qubit controlled-Z operation multiplies a computational-basis amplitude by `-1` exactly when all 18 selected qubits are in state `1`, and leaves every other computational-basis amplitude unchanged. This operation is part of the ansatz only; it is not folded into the Hamiltonian.

### Ansatz Structure

Start from the computational-basis Neel state with qubits listed in `initial_ones` set to `1`. Use `n_layers` variational layers. Each layer applies a trainable `RX-RZ-RY` block on every qubit and then applies the fixed 18-qubit controlled-Z operation on `selected_qubits`. Thus the repeated structure is `(rotation block -> controlled-Z)`.

Initialize all trainable rotation angles independently from a Gaussian distribution with mean `0` and standard deviation `initial_parameter_scale`, using the configured seed. Optimize all rotation angles with Adam at the configured learning rate for exactly `max_steps` updates.

### Energy And Reference

Evaluate the Hamiltonian expectation by matrix-vector product rather than constructing a dense Hamiltonian matrix. The evaluator independently computes a strict sparse-Lanczos ground-state reference for the same TFIM Hamiltonian and reports the VQE energy-density gap.

## Solution Interface

The solution file must be named `solution_10.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only NumPy-format quantities that the evaluator checks or reports.

Required result keys:

- `energy_history`: NumPy array with shape `(max_steps,)`.
- `final_parameters`: final rotation tensor with shape `(n_layers, n_qubits, 3)`.

Each history records one value per optimizer update, evaluated immediately before applying that update. The evaluator derives initial/final energy density from the first/last entries of `energy_history`. The largest gate size and requested step count are fixed by the evaluator configuration and are not returned by the solution.

## Evaluation Interface

The evaluator file is `evaluate_10.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_10.py --solution solution_10
```

The evaluator consumes only the returned result dictionary. It prints solution time, exact-reference time, initial and final energy densities, exact ground-state energy density, VQE gap, energy-history length, largest gate size, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `energy_history.shape == (max_steps,)`.
- `final_parameters.shape == (n_layers, n_qubits, 3)`.
- The evaluator-computed largest ansatz gate size is `18`.
- The final energy density is at least `minimum_energy_improvement` lower than the initial energy density.
- The returned history is finite.
- The final energy density is not below the exact ground-state energy density beyond `exact_lower_bound_tolerance`.
- The final energy-density gap relative to the exact reference is at most `maximum_energy_density_gap`. This threshold is intentionally loose because the benchmark is about expressing and optimizing through the large `CMZ` hyperedge, not about solving the TFIM ground state to high precision.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_10.py` can be evaluated with:

```bash
python evaluate_10.py --solution solution_10
```

Observed TensorCircuit-NG/JAX baseline in the current validation environment with the default 200-step configuration:

- Solution time: `66.30s`.
- Exact reference time: `30.69s`.
- Initial energy density: `0.9718770385`.
- Final energy density: `-1.1766326427`.
- Exact ground energy density: `-1.2925285569`.
- VQE energy-density gap: `0.1158959142`.
- Energy improvement: `2.1485096812`.

## Implementation Hint

Use the built-in `c.cmz(*selected_qubits)` gate in the ansatz. Use `tc.quantum.PauliStringSum2MVP` for the TFIM Hamiltonian, and scan over repeated variational layers so the repeated circuit structure is expressed as a tensor program. A TensorCircuit-NG baseline may configure an OMECO path searcher internally for the hyperedge contractions; this is an implementation detail and is not part of the framework-neutral evaluator configuration.
