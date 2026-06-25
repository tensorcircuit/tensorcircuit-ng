# Problem 1: DMRG-MPS Input With Variational Circuit Refinement

## Goal

Implement a variational quantum circuit whose input state is a DMRG-generated matrix-product-state approximation to the ground state of a 1D TFIM Hamiltonian. The solution must use that MPS as the initial state of a regular circuit-like variational computation, optimize a shallow circuit on top of it, and return NumPy-format results for evaluator-side validation.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 32,
    "field": 1.05,
    "dmrg_chi": 8,
    "dmrg_sweeps": 2,
    "n_layers": 4,
    "max_steps": 500,
    "learning_rate": 0.005,
}
```

Before timing `run_solution(config)`, the evaluator runs DMRG itself and augments this dictionary with:

- `dmrg_state`: the normalized quimb MPS DMRG state.
- `dmrg_energy`: the corresponding DMRG energy.

These runtime keys are intentionally supplied by the evaluator so the DMRG preparation time is excluded from solution timing and every solution optimizes from the same deterministic MPS input.

The Hamiltonian is the open-boundary transverse-field Ising model

$$
H = -\sum_{i=0}^{30} Z_i Z_{i+1} - 1.05 \sum_{i=0}^{31} X_i.
$$

The variational circuit has four brickwork layers on top of the DMRG-MPS input state. Each layer applies trainable one-qubit $R_Z R_Y R_Z$ rotations on all qubits, followed by trainable nearest-neighbor two-qubit interactions $\exp[-i(\theta_{xx} X X + \theta_{yy} Y Y + \theta_{zz} Z Z)]$ on even bonds in even-numbered layers and odd bonds in odd-numbered layers. The optimizer is Adam with learning rate `0.005` for exactly 500 steps. Do not use early stopping.

The refinement is intended as a small correction on top of the supplied DMRG-MPS input, not as a replacement by a far-from-identity random circuit.

## Solution Interface

The solution file must be named `solution_1.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return a dictionary whose values are NumPy arrays or NumPy-compatible scalars.

Required result keys:

- `energy_history`: NumPy array with length `config["max_steps"]`.

Each history records one value per optimizer update, evaluated immediately before applying that update. The evaluator derives the initial and final variational energies from the first and last entries of `energy_history`. The solution may use any quantum software framework, but it must consume only the evaluator-provided configuration, including the supplied `dmrg_state`, and return only this NumPy-format dictionary. The solution should not run DMRG internally.

## Evaluation Interface

The evaluator file is `evaluate_1.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_1.py --solution solution_1
```

The evaluator computes the DMRG-MPS input before timing, passes that quimb MPS into the solution through `config["dmrg_state"]`, and then consumes only the returned result dictionary. It reports compact energy and timing summaries derived from the returned history together with the supplied DMRG reference.

- solution module name,
- end-to-end solution time, excluding evaluator-side DMRG preparation,
- DMRG reference energy,
- initial variational energy,
- final variational energy,
- energy improvement from circuit refinement,
- returned result keys.

## Passing Criteria

A run is considered functionally successful when the returned history has the expected shape, remains finite, and the refined circuit energy stays consistent with the evaluator-supplied DMRG reference without materially degrading it. All returned values must be NumPy arrays or NumPy-compatible scalars.

The evaluator should report these metrics directly so another framework's `solution_1.py` can be compared without changing `evaluate_1.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_1.py` can be evaluated with:

```bash
python evaluate_1.py --solution solution_1
```

Observed TensorCircuit-NG/JAX baseline in the current validation environment with the default 32-qubit configuration:

- End-to-end solution time: `27.22s`.
- DMRG reference energy: `-41.50400177`.
- Initial variational energy: `-41.50401306`.
- Final variational energy: `-41.50411606`.
- Final minus DMRG reference: `-1.14291145e-04`.
- Energy improvement from circuit refinement: `1.02996826e-04`.

This baseline includes first compilation/path setup, all 500 optimizer updates, and result materialization into NumPy arrays. It excludes evaluator-side DMRG state generation.

## Implementation Hint

An efficient TensorCircuit-NG implementation can convert the evaluator-provided quimb MPS with `tc.quantum.quimb2qop(config["dmrg_state"])`, inject it into a regular circuit input, build the four-layer variational circuit on top, construct the fixed 63 TFIM Pauli terms inside the solution from the Hamiltonian definition, and evaluate them through `vmap(parameterized_measurements)`. OMECO can be used as the tensor-network contraction planner. The problem itself does not require a specific implementation strategy.
