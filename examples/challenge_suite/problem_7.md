# Problem 7: 50-Qubit Two-Excitation VQE

## Goal

Optimize a particle-number-preserving variational circuit for a 50-qubit hard-core-boson Hamiltonian while staying entirely inside the two-excitation subspace. The full Hilbert space has dimension `2^50`, but the computation should use the `C(50, 2) = 1225` dimensional symmetry sector.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 50,
    "n_particles": 2,
    "initial_occupied": [16, 33],
    "interaction": 0.7,
    "n_layers": 20,
    "initial_parameter_scale": 0.01,
    "max_steps": 500,
    "learning_rate": 0.01,
}
```

Use the open-boundary hard-core-boson Hamiltonian

$$H = -\sum_{i=0}^{48}(\sigma_i^+\sigma_{i+1}^- + \sigma_i^-\sigma_{i+1}^+) + 0.7\sum_{i=0}^{48} n_i n_{i+1} + \sum_{i=0}^{49}\mu_i n_i,$$

where `n_i = |1><1|_i` and

$$\mu_i = 0.25 \cos(2\pi i / 50) + 0.10(-1)^i.$$

### Ansatz Structure

Start from the computational-basis state with exactly two excitations at sites `16` and `33`. Use 20 brickwork layers, equivalently 10 repeated even+odd blocks. In each block, first apply the even-bond layer on bonds `(0,1), (2,3), ..., (48,49)`, then apply the odd-bond layer on bonds `(1,2), (3,4), ..., (47,48)`. Each active bond `(i, i+1)` has three independent trainable real parameters `(theta, alpha, beta)` and applies the number-preserving gate `G_i(theta, alpha, beta) = RZ_i(alpha) RZ_{i+1}(beta) ISWAP_{i,i+1}(theta)`, with gates applied from right to left in this expression.

The single-site phase is `RZ(phi) = exp(-i phi Z / 2)`. The mixing gate is the identity on `|00>` and `|11>`, and acts on the ordered subspace `(|01>, |10>)` as

```text
[[cos(pi theta / 2), i sin(pi theta / 2)],
 [i sin(pi theta / 2), cos(pi theta / 2)]]
```

All trainable parameters are initialized independently from a normal distribution with mean `0` and standard deviation `initial_parameter_scale`.

### Loss Function

Train all gate parameters by minimizing `<H>` with Adam at learning rate `0.01` for exactly 500 optimizer updates. Do not use early stopping, because the evaluator measures fixed-work end-to-end runtime.

## Solution Interface

The solution file must be named `solution_7.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `energy_history`: NumPy array with length equal to the number of optimizer updates.

`energy_history` records one energy value per optimizer update, evaluated immediately before applying that update. The evaluator derives initial and final energy from the first and last entries of `energy_history`. The excitation number and requested step count are fixed by the evaluator configuration and are not returned by the solution.

The solution may use any quantum software framework, but it must consume only the evaluator-provided configuration and return only this NumPy-format dictionary.

## Evaluation Interface

The evaluator file is `evaluate_7.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_7.py --solution solution_7
```

The evaluator consumes only the returned result dictionary. It builds the exact sparse Hamiltonian in the two-excitation subspace, prints the subspace dimension, full Hilbert-space dimension, initial energy, final VQE energy, exact subspace ground-state energy, final excitation-number expectation, energy-history length, steps run, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- The two-excitation subspace dimension is `1225`.
- `len(energy_history) == 500`.
- The final VQE energy is lower than the initial energy.
- The final VQE energy is no lower than the exact sparse ground-state energy beyond numerical tolerance.
- The final excitation-number expectation equals `2` to numerical tolerance `1e-12`.
- All returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these metrics directly so another framework's `solution_7.py` can be compared without changing `evaluate_7.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_7.py` can be evaluated with:

```bash
python evaluate_7.py --solution solution_7
```

Observed TensorCircuit-NG baseline with the default 500-step configuration:

- End-to-end solution time: `182.03s`.
- Initial energy: `-0.2424087254`.
- Final VQE energy: `-4.3166061360`.
- Exact subspace ground energy: `-4.3306100384`.
- Energy history length: `500`.

## Implementation Hint

For a TensorCircuit-NG/JAX baseline, use `tc.U1Circuit` with `k=2` so the state lives directly in the two-excitation sector. `U1Circuit.iswap` implements the specified `ISWAP(theta)` mixing gate and `rz` implements the specified single-site phase. Build the fixed Hamiltonian as one Pauli-string sum and evaluate it with `U1Circuit.expectation_pss`.
