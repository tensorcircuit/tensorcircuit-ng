# Problem 5: Custom Non-Unitary Gate Cooling

## Goal

Optimize a variational cooling circuit built from trainable non-unitary filters. The task is to apply one-qubit and two-qubit exponential filters, rescale the many-body state to unit length after every cooling layer, differentiate through the rescaling, and lower the energy density of an 18-qubit transverse-field Ising Hamiltonian.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 18,
    "transverse_field": 1.10,
    "n_layers": 10,
    "initial_filter_strength": 0.01,
    "max_steps": 600,
    "learning_rate": 0.02,
}
```

Use the open-boundary transverse-field Ising Hamiltonian

$$H = -\sum_{i=0}^{16} Z_i Z_{i+1} - 1.10 \sum_{i=0}^{17} X_i.$$

Start from `|+>^18`. Use ten cooling layers indexed by `l = 0, ..., 9`. In layer `l`, apply the same trainable one-qubit filter `exp(a_l X_i)` to every qubit, then apply the same trainable two-qubit filter `exp(b_l Z_i Z_{i+1})` to the brickwork bonds of that layer. Even-numbered layers use bonds `(0,1), (2,3), ..., (16,17)`, and odd-numbered layers use bonds `(1,2), (3,4), ..., (15,16)`.

Initialize every `a_l` and every `b_l` to `initial_filter_strength = 0.01`. Rescale the full state to unit length after every cooling layer.

### Loss Function

Train the objective

$$\mathcal{L} = \langle H \rangle / 18$$

for exactly 600 Adam updates at learning rate `0.02`. Do not use early stopping.

## Solution Interface

The solution file must be named `solution_5.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `final_a`: NumPy array with shape `(5, 2)` containing learned one-qubit filter strengths, where each row is one even+odd block and the two columns are the even and odd sublayers.
- `final_b`: NumPy array with shape `(5, 2)` containing learned two-qubit filter strengths, with the same block and sublayer layout.
- `energy_density_history`: NumPy array with length `config["max_steps"]`.

`energy_density_history` records one value per optimizer update, evaluated immediately before applying that update. The evaluator derives initial and final energy density from the first and last entries of `energy_density_history`.

The solution may use any quantum software framework, but it must consume only the evaluator-provided configuration and return only this NumPy-format dictionary.

## Evaluation Interface

The evaluator file is `evaluate_5.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_5.py --solution solution_5
```

The evaluator consumes only the returned result dictionary. It computes the exact 18-qubit sparse ground-state energy independently, then prints initial and final energy density, exact ground-state energy density, learned filter strengths, energy-history length, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default 600-step configuration:

- `len(energy_density_history) == 600`.
- `final_a.shape == (5, 2)` and `final_b.shape == (5, 2)`.
- The final energy density is lower than the initial energy density.
- The final energy density is no lower than the exact sparse ground-state energy density beyond numerical tolerance.
- All returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these metrics directly so another framework's `solution_5.py` can be compared without changing `evaluate_5.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_5.py` can be evaluated with:

```bash
python evaluate_5.py --solution solution_5
```

Observed TensorCircuit-NG baseline with the default 600-step configuration:

- End-to-end solution time: `54.55s`.
- Exact sparse ground energy density: `-1.3269012239`.
- Initial energy density: `-1.1720404625`.
- Final energy density: `-1.3267327547`.

## Implementation Hint

For a TensorCircuit-NG/JAX baseline, `exp(a X)` and `exp(b Z Z)` can be implemented with ordinary `Circuit.rx` and `Circuit.rzz` calls by passing imaginary angles. TensorCircuit's `RX/RZZ` gates use the standard half-angle convention, so `theta=2.0j * strength` implements the stated filters. Treat one even+odd pair as the repeated scan unit, and evaluate the fixed TFIM Hamiltonian with a matrix-free MVP followed by `<psi|Hpsi>`.
