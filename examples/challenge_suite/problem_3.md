# Problem 3: Probability-Aware Post-Selected Many-Body Cooling

## Goal

Optimize a 12-qubit variational cooling circuit whose dynamics include explicit post-selection in the computational basis. The objective combines the final transverse-field Ising energy density with a differentiable penalty for low post-selection probability, so the selected measurement trajectory and its probability both affect training.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 12,
    "transverse_field": 0.9,
    "n_steps": 10,
    "log_probability_weight": 0.05,
    "max_steps": 300,
    "learning_rate": 0.01,
}
```

The target Hamiltonian is the open-boundary transverse-field Ising model

$$H_{\mathrm{target}} = -\sum_{i=0}^{10} Z_i Z_{i+1} - 0.9 \sum_{i=0}^{11} X_i.$$

The initial state is `|+>^12`.

### Cooling Ansatz

Use ten time steps. Step `t` applies trainable nearest-neighbor gates `exp[-i(theta_xx XX + theta_zz ZZ)]` on the brickwork bonds for that step, then trainable `RX` rotations on every qubit, then the post-selected `Z` measurement branch defined below. Even-numbered steps use bonds `(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)`. Odd-numbered steps use bonds `(1,2), (3,4), (5,6), (7,8), (9,10)`. Each active bond has two scalar parameters, `theta_xx` and `theta_zz`, and each step has 12 scalar `RX` angles.

### Post-Selected Measurement Branch

After every time step, post-select the `+1` (`Z`-up, computational-basis `0`) outcome on each even-indexed qubit `0,2,4,6,8,10`, one measurement event at a time. This means post-selection is applied after an even-bond step and also after an odd-bond step, not only after a full even+odd block.

For each event, compute the branch probability and record `log(p_event + 1e-12)`. There are `10 * 6 = 60` selected measurement events in total. The total post-selection probability is the product of these 60 event probabilities.

### Loss Function

Train

$$\mathrm{loss} = \langle H_{\mathrm{target}}\rangle / 12 - 0.05 \cdot \mathrm{mean}\left[\log(p_{\mathrm{event}} + 10^{-12})\right]$$

with Adam at learning rate `0.01` for exactly 300 optimizer updates. Do not use early stopping.

## Solution Interface

The solution file must be named `solution_3.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `energy_density_history`: NumPy array with length `config["max_steps"]`.
- `success_probability_history`: NumPy array with length `config["max_steps"]`.
- `mean_log_probability_history`: NumPy array with length `config["max_steps"]`.
- `loss_history`: NumPy array with length `config["max_steps"]`.

Each history records one value per optimizer update, evaluated immediately before applying that update. The evaluator derives initial/final energy density, final success probability, final mean log probability, and initial/final loss from these histories.

The solution may use any quantum software framework, but it must consume only the evaluator-provided configuration and return only this NumPy-format dictionary.

## Evaluation Interface

The evaluator file is `evaluate_3.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_3.py --solution solution_3
```

The evaluator consumes only the returned result dictionary. It independently builds a sparse TFIM Hamiltonian, computes the exact ground-state energy by sparse diagonalization, and prints:

- solution module name,
- end-to-end solution time,
- exact sparse ground-state energy density,
- initial and final energy density,
- final total post-selection probability,
- final mean log event probability,
- initial and final total loss,
- training history lengths,
- returned result keys.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default 300-step configuration:

- `len(energy_density_history) == 300`, `len(success_probability_history) == 300`, `len(mean_log_probability_history) == 300`, and `len(loss_history) == 300`.
- The final loss is lower than the initial loss, derived from `loss_history`.
- The final energy density is lower than the initial energy density, derived from `energy_density_history`.
- The final energy density is at most `1.0` above the exact sparse ground-state energy density. This is intentionally loose and catches incorrect Hamiltonian/sign conventions without making the benchmark a high-precision cooling solver.
- The final success probability derived from `success_probability_history` is in `(0, 1]`.
- The final success probability matches `exp(60 * final_mean_log_probability)`, where both values are derived from their histories.
- All returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these metrics directly so another framework's `solution_3.py` can be compared without changing `evaluate_3.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_3.py` can be evaluated with:

```bash
python evaluate_3.py --solution solution_3
```

Observed TensorCircuit-NG baseline with the default 300-step configuration:

- End-to-end solution time: `2.46s`.
- Exact sparse ground energy density: `-1.17548668`.
- Initial energy density: `-0.44931078`.
- Final energy density: `-1.02593327`.
- Final success probability: `1.55883860e-02`.
- Final mean log event probability: `-6.93538189e-02`.
- Initial loss: `-0.44583517`.
- Final loss: `-1.02246559`.

## Implementation Hint

For a TensorCircuit-NG/JAX baseline, treat one even+odd cooling block as the repeated unit and scan over the five blocks with `jax.lax.scan`. Use `Circuit.post_select(q, keep=0)` for the selected `Z`-up branch. The fixed TFIM Hamiltonian can be evaluated with a matrix-free MVP and `<psi|Hpsi>`.
