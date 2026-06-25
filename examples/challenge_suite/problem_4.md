# Problem 4: Trainable Kraus Noise Calibration From Multi-Circuit Data

## Goal

Fit the two parameters of a user-defined asymmetric bit-flip channel from synthetic multi-circuit observable data. The task includes generating the synthetic target observable table by running the same noisy probe circuits at the true channel probabilities, then fitting the channel probabilities from that generated table. The evaluator provides the true probabilities and probe configuration, but it does not provide precomputed target observables. The task is not to use a built-in noise model, but to express the channel as Kraus tensor algebra, insert it after fixed entangling operations, differentiate through the noisy circuits, and recover the true transition probabilities from expectation values.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 12,
    "entangler_angle": 0.31,
    "true_p01": 0.034,
    "true_p10": 0.011,
    "initial_p01": 0.070,
    "initial_p10": 0.040,
    "max_steps": 120,
    "learning_rate": 0.04,
    "probability_absolute_tolerance": 2e-4,
}
```

The asymmetric bit-flip channel acts on one qubit as

$$K_0 = \sqrt{1-p_{01}} |0\rangle\langle 0| + \sqrt{1-p_{10}} |1\rangle\langle 1|,$$

$$K_1 = \sqrt{p_{10}} |0\rangle\langle 1|,\qquad K_2 = \sqrt{p_{01}} |1\rangle\langle 0|.$$

Here `p01` is the probability for `0 -> 1` and `p10` is the probability for `1 -> 0`. Inside `run_solution(config)`, first generate the synthetic target data from the true values `p01 = 0.034` and `p10 = 0.011`, using exactly the probe circuits and observables defined below. Then initialize the fitted probabilities from `initial_p01` and `initial_p10` and optimize them against that generated target table.

### Probe Circuits

Use four fixed 12-qubit inputs and the same noisy probe circuit for every input. The four inputs are a GHZ state `(|0...0> + |1...1>) / sqrt(2)`, Bell-like pair states `(|01> + |10>) / sqrt(2)` on pairs `(0,1), (2,3), ..., (10,11)`, `|0>^12`, and `|+>^12`.

The shared probe circuit is one even-bond brickwork entangler layer. On every bond `(0,1), (2,3), ..., (10,11)`, apply `RXX(0.31)` with convention `RXX(theta) = exp(-i theta X_i X_j / 2)`, and then apply the asymmetric bit-flip channel independently to the two qubits in that bond.

All probes therefore have identical noisy entangler structure and differ only in the initial state.

### Observables And Loss

For each probe circuit, use the noisy expectations of all single-site observables `Z_i` for `i = 0, ..., 11` and the full-chain parity `Z_0 Z_1 ... Z_11`. This gives a target table with shape `(4, 13)`.

Parameterize the fitted probabilities by two unconstrained scalars,

$$p_{01} = \operatorname{sigmoid}(r_{01}),\qquad p_{10} = \operatorname{sigmoid}(r_{10}).$$

Initialize these probabilities to `initial_p01 = 0.070` and `initial_p10 = 0.040`. Train the mean squared error between the fitted observable table and the synthetic target table for exactly 120 Adam updates at learning rate `0.04`. Do not use early stopping.

## Solution Interface

The solution file must be named `solution_4.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `loss_history`: NumPy array with length `config["max_steps"]`.
- `final_probabilities`: NumPy array with shape `(2,)`, storing final `[p01, p10]`.
- `fitted_expectations`: NumPy array with shape `(4, 13)`.

`loss_history` records one loss value per optimizer update, evaluated immediately before applying that update. The solution may use any quantum software framework, but it must consume only the evaluator-provided configuration and return only this NumPy-format dictionary.

## Evaluation Interface

The evaluator file is `evaluate_4.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_4.py --solution solution_4
```

The evaluator consumes only the returned result dictionary. It prints the true probabilities, initialized probabilities, fitted probabilities, absolute parameter errors, initial and final loss, trace-preserving error for the fitted Kraus operators, loss-history length, fitted expectation shape, returned keys, and pass/fail criteria.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default 120-step configuration:

- `len(loss_history) == 120`.
- `final_probabilities.shape == (2,)` and `fitted_expectations.shape == (4, 13)`.
- The final loss is lower than the initial loss, derived from `loss_history`.
- The absolute errors of both fitted probabilities are at most `probability_absolute_tolerance`.
- The fitted Kraus operators satisfy `max(abs(sum_a K_a^\dagger K_a - I)) <= 1e-8`.
- All returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these metrics directly so another framework's `solution_4.py` can be compared without changing `evaluate_4.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_4.py` can be evaluated with:

```bash
python evaluate_4.py --solution solution_4
```

Observed TensorCircuit-NG/JAX baseline in the current validation environment with the entangled-probe default configuration:

- End-to-end solution time: `47.45s`.
- Initial loss: `6.70448504e-03`.
- Final loss: `3.09971142e-08`.
- Fitted `p01`: `0.03398004`.
- Fitted `p10`: `0.01109863`.
- Trace-preserving error: `1.11022302e-16`.

## Implementation Hint

For a TensorCircuit-NG/JAX baseline, use `DMCircuit` with `apply_general_kraus` to insert the custom one-qubit channel after each entangling operation. The probes share one noisy entangler structure and differ only in their initial states, keeping the benchmark focused on the trainable channel. Keep the Kraus operators as differentiable tensor algebra, enforce positivity with sigmoid-parameterized probabilities, and verify trace preservation by contracting `sum_a K_a^\dagger K_a`.
