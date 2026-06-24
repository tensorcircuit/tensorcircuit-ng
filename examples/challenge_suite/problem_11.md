# Problem 11: Mixed-Mode Differentiable Quantum Sensor Training

## Goal

Train a variational quantum sensor whose learned object is a full local response matrix, not a single expectation value. The task combines forward-mode differentiation with respect to two physical field parameters and reverse-mode differentiation with respect to hundreds of trainable circuit parameters.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 20,
    "n_field_params": 2,
    "n_layers": 6,
    "max_steps": 300,
    "learning_rate": 0.01,
    "initial_parameter_scale": 0.02,
    "readout_penalty_weight": 0.05,
    "seed": 2037,
    "final_response_mse_tolerance": 1e-9,
}
```

Use 20 qubits arranged on an open chain. Prepare a trainable probe state from `|0>^20` using six brickwork layers. Each layer applies trainable `RZ-RY` rotations on every qubit, followed by trainable nearest-neighbor `RZZ` gates on alternating bonds. Even layers use bonds `(0,1), (2,3), ...`; odd layers use bonds `(1,2), (3,4), ...`. The `RZZ` convention is `RZZ(theta) = exp(-i theta Z_i Z_j / 2)`.

Encode a two-parameter weak field `lambda = (lambda_0, lambda_1)` by applying

```text
U(lambda) = exp[-i sum_i (lambda_0 + lambda_1 x_i) Z_i],
```

where `x_i = 2 i / 19 - 1`. Define the 20-dimensional readout vector

```text
f_i(theta, lambda) = <X_i>
```

after sensing. At `lambda = (0, 0)`, compute the response matrix

```text
R_{i,a}(theta) = partial f_i(theta, lambda) / partial lambda_a
```

for all sites `i = 0, ..., 19` and physical field parameters `a = 0, 1`. Use the target response matrix

```text
T_{i,0} = 1,    T_{i,1} = x_i.
```

Train the objective

```text
loss = mean((R - T)^2) + 0.05 * mean(f_i(theta, 0)^2)
```

with Adam at learning rate `0.01` for exactly 300 optimizer updates. Do not use early stopping.

## Solution Interface

The solution file must be named `solution_11.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only NumPy-format quantities that the evaluator checks or reports.

Required result keys:

- `loss_history`: NumPy array with shape `(max_steps,)`.
- `response_mse_history`: NumPy array with shape `(max_steps,)`.
- `readout_penalty_history`: NumPy array with shape `(max_steps,)`.
- `final_response_matrix`: NumPy array with shape `(n_qubits, n_field_params)`.
- `final_zero_field_readouts`: NumPy array with shape `(n_qubits,)`.
- `final_grad_norm`: scalar float.

Each history records one value per optimizer update, evaluated immediately before applying that update. The evaluator recomputes final response MSE, final readout penalty, and final total loss from the returned arrays.

## Evaluation Interface

The evaluator file is `evaluate_11.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_11.py --solution solution_11
```

The evaluator consumes only the returned result dictionary. It prints the end-to-end solution time, initial and final response-matrix MSE, final zero-field readout penalty, final total loss, final gradient norm, trainable parameter count, response dimensions, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `loss_history.shape == response_mse_history.shape == readout_penalty_history.shape == (300,)`.
- `final_response_matrix.shape == (20, 2)`.
- `final_zero_field_readouts.shape == (20,)`.
- The final response-matrix MSE is lower than the initial response-matrix MSE.
- The final response-matrix MSE is at most `1e-9`.
- The final loss is lower than the initial loss.
- The final gradient norm is finite.
- All returned arrays contain finite values.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_11.py` can be evaluated with:

```bash
python evaluate_11.py --solution solution_11
```

A verified TensorCircuit-NG/JAX baseline run with the default configuration performed `300` optimizer updates and produced response matrix shape `(20, 2)`, initial response MSE `6.85145557e-01`, final response MSE `7.06921946e-12`, final zero-field readout penalty `4.91542769e-09`, initial total loss `6.85256898e-01`, final total loss `2.52840604e-10`, final gradient norm `2.40408826e-06`, and overall `PASS`. The evaluator-measured `run_solution(config)` time for that run was `48.55s`; this time is a reference measurement only and is not a passing criterion.

## Implementation Hint

Compute the two columns of `R` by forward-mode derivatives with respect to the two physical field parameters, then differentiate the scalar response-matching loss by reverse mode with respect to the much larger set of trainable circuit parameters. In TensorCircuit-NG/JAX, one efficient implementation configures an OMECo contraction path searcher, scans over even/odd layer pairs while constructing the probe state, computes all local `X_i` readouts from the resulting state via local coherences, applies `tc.backend.jacfwd` only to the two-parameter sensing map, and wraps the outer scalar loss in `tc.backend.value_and_grad`.
