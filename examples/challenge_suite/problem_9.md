# Problem 9: 200-Qubit Local Light-Cone Optimization

## Goal

Optimize a local observable in the middle of a 200-qubit shallow variational circuit without constructing the full `2^200` statevector. The task is designed to test whether a framework can exploit the finite light cone of a local measurement: although the circuit is defined on 200 qubits, the target observable `<Z_100>` only depends on a small causal cone for an eight-layer nearest-neighbor ansatz.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 200,
    "observable_index": 100,
    "n_layers": 8,
    "max_steps": 100,
    "n_restarts": 200,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.02,
    "seed": 2035,
    "success_threshold": 0.9,
}
```

Start from `|+>^200`. The variational ansatz has eight brickwork layers. Each layer first applies trainable `RY` and `RZ` rotations on every qubit, then applies trainable nearest-neighbor `RZZ` gates on alternating bonds. Even layers use bonds `(0,1), (2,3), ...`; odd layers use bonds `(1,2), (3,4), ...`. The `RZZ` convention is `RZZ(theta) = exp(-i theta Z_i Z_j / 2)`.

For each of the 200 restarts, initialize all trainable parameters independently from a Gaussian distribution with mean `0` and standard deviation `initial_parameter_scale`, using seed `seed + restart_index`. Optimize the loss

```text
loss(theta) = - <Z_100>(theta)
```

with Adam at learning rate `0.03` for exactly `max_steps` updates. Do not use early stopping.

### Local-Minimum Landscape Probe

This is a local-control landscape problem rather than a single trajectory demo. The 200 random initializations probe how often the same shallow 200-qubit ansatz flows into high-polarization local basins around the central observable. The reported final-expectation variance and success fraction quantify the spread of local minima reached by the optimizer.

The default run performs 20,000 optimizer updates:

```bash
python evaluate_9.py --solution solution_9
```

The scientific quantities are the distribution of final local polarization over random initializations, including the best final expectation, mean final expectation, variance, standard deviation, and success fraction above the configured threshold.

## Solution Interface

The solution file must be named `solution_9.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only NumPy-format quantities that the evaluator checks or reports.

Required result keys:

- `observable_history`: NumPy array with shape `(n_restarts, max_steps)`.
- `final_grad_norms`: NumPy array with shape `(n_restarts,)`.

`observable_history` records one `<Z_100>` value per optimizer update and restart, evaluated immediately before applying that update. The evaluator derives per-restart initial and final expectations from the first and last columns of `observable_history`.

The evaluator recomputes the best, mean, variance, standard deviation, and success fraction from the returned arrays. It does not trust self-reported implementation flags as passing criteria.

## Evaluation Interface

The evaluator file is `evaluate_9.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_9.py --solution solution_9
```

The evaluator consumes only the returned result dictionary. It prints the end-to-end solution time, optimization summary, final-expectation variance, history shape, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default 200-restart configuration:

- `observable_history.shape == (n_restarts, max_steps)`.
- The initial and final expectation arrays derived from `observable_history` both have shape `(n_restarts,)`.
- The mean final `<Z_100>` is larger than the mean initial `<Z_100>`.
- The best final `<Z_100>` is at least `success_threshold`.
- All final gradient norms are finite.
- The final-expectation variance recomputed by the evaluator is finite and non-negative.
- All returned values are NumPy arrays or NumPy-compatible scalars.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_9.py` can be evaluated with:

```bash
python evaluate_9.py --solution solution_9
```

A verified TensorCircuit-NG/JAX baseline run with the default 200-restart configuration performed `20,000` optimizer updates and produced observable history shape `(200, 100)`, mean initial `<Z_100>` `0.0052354468`, mean final `<Z_100>` `0.9999909070`, final-expectation variance `1.6940038616e-11`, final-expectation standard deviation `4.1158278166e-06`, best final `<Z_100>` `1.0000007153`, success fraction `1.000000`, mean final gradient norm `9.68467597e-03`, and overall `PASS`. The evaluator-measured `run_solution(config)` time for that run was `44.92s`; this time is a reference measurement only and is not a passing criterion.

## Implementation Hint

Use `tc.Circuit` with `enable_lightcone=True` in `circuit.expectation_ps(z=[observable_index], enable_lightcone=True)`, and JIT compile the value-and-gradient function once. The key discipline is to keep the circuit structure, parameter shape, target observable, and light-cone flag fixed across all optimizer updates and restarts so the compiled executable is reused throughout the 20,000-update landscape probe.
