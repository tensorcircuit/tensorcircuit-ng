# Problem 11: Spin-1 Haldane-Chain VQE With String-Order Verification

## Goal

Find a low-energy variational state for an open spin-1 chain in the Haldane phase and verify that the optimized state reproduces the expected nonlocal string order. The task is not only to lower the energy, but to produce a physically consistent final state whose hidden antiferromagnetic order can be checked from fixed string correlators.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_sites": 12,
    "n_layers": 5,
    "beta": 0.20,
    "single_ion_anisotropy": 0.15,
    "max_steps": 500,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.05,
    "seed": 2041,
    "minimum_energy_improvement": 5e-3,
    "maximum_energy_density_gap": 0.12,
    "maximum_string_order_mae": 0.12,
}
```

Use an open-boundary spin-1 chain with local basis ordered as `|+1>`, `|0>`, `|-1>`. The local spin operators are

```text
Sx = (1 / sqrt(2)) [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
Sy = (1 / sqrt(2)) [[0, -i, 0], [i, 0, -i], [0, i, 0]]
Sz = [[1, 0, 0], [0, 0, 0], [0, 0, -1]].
```

The Hamiltonian is

```text
H = sum_{i=0}^{n_sites-2} [Si · S{i+1} + beta (Si · S{i+1})^2]
    + single_ion_anisotropy sum_{i=0}^{n_sites-1} (Sz_i)^2,
```

where

```text
Si · Sj = Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j.
```

For the default configuration this is an `n_sites = 12` spin-1 chain with `beta = 0.20` and `single_ion_anisotropy = 0.15`.

## Variational Ansatz

Start from the fixed Neel product state

```text
|+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1>.
```

Use `n_layers = 5` repeated brickwork layers. In layer `l`, apply the following operations in order.

### 1. Single-Site Rotation Block

For every site `i = 0, ..., n_sites - 1`, apply

```text
Rz(alpha_{l,i}) Ry(beta_{l,i}) Rz(gamma_{l,i}),
```

with

```text
Rz(phi) = exp(-i phi Sz),
Ry(theta) = exp(-i theta Sy).
```

### 2. Even-Bond Entangling Block

For every even bond `(0,1), (2,3), ..., (n_sites - 2, n_sites - 1)`, apply

```text
U_even(theta_{l,b}, phi_{l,b})
  = exp[-i theta_{l,b} (Sx⊗Sx + Sy⊗Sy)
        - i phi_{l,b} (Sz⊗Sz)
        - i beta (Si · S{i+1})^2].
```

### 3. Odd-Bond Entangling Block

For every odd bond `(1,2), (3,4), ..., (n_sites - 3, n_sites - 2)`, apply an independently parameterized gate of the same form,

```text
U_odd(theta'_{l,b}, phi'_{l,b})
  = exp[-i theta'_{l,b} (Sx⊗Sx + Sy⊗Sy)
        - i phi'_{l,b} (Sz⊗Sz)
        - i beta (Si · S{i+1})^2].
```

Each layer therefore has

```text
3 * n_sites + 2 * (n_sites - 1)
```

real parameters. For the default `n_sites = 12` configuration, each layer has `58` parameters and the full ansatz has `290` parameters.

Initialize every trainable parameter independently from a Gaussian distribution with mean `0` and standard deviation `initial_parameter_scale`, using the configured seed. Optimize all parameters with Adam at learning rate `learning_rate` for exactly `max_steps = 500` updates. Do not use early stopping.

## String-Order Diagnostics

In addition to the final energy, verify the Haldane-phase string correlators

```text
O_string^z(i, j)
  = < Sz_i [prod_{k=i+1}^{j-1} exp(i pi Sz_k)] Sz_j >.
```

The evaluator checks the three fixed strings

```text
(i, j) in {(0, 11), (1, 10), (2, 9)}.
```

For spin-1, the middle factor is

```text
exp(i pi Sz) = diag(-1, 1, -1).
```

## Solution Interface

The solution file must be named `solution_11.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform only the optimization and return only the NumPy-format quantities that the evaluator checks or reports.

Required result keys:

- `energy_density_history`: NumPy array with shape `(max_steps,)`.
- `final_energy_density`: scalar energy density evaluated after the last optimizer update.
- `final_string_orders`: NumPy array with shape `(3,)`, ordered as `[(0, 11), (1, 10), (2, 9)]`.

Each energy-history entry records the energy density evaluated immediately before the corresponding optimizer update.

## Evaluation Interface

The evaluator file is `evaluate_11.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_11.py --solution solution_11
```

The evaluator consumes only the returned result dictionary. Outside the timed `run_solution(config)` call, it independently computes the exact ground-state energy density and exact string correlators for the fixed Hamiltonian, then compares those exact quantities against the final energy density and final string correlators reported by the solution.

It prints:

- solution module name,
- end-to-end solution time,
- exact-reference time,
- initial and final variational energy density,
- exact ground-state energy density,
- final energy-density gap,
- the three final and exact string correlators,
- string-order mean absolute error,
- returned array shapes and keys.

The evaluator does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `energy_density_history.shape == (max_steps,)`.
- `final_string_orders.shape == (3,)`.
- The final energy density is at least `minimum_energy_improvement` lower than the initial energy density.
- The returned arrays contain finite values.
- The final energy density is no more than `maximum_energy_density_gap` above the exact ground-state density.
- The mean absolute error of the three checked string correlators is at most `maximum_string_order_mae = 0.12`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_11.py` can be evaluated with:

```bash
python evaluate_11.py --solution solution_11
```

A verified TensorCircuit-NG/JAX baseline run in the current validation environment with the default configuration produced end-to-end solution time `68.08s`, exact-reference time `4.63s`, initial energy density `-0.0546756685`, final energy density `-0.7045351863`, exact ground-state energy density `-0.7736449389`, energy-density gap `0.0691097526`, string correlators `(-0.3860595226, -0.3445219994, -0.4002564847)`, string-order mean absolute error `0.0551720974`, and overall `PASS`. The evaluator-measured `run_solution(config)` time is a reference measurement only and is not a passing criterion.

## Implementation Hint

Because every layer has the same fixed structure, it is often advantageous to stage one layer as a repeated tensor-program body over depth rather than tracing four separate copies of the layer into the compiled program. The energy and the checked string correlators are both sums of short-range or fixed-support observables, so they can be evaluated directly from the final variational state without introducing auxiliary sampling.
