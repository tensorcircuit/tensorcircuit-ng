# Problem 8: 7x7 Mixed-Axis Grid Tensor-Network Sampling

## Goal

Sample from a 49-qubit shallow two-dimensional circuit without constructing the full `2^49` statevector. The entangling graph is a 7 by 7 square grid, and the intended computation samples by contracting the corresponding tensor network directly rather than by reducing the problem to a one-dimensional MPS-friendly geometry.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "grid_side": 7,
    "n_qubits": 49,
    "n_samples": 8192,
    "ry_offset": 0.19,
    "ry_row_sin_scale": 0.07,
    "ry_row_sin_frequency": 0.83,
    "ry_col_cos_scale": 0.05,
    "ry_col_cos_frequency": 0.61,
    "ry_diag_sin_scale": 0.03,
    "ry_diag_sin_frequency": 0.29,
    "rzz_offset": 0.31,
    "rzz_edge_sin_scale": 0.09,
    "rzz_edge_sin_frequency": 0.47,
    "rzz_site_cos_scale": 0.06,
    "rzz_site_cos_frequency": 0.38,
    "rxx_offset": 0.27,
    "rxx_edge_cos_scale": 0.08,
    "rxx_edge_cos_frequency": 0.41,
    "rxx_site_sin_scale": 0.07,
    "rxx_site_sin_frequency": 0.33,
    "rx_offset": 0.17,
    "rx_row_cos_scale": 0.06,
    "rx_row_cos_frequency": 0.52,
    "rx_col_sin_scale": 0.04,
    "rx_col_sin_frequency": 0.44,
    "rx_diag_cos_scale": 0.02,
    "rx_diag_cos_frequency": 0.25,
    "single_z_tolerance": 0.03,
    "hidden_z_string_max_tolerance": 0.05,
    "hidden_z_string_mean_tolerance": 0.015,
}
```

Index the qubits by grid coordinate `(r, c)` with linear index `q = 7 r + c`. Start from `|0>^49`, apply one position-dependent `Ry` gate on every qubit, apply one full horizontal nearest-neighbor `RZZ` layer, apply one full vertical nearest-neighbor `RXX` layer, apply one position-dependent `Rx` gate on every qubit, and sample in the computational basis. The conventions are `RZZ(theta) = exp(-i theta Z_i Z_j / 2)` and `RXX(theta) = exp(-i theta X_i X_j / 2)`.

### Single-Qubit Layers

For every site `(r, c)`, the first-layer rotation angle is

```text
alpha(r, c) = 0.19 + 0.07 sin(0.83 (r + 1)) + 0.05 cos(0.61 (c + 1)) + 0.03 sin(0.29 (r + c + 2)).
```

Apply `Ry(alpha(r,c))` at every site in row-major order.

For every site `(r, c)`, the final-layer rotation angle is

```text
gamma(r, c) = 0.17 + 0.06 cos(0.52 (r + 1)) - 0.04 sin(0.44 (c + 1)) + 0.02 cos(0.25 (r + c + 2)).
```

Apply `Rx(gamma(r,c))` at every site in row-major order.

### Horizontal RZZ Layer

Apply `RZZ` on every horizontal nearest-neighbor edge `((r,c),(r,c+1))` in row-major edge order. If the edge is the `k_h`-th horizontal edge in that order, its angle is

```text
beta_h(r, c, k_h) = 0.31 + 0.09 sin(0.47 (k_h + 1)) + 0.06 cos(0.38 (2 r + c + 1)).
```

### Vertical RXX Layer

Apply `RXX` on every vertical nearest-neighbor edge `((r,c),(r+1,c))` in row-major edge order. If the edge is the `k_v`-th vertical edge in that order, its angle is

```text
beta_v(r, c, k_v) = 0.27 + 0.08 cos(0.41 (k_v + 1)) + 0.07 sin(0.33 (r + 2 c + 1)).
```

## Observable Checks

The evaluator consumes only the returned computational-basis samples. It converts each sampled bit `b in {0,1}` to `z = 1 - 2b` and estimates a fixed hidden set of `Z`-string observables from sample averages. The hidden checks include single-site `Z_i`, selected two-site `Z_i Z_j`, local patch parities, and longer-range `prod_{i in S} Z_i` strings. The exact reference values are precomputed once from the fixed circuit and are not disclosed in the problem statement.

The reported metrics include the maximum absolute error over the hidden single-site `Z_i` checks, the maximum absolute error over the full hidden `Z`-string set, and the mean absolute error over the full hidden `Z`-string set.

## Solution Interface

The solution file must be named `solution_8.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should implement the fixed circuit, sample from the computational-basis distribution without materializing the full statevector or dense output probability vector, and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `samples`: integer NumPy array with shape `(n_samples, n_qubits)` containing computational-basis bits `0` and `1`.

## Evaluation Interface

The evaluator file is `evaluate_8.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_8.py --solution solution_8
```

The evaluator consumes only the returned result dictionary. It prints sampled bitstrings, the number of hidden observables checked, the maximum single-site `Z_i` absolute error over the hidden single-site set, the maximum hidden `Z`-string absolute error, the mean hidden `Z`-string absolute error, sample shape, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `samples.shape == (8192, 49)`.
- Every sample entry is either `0` or `1`.
- The maximum hidden single-site `Z_i` finite-sample error is at most `0.03`.
- The maximum hidden `Z`-string finite-sample error is at most `0.05`.
- The mean hidden `Z`-string finite-sample error is at most `0.015`.
- All returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these metrics directly so another framework's `solution_8.py` can be compared without changing `evaluate_8.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_8.py` can be evaluated with:

```bash
python evaluate_8.py --solution solution_8
```

A correct baseline should pass the observable checks with ordinary finite-sample fluctuations at the default 8192-shot configuration. In the current validation environment, the TensorCircuit-NG/JAX baseline produced sample shape `(8192, 49)`, checked `44` hidden observables, had maximum single-site `Z` absolute error `0.0046486279`, maximum hidden `Z`-string absolute error `0.0188068181`, mean hidden `Z`-string absolute error `0.0039698657`, and evaluator-measured `run_solution(config)` time `119.10s`. This time is a reference measurement only and is not a passing criterion.

## Implementation Hint

Use `tc.Circuit` to build the circuit directly on the 49-qubit grid, apply the parameterized single-qubit and entangling layers in the required order, and call `Circuit.sample(allow_state=False, format="sample_bin")` or an equivalent tensor-network sampling path.
