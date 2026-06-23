# Problem 8: 7x7 Grid RZZ Tensor-Network Sampling

## Goal

Sample from a 49-qubit shallow two-dimensional IQP-style circuit without constructing the full `2^49` statevector. The task is designed so a one-dimensional MPS representation is not the natural simulation object: the entangling graph is a 7 by 7 square grid, while TensorCircuit can sample from the circuit by contracting the corresponding tensor network directly.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "grid_side": 7,
    "n_qubits": 49,
    "n_samples": 8192,
    "theta_offset": 0.43,
    "theta_sin_scale": 0.17,
    "theta_sin_frequency": 0.37,
    "theta_cos_scale": 0.11,
    "theta_cos_frequency": 0.19,
    "single_z_tolerance": 0.1,
    "edge_zz_tolerance": 0.1,
    "parity_tolerance": 1e-6,
}
```

Index the qubits by grid coordinate `(r, c)` with linear index `q = 7 r + c`. Start from `|0>^49`, apply `H` to every qubit, apply one full layer of `RZZ(theta_e)` gates on every nearest-neighbor grid edge, apply `H` to every qubit again, and sample in the computational basis. The `RZZ` convention is `RZZ(theta) = exp(-i theta Z_i Z_j / 2)`.

### Grid RZZ Layer

The grid edge layer is split into four non-overlapping sublayers applied in this order: horizontal edges with even column `c`, horizontal edges with odd column `c`, vertical edges with even row `r`, and vertical edges with odd row `r`. Every nearest-neighbor grid edge appears exactly once. If an edge is the `k`-th edge in this ordered list and has endpoints `(u, v)`, its angle is

```text
theta_k = 0.43 + 0.17 sin(0.37 (k + 1)) + 0.11 cos(0.19 (u + 2 v + 1)).
```

### Observable Checks

The evaluator computes empirical values from the returned samples for every single-site `Z_i`, every grid-edge `Z_i Z_j`, and the full-grid parity `prod_i Z_i`. For samples `b in {0,1}`, it maps each bit to `z = 1 - 2b`, estimates `<Z_i>` by averaging `z_i` over the 8192 shots, and estimates `<Z_i Z_j>` by averaging `z_i z_j` over the same shots. The reported single-site Z error is `max_i |mean(z_i) - exact(<Z_i>)|`, and the reported grid-edge ZZ error is `max_(i,j) |mean(z_i z_j) - exact(<Z_i Z_j>)|` over all nearest-neighbor grid edges. It independently computes exact references using the IQP moment identity. After the final Hadamard layer, `Z` moments are `X` moments before that layer. For the state produced by the grid `RZZ` layer from `|+>^49`, `<prod_{i in S} X_i> = prod_{e in boundary(S)} cos(theta_e)`, where `boundary(S)` contains grid edges with exactly one endpoint in `S`. In particular, the full-grid parity is exactly `1`.

## Solution Interface

The solution file must be named `solution_8.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should build the TensorCircuit circuit, sample from the tensor network with `allow_state=False`, and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `samples`: integer NumPy array with shape `(n_samples, n_qubits)` containing computational-basis bits `0` and `1`.
- `largest_index_dimension`: integer scalar reporting the largest tensor index dimension used by the qubit tensor network.

## Evaluation Interface

The evaluator file is `evaluate_8.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_8.py --solution solution_8
```

The evaluator consumes only the returned result dictionary. It prints sampled bitstrings, empirical and exact full-grid parity, maximum absolute error over all single-site `Z_i`, maximum absolute error over all grid-edge `Z_i Z_j`, sample shape, largest tensor index dimension, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `samples.shape == (8192, 49)`.
- Every sample entry is either `0` or `1`.
- The maximum single-site `Z_i` finite-sample error is at most `0.1`.
- The maximum grid-edge `Z_i Z_j` finite-sample error is at most `0.1`.
- The full-grid parity error is at most `1e-6`.
- The largest reported tensor index dimension is exactly `2`.
- All returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these metrics directly so another framework's `solution_8.py` can be compared without changing `evaluate_8.py`.

## Reference TensorCircuit-NG Baseline Run

A verified TensorCircuit-NG/JAX baseline run with the default 8192-shot configuration produced sample shape `(8192, 49)`, full-grid parity absolute error `0.0000000000`, maximum single-site Z absolute error `0.0279209603`, maximum grid-edge ZZ absolute error `0.0195386093`, and overall `PASS`. The evaluator-measured `run_solution(config)` time for that run was `98.90s`; this time is a reference measurement only and is not a passing criterion.

## Implementation Hint

Use `tc.Circuit` rather than a dense statevector or a one-dimensional MPS. A practical TensorCircuit-NG baseline configures an OMECO `TreeSA` path searcher with at least `ntrials=32` and `niters=32` through `tc.set_contractor("custom", optimizer=..., preprocessing=True)`, then calls `Circuit.sample(batch=n_samples, allow_state=False, format="sample_bin", status=...)` so sampling is performed by tensor-network contraction instead of materializing the full output distribution. Passing a fixed per-shot `status` array makes the baseline samples reproducible while still using the JIT-compiled perfect-sampling path.
