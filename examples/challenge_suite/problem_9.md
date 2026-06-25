# Problem 9: Random Local Light-Cone Optimization

## Goal

Optimize two local Pauli observables in a large irregular but local variational circuit without constructing the full `2^512` statevector. The task is designed to test whether a framework can exploit finite causal cones for local measurements when the circuit is no longer a regular brickwork pattern.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`. In addition to these scalar fields, the evaluator materializes and passes the full `gate_tape`, `parameter_count`, `pauli_terms`, and `pauli_cone_sizes` described below.

```python
{
    "n_qubits": 512,
    "n_layers": 6,
    "max_steps": 100,
    "n_restarts": 200,
    "learning_rate": 0.03,
    "initial_parameter_scale": 0.02,
    "seed": 2035,
    "edge_keep_prob": 0.24,
    "success_threshold": 1.0,
}
```

Start from `|+>^512`. The variational ansatz is a seeded random local circuit on a two-leg ladder graph. The evaluator constructs a framework-neutral gate tape and passes it directly in `config["gate_tape"]`, so the solution does not have to infer the circuit from prose.

Each gate tape entry is a tuple:

- Single-qubit trainable gate: `(gate_name, qubit, parameter_index)`.
- Two-qubit trainable gate: `(gate_name, qubit_a, qubit_b, parameter_index)`.

Here `gate_name` is one of `"rx"`, `"ry"`, `"rz"`, `"rxx"`, `"ryy"`, or `"rzz"`. Each gate consumes exactly one scalar trainable parameter from the full parameter vector at `parameter_index`. The two-qubit gate convention is `exp(-i theta P_i P_j / 2)` for `RXX`, `RYY`, and `RZZ`.

The evaluator uses this exact deterministic generator:

```python
def ladder_edges(n_qubits):
    assert n_qubits % 2 == 0
    n_columns = n_qubits // 2
    edges = []
    for column in range(n_columns):
        edges.append((2 * column, 2 * column + 1))
        if column + 1 < n_columns:
            edges.append((2 * column, 2 * (column + 1)))
            edges.append((2 * column + 1, 2 * (column + 1) + 1))
            if column % 2 == 0:
                edges.append((2 * column, 2 * (column + 1) + 1))
            else:
                edges.append((2 * column + 1, 2 * (column + 1)))
    return edges


def generate_gate_tape(config):
    rng = np.random.default_rng(config["seed"])
    single_axes = ("rx", "ry", "rz")
    two_axes = ("rxx", "ryy", "rzz")
    graph_edges = ladder_edges(config["n_qubits"])
    gate_tape = []
    parameter_index = 0

    for _ in range(config["n_layers"]):
        for qubit in range(config["n_qubits"]):
            axis = single_axes[int(rng.integers(len(single_axes)))]
            gate_tape.append((axis, qubit, parameter_index))
            parameter_index += 1

        edges = list(graph_edges)
        rng.shuffle(edges)
        used = set()
        for qubit_a, qubit_b in edges:
            if qubit_a in used or qubit_b in used:
                continue
            if rng.random() > config["edge_keep_prob"]:
                continue
            axis = two_axes[int(rng.integers(len(two_axes)))]
            gate_tape.append((axis, qubit_a, qubit_b, parameter_index))
            parameter_index += 1
            used.add(qubit_a)
            used.add(qubit_b)

    return tuple(gate_tape), parameter_index
```

For the default configuration this produces a `gate_tape` of length `3897`, including `825` two-qubit gates. The two measured local Pauli terms are passed as:

```python
pauli_terms = (
    (0.5645931361768194, (("x", 388), ("z", 390))),
    (1.0, (("x", 16), ("y", 19))),
)
```

The optimized objective is

```text
local_objective(theta)
  = 0.5645931361768194 * <X_388 Z_390>(theta)
    + <X_16 Y_19>(theta)

loss(theta) = - local_objective(theta)
```

The evaluator also passes `pauli_cone_sizes` for reporting only; with the default gate tape the two cone sizes are `18` and `15`.

For each of the 200 restarts, initialize the full trainable parameter vector independently from a Gaussian distribution with mean `0` and standard deviation `initial_parameter_scale`, using seed `seed + 100000 + restart_index`. Optimize the loss with Adam at learning rate `0.03` for exactly `max_steps` updates. Do not use early stopping.

### Local Light-Cone Probe

The full circuit has 512 qubits and thousands of parameterized gates, while the two requested Pauli terms only depend on small but irregular causal cones. The circuit is local, but because each layer uses a seeded random matching and random gate axes, the causal cones are not simple contiguous brickwork intervals.

The default run performs 20,000 restart-step updates:

```bash
python evaluate_9.py --solution solution_9
```

The scientific quantities are the distribution of final local objective values over random initializations, including the best final expectation, mean final expectation, variance, standard deviation, and success fraction above the configured threshold.

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

`observable_history` records one local objective value per optimizer update and restart, evaluated immediately before applying that update. The evaluator derives per-restart initial and final objective values from the first and last columns of `observable_history`.

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
- The initial and final objective arrays derived from `observable_history` both have shape `(n_restarts,)`.
- The mean final local objective is larger than the mean initial local objective.
- The best final local objective is at least `success_threshold`.
- The final-objective variance recomputed by the evaluator is finite and non-negative.
- All returned values are NumPy arrays or NumPy-compatible scalars.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_9.py` can be evaluated with:

```bash
python evaluate_9.py --solution solution_9
```

The baseline constructs the full 512-qubit `tc.Circuit` from `config["gate_tape"]`, evaluates the two Pauli terms with `enable_lightcone=True`, and JIT compiles the batched optimizer step across restarts. The key discipline is to keep the full gate tape, parameter shape, target Pauli terms, and light-cone flag fixed across all optimizer updates and restarts so the compiled executable is reused throughout the 20,000-update landscape probe.
