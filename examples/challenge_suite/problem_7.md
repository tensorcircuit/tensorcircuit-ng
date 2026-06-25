# Problem 7: 16-Qubit Measurement-Feedback VQE

## Goal

Optimize a variational quantum protocol whose energy objective is defined by an average over mid-circuit measurement trajectories. The task is not a static VQE over a single deterministic circuit: each trajectory contains projective ancilla measurements, and the subsequent two-qubit feedback gates depend explicitly on those measured classical bits.

The problem is designed to test whether a framework can express and optimize a hybrid quantum-classical protocol with:

- parameterized data and ancilla layers,
- projective mid-circuit ancilla measurement,
- branch-dependent conditional two-qubit gates,
- trajectory-batched energy estimation inside a gradient-based VQE loop.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_data_qubits": 8,
    "n_ancilla_qubits": 8,
    "n_qubits": 16,
    "n_layers": 2,
    "n_trajectories": 128,
    "initial_parameter_scale": 0.1,
    "max_steps": 100,
    "learning_rate": 0.02,
    "seed": 2047,
    "transverse_field": 1.05,
    "minimum_improvement": 0.3,
    "target_final_energy": -8.3,
}
```

Data qubits are indexed as `0,1,...,7`. Ancilla qubits are indexed as `8,9,...,15`, with ancilla `8 + i` paired to data qubit `i`.

The Hamiltonian acts only on the 8 data qubits and is the open-boundary transverse-field Ising model

$$
H = -\sum_{i=0}^{6} Z_i Z_{i+1} - 1.05 \sum_{i=0}^{7} X_i.
$$

## Variational Protocol

Each layer consists of the following operations, applied in this order.

### 1. Data Single-Qubit Layer

Apply trainable `RY` rotations on all data qubits:

```text
RY(theta_data[layer, i]) on data qubit i,  i = 0,...,7.
```

### 2. Ancilla Single-Qubit Layer

Apply trainable `RY` rotations on all ancilla qubits:

```text
RY(theta_anc[layer, i]) on ancilla qubit 8 + i,  i = 0,...,7.
```

### 3. Pre-Measurement Entangling Layer

For each pair `(data_i, ancilla_i)`, apply a trainable two-qubit entangler

```text
RZZ(theta_ent[layer, i]) on qubits (8 + i, i).
```

### 4. Mid-Circuit Ancilla Measurement

Measure each ancilla qubit `8 + i` in the computational basis, obtaining trajectory bit

```text
m[layer, i] in {0, 1}.
```

Use exactly `n_trajectories = 128` trajectories per objective evaluation, and keep the per-trajectory measurement randomness fixed across optimizer updates so the objective is a reproducible trajectory average rather than an optimizer-side resampling procedure.

### 5. Conditional Feedback Layer

For each pair `(ancilla_i, data_i)`, apply a conditional two-qubit `RZZ` gate:

- if `m[layer, i] = 0`, apply `RZZ(theta_fb0[layer, i])` on `(8 + i, i)`;
- if `m[layer, i] = 1`, apply `RZZ(theta_fb1[layer, i])` on `(8 + i, i)`.

These two feedback angles are independent trainable parameters.

### 6. Data Post-Processing Layer

Apply trainable `RZ` rotations on all data qubits:

```text
RZ(theta_post[layer, i]) on data qubit i,  i = 0,...,7.
```

## Objective Function

For a fixed parameter vector and a fixed batch of `128` trajectory uniforms, the protocol produces `128` measurement-conditioned trajectories. Let `E_t` be the data-Hamiltonian expectation value on trajectory `t`. The objective is the trajectory-averaged energy

$$
E_{\mathrm{avg}} = \frac{1}{128} \sum_{t=1}^{128} E_t.
$$

Train all variational parameters by minimizing `E_avg` with Adam at learning rate `0.02` for exactly `max_steps = 100` optimizer updates. Do not use early stopping.

All trainable parameters are initialized independently from a normal distribution with mean `0` and standard deviation `initial_parameter_scale`, using the configured seed.

## Solution Interface

The solution file must be named `solution_7.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should return only NumPy-format quantities consumed by the evaluator.

Required result keys:

- `energy_history`: NumPy array of length `config["max_steps"]`
- `final_trajectory_energies`: NumPy array with shape `(config["n_trajectories"],)`

`energy_history[t]` records the trajectory-averaged energy evaluated immediately before optimizer update `t`.

`final_trajectory_energies[k]` records the final per-trajectory data-Hamiltonian energy for the same fixed batch of `128` trajectory uniforms used by the objective.

## Evaluation Interface

The evaluator file is `evaluate_7.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_7.py --solution solution_7
```

The evaluator consumes only the returned result dictionary. It prints:

- solution module name,
- end-to-end solution time,
- initial trajectory-averaged energy,
- final trajectory-averaged energy,
- total energy improvement,
- mean and standard deviation of final trajectory energies,
- energy-history length,
- returned result keys.

It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `len(energy_history) == 100`
- `final_trajectory_energies.shape == (128,)`
- the final trajectory-averaged energy is lower than the initial trajectory-averaged energy
- the total energy improvement is at least `minimum_improvement = 0.3`
- the final trajectory-averaged energy is at most `target_final_energy = -8.3`
- all returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these quantities directly so another framework's `solution_7.py` can be compared without changing `evaluate_7.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_7.py` can be evaluated with:

```bash
python evaluate_7.py --solution solution_7
```

Observed TensorCircuit-NG/JAX baseline in the current validation environment with the default 100-step configuration:

- End-to-end solution time: `46.62s`.
- Initial trajectory-averaged energy: `-6.9331135750`.
- Final trajectory-averaged energy: `-9.6408786774`.
- Energy improvement: `2.7077651024`.
- Final trajectory energy mean/std: `-9.6408815384 / 0.0000018564`.
- Energy history length: `100`.
- Overall: `PASS`.

This time is a reference measurement only and is not a passing criterion.

## Implementation Hint

For a TensorCircuit-NG/JAX baseline, use `cond_measure` for ancilla measurements, batch the `128` fixed trajectories with `vmap`, and JIT-compile the full optimizer step rather than only `value_and_grad`, so each update reuses one compiled trajectory-averaged objective-and-update path.
