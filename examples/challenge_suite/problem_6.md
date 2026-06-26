# Problem 6: Digital-Analog Hybrid VQE With Trainable Analog Blocks

## Goal

Optimize a hybrid digital-analog variational ansatz to minimize the energy density of a 14-qubit Hamiltonian. Each ansatz block interleaves a trainable analog Hamiltonian evolution with a trainable layer of local digital rotations. The analog evolution time, coupling strength, and detuning are all trainable, so the optimizer simultaneously shapes the analog physics and the digital pulse sequence.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 14,
    "n_blocks": 4,
    "t_min": 0.05,
    "t_max": 0.50,
    "ode_rtol": 1e-6,
    "ode_atol": 1e-6,
    "ode_max_steps": 16,
    "max_steps": 100,
    "learning_rate": 0.12,
}
```

Use the open-boundary 14-qubit target Hamiltonian

$$H_{\text{target}} = \sum_{i=0}^{12} (0.7\, X_i X_{i+1} + 0.7\, Y_i Y_{i+1} + 1.1\, Z_i Z_{i+1}) + 0.25 \sum_{i=0}^{13} (-1)^i Z_i.$$

Start from the Néel state $|01010101010101\rangle$.

### Ansatz Structure

Use `n_blocks = 4` hybrid blocks indexed by $l = 0, \ldots, 3$. In block $l$:

1. **Analog evolution**: evolve under the trainable time-independent Hamiltonian

$$H_{\text{analog}}^{(l)} = J_l \sum_{i=0}^{12} (X_i X_{i+1} + Y_i Y_{i+1}) + \Delta_l \sum_{i=0}^{13} (-1)^i Z_i$$

for trainable time $t_l$, where

$$t_l = t_{\min} + (t_{\max} - t_{\min}) \cdot \sigma(s_l), \quad J_l = \tanh(j_l), \quad \Delta_l = \tanh(d_l),$$

with $\sigma(x) = 1/(1+e^{-x})$ and unconstrained trainable scalars $s_l, j_l, d_l$. This bounds the evolution time in $(t_{\min}, t_{\max})$ and bounds the couplings in $(-1, 1)$.

The analog evolution in each block is intended to be implemented as a continuous-time ODE/Schrödinger evolution, not as a Trotterized gate sequence. The presence of `ode_rtol`, `ode_atol`, and `ode_max_steps` in the config is part of the specification:

- Use an actual ODE solver or framework-native differential-equation integrator for the analog block.
- Respect `ode_rtol` and `ode_atol` as numerical tolerances.
- Treat `ode_max_steps` as an ODE/integrator step-control bound, not as a product-formula or Trotter slice count.
- Do not replace the analog block with Lie-Trotter, Suzuki-Trotter, operator splitting, or any other sliced digital approximation of the requested continuous-time evolution.

2. **Digital rotations**: apply trainable $RZ(\alpha_{l,k})\,RY(\beta_{l,k})\,RZ(\gamma_{l,k})$ on every qubit $k$.

### Initialization

Initialize all $s_l$ to $0$ (giving $t_l = (t_{\min}+t_{\max})/2 = 0.275$), all $j_l$ and $d_l$ to $0.1$, and all rotation angles $\alpha_{l,k}, \beta_{l,k}, \gamma_{l,k}$ independently from $\mathcal{N}(0, 0.1^2)$.

### Loss Function

Train the objective

$$\mathcal{L} = \langle H_{\text{target}} \rangle / 14$$

for exactly `max_steps = 100` Adam updates at learning rate `0.12`. Do not use early stopping.

## Solution Interface

The solution file must be named `solution_6.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `final_analog_times`: NumPy array with shape `(n_blocks,)` containing the learned analog evolution times $t_l$.
- `final_analog_couplings`: NumPy array with shape `(n_blocks,)` containing the learned XY couplings $J_l = \tanh(j_l)$.
- `final_analog_detunings`: NumPy array with shape `(n_blocks,)` containing the learned detunings $\Delta_l = \tanh(d_l)$.
- `energy_density_history`: NumPy array with length `config["max_steps"]`.

`energy_density_history` records one value per optimizer update, evaluated immediately before applying that update. The evaluator derives initial and final energy density from the first and last entries of `energy_density_history`.

The solution may use any quantum software framework, but it must consume only the evaluator-provided configuration and return only this NumPy-format dictionary.

## Evaluation Interface

The evaluator file is `evaluate_6.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_6.py --solution solution_6
```

The evaluator consumes only the returned result dictionary. It computes the exact 14-qubit sparse ground-state energy independently using `scipy.sparse.linalg.eigsh`, then prints initial and final energy density, exact ground-state energy density, learned analog parameters, energy history length, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default 100-step configuration:

- `len(energy_density_history) == 100`.
- `final_analog_times.shape == (n_blocks,)` with all values in `(t_min, t_max)`.
- `final_analog_couplings.shape == (n_blocks,)` with all values in `(-1, 1)`.
- `final_analog_detunings.shape == (n_blocks,)` with all values in `(-1, 1)`.
- The final energy density is lower than the initial energy density.
- The final energy density is no lower than the exact sparse ground-state energy density beyond numerical tolerance `1e-6`.
- The final energy density is at most `1.0` above the exact sparse ground-state energy density.
- All returned values are NumPy arrays or NumPy-compatible scalars.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_6.py` can be evaluated with:

```bash
python evaluate_6.py --solution solution_6
```

Observed TensorCircuit-NG baseline with the default 100-step configuration:

- End-to-end solution time: `26.83s`.
- Exact sparse ground energy density: `-1.6025540829`.
- Initial energy density: `-0.5182266235`.
- Final energy density: `-1.5754302740`.

## Implementation Hint

For a TensorCircuit-NG/JAX baseline, implement each analog block directly with `diffrax.diffeqsolve` using the time-independent vector field `f(t, y) = -i (J_l H_{XY}(y) + Delta_l H_{field}(y))`, where `H_XY`, `H_field`, and `H_target` are fixed matrix-free MVP functions built once outside the training loop with `tc.quantum.PauliStringSum2MVP`. Use `diffrax.Tsit5` with `diffrax.PIDController` tolerances. Interleave each analog evolution with a `tc.Circuit` digital rotation layer applied to the output state.
