# Problem 2: Entanglement-Profile-Constrained VQE

## Goal

Optimize a variational quantum eigensolver for a 12-qubit open-boundary XXZ chain with a staggered field. The objective combines the final energy density with a penalty that matches a prescribed half-chain Renyi-2 entropy profile at three fixed circuit checkpoints.

In this problem, an "entanglement profile" means a fixed list of target Renyi-2 entropies at fixed circuit checkpoints. It is not inferred from the ground state and it is not a sampled observable. The optimizer is explicitly penalized when the half-chain entropy at checkpoint `b` differs from the evaluator-provided scalar `target_entropies[b]`.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 12,
    "zz_anisotropy": 1.2,
    "staggered_field": 0.35,
    "n_layers": 6,
    "subsystem_size": 6,
    "target_entropies": <NumPy real array with shape (3,)>,
    "entropy_weight": 0.25,
    "max_steps": 500,
    "learning_rate": 0.015,
}
```

The Hamiltonian is the open-boundary XXZ chain with a staggered field,

$$H = \sum_{i=0}^{10} (X_i X_{i+1} + Y_i Y_{i+1} + 1.2 Z_i Z_{i+1}) + 0.35 \sum_{i=0}^{11} (-1)^i Z_i.$$

The initial state is the computational basis state `|010101010101>`, where qubit 0 is the leftmost bit.

### Ansatz Structure

The ansatz has three repeated even+odd brickwork blocks. Block `b` consists of two trainable sublayers applied in this order:

1. Even-bond sublayer: apply trainable `RY(theta_y) RZ(theta_z)` rotations on every qubit, then apply trainable nearest-neighbor interactions `exp[-i(theta_xx XX + theta_yy YY + theta_zz ZZ)]` on even bonds `(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)`.
2. Odd-bond sublayer: apply a separate trainable `RY(theta_y) RZ(theta_z)` rotation pair on every qubit, then apply separate trainable nearest-neighbor interactions `exp[-i(theta_xx XX + theta_yy YY + theta_zz ZZ)]` on odd bonds `(1,2), (3,4), (5,6), (7,8), (9,10)`.

There are therefore six trainable sublayers in total, but only three entanglement checkpoints. One block contains `2 * 12 + 3 * 6 + 2 * 12 + 3 * 5 = 81` scalar parameters, so the default ansatz has `3 * 81 = 243` trainable scalars.

### Entanglement Profile

After block `b = 0, 1, 2`, let `|psi_b>` be the normalized output state of that entire even+odd block. Compute the reduced density matrix of the left half,

$$\rho_A^{(b)} = \operatorname{Tr}_{6,7,8,9,10,11}\left(|\psi_b\rangle\langle\psi_b|\right),$$

where subsystem `A` is qubits `0,1,2,3,4,5`. Then compute the second Renyi entropy

$$S_2^{(b)} = -\log\operatorname{Tr}\left[(\rho_A^{(b)})^2\right].$$

The target entropy profile is the evaluator-provided vector

```python
target_entropies = np.array([0.30, 0.60, 0.80], dtype=np.float32)
```

This means the loss should penalize the three deviations `S2_after_block_1 - 0.30`, `S2_after_block_2 - 0.60`, and `S2_after_block_3 - 0.80`. No entropy is required after individual even or odd sublayers.

### Loss Function

Train

$$\mathrm{loss} = E / 12 + 0.25 \cdot \mathrm{mean}\left[(S2_b - S2^{target}_b)^2\right]$$

with Adam at learning rate `0.015` for exactly 500 optimizer updates. Do not use early stopping.

The energy `E` is the expectation value of the XXZ Hamiltonian on the final state after block 3. The entropy term uses only the three block-checkpoint Renyi-2 values defined above. Trainable parameters are initialized from a Gaussian distribution with standard deviation `0.02`.

## Solution Interface

The solution file must be named `solution_2.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only the NumPy-format quantities that the evaluator checks.

Required result keys:

- `initial_energy_density`: scalar float.
- `final_energy_density`: scalar float.
- `final_block_entropies`: NumPy array with shape `(3,)`.
- `final_entropy_mse`: scalar float.
- `initial_loss`: scalar float.
- `final_loss`: scalar float.
- `energy_density_history`: NumPy array with length `config["max_steps"]`.
- `loss_history`: NumPy array with length `config["max_steps"]`.
- `entropy_mse_history`: NumPy array with length `config["max_steps"]`.
- `entropy_history`: NumPy array with shape `(config["max_steps"], 3)`.

The solution may use any quantum software framework, but it must consume only the evaluator-provided configuration and return only this NumPy-format dictionary.

## Evaluation Interface

The evaluator file is `evaluate_2.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_2.py --solution solution_2
```

The evaluator consumes only the returned result dictionary. It independently builds a sparse XXZ Hamiltonian, computes the exact ground-state energy by sparse diagonalization, and prints:

- solution module name,
- end-to-end solution time,
- exact sparse ground-state energy density,
- initial and final energy density,
- final three block entropies,
- target three block entropies,
- entropy-profile mean squared error and root mean squared error,
- initial and final total loss,
- training history lengths and entropy-history shape,
- returned result keys.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default 500-step configuration:

- `len(energy_density_history) == 500`, `len(loss_history) == 500`, and `len(entropy_mse_history) == 500`.
- `entropy_history.shape == (500, 3)`.
- `final_loss < initial_loss`.
- `final_energy_density < initial_energy_density`.
- The reported final entropy-profile MSE matches the MSE recomputed from `final_block_entropies` and `target_entropies`.
- All returned values are NumPy arrays or NumPy-compatible scalars.

The evaluator reports these metrics directly so another framework's `solution_2.py` can be compared without changing `evaluate_2.py`.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_2.py` can be evaluated with:

```bash
python evaluate_2.py --solution solution_2
```

Observed TensorCircuit-NG baseline with the default 500-step configuration:

- End-to-end solution time: `5.62s`.
- Exact sparse ground energy density: `-2.00037162`.
- Initial energy density: `-0.74485052`.
- Final energy density: `-1.99162102`.
- Final block entropies: `[0.295546, 0.586851, 0.266978]`.
- Target entropies: `[0.30, 0.60, 0.80]`.
- Entropy-profile MSE: `9.47683677e-02`.
- Entropy-profile RMSE: `3.07844716e-01`.
- Initial loss: `-0.65404928`.
- Final loss: `-1.96792901`.

The MSE is the mean of the three squared entropy deviations. In the baseline above, the third entropy deviation is about `-0.534`, whose square is about `0.285`; after averaging over three checkpoints this contributes about `0.095` to the MSE. The RMSE is printed because it is on the same scale as the entropy deviations and makes this mismatch more visible.

## Implementation Hint

For a TensorCircuit-NG/JAX baseline, treat one even+odd brickwork block as the repeated unit and scan over the three blocks with `jax.lax.scan`. This keeps the repeated circuit structure compact for JIT compilation while still returning the three block-level entropy values needed by the loss.

The XXZ terms are fixed by the problem definition, so a TensorCircuit-NG baseline can define a matrix-free MVP and compute `<psi|Hpsi>` directly.
