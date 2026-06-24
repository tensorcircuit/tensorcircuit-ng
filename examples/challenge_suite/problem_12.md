# Problem 12: Variational Circuit to MPS Overlap Optimization

## Goal

Optimize a variational circuit state against a target matrix-product state without converting the target into a gate-preparation circuit. The loss is the direct tensor-network overlap between a DMRG-MPS bra and a trainable circuit ket.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 32,
    "zz_anisotropy": 1.4,
    "staggered_field": 0.2,
    "dmrg_chi": 8,
    "dmrg_sweeps": 4,
    "dmrg_tolerance": 1e-7,
    "n_layers": 2,
    "max_steps": 5000,
    "learning_rate": 0.02,
    "initial_parameter_scale": 0.02,
    "seed": 2039,
    "fidelity_threshold": 0.85,
}
```

Before timing `run_solution(config)`, the evaluator runs DMRG itself and augments this dictionary with:

- `dmrg_state`: the normalized quimb MPS target state.

This runtime key is intentionally supplied by the evaluator so the DMRG preparation time is excluded from solution timing and every solution optimizes against the same deterministic MPS target.

Use an open-boundary 32-qubit XXZ Hamiltonian

```text
H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + 1.4 Z_i Z_{i+1})
    + 0.2 sum_i (-1)^i Z_i.
```

Obtain a normalized DMRG-MPS target state for this Hamiltonian with maximum bond dimension `chi = 8` and four DMRG sweeps. Prepare a variational circuit state from the 32-qubit Neel product state `|0101...01>`. Use two brickwork variational layers. Each layer applies a trainable nearest-neighbor two-qubit `SU4` gate on alternating bonds. The first layer uses bonds `(0,1), (2,3), ...`; the second layer uses bonds `(1,2), (3,4), ...`. Each `SU4` gate has 15 trainable parameters, so the default circuit has `465` trainable parameters.

Optimize all circuit parameters to maximize

```text
F(theta) = |<psi_MPS | psi_circuit(theta)>|^2.
```

Train the loss `1 - F(theta)` with Adam at learning rate `0.02` for exactly 5000 optimizer updates. Do not use early stopping.

## Solution Interface

The solution file must be named `solution_12.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only NumPy-format quantities that the evaluator checks or reports. It should consume the evaluator-provided `dmrg_state` and should not run DMRG internally.

Required result keys:

- `loss_history`: NumPy array with shape `(max_steps,)`.
- `fidelity_history`: NumPy array with shape `(max_steps,)`.
- `final_parameters`: NumPy array with shape `(parameter_count,)`.
- `final_overlap_phase`: scalar float.
- `final_grad_norm`: scalar float.

Each history records one value per optimizer update, evaluated immediately before applying that update. The evaluator recomputes parameter count, derives final fidelity and final loss from the histories, and checks all shapes and scalar quantities from the returned arrays.

## Evaluation Interface

The evaluator file is `evaluate_12.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_12.py --solution solution_12
```

The evaluator computes the DMRG-MPS target before timing, passes that quimb MPS into the solution through `config["dmrg_state"]`, and then consumes only the returned result dictionary. It prints the end-to-end solution time excluding evaluator-side DMRG preparation, initial and final fidelity, final overlap phase, final gradient norm, target MPS bond dimension, history shape, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `loss_history.shape == fidelity_history.shape == (5000,)`.
- `final_parameters.shape == (parameter_count,)`.
- The final fidelity is larger than the initial fidelity.
- The final loss is lower than the initial loss.
- The final fidelity is at least `0.85`.
- The final fidelity is bounded by `1` up to numerical tolerance.
- The target MPS maximum bond dimension is positive and at most `dmrg_chi`.
- All returned arrays and scalar metrics contain finite values.

## TC-NG Baseline

The TensorCircuit-NG solution in `solution_12.py` can be evaluated with:

```bash
python evaluate_12.py --solution solution_12
```

A verified TensorCircuit-NG/JAX baseline run with the default configuration performed `5000` optimizer updates and produced fidelity history shape `(5000,)`, initial fidelity `1.92078686e-09`, final fidelity `8.70016992e-01`, initial loss `1.00000000e+00`, final loss `1.29983008e-01`, final overlap phase `2.27205586e+00`, final gradient norm `4.97358255e-02`, target MPS maximum bond dimension `8`, and overall `PASS` with the `0.85` fidelity threshold. The evaluator-measured `run_solution(config)` time for that run was `14.21s` with `XLA_FLAGS=--xla_disable_hlo_passes=fusion`; this time is a reference measurement only and is not a passing criterion.

## Implementation Hint

Contract the target MPS bra directly with the variational circuit ket as one tensor network, e.g. convert the evaluator-provided quimb MPS with `tc.quantum.quimb2qop(config["dmrg_state"])` and evaluate `target_mps.adjoint() @ circuit.quvector()`. Apply TensorCircuit's built-in `circuit.su4(i, i + 1, theta=theta_15)` on each active bond, where `theta_15` is a length-15 parameter slice. The target MPS does not need to be compiled into a gate-preparation circuit, and the circuit ket does not need to be converted into a dense target vector. In TensorCircuit-NG/JAX, an efficient baseline configures an OMECo contraction path searcher once, JIT-compiles the overlap value-and-gradient function once, and then reuses the compiled executable across the 5000 Adam updates. For this SU4-heavy contraction workload, disabling XLA's HLO fusion pass can substantially reduce first-call compile time while preserving the required fidelity threshold.
