# Circuit Models and API

Use this file for cross-cutting circuit invariants, serialization behavior, and model-specific API constraints.

## Class structure

- The main hierarchy is `AbstractCircuit -> BaseCircuit -> Circuit/DMCircuit`.
- `MPSCircuit`, `U1Circuit`, and `StabilizerCircuit` branch directly from `AbstractCircuit`.
- `QuditCircuit` and `AnalogCircuit` are wrappers around an internal dense `Circuit`.
- When adding methods to `AbstractCircuit`, check every subclass initializer and state representation instead of assuming `Circuit`-style construction.

## Inverse and reconstruction semantics

- `AbstractCircuit.inverse()` rebuilds the circuit without carrying through runtime state inputs such as `inputs`, `mps_inputs`, `dminputs`, `tableau_inputs`, `tensors`, or `wavefunction`.
- As a result, an inverted circuit restarts from the default zero state unless the caller explicitly rebuilds it with replacement inputs.
- If downstream code needs `replace_inputs()`, rebuild from QIR with a placeholder input tensor so the reconstructed circuit has a mutable input node.

## Analog circuit inversion

- For an analog block on `[t0, t1]`, invert time-dependent evolution with `-H(t0 + t1 - t)` while keeping the original forward time grid. For time-independent or mutually commuting Hamiltonians, negating `H(t)` alone is sufficient.
- Negating the time array instead tends to send ODE solvers through an invalid backward-time setup and can produce NaNs.

## QIR rules

- `to_qir()` and `from_qir()` are the canonical serialization boundary across circuit types.
- New gates should record enough information in `gatef` and `parameters` for reconstruction without hidden context.
- For noise or channel instructions, store an explicit channel marker and replay them through the same delayed-channel application path during reconstruction. Do not rely on `TypeError` probing or method-name heuristics to distinguish gates from channels.
- Cross-class `from_qir(...)` is expected to work only when the destination class supports the gates in the serialized program and the caller supplies any model-specific constructor arguments through `circuit_params`.

## MPS measurement and truncation diagnostics

- Normalize the conditional state after each sampled outcome, not only the probability vector. Center moves with `normalize=False` otherwise retain the probability of the measured prefix in the state scale and can underflow on long chains. Complete left-to-right sampling can propagate a bond vector through right-canonical tensors without QR; arbitrary measurement orders still need the general center-moving path.
- Track truncation at every state-reduction route, including `reduce_dimension` used by MPO and multi-site gates. Use discarded weight relative to the same post-operator decomposition; a pre-gate norm is not the denominator for nonunitary operators. `get_fidelity()` tracks truncations after input construction, survives normalization, and is not an exact overlap or a postselection probability.
- Zero-norm reductions leave accumulated fidelity unchanged. Protect the denominator before division; masking a `0/0` result afterward can still contaminate gradients under JIT.
