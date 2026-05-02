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

- The inverse of analog evolution should negate the Hamiltonian while keeping the original forward time grid.
- Negating the time array instead tends to send ODE solvers through an invalid backward-time setup and can produce NaNs.

## QIR rules

- `to_qir()` and `from_qir()` are the canonical serialization boundary across circuit types.
- New gates should record enough information in `gatef` and `parameters` for reconstruction without hidden context.
- Cross-class `from_qir(...)` is expected to work only when the destination class supports the gates in the serialized program and the caller supplies any model-specific constructor arguments through `circuit_params`.
