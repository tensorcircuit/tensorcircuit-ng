# Circuit Class Architecture & API Consistency

1.  **Inheritance Hierarchy**:
    - The circuit classes follow: `AbstractCircuit` → `BaseCircuit` → `Circuit` / `DMCircuit`. Side classes (`MPSCircuit`, `U1Circuit`, `StabilizerCircuit`) extend `AbstractCircuit` directly. `QuditCircuit` and `AnalogCircuit` are wrappers that compose an internal `Circuit`.
    - **Protocol**: When adding a method to `AbstractCircuit`, verify it works for all subclasses. Not all share the same `__init__` signature — `MPSCircuit` uses `mps_inputs`, `DMCircuit` uses `dminputs`, etc.

2.  **`inverse()` Filters `inputs` from `circuit_params`**:
    - `AbstractCircuit.inverse()` explicitly excludes `inputs`, `mps_inputs`, `dminputs`, `tableau_inputs`, `tensors`, `wavefunction` from the reconstructed circuit's params (line ~440 of `abstractcircuit.py`). This means inverted circuits always start from `|0⟩`.
    - **Pitfall**: If you need the inverted circuit to support `replace_inputs()` (e.g., in `AnalogCircuit.state()` pipeline), you must rebuild it via `Circuit.from_qir(qir, circuit_params={"nqubits": n, "inputs": placeholder})` to ensure the circuit has a single input tensor node.
    - **Root Cause**: `replace_inputs` requires `self.inputs is not None`. When `inputs` is filtered out, `Circuit.__init__` uses `all_zero_nodes()` which creates separate nodes with no single input tensor to overwrite.

3.  **`AnalogCircuit` Inverse — Negate Hamiltonian, Not Time**:
    - The physical inverse of $e^{-iHT}$ is $e^{+iHT} = e^{-i(-H)T}$, achieved by negating the Hamiltonian.
    - **Pitfall**: Negating the time array (e.g., `[0, 0.5]` → `[-0, -0.5]`) makes the ODE solver receive backward-running time (`times[0] > times[1]`), causing NaN divergence in both `jax.experimental.ode.odeint` and `diffrax`.
    - **Protocol**: Create a wrapper `neg_ham = lambda t, _orig=block.hamiltonian_func: -_orig(t)` and pass it with the original (positive) time array.

4.  **QIR Roundtrip Consistency**:
    - `to_qir()` / `from_qir()` is the canonical serialization path. All circuit classes should produce compatible QIR entries with consistent `gatef` and `parameters` fields.
    - **Protocol**: When adding a new gate to a circuit class, ensure `apply_general_gate` records a proper QIR entry with `gatef` pointing to the gate factory and `parameters` containing all reconstruction-necessary kwargs.
    - **Cross-class conversion**: `ClassA.from_qir(ClassB.to_qir())` works if both classes support the gates in the QIR. Use `circuit_params` dict to pass class-specific init args (e.g., `{"nqubits": n, "k": k}` for `U1Circuit`).
