# Simulation Patterns

Use this file for model-specific performance patterns that are broader than one bug fix.

## U(1) and symmetry-restricted simulation

- When particle number is conserved, stay inside the compressed U(1) subspace instead of reconstructing dense `2^N` states.
- For entropy or reduced-density-matrix work, form the smaller covariance matrix first and diagonalize that instead of taking a full SVD of the rectangular Schmidt matrix.
- Cache structural basis-generation helpers at module scope so repeated traces do not rebuild the same combinatorial maps.
- For large qubit counts, use `int64` backend tensors for bit operations and rely on broadcasting instead of allocating full helper arrays when a scalar factor is enough.

## Qudit and sparse-model workflows

- Keep qudit code dimension-agnostic. Use loops over `range(d)` or qudit index pairs instead of hardcoding qutrit-specific structure.
- For expressive qudit ansaetze, include both all diagonal phase terms and all off-diagonal mixing terms rather than a shallow qubit-style template.
- For large Hilbert spaces, prefer sparse Hamiltonian construction and the repo's sparse expectation helpers over dense matrix assembly.

## Pauli and Heisenberg evolution

- Pauli propagation evolves observables backward through the circuit, so apply gates in reversed chronological order.
- When operator coefficients are traced tensors, use functional indexed updates rather than Python-side mutation or direct array assignment.
- If a Heisenberg-style expectation is used as a loss, return its real part explicitly before differentiation.
