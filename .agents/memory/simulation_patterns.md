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

## Fixed-chi masked MPS/DMRG

- In fixed-bond-dimension masked MPS/DMRG workflows, the mask constrains the local optimization variable, not the stored canonical tensor itself. After QR or other canonicalization steps, boundary tensors can become dense again as part of gauge completion.
- Do not remask canonical tensors after QR just to preserve visual sparsity. That generally breaks canonicality; if exact structural zeros are required, use explicit boundary bond dimensions or a canonicalization scheme defined on the active subspace.

## Pauli and Heisenberg evolution

- Pauli propagation evolves observables backward through the circuit, so apply gates in reversed chronological order.
- When operator coefficients are traced tensors, use functional indexed updates rather than Python-side mutation or direct array assignment.
- If a Heisenberg-style expectation is used as a loss, return its real part explicitly before differentiation.
- Dense Pauli propagation enumerates the full `k`-local Pauli basis, so it is useful as a correctness reference but scales combinatorially in `N` and `k`.
- Sparse Pauli propagation stores only a `buffer_size` set of bit-packed Pauli strings plus coefficients. Its truncation semantics should always be: drop terms above locality `k`, aggregate duplicate codes, then keep top `buffer_size` by coefficient magnitude.
- In sparse Pauli propagation, aggregation must retain only one row per unique code. After sort-and-segment aggregation, non-boundary duplicate rows can still be selected by top-k to fill the buffer and must be zeroed to avoid double counting.
- Sparse Pauli initial-state construction should follow the same aggregate-then-top-k rule as gate propagation; never truncate initial terms by input order before duplicate aggregation.
- Sparse Pauli bit-packing uses signed `int64` words, so keep 31 qubits per word rather than 32. The 32nd two-bit Pauli slot would occupy the sign bit and can overflow or corrupt sentinel handling.
- Current Pauli propagation supports only one- and two-qubit gates. Unsupported higher-arity gates should fail fast instead of being silently ignored or routed through a two-qubit PTM path.
