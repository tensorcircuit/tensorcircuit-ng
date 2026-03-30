# Qudit Simulation & Advanced Models

1.  **Ansatz Expressibility**:
    - For $d > 2$ (Qudits), simple "Hardware Efficient" ansätze (single layer of rotations) are often insufficient to reach ground states of complex Hamiltonians (e.g. Potts model).
    - **Protocol**: Ensure high expressibility by parameterizing _all_ $d$ diagonal phases (`rz` on levels $0 \dots d-1$) and _all_ off-diagonal mixing angles (`ry` on pairs $(j, k)$).
    - **Dimension Agnostic Code**: Write code using variables `d` and loops `range(d)` instead of hardcoding `d=3`. This allows the same script to simulate qutrits, ququarts, etc. seamlessly.

2.  **Sparse Matrix Hamiltonian**:
    - For larger Hilbert spaces ($d^N \gg 10^3$), dense matrix construction explodes in memory.
    - **Protocol**: Construct Hamiltonians using `scipy.sparse` (COO format), but **strictly prefer** `tc.quantum.PauliStringSum2COO` for Pauli sums as it is significantly faster than manual `kron` construction.
    - **Integration**: Convert to JAX Sparse via `tc.backend.coo_sparse_matrix(indices, values, shape)` and use `tc.templates.measurements.sparse_expectation` or `tc.templates.measurements.mpo_expectation` for large system evaluations. This provides massive speedups and enables simulation of larger $N$ or $d$.
