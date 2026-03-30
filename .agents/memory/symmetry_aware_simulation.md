# Symmetry-Aware Simulation (U1Circuit)

1.  **Exploiting Particle Number Conservation**:
    - **Principle**: For systems where the total number of excitations $K$ is preserved, the valid Hilbert space dimension $\binom{N}{K}$ is orders of magnitude smaller than $2^N$.
    - **Scaling**: This enables simulation of large systems (e.g., $N=60, K=4$) that are mathematically impossible in dense simulators.
    - **Protocol**: Perform all calculations (gates, expectations, RDM) directly within the compressed U(1) subspace to maximize performance and minimize memory.

2.  **Schmidt/GEMM Advantage for Entropy**:
    - **Optimization**: Calculating Entanglement Entropy $S = -\sum p_i \ln p_i$ only requires the non-zero eigenvalues of the Reduced Density Matrix (RDM).
    - **GEMM Strategy**: Instead of a full SVD on a rectangular Schmidt matrix $M$ ($D_A \times D_B$), always contract out the larger dimension first to form a smaller covariance matrix: $\rho = M M^\dagger$ if $D_A \le D_B$, else $M^\dagger M$.
    - **Benefit**: This trades a branch-heavy, iterative SVD for a massively parallel GEMM (Matrix Multiplication) and a tiny `eigh`. On GPUs, this is significantly faster and enables analysis of subsystems far larger than can be stored as dense RDMs.

3.  **JIT and Trace-Time Caching**:
    - **Problem**: In backends like JAX, combinatorial logic (like generating basis states via `itertools.combinations`) during the tracing phase can be a major bottleneck for large subspaces.
    - **Protocol**: Refactor structural space mapping (e.g., basis generation, subsystem bit extraction) into module-level helper functions wrapped with `@functools.lru_cache`.
    - **Benefit**: This ensures the expensive "blueprint" of the symmetry sectors is calculated once per $(N, K)$ configuration and reused across re-traces, making variational optimization and repeated analyses nearly instantaneous.

4.  **Backend-Agnostic Bitwise Protection**:
    - **Protocol**: For large qubit counts ($N > 30$), bit manipulation must use `int64` backend tensors to prevent Python/C integer overflows.
    - **Optimization**: Initialize un-mutated vectors (like `z_factor` in `expectation_ps`) as scalar tensors (`1.0`) instead of full arrays. Leverage native backend broadcasting to save memory and improve kernel fusion during JIT.
