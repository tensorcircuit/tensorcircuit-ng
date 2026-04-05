# Testing, Robustness & Benchmarking

## Testing & Robustness

1.  **Pytest Acceleration**:
    - **Rule**: Use `pytest -n auto` to run tests in parallel, which significantly reduces execution time by utilizing multiple CPU cores.
    - **Note**: This requires `pytest-xdist`, which is included in `requirements-dev.txt`.

2.  **Sparse Matrix Compatibility**:
    - TensorCircuit supports multiple sparse formats (Scipy CSR/COO, JAX BCOO, TensorFlow SparseTensor).
    - **Protocol**: When handling sparse outputs, do NOT assume specific attributes like `.row` or `.col` exist (missing in JAX BCOO or Scipy CSR).
    - **Check**: Use `hasattr(obj, "tocoo")` to detect Scipy sparse matrices and convert them if a standard COO interface is needed.
    - **Comparison**: Compare sparse matrices by subtraction (`abs(A - B).max() ≈ 0`) rather than element-wise attribute checks, to be robust against format differences.

## Benchmarking External Libraries

1.  **Environment Isolation**:
    - When benchmarking against external libraries (e.g., QuSpin, Quimb, Qiskit) that may have conflicting dependencies, create a dedicated Conda environment (e.g., `bench_quspin`).
    - Install the external library there using pip from the specific conda environment and run the benchmark script using that specific interpreter.

2.  **Fair Comparison**:
    - Ensure "apples-to-apples" settings (e.g., same precision `complex128`) to support valid performance claims.
