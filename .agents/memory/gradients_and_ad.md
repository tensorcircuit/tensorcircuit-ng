# Gradients, AD & Stochastic Optimization

## Automatic Differentiation (AD) & Complex Numbers

1.  **JAX Complex AD (Wirtinger Calculus)**:
    - **Protocol**: JAX's `custom_vjp` for complex-to-complex functions expects the backward pass to return the Wirtinger derivative ($\partial L / \partial z^*$). In contrast, TensorFlow's `custom_gradient` returns the mathematical gradient ($\nabla_z L = (\partial L / \partial z^*)^*$).
    - **Implementation Guide**: To port a complex gradient from TF to JAX:
      1. Conjugate incoming tangents: `dq, dr = dq.conj(), dr.conj()`.
      2. Perform the gradient calculation logic (same as TF).
      3. Conjugate the final result: `return (result.conj(),)`.
    - **Pitfall**: Forgetting the final `.conj()` on the result, especially in recursive branches (like Wide-Matrix QR), will lead to gradients that mismatch numerical results by a complex conjugate or incorrect sign.

2.  **Wide Matrix QR/SVD AD Instability**:
    - **Observation**: JAX native AD does not support non-square (wide/tall) matrix QR decomposition. TensorCircuit provides custom implementations (`adaware_qr`).

3.  **Isolation Strategy for Gradient Debugging**:
    - **Protocol**: When AD gradients mismatch numerical gradients in a complex system like `MPSCircuit`:
      1. Isolate the **Library API** (e.g. `apply_nqubit_gate`) in a minimal script.
      2. Create a **Pure Backend Chain** (e.g., a chain of 3 `backend.qr` calls) that mimics the internal logic.
      3. Compare both against TensorFlow. If the Pure Backend Chain fails in JAX but the Library API works in TF, the issue is likely in the `jax_ops.py` VJP implementation of the specific primitive.

## Stochastic Gradient Optimization

1.  **Common Random Numbers (CRN) for Variance Reduction**:
    - **Principle**: When calculating gradients of stochastic functions (e.g., via parameter shift), using the same random seed/status for both the positive and negative shifts ($f(x+\Delta, \xi)$ and $f(x-\Delta, \xi)$) significantly reduces the variance of the gradient estimate. The noise $\xi$ partially cancels out in the numerator.
    - **Protocol**: For simulation-based optimization, strictly prefer CRN over independent noise. In `parameter_shift_grad_v2`, this is achieved by providing a status tensor of shape `[size, shots]` (where `size` is the number of parameters).

2.  **Backend-Specific `random_split` Behavior**:
    - **JAX**: `random_split(key)` is a core functional transformation that generates independent subkeys. Passing non-key tensors (like floats) typically raises a `TypeError`.
    - **Other Backends (TF/NumPy)**: In the `AbstractBackend` (and thus TF), `random_split(key)` is often a **silent no-op** that returns `(key, key)` regardless of input type.
    - **Pitfall**: A `try-except` block around `random_split` is **unreliable** for detecting JAX PRNG keys. On TensorFlow, it will "successfully" process status tensors (floats) as keys, leading to incorrect stacking (e.g., `[1, batch, ...]` shapes) and cascading `ValueError` in downstream measurement logic.
    - **Protocol**: Use explicit backend and type checks (e.g., `backend.name == "jax" and not backend.is_tensor(arg)`) to distinguish between JAX PRNG keys and external status tensors.

3.  **Backend Vectorization (Vmap) Implementation Differences**:
    - **JAX (`vmap`)**: Uses tracers that "hide" the batch dimension. Code written for a single instance (e.g., shape `[shots]`) typically works without modification because JAX intercepts shape checks.
    - **TensorFlow (`vectorized_map`)**: Physically prepends a leading batch dimension to all tensors. 
    - **Protocol**: To support TensorFlow vectorization, code must be "rank-polymorphic" (avoiding hardcoded rank checks like `len(shape) == 1`). Currently, high-level stochastic methods in `tensorcircuit` (like `sample_expectation_ps`) may only be differentiable via `vmap` on JAX due to these literal batching conflicts in the TF backend.

4.  **Verification of Stochastic Gradients**:
    - **Protocol**: Always verify stochastic gradient implementations (e.g., those using `sample`) by comparing them against the exact analytical gradient of the same circuit (using `expectation_ps`).
    - **Tolerance**: Use a statistical tolerance proportional to $1/\sqrt{\text{shots}}$. For 1000 shots, an `atol` of `5e-2` is a robust threshold for automated tests.
