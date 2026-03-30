# Backend Quirks & API Wrapper Behavior

1.  **JAX Vmap in TensorCircuit Wrapper**:
    - When using `tc.backend.vmap` (alias `K.vmap`) with the JAX backend, do **not** use JAX-native arguments like `in_axes`. The TC wrapper unifies behavior and exposes `vectorized_argnums` (like TensorFlow) instead.
    - **Protocol**: Always use `vectorized_argnums=(0, 1, ...)` to specify batched arguments, regardless of the backend (JAX/TF/Torch). Passing `in_axes` will raise a `TypeError` because the wrapper function definition doesn't accept it.

2.  **Common Backend APIs & Missing Methods**:
    - The `tc.backend` interface provides unified methods like `K.zeros_like`, `K.ones_like`, and `K.scatter`.
    - **Protocols for Missing Methods**:
      - `K.any` -> `K.sum(probs) > 0`
      - `K.minimum(a,b)` -> `K.where(a < b, a, b)`
      - `K.outer(a, b)` -> `a[:, None] * b[None, :]` (Broadcasting)

3.  **JIT Buffer Management & Dynamic Updates**:
    - Updating a fixed-size buffer with a dynamic number of new terms in JAX JIT is challenging due to static shape requirements.
    - **Pitfall**: `jax.lax.dynamic_update_slice` requires static slice shapes and can behave unexpectedly if logic implies dynamic sizes.
    - **Protocol**: Use **masked updates** (`K.where`). Create a mask for valid insertion indices and map source indices to destination indices using modular arithmetic or standard indexing, masked by the valid region. This maintains static graph shapes.

4.  **Lexsort Priority Consistency**:
    - **Protocol**: In TensorCircuit, `lexsort(keys)` consistently treats the **last** key in the sequence as the **primary** sort key, following the NumPy convention.
    - **Pitfall**: Different backend libraries might have different default priorities (e.g. TensorFlow's `argsort` doesn't natively support lexsort, and custom implementations must be careful). Always verify and test against NumPy's behavior.

5.  **Migrating from PyTorch/TF to JAX**:
    - **Protocol**: Always prefer the JAX backend for high-performance quantum kernels.
    - **Hybrid Workflow**: If the project requires PyTorch (e.g., for specific NN layers or legacy optimizers), use the `jax` backend for the quantum part and bridge it using `tc.interfaces.torch_interface`. This maintains an end-to-end differentiable graph while leveraging JAX's JIT and vectorization for the quantum bottle-necks.

6.  **Numpy Operations on Backend Tensors**:
    - Avoid using native `numpy` functions on backend tensors (like `np.diag(tensor)`). This breaks JAX tracing, as the tensor might be an abstract Tracer during `jit`.
    - **Protocol**: Always use the equivalent backend method such as `tc.backend.diagflat(tensor)` instead of `np.diag`.
