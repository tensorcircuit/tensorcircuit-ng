# Autodiff and Stochastic Optimization

Use this file for complex gradients, backend AD differences, and noisy-gradient workflows.

## Complex AD

- JAX `custom_vjp` for complex outputs expects Wirtinger-style cotangents, which differs from TensorFlow's `custom_gradient` convention.
- When porting a complex gradient rule from TensorFlow to JAX, conjugate incoming cotangents and conjugate the returned gradient. Missing the final conjugation usually produces a sign or complex-conjugate mismatch.
- Wide or tall QR/SVD paths are not safely covered by native JAX AD. Use the repo's custom AD-aware primitives when the matrix geometry is not square.

## Debugging gradient mismatches

- Reduce the problem to a minimal library API call first.
- Then build a pure-backend reproduction of the same primitive chain.
- If the backend-only reproduction fails while the higher-level TensorFlow path is correct, the bug is usually in the backend VJP/JVP implementation rather than the model code.

## Stochastic gradients

- Prefer common random numbers for parameter-shift or Monte Carlo gradients so the positive and negative shifts share the same randomness.
- Do not detect JAX PRNG keys with `try/except` around `random_split`; other backends may accept non-key tensors and silently do the wrong thing.
- TensorFlow-style vectorization materializes a leading batch axis, so stochastic code should be rank-polymorphic rather than assuming single-example tensor ranks.
- Validate stochastic gradients against an exact expectation-based gradient when one exists, and scale tolerances with shot noise instead of using a backend-independent hard threshold.

## Loss shape requirements

- Gradient entry points should return a real scalar. Even physically real expectations should be wrapped with `K.real(...)` before being used as optimization losses.
