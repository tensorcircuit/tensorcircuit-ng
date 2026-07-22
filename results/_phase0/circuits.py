"""Parameterized circuit builders so XLA cannot constant-fold (review §5.1). theta is a runtime arg."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")


def build_parameterized_circuit(theta, n, depth):
    """theta: 1-D array of length >= depth*n (rz angles, runtime). Returns a Circuit."""
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    k = 0
    for _ in range(depth):
        for i in range(0, n - 1, 2):
            c.cnot(i, i + 1)
        for i in range(1, n - 1, 2):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rz(i, theta=theta[k])
            k += 1
    return c


def expectation_fn(n, depth):
    """Returns f(theta)->scalar, jax.jit-able, runtime-parametric."""

    def f(theta):
        c = build_parameterized_circuit(theta, n, depth)
        return c.expectation((tc.gates.z(), [0]))

    return jax.jit(f)


def verify_dynamic(theta0, theta1, n, depth):
    f = expectation_fn(n, depth)
    # c.expectation() returns a complex scalar (Hermitian Z -> mathematically real);
    # float() on a complex array raises TypeError, so take the real part.
    v0 = float(f(theta0).real)
    v1 = float(f(theta1).real)
    hlo = str(f.lower(theta0).compiler_ir(dialect="stablehlo"))
    # stablehlo with a runtime param has a function arg and is not a single constant
    has_param = ("%arg" in hlo or "parameter" in hlo.lower()) and not (
        hlo.strip().endswith("}") and "constant" in hlo and "dot_general" not in hlo
    )
    return {
        "output_changes": v0 != v1,
        "hlo_has_runtime_param": bool(has_param),
        "v0": v0,
        "v1": v1,
    }
