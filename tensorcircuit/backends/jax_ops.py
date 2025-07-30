"""
Customized ops for ML framework
"""

# pylint: disable=invalid-name
# pylint: disable=unused-variable


from typing import Any, Tuple, Sequence
from functools import partial

import jax
import jax.numpy as jnp

Array = Any  # jnp.array


@jax.custom_vjp
def adaware_svd(A: Array) -> Any:
    u, s, v = jnp.linalg.svd(A, full_matrices=False)
    return (u, s, v)


def _safe_reciprocal(x: Array, epsilon: float = 1e-15) -> Array:
    return x / (x * x + epsilon)


def jaxsvd_fwd(A: Array) -> Any:
    u, s, v = adaware_svd(A)
    return (u, s, v), (u, s, v)


def jaxsvd_bwd(r: Sequence[Array], tangents: Sequence[Array]) -> Tuple[Array]:
    # https://arxiv.org/pdf/1909.02659.pdf
    u, s, v = r
    du, ds, dv = tangents
    v = jnp.conj(jnp.transpose(v))
    dv = jnp.conj(jnp.transpose(dv))
    m = u.shape[-2]
    n = v.shape[-2]

    F = s * s - (s * s)[:, None]
    F = _safe_reciprocal(F) - jnp.diag(jnp.diag(_safe_reciprocal(F)))
    # note the double character of ``jnp.diag``` ...
    S = jnp.diag(s)
    dAs = jnp.conj(u) @ jnp.diag(ds) @ jnp.transpose(v)

    J = F * (jnp.transpose(u) @ du)
    dAu = jnp.conj(u) @ (J + jnp.transpose(jnp.conj(J))) @ S @ jnp.transpose(v)

    K = F * (jnp.transpose(v) @ dv)
    dAv = jnp.conj(u) @ S @ (K + jnp.conj(jnp.transpose(K))) @ jnp.transpose(v)

    Sinv = jnp.diag(_safe_reciprocal(s))
    L = jnp.diag(jnp.diag(jnp.transpose(v) @ dv)) @ Sinv
    dAc = 1 / 2.0 * jnp.conj(u) @ (jnp.conj(L) - L) @ jnp.transpose(v)

    grad_a = dAv + dAu + dAs + dAc

    if m > n:
        grad_a += (du - jnp.conj(u) @ jnp.transpose(u) @ du) @ Sinv @ jnp.transpose(v)
    elif m < n:
        grad_a += (
            jnp.conj(u)
            @ Sinv
            @ jnp.transpose(jnp.conj(dv))
            @ (jnp.eye(n) - jnp.conj(v) @ jnp.transpose(v))
        )
    # m=n do nothing

    return (grad_a,)


adaware_svd.defvjp(jaxsvd_fwd, jaxsvd_bwd)  # type: ignore

adaware_svd_jit = jax.jit(adaware_svd)


qr_epsilon = 1e-8


@jax.custom_vjp
def adaware_qr(A: Array) -> Any:
    q, r = jnp.linalg.qr(A)
    return (q, r)


def jaxqr_fwd(A: Array) -> Any:
    q, r = adaware_qr(A)
    return (q, r), (A, q, r)


def jaxqr_bwd(res: Sequence[Array], tangents: Sequence[Array]) -> Tuple[Array]:
    a, q, r = res
    dq, dr = tangents
    dq = dq.conj()
    dr = dr.conj()

    def _TriangularSolve(x: Array, r: Array) -> Array:
        return jax.scipy.linalg.solve_triangular(  # type: ignore
            r, x.T.conj(), lower=False, trans=0
        ).T.conj()

    def _QrGradSquareAndDeepMatrices(q: Array, r: Array, dq: Array, dr: Array) -> Array:
        # Modification begins
        rdiag = jnp.diag(r)
        rdiag = (jnp.abs(rdiag) < qr_epsilon) * qr_epsilon + (
            jnp.abs(rdiag) >= qr_epsilon
        ) * rdiag
        diag_indices = jnp.arange(min(r.shape[-2:]))
        r = r.at[diag_indices, diag_indices].set(rdiag)
        # Modification ends

        qdq = q.T.conj().dot(dq)
        qdq_ = qdq - qdq.T.conj()
        rdr = r.dot(dr.T.conj())
        rdr_ = rdr - rdr.T.conj()
        tril = jnp.tril(qdq_ + rdr_)

        grad_a = q.dot(dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - q.dot(qdq), r)
        ret = grad_a + grad_b

        if jnp.iscomplexobj(q):
            m = rdr - qdq.T.conj()
            length = m.shape[0]
            eyem = jnp.zeros_like(m)
            eyem = eyem.at[jnp.arange(length), jnp.arange(length)].set(jnp.diag(m))
            correction = eyem - jnp.real(eyem)
            ret = ret + _TriangularSolve(q.dot(correction.T.conj()), r)

        return ret.conj()

    num_rows, num_cols = q.shape[-2], r.shape[-1]

    if num_rows >= num_cols:
        result = _QrGradSquareAndDeepMatrices(q, r, dq, dr)
        return (result,)

    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = q.dot(dv)
    dx = _QrGradSquareAndDeepMatrices(q, u, dq + y.dot(dv.T.conj()), du)
    result = jnp.concatenate([dx, dy], axis=-1)
    return (result,)


adaware_qr.defvjp(jaxqr_fwd, jaxqr_bwd)  # type: ignore

adaware_qr_jit = jax.jit(adaware_qr)


@jax.custom_vjp
def adaware_eigh(A: Array) -> Array:
    e, v = jnp.linalg.eigh(A)
    return e, v


def jaxeigh_fwd(A: Array) -> Array:
    e, v = jnp.linalg.eigh(A)
    return (e, v), (A, e, v)


def jaxeigh_bwd(r: Array, tangents: Array) -> Array:
    a, e, v = r
    de, dv = tangents
    eye_n = jnp.eye(a.shape[-1], dtype=a.dtype)
    f = _safe_reciprocal(e[..., jnp.newaxis, :] - e[..., jnp.newaxis] + eye_n) - eye_n
    middle = jnp.diag(de) + jnp.multiply(f, (v.T @ dv))
    grad_a = jnp.conj(v) @ middle @ v.T
    return (grad_a,)


# denegerate eigev values lead nan in eigh gradients, while tf has fixed that long ago

adaware_eigh.defvjp(jaxeigh_fwd, jaxeigh_bwd)
adaware_eigh_jit = jax.jit(adaware_eigh)


@partial(jax.jit, static_argnums=[0, 2])
def bessel_jv_jax_rescaled(k: int, x: jnp.ndarray, M: int) -> jnp.ndarray:
    """
    Computes Bessel function Jv using Miller's algorithm with dynamic rescaling,
    implemented in JAX.
    """
    if M <= k:
        raise ValueError(
            f"Recurrence length M ({M}) must be greater than the required order k ({k})."
        )

    # Use vmap to handle array inputs for x efficiently.
    # We map _bessel_jv_scalar_rescaled over the last dimension of x.
    return _bessel_jv_scalar_rescaled(k, M, x)


def _bessel_jv_scalar_rescaled(k: int, M: int, x_val: jnp.ndarray) -> jnp.ndarray:
    """
    JAX implementation for a scalar input x_val.
    This function will be vmapped for array inputs.
    """
    rescale_threshold = 1e250

    # Define the body of the recurrence loop
    def recurrence_body(i, state):  # type: ignore
        # M - i is the current 'm' value in the original loop.
        # Loop from M down to 1. jax.lax.fori_loop goes from lower to upper-1.
        # So for m from M down to 1, we map i from 0 to M-1.
        # Current m_val = M - i
        # The loop range for m in numpy was `range(M, 0, -1)`, which means m goes from M, M-1, ..., 1.
        # For lax.fori_loop (start, stop, body_fn, init_val), start is inclusive, stop is exclusive.
        # So to iterate M times for m from M down to 1, we do i from 0 to M-1.
        # m_val = M - i means that for i=0, m_val=M; for i=M-1, m_val=1.
        m_val = M - i
        f_m, f_m_p1, f_vals = state

        # If x_val is near zero, this division could be an issue,
        # but the outer lax.cond handles the x_val near zero case before this loop runs.
        f_m_m1 = (2.0 * m_val / x_val) * f_m - f_m_p1

        # --- Rescaling Step ---
        # jax.lax.cond requires all branches to return the exact same type and shape.
        def rescale_branch(vals):  # type: ignore
            f_m_val, f_m_p1_val, f_vals_arr = vals
            scale_factor = f_m_m1
            # Return new f_m, f_m_p1, updated f_vals_arr, and the new f_m_m1 value (which is 1.0)
            return (
                f_m_val / scale_factor,
                f_m_p1_val / scale_factor,
                f_vals_arr / scale_factor,
                1.0,
            )

        def no_rescale_branch(vals):  # type: ignore
            f_m_val, f_m_p1_val, f_vals_arr = (
                vals  # Unpack to keep signatures consistent
            )
            # Return original f_m, f_m_p1, original f_vals_arr, and the computed f_m_m1
            return (f_m_val, f_m_p1_val, f_vals_arr, f_m_m1)

        f_m_rescaled, f_m_p1_rescaled, f_vals_rescaled, f_m_m1_effective = jax.lax.cond(
            jnp.abs(f_m_m1) > rescale_threshold,
            rescale_branch,
            no_rescale_branch,
            (f_m, f_m_p1, f_vals),  # Arguments passed to branches
        )

        # Update f_vals at index m_val - 1. JAX uses .at[idx].set(val) for non-in-place updates.
        f_vals_updated = f_vals_rescaled.at[m_val - 1].set(f_m_m1_effective)

        # Return new state for the next iteration: (new f_m, new f_m_p1, updated f_vals)
        return (f_m_m1_effective, f_m_rescaled, f_vals_updated)

    # Initial state for the recurrence loop
    f_m_p1_init = 0.0
    f_m_init = 1e-30  # Start with a very small number
    f_vals_init = jnp.zeros(M + 1).at[M].set(f_m_init)

    # Use jax.lax.fori_loop for the backward recurrence
    # Loop from i = 0 to M-1 (total M iterations)
    # The 'body' function gets current 'i' and 'state', returns 'new_state'.
    # We don't need the final f_m_p1, only f_m and f_vals.
    final_f_m, _, f_vals = jax.lax.fori_loop(
        0, M, recurrence_body, (f_m_init, f_m_p1_init, f_vals_init)
    )

    # Normalization using Neumann's sum rule
    even_sum = jnp.sum(f_vals[2::2])
    norm_const = f_vals[0] + 2.0 * even_sum

    # Handle division by near-zero normalization constant
    norm_const_safe = jnp.where(jnp.abs(norm_const) < 1e-12, 1e-12, norm_const)

    # Conditional logic for x_val close to zero
    def x_is_zero_case() -> jnp.ndarray:
        # For x=0, J_0(0)=1, J_k(0)=0 for k>0
        return jnp.zeros(k).at[0].set(1.0)

    def x_is_not_zero_case() -> jnp.ndarray:
        return f_vals[:k] / norm_const_safe  # type: ignore

    # Use lax.cond to select between the two cases based on x_val
    return jax.lax.cond(jnp.abs(x_val) < 1e-12, x_is_zero_case, x_is_not_zero_case)  # type: ignore
