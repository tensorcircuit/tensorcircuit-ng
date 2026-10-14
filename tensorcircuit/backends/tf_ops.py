"""
Customized ops for ML framework
"""

# pylint: disable=invalid-name

from typing import Any

import tensorflow as tf

Array = Any  # tensorflow Tensor

qr_epsilon = 1e-8


@tf.custom_gradient  # type: ignore[misc]
def adaware_sqrtmh(value: Array) -> Any:
    """PSD square root with a fixed-rank response at zero eigenvalues."""
    e, v = tf.linalg.eigh(value)
    s = tf.sqrt(tf.maximum(tf.math.real(e), 0))
    result = (v * tf.cast(s[..., None, :], v.dtype)) @ tf.linalg.adjoint(v)

    def backward(gradient: Array) -> Array:
        denominator = s[..., :, None] + s[..., None, :]
        positive = denominator > 0
        denominator = tf.cast(tf.where(positive, denominator, 1), v.dtype)
        gradient = (gradient + tf.linalg.adjoint(gradient)) / 2
        projected = tf.linalg.adjoint(v) @ gradient @ v
        response = tf.where(positive, projected / denominator, 0)
        return v @ response @ tf.linalg.adjoint(v)

    return result, backward


def tfqr_grad(a: Array, q: Array, r: Array, dq: Array, dr: Array) -> Array:
    """Get the gradient for Qr."""

    if (
        r.shape.ndims is None
        or r.shape.as_list()[-2] is None
        or r.shape.as_list()[-1] is None
    ):
        raise NotImplementedError(
            "QrGrad not implemented with dynamic shapes. "
            f"Received r.shape: {r.shape}"
        )
    if (
        r.shape.dims[-2].value > r.shape.dims[-1].value
        and q.shape.dims[-2].value == q.shape.dims[-1].value
    ):
        raise NotImplementedError(
            "QrGrad not implemented when nrows > ncols "
            "and full_matrices is true. Received r.shape="
            f"{r.shape} with nrows={r.shape.dims[-2]}"
            f"and ncols={r.shape.dims[-1]}."
        )

    def _TriangularSolve(x: Array, r: Array) -> Array:
        """Equivalent to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
        return tf.linalg.adjoint(
            tf.linalg.triangular_solve(
                r, tf.linalg.adjoint(x), lower=False, adjoint=False
            )
        )

    def _QrGradSquareAndDeepMatrices(q: Array, r: Array, dq: Array, dr: Array) -> Array:
        """
        Get the gradient for matrix orders num_rows >= num_cols and full_matrices is false.
        """

        # Modification begins
        rdiag = tf.linalg.diag_part(r)
        small_indices = tf.where(tf.math.abs(rdiag) < qr_epsilon)
        length = tf.shape(small_indices)[0]
        newvalues = tf.ones((length,), dtype=rdiag.dtype) * qr_epsilon
        rdiag = tf.tensor_scatter_nd_update(rdiag, small_indices, newvalues)
        r = tf.linalg.set_diag(r, rdiag)
        # Modification ends

        qdq = tf.matmul(q, dq, adjoint_a=True)
        qdq_ = qdq - tf.linalg.adjoint(qdq)
        rdr = tf.matmul(r, dr, adjoint_b=True)
        rdr_ = rdr - tf.linalg.adjoint(rdr)
        tril = tf.linalg.band_part(qdq_ + rdr_, -1, 0)

        grad_a = tf.matmul(q, dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - tf.matmul(q, qdq), r)
        ret = grad_a + grad_b

        if q.dtype.is_complex:
            m = rdr - tf.linalg.adjoint(qdq)
            eyem = tf.linalg.set_diag(tf.zeros_like(m), tf.linalg.diag_part(m))
            correction = eyem - tf.cast(tf.math.real(eyem), q.dtype)
            ret = ret + _TriangularSolve(tf.matmul(q, tf.linalg.adjoint(correction)), r)

        return ret

    num_rows, num_cols = q.shape.dims[-2].value, r.shape.dims[-1]

    if num_rows >= num_cols:
        return _QrGradSquareAndDeepMatrices(q, r, dq, dr)

    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = tf.matmul(q, dv)
    dx = _QrGradSquareAndDeepMatrices(q, u, dq + tf.matmul(y, dv, adjoint_b=True), du)
    return tf.concat([dx, dy], axis=-1)


@tf.custom_gradient  # type: ignore
def tfqr(A: Array) -> Any:
    q, r = tf.linalg.qr(A)

    def grad(dq: Array, dr: Array) -> Any:
        return tfqr_grad(A, q, r, dq, dr)

    return (q, r), grad
