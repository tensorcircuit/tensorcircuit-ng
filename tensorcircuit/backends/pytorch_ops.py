"""
Customized ops for ML framework
"""

# pylint: disable=invalid-name

from typing import Any

import torch  # type: ignore

Array = Any  # torch Tensor

qr_epsilon = 1e-8


class _PSDSquareRoot(torch.autograd.Function):
    """Differentiate the PSD square root using cached spectral factors."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value: Array, s: Array, v: Array) -> Array:
        return (v * s[..., None, :]) @ v.adjoint()

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Array) -> None:
        ctx.save_for_backward(inputs[1], inputs[2])

    @staticmethod
    def backward(ctx: Any, gradient: Array) -> Any:
        s, v = ctx.saved_tensors
        denominator = s[..., :, None] + s[..., None, :]
        positive = denominator > 0
        gradient = (gradient + gradient.adjoint()) / 2
        projected = v.adjoint() @ gradient @ v
        response = torch.where(
            positive, projected / torch.where(positive, denominator, 1), 0
        )
        return v @ response @ v.adjoint(), None, None


def sqrtmh_psd(value: Array) -> Array:
    """PSD square root with a fixed-rank response at zero eigenvalues."""
    # Keep factors in the graph for higher derivatives, without repeating eigh.
    e, v = torch.linalg.eigh(value)
    s = torch.sqrt(torch.clamp(e, min=0))
    return _PSDSquareRoot.apply(value, s, v)


def torchqr_grad(
    a: Array,
    q: Array,
    r: Array,
    dq: Array,
    dr: Array,
) -> Array:
    """Get the gradient for Qr."""

    if r.shape[-2] > r.shape[-1] and q.shape[-2] == q.shape[-1]:
        raise NotImplementedError(
            "QrGrad not implemented when nrows > ncols "
            "and full_matrices is true. Received r.shape="
            f"{r.shape} with nrows={r.shape[-2]}"
            f"and ncols={r.shape[-1]}."
        )

    def _TriangularSolve(x: Array, r: Array) -> Array:
        """Equivalent to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
        return torch.linalg.solve_triangular(
            r, x.adjoint(), upper=True, unitriangular=False
        ).adjoint()

    def _QrGradSquareAndDeepMatrices(q: Array, r: Array, dq: Array, dr: Array) -> Array:
        """
        Get the gradient for matrix orders num_rows >= num_cols and full_matrices is false.
        """

        # Modification begins
        rdiag = torch.linalg.diagonal(r)
        # if abs(rdiag[i]) < qr_epsilon then rdiag[i] = qr_epsilon otherwise keep the old value
        qr_epsilon_diag = torch.ones_like(rdiag) * qr_epsilon
        rdiag = torch.where(rdiag.abs() < qr_epsilon, qr_epsilon_diag, rdiag)
        r = torch.diagonal_scatter(r, rdiag, dim1=-2, dim2=-1)
        # Modification ends

        qdq = torch.matmul(q.adjoint(), dq)
        qdq_ = qdq - qdq.adjoint()
        rdr = torch.matmul(r, dr.adjoint())
        rdr_ = rdr - rdr.adjoint()
        tril = torch.tril(qdq_ + rdr_)

        grad_a = torch.matmul(q, dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - torch.matmul(q, qdq), r)
        ret = grad_a + grad_b

        if q.is_complex():
            m = rdr - qdq.adjoint()
            eyem = torch.diagonal_scatter(
                torch.zeros_like(m), torch.linalg.diagonal(m), dim1=-2, dim2=-1
            )
            correction = eyem - torch.real(eyem).to(dtype=q.dtype)
            ret = ret + _TriangularSolve(torch.matmul(q, correction.adjoint()), r)

        return ret

    num_rows, num_cols = q.shape[-2], r.shape[-1]

    if num_rows >= num_cols:
        return _QrGradSquareAndDeepMatrices(q, r, dq, dr)

    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = torch.matmul(q, dv)
    dx = _QrGradSquareAndDeepMatrices(q, u, dq + torch.matmul(y, dv.adjoint()), du)
    return torch.cat([dx, dy], dim=-1)


class torchqr(torch.autograd.Function):
    """
    Customized backward of qr for better numerical stability
    """

    @staticmethod
    def forward(a: Array) -> Any:
        q, r = torch.linalg.qr(a, mode="reduced")
        return q, r

    # setup_context is responsible for calling methods and/or assigning to
    # the ctx object. Please do not do additional compute (e.g. add
    # Tensors together) in setup_context.
    # https://pytorch.org/docs/master/notes/extending.func.html
    @staticmethod
    def setup_context(ctx, inputs, output):
        (a,) = inputs
        q, r = output
        # Tensors must be saved via ctx.save_for_backward. Please do not
        # assign them directly onto the ctx object.
        ctx.save_for_backward(a, q, r)
        # Non-tensors may be saved by assigning them as attributes on the ctx object.
        # ctx.dim = dim

    @staticmethod
    def backward(ctx, dq: Array, dr: Array) -> Any:
        a, q, r = ctx.saved_tensors
        return torchqr_grad(a, q, r, dq, dr)
