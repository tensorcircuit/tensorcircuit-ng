"""
Fixed-work pure-JAX Krylov primitives used by TNALG solvers.
"""

from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

Array = Any
_TAYLOR_SUBSTEPS = 8


def _safe_norm(vector: Array) -> Array:
    """
    Use the zero subgradient for a zero norm, without differentiating ``sqrt(0)``.
    """
    squared = jnp.sum(jnp.real(vector * jnp.conj(vector)))
    return jnp.where(squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1)), 0)


@jax.custom_jvp
def _projected_exponential(matrix: Array, coefficient: Array) -> Array:
    """
    Apply a real-symmetric projected exponential to its first basis vector.
    """
    eigenvalues, vectors = jnp.linalg.eigh(matrix)
    return vectors @ (jnp.exp(coefficient * eigenvalues) * vectors[0])


@_projected_exponential.defjvp
def _projected_exponential_jvp(primals: Any, tangents: Any) -> Any:
    matrix, coefficient = primals
    matrix_tangent, coefficient_tangent = tangents
    eigenvalues, vectors = jnp.linalg.eigh(matrix)
    exponentials = jnp.exp(coefficient * eigenvalues)
    half_difference = coefficient * (eigenvalues[:, None] - eigenvalues[None, :]) / 2
    small = jnp.abs(half_difference) < 1e-3
    safe = jnp.where(small, 1, half_difference)
    sinhc = jnp.where(
        small,
        1 + half_difference**2 / 6 + half_difference**4 / 120,
        jnp.sinh(safe) / safe,
    )
    divided_difference = (
        coefficient
        * jnp.exp(coefficient * (eigenvalues[:, None] + eigenvalues[None, :]) / 2)
        * sinhc
    )
    projected_tangent = vectors.T @ matrix_tangent @ vectors
    tangent = vectors @ ((divided_difference * projected_tangent) @ vectors[0])
    tangent = tangent + coefficient_tangent * (
        vectors @ (eigenvalues * exponentials * vectors[0])
    )
    return vectors @ (exponentials * vectors[0]), tangent


def _lanczos_basis(
    matvec: Callable[[Array], Array],
    flat: Array,
    max_dim: int,
    reorthogonalize: bool,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Build a fixed Lanczos basis without tracing one matvec per iteration.
    """
    dimension = flat.shape[0]
    krylov_dim = min(max_dim, dimension)
    if krylov_dim < 1:
        raise ValueError("Krylov vector space must be non-empty")
    vector_norm = _safe_norm(flat)
    current = flat / jnp.where(vector_norm == 0, 1, vector_norm)
    basis = jnp.zeros((krylov_dim, dimension), dtype=flat.dtype)
    previous = jnp.zeros_like(current)
    previous_beta = jnp.asarray(0, dtype=jnp.real(flat).dtype)

    def residual_and_alpha(
        iteration: Array,
        current: Array,
        previous: Array,
        previous_beta: Array,
        basis: Array,
    ) -> Tuple[Array, Array, Array]:
        action = matvec(current)
        action_squared = jnp.real(jnp.vdot(action, action))
        residual = action - previous_beta * previous
        alpha = jnp.real(jnp.vdot(current, residual))
        residual = residual - alpha * current
        if reorthogonalize:
            projections = jnp.einsum("kd,d->k", jnp.conj(basis), residual)
            residual = residual - jnp.einsum(
                "k,k,kd->d", jnp.arange(krylov_dim) <= iteration, projections, basis
            )
        return residual, alpha, action_squared

    def advance(carry: Tuple[Array, ...], iteration: Array) -> Any:
        current, previous, previous_beta, basis, active = carry
        basis = basis.at[iteration].set(jnp.where(active, current, 0))
        residual, alpha, action_squared = residual_and_alpha(
            iteration, current, previous, previous_beta, basis
        )
        last = iteration == krylov_dim - 1
        squared_norm = jnp.sum(jnp.real(residual * jnp.conj(residual)))
        # Numerical breakdown must not normalize roundoff into a new direction.
        tolerance = 8 * jnp.finfo(jnp.real(flat).dtype).eps
        nonzero = squared_norm > tolerance**2 * action_squared
        beta = jnp.where(
            nonzero | last, jnp.sqrt(jnp.where(last | ~nonzero, 1, squared_norm)), 0
        )
        next_active = active & (beta > 0)
        next_vector = residual / jnp.where(beta == 0, 1, beta)
        return (
            (
                jnp.where(next_active, next_vector, current),
                current,
                jnp.where(active, beta, 0),
                basis,
                next_active,
            ),
            (jnp.where(active, alpha, 0), jnp.where(active, beta, 0), active),
        )

    carry, (alphas, betas, basis_active) = jax.lax.scan(
        advance,
        (current, previous, previous_beta, basis, vector_norm > 0),
        jnp.arange(krylov_dim),
    )
    return carry[3], basis_active, alphas, betas[:-1], vector_norm


def _lanczos_exponential(
    matvec: Callable[[Array], Array],
    vector: Array,
    coefficient: Array,
    max_dim: int,
    reorthogonalize: bool = True,
) -> Tuple[Array, dict[str, Array]]:
    """
    Apply ``exp(coefficient * H)`` with a static Lanczos work budget.
    """
    flat = jnp.reshape(vector, (-1,))
    basis, basis_active, alphas, betas, vector_norm = _lanczos_basis(
        matvec, flat, max_dim, reorthogonalize
    )
    krylov_dim = alphas.shape[0]
    projected = jnp.diag(alphas)
    if krylov_dim > 1:
        projected = projected + jnp.diag(betas, 1)
        projected = projected + jnp.diag(betas, -1)
    inactive_energy = jnp.sum(jnp.abs(alphas)) + 2 * jnp.sum(jnp.abs(betas)) + 1
    projected = projected + jnp.diag(jnp.where(basis_active, 0, inactive_energy))
    projected_vector = _projected_exponential(projected, coefficient)
    result = jnp.einsum("kd,k->d", basis, projected_vector) * vector_norm
    return jnp.reshape(result, vector.shape), {
        "finite": jnp.all(jnp.isfinite(result)),
        "breakdown": ~jnp.all(basis_active),
        "residual": betas[-1] if krylov_dim > 1 else jnp.zeros((), dtype=alphas.dtype),
    }


def _taylor_exponential(
    matvec: Callable[[Array], Array], vector: Array, coefficient: Array, max_dim: int
) -> Array:
    """Matrix-free analytic continuation for derivatives at Lanczos breakdown.

    Centering removes the extensive scalar phase. Fixed substeps and a
    degree of at least 16 keep the continuation linear in each JVP tangent;
    increasing the Krylov budget also increases its polynomial degree.
    """
    squared_norm = jnp.real(jnp.vdot(vector, vector))
    shift = jnp.real(jnp.vdot(vector, matvec(vector))) / jnp.where(
        squared_norm > 0, squared_norm, 1
    )
    increment = coefficient / _TAYLOR_SUBSTEPS

    def substep(value: Array, _: Any) -> Tuple[Array, None]:
        def term(
            carry: Tuple[Array, Array], order: Array
        ) -> Tuple[Tuple[Array, Array], None]:
            previous, total = carry
            following = increment * (matvec(previous) - shift * previous) / order
            return (following, total + following), None

        (_, result), _ = jax.lax.scan(
            term, (value, value), jnp.arange(1, max(16, 2 * max_dim) + 1)
        )
        return result, None

    result, _ = jax.lax.scan(substep, vector, None, length=_TAYLOR_SUBSTEPS)
    return jnp.exp(coefficient * shift) * result


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _exponential_with_continuation(
    matvec: Any,
    max_dim: int,
    reorthogonalize: bool,
    vector: Array,
    coefficient: Array,
    constants: Any,
) -> Any:
    return _lanczos_exponential(
        lambda value: matvec(value, *constants),
        vector,
        coefficient,
        max_dim,
        reorthogonalize,
    )


@_exponential_with_continuation.defjvp
def _exponential_jvp(
    matvec: Any, max_dim: int, reorthogonalize: bool, primals: Any, tangents: Any
) -> Any:
    def ordinary(vector: Array, coefficient: Array, constants: Any) -> Any:
        return _lanczos_exponential(
            lambda value: matvec(value, *constants),
            vector,
            coefficient,
            max_dim,
            reorthogonalize,
        )

    primal = ordinary(*primals)

    def continuation(_: Any) -> Any:
        def apply(vector: Array, coefficient: Array, constants: Any) -> Array:
            return _taylor_exponential(
                lambda value: matvec(value, *constants), vector, coefficient, max_dim
            )

        tangent = jax.jvp(apply, primals, tangents)[1]
        return tangent, {
            "finite": jnp.zeros((), dtype=jax.dtypes.float0),
            "breakdown": jnp.zeros((), dtype=jax.dtypes.float0),
            "residual": jnp.zeros_like(primal[1]["residual"]),
        }

    tangent = jax.lax.cond(
        primal[1]["breakdown"],
        continuation,
        lambda _: jax.jvp(ordinary, primals, tangents)[1],
        None,
    )
    return primal, tangent


def expm_action_hermitian(
    matvec: Callable[[Array], Array],
    vector: Array,
    coefficient: Array,
    max_dim: int,
    reorthogonalize: bool = True,
) -> Tuple[Array, dict[str, Array]]:
    """
    Apply a fixed-budget Lanczos exponential with a breakdown-safe first JVP.

    Ordinary points differentiate the Lanczos algorithm. At numerical breakdown,
    its singular basis coordinates are replaced only in the derivative by a
    centered, fixed-work Taylor continuation of the matrix exponential.
    Convergence of that derivative must be checked with the solver budget.
    The forward sweep always uses Lanczos, without an additional matvec body.

    :param matvec: Hermitian matrix-vector product.
    :type matvec: Callable[[Array], Array]
    :param vector: Initial vector to evolve.
    :type vector: Array
    :param coefficient: Scalar multiplying the effective operator in the
        exponential.
    :type coefficient: Array
    :param max_dim: Maximum Lanczos basis dimension.
    :type max_dim: int
    :param reorthogonalize: Whether to reorthogonalize the basis.
    :type reorthogonalize: bool
    :return: Evolved vector and Lanczos diagnostic report.
    :rtype: Tuple[Array, dict[str, Array]]
    """
    shape = vector.shape
    flat = jnp.reshape(vector, (-1,))
    converted, constants = jax.closure_convert(matvec, flat)
    result, report = _exponential_with_continuation(
        converted, max_dim, reorthogonalize, flat, jnp.asarray(coefficient), constants
    )
    return jnp.reshape(result, shape), report


def _lanczos_eigenpair(
    matvec: Callable[[Array], Array],
    flat: Array,
    max_dim: int,
    reorthogonalize: bool,
) -> Tuple[Array, Array, Array]:
    """Return the lowest Ritz pair and whether the basis broke down."""
    basis, basis_active, alphas, betas, _ = _lanczos_basis(
        matvec, flat, max_dim, reorthogonalize
    )
    krylov_dim = alphas.shape[0]
    projected = jnp.diag(alphas)
    if krylov_dim > 1:
        projected = projected + jnp.diag(betas, 1)
        projected = projected + jnp.diag(betas, -1)
    inactive_energy = jnp.sum(jnp.abs(alphas)) + 2 * jnp.sum(jnp.abs(betas)) + 1
    # Distinct inactive roots keep the unused ordinary JVP finite under vmap.
    projected = projected + jnp.diag(
        jnp.where(basis_active, 0, inactive_energy * (1 + jnp.arange(krylov_dim)))
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(projected)
    candidate = jnp.einsum("kd,k->d", basis, eigenvectors[:, 0])
    candidate_norm = _safe_norm(candidate)
    candidate = candidate / jnp.where(candidate_norm == 0, 1, candidate_norm)
    return candidate, jnp.real(eigenvalues[0]), ~jnp.all(basis_active)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _eigenpair_with_response(
    matvec: Any,
    max_dim: int,
    reorthogonalize: bool,
    vector: Array,
    constants: Any,
) -> Any:
    return _lanczos_eigenpair(
        lambda value: matvec(value, *constants), vector, max_dim, reorthogonalize
    )


@_eigenpair_with_response.defjvp
def _eigenpair_jvp(
    matvec: Any, max_dim: int, reorthogonalize: bool, primals: Any, tangents: Any
) -> Any:
    vector, constants = primals
    vector_tangent, constants_tangent = tangents
    # Higher-order AD must reuse the response rule for the primal eigenpair.
    primal = _eigenpair_with_response(matvec, max_dim, reorthogonalize, *primals)
    candidate, eigenvalue, breakdown = primal

    def project(value: Array) -> Array:
        return value - candidate * jnp.vdot(candidate, value)

    action_tangent = jax.jvp(
        lambda parameters: matvec(candidate, *parameters),
        (constants,),
        (constants_tangent,),
    )[1]
    energy_tangent = jnp.real(jnp.vdot(candidate, action_tangent))

    def shifted(value: Array) -> Array:
        perpendicular = project(value)
        return project(
            matvec(perpendicular, *constants) - eigenvalue * perpendicular
        ) + candidate * jnp.vdot(candidate, value)

    def solve(action: Any, rhs: Array) -> Array:
        return cg(
            action,
            rhs,
            tol=8 * jnp.finfo(jnp.real(vector).dtype).eps,
            maxiter=max_dim,
        )[0]

    # Keep RHS-dependent CG setup inside the linear differentiation boundary.
    tangent = jax.lax.custom_linear_solve(
        shifted,
        jnp.where(breakdown, -project(action_tangent), 0),
        solve=solve,
        transpose_solve=solve,
    )
    tangent = project(tangent)
    if jnp.issubdtype(vector.dtype, jnp.complexfloating):
        # Lanczos fixes the phase through a real overlap with its seed.
        overlap = jnp.real(jnp.vdot(vector, candidate))
        phase = jnp.imag(
            jnp.vdot(vector_tangent, candidate) + jnp.vdot(vector, tangent)
        ) / jnp.where(overlap == 0, 1, overlap)
        tangent = tangent - 1j * phase * candidate

    def ordinary(initial: Array, parameters: Any) -> Any:
        return _lanczos_eigenpair(
            lambda value: matvec(value, *parameters), initial, max_dim, reorthogonalize
        )

    ordinary_tangent = jax.jvp(ordinary, primals, tangents)[1]
    return primal, (
        jnp.where(breakdown, tangent, ordinary_tangent[0]),
        jnp.where(breakdown, energy_tangent, ordinary_tangent[1]),
        jnp.zeros((), dtype=jax.dtypes.float0),
    )


def lowest_eigenvector_hermitian(
    matvec: Callable[[Array], Array],
    vector: Array,
    max_dim: int,
    reorthogonalize: bool = True,
    return_report: bool = True,
) -> Any:
    """
    Find a lowest Ritz vector using a fixed-work Lanczos iteration.

    Without breakdown, derivatives follow the finite Lanczos algorithm. At
    numerical breakdown, use the implicit response of an isolated ground state
    instead of differentiating singular basis coordinates. The matrix-free
    response solve uses at most ``max_dim`` conjugate-gradient iterations;
    check derivative convergence by increasing this budget. This rule requires
    a nonzero gap in the supplied vector space, including a fixed symmetry
    sector. It does not define derivatives at a degenerate ground state or
    guarantee that an invariant starting subspace contains the ground state.
    The complex vector's phase follows its real overlap with the initial vector.

    :param matvec: Hermitian matrix-vector product.
    :type matvec: Callable[[Array], Array]
    :param vector: Initial vector for the Lanczos iteration.
    :type vector: Array
    :param max_dim: Maximum Lanczos basis dimension and maximum number of
        iterations in the breakdown response solve.
    :type max_dim: int
    :param reorthogonalize: Whether to reorthogonalize the basis.
    :type reorthogonalize: bool
    :param return_report: Whether to return the Lanczos diagnostic report.
    :type return_report: bool
    :return: Lowest Ritz vector and eigenvalue, optionally with diagnostics.
    :rtype: Any
    """
    flat = jnp.reshape(vector, (-1,))
    converted, constants = jax.closure_convert(matvec, flat)
    candidate, eigenvalue, breakdown = _eigenpair_with_response(
        converted, max_dim, reorthogonalize, flat, constants
    )
    result = jnp.reshape(candidate, vector.shape), eigenvalue
    if not return_report:
        return result
    residual_norm = _safe_norm(matvec(candidate) - eigenvalue * candidate)
    return (
        *result,
        {
            "finite": jnp.all(jnp.isfinite(candidate)),
            "breakdown": breakdown,
            "residual": residual_norm,
        },
    )
