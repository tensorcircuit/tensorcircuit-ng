"""
Backend-aware matrix-function actions, Ritz eigenpairs, and spectral measures.

The public routines in this module operate on dense, sparse, ``LinearOperator``
and matrix-vector-product inputs.  Solver sizes are Python values so that the
recurrences have fixed shapes under JIT; query values remain backend tensors.
"""

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence, Tuple, Union

from .cons import backend, dtypestr, idtypestr, rdtypestr
from .quantum import aslinearoperator

Tensor = Any


def _as_tensor(value: Any) -> Tensor:
    """
    Convert scalars and simple Python sequences through the active backend.
    """
    try:
        return backend.convert_to_tensor(value)
    except (TypeError, ValueError):
        if isinstance(value, (list, tuple)):
            return backend.stack([_as_tensor(v) for v in value])
        raise


def _rank(value: Tensor) -> int:
    return len(backend.shape_tuple(value))


def _concrete_float(value: Tensor) -> Optional[float]:
    """
    Return an eager scalar as a Python float, or ``None`` for a tracer.
    """
    try:
        eager_value = backend.numpy(value)
        if hasattr(eager_value, "real"):
            eager_value = eager_value.real
        return float(eager_value)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _concrete_query_failure_details(mask: Tensor, query: Tensor) -> str:
    """Return concise eager diagnostics for invalid scalar or rank-one queries."""
    try:
        flags = backend.numpy(mask).tolist()
        values = backend.numpy(query).tolist()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return ""
    if _rank(query) == 0:
        return f" for z={values!r}"
    indices = [index for index, flag in enumerate(flags) if flag]
    shown = indices[:8]
    shown_values = [values[index] for index in shown]
    suffix = "..." if len(indices) > len(shown) else ""
    return (
        f" for {len(indices)} query rows; first failing indices {shown}{suffix} "
        f"have z={shown_values}{suffix}"
    )


def _check_query(value: Any, name: str) -> Tuple[Tensor, bool]:
    tensor = _as_tensor(value)
    rank = _rank(tensor)
    if rank not in (0, 1):
        raise ValueError(f"{name} must be a scalar or rank-one tensor.")
    return tensor, rank == 0


def _check_bounds(bounds: Sequence[float]) -> Tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("bounds must contain (emin, emax).")
    emin, emax = float(bounds[0]), float(bounds[1])
    if not math.isfinite(emin) or not math.isfinite(emax) or emin >= emax:
        raise ValueError("bounds must be finite and satisfy emin < emax.")
    return emin, emax


@dataclass(frozen=True)
class KrylovConfig:
    """
    Configuration for a fixed-shape Hermitian Lanczos recurrence.

    A Lanczos run approximates ``f(H) @ vector`` or the lowest Ritz eigenpair by
    projecting the Hermitian operator ``H`` onto at most ``max_dim`` basis
    vectors. The size is fixed before tracing so the recurrence can be compiled
    by JAX and reused for different vectors or function queries.

    :ivar max_dim: Maximum number of Krylov basis vectors.
    :ivar reorthogonalization: Whether to apply full reorthogonalization or only
        the three-term recurrence.
    :ivar breakdown_tol: Relative residual threshold for padded recurrence entries;
        ``None`` selects a dtype-dependent default.
    """

    max_dim: int
    reorthogonalization: Literal["none", "full"] = "full"
    breakdown_tol: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_dim, int) or isinstance(self.max_dim, bool):
            raise TypeError("max_dim must be an integer.")
        if self.max_dim < 1:
            raise ValueError("max_dim must be >= 1.")
        if self.reorthogonalization not in ("none", "full"):
            raise ValueError("reorthogonalization must be 'none' or 'full'.")
        if self.breakdown_tol is not None:
            if not math.isfinite(float(self.breakdown_tol)) or self.breakdown_tol < 0:
                raise ValueError("breakdown_tol must be a non-negative finite number.")


@dataclass(frozen=True)
class ChebyshevConfig:
    """
    Configuration for a Chebyshev polynomial recurrence.

    The operator spectrum must lie inside ``bounds=(emin, emax)``. The
    recurrence applies Chebyshev polynomials to the affine-rescaled operator
    ``(H - (emin + emax) / 2) / ((emax - emin) / 2)``.

    :ivar order: Number of Chebyshev moments or coefficients.
    :ivar bounds: Ascending spectral bounds ``(emin, emax)`` enclosing the Hermitian operator spectrum.
    :ivar kernel: DOS damping kernel. The recurrence itself does not use this
        field; it is consumed by :func:`kernel_weights`.
    :ivar lorentz_lambda: Positive Lorentz damping parameter.
    :ivar scaling_steps: Number of fixed repeated steps used by
        :func:`exponential_action`. Direct coefficient generation and
        polynomial actions remain single-step operations. For half
        imaginary-time evolution with ``energy_shift`` near ``E_min``, a
        complex128 starting heuristic is ``0.5 * beta * (E_min - bounds[0]) /
        scaling_steps <= 15``; this is not an error guarantee, and complex64
        generally requires a lower value.
    :ivar tail_decay_tolerance: Positive threshold for the final-to-leading
        resolvent coefficient magnitude ratio. This is a decay diagnostic,
        not a relative-error tolerance.
    """

    order: int
    bounds: Tuple[float, float]
    kernel: Literal["jackson", "lorentz", "dirichlet"] = "jackson"
    lorentz_lambda: float = 4.0
    scaling_steps: int = 1
    tail_decay_tolerance: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise TypeError("order must be an integer.")
        if self.order < 1:
            raise ValueError("order must be >= 1.")
        object.__setattr__(self, "bounds", _check_bounds(self.bounds))
        if self.kernel not in ("jackson", "lorentz", "dirichlet"):
            raise ValueError("kernel must be jackson, lorentz, or dirichlet.")
        if not math.isfinite(float(self.lorentz_lambda)) or self.lorentz_lambda <= 0:
            raise ValueError("lorentz_lambda must be positive and finite.")
        if not isinstance(self.scaling_steps, int) or isinstance(
            self.scaling_steps, bool
        ):
            raise TypeError("scaling_steps must be an integer.")
        if self.scaling_steps < 1:
            raise ValueError("scaling_steps must be >= 1.")
        if (
            not math.isfinite(float(self.tail_decay_tolerance))
            or self.tail_decay_tolerance <= 0
        ):
            raise ValueError("tail_decay_tolerance must be positive and finite.")


@dataclass(frozen=True)
class TaylorConfig:
    """
    Configuration for a fixed scaling-and-Taylor exponential action.

    The action evaluates ``exp(coefficient * H) @ vector`` using ``degree``
    Taylor terms at each of ``scaling_steps`` fixed steps. Both values are
    static Python integers so the same compiled schedule is reused.

    :ivar degree: Number of Taylor terms evaluated at each scaling step.
    :ivar scaling_steps: Number of scaling-and-squaring steps.
    """

    degree: int
    scaling_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.degree, int) or isinstance(self.degree, bool):
            raise TypeError("degree must be an integer.")
        if not isinstance(self.scaling_steps, int) or isinstance(
            self.scaling_steps, bool
        ):
            raise TypeError("scaling_steps must be an integer.")
        if self.degree < 0:
            raise ValueError("degree must be >= 0.")
        if self.scaling_steps < 1:
            raise ValueError("scaling_steps must be >= 1.")


class LanczosRecurrence(NamedTuple):
    """
    Coefficients and diagnostics returned by one Lanczos projection.

    ``diagonal[j]`` is ``<q_j|H|q_j>`` and ``off_diagonal[j]`` couples
    ``q_j`` to ``q_{j+1}`` in the small projected tridiagonal matrix.
    ``active[j]`` is true when that basis position represents a real Krylov
    vector; positions after an early breakdown are zero padding and must not
    be interpreted as additional eigenvalues.

    :ivar diagonal: Diagonal recurrence coefficients with shape ``(max_dim,)``.
    :ivar off_diagonal: Off-diagonal coefficients with shape ``(max_dim - 1,)``.
    :ivar vector_norm: Norm of the unnormalized seed vector.
    :ivar active: Boolean mask for valid recurrence entries with shape ``(max_dim,)``.
    :ivar residual_norm: Final residual norm before padding.
    """

    diagonal: Tensor
    off_diagonal: Tensor
    vector_norm: Tensor
    active: Tensor
    residual_norm: Tensor


class KrylovProjection(NamedTuple):
    """
    Result of projecting a Hermitian operator onto a seed-generated Krylov space.

    ``basis[:, j]`` is the normalized basis vector ``q_j`` and
    ``recurrence`` describes the seed-generated Krylov recurrence. The basis
    always has the configured fixed width; columns after breakdown are zero
    padding. The projected operator is always the fixed-width tridiagonal
    recurrence with inactive slots padded outside its spectral interval.
    Gradients of perturbations that open a subspace after breakdown are not
    supported; such perturbations require a fresh projection with a resolved
    nonzero residual.

    :ivar basis: Krylov basis with shape ``(dimension, max_dim)``; columns after breakdown are zero padded.
    :ivar recurrence: Recurrence coefficients and a mask identifying valid basis columns.
    """

    basis: Tensor
    recurrence: LanczosRecurrence


class ChebyshevMomentResult(NamedTuple):
    """
    Raw Chebyshev moments together with the metadata needed to interpret them.

    For a probe ``r``, ``moments[n]`` represents
    ``<r|T_n(H_tilde)|r>`` (or one such row per probe), where ``H_tilde`` is
    determined by ``bounds``. A rank-one result is a single bilinear sequence;
    trace estimators require the rank-two, one-row-per-probe form. Reusing this
    record avoids repeating Hamiltonian matrix-vector products for later
    queries.

    :ivar moments: Raw moments with shape ``(order,)`` or ``(num_probes, order)``.
    :ivar bounds: Ascending spectral bounds with shape ``(2,)``.
    :ivar dimension: Hilbert-space dimension stored as a scalar integer tensor.
    """

    moments: Tensor
    bounds: Tensor
    dimension: Tensor


class LanczosMeasure(NamedTuple):
    """
    Spectral quadrature data retained for each stochastic probe.

    ``sum_j weights[r, j] f(nodes[r, j])`` approximates
    ``<probe_r|f(H)|probe_r>``. Each weight already includes
    ``||probe_r||^2``. ``active[r, j]`` says whether that node belongs to the
    actual Krylov projection; false entries are fixed-shape padding and
    contribute nothing.

    :ivar nodes: Ritz nodes with shape ``(num_probes, max_dim)``.
    :ivar weights: Quadrature weights with shape ``(num_probes, max_dim)``.
    :ivar active: Boolean mask identifying valid Ritz nodes.
    :ivar dimension: Hilbert-space dimension stored as a scalar integer tensor.
    """

    nodes: Tensor
    weights: Tensor
    active: Tensor
    dimension: Tensor


def _operator_shape(operator: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(operator, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except (TypeError, ValueError):
        return None


def _prepare_operator(
    operator: Any,
    *,
    vector: Optional[Tensor] = None,
    probes: Optional[Tensor] = None,
    dimension: Optional[int] = None,
) -> Tuple[Any, int]:
    """
    Resolve a dimension and adapt an operator without materializing it.
    """
    op_shape = _operator_shape(operator)
    if op_shape is not None:
        if len(op_shape) != 2 or op_shape[0] != op_shape[1]:
            raise ValueError("operator must have a statically known square shape.")
        if dimension is not None and int(dimension) != op_shape[0]:
            raise ValueError("dimension disagrees with operator.shape.")
        dimension = op_shape[0]
    if vector is not None:
        if _rank(vector) != 1:
            raise ValueError("vector must be one-dimensional.")
        vector_dim = backend.shape_tuple(vector)[0]
        if dimension is not None and dimension != vector_dim:
            raise ValueError("dimension disagrees with vector length.")
        dimension = vector_dim
    if probes is not None:
        if _rank(probes) != 2:
            raise ValueError("probes must have shape (num_probes, dimension).")
        probe_dim = backend.shape_tuple(probes)[1]
        if dimension is not None and dimension != probe_dim:
            raise ValueError("dimension disagrees with probe width.")
        dimension = probe_dim
    if dimension is None:
        raise ValueError(
            "dimension is required for a bare callable without a state or probes."
        )
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    adapted = aslinearoperator(operator, shape=(dimension, dimension), dtype=dtypestr)
    return adapted, dimension


def _resolved_breakdown_tol(config: KrylovConfig) -> float:
    if config.breakdown_tol is not None:
        return float(config.breakdown_tol)
    return 1.0e-6 if dtypestr == "complex64" else 1.0e-12


def _safe_norm(vector: Tensor) -> Tensor:
    squared = backend.real(backend.sum(backend.conj(vector) * vector))
    positive = squared > 0
    return backend.where(
        positive, backend.sqrt(backend.where(positive, squared, 1.0)), 0.0
    )


def _check_nonzero(vector: Tensor) -> None:
    value = _concrete_float(backend.norm(vector))
    if value is not None and value == 0.0:
        raise ValueError("Lanczos and matrix-function actions reject a zero vector.")


def _one_hot(index: Tensor, size: int, dtype: Optional[str] = None) -> Tensor:
    if dtype is None:
        dtype = dtypestr
    result = backend.onehot(index, size)
    return backend.cast(result, dtype)


def _lanczos_core(
    operator: Any, vector: Tensor, config: KrylovConfig, dimension: int
) -> KrylovProjection:
    vector = backend.cast(vector, dtypestr)
    _check_nonzero(vector)
    vector_norm = _safe_norm(vector)
    safe_norm = backend.where(vector_norm > 0, vector_norm, 1.0)
    q = vector / backend.cast(safe_norm, dtypestr)
    q_previous = backend.zeros([dimension], dtype=dtypestr)
    basis = backend.zeros([dimension, config.max_dim], dtype=dtypestr)
    diagonal = backend.zeros([config.max_dim], dtype=dtypestr)
    off_diagonal = backend.zeros([max(config.max_dim - 1, 0)], dtype=dtypestr)
    active = backend.zeros([config.max_dim], dtype="bool")
    alive = backend.convert_to_tensor(True, dtype="bool")
    previous_beta = backend.convert_to_tensor(0.0, dtype=rdtypestr)
    residual_norm = backend.convert_to_tensor(0.0, dtype=rdtypestr)
    tolerance = backend.convert_to_tensor(
        _resolved_breakdown_tol(config), dtype=rdtypestr
    )

    def lanczos_step(carry: Tuple[Tensor, ...], index: Tensor) -> Tuple[Tensor, ...]:
        (
            current,
            previous,
            beta_previous,
            current_basis,
            current_diagonal,
            current_off_diagonal,
            current_active,
            current_alive,
            current_residual,
        ) = carry

        def active_step() -> Tuple[Tensor, ...]:
            residual = operator @ current
            residual = residual - backend.cast(beta_previous, dtypestr) * previous
            alpha = backend.real(backend.sum(backend.conj(current) * residual))
            residual = residual - backend.cast(alpha, dtypestr) * current
            if config.reorthogonalization == "full":
                overlaps = backend.tensordot(
                    backend.conj(backend.transpose(current_basis)), residual, axes=1
                )
                residual = residual - backend.matvec(current_basis, overlaps)
            beta = _safe_norm(residual)
            scale = backend.where(backend.abs(alpha) > 1.0, backend.abs(alpha), 1.0)
            next_alive = (beta > tolerance * scale) & (index < config.max_dim - 1)
            safe_beta = backend.where(next_alive, beta, 1.0)
            next_vector = residual / backend.cast(safe_beta, dtypestr)
            next_vector = next_vector * backend.cast(next_alive, dtypestr)

            current_slot = _one_hot(index, config.max_dim)
            new_basis = current_basis + backend.reshape(
                current, [dimension, 1]
            ) * backend.reshape(current_slot, [1, config.max_dim])
            new_diagonal = current_diagonal + current_slot * backend.cast(
                alpha, dtypestr
            )
            if config.max_dim > 1:
                off_index = backend.where(
                    index < config.max_dim - 1,
                    index,
                    backend.convert_to_tensor(config.max_dim - 2, dtype=idtypestr),
                )
                off_slot = _one_hot(off_index, config.max_dim - 1)
                off_slot = off_slot * backend.cast(
                    (index < config.max_dim - 1) & next_alive, dtypestr
                )
                new_off_diagonal = current_off_diagonal + off_slot * backend.cast(
                    beta, dtypestr
                )
            else:
                new_off_diagonal = current_off_diagonal
            new_active = backend.where(current_slot != 0, True, current_active)
            return (
                next_vector,
                current,
                beta,
                new_basis,
                new_diagonal,
                new_off_diagonal,
                new_active,
                next_alive,
                beta,
            )

        def inactive_step() -> Tuple[Tensor, ...]:
            return (
                current,
                previous,
                backend.convert_to_tensor(0.0, dtype=rdtypestr),
                current_basis,
                current_diagonal,
                current_off_diagonal,
                current_active,
                current_alive,
                current_residual,
            )

        return backend.cond(current_alive, active_step, inactive_step)  # type: ignore[no-any-return]

    final = backend.scan(
        lanczos_step,
        backend.arange(config.max_dim),
        (
            q,
            q_previous,
            previous_beta,
            basis,
            diagonal,
            off_diagonal,
            active,
            alive,
            residual_norm,
        ),
    )
    recurrence = LanczosRecurrence(
        diagonal=final[4],
        off_diagonal=final[5],
        vector_norm=vector_norm,
        active=final[6],
        residual_norm=final[8],
    )
    return KrylovProjection(
        basis=final[3],
        recurrence=recurrence,
    )


def lanczos_project(
    operator: Any,
    vector: Tensor,
    config: KrylovConfig,
    *,
    dimension: Optional[int] = None,
) -> KrylovProjection:
    """
    Build a fixed-shape Hermitian Lanczos projection for a seed vector.

    The operator must be square and Hermitian. The recurrence uses backend
    ``scan`` and is JIT compatible; ``config`` is static and the seed remains
    differentiable.

    :param operator: Dense or sparse square operator, linear operator, or matrix-vector-product callable.
    :type operator: Any
    :param vector: One-dimensional seed vector with shape ``(dimension,)``.
    :type vector: Tensor
    :param config: Static Krylov dimension and recurrence policy.
    :type config: KrylovConfig
    :param dimension: Optional dimension required for a bare callable when it cannot be inferred from ``vector``.
    :type dimension: Optional[int]
    :return: A :class:`KrylovProjection` containing the fixed-shape ``basis``
        and ``recurrence``. ``recurrence.active[j]`` is true for a generated
        Krylov vector and false for fixed-width padding after an early
        breakdown. The function evaluator uses only the generated tridiagonal
        recurrence; gradients of perturbations that open a subspace exactly at
        breakdown are not supported.
    :rtype: KrylovProjection
    """
    if not isinstance(config, KrylovConfig):
        raise TypeError("config must be a KrylovConfig.")
    vector = _as_tensor(vector)
    adapted, dimension = _prepare_operator(operator, vector=vector, dimension=dimension)
    return _lanczos_core(adapted, vector, config, dimension)


def _masked_tridiagonal(
    projection: KrylovProjection,
) -> Tuple[Tensor, Tensor]:
    """
    Build the projected matrix while separating fixed-width padding.
    """
    active = backend.cast(projection.recurrence.active, "bool")
    diagonal = backend.cast(projection.recurrence.diagonal, dtypestr)
    projected = backend.diagflat(diagonal)
    if backend.shape_tuple(projection.recurrence.off_diagonal)[0] > 0:
        upper = backend.diagflat(
            backend.cast(projection.recurrence.off_diagonal, dtypestr), k=1
        )
        projected = projected + upper + backend.adjoint(upper)
    bound = backend.max(backend.sum(backend.abs(projected), axis=1)) + 1.0
    sentinel = backend.cast(bound + 1.0, dtypestr)
    inactive_offsets = backend.cast(
        backend.arange(backend.shape_tuple(diagonal)[0]), dtypestr
    )
    padded_diagonal = backend.where(active, diagonal, sentinel + inactive_offsets)
    active_matrix = backend.cast(active[:, None] & active[None, :], dtypestr)
    matrix = projected * active_matrix
    matrix = matrix - backend.diagflat(backend.where(active, diagonal, 0.0))
    matrix = matrix + backend.diagflat(padded_diagonal)
    return matrix, backend.real(sentinel)


def _projection_spectral_data(
    projection: KrylovProjection,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    matrix, sentinel = _masked_tridiagonal(projection)
    nodes, eigenvectors = backend.eigh(matrix)
    nodes = backend.real(nodes)
    active = nodes < sentinel
    first_abs = backend.abs(eigenvectors[0, :])
    norm_abs = backend.abs(projection.recurrence.vector_norm)
    weights = (first_abs * first_abs) * (norm_abs * norm_abs)
    weights = backend.where(active, weights, backend.zeros_like(weights))
    return nodes, eigenvectors, weights, active


def _safe_ritz_nodes(nodes: Tensor, active: Tensor) -> Tensor:
    node_count = backend.shape_tuple(nodes)[-1]
    first_active = backend.argmax(backend.cast(active, rdtypestr), axis=-1)
    selector = _one_hot(first_active, node_count, backend.dtype(nodes))
    anchor = backend.sum(nodes * selector, axis=-1, keepdims=_rank(active) > 1)
    return backend.where(active, nodes, anchor)


def _mask_active(values: Tensor, active: Tensor) -> Tensor:
    if _rank(values) == _rank(active) + 1:
        active = backend.reshape(active, [1] + list(backend.shape_tuple(active)))
    return backend.where(active, values, backend.zeros_like(values))


def _projection_function_values(
    projection: KrylovProjection, function: Callable[[Tensor], Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    nodes, eigenvectors, weights, active = _projection_spectral_data(projection)
    safe_nodes = _safe_ritz_nodes(nodes, active)
    values = _as_tensor(function(safe_nodes))
    if (
        _rank(values) not in (1, 2)
        or backend.shape_tuple(values)[-1] != backend.shape_tuple(nodes)[0]
    ):
        raise ValueError(
            "function must return shape (max_dim,) or (num_queries, max_dim)."
        )
    return nodes, eigenvectors, weights, active, _mask_active(values, active)


def lanczos_apply(
    projection: KrylovProjection,
    function: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    Apply a scalar function to a prepared Lanczos projection.

    This is the matrix-free operation ``f(H) @ vector`` after the Lanczos
    basis has already been constructed. It diagonalizes only the small
    projected matrix, evaluates ``function`` on its Ritz eigenvalues, and
    transforms the result back to the original vector space. It is useful when
    the same projection is queried with several functions or times.

    The callback receives safe Ritz nodes with shape ``(max_dim,)`` and may
    return ``(max_dim,)`` or ``(num_queries, max_dim)``. The output has shape
    ``(dimension,)`` or ``(num_queries, dimension)``; inactive padded nodes are
    masked before accumulation.

    :param projection: Seed-dependent Lanczos projection.
    :type projection: KrylovProjection
    :param function: Backend-compatible scalar-function callback on Ritz nodes.
    :type function: Callable[[Tensor], Tensor]
    :return: The unnormalized action ``f(H) @ vector``.
    :rtype: Tensor
    """
    if not isinstance(projection, KrylovProjection):
        raise TypeError("projection must be a KrylovProjection.")
    _, eigenvectors, _, _, values = _projection_function_values(projection, function)
    values = backend.cast(values, dtypestr)
    first_component = backend.conj(eigenvectors[0, :])
    coefficients = values * first_component
    transformed = backend.tensordot(
        coefficients, backend.transpose(eigenvectors), axes=1
    )
    if _rank(values) == 1:
        result = backend.matvec(projection.basis, transformed)
    else:
        result = backend.tensordot(
            transformed, backend.transpose(projection.basis), axes=1
        )
    result = result * backend.cast(projection.recurrence.vector_norm, dtypestr)
    return backend.cast(result, dtypestr)


def lanczos_lowest_eigenpair(
    operator: Any,
    vector: Tensor,
    config: KrylovConfig,
    *,
    dimension: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Approximate the lowest Ritz eigenpair of a Hermitian operator by Lanczos.

    The seed must overlap the desired eigenspace. Increase ``config.max_dim``
    and check ``||H @ state - energy * state||`` to assess convergence.

    :param operator: Hermitian dense or sparse operator, linear operator, or matrix-vector-product callable.
    :type operator: Any
    :param vector: Nonzero one-dimensional seed vector.
    :type vector: Tensor
    :param config: Static Krylov dimension and recurrence policy.
    :type config: KrylovConfig
    :param dimension: Optional dimension checked against the seed and operator.
    :type dimension: Optional[int]
    :return: Lowest Ritz energy and normalized Ritz vector.
    :rtype: Tuple[Tensor, Tensor]
    """
    projection = lanczos_project(operator, vector, config, dimension=dimension)
    nodes, eigenvectors, _, _ = _projection_spectral_data(projection)
    state = backend.matvec(projection.basis, eigenvectors[:, 0])
    return nodes[0], state / backend.cast(_safe_norm(state), dtypestr)


def lanczos_quadrature(
    projection: KrylovProjection,
    function: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    Evaluate ``<v|f(H)|v>`` from a prepared Lanczos projection.

    The seed vector need not be normalized. The function is evaluated only on
    the valid Ritz eigenvalues and the returned scalar includes the squared
    norm of the original seed.

    The operator is assumed Hermitian. The seed is not normalized, so the
    result includes its squared norm; query-valued callbacks are supported.

    :param projection: Seed-dependent Lanczos projection.
    :type projection: KrylovProjection
    :param function: Backend-compatible scalar-function callback on Ritz nodes.
    :type function: Callable[[Tensor], Tensor]
    :return: Scalar or rank-one query values of the projected quadratic form.
    :rtype: Tensor
    """
    if not isinstance(projection, KrylovProjection):
        raise TypeError("projection must be a KrylovProjection.")
    _, _, weights, _, values = _projection_function_values(projection, function)
    return backend.sum(values * backend.cast(weights, backend.dtype(values)), axis=-1)


def lanczos_resolvent(
    projection_or_measure: Union[KrylovProjection, LanczosMeasure],
    z: Tensor,
) -> Tensor:
    """
    Evaluate the projected resolvent ``(z - H)^-1`` spectrally.

    The result is the quadratic form represented by the prepared Lanczos
    measure, not a dense inverse of the original operator. A rank-one query
    returns one value per query point. The query must avoid the actual Ritz
    poles; fixed-width padding is ignored.

    Active Ritz poles remain mathematical singularities. Inactive padded slots
    are replaced by safe denominators and contribute zero.

    :param projection_or_measure: A prepared projection or per-probe SLQ measure.
    :type projection_or_measure: Union[KrylovProjection, LanczosMeasure]
    :param z: Scalar or rank-one complex resolvent query.
    :type z: Tensor
    :return: Projected resolvent values, with one query axis when ``z`` is rank one.
    :rtype: Tensor
    """
    z = backend.cast(_as_tensor(z), dtypestr)
    z, scalar = _check_query(z, "z")
    if isinstance(projection_or_measure, KrylovProjection):
        nodes, _, weights, active = _projection_spectral_data(projection_or_measure)
    elif isinstance(projection_or_measure, LanczosMeasure):
        nodes = projection_or_measure.nodes
        weights = projection_or_measure.weights
        active = projection_or_measure.active
    else:
        raise TypeError("expected a KrylovProjection or LanczosMeasure.")
    nodes = backend.cast(nodes, dtypestr)
    weights = backend.cast(weights, dtypestr)
    if scalar:
        denominator = backend.where(active, z - nodes, 1.0)
        return backend.sum(
            backend.where(active, weights / denominator, backend.zeros_like(weights)),
            axis=-1,
        )
    active = backend.reshape(active, [1] + list(backend.shape_tuple(active)))
    denominator = backend.where(
        active,
        backend.reshape(z, [-1] + [1] * _rank(nodes)) - nodes,
        1.0,
    )
    return backend.sum(
        backend.where(active, weights / denominator, backend.zeros_like(denominator)),
        axis=-1,
    )


def _measure_one(
    operator: Any, probe: Tensor, config: KrylovConfig, dimension: int
) -> Tensor:
    projection = _lanczos_core(operator, probe, config, dimension)
    nodes, _, weights, active = _projection_spectral_data(projection)
    active = backend.cast(active, rdtypestr)
    return backend.concat(
        [
            backend.cast(backend.real(nodes), rdtypestr),
            backend.cast(backend.real(weights), rdtypestr),
            active,
        ],
        axis=0,
    )


def slq_measure(
    operator: Any,
    probes: Tensor,
    config: KrylovConfig,
    *,
    dimension: Optional[int] = None,
    probe_batch_size: Optional[int] = None,
) -> LanczosMeasure:
    """
    Construct reusable stochastic Lanczos quadrature measures.

    Each probe is projected independently and reduced to a short weighted
    discrete spectrum. The returned measure can be reused for DOS, partition
    function, thermal, and resolvent calculations without repeating
    Hamiltonian matrix-vector products.

    The operator must be Hermitian and ``probes`` has shape
    ``(num_probes, dimension)``. A batch size uses a padded fixed-block
    scan to bound compiled peak memory; padded probe results are discarded.

    :param operator: Dense or sparse square operator, linear operator, or matrix-vector-product callable.
    :type operator: Any
    :param probes: Explicit probe matrix with shape ``(num_probes, dimension)``.
    :type probes: Tensor
    :param config: Static Krylov dimension and recurrence policy.
    :type config: KrylovConfig
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param probe_batch_size: Optional positive static block size for probe batching.
    :type probe_batch_size: Optional[int]
    :return: A :class:`LanczosMeasure` with ``nodes`` and ``weights`` of shape
        ``(num_probes, max_dim)``. ``active`` marks which entries are genuine
        Ritz nodes; padded entries are ignored by all evaluators.
    :rtype: LanczosMeasure
    """
    if not isinstance(config, KrylovConfig):
        raise TypeError("config must be a KrylovConfig.")
    probes = _as_tensor(probes)
    if _rank(probes) != 2:
        raise ValueError("probes must have shape (num_probes, dimension).")
    num_probes = backend.shape_tuple(probes)[0]
    if num_probes < 1:
        raise ValueError("probes must contain at least one vector.")
    if probe_batch_size is not None and (
        not isinstance(probe_batch_size, int) or probe_batch_size < 1
    ):
        raise ValueError("probe_batch_size must be a positive integer or None.")
    adapted, dimension = _prepare_operator(operator, probes=probes, dimension=dimension)

    if probe_batch_size is None:
        packed = backend.vmap(
            lambda probe: _measure_one(adapted, probe, config, dimension)
        )(probes)
    else:
        packed = _chunked_vmap(
            lambda probe: _measure_one(adapted, probe, config, dimension),
            probes,
            probe_batch_size,
        )
    max_dim = config.max_dim
    nodes = backend.cast(packed[:, :max_dim], rdtypestr)
    weights = backend.cast(packed[:, max_dim : 2 * max_dim], rdtypestr)
    active = packed[:, 2 * max_dim :] != 0
    return LanczosMeasure(
        nodes=nodes,
        weights=weights,
        active=active,
        dimension=backend.convert_to_tensor(dimension, dtype=idtypestr),
    )


def _sample_statistics(
    samples: Tensor, scale: Tensor = 1.0, with_std: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Aggregate samples whose final axis indexes probes.

    When ``with_std`` is false, only the probe mean is formed and the returned
    error tensor is a placeholder that is ignored by the public caller.
    """
    samples = samples * backend.cast(_as_tensor(scale), backend.dtype(samples))
    count = backend.shape_tuple(samples)[-1]
    value = backend.mean(samples, axis=(-1,))
    if not with_std:
        return value, backend.zeros_like(value, dtype=rdtypestr)
    if count < 2:
        raise ValueError("at least two probes are required for a standard error.")
    centered = samples - value[..., None]
    variance = backend.real(
        backend.mean(backend.abs(centered) * backend.abs(centered), axis=(-1,))
    )
    standard_error = backend.sqrt(variance / count * (count / (count - 1)))
    return value, standard_error


def _evaluate_measure_function(
    measure: LanczosMeasure, function: Callable[[Tensor], Tensor]
) -> Tensor:
    safe_nodes = _safe_ritz_nodes(measure.nodes, measure.active)
    values = _as_tensor(function(safe_nodes))
    values = _mask_active(values, measure.active)
    if _rank(values) == 2:
        if backend.shape_tuple(values) != backend.shape_tuple(measure.nodes):
            raise ValueError("function must return one value per probe and Ritz node.")
        return backend.sum(
            values * backend.cast(measure.weights, backend.dtype(values)), axis=-1
        )
    if _rank(values) == 3:
        if backend.shape_tuple(values)[-2:] != backend.shape_tuple(measure.nodes):
            raise ValueError(
                "query-valued functions must return (num_queries, num_probes, max_dim)."
            )
        return backend.sum(
            values * backend.cast(measure.weights[None, :, :], backend.dtype(values)),
            axis=-1,
        )
    raise ValueError(
        "function must return (num_probes, max_dim) or "
        "(num_queries, num_probes, max_dim)."
    )


def _chunked_vmap(
    function: Callable[[Tensor], Tensor], probes: Tensor, probe_batch_size: int
) -> Tensor:
    """
    Map a probe function over fixed-size padded blocks with one scan.
    """
    num_probes, dimension = backend.shape_tuple(probes)
    probe_batch_size = min(probe_batch_size, num_probes)
    num_blocks = (num_probes + probe_batch_size - 1) // probe_batch_size
    padded_count = num_blocks * probe_batch_size
    padding = padded_count - num_probes
    if padding:
        probes = backend.concat(
            [probes, backend.tile(probes[-1:], [padding, 1])], axis=0
        )
    blocks = backend.reshape(probes, [num_blocks, probe_batch_size, dimension])

    def scan_step(carry: Tensor, block: Tensor) -> Tuple[Tensor, Tensor]:
        del carry
        return backend.convert_to_tensor(0.0, dtype=rdtypestr), backend.vmap(function)(
            block
        )

    _, block_values = backend.jaxy_scan(
        scan_step,
        backend.convert_to_tensor(0.0, dtype=rdtypestr),
        blocks,
    )
    value_shape = backend.shape_tuple(block_values)
    flat_shape = [padded_count] + list(value_shape[2:])
    values = backend.reshape(block_values, flat_shape)
    return values[:num_probes]


def evaluate_slq_trace(
    measure: LanczosMeasure,
    function: Callable[[Tensor], Tensor],
    *,
    probe_scale: Tensor = 1.0,
    with_std: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Evaluate and statistically aggregate a function on an SLQ measure.

    The callback receives one row of Ritz nodes for every probe and must return
    one value per node, or one such array per query. The values are integrated
    against the stored quadrature weights before the probe mean and standard
    error are formed.

    The final sample axis indexes probes. The reported standard error is the
    unbiased sample standard error of the probe estimator.

    :param measure: Prepared per-probe SLQ measure.
    :type measure: LanczosMeasure
    :param function: Backend-compatible scalar-function callback on Ritz nodes.
    :type function: Callable[[Tensor], Tensor]
    :param probe_scale: Multiplicative factor converting the probe mean to the requested trace normalization.
    :type probe_scale: Tensor
    :param with_std: If true, return ``(value, standard_error)``; otherwise
        return only the estimated function trace.
    :type with_std: bool, optional
    :return: Estimated function trace, optionally paired with its probe
        standard error.
    :rtype: Union[Tensor, Tuple[Tensor, Tensor]]
    """
    if not isinstance(measure, LanczosMeasure):
        raise TypeError("measure must be a LanczosMeasure.")
    samples = _evaluate_measure_function(measure, function)
    value, standard_error = _sample_statistics(samples, probe_scale, with_std=with_std)
    if with_std:
        return value, standard_error
    return value


def _scaled_hamiltonian(
    operator: Any, config: ChebyshevConfig
) -> Callable[[Tensor], Tensor]:
    emin, emax = config.bounds
    scale = backend.cast((emax - emin) / 2.0, dtypestr)
    center = backend.cast((emax + emin) / 2.0, dtypestr)

    def apply(vector: Tensor) -> Tensor:
        return (operator @ vector - center * vector) / scale

    return apply


def _check_coefficients(coefficients: Any, order: int) -> Tensor:
    coefficients = _as_tensor(coefficients)
    rank = _rank(coefficients)
    if rank not in (1, 2) or backend.shape_tuple(coefficients)[-1] != order:
        raise ValueError(
            f"coefficients must have shape ({order},) or (num_queries, {order})."
        )
    return coefficients


def chebyshev_action(
    operator: Any,
    vector: Tensor,
    coefficients: Tensor,
    config: ChebyshevConfig,
    *,
    dimension: Optional[int] = None,
) -> Tensor:
    """
    Compute a fixed-order Chebyshev polynomial action.

    The function evaluates ``sum_n coefficients[n] * T_n(H_tilde) @ vector``
    without materializing all intermediate Chebyshev vectors. This is the
    reusable kernel behind real-time evolution and other polynomial matrix
    functions.

    The operator must be Hermitian and its spectrum must be enclosed by
    ``config.bounds``. The recurrence is streamed with backend ``scan``.
    Coefficients have shape ``(order,)`` or ``(num_queries, order)`` and
    outputs have shape ``(dimension,)`` or ``(num_queries, dimension)``.

    :param operator: Dense or sparse square operator, linear operator, or matrix-vector-product callable.
    :type operator: Any
    :param vector: One-dimensional input vector with shape ``(dimension,)``.
    :type vector: Tensor
    :param coefficients: Chebyshev coefficients with one row per query at most.
    :type coefficients: Tensor
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :return: Unnormalized polynomial action.
    :rtype: Tensor
    """
    if not isinstance(config, ChebyshevConfig):
        raise TypeError("config must be a ChebyshevConfig.")
    vector = backend.cast(_as_tensor(vector), dtypestr)
    adapted, dimension = _prepare_operator(operator, vector=vector, dimension=dimension)
    coefficients = backend.cast(
        _check_coefficients(coefficients, config.order), dtypestr
    )
    apply_scaled = _scaled_hamiltonian(adapted, config)
    coefficient_zero = coefficients[..., 0][..., None]
    result = coefficient_zero * vector
    if config.order == 1:
        return backend.cast(result, dtypestr)
    previous = vector
    current = apply_scaled(vector)
    result = result + coefficients[..., 1][..., None] * current

    def recurrence(
        carry: Tuple[Tensor, Tensor, Tensor], index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        previous_vector, current_vector, current_result = carry
        next_vector = (
            backend.cast(2.0, dtypestr) * apply_scaled(current_vector) - previous_vector
        )
        next_result = current_result + coefficients[..., index][..., None] * next_vector
        return current_vector, next_vector, next_result

    if config.order > 2:
        _, _, result = backend.scan(
            recurrence,
            backend.arange(2, config.order),
            (previous, current, result),
        )
    return backend.cast(result, dtypestr)


def chebyshev_moments(
    operator: Any,
    right_vector: Tensor,
    config: ChebyshevConfig,
    *,
    left_vector: Optional[Tensor] = None,
    dimension: Optional[int] = None,
) -> ChebyshevMomentResult:
    """
    Compute ``<left|T_n(H_tilde)|right>`` for all requested orders.

    The operator is rescaled using ``config.bounds`` and the recurrence is
    streamed one order at a time. The returned moments can be contracted with
    many later coefficient vectors without applying the operator again.

    The operator must be Hermitian with spectrum enclosed by ``config.bounds``.
    Both vectors are rank one with length ``dimension``; the returned moments
    have shape ``(order,)``.

    :param operator: Dense or sparse square operator, linear operator, or matrix-vector-product callable.
    :type operator: Any
    :param right_vector: Right vector with shape ``(dimension,)``.
    :type right_vector: Tensor
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :param left_vector: Optional left vector with the same shape as ``right_vector``.
    :type left_vector: Optional[Tensor]
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :return: Raw moments and bounds metadata.
    :rtype: ChebyshevMomentResult
    """
    if not isinstance(config, ChebyshevConfig):
        raise TypeError("config must be a ChebyshevConfig.")
    right_vector = backend.cast(_as_tensor(right_vector), dtypestr)
    if left_vector is None:
        left_vector = right_vector
    else:
        left_vector = backend.cast(_as_tensor(left_vector), dtypestr)
    adapted, dimension = _prepare_operator(
        operator, vector=right_vector, dimension=dimension
    )
    if _rank(left_vector) != 1 or backend.shape_tuple(left_vector)[0] != dimension:
        raise ValueError(
            "left_vector must have the same one-dimensional shape as right_vector."
        )
    apply_scaled = _scaled_hamiltonian(adapted, config)
    moments = backend.zeros([config.order], dtype=dtypestr)
    moment_zero = backend.sum(backend.conj(left_vector) * right_vector)
    moments = moments + _one_hot(0, config.order) * moment_zero
    if config.order > 1:
        previous = right_vector
        current = apply_scaled(previous)
        moment_one = backend.sum(backend.conj(left_vector) * current)
        moments = moments + _one_hot(1, config.order) * moment_one

        def recurrence(
            carry: Tuple[Tensor, Tensor, Tensor], index: Tensor
        ) -> Tuple[Tensor, Tensor, Tensor]:
            previous_vector, current_vector, current_moments = carry
            next_vector = (
                backend.cast(2.0, dtypestr) * apply_scaled(current_vector)
                - previous_vector
            )
            moment = backend.sum(backend.conj(left_vector) * next_vector)
            next_moments = current_moments + _one_hot(index, config.order) * moment
            return current_vector, next_vector, next_moments

        if config.order > 2:
            _, _, moments = backend.scan(
                recurrence,
                backend.arange(2, config.order),
                (previous, current, moments),
            )
    return ChebyshevMomentResult(
        moments=moments,
        bounds=backend.stack(
            [
                backend.convert_to_tensor(config.bounds[0]),
                backend.convert_to_tensor(config.bounds[1]),
            ]
        ),
        dimension=backend.convert_to_tensor(dimension, dtype=idtypestr),
    )


def _trace_chebyshev_one(
    operator: Any, vector: Tensor, config: ChebyshevConfig
) -> Tensor:
    apply_scaled = _scaled_hamiltonian(operator, config)
    moments = backend.zeros([config.order], dtype=dtypestr)
    first = vector
    moment_zero = backend.real(backend.sum(backend.conj(first) * first))
    moments = moments + _one_hot(0, config.order) * backend.cast(moment_zero, dtypestr)
    if config.order == 1:
        return moments
    second = apply_scaled(first)
    moment_one = backend.real(backend.sum(backend.conj(first) * second))
    moments = moments + _one_hot(1, config.order) * backend.cast(moment_one, dtypestr)
    max_pair = (config.order - 1) // 2

    def recurrence(
        carry: Tuple[Tensor, Tensor, Tensor], index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        previous, current, current_moments = carry
        following = backend.cast(2.0, dtypestr) * apply_scaled(current) - previous
        even_value = (
            2.0 * backend.real(backend.sum(backend.conj(current) * current))
            - moment_zero
        )
        odd_value = (
            2.0 * backend.real(backend.sum(backend.conj(current) * following))
            - moment_one
        )
        even_index = 2 * index
        odd_unclamped = even_index + 1
        odd_valid = odd_unclamped < config.order
        odd_index = backend.where(
            odd_valid,
            odd_unclamped,
            backend.convert_to_tensor(config.order - 1, dtype=idtypestr),
        )
        even_slot = _one_hot(even_index, config.order)
        odd_slot = _one_hot(odd_index, config.order) * backend.cast(odd_valid, dtypestr)
        updated = current_moments + even_slot * backend.cast(even_value, dtypestr)
        updated = updated + odd_slot * backend.cast(odd_value, dtypestr)
        return current, following, updated

    if max_pair >= 1:
        _, _, moments = backend.scan(
            recurrence,
            backend.arange(1, max_pair + 1),
            (first, second, moments),
        )
    return moments


def stochastic_chebyshev_moments(
    operator: Any,
    probes: Tensor,
    config: ChebyshevConfig,
    *,
    dimension: Optional[int] = None,
    probe_batch_size: Optional[int] = None,
) -> ChebyshevMomentResult:
    """
    Compute same-vector Chebyshev moments for explicit trace probes.

    For each row ``probe[r]`` this computes the diagonal moments
    ``<probe[r]|T_n(H_tilde)|probe[r]>`` used by stochastic trace estimates.
    ``probe_batch_size`` changes only the execution schedule; it does not
    change the returned probe-by-order array.

    ``probes`` has shape ``(num_probes, dimension)`` and the result stores
    moments with shape ``(num_probes, order)``. Optional batching uses a padded
    fixed-block scan.

    :param operator: Dense or sparse square operator, linear operator, or matrix-vector-product callable.
    :type operator: Any
    :param probes: Explicit probe matrix with shape ``(num_probes, dimension)``.
    :type probes: Tensor
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param probe_batch_size: Optional positive static block size for probe batching.
    :type probe_batch_size: Optional[int]
    :return: Raw moments, bounds, and dimension metadata.
    :rtype: ChebyshevMomentResult
    """
    if not isinstance(config, ChebyshevConfig):
        raise TypeError("config must be a ChebyshevConfig.")
    probes = backend.cast(_as_tensor(probes), dtypestr)
    if _rank(probes) != 2:
        raise ValueError("probes must have shape (num_probes, dimension).")
    if backend.shape_tuple(probes)[0] < 1:
        raise ValueError("probes must contain at least one vector.")
    if probe_batch_size is not None and (
        not isinstance(probe_batch_size, int) or probe_batch_size < 1
    ):
        raise ValueError("probe_batch_size must be a positive integer or None.")
    adapted, dimension = _prepare_operator(operator, probes=probes, dimension=dimension)

    if probe_batch_size is None:
        moments = backend.vmap(
            lambda probe: _trace_chebyshev_one(adapted, probe, config)
        )(probes)
    else:
        moments = _chunked_vmap(
            lambda probe: _trace_chebyshev_one(adapted, probe, config),
            probes,
            probe_batch_size,
        )
    return ChebyshevMomentResult(
        moments=moments,
        bounds=backend.stack(
            [
                backend.convert_to_tensor(config.bounds[0]),
                backend.convert_to_tensor(config.bounds[1]),
            ]
        ),
        dimension=backend.convert_to_tensor(dimension, dtype=idtypestr),
    )


def evaluate_chebyshev_moments(
    moments: ChebyshevMomentResult,
    coefficients: Tensor,
    *,
    kernel_weights: Optional[Tensor] = None,
) -> Tensor:
    """
    Evaluate one or many scalar functions from raw Chebyshev moments.

    This performs only the coefficient--moment contraction. It is therefore
    cheap compared with moment preparation and is intended for repeated DOS,
    thermal, or response queries sharing the same Hamiltonian moments.

    Coefficients have shape ``(order,)`` or ``(num_queries, order)`` and are
    contracted with rank-one or probe-batched moments.

    :param moments: Prepared raw Chebyshev moments.
    :type moments: ChebyshevMomentResult
    :param coefficients: Chebyshev coefficients with one row per query at most.
    :type coefficients: Tensor
    :param kernel_weights: Optional damping weights with shape ``(order,)``.
    :type kernel_weights: Optional[Tensor]
    :return: Scalar, query-vector, or per-probe function values.
    :rtype: Tensor
    """
    if not isinstance(moments, ChebyshevMomentResult):
        raise TypeError("moments must be a ChebyshevMomentResult.")
    raw = _as_tensor(moments.moments)
    coefficients = _check_coefficients(coefficients, backend.shape_tuple(raw)[-1])
    if kernel_weights is not None:
        weights = _as_tensor(kernel_weights)
        if (
            _rank(weights) != 1
            or backend.shape_tuple(weights)[0] != backend.shape_tuple(raw)[-1]
        ):
            raise ValueError("kernel_weights must have one entry per moment.")
        raw = raw * backend.cast(weights, backend.dtype(raw))
    if _rank(raw) == 1:
        return backend.tensordot(coefficients, raw, axes=1)
    if _rank(coefficients) == 1:
        return backend.tensordot(raw, coefficients, axes=([1], [0]))
    return backend.tensordot(coefficients, backend.transpose(raw), axes=([1], [0]))


def evaluate_chebyshev_trace(
    moments: ChebyshevMomentResult,
    coefficients: Tensor,
    *,
    kernel_weights: Optional[Tensor] = None,
    probe_scale: Tensor = 1.0,
    with_std: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Evaluate stochastic Chebyshev moments and report probe uncertainty.

    The function first contracts each probe's moments with the supplied
    coefficient vector, then returns the scaled probe mean and its standard
    error. Query rows are preserved in the leading output dimension.

    ``moments`` must have shape ``(num_probes, order)`` and ``coefficients``
    has shape ``(order,)`` or ``(num_queries, order)``. A rank-one bilinear
    sequence is not a trace estimate; use :func:`evaluate_chebyshev_moments`
    for that case. ``probe_scale`` converts the probe mean into the requested
    trace normalization; the standard error uses paired probe samples.

    :param moments: Prepared raw Chebyshev moments.
    :type moments: ChebyshevMomentResult
    :param coefficients: Chebyshev coefficients with one row per query at most.
    :type coefficients: Tensor
    :param kernel_weights: Optional damping weights with shape ``(order,)``.
    :type kernel_weights: Optional[Tensor]
    :param probe_scale: Multiplicative factor converting the probe mean to the requested trace normalization.
    :type probe_scale: Tensor
    :param with_std: If true, return ``(value, standard_error)``; otherwise
        return only the estimated trace.
    :type with_std: bool, optional
    :return: Estimated trace values, optionally paired with probe standard
        errors.
    :rtype: Union[Tensor, Tuple[Tensor, Tensor]]
    """
    _validate_trace_moments(moments)
    values = evaluate_chebyshev_moments(
        moments, coefficients, kernel_weights=kernel_weights
    )
    value, standard_error = _sample_statistics(values, probe_scale, with_std=with_std)
    if with_std:
        return value, standard_error
    return value


def _validate_trace_moments(moments: ChebyshevMomentResult) -> None:
    if not isinstance(moments, ChebyshevMomentResult):
        raise TypeError("moments must be a ChebyshevMomentResult.")
    if _rank(moments.moments) != 2:
        raise ValueError(
            "trace estimators require moments with shape "
            "(num_probes, order); use evaluate_chebyshev_moments for a "
            "single bilinear moment sequence."
        )


def _chebyshev_quadrature_from_bounds(
    bounds: Tensor, quadrature_order: int
) -> Tuple[Tensor, Tensor]:
    theta = (backend.cast(backend.arange(quadrature_order), rdtypestr) + 0.5) * (
        math.pi / quadrature_order
    )
    x = backend.cos(theta)
    bounds = _as_tensor(bounds)
    emin, emax = bounds[0], bounds[1]
    energies = (emax + emin) / 2.0 + (emax - emin) / 2.0 * x
    return theta, energies


def _chebyshev_coefficients_from_bounds(
    function: Callable[[Tensor], Tensor],
    order: int,
    bounds: Tensor,
    quadrature_order: Optional[int] = None,
) -> Tensor:
    if backend.name not in ("numpy", "jax"):
        raise NotImplementedError(
            "Chebyshev coefficient generation is currently supported only "
            "by the NumPy and JAX backends."
        )
    if quadrature_order is None:
        quadrature_order = max(256, 4 * order)
    if not isinstance(quadrature_order, int) or quadrature_order < order:
        raise ValueError("quadrature_order must be an integer >= order.")
    _, energies = _chebyshev_quadrature_from_bounds(bounds, quadrature_order)
    values = _as_tensor(function(energies))
    if _rank(values) != 1 or backend.shape_tuple(values)[0] != quadrature_order:
        raise ValueError("function must return one value per quadrature node.")
    values = backend.cast(values, dtypestr)
    extended = backend.concat([values, backend.reverse(values)], axis=0)
    transformed = backend.fft(extended)
    orders = backend.cast(backend.arange(order), rdtypestr)
    phase = backend.exp(
        backend.cast(-0.5j * math.pi / quadrature_order, dtypestr)
        * backend.cast(orders, dtypestr)
    )
    raw = phase * transformed[:order] / quadrature_order
    factors = backend.concat(
        [
            backend.convert_to_tensor([0.5], dtype=rdtypestr),
            backend.ones([order - 1], dtype=rdtypestr),
        ],
        axis=0,
    )
    return raw * backend.cast(factors, dtypestr)


def chebyshev_coefficients(
    function: Callable[[Tensor], Tensor],
    config: ChebyshevConfig,
    *,
    quadrature_order: Optional[int] = None,
) -> Tensor:
    """
    Sample a scalar function and compute Chebyshev coefficients with an FFT-DCT.

    This generic helper is intended for user-supplied scalar functions on the
    NumPy and JAX backends. Built-in exponential, resolvent, Fermi--Dirac, and
    delta-function coefficient generators use analytic or backend-specialized
    formulas where available; no TensorFlow, PyTorch, or CuPy DCT fallback is
    provided.

    The callback receives Chebyshev nodes mapped to ``config.bounds`` and
    returns one value per node. This generic helper is differentiable when the
    callback and backend support it; built-in coefficient generators may use
    analytic special-function paths instead.

    :param function: Backend-compatible callback returning one value per node.
    :type function: Callable[[Tensor], Tensor]
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :param quadrature_order: Optional static number of quadrature nodes; it must be at least ``config.order``.
    :type quadrature_order: Optional[int]
    :return: Coefficients with shape ``(order,)``.
    :rtype: Tensor
    """
    if not isinstance(config, ChebyshevConfig):
        raise TypeError("config must be a ChebyshevConfig.")
    return _chebyshev_coefficients_from_bounds(
        function, config.order, _as_tensor(config.bounds), quadrature_order
    )


def _exponential_coefficients_from_bounds(
    coefficient: Tensor,
    order: int,
    bounds: Tensor,
    reference: Tensor = 0.0,
    scaled: bool = False,
) -> Tensor:
    if backend.name not in ("numpy", "jax"):
        raise NotImplementedError(
            "Analytic Chebyshev exponential coefficients are currently "
            "supported only by the NumPy and JAX backends."
        )
    coefficient = backend.cast(_as_tensor(coefficient), dtypestr)
    coefficient, scalar = _check_query(coefficient, "coefficient")
    reference = backend.cast(_as_tensor(reference), dtypestr)
    scale = backend.cast((bounds[1] - bounds[0]) / 2.0, dtypestr)
    mixed = (backend.real(coefficient) != 0.0) & (backend.imag(coefficient) != 0.0)
    invalid = _concrete_float(backend.max(backend.cast(mixed, rdtypestr)))
    if invalid is not None and invalid > 0.0:
        raise ValueError(
            "Chebyshev exponential coefficients require a purely real or "
            "purely imaginary coefficient."
        )

    def one(value: Tensor) -> Tensor:
        center = (bounds[1] + bounds[0]) / 2.0 - reference
        orders = backend.cast(backend.arange(order), rdtypestr)
        recurrence_length = order + 4 * math.ceil(math.sqrt(order)) + 32
        factors = backend.concat(
            [
                backend.convert_to_tensor([1.0], dtype=rdtypestr),
                backend.ones([order - 1], dtype=rdtypestr) * 2.0,
            ],
            axis=0,
        )
        if scaled:
            bessel = backend.special_ive(
                order,
                backend.real(-value * scale),
                recurrence_length,
            )
            phase = backend.cos(math.pi * orders)
            return backend.exp(value * center) * factors * phase * bessel

        def real_coefficients() -> Tensor:
            argument = backend.real(value * scale)
            magnitude = backend.abs(argument)
            bessel = backend.special_ive(order, magnitude, recurrence_length)
            phase = backend.where(
                argument < 0.0,
                backend.cos(math.pi * orders),
                backend.ones([order], dtype=rdtypestr),
            )
            prefactor = backend.exp(value * center + backend.cast(magnitude, dtypestr))
            return backend.cast(prefactor * factors * phase * bessel, dtypestr)

        def complex_coefficients() -> Tensor:
            bessel = backend.special_jv(
                order,
                1.0j * value * scale,
                recurrence_length,
            )
            phase = backend.exp(-0.5j * math.pi * orders)
            return backend.cast(
                backend.exp(value * center) * factors * phase * bessel, dtypestr
            )

        result = backend.cond(
            backend.imag(value) == 0.0,
            real_coefficients,
            complex_coefficients,
        )
        invalid_coefficients = backend.ones([order], dtype=dtypestr) * backend.cast(
            float("nan"), dtypestr
        )
        is_mixed = (backend.real(value) != 0.0) & (backend.imag(value) != 0.0)
        return backend.where(is_mixed, invalid_coefficients, result)

    if scalar:
        return one(coefficient)
    return backend.vmap(one)(coefficient)


def exponential_coefficients(coefficient: Tensor, config: ChebyshevConfig) -> Tensor:
    """
    Return analytic Chebyshev coefficients for ``exp(coefficient * H)``.

    For a purely imaginary coefficient this is the real-time propagator
    expansion; for a purely real coefficient it is the imaginary-time
    expansion. Mixed real-imaginary coefficients are unsupported. The returned
    coefficients are ready for :func:`chebyshev_action` and have one row per
    query when ``coefficient`` is rank one.

    The coefficient query is scalar or rank one, yielding shape ``(order,)``
    or ``(num_queries, order)``. NumPy and JAX use their backend Bessel paths.
    Other backends raise ``NotImplementedError`` for this Bessel-dependent
    coefficient generator; no silent DCT fallback is used.

    :param coefficient: Scalar or rank-one exponential coefficient query.
    :type coefficient: Tensor
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :return: Chebyshev coefficients for the exponential matrix function.
    :rtype: Tensor
    """
    return _exponential_coefficients_from_bounds(
        coefficient, config.order, _as_tensor(config.bounds)
    )


def _multiply_chebyshev_affine(
    coefficients: Tensor, output_order: int, scale: Tensor, center: Tensor
) -> Tensor:
    first = coefficients[..., 1:2] / 2.0
    if output_order == 1:
        x_coefficients = first
    else:
        second = coefficients[..., 0:1] + coefficients[..., 2:3] / 2.0
        middle = (
            coefficients[..., 1 : output_order - 1]
            + coefficients[..., 3 : output_order + 1]
        ) / 2.0
        x_coefficients = backend.concat([first, second, middle], axis=-1)
    return center * coefficients[..., :output_order] + scale * x_coefficients


def _energy_polynomial_coefficients(
    coefficient: Tensor,
    order: int,
    bounds: Tensor,
    reference: Tensor,
    power: int,
    centered: bool,
) -> Tensor:
    scale = (bounds[1] - bounds[0]) / 2.0
    center = (bounds[1] + bounds[0]) / 2.0
    if centered:
        center = center - reference
    coefficients = _exponential_coefficients_from_bounds(
        coefficient, order + power, bounds, reference, scaled=True
    )
    for output_order in range(order + power - 1, order - 1, -1):
        coefficients = _multiply_chebyshev_affine(
            coefficients, output_order, scale, center
        )
    return coefficients


def resolvent_coefficients(z: Tensor, config: ChebyshevConfig) -> Tensor:
    """
    Return analytic Chebyshev coefficients for ``1 / (z - H)``.

    The coefficients describe the resolvent of the operator whose spectrum is
    enclosed by ``config.bounds``. They can be contracted with moments or used
    in a Chebyshev action; ``z`` must not lie on the enclosed real interval.

    The formula uses the branch with a decaying Chebyshev ratio. Concrete
    queries are rejected when the final-to-leading coefficient ratio reaches
    ``config.tail_decay_tolerance``. This is a decay diagnostic rather than a
    relative-error tolerance. Under JIT the corresponding coefficient rows are
    NaN so invalid truncations cannot silently propagate.

    :param z: Scalar or rank-one complex resolvent query.
    :type z: Tensor
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :return: Coefficients with shape ``(order,)`` or ``(num_queries, order)``.
    :rtype: Tensor
    """
    z = backend.cast(_as_tensor(z), dtypestr)
    z, scalar = _check_query(z, "z")
    real = backend.real(z)
    on_enclosed_real_interval = (
        (backend.imag(z) == 0.0)
        & (real >= config.bounds[0])
        & (real <= config.bounds[1])
    )
    invalid = _concrete_float(
        backend.max(backend.cast(on_enclosed_real_interval, rdtypestr))
    )
    if invalid is not None and invalid > 0.0:
        details = _concrete_query_failure_details(on_enclosed_real_interval, z)
        raise ValueError(
            "resolvent query lies on the real interval enclosed by the spectral "
            f"bounds{details}."
        )

    scale = backend.cast((config.bounds[1] - config.bounds[0]) / 2.0, dtypestr)
    center = backend.cast((config.bounds[1] + config.bounds[0]) / 2.0, dtypestr)
    normalized = (z - center) / scale
    root = backend.sqrt(normalized * normalized - 1.0)
    q = normalized - root
    flip = backend.abs(q) > 1.0
    root = backend.where(flip, -root, root)
    q = normalized - root
    tail_ratio = backend.cast(2.0, rdtypestr) * backend.power(
        backend.abs(q), config.order - 1
    )
    nondecaying = tail_ratio >= backend.cast(config.tail_decay_tolerance, rdtypestr)
    invalid = _concrete_float(backend.max(backend.cast(nondecaying, rdtypestr)))
    if invalid is not None and invalid > 0.0:
        details = _concrete_query_failure_details(nondecaying, z)
        raise ValueError(
            "resolvent Chebyshev coefficient tail exceeds "
            f"tail_decay_tolerance={config.tail_decay_tolerance:g}{details}; "
            "increase order or broadening."
        )

    def one(value_root: Tensor, value_q: Tensor, invalid_tail: Tensor) -> Tensor:
        orders = backend.cast(backend.arange(config.order), dtypestr)
        powers = backend.power(value_q, orders)
        factors = backend.concat(
            [
                backend.convert_to_tensor([1.0], dtype=rdtypestr),
                backend.ones([config.order - 1], dtype=rdtypestr) * 2.0,
            ],
            axis=0,
        )
        coefficients = backend.cast(factors, dtypestr) * powers / (scale * value_root)
        nan_coefficients = backend.ones([config.order], dtype=dtypestr) * backend.cast(
            float("nan"), dtypestr
        )
        return backend.where(invalid_tail, nan_coefficients, coefficients)

    if scalar:
        coefficients = one(root, q, nondecaying)
    else:
        coefficients = backend.vmap(one, vectorized_argnums=(0, 1, 2))(
            root, q, nondecaying
        )
    return coefficients


def fermi_dirac_coefficients(
    beta: Tensor, chemical_potential: Tensor, config: ChebyshevConfig
) -> Tensor:
    """
    Return Chebyshev coefficients for the Fermi--Dirac function.

    The query parameters are the inverse temperature and chemical potential;
    scalar and rank-one inputs are broadcast into coefficient rows for later
    contraction with prepared Chebyshev moments.

    Scalar queries produce one coefficient vector. Rank-one queries are paired
    elementwise, except that a scalar query broadcasts over the other vector.

    :param beta: Inverse-temperature scalar or rank-one query.
    :type beta: Tensor
    :param chemical_potential: Chemical-potential scalar or rank-one query.
    :type chemical_potential: Tensor
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :return: Coefficients with shape ``(order,)`` or ``(num_queries, order)``.
    :rtype: Tensor
    """
    beta, beta_scalar = _check_query(beta, "beta")
    chemical_potential, mu_scalar = _check_query(
        chemical_potential, "chemical_potential"
    )
    if not beta_scalar and not mu_scalar:
        if backend.shape_tuple(beta)[0] != backend.shape_tuple(chemical_potential)[0]:
            raise ValueError(
                "beta and chemical_potential must have matching query lengths."
            )

    def one(b: Tensor, mu: Tensor) -> Tensor:
        return chebyshev_coefficients(
            lambda energy: backend.sigmoid(-b * (energy - mu)),
            config,
        )

    if beta_scalar and mu_scalar:
        return one(beta, chemical_potential)
    if beta_scalar:
        return backend.vmap(lambda mu: one(beta, mu))(chemical_potential)
    if mu_scalar:
        return backend.vmap(lambda b: one(b, chemical_potential))(beta)
    return backend.vmap(one, vectorized_argnums=(0, 1))(beta, chemical_potential)


def _validate_inside(energies: Tensor, bounds: Tuple[float, float], name: str) -> None:
    values = _concrete_float(backend.min(energies))
    maximum = _concrete_float(backend.max(energies))
    if values is not None and maximum is not None:
        if values <= bounds[0] or maximum >= bounds[1]:
            raise ValueError(f"{name} must lie strictly inside the spectral bounds.")


def delta_coefficients(energies: Tensor, config: ChebyshevConfig) -> Tensor:
    """
    Return Chebyshev coefficients for the density kernel ``delta(E - H)``.

    The finite-order series is the kernel used by KPM DOS reconstruction. Each
    requested energy produces one coefficient row, and the energy must lie
    strictly inside the configured spectral interval.

    Query energies must lie strictly inside the configured bounds because the
    Chebyshev density kernel has endpoint singularities.

    :param energies: Scalar or rank-one energy query.
    :type energies: Tensor
    :param config: Static polynomial order and ascending spectral bounds.
    :type config: ChebyshevConfig
    :return: Coefficients with shape ``(order,)`` or ``(num_queries, order)``.
    :rtype: Tensor
    """
    energies, scalar = _check_query(energies, "energies")
    _validate_inside(energies, config.bounds, "energies")
    emin, emax = config.bounds
    scale = (emax - emin) / 2.0
    center = (emax + emin) / 2.0

    def one(energy: Tensor) -> Tensor:
        x = (energy - center) / scale
        denominator = math.pi * scale * backend.sqrt(1.0 - x * x)
        orders = backend.cast(backend.arange(config.order), rdtypestr)
        theta = backend.acos(x)
        factors = backend.concat(
            [
                backend.ones([1], dtype=rdtypestr),
                backend.ones([config.order - 1], dtype=rdtypestr) * 2.0,
            ],
            axis=0,
        )
        return factors * backend.cos(orders * theta) / denominator

    if scalar:
        return one(energies)
    return backend.vmap(one)(energies)


def kernel_weights(config: ChebyshevConfig) -> Tensor:
    """
    Return KPM damping weights.

    The weights multiply raw Chebyshev moments before DOS reconstruction and
    are selected by ``config.kernel``. They do not alter the Hamiltonian
    recurrence or the stored moment metadata.

    The returned rank-one tensor has length ``config.order`` and is applied
    elementwise to raw moments during reconstruction.

    :param config: Chebyshev order, bounds, and damping-kernel choice.
    :type config: ChebyshevConfig
    :return: Damping weights with shape ``(order,)``.
    :rtype: Tensor
    """
    if not isinstance(config, ChebyshevConfig):
        raise TypeError("config must be a ChebyshevConfig.")
    if config.order < 2:
        raise ValueError("DOS reconstruction requires at least two moments.")
    n = backend.cast(backend.arange(config.order), rdtypestr)
    if config.kernel == "dirichlet":
        return backend.ones([config.order], dtype=rdtypestr)
    if config.kernel == "jackson":
        m = float(config.order)
        angle = math.pi / m
        return (
            (m - n) * backend.cos(angle * n) + backend.sin(angle * n) / math.tan(angle)
        ) / m
    return backend.sinh(config.lorentz_lambda * (1.0 - n / config.order)) / math.sinh(
        config.lorentz_lambda
    )


def _taylor_action_scalar(
    operator: Any, vector: Tensor, coefficient: Tensor, config: TaylorConfig
) -> Tensor:
    if config.degree == 0:
        return vector

    def taylor_step(
        carry: Tuple[Tensor, Tensor], index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        total, term = carry
        denominator = backend.cast(config.scaling_steps * (index + 1), dtypestr)
        term = (coefficient / denominator) * (operator @ term)
        return total + term, term

    def scaling_step(state: Tensor, _: Tensor) -> Tensor:
        total, _ = backend.scan(
            taylor_step,
            backend.arange(config.degree),
            (state, state),
        )
        return total

    return backend.scan(
        scaling_step,
        backend.arange(config.scaling_steps),
        vector,
    )


def _taylor_action(
    operator: Any, vector: Tensor, coefficient: Tensor, config: TaylorConfig
) -> Tensor:
    coefficient, scalar = _check_query(coefficient, "coefficient")
    if scalar:
        return _taylor_action_scalar(operator, vector, coefficient, config)
    return backend.vmap(
        lambda item: _taylor_action_scalar(operator, vector, item, config)
    )(coefficient)


def _chebyshev_exponential_action(
    operator: Any,
    vector: Tensor,
    coefficient: Tensor,
    config: ChebyshevConfig,
    reference: Tensor,
    dimension: Optional[int],
) -> Tensor:
    coefficient, scalar = _check_query(coefficient, "coefficient")
    adapted, dimension = _prepare_operator(operator, vector=vector, dimension=dimension)

    def one(value: Tensor) -> Tensor:
        step_coefficient = value / backend.cast(config.scaling_steps, dtypestr)
        coefficients = _exponential_coefficients_from_bounds(
            step_coefficient,
            config.order,
            _as_tensor(config.bounds),
            reference,
        )

        def step(state: Tensor, _: Tensor) -> Tensor:
            return chebyshev_action(
                adapted, state, coefficients, config, dimension=dimension
            )

        return backend.scan(step, backend.arange(config.scaling_steps), vector)

    if scalar:
        return one(coefficient)
    return backend.vmap(one)(coefficient)


def _exponential_action(
    operator: Any,
    vector: Tensor,
    coefficient: Tensor,
    method: Union[KrylovConfig, ChebyshevConfig, TaylorConfig],
    *,
    dimension: Optional[int] = None,
    energy_shift: Optional[Tensor] = None,
    restore_shift: bool = True,
) -> Tensor:
    vector = backend.cast(_as_tensor(vector), dtypestr)
    shift = None
    action_operator = operator
    if energy_shift is not None:
        shift, shift_scalar = _check_query(energy_shift, "energy_shift")
        if not shift_scalar:
            raise ValueError("energy_shift must be a scalar.")
        shift = backend.cast(shift, dtypestr)
    if shift is not None and isinstance(method, (KrylovConfig, TaylorConfig)):
        adapted_operator, resolved_dimension = _prepare_operator(
            operator, vector=vector, dimension=dimension
        )

        def shifted_matvec(state: Tensor) -> Tensor:
            return adapted_operator @ state - shift * state

        action_operator = shifted_matvec
        dimension = resolved_dimension
    if isinstance(method, KrylovConfig):
        projection = lanczos_project(
            action_operator, vector, method, dimension=dimension
        )
        coefficient = backend.cast(_as_tensor(coefficient), dtypestr)
        result = lanczos_apply(
            projection,
            lambda nodes: (
                backend.exp(coefficient[..., None] * backend.cast(nodes, dtypestr))
                if _rank(coefficient) == 1
                else backend.exp(coefficient * backend.cast(nodes, dtypestr))
            ),
        )
    elif isinstance(method, ChebyshevConfig):
        if method.scaling_steps == 1:
            coefficients = (
                exponential_coefficients(coefficient, method)
                if shift is None
                else _exponential_coefficients_from_bounds(
                    coefficient, method.order, _as_tensor(method.bounds), shift
                )
            )
            result = chebyshev_action(
                operator, vector, coefficients, method, dimension=dimension
            )
        else:
            reference = (
                backend.convert_to_tensor(0.0, dtype=dtypestr)
                if shift is None
                else shift
            )
            result = _chebyshev_exponential_action(
                operator,
                vector,
                backend.cast(_as_tensor(coefficient), dtypestr),
                method,
                reference,
                dimension,
            )
    elif isinstance(method, TaylorConfig):
        adapted, resolved_dimension = _prepare_operator(
            action_operator, vector=vector, dimension=dimension
        )
        result = _taylor_action(
            adapted, vector, backend.cast(_as_tensor(coefficient), dtypestr), method
        )
        dimension = resolved_dimension
    else:
        raise TypeError(
            "method must be a KrylovConfig, ChebyshevConfig, or TaylorConfig."
        )
    if shift is not None and restore_shift:
        coefficient_tensor = backend.cast(_as_tensor(coefficient), dtypestr)
        factor = backend.exp(coefficient_tensor * shift)
        if _rank(factor) == 1:
            result = result * factor[..., None]
        else:
            result = result * factor
    return result


def exponential_action(
    operator: Any,
    vector: Tensor,
    coefficient: Tensor,
    method: Union[KrylovConfig, ChebyshevConfig, TaylorConfig],
    *,
    dimension: Optional[int] = None,
    energy_shift: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute the unnormalized action ``exp(coefficient * H) @ vector``.

    This is the common execution entry point for the fixed-shape Krylov,
    Chebyshev, and Taylor implementations. Choose the method through its
    configuration object; the operator can remain a matrix-free MVP callable.
    A rank-one coefficient query returns one output vector per query.

    The operator may be dense, sparse, matrix-free, or a linear operator.
    ``method`` is static for JIT; ``vector`` has shape ``(dimension,)`` and a
    rank-one coefficient query produces shape ``(num_queries, dimension)``.
    ``ChebyshevConfig.scaling_steps`` applies repeated shorter polynomial
    actions through a fixed backend scan. No normalization is performed.

    :param operator: Square operator or matrix-vector-product callable.
    :type operator: Any
    :param vector: One-dimensional input vector with shape ``(dimension,)``.
    :type vector: Tensor
    :param coefficient: Scalar or rank-one exponential coefficient query.
    :type coefficient: Tensor
    :param method: Static Krylov, Chebyshev, or Taylor configuration.
    :type method: Union[KrylovConfig, ChebyshevConfig, TaylorConfig]
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param energy_shift: Optional scalar shift restored analytically after the
        matrix-function action.
    :type energy_shift: Optional[Tensor]
    :return: Unnormalized matrix-function action.
    :rtype: Tensor
    """
    return _exponential_action(
        operator,
        vector,
        coefficient,
        method,
        dimension=dimension,
        energy_shift=energy_shift,
    )


def estimate_spectral_bounds(
    operator: Any,
    initial_vector: Tensor,
    config: KrylovConfig,
    *,
    dimension: Optional[int] = None,
    padding: float = 0.01,
) -> Tuple[Tensor, Tensor]:
    """
    Estimate ascending spectral bounds from an explicit Lanczos seed.

    This helper diagonalizes only the small projected Lanczos matrix. It
    returns Python floats eagerly and backend scalars under JIT. The bounds
    describe the spectrum visible from ``initial_vector``; increase
    ``max_dim`` or use a better seed when a full-operator enclosure is required.
    Construct a static :class:`ChebyshevConfig` from eager bounds outside JIT.

    :param operator: Hermitian square operator or matrix-vector-product callable.
    :type operator: Any
    :param initial_vector: One-dimensional nonzero seed vector.
    :type initial_vector: Tensor
    :param config: Static Krylov dimension and recurrence policy.
    :type config: KrylovConfig
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param padding: Static Python relative padding applied to the active Ritz range.
    :type padding: float
    :return: Ascending spectral bounds ``(emin, emax)``.
    :rtype: Tuple[Tensor, Tensor]
    """
    if padding < 0 or not math.isfinite(float(padding)):
        raise ValueError("padding must be a finite non-negative number.")
    projection = lanczos_project(operator, initial_vector, config, dimension=dimension)
    values, _, _, active = _projection_spectral_data(projection)
    infinity = backend.convert_to_tensor(float("inf"), dtype=rdtypestr)
    emin = backend.min(backend.where(active, values, infinity))
    emax = backend.max(backend.where(active, values, -infinity))
    width = emax - emin
    scale = backend.where(backend.abs(emax) > 1.0, backend.abs(emax), 1.0)
    margin = backend.where(width == 0, scale, width) * padding
    lower, upper = emin - margin, emax + margin
    concrete_lower, concrete_upper = _concrete_float(lower), _concrete_float(upper)
    if concrete_lower is not None and concrete_upper is not None:
        return concrete_lower, concrete_upper
    return lower, upper
