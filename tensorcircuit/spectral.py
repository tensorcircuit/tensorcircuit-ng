"""
Physics-facing density-of-states, thermal, and response estimators.

The module separates physical quantities from the numerical reductions in
``tensorcircuit.matrixfunc``. A reader can therefore start from a Hamiltonian,
an observable, and a few probes without choosing a recurrence implementation.
The lower-level module remains available when a calculation needs explicit
control over a Krylov projection or Chebyshev moments.

::

    physics quantities
             |-------------------|-------------------|
            DOS               Thermal             Response
             |                   |                   |
        KPM / SLQ      partition / energy      resolvent / time
             |                   |                   |
       Chebyshev /     SLQ / Chebyshev /      Krylov / Chebyshev /
          Lanczos            typicality              Taylor

DOS is a density reconstructed from a Chebyshev moment sequence (KPM) or a
broadened Lanczos spectral measure (SLQ). Thermal functions evaluate traces of
``exp(-beta * H)`` and ratios of the corresponding energy moments. Response
functions evaluate resolvents or time correlations. These physical names are
the intended user-facing entry points; ``matrixfunc`` supplies their reusable
actions, projections, and moment contractions.
"""

import math
from typing import Any, Literal, Optional, Tuple, Union

from .cons import backend, dtypestr, rdtypestr
from . import matrixfunc
from .matrixfunc import (
    _as_tensor,
    _check_query,
    _concrete_float,
    _rank,
    ChebyshevConfig,
    ChebyshevMomentResult,
    KrylovConfig,
    LanczosMeasure,
    TaylorConfig,
)

Tensor = Any
Estimate = Union[Tensor, Tuple[Tensor, Tensor]]
_DEFAULT_KRYLOV_CONFIG = KrylovConfig(max_dim=32)


def _check_nonnegative(value: Tensor, name: str) -> None:
    minimum = _concrete_float(backend.min(value))
    if minimum is not None and minimum < 0:
        raise ValueError(f"{name} must be non-negative.")


def _check_positive(value: Tensor, name: str) -> None:
    minimum = _concrete_float(backend.min(value))
    if minimum is not None and minimum <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_probes(probes: Tensor) -> Tensor:
    probes = _as_tensor(probes)
    if _rank(probes) != 2:
        raise ValueError("probes must have shape (num_probes, dimension).")
    if backend.shape_tuple(probes)[0] < 1:
        raise ValueError("stochastic spectral estimators require at least one probe.")
    return probes


def _normalization_factor(
    normalization: Literal["probability", "states"], dimension: Tensor
) -> Tensor:
    if normalization not in ("probability", "states"):
        raise ValueError("normalization must be 'probability' or 'states'.")
    if normalization == "states":
        return backend.convert_to_tensor(1.0, dtype=rdtypestr)
    return backend.cast(dimension, rdtypestr)


def _format_estimate(value: Tensor, standard_error: Tensor, with_std: bool) -> Estimate:
    """
    Apply the package-wide ``with_std`` return convention.
    """
    if with_std:
        return value, standard_error
    return value


def _ratio_statistics(
    numerator: Tensor, denominator: Tensor, with_std: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Estimate a ratio from paired samples with a delta-method error.
    """
    count = backend.shape_tuple(numerator)[-1]
    mean_n = backend.mean(numerator, axis=(-1,))
    mean_d = backend.mean(denominator, axis=(-1,))
    value = mean_n / mean_d
    if not with_std:
        return value, backend.zeros_like(value, dtype=rdtypestr)
    if count < 2:
        raise ValueError("at least two paired samples are required.")
    centered_n = numerator - mean_n[..., None]
    centered_d = denominator - mean_d[..., None]
    variance_n = backend.real(
        backend.mean(backend.abs(centered_n) * backend.abs(centered_n), axis=(-1,))
    )
    variance_d = backend.real(
        backend.mean(backend.abs(centered_d) * backend.abs(centered_d), axis=(-1,))
    )
    covariance = backend.real(
        backend.mean(centered_n * backend.conj(centered_d), axis=(-1,))
    )
    correction = count / (count - 1)
    variance_n = variance_n * correction
    variance_d = variance_d * correction
    covariance = covariance * correction
    delta_variance = (
        variance_n / (mean_d * mean_d)
        + mean_n * mean_n * variance_d / (mean_d * mean_d * mean_d * mean_d)
        - 2.0 * mean_n * covariance / (mean_d * mean_d * mean_d)
    ) / count
    return value, backend.sqrt(backend.abs(delta_variance))


def _log_statistics(samples: Tensor, with_std: bool = True) -> Tuple[Tensor, Tensor]:
    samples = backend.real(samples)
    count = backend.shape_tuple(samples)[-1]
    mean = backend.mean(samples, axis=(-1,))
    concrete_mean = _concrete_float(backend.min(mean))
    if concrete_mean is not None and concrete_mean <= 0:
        raise ValueError("partition-function estimate is non-positive.")
    value = backend.log(mean)
    if not with_std:
        return value, backend.zeros_like(value, dtype=rdtypestr)
    if count < 2:
        raise ValueError("at least two probes are required.")
    centered = samples - mean[..., None]
    variance = backend.real(
        backend.mean(backend.abs(centered) * backend.abs(centered), axis=(-1,))
    )
    stderr_mean = backend.sqrt(variance / count * (count / (count - 1)))
    return value, stderr_mean / backend.abs(mean)


def random_trace_probes(
    num_probes: int,
    dimension: int,
    *,
    distribution: Literal["rademacher", "random_phase"] = "random_phase",
    status: Optional[Tensor] = None,
) -> Tensor:
    """
    Generate explicit Hutchinson probes.

    The returned tensor has shape ``(num_probes, dimension)``; both supported
    distributions have ``E[r r^dagger] = I`` and use unit ``probe_scale``.
    Without ``status``, draw uniforms from the backend's implicit random state;
    use this mode to prepare probes before JIT compilation. Pass explicit
    uniform ``status`` when generating probes inside a compiled function.

    :param num_probes: Number of probe rows to generate.
    :type num_probes: int
    :param dimension: Hilbert-space dimension of each probe.
    :type dimension: int
    :param distribution: Probe distribution, either ``"rademacher"`` or ``"random_phase"``.
    :type distribution: Literal["rademacher", "random_phase"]
    :param status: Optional external uniform random tensor with shape
        ``(num_probes, dimension)`` and values in ``[0, 1)``.
    :type status: Optional[Tensor]
    :return: Probe matrix with shape ``(num_probes, dimension)``.
    :rtype: Tensor
    """
    if not isinstance(num_probes, int) or num_probes < 1:
        raise ValueError("num_probes must be a positive integer.")
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if distribution not in ("rademacher", "random_phase"):
        raise ValueError("distribution must be 'rademacher' or 'random_phase'.")
    if status is None:
        uniform = backend.implicit_randu(shape=[num_probes, dimension], dtype=rdtypestr)
    else:
        uniform = backend.cast(backend.convert_to_tensor(status), rdtypestr)
        if backend.shape_tuple(uniform) != (num_probes, dimension):
            raise ValueError("status must have shape (num_probes, dimension).")
    if distribution == "rademacher":
        one = backend.ones_like(uniform, dtype=rdtypestr)
        return backend.where(uniform < 0.5, one, -one)
    return backend.exp(
        backend.cast(2.0j * math.pi, dtypestr) * backend.cast(uniform, dtypestr)
    )


def _validate_prepared_bounds(
    moments: ChebyshevMomentResult, bounds: Tuple[float, float]
) -> Tensor:
    """
    Reject eager mismatches and mark traced mismatches for NaN coefficients.

    JAX tracers cannot be compared on the host while tracing. In that case,
    the returned tensor mask checks the bounds at execution time.
    """
    lower = _concrete_float(moments.bounds[0])
    upper = _concrete_float(moments.bounds[1])
    if (
        lower is not None
        and upper is not None
        and not (
            math.isclose(lower, bounds[0], rel_tol=1.0e-6, abs_tol=1.0e-6)
            and math.isclose(upper, bounds[1], rel_tol=1.0e-6, abs_tol=1.0e-6)
        )
    ):
        raise ValueError(
            "prepared moments bounds do not match the evaluation configuration."
        )
    expected = backend.convert_to_tensor(bounds, dtype=rdtypestr)
    difference = backend.abs(backend.cast(moments.bounds, rdtypestr) - expected)
    tolerance = 1.0e-6 * (1.0 + backend.abs(expected))
    return backend.max(backend.cast(difference > tolerance, rdtypestr)) == 0.0


def density_of_states_from_moments(
    moments: ChebyshevMomentResult,
    energies: Tensor,
    *,
    config: ChebyshevConfig,
    probe_scale: Tensor = 1.0,
    normalization: Literal["probability", "states"] = "probability",
    with_std: bool = False,
) -> Estimate:
    """
    Reconstruct a DOS from prepared stochastic Chebyshev moments.

    ``moments`` must have shape ``(num_probes, config.order)`` and must have
    been generated with exactly the same ascending bounds as ``config``.
    ``energies`` is scalar or rank one and must lie strictly inside those
    bounds. ``normalization="probability"`` divides by the Hilbert-space
    dimension; ``"states"`` returns the state-counting DOS.

    :param moments: Prepared Chebyshev moments with shape ``(num_probes, order)``.
    :type moments: ChebyshevMomentResult
    :param energies: Scalar or rank-one energy query inside the configured bounds.
    :type energies: Tensor
    :param config: Chebyshev configuration matching the prepared moments. Its
        ``kernel`` selects the DOS damping rule.
    :type config: ChebyshevConfig
    :param probe_scale: Multiplicative trace-normalization factor.
    :type probe_scale: Tensor
    :param normalization: Either ``"probability"`` or ``"states"``.
    :type normalization: Literal["probability", "states"]
    :param with_std: If true, return ``(dos, standard_error)``; otherwise
        return only the DOS estimate.
    :type with_std: bool, optional
    :return: DOS values, optionally paired with their probe standard errors.
    :rtype: Estimate
    """
    if not isinstance(moments, ChebyshevMomentResult):
        raise TypeError("moments must be a ChebyshevMomentResult.")
    matrixfunc._validate_trace_moments(moments)
    if not isinstance(config, ChebyshevConfig):
        raise TypeError("config must be a ChebyshevConfig.")
    if config.order < 2:
        raise ValueError("DOS reconstruction requires at least two moments.")
    if backend.shape_tuple(moments.moments)[-1] != config.order:
        raise ValueError("moment order does not match ChebyshevConfig.order.")
    bounds_match = _validate_prepared_bounds(moments, config.bounds)
    energies, _ = _check_query(energies, "energies")
    matrixfunc._validate_inside(energies, config.bounds, "energies")
    coefficients = matrixfunc.delta_coefficients(energies, config)
    coefficients = backend.where(
        bounds_match,
        coefficients,
        backend.cast(float("nan"), backend.dtype(coefficients)),
    )
    result = matrixfunc.evaluate_chebyshev_trace(
        moments,
        coefficients,
        kernel_weights=matrixfunc.kernel_weights(config),
        probe_scale=probe_scale,
        with_std=with_std,
    )
    factor = _normalization_factor(normalization, moments.dimension)
    if with_std:
        value, standard_error = result
        return (
            backend.real(value) / factor,
            standard_error / backend.abs(factor),
        )
    return backend.real(result) / factor


def density_of_states_from_slq(
    measure: LanczosMeasure,
    energies: Tensor,
    *,
    broadening: Tensor,
    lineshape: Literal["lorentzian", "gaussian"] = "lorentzian",
    probe_scale: Tensor = 1.0,
    normalization: Literal["probability", "states"] = "probability",
    with_std: bool = False,
) -> Estimate:
    """
    Reconstruct a broadened DOS from a prepared SLQ measure.

    ``energies`` is scalar or rank one, while ``broadening`` is strictly
    positive. The measure must contain explicit active Ritz-node masks; padded
    inactive slots never contribute to the lineshape.

    :param measure: Prepared per-probe SLQ measure.
    :type measure: LanczosMeasure
    :param energies: Scalar or rank-one energy query.
    :type energies: Tensor
    :param broadening: Strictly positive Lorentzian or Gaussian width.
    :type broadening: Tensor
    :param lineshape: Broadening profile, either ``"lorentzian"`` or ``"gaussian"``.
    :type lineshape: Literal["lorentzian", "gaussian"]
    :param probe_scale: Multiplicative trace-normalization factor.
    :type probe_scale: Tensor
    :param normalization: Either ``"probability"`` or ``"states"``.
    :type normalization: Literal["probability", "states"]
    :param with_std: If true, return ``(dos, standard_error)``; otherwise
        return only the broadened DOS estimate.
    :type with_std: bool, optional
    :return: Broadened DOS values, optionally paired with their probe standard
        errors.
    :rtype: Estimate
    """
    if not isinstance(measure, LanczosMeasure):
        raise TypeError("measure must be a LanczosMeasure.")
    if lineshape not in ("lorentzian", "gaussian"):
        raise ValueError("lineshape must be 'lorentzian' or 'gaussian'.")
    broadening = backend.cast(_as_tensor(broadening), rdtypestr)
    _check_positive(broadening, "broadening")
    energies, scalar = _check_query(energies, "energies")
    nodes = backend.cast(measure.nodes, rdtypestr)
    weights = backend.cast(measure.weights, rdtypestr)
    weights = weights * backend.cast(measure.active, rdtypestr)
    if scalar:
        difference = energies - nodes
        if lineshape == "lorentzian":
            line = broadening / (
                math.pi * (difference * difference + broadening * broadening)
            )
        else:
            line = backend.exp(
                -((difference / broadening) * (difference / broadening))
            ) / (math.sqrt(math.pi) * broadening)
        samples = backend.sum(weights * line, axis=-1)
    else:
        difference = energies[:, None, None] - nodes[None, :, :]
        if lineshape == "lorentzian":
            line = broadening / (
                math.pi * (difference * difference + broadening * broadening)
            )
        else:
            ratio = difference / broadening
            line = backend.exp(-(ratio * ratio)) / (math.sqrt(math.pi) * broadening)
        samples = backend.sum(weights[None, :, :] * line, axis=-1)
    value, standard_error = matrixfunc._sample_statistics(
        samples, probe_scale, with_std=with_std
    )
    factor = _normalization_factor(normalization, measure.dimension)
    return _format_estimate(
        value / factor, standard_error / backend.abs(factor), with_std
    )


def density_of_states(
    operator: Any,
    energies: Tensor,
    *,
    method: Optional[Union[ChebyshevConfig, KrylovConfig]] = None,
    probes: Tensor,
    dimension: Optional[int] = None,
    probe_scale: Tensor = 1.0,
    probe_batch_size: Optional[int] = None,
    normalization: Literal["probability", "states"] = "probability",
    broadening: Optional[Tensor] = None,
    lineshape: Literal["lorentzian", "gaussian"] = "lorentzian",
    with_std: bool = False,
) -> Estimate:
    """
    Estimate a normalized or state-counting density of states.

    If ``method`` is omitted, a 32-vector Lanczos measure is used. The
    ``ChebyshevConfig`` selects Chebyshev moments and uses its ``kernel`` for
    reconstruction; ``KrylovConfig`` selects SLQ and requires positive
    broadening. Probe rows have shape ``(num_probes, dimension)`` and optional
    batching is static.

    :param operator: Hermitian square operator or matrix-vector-product callable.
    :type operator: Any
    :param energies: Scalar or rank-one energy query.
    :type energies: Tensor
    :param method: Optional Chebyshev or SLQ configuration. The default is a
        32-vector Lanczos measure; SLQ still requires an explicit physical
        ``broadening`` because its units cannot be inferred from the operator.
    :type method: Optional[Union[ChebyshevConfig, KrylovConfig]]
    :param probes: Explicit probe matrix with shape ``(num_probes, dimension)``.
    :type probes: Tensor
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param probe_scale: Multiplicative trace-normalization factor.
    :type probe_scale: Tensor
    :param probe_batch_size: Optional positive static probe block size.
    :type probe_batch_size: Optional[int]
    :param normalization: Either ``"probability"`` or ``"states"``.
    :type normalization: Literal["probability", "states"]
    :param broadening: Required positive width for SLQ and forbidden for
        Chebyshev DOS reconstruction.
    :type broadening: Optional[Tensor]
    :param lineshape: SLQ broadening profile.
    :type lineshape: Literal["lorentzian", "gaussian"]
    :param with_std: If true, return ``(dos, standard_error)``; otherwise
        return only the DOS estimate.
    :type with_std: bool, optional
    :return: DOS estimate, optionally paired with its probe standard error.
    :rtype: Estimate
    """
    probes = _require_probes(probes)
    if method is None:
        method = _DEFAULT_KRYLOV_CONFIG
    if isinstance(method, ChebyshevConfig):
        if broadening is not None:
            raise ValueError(
                "Chebyshev DOS resolution is set by order and kernel; broadening must be None."
            )
        moments = matrixfunc.stochastic_chebyshev_moments(
            operator,
            probes,
            method,
            dimension=dimension,
            probe_batch_size=probe_batch_size,
        )
        return density_of_states_from_moments(
            moments,
            energies,
            config=method,
            probe_scale=probe_scale,
            normalization=normalization,
            with_std=with_std,
        )
    if isinstance(method, KrylovConfig):
        if broadening is None:
            raise ValueError("SLQ DOS reconstruction requires broadening > 0.")
        measure = matrixfunc.slq_measure(
            operator,
            probes,
            method,
            dimension=dimension,
            probe_batch_size=probe_batch_size,
        )
        return density_of_states_from_slq(
            measure,
            energies,
            broadening=broadening,
            lineshape=lineshape,
            probe_scale=probe_scale,
            normalization=normalization,
            with_std=with_std,
        )
    raise TypeError("method must be a ChebyshevConfig or KrylovConfig.")


def _slq_relative_data(
    measure: LanczosMeasure, energy_shift: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Prepare automatically centered Ritz nodes and active weights."""
    if energy_shift is not None:
        _, shift_scalar = _check_query(energy_shift, "energy_shift")
        if not shift_scalar:
            raise ValueError("energy_shift must be a scalar.")
    nodes = backend.cast(measure.nodes, rdtypestr)
    active = measure.active
    infinity = backend.convert_to_tensor(float("inf"), dtype=rdtypestr)
    reference = backend.min(backend.where(active, nodes, infinity))
    relative_nodes = backend.where(active, nodes - reference, 0.0)
    active_weights = measure.weights * backend.cast(active, rdtypestr)
    return relative_nodes, active_weights, reference


def _slq_partition_samples(
    measure: LanczosMeasure, beta: Tensor, energy_shift: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    beta, scalar = _check_query(beta, "beta")
    _check_nonnegative(beta, "beta")
    relative_nodes, active_weights, reference = _slq_relative_data(
        measure, energy_shift
    )
    correction = -beta * reference
    if scalar:
        samples = backend.sum(
            active_weights * backend.exp(-beta * relative_nodes), axis=-1
        )
        return samples, correction
    samples = backend.sum(
        active_weights[None, :, :]
        * backend.exp(-beta[:, None, None] * relative_nodes[None, :, :]),
        axis=-1,
    )
    return samples, correction


def log_partition_function_from_slq(
    measure: LanczosMeasure,
    beta: Tensor,
    *,
    probe_scale: Tensor = 1.0,
    energy_shift: Optional[Tensor] = None,
    with_std: bool = False,
) -> Estimate:
    """
    Evaluate ``log Tr exp(-beta H)`` from a prepared SLQ measure.

    ``beta`` is non-negative and scalar or rank one. Per-probe shifted weights
    are combined with a log estimator. SLQ automatically uses its smallest
    active Ritz node as the numerical reference; ``energy_shift`` is accepted
    only for API compatibility with the Chebyshev path.

    :param measure: Prepared per-probe SLQ measure.
    :type measure: LanczosMeasure
    :param beta: Non-negative inverse temperature, scalar or rank one.
    :type beta: Tensor
    :param probe_scale: Multiplicative trace-normalization factor.
    :type probe_scale: Tensor
    :param energy_shift: Optional scalar reference accepted for API compatibility; SLQ chooses its own stable reference.
    :type energy_shift: Optional[Tensor]
    :param with_std: If true, return ``(log_z, standard_error)``; otherwise
        return only ``log_z``.
    :type with_std: bool, optional
    :return: The logarithm of the partition function, optionally paired with
        its probe standard error.
    :rtype: Estimate
    """
    if not isinstance(measure, LanczosMeasure):
        raise TypeError("measure must be a LanczosMeasure.")
    samples, correction = _slq_partition_samples(measure, beta, energy_shift)
    value, standard_error = _log_statistics(
        samples * _as_tensor(probe_scale), with_std=with_std
    )
    return _format_estimate(value + correction, standard_error, with_std)


def log_partition_function_from_moments(
    moments: ChebyshevMomentResult,
    beta: Tensor,
    *,
    probe_scale: Tensor = 1.0,
    energy_shift: Optional[Tensor] = None,
    with_std: bool = False,
) -> Estimate:
    """
    Evaluate ``log Tr exp(-beta H)`` from prepared Chebyshev moments.

    The moments must have shape ``(num_probes, order)``; a rank-one bilinear
    sequence cannot represent a trace. The moments retain their original
    bounds. Exponential coefficients are evaluated from the shared backend
    Bessel path around the supplied energy reference, which is restored
    analytically.

    :param moments: Prepared Chebyshev moments.
    :type moments: ChebyshevMomentResult
    :param beta: Non-negative inverse temperature, scalar or rank one.
    :type beta: Tensor
    :param probe_scale: Multiplicative trace-normalization factor.
    :type probe_scale: Tensor
    :param energy_shift: Optional scalar energy reference removed before exponentiation.
    :type energy_shift: Optional[Tensor]
    :param with_std: If true, return ``(log_z, standard_error)``; otherwise
        return only ``log_z``.
    :type with_std: bool, optional
    :return: The logarithm of the partition function, optionally paired with
        its probe standard error.
    :rtype: Estimate
    """
    if not isinstance(moments, ChebyshevMomentResult):
        raise TypeError("moments must be a ChebyshevMomentResult.")
    matrixfunc._validate_trace_moments(moments)
    beta, _ = _check_query(beta, "beta")
    _check_nonnegative(beta, "beta")
    if energy_shift is None:
        shift = backend.convert_to_tensor(0.0, dtype=rdtypestr)
    else:
        shift, shift_scalar = _check_query(energy_shift, "energy_shift")
        if not shift_scalar:
            raise ValueError("energy_shift must be a scalar.")
        shift = backend.cast(shift, rdtypestr)
    beta_tensor = beta
    order = backend.shape_tuple(moments.moments)[-1]
    bounds = moments.bounds
    coefficients = matrixfunc._energy_polynomial_coefficients(
        -beta_tensor, order, bounds, shift, 0, False
    )
    values = matrixfunc.evaluate_chebyshev_moments(moments, coefficients)
    scale = backend.cast((bounds[1] - bounds[0]) / 2.0, rdtypestr)
    estimate, standard_error = _log_statistics(
        values * _as_tensor(probe_scale), with_std=with_std
    )
    return _format_estimate(
        estimate + beta * scale - beta * shift, standard_error, with_std
    )


def log_partition_function(
    operator: Any,
    beta: Tensor,
    *,
    method: Optional[Union[KrylovConfig, ChebyshevConfig]] = None,
    probes: Tensor,
    dimension: Optional[int] = None,
    probe_scale: Tensor = 1.0,
    probe_batch_size: Optional[int] = None,
    energy_shift: Optional[Tensor] = None,
    with_std: bool = False,
) -> Estimate:
    """
    Estimate the stable logarithm of a partition function.

    ``method`` is either ``KrylovConfig`` or ``ChebyshevConfig``; ``probes``
    has shape ``(num_probes, dimension)`` and contains at least one row. A
    standard error requires at least two rows.

    :param operator: Hermitian square operator or matrix-vector-product callable.
    :type operator: Any
    :param beta: Non-negative inverse temperature, scalar or rank one.
    :type beta: Tensor
    :param method: Krylov or Chebyshev configuration.
    :type method: Union[KrylovConfig, ChebyshevConfig]
    :param probes: Explicit probe matrix with shape ``(num_probes, dimension)``.
    :type probes: Tensor
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param probe_scale: Multiplicative trace-normalization factor.
    :type probe_scale: Tensor
    :param probe_batch_size: Optional positive static probe block size.
    :type probe_batch_size: Optional[int]
    :param energy_shift: Optional scalar energy reference removed before exponentiation.
    :type energy_shift: Optional[Tensor]
    :param with_std: If true, return ``(log_z, standard_error)``; otherwise
        return only ``log_z``.
    :type with_std: bool, optional
    :return: The logarithm of the partition function, optionally paired with
        its probe standard error.
    :rtype: Estimate
    """
    probes = _require_probes(probes)
    if method is None:
        method = _DEFAULT_KRYLOV_CONFIG
    if isinstance(method, KrylovConfig):
        measure = matrixfunc.slq_measure(
            operator,
            probes,
            method,
            dimension=dimension,
            probe_batch_size=probe_batch_size,
        )
        return log_partition_function_from_slq(
            measure,
            beta,
            probe_scale=probe_scale,
            energy_shift=energy_shift,
            with_std=with_std,
        )
    if isinstance(method, ChebyshevConfig):
        moments = matrixfunc.stochastic_chebyshev_moments(
            operator,
            probes,
            method,
            dimension=dimension,
            probe_batch_size=probe_batch_size,
        )
        return log_partition_function_from_moments(
            moments,
            beta,
            probe_scale=probe_scale,
            energy_shift=energy_shift,
            with_std=with_std,
        )
    raise TypeError("method must be a KrylovConfig or ChebyshevConfig.")


def partition_function(
    operator: Any,
    beta: Tensor,
    *,
    method: Optional[Union[KrylovConfig, ChebyshevConfig]] = None,
    probes: Tensor,
    dimension: Optional[int] = None,
    probe_scale: Tensor = 1.0,
    probe_batch_size: Optional[int] = None,
    energy_shift: Optional[Tensor] = None,
    with_std: bool = False,
) -> Estimate:
    """
    Exponentiate :func:`log_partition_function` explicitly.

    This convenience conversion can overflow when the physical partition
    function itself is outside the active floating-point range.

    :param operator: Hermitian square operator or matrix-vector-product callable.
    :type operator: Any
    :param beta: Non-negative inverse temperature, scalar or rank one.
    :type beta: Tensor
    :param method: Krylov or Chebyshev configuration.
    :type method: Union[KrylovConfig, ChebyshevConfig]
    :param probes: Explicit probe matrix with shape ``(num_probes, dimension)``.
    :type probes: Tensor
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param probe_scale: Multiplicative trace-normalization factor.
    :type probe_scale: Tensor
    :param probe_batch_size: Optional positive static probe block size.
    :type probe_batch_size: Optional[int]
    :param energy_shift: Optional scalar energy reference removed before exponentiation.
    :type energy_shift: Optional[Tensor]
    :param with_std: If true, return ``(z, standard_error)``; otherwise return
        only the partition function.
    :type with_std: bool, optional
    :return: The partition function, optionally paired with its propagated
        probe standard error.
    :rtype: Estimate
    """
    logz = log_partition_function(
        operator,
        beta,
        method=method,
        probes=probes,
        dimension=dimension,
        probe_scale=probe_scale,
        probe_batch_size=probe_batch_size,
        energy_shift=energy_shift,
        with_std=with_std,
    )
    if with_std:
        logz_value, logz_error = logz
        value = backend.exp(logz_value)
        return value, backend.abs(value) * logz_error
    return backend.exp(logz)


def free_energy(logz: Estimate, beta: Tensor, *, with_std: bool = False) -> Estimate:
    """
    Compute ``F=-log(Z)/beta`` with propagated uncertainty.

    ``beta`` must be strictly positive and has the same scalar/query shape as
    ``logz`` may be either the scalar result of ``log_partition_function`` or
    the ``(value, standard_error)`` pair returned when ``with_std=True``.

    :param logz: Log partition-function estimate.
    :type logz: Estimate
    :param beta: Strictly positive inverse temperature.
    :type beta: Tensor
    :param with_std: If true, return ``(free_energy, standard_error)``.
    :type with_std: bool, optional
    :return: Free energy, optionally paired with its propagated standard error.
    :rtype: Estimate
    """
    if isinstance(logz, tuple):
        logz_value, logz_error = logz
    else:
        logz_value = logz
        logz_error = backend.zeros_like(logz_value, dtype=rdtypestr)
    beta, _ = _check_query(beta, "beta")
    _check_positive(beta, "beta")
    return _format_estimate(
        -logz_value / beta,
        logz_error / backend.abs(beta),
        with_std,
    )


def _slq_thermal_terms(
    measure: LanczosMeasure,
    beta: Tensor,
    energy_shift: Optional[Tensor],
    centered: bool,
) -> Tuple[Tensor, Tensor]:
    """Build shifted SLQ weights and energy nodes once for related moments."""
    beta, scalar = _check_query(beta, "beta")
    _check_nonnegative(beta, "beta")
    relative_nodes, active_weights, reference = _slq_relative_data(
        measure, energy_shift
    )
    physical_nodes = relative_nodes + reference
    sample_nodes = relative_nodes if centered else physical_nodes
    if scalar:
        return backend.exp(-beta * relative_nodes) * active_weights, sample_nodes
    return (
        backend.exp(-beta[:, None, None] * relative_nodes[None, :, :])
        * active_weights[None, :, :],
        sample_nodes,
    )


def _slq_thermal_samples(
    measure: LanczosMeasure,
    beta: Tensor,
    power: int,
    energy_shift: Optional[Tensor],
    centered: bool = False,
) -> Tuple[Tensor, Tensor]:
    weights, sample_nodes = _slq_thermal_terms(measure, beta, energy_shift, centered)
    if _rank(weights) == 3:
        sample_nodes = sample_nodes[None, :, :]
    return (
        backend.sum(weights * (sample_nodes**power), axis=-1),
        backend.sum(weights, axis=-1),
    )


def _moment_thermal_samples(
    moments: ChebyshevMomentResult,
    beta: Tensor,
    power: int,
    energy_shift: Optional[Tensor],
    centered: bool = False,
    denominator: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    beta, _ = _check_query(beta, "beta")
    _check_nonnegative(beta, "beta")
    if energy_shift is None:
        shift = backend.convert_to_tensor(0.0, dtype=rdtypestr)
    else:
        shift, shift_scalar = _check_query(energy_shift, "energy_shift")
        if not shift_scalar:
            raise ValueError("energy_shift must be a scalar.")
        shift = backend.cast(shift, rdtypestr)
    order = backend.shape_tuple(moments.moments)[-1]
    numerator = matrixfunc.evaluate_chebyshev_moments(
        moments,
        matrixfunc._energy_polynomial_coefficients(
            -beta, order, moments.bounds, shift, power, centered
        ),
    )
    if denominator is None:
        denominator = matrixfunc.evaluate_chebyshev_moments(
            moments,
            matrixfunc._energy_polynomial_coefficients(
                -beta, order, moments.bounds, shift, 0, centered
            ),
        )
    return numerator, denominator


def thermal_energy(
    measure: Union[LanczosMeasure, ChebyshevMomentResult],
    beta: Tensor,
    *,
    energy_shift: Optional[Tensor] = None,
    with_std: bool = False,
) -> Estimate:
    """
    Evaluate ``<H>_beta`` from a reusable SLQ measure or trace Chebyshev
    moments.

    Chebyshev moments must have shape ``(num_probes, order)``. A rank-one
    bilinear sequence is not a trace. ``energy_shift`` stabilizes Chebyshev
    weights and cancels from the final ratio; SLQ chooses its own Ritz-node
    reference. ``beta`` is non-negative and scalar or rank one.

    :param measure: Reusable SLQ measure or Chebyshev moments.
    :type measure: Union[LanczosMeasure, ChebyshevMomentResult]
    :param beta: Non-negative inverse temperature, scalar or rank one.
    :type beta: Tensor
    :param energy_shift: Optional scalar energy reference used by Chebyshev
        moments; SLQ chooses its own stable reference.
    :type energy_shift: Optional[Tensor]
    :param with_std: If true, return ``(energy, standard_error)``; otherwise
        return only the thermal energy.
    :type with_std: bool, optional
    :return: Thermal energy, optionally paired with its probe standard error.
    :rtype: Estimate
    """
    if isinstance(measure, LanczosMeasure):
        numerator, denominator = _slq_thermal_samples(measure, beta, 1, energy_shift)
    elif isinstance(measure, ChebyshevMomentResult):
        matrixfunc._validate_trace_moments(measure)
        numerator, denominator = _moment_thermal_samples(measure, beta, 1, energy_shift)
    else:
        raise TypeError("measure must be a LanczosMeasure or ChebyshevMomentResult.")
    numerator = backend.real(numerator)
    denominator = backend.real(denominator)
    value, standard_error = _ratio_statistics(numerator, denominator, with_std=with_std)
    return _format_estimate(value, standard_error, with_std)


def _heat_capacity_statistics(
    denominator: Tensor,
    first: Tensor,
    second: Tensor,
    beta: Tensor,
    with_std: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Apply the delta method to the joint ``(Z, HZ, H2Z)`` samples.
    """
    count = backend.shape_tuple(denominator)[-1]
    mean_d = backend.mean(denominator, axis=(-1,))
    mean_1 = backend.mean(first, axis=(-1,))
    mean_2 = backend.mean(second, axis=(-1,))
    mean = mean_1 / mean_d
    value = beta * beta * (mean_2 / mean_d - mean * mean)
    if not with_std:
        return value, backend.zeros_like(value, dtype=rdtypestr)
    if count < 2:
        raise ValueError("at least two paired samples are required.")

    centered_d = denominator - mean_d[..., None]
    centered_1 = first - mean_1[..., None]
    centered_2 = second - mean_2[..., None]
    correction = count / (count - 1)
    covariance_dd = (
        backend.real(backend.mean(centered_d * centered_d, axis=(-1,))) * correction
    )
    covariance_d1 = (
        backend.real(backend.mean(centered_d * centered_1, axis=(-1,))) * correction
    )
    covariance_d2 = (
        backend.real(backend.mean(centered_d * centered_2, axis=(-1,))) * correction
    )
    covariance_11 = (
        backend.real(backend.mean(centered_1 * centered_1, axis=(-1,))) * correction
    )
    covariance_12 = (
        backend.real(backend.mean(centered_1 * centered_2, axis=(-1,))) * correction
    )
    covariance_22 = (
        backend.real(backend.mean(centered_2 * centered_2, axis=(-1,))) * correction
    )
    gradient_d = (
        beta
        * beta
        * (-mean_2 / (mean_d * mean_d) + 2.0 * mean_1 * mean_1 / (mean_d**3))
    )
    gradient_1 = beta * beta * (-2.0 * mean_1 / (mean_d * mean_d))
    gradient_2 = beta * beta / mean_d
    delta_variance = (
        gradient_d * gradient_d * covariance_dd
        + gradient_1 * gradient_1 * covariance_11
        + gradient_2 * gradient_2 * covariance_22
        + 2.0 * gradient_d * gradient_1 * covariance_d1
        + 2.0 * gradient_d * gradient_2 * covariance_d2
        + 2.0 * gradient_1 * gradient_2 * covariance_12
    ) / count
    return value, backend.sqrt(backend.abs(delta_variance))


def heat_capacity(
    measure: Union[LanczosMeasure, ChebyshevMomentResult],
    beta: Tensor,
    *,
    energy_shift: Optional[Tensor] = None,
    with_std: bool = False,
) -> Estimate:
    """
    Evaluate ``beta**2 Var(H)`` from a trace measure with joint covariance
    propagation.

    Chebyshev moments must have shape ``(num_probes, order)``. A rank-one
    bilinear sequence is not a trace. The standard error is obtained by
    applying the delta method directly to the paired per-probe
    ``(Z, HZ, H2Z)`` samples, including all covariances.

    :param measure: Reusable SLQ measure or Chebyshev moments.
    :type measure: Union[LanczosMeasure, ChebyshevMomentResult]
    :param beta: Non-negative inverse temperature, scalar or rank one.
    :type beta: Tensor
    :param energy_shift: Optional scalar energy reference used by Chebyshev
        moments; SLQ chooses its own stable reference.
    :type energy_shift: Optional[Tensor]
    :param with_std: If true, return ``(heat_capacity, standard_error)``;
        otherwise return only the heat capacity.
    :type with_std: bool, optional
    :return: Heat capacity, optionally paired with its joint-covariance standard
        error.
    :rtype: Estimate
    """
    if isinstance(measure, LanczosMeasure):
        weights, sample_nodes = _slq_thermal_terms(
            measure, beta, energy_shift, centered=True
        )
        if _rank(weights) == 3:
            sample_nodes = sample_nodes[None, :, :]
        denominator = backend.sum(weights, axis=-1)
        first = backend.sum(weights * sample_nodes, axis=-1)
        second = backend.sum(weights * sample_nodes * sample_nodes, axis=-1)
    elif isinstance(measure, ChebyshevMomentResult):
        matrixfunc._validate_trace_moments(measure)
        first, denominator = _moment_thermal_samples(
            measure, beta, 1, energy_shift, centered=True
        )
        second, _ = _moment_thermal_samples(
            measure,
            beta,
            2,
            energy_shift,
            centered=True,
            denominator=denominator,
        )
    else:
        raise TypeError("measure must be a LanczosMeasure or ChebyshevMomentResult.")
    first = backend.real(first)
    second = backend.real(second)
    denominator = backend.real(denominator)
    beta, _ = _check_query(beta, "beta")
    value, standard_error = _heat_capacity_statistics(
        denominator, first, second, beta, with_std=with_std
    )
    return _format_estimate(value, standard_error, with_std)


def thermal_expectation(
    operator: Any,
    observable: Any,
    beta: Tensor,
    *,
    probes: Tensor,
    method: Optional[Union[KrylovConfig, ChebyshevConfig, TaylorConfig]] = None,
    dimension: Optional[int] = None,
    probe_batch_size: Optional[int] = None,
    energy_shift: Optional[Tensor] = None,
    with_std: bool = False,
) -> Estimate:
    """
    Estimate ``Tr(exp(-beta H) O) / Tr(exp(-beta H))`` by typicality.

    ``observable`` and ``operator`` are Hermitian matrix-free operators, and
    ``probes`` has shape ``(num_probes, dimension)``. The result is scalar or
    rank one according to ``beta``. Shifted imaginary-time states omit their
    common exponential factor, so the ratio remains finite at large beta.

    :param operator: Hermitian Hamiltonian operator or matrix-vector-product callable.
    :type operator: Any
    :param observable: Observable operator with the same dimension as ``operator``.
    :type observable: Any
    :param beta: Non-negative inverse temperature, scalar or rank one.
    :type beta: Tensor
    :param probes: Explicit probe matrix with shape ``(num_probes, dimension)``.
    :type probes: Tensor
    :param method: Static Krylov, Chebyshev, or Taylor configuration.
    :type method: Union[KrylovConfig, ChebyshevConfig, TaylorConfig]
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :param probe_batch_size: Optional positive static probe block size.
    :type probe_batch_size: Optional[int]
    :param energy_shift: Optional scalar shift omitted from the intermediate state scale.
    :type energy_shift: Optional[Tensor]
    :param with_std: If true, return ``(expectation, standard_error)``;
        otherwise return only the expectation.
    :type with_std: bool, optional
    :return: Thermal expectation, optionally paired with its paired-probe
        standard error.
    :rtype: Estimate
    """
    probes = _require_probes(probes)
    if method is None:
        method = _DEFAULT_KRYLOV_CONFIG
    beta, scalar = _check_query(beta, "beta")
    _check_nonnegative(beta, "beta")
    hamiltonian, dimension = matrixfunc._prepare_operator(
        operator, probes=probes, dimension=dimension
    )
    observable_operator, _ = matrixfunc._prepare_operator(
        observable, dimension=dimension
    )

    def one(probe: Tensor) -> Tensor:
        state = matrixfunc._exponential_action(
            hamiltonian,
            probe,
            -0.5 * beta,
            method,
            dimension=dimension,
            energy_shift=energy_shift,
            restore_shift=False,
        )
        if _rank(state) == 1:
            observed = observable_operator @ state
            numerator = backend.real(backend.sum(backend.conj(state) * observed))
            denominator = backend.real(backend.sum(backend.conj(state) * state))
            return backend.stack([numerator, denominator])
        observed = backend.vmap(lambda item: observable_operator @ item)(state)
        numerator = backend.real(backend.sum(backend.conj(state) * observed, axis=-1))
        denominator = backend.real(backend.sum(backend.conj(state) * state, axis=-1))
        return backend.stack([numerator, denominator], axis=0)

    if probe_batch_size is None:
        packed = backend.vmap(one)(probes)
    else:
        if not isinstance(probe_batch_size, int) or probe_batch_size < 1:
            raise ValueError("probe_batch_size must be a positive integer or None.")
        packed = matrixfunc._chunked_vmap(one, probes, probe_batch_size)
    if scalar:
        numerator = packed[:, 0]
        denominator = packed[:, 1]
    else:
        numerator = backend.transpose(packed[:, 0, :])
        denominator = backend.transpose(packed[:, 1, :])
    value, standard_error = _ratio_statistics(numerator, denominator, with_std=with_std)
    return _format_estimate(value, standard_error, with_std)


def _resolvent_bilinear_krylov(
    operator: Any,
    right_vector: Tensor,
    z: Tensor,
    left_vector: Optional[Tensor],
    method: KrylovConfig,
    dimension: Optional[int],
) -> Tensor:
    projection = matrixfunc.lanczos_project(
        operator, right_vector, method, dimension=dimension
    )
    z = backend.cast(z, dtypestr)
    if left_vector is None:
        return matrixfunc.lanczos_resolvent(projection, z)
    result = matrixfunc.lanczos_apply(
        projection,
        lambda nodes: (
            1.0 / (z[..., None] - backend.cast(nodes, dtypestr))
            if _rank(z) == 1
            else 1.0 / (z - backend.cast(nodes, dtypestr))
        ),
    )
    left_vector = backend.cast(_as_tensor(left_vector), dtypestr)
    if _rank(result) == 1:
        return backend.sum(backend.conj(left_vector) * result)
    return backend.tensordot(backend.conj(left_vector), result, axes=([0], [1]))


def resolvent_bilinear(
    operator: Any,
    right_vector: Tensor,
    z: Tensor,
    *,
    left_vector: Optional[Tensor] = None,
    method: Optional[Union[KrylovConfig, ChebyshevConfig]] = None,
    dimension: Optional[int] = None,
) -> Tensor:
    """
    Evaluate ``<left|(z-H)^-1|right>`` without forming an inverse.

    ``right_vector`` and an optional ``left_vector`` are rank-one vectors;
    ``z`` is scalar or rank one. Krylov and Chebyshev methods assume a
    Hermitian operator and preserve the vectors' unnormalized amplitudes.

    :param operator: Hermitian square operator or matrix-vector-product callable.
    :type operator: Any
    :param right_vector: Right vector with shape ``(dimension,)``.
    :type right_vector: Tensor
    :param z: Scalar or rank-one complex resolvent query.
    :type z: Tensor
    :param left_vector: Optional left vector with the same shape as ``right_vector``.
    :type left_vector: Optional[Tensor]
    :param method: Krylov or Chebyshev configuration.
    :type method: Union[KrylovConfig, ChebyshevConfig]
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :return: Bilinear resolvent values with the query shape of ``z``.
    :rtype: Tensor
    """
    right_vector = backend.cast(_as_tensor(right_vector), dtypestr)
    z, _ = _check_query(z, "z")
    z = backend.cast(z, dtypestr)
    if method is None:
        method = _DEFAULT_KRYLOV_CONFIG
    if isinstance(method, KrylovConfig):
        return _resolvent_bilinear_krylov(
            operator, right_vector, z, left_vector, method, dimension
        )
    if isinstance(method, ChebyshevConfig):
        moments = matrixfunc.chebyshev_moments(
            operator,
            right_vector,
            method,
            left_vector=left_vector,
            dimension=dimension,
        )
        return matrixfunc.evaluate_chebyshev_moments(
            moments, matrixfunc.resolvent_coefficients(z, method)
        )
    raise TypeError("method must be a KrylovConfig or ChebyshevConfig.")


def zero_temperature_greens_function(
    operator: Any,
    frequencies: Tensor,
    *,
    ground_energy: Tensor,
    particle_right: Tensor,
    particle_left: Optional[Tensor] = None,
    hole_right: Optional[Tensor] = None,
    hole_left: Optional[Tensor] = None,
    broadening: Tensor,
    branch_sign: Literal[-1, 1] = 1,
    method: Optional[Union[KrylovConfig, ChebyshevConfig]] = None,
    dimension: Optional[int] = None,
) -> Tensor:
    """
    Evaluate the retarded zero-temperature particle/hole Green function.

    ``frequencies`` is scalar or rank one, ``broadening`` is strictly positive,
    and particle/hole insertion vectors must be supplied as matching pairs.
    The returned tensor has the same query shape as ``frequencies``.

    :param operator: Hermitian square operator or matrix-vector-product callable.
    :type operator: Any
    :param frequencies: Scalar or rank-one frequency query.
    :type frequencies: Tensor
    :param ground_energy: Ground-state energy reference.
    :type ground_energy: Tensor
    :param particle_right: Particle insertion vector.
    :type particle_right: Tensor
    :param particle_left: Optional particle bra vector.
    :type particle_left: Optional[Tensor]
    :param hole_right: Optional hole insertion vector; must be paired with ``hole_left``.
    :type hole_right: Optional[Tensor]
    :param hole_left: Optional hole bra vector; must be paired with ``hole_right``.
    :type hole_left: Optional[Tensor]
    :param broadening: Strictly positive retarded broadening.
    :type broadening: Tensor
    :param branch_sign: Sign multiplying the hole contribution.
    :type branch_sign: Literal[-1, 1]
    :param method: Krylov or Chebyshev configuration.
    :type method: Union[KrylovConfig, ChebyshevConfig]
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :return: Retarded Green-function values with the query shape of ``frequencies``.
    :rtype: Tensor
    """
    if branch_sign not in (-1.0, 1.0):
        raise ValueError("branch_sign must be +1.0 or -1.0.")
    if method is None:
        method = _DEFAULT_KRYLOV_CONFIG
    frequencies, _ = _check_query(frequencies, "frequencies")
    broadening = backend.cast(_as_tensor(broadening), rdtypestr)
    _check_positive(broadening, "broadening")
    ground_energy = backend.cast(_as_tensor(ground_energy), rdtypestr)
    particle_right = backend.cast(_as_tensor(particle_right), dtypestr)
    if (hole_right is None) != (hole_left is None):
        raise ValueError("hole_right and hole_left must be supplied together.")
    complex_ground = backend.cast(ground_energy, dtypestr)
    complex_broadening = backend.cast(1.0j, dtypestr) * backend.cast(
        broadening, dtypestr
    )
    z_particle = (
        backend.cast(frequencies, dtypestr) + complex_ground + complex_broadening
    )
    particle = resolvent_bilinear(
        operator,
        particle_right,
        z_particle,
        left_vector=particle_left,
        method=method,
        dimension=dimension,
    )
    if hole_right is None:
        return particle
    z_hole = -backend.cast(frequencies, dtypestr) + complex_ground - complex_broadening
    hole = resolvent_bilinear(
        operator,
        backend.cast(_as_tensor(hole_right), dtypestr),
        z_hole,
        left_vector=backend.cast(_as_tensor(hole_left), dtypestr),
        method=method,
        dimension=dimension,
    )
    return particle - backend.cast(branch_sign, dtypestr) * hole


def spectral_function(green_function: Tensor) -> Tensor:
    """
    Return the spectral density ``-Im(G^R)/pi`` elementwise.

    ``green_function`` may have any query shape, which is preserved in the
    returned real-valued tensor.

    :param green_function: Retarded Green-function values.
    :type green_function: Tensor
    :return: Spectral density ``-Im(green_function) / pi``.
    :rtype: Tensor
    """
    return -backend.imag(green_function) / math.pi


def time_correlation(
    operator: Any,
    left_vector: Tensor,
    right_vector: Tensor,
    times: Tensor,
    *,
    method: Optional[Union[KrylovConfig, ChebyshevConfig, TaylorConfig]] = None,
    energy_shift: Tensor = 0.0,
    dimension: Optional[int] = None,
) -> Tensor:
    """
    Evaluate ``<left|exp(-i(H-shift)t)|right>``.

    ``times`` is non-negative and scalar or rank one; unlike
    :func:`fft_spectrum`, this function does not require a uniform grid.
    ``energy_shift`` changes only the phase convention.

    :param operator: Hermitian square operator or matrix-vector-product callable.
    :type operator: Any
    :param left_vector: Left vector with shape ``(dimension,)``.
    :type left_vector: Tensor
    :param right_vector: Right vector with shape ``(dimension,)``.
    :type right_vector: Tensor
    :param times: Non-negative scalar or rank-one time query.
    :type times: Tensor
    :param method: Static Krylov, Chebyshev, or Taylor configuration.
    :type method: Union[KrylovConfig, ChebyshevConfig, TaylorConfig]
    :param energy_shift: Scalar energy shift used only for the returned phase.
    :type energy_shift: Tensor
    :param dimension: Optional operator dimension for a bare callable.
    :type dimension: Optional[int]
    :return: Correlation values with the query shape of ``times``.
    :rtype: Tensor
    """
    times, scalar = _check_query(times, "times")
    _check_nonnegative(times, "times")
    if method is None:
        method = _DEFAULT_KRYLOV_CONFIG
    right_vector = backend.cast(_as_tensor(right_vector), dtypestr)
    left_vector = backend.cast(_as_tensor(left_vector), dtypestr)
    action = matrixfunc.exponential_action(
        operator,
        right_vector,
        -1.0j * backend.cast(times, dtypestr),
        method,
        dimension=dimension,
    )
    shift = backend.cast(_as_tensor(energy_shift), dtypestr)
    phase = backend.cast(1.0j, dtypestr) * shift * backend.cast(times, dtypestr)
    if scalar:
        action = action * backend.exp(phase)
        return backend.sum(backend.conj(left_vector) * action)
    action = action * backend.exp(phase)[..., None]
    return backend.tensordot(backend.conj(left_vector), action, axes=([0], [1]))


def _window_values(length: int, window: str) -> Tensor:
    if window not in ("none", "hann", "hamming", "blackman"):
        raise ValueError("unsupported FFT window.")
    if window == "none" or length == 1:
        return backend.ones([length], dtype=rdtypestr)
    index = backend.cast(backend.arange(length), rdtypestr)
    phase = 2.0 * math.pi * index / (length - 1)
    if window == "hann":
        return 0.5 * (1.0 - backend.cos(phase))
    if window == "hamming":
        return 0.54 - 0.46 * backend.cos(phase)
    return 0.42 - 0.5 * backend.cos(phase) + 0.08 * backend.cos(2.0 * phase)


def fft_spectrum(
    correlation: Tensor,
    times: Tensor,
    *,
    window: Literal["none", "hann", "hamming", "blackman"] = "none",
    zero_padding: int = 0,
    convention: Literal["positive", "negative"] = "positive",
) -> Tuple[Tensor, Tensor]:
    """
    Fourier transform a uniformly sampled correlation with explicit signs.

    ``times`` must be rank one, start at zero, and be uniformly spaced.
    ``zero_padding`` is a static number of appended samples. The transform
    uses the backend FFT primitive, returns angular frequencies in fft-shifted
    ascending order, and evaluates ``dt * sum_j exp(+-i omega t_j) C_j``.

    :param correlation: Rank-one uniformly sampled correlation values.
    :type correlation: Tensor
    :param times: Rank-one time grid beginning at zero with uniform spacing.
    :type times: Tensor
    :param window: Window function, one of ``"none"``, ``"hann"``, ``"hamming"``, or ``"blackman"``.
    :type window: Literal["none", "hann", "hamming", "blackman"]
    :param zero_padding: Number of static zero samples appended before the FFT.
    :type zero_padding: int
    :param convention: Transform sign convention, either ``"positive"`` or ``"negative"``.
    :type convention: Literal["positive", "negative"]
    :return: Shifted angular frequencies and the corresponding spectrum.
    :rtype: Tuple[Tensor, Tensor]
    """
    correlation = _as_tensor(correlation)
    times = _as_tensor(times)
    if _rank(correlation) != 1 or _rank(times) != 1:
        raise ValueError("correlation and times must be rank-one tensors.")
    if backend.shape_tuple(correlation)[0] != backend.shape_tuple(times)[0]:
        raise ValueError("correlation and times must have the same length.")
    if not isinstance(zero_padding, int) or zero_padding < 0:
        raise ValueError("zero_padding must be a non-negative integer.")
    if convention not in ("positive", "negative"):
        raise ValueError("convention must be 'positive' or 'negative'.")
    length = backend.shape_tuple(times)[0]
    if length < 1:
        raise ValueError("times must not be empty.")
    concrete_times = None
    try:
        concrete_times = [float(x) for x in backend.numpy(times)]
    except (TypeError, ValueError, RuntimeError):
        pass
    if concrete_times is not None:
        if abs(concrete_times[0]) > 1.0e-12:
            raise ValueError("times must begin at zero.")
        if length > 1:
            spacing = concrete_times[1] - concrete_times[0]
            if spacing <= 0:
                raise ValueError("times must be strictly increasing.")
            for left, right in zip(concrete_times[1:-1], concrete_times[2:]):
                if abs((right - left) - spacing) > 1.0e-6 * max(1.0, abs(spacing)):
                    raise ValueError("times must be uniformly spaced.")
    if length > 1:
        dt = times[1] - times[0]
    else:
        dt = backend.convert_to_tensor(1.0, dtype=rdtypestr)
    total_length = length + zero_padding
    weighted = correlation * backend.cast(
        _window_values(length, window), backend.dtype(correlation)
    )
    if zero_padding:
        weighted = backend.concat(
            [weighted, backend.zeros([zero_padding], dtype=backend.dtype(weighted))],
            axis=0,
        )
    index = backend.cast(backend.arange(total_length), rdtypestr)
    wrapped = backend.where(
        index < (total_length + 1) // 2,
        index,
        index - total_length,
    )
    frequencies = 2.0 * math.pi * wrapped / (total_length * dt)
    if convention == "positive":
        spectrum = backend.conj(backend.fft(backend.conj(weighted)))
    else:
        spectrum = backend.fft(weighted)
    spectrum = backend.cast(dt, backend.dtype(spectrum)) * spectrum
    shift = (total_length + 1) // 2
    if shift:
        frequencies = backend.concat([frequencies[shift:], frequencies[:shift]], axis=0)
        spectrum = backend.concat([spectrum[shift:], spectrum[:shift]], axis=0)
    return frequencies, spectrum
