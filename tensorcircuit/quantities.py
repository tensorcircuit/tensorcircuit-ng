"""Quantum information quantities and state transformations."""

# pylint: disable=invalid-name

import math
import os
from functools import partial, reduce
from operator import matmul
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .cons import backend, contractor, dtypestr, idtypestr, rdtypestr
from .gates import Gate
from .quantum import QuOperator, _infer_num_sites, _resolve_cut_or_subsystem

Tensor = Any


def op2tensor(
    fn: Callable[..., Any], op_argnums: Union[int, Sequence[int]] = 0
) -> Callable[..., Any]:
    from .interfaces.tensortrans import args_to_tensor

    return args_to_tensor(fn, op_argnums, qop_to_tensor=True, cast_dtype=False)


# def op2tensor(
#     fn: Callable[..., Any], op_argnums: Union[int, Sequence[int]] = 0
# ) -> Callable[..., Any]:
#     if isinstance(op_argnums, int):
#         op_argnums = [op_argnums]

#     @wraps(fn)
#     def wrapper(*args: Any, **kwargs: Any) -> Any:
#         nargs = list(args)
#         for i in op_argnums:  # type: ignore
#             if isinstance(args[i], QuOperator):
#                 nargs[i] = args[i].copy().eval_matrix()
#         out = fn(*nargs, **kwargs)
#         return out

#     return wrapper


@op2tensor
def entropy(rho: Union[Tensor, QuOperator], eps: Optional[float] = None) -> Tensor:
    """
    Compute the entropy from the given density matrix ``rho``.

    :Example:

    .. code-block:: python

        @partial(tc.backend.jit, jit_compile=False, static_argnums=(1, 2))
        def entanglement1(param, n, nlayers):
            c = tc.Circuit(n)
            c = tc.templates.blocks.example_block(c, param, nlayers)
            w = c.wavefunction()
            rm = qu.reduced_density_matrix(w, int(n / 2))
            return qu.entropy(rm)

        @partial(tc.backend.jit, jit_compile=False, static_argnums=(1, 2))
        def entanglement2(param, n, nlayers):
            c = tc.Circuit(n)
            c = tc.templates.blocks.example_block(c, param, nlayers)
            w = c.get_quvector()
            rm = w.reduced_density([i for i in range(int(n / 2))])
            return qu.entropy(rm)

    >>> param = tc.backend.ones([6, 6])
    >>> tc.backend.trace(param)
    >>> entanglement1(param, 6, 3)
    1.3132654
    >>> entanglement2(param, 6, 3)
    1.3132653

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param eps: Epsilon, default is 1e-12.
    :type eps: float
    :return: Entropy on the given density matrix.
    :rtype: Tensor
    """
    eps_env = os.environ.get("TC_QUANTUM_ENTROPY_EPS")
    if eps is None and eps_env is None:
        eps = 1e-12
    elif eps is None and eps_env is not None:
        eps = 10 ** (-int(eps_env))
    rho = rho + eps * backend.cast(backend.eye(rho.shape[-1]), rho.dtype)  # type: ignore
    lbd = backend.real(backend.eigh(rho)[0])
    lbd = backend.relu(lbd)
    lbd /= backend.sum(lbd)
    # we need the matrix anyway for AD.
    entropy = -backend.sum(lbd * backend.log(lbd + eps))
    return backend.real(entropy)


@op2tensor
def anti_flatness(rho: Union[Tensor, QuOperator]) -> Tensor:
    r"""
    Compute the anti-flatness of a normalized density matrix.

    For the eigenvalues :math:`\lambda_i` of ``rho``, anti-flatness is

    .. math::

        \mathcal{F} = \operatorname{Tr}(\rho^3)
        - \left[\operatorname{Tr}(\rho^2)\right]^2.

    The input is expected to be a square, trace-one density matrix.  The
    expression is evaluated with matrix products rather than an eigendecomposition,
    so it remains differentiable and compatible with backend JIT transforms.
    Anti-flatness is zero for a rank-one spectrum and for a flat spectrum on
    its support; it is a spectral diagnostic, not by itself a complete
    non-stabilizerness measure.

    :param rho: The normalized density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :return: The anti-flatness of the density matrix.
    :rtype: Tensor
    """
    rho_squared = backend.matmul(rho, rho)
    purity = backend.real(backend.trace(rho_squared))
    third_moment = backend.real(backend.sum(rho_squared * backend.transpose(rho)))
    return third_moment - purity * purity


def trace_product(*o: Union[Tensor, QuOperator]) -> Tensor:
    """
    Compute the trace of several inputs ``o`` as tensor or ``QuOperator``.

    .. math ::

        \\operatorname{Tr}(\\prod_i O_i)

    :Example:

    >>> o = np.ones([2, 2])
    >>> h = np.eye(2)
    >>> qu.trace_product(o, h)
    2.0
    >>> oq = qu.QuOperator.from_tensor(o)
    >>> hq = qu.QuOperator.from_tensor(h)
    >>> qu.trace_product(oq, hq)
    array([[2.]])
    >>> qu.trace_product(oq, h)
    array([[2.]])
    >>> qu.trace_product(o, hq)
    array([[2.]])

    :return: The trace of several inputs.
    :rtype: Tensor
    """
    prod = reduce(matmul, o)
    if isinstance(prod, QuOperator):
        return prod.trace().eval_matrix()
    return backend.trace(prod)


@op2tensor
def entanglement_entropy(
    state: Tensor,
    cut: Union[int, List[int], Tuple[int, ...], None] = None,
    *,
    subsystem_to_keep: Optional[Sequence[int]] = None,
    subsystems_to_trace_out: Optional[Sequence[int]] = None,
    dim: Optional[int] = None,
) -> Tensor:
    """
    Compute the von Neumann entanglement entropy of ``state`` across the bipartition
    defined by ``cut``.

    The traced-out subsystem can be specified via the legacy ``cut`` argument
    (``int`` = ``list(range(cut))`` i.e. ``[0, cut)``; or an explicit site list)
    or, preferably, via exactly one of ``subsystem_to_keep`` /
    ``subsystems_to_trace_out``. See :func:`reduced_density_matrix` for the
    full resolution semantics; if both ``cut`` and a new argument are given,
    the new argument wins and ``cut`` is ignored (with a ``UserWarning``).

    :param state: wavefunction or density matrix of the full system
    :type state: Tensor
    :param cut: legacy trace-out specification; prefer the dual arguments below.
    :type cut: Union[int, List[int], Tuple[int, ...], None]
    :param subsystem_to_keep: sites to keep (all others are traced out).
        Mutually exclusive with ``subsystems_to_trace_out``.
    :type subsystem_to_keep: Optional[Sequence[int]]
    :param subsystems_to_trace_out: sites to trace out.
    :type subsystems_to_trace_out: Optional[Sequence[int]]
    :param dim: dimension of qudit system, defaults to 2
    :type dim: Optional[int]
    :return: the von Neumann entanglement entropy :math:`S = -\\mathrm{Tr}(\\rho \\log\\rho)`
    :rtype: Tensor
    """
    d = 2 if dim is None else dim
    if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        n = _infer_num_sites(state.shape[0], d)
    else:
        n = _infer_num_sites(int(backend.sizen(state)), d)
    traceout = _resolve_cut_or_subsystem(
        n, cut, subsystem_to_keep, subsystems_to_trace_out, name="entanglement_entropy"
    )
    rho = reduced_density_matrix(state, subsystems_to_trace_out=traceout, dim=dim)
    return entropy(rho)


@op2tensor
def entanglement_anti_flatness(
    state: Tensor,
    *,
    subsystem_to_keep: Optional[Sequence[int]] = None,
    subsystems_to_trace_out: Optional[Sequence[int]] = None,
    dim: Optional[int] = None,
) -> Tensor:
    r"""
    Compute the anti-flatness of a subsystem entanglement spectrum.

    The subsystem is specified via exactly one of ``subsystem_to_keep`` /
    ``subsystems_to_trace_out``.  The argument resolution follows the new
    subsystem specification used by :func:`entanglement_entropy`.

    :param state: Wavefunction or density matrix of the full system.
    :type state: Tensor
    :param subsystem_to_keep: Sites to keep; all others are traced out.
    :type subsystem_to_keep: Optional[Sequence[int]]
    :param subsystems_to_trace_out: Sites to trace out.
    :type subsystems_to_trace_out: Optional[Sequence[int]]
    :param dim: Local qudit dimension, defaulting to 2.
    :type dim: Optional[int]
    :return: The subsystem anti-flatness.
    :rtype: Tensor
    """
    d = 2 if dim is None else dim
    if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        n = _infer_num_sites(state.shape[0], d)
    else:
        n = _infer_num_sites(int(backend.sizen(state)), d)
    traceout = _resolve_cut_or_subsystem(
        n,
        None,
        subsystem_to_keep,
        subsystems_to_trace_out,
        name="entanglement_anti_flatness",
    )
    rho = reduced_density_matrix(state, subsystems_to_trace_out=traceout, dim=dim)
    return anti_flatness(rho)


def reduced_wavefunction(
    state: Tensor,
    cut: Union[List[int], Tuple[int, ...], None] = None,
    measure: Optional[List[int]] = None,
    dim: Optional[int] = None,
    *,
    subsystem_to_keep: Optional[Sequence[int]] = None,
    subsystems_to_trace_out: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    Compute the reduced wavefunction from the quantum state ``state``.
    The fixed measure result is guaranteed by users,
    otherwise final normalization may required in the return

    The reduced (measured) sites can be specified via the legacy ``cut``
    argument (an explicit site list) or, preferably, via exactly one of
    ``subsystem_to_keep`` / ``subsystems_to_trace_out``. See
    :func:`reduced_density_matrix` for the resolution semantics; if both
    ``cut`` and a new argument are given, the new argument wins and ``cut``
    is ignored (with a ``UserWarning``).

    :param state: wavefunction of the full system
    :type state: Tensor
    :param cut: the list of position for qubit to be reduced; prefer the dual
        arguments below.
    :type cut: Union[List[int], Tuple[int, ...], None]
    :param measure: the fixed results of given qubits in the same shape list as the reduced sites
    :type measure: List[int]
    :return: the (unnormalized) reduced wavefunction on the remaining sites
    :rtype: Tensor
    :param dim: dimension of qudit system
    :type dim: int
    :param subsystem_to_keep: sites to keep (all others are reduced).
        Mutually exclusive with ``subsystems_to_trace_out``.
    :type subsystem_to_keep: Optional[Sequence[int]]
    :param subsystems_to_trace_out: sites to reduce (measure out).
    :type subsystems_to_trace_out: Optional[Sequence[int]]
    """
    dim = 2 if dim is None else dim
    s = backend.reshaped(state, dim)
    n = len(backend.shape_tuple(s))
    traceout = _resolve_cut_or_subsystem(
        n, cut, subsystem_to_keep, subsystems_to_trace_out, name="reduced_wavefunction"
    )
    if measure is None:
        measure = [0 for _ in traceout]
    s_node = Gate(s)
    end_nodes = []
    for c, m in zip(traceout, measure):
        oh = backend.cast(
            backend.one_hot(backend.cast(backend.convert_to_tensor(m), "int32"), dim),
            dtypestr,
        )
        end_node = Gate(backend.convert_to_tensor(oh))
        end_nodes.append(end_node)
        s_node[c] ^ end_node[0]
    new_node = contractor(
        [s_node] + end_nodes,
        output_edge_order=[s_node[i] for i in range(n) if i not in traceout],
    )
    return backend.reshape(new_node.tensor, [-1])


def reduced_density_matrix(
    state: Union[Tensor, QuOperator],
    cut: Union[int, List[int], Tuple[int, ...], None] = None,
    p: Optional[Tensor] = None,
    normalize: bool = True,
    dim: Optional[int] = None,
    *,
    subsystem_to_keep: Optional[Sequence[int]] = None,
    subsystems_to_trace_out: Optional[Sequence[int]] = None,
) -> Union[Tensor, QuOperator]:
    r"""
    Compute the reduced density matrix from the quantum state ``state``.

    The subsystem can be specified in either of two equivalent ways:

    * ``cut`` (legacy): the indices to trace out. If ``cut`` is an ``int``,
      it indicates ``[0, cut)`` (i.e. ``list(range(cut))``) as the traced-out
      region; if it is a sequence, it is the explicit site list.
    * ``subsystem_to_keep`` / ``subsystems_to_trace_out`` (preferred): exactly
      one must be given; the other is inferred as the complement. These are
      mutually exclusive with each other.

    If both ``cut`` and one of the new arguments are given, the new argument
    takes precedence and ``cut`` is ignored (a ``UserWarning`` is emitted).

    :param state: The normalized quantum state in form of Tensor or QuOperator.
    :type state: Union[Tensor, QuOperator]
    :param cut: legacy trace-out specification (int = ``list(range(cut))``,
        i.e. ``[0, cut)``; or an explicit site list). Prefer
        ``subsystem_to_keep``/``subsystems_to_trace_out``.
    :type cut: Union[int, List[int], Tuple[int, ...], None]
    :param p: Optional diagonal weights on the traced-out subsystem, applied
        before the partial trace. Not supported for ``QuOperator`` inputs.
    :type p: Optional[Tensor]
    :return: The reduced density matrix.
    :rtype: Union[Tensor, QuOperator]
    :param normalize: If ``p`` is given, whether to renormalize the weighted
        reduced state to trace one. With ``p=None`` and a normalized input,
        this has no effect.
    :type normalize: bool
    :param dim: dimension of qudit system
    :type dim: int
    :param subsystem_to_keep: sites to keep (all others are traced out).
        Mutually exclusive with ``subsystems_to_trace_out``.
    :type subsystem_to_keep: Optional[Sequence[int]]
    :param subsystems_to_trace_out: sites to trace out.
    :type subsystems_to_trace_out: Optional[Sequence[int]]
    """
    dim = 2 if dim is None else dim
    if isinstance(state, QuOperator):
        if p is not None:
            raise NotImplementedError(
                "p arguments is not supported when state is a `QuOperator`"
            )
        # A square QuOperator (out and in edges both present) can be partially
        # traced directly. A pure-state vector (QuVector / QuAdjointVector) has
        # no in-edges, so partial_trace would IndexError; route it through
        # reduced_density, which projects first (|psi><psi|) then traces.
        n = len(state.out_edges) if state.is_vector() else len(state.in_edges)
        traceout = _resolve_cut_or_subsystem(
            n,
            cut,
            subsystem_to_keep,
            subsystems_to_trace_out,
            name="reduced_density_matrix",
        )
        if state.is_vector() or state.is_adjoint_vector():
            return state.reduced_density(subsystems_to_trace_out=traceout)  # type: ignore[attr-defined]
        return state.partial_trace(traceout)
    if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        # density operator
        freedom = _infer_num_sites(state.shape[0], dim)
        traceout = _resolve_cut_or_subsystem(
            freedom,
            cut,
            subsystem_to_keep,
            subsystems_to_trace_out,
            name="reduced_density_matrix",
        )
        left = traceout + [i for i in range(freedom) if i not in traceout]
        right = [i + freedom for i in left]

        rho = backend.reshape(state, [dim] * (2 * freedom))
        rho = backend.transpose(rho, perm=left + right)
        rho = backend.reshape(
            rho,
            [
                dim ** len(traceout),
                dim ** (freedom - len(traceout)),
                dim ** len(traceout),
                dim ** (freedom - len(traceout)),
            ],
        )
        if p is None:
            # correct but per-axis loop with tf.einsum fails on high-dim tensors
            rho = backend.trace(rho, axis1=0, axis2=2)
        else:
            p = backend.reshape(p, [-1])
            rho = backend.einsum("a,aiaj->ij", p, rho)
        rho = backend.reshape(
            rho, [dim ** (freedom - len(traceout)), dim ** (freedom - len(traceout))]
        )
        if normalize:
            rho /= backend.trace(rho)

    else:
        w = state / backend.norm(state)
        size = int(backend.sizen(state))
        freedom = _infer_num_sites(size, dim)
        traceout = _resolve_cut_or_subsystem(
            freedom,
            cut,
            subsystem_to_keep,
            subsystems_to_trace_out,
            name="reduced_density_matrix",
        )
        perm = [i for i in range(freedom) if i not in traceout]
        perm = perm + traceout
        w = backend.reshape(w, [dim] * freedom)
        w = backend.transpose(w, perm=perm)
        w = backend.reshape(w, [-1, dim ** len(traceout)])
        if p is None:
            rho = w @ backend.adjoint(w)
        else:
            rho = w @ backend.diagflat(p) @ backend.adjoint(w)
            if normalize:
                rho /= backend.trace(rho)

    return rho


def free_energy(
    rho: Union[Tensor, QuOperator],
    h: Union[Tensor, QuOperator],
    beta: float = 1,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute the free energy of the given density matrix.

    :Example:

    >>> rho = np.array([[1.0, 0], [0, 0]])
    >>> h = np.array([[-1.0, 0], [0, 1]])
    >>> qu.free_energy(rho, h, 0.5)
    -0.9999999999979998
    >>> hq = qu.QuOperator.from_tensor(h)
    >>> qu.free_energy(rho, hq, 0.5)
    array([[-1.]])

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param h: Hamiltonian operator in form of Tensor or QuOperator.
    :type h: Union[Tensor, QuOperator]
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :param eps: Epsilon, default is 1e-12.
    :type eps: float, optional

    :return: The free energy of the given density matrix with the Hamiltonian operator.
    :rtype: Tensor
    """
    energy = backend.real(trace_product(rho, h))
    s = entropy(rho, eps)
    return backend.real(energy - s / beta)


def renyi_entropy(rho: Union[Tensor, QuOperator], k: int = 2) -> Tensor:
    """
    Compute the Renyi entropy of order :math:`k` by given density matrix.

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param k: The order of Renyi entropy, default is 2.
    :type k: int, optional
    :return: The :math:`k` th order of Renyi entropy.
    :rtype: Tensor
    """
    s = 1 / (1 - k) * backend.real(backend.log(trace_product(*[rho] * k)))
    return s


def _fwht(vector: Tensor) -> Tensor:
    """Apply the unnormalized fast Walsh-Hadamard transform to a vector."""
    size = int(backend.sizen(vector))
    if size == 0 or size & (size - 1):
        raise ValueError("The FWHT input size must be a positive power of two.")

    transformed = vector
    block_size = 1
    while block_size < size:
        blocks = backend.reshape(transformed, [-1, 2 * block_size])
        left = blocks[:, :block_size]
        right = blocks[:, block_size:]
        transformed = backend.reshape(
            backend.stack([left + right, left - right], axis=1), [-1]
        )
        block_size *= 2
    return transformed


def stabilizer_renyi_entropy(
    state: Tensor,
    alpha: int = 2,
    status: Optional[Tensor] = None,
    with_std: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""
    Compute the pure-state qubit stabilizer Rényi entropy with FWHT.

    This implementation currently supports qubit states only.  For an
    ``N``-qubit pure state ``|psi>`` and ``d=2**N``, the Pauli
    correlators are grouped by ``x`` using

    ``f_x[b] = conjugate(psi[b]) * psi[b xor x]``.

    An unnormalized fast Walsh-Hadamard transform of ``f_x`` returns all
    correlators in the ``x`` family simultaneously.  If ``status`` is not
    provided, all ``x`` families are evaluated exactly.  If ``status`` is a
    rank-one tensor with values in ``[0, 1)``, ``x=0`` is evaluated exactly
    and each status value samples one nonzero ``x`` family uniformly.  The
    latter mode gives an unbiased estimator of the moment before the final
    logarithm and is convenient for JIT-compiled functions because its
    randomness is supplied externally.  If ``with_std`` is true, the sampled
    family moments are also used to estimate the standard error of the final
    entropy.  This adds only scalar accumulators to the sampling scan and does
    not store all sampled family moments.  More precisely, if ``s_m`` is the
    sample standard deviation of the sampled family moments, the returned
    error bar is the first-order estimate
    ``(d - 1) * s_m / (sqrt(K) * abs(1 - alpha) * log(2) * S)`` for the
    entropy estimator, where ``K`` is the number of nonzero ``x`` samples and
    ``S`` is the estimated moment.  Thus it is the estimated standard
    deviation of the final entropy across repeated Monte Carlo runs, rather
    than the raw standard deviation of the family moments.
    A fixed Clifford unitary may be inserted into the state-preparation
    circuit before obtaining ``state``.  Since Clifford unitaries permute the
    Pauli operators, this does not change the exact entropy, but it can change
    the sampling variance.  For structured states, applying Hadamard gates to
    selected computational-basis qubits is a simple user-controlled
    preconditioning option.  The choice is state-dependent and is not needed
    for the exact mode.

    :param state: Pure-state qubit wavefunction with shape ``(2**N,)``.
    :type state: Tensor
    :param alpha: Integer Rényi order, currently restricted to ``alpha >= 2``.
    :type alpha: int, optional
    :param status: Optional external uniform random tensor with shape
        ``(number_of_samples,)`` and values in ``[0, 1)``.
    :type status: Optional[Tensor]
    :param with_std: If true, return a tuple containing the entropy and an
        estimated standard error.  The standard error is zero in exact mode;
        sampled mode requires at least two ``status`` values.
    :type with_std: bool, optional
    :return: The stabilizer Rényi entropy ``M_alpha``.
        If ``with_std`` is true, return ``(entropy, standard_error)``.
    :rtype: Union[Tensor, Tuple[Tensor, Tensor]]
    """
    if not isinstance(alpha, int) or alpha < 2:
        raise ValueError("alpha must be an integer greater than or equal to 2.")

    state = backend.cast(backend.convert_to_tensor(state), dtypestr)
    if len(state.shape) != 1:
        raise ValueError("state must be a rank-one pure-state wavefunction.")

    dimension = int(backend.sizen(state))
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError(
            "The state size must be a power of two with at least one qubit."
        )
    nqubits = int(math.log2(dimension))

    state = state / backend.norm(state)
    basis = backend.arange(dimension)

    def family_moment(x: Tensor) -> Tensor:
        shifted_state = backend.gather1d(state, backend.bitwise_xor(basis, x))
        family = _fwht(backend.conj(state) * shifted_state)
        magnitude = backend.abs(family)
        moment_power = magnitude * magnitude
        for _ in range(1, alpha):
            moment_power = moment_power * magnitude * magnitude
        return backend.sum(moment_power) / dimension**alpha

    zero = backend.convert_to_tensor(0.0, dtype=rdtypestr)
    if status is None:
        x_values = backend.arange(dimension)
        moment = backend.scan(lambda carry, x: carry + family_moment(x), x_values, zero)
        entropy = backend.real(
            backend.log(moment) / math.log(2.0) / (1 - alpha) - nqubits
        )
        if with_std:
            return entropy, zero
        return entropy
    else:
        status = backend.cast(backend.convert_to_tensor(status), rdtypestr)
        if len(status.shape) != 1 or int(status.shape[0]) == 0:
            raise ValueError("status must be a non-empty rank-one tensor.")

        sample_count = int(status.shape[0])
        if with_std and sample_count < 2:
            raise ValueError(
                "status must contain at least two samples when with_std is true."
            )
        sampled_x = backend.cast(backend.floor(status * (dimension - 1)), idtypestr)
        sampled_x = backend.clip(sampled_x, 0, dimension - 2) + 1
        if with_std:

            def accumulate_sample_stats(
                carry: Tuple[Tensor, Tensor], x: Tensor
            ) -> Tuple[Tensor, Tensor]:
                sample = family_moment(x)
                return carry[0] + sample, carry[1] + sample * sample

            sampled_moment, sampled_moment_square = backend.scan(
                accumulate_sample_stats, sampled_x, (zero, zero)
            )
            sample_mean = sampled_moment / sample_count
            sample_variance = backend.relu(
                (sampled_moment_square - sampled_moment * sample_mean)
                / (sample_count - 1)
            )
        else:
            sampled_moment = backend.scan(
                lambda carry, x: carry + family_moment(x), sampled_x, zero
            )
        moment = family_moment(backend.convert_to_tensor(0, dtype=idtypestr))
        moment += (dimension - 1) * sampled_moment / sample_count
        entropy = backend.real(
            backend.log(moment) / math.log(2.0) / (1 - alpha) - nqubits
        )
        if with_std:
            moment_std = (dimension - 1) * backend.sqrt(sample_variance / sample_count)
            entropy_std = moment_std / (abs(1 - alpha) * math.log(2.0) * moment)
            return entropy, backend.real(entropy_std)
        return entropy


def renyi_free_energy(
    rho: Union[Tensor, QuOperator],
    h: Union[Tensor, QuOperator],
    beta: float = 1,
    k: int = 2,
) -> Tensor:
    """
    Compute the Renyi free energy of the corresponding density matrix and Hamiltonian.

    :Example:

    >>> rho = np.array([[1.0, 0], [0, 0]])
    >>> h = np.array([[-1.0, 0], [0, 1]])
    >>> qu.renyi_free_energy(rho, h, 0.5)
    -1.0
    >>> qu.free_energy(rho, h, 0.5)
    -0.9999999999979998

    :param rho: The density matrix in form of Tensor or QuOperator.
    :type rho: Union[Tensor, QuOperator]
    :param h: Hamiltonian operator in form of Tensor or QuOperator.
    :type h: Union[Tensor, QuOperator]
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :param k: The order of Renyi entropy, default is 2.
    :type k: int, optional
    :return: The :math:`k` th order of Renyi entropy.
    :rtype: Tensor
    """
    energy = backend.real(trace_product(rho, h))
    s = renyi_entropy(rho, k)
    return backend.real(energy - s / beta)


def taylorlnm(x: Tensor, k: int) -> Tensor:
    """
    Taylor expansion of :math:`ln(x+1)`.

    :param x: The density matrix in form of Tensor.
    :type x: Tensor
    :param k: The :math:`k` th order, default is 2.
    :type k: int, optional
    :return: The :math:`k` th order of Taylor expansion of :math:`ln(x+1)`.
    :rtype: Tensor
    """
    dtype = x.dtype
    s = x.shape[-1]
    eye = backend.eye(s, dtype=dtype)
    y = 1 / k * (-1) ** (k + 1) * eye
    for i in reversed(range(k)):
        y = y @ x
        if i > 0:
            y += 1 / (i) * (-1) ** (i + 1) * eye
    return y


def truncated_free_energy(
    rho: Tensor, h: Tensor, beta: float = 1, k: int = 2
) -> Tensor:
    """
    Compute the truncated free energy from the given density matrix ``rho``.

    :param rho: The density matrix in form of Tensor.
    :type rho: Tensor
    :param h: Hamiltonian operator in form of Tensor.
    :type h: Tensor
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :param k: The :math:`k` th order, defaults to 2
    :type k: int, optional
    :return: The :math:`k` th order of the truncated free energy.
    :rtype: Tensor
    """
    dtype = rho.dtype
    s = rho.shape[-1]
    tyexpand = rho @ taylorlnm(rho - backend.eye(s, dtype=dtype), k - 1)
    renyi = -backend.real(backend.trace(tyexpand))
    energy = backend.real(trace_product(rho, h))
    return energy - renyi / beta


@op2tensor
def partial_transpose(
    rho: Tensor, transposed_sites: List[int], dim: Optional[int] = None
) -> Tensor:
    """
    Compute the partial transpose of a density matrix on the given sites.

    :param rho: density matrix
    :type rho: Tensor
    :param transposed_sites: sites int list to be transposed
    :type transposed_sites: List[int]
    :param dim: dimension of qudit system
    :type dim: int
    :return: the partially transposed density matrix
    :rtype: Tensor
    """
    dim = 2 if dim is None else dim
    rho = backend.reshaped(rho, dim)
    rho_node = Gate(rho)
    n = len(rho.shape) // 2
    left_edges = []
    right_edges = []
    for i in range(n):
        if i not in transposed_sites:
            left_edges.append(rho_node[i])
            right_edges.append(rho_node[i + n])
        else:
            left_edges.append(rho_node[i + n])
            right_edges.append(rho_node[i])
    rhot_op = QuOperator(out_edges=left_edges, in_edges=right_edges)
    rhot = rhot_op.eval_matrix()
    return rhot


@op2tensor
def entanglement_negativity(
    rho: Tensor, transposed_sites: List[int], dim: Optional[int] = None
) -> Tensor:
    """
    Compute the entanglement negativity of ``rho`` across the bipartition
    defined by ``transposed_sites``.

    :param rho: density matrix of the bipartite system
    :type rho: Tensor
    :param transposed_sites: sites to transpose when forming the partial transpose
    :type transposed_sites: List[int]
    :param dim: dimension of qudit system
    :type dim: int
    :return: the entanglement negativity :math:`(\\lVert\\rho^{T_A}\\rVert_1 - 1)/2`
    :rtype: Tensor
    """
    rhot = partial_transpose(rho, transposed_sites, dim=dim)
    es = backend.eigvalsh(rhot)
    rhot_m = backend.sum(backend.abs(es))
    return (rhot_m - 1.0) / 2.0


@op2tensor
def log_negativity(
    rho: Tensor, transposed_sites: List[int], base: str = "e", dim: Optional[int] = None
) -> Tensor:
    """
    Compute the logarithmic negativity of ``rho``, i.e. the log of
    :math:`\\lVert\\rho^{T_A}\\rVert_1`.

    :param rho: density matrix of the bipartite system
    :type rho: Tensor
    :param transposed_sites: sites to transpose when forming the partial transpose
    :type transposed_sites: List[int]
    :param base: whether use 2 based log or e based log, defaults to "e"
    :type base: str, optional
    :param dim: dimension of qudit system
    :type dim: int
    :return: the logarithmic negativity
    :rtype: Tensor
    """
    dim = 2 if dim is None else dim
    rhot = partial_transpose(rho, transposed_sites, dim)
    es = backend.eigvalsh(rhot)
    rhot_m = backend.sum(backend.abs(es))
    een = backend.log(rhot_m)
    if base in ["2", 2]:
        return een / backend.cast(backend.log(2.0), rdtypestr)
    return een


@partial(op2tensor, op_argnums=(0, 1))
def trace_distance(rho: Tensor, rho0: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute the trace distance between two density matrix ``rho`` and ``rho0``.

    :param rho: The density matrix in form of Tensor.
    :type rho: Tensor
    :param rho0: The density matrix in form of Tensor.
    :type rho0: Tensor
    :param eps: Epsilon, defaults to 1e-12
    :type eps: float, optional
    :return: The trace distance between two density matrix ``rho`` and ``rho2``.
    :rtype: Tensor
    """
    d2 = rho - rho0
    d2 = backend.adjoint(d2) @ d2
    lbds = backend.real(backend.eigh(d2)[0])
    lbds = backend.relu(lbds)
    return 0.5 * backend.sum(backend.sqrt(lbds + eps))


@partial(op2tensor, op_argnums=(0, 1))
def fidelity(rho: Tensor, rho0: Tensor) -> Tensor:
    """
    Return the squared fidelity scalar between two states rho and rho0.

    .. math::

        \\left( \\operatorname{Tr}\\left[\\sqrt{\\sqrt{rho} rho_0 \\sqrt{rho}}\\right] \\right)^2

    Note
    ----
    This returns the squared Uhlmann fidelity ``F**2``, not ``F`` itself.
    For the unsquared fidelity, take ``backend.sqrt`` of the result.
    The implementation uses the equivalent squared trace norm of
    ``sqrt(rho) @ sqrt(rho0)``, with PSD matrix square roots. At rank boundaries,
    floating-point perturbations can cause errors of order the square root of
    machine precision. Derivatives along rank-changing paths need not be finite.

    :param rho: The density matrix in form of Tensor.
    :type rho: Tensor
    :param rho0: The density matrix in form of Tensor.
    :type rho0: Tensor
    :return: The squared fidelity scalar between ``rho`` and ``rho0``.
    :rtype: Tensor
    """
    product = backend.sqrtmh(rho) @ backend.sqrtmh(rho0)
    singular_values = backend.real(backend.svd(product)[1])
    return backend.sum(singular_values) ** 2


@op2tensor
def gibbs_state(h: Tensor, beta: float = 1) -> Tensor:
    """
    Compute the Gibbs state of the given Hamiltonian operator ``h``.

    :param h: Hamiltonian operator in form of Tensor.
    :type h: Tensor
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :return: The Gibbs state of ``h`` with the given ``beta``.
    :rtype: Tensor
    """
    rho = backend.expm(-beta * h)
    rho /= backend.trace(rho)
    return rho


@op2tensor
def double_state(h: Tensor, beta: float = 1) -> Tensor:
    """
    Compute the double state of the given Hamiltonian operator ``h``.

    :param h: Hamiltonian operator in form of Tensor.
    :type h: Tensor
    :param beta: Constant for the optimization, default is 1.
    :type beta: float, optional
    :return: The double state of ``h`` with the given ``beta``.
    :rtype: Tensor
    """
    rho = backend.expm(-beta / 2 * h)
    state = backend.reshape(rho, [-1])
    norm = backend.norm(state)
    return state / norm


@op2tensor
def mutual_information(
    s: Tensor,
    cut: Union[int, List[int], Tuple[int, ...], None] = None,
    dim: Optional[int] = None,
    *,
    subsystem_to_keep: Optional[Sequence[int]] = None,
    subsystems_to_trace_out: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    Mutual information between the two sides of the bipartition described by ``cut``.

    The traced-out subsystem can be specified via the legacy ``cut`` argument
    (``int`` = ``list(range(cut))`` i.e. ``[0, cut)``; or an explicit site list)
    or, preferably, via exactly one of ``subsystem_to_keep`` /
    ``subsystems_to_trace_out``. See :func:`reduced_density_matrix` for the
    full resolution semantics; if both ``cut`` and a new argument are given,
    the new argument wins and ``cut`` is ignored (with a ``UserWarning``).

    :param s: The density matrix in form of Tensor.
    :type s: Tensor
    :param cut: legacy trace-out specification; prefer the dual arguments below.
    :type cut: Union[int, List[int], Tuple[int, ...], None]
    :param dim: dimension of qudit system, defaults to 2.
    :type dim: Optional[int]
    :param subsystem_to_keep: sites to keep (all others form the other subsystem).
        Mutually exclusive with ``subsystems_to_trace_out``.
    :type subsystem_to_keep: Optional[Sequence[int]]
    :param subsystems_to_trace_out: sites to trace out (defines subsystem A).
    :type subsystems_to_trace_out: Optional[Sequence[int]]
    :return: The mutual information between AB subsystem described by ``cut``.
    :rtype: Tensor
    """
    dim = 2 if dim is None else dim
    if len(s.shape) == 2 and s.shape[0] == s.shape[1]:
        # mixed state
        n = _infer_num_sites(s.shape[0], dim=dim)
        traceout = _resolve_cut_or_subsystem(
            n,
            cut,
            subsystem_to_keep,
            subsystems_to_trace_out,
            name="mutual_information",
        )
        hab = entropy(s)

        # subsystem a
        rhoa = reduced_density_matrix(s, subsystems_to_trace_out=traceout, dim=dim)
        ha = entropy(rhoa)

        # need subsystem b as well
        other = tuple(i for i in range(n) if i not in traceout)
        rhob = reduced_density_matrix(s, subsystems_to_trace_out=other, dim=dim)  # type: ignore
        hb = entropy(rhob)

    # pure system
    else:
        n = _infer_num_sites(int(backend.sizen(s)), dim=dim)
        traceout = _resolve_cut_or_subsystem(
            n,
            cut,
            subsystem_to_keep,
            subsystems_to_trace_out,
            name="mutual_information",
        )
        hab = 0.0
        rhoa = reduced_density_matrix(s, subsystems_to_trace_out=traceout, dim=dim)
        ha = hb = entropy(rhoa)

    return ha + hb - hab
