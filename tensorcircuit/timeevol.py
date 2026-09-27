"""
Analog time evolution engines
"""

from typing import Any, Tuple, Optional, Callable, Sequence, Dict
from functools import partial
import math
import warnings

import numpy as np

from .cons import backend, dtypestr, rdtypestr, contractor
from .gates import Gate
from . import matrixfunc
from .quantum import aslinearoperator
from .utils import arg_alias

Tensor = Any
Circuit = Any


# Double-precision truncation bounds from Al-Mohy and Higham (2011), table 3.1.
# They are used outside JIT by ``estimate_expm_multiply_parameters``.
_EXPM_MULTIPLY_THETA = {
    1: 2.29e-16,
    2: 2.58e-8,
    3: 1.39e-5,
    4: 3.40e-4,
    5: 2.40e-3,
    6: 9.07e-3,
    7: 2.38e-2,
    8: 5.00e-2,
    9: 8.96e-2,
    10: 1.44e-1,
    11: 2.14e-1,
    12: 3.00e-1,
    13: 4.00e-1,
    14: 5.14e-1,
    15: 6.41e-1,
    16: 7.81e-1,
    17: 9.31e-1,
    18: 1.09,
    19: 1.26,
    20: 1.44,
    21: 1.62,
    22: 1.82,
    23: 2.01,
    24: 2.22,
    25: 2.43,
    26: 2.64,
    27: 2.86,
    28: 3.08,
    29: 3.31,
    30: 3.54,
    35: 4.7,
    40: 6.0,
    45: 7.2,
    50: 8.5,
    55: 9.9,
}


def _legacy_lanczos_projection(
    hamiltonian: Any, initial_vector: Tensor, subspace_dimension: int
) -> Tuple[Tensor, Tensor]:
    if subspace_dimension < 1:
        raise ValueError("subspace_dimension must be positive.")
    projection = matrixfunc.lanczos_project(
        hamiltonian,
        initial_vector,
        matrixfunc.KrylovConfig(max_dim=subspace_dimension),
    )
    recurrence = projection.recurrence
    diagonal = backend.cast(recurrence.diagonal, dtypestr)
    if subspace_dimension == 1:
        return projection.basis, backend.diagflat(diagonal)
    connected = recurrence.active[:-1] & recurrence.active[1:]
    off_diagonal = backend.cast(recurrence.off_diagonal, dtypestr)
    off_diagonal = off_diagonal * backend.cast(connected, dtypestr)
    upper = backend.diagflat(off_diagonal, k=1)
    return projection.basis, backend.diagflat(diagonal) + upper + backend.adjoint(  # type: ignore[no-any-return]
        upper
    )


def lanczos_iteration_scan(
    hamiltonian: Any, initial_vector: Tensor, subspace_dimension: int
) -> Tuple[Tensor, Tensor]:
    """
    Build a Lanczos basis and projected Hamiltonian.

    This compatibility entry point now delegates to the shared fixed-shape
    matrix-function Lanczos kernel. Use :func:`tensorcircuit.matrixfunc.lanczos_project`
    for new code.

    :param hamiltonian: Hermitian matrix, linear operator, or matrix-vector-product callable.
    :type hamiltonian: Any
    :param initial_vector: Nonzero seed vector with shape ``(dimension,)``.
    :type initial_vector: Tensor
    :param subspace_dimension: Maximum Krylov basis size.
    :type subspace_dimension: int
    :return: Basis and fixed-size projected tridiagonal matrix. Zero padding may
        remain after exact Lanczos breakdown; use
        :func:`tensorcircuit.matrixfunc.lanczos_project` when the active mask is
        required.
    :rtype: Tuple[Tensor, Tensor]
    """
    return _legacy_lanczos_projection(hamiltonian, initial_vector, subspace_dimension)


def lanczos_iteration(
    hamiltonian: Any, initial_vector: Tensor, subspace_dimension: int
) -> Tuple[Tensor, Tensor]:
    """
    Build a Lanczos basis and projected Hamiltonian.

    This is the historical non-scan name retained as a thin compatibility
    wrapper around the shared scan-based kernel. Use
    :func:`tensorcircuit.matrixfunc.lanczos_project` for new code.

    :param hamiltonian: Hermitian matrix, linear operator, or matrix-vector-product callable.
    :type hamiltonian: Any
    :param initial_vector: Nonzero seed vector with shape ``(dimension,)``.
    :type initial_vector: Tensor
    :param subspace_dimension: Maximum Krylov basis size.
    :type subspace_dimension: int
    :return: Basis and fixed-size projected tridiagonal matrix. Zero padding may
        remain after exact Lanczos breakdown; use
        :func:`tensorcircuit.matrixfunc.lanczos_project` when the active mask is
        required.
    :rtype: Tuple[Tensor, Tensor]
    """
    return _legacy_lanczos_projection(hamiltonian, initial_vector, subspace_dimension)


def krylov_evol(
    hamiltonian: Any,
    initial_state: Tensor,
    times: Tensor,
    subspace_dimension: Optional[int] = None,
    callback: Optional[Callable[[Any], Any]] = None,
    scan_impl: Optional[bool] = None,
    *,
    config: Optional[matrixfunc.KrylovConfig] = None,
) -> Any:
    """
    Perform quantum state time evolution using Krylov subspace method.

    :param hamiltonian: Hermitian sparse matrix, dense matrix, LinearOperator, or
        MVP callable implementing ``H @ state``.
    :type hamiltonian: Any
    :param initial_state: Initial quantum state
    :type initial_state: Tensor
    :param times: List of time points. The propagator is ``exp(-1j * t * H)``.
    :type times: Tensor
    :param subspace_dimension: Krylov subspace dimension
    :type subspace_dimension: int
    :param callback: Optional callback function applied to quantum state at
                  each evolution time point, return some observables
    :type callback: Optional[Callable[[Any], Any]], optional
    :param scan_impl: Deprecated compatibility flag. It no longer selects an
        implementation; every call uses the fixed-shape scan kernel.
    :type scan_impl: bool, optional
    :param config: Optional shared :class:`~tensorcircuit.matrixfunc.KrylovConfig`.
        When supplied, ``subspace_dimension`` and ``scan_impl`` must be omitted;
        its fixed ``max_dim`` controls the JIT-compatible matrix-function path.
    :type config: Optional[matrixfunc.KrylovConfig]
    :return: List of evolved quantum states, or list of callback function results
        (if callback provided)
    :rtype: Any
    """
    if config is None:
        if subspace_dimension is None:
            raise ValueError(
                "subspace_dimension must be provided when config is absent."
            )
        if subspace_dimension < 1:
            raise ValueError("subspace_dimension must be positive.")
        config = matrixfunc.KrylovConfig(max_dim=subspace_dimension)
    else:
        if subspace_dimension is not None or scan_impl is not None:
            raise ValueError(
                "config cannot be combined with subspace_dimension or scan_impl."
            )
        if not isinstance(config, matrixfunc.KrylovConfig):
            raise TypeError("config must be a KrylovConfig.")

    if scan_impl is not None:
        warnings.warn(
            "scan_impl is deprecated and ignored; the shared scan kernel is used.",
            DeprecationWarning,
            stacklevel=2,
        )
    del scan_impl
    initial_state = backend.cast(initial_state, dtypestr)
    times = backend.convert_to_tensor(times, dtype=dtypestr)
    states = matrixfunc.exponential_action(
        hamiltonian, initial_state, -1.0j * times, config
    )
    if backend.shape_tuple(times) == ():
        states = backend.reshape(states, [1, backend.shape_tuple(initial_state)[0]])
    if callback is not None:
        return backend.stack([callback(state) for state in states])
    return states


def estimate_expm_multiply_parameters(
    t_max: float, norm_bound: float
) -> Tuple[int, int]:
    """
    Choose a static Taylor degree and scaling count for ``expm_multiply_evol``.

    The returned pair is selected from the Al-Mohy--Higham truncation bounds
    using ``norm_bound >= ||H - trace(H) I / n||_1`` and is intended to be
    computed outside JIT. It is conservative when ``norm_bound`` is an upper
    bound rather than an exact norm. Unlike SciPy, it does not run dynamic
    power-norm estimators; this keeps schedule selection independent of an MVP
    implementation. The tabulated bounds target double-precision truncation,
    so complex64 workloads should be validated at their required tolerance.

    :param t_max: Largest absolute evolution time to be used by the compiled kernel.
    :type t_max: float
    :param norm_bound: Upper bound on the 1-norm of the trace-shifted Hamiltonian.
    :type norm_bound: float
    :return: Static Taylor degree ``m`` and scaling count ``s``.
    :rtype: Tuple[int, int]
    """
    t_max = float(t_max)
    norm_bound = float(norm_bound)
    if not math.isfinite(t_max) or t_max < 0:
        raise ValueError("t_max must be a finite non-negative number.")
    if not math.isfinite(norm_bound) or norm_bound < 0:
        raise ValueError("norm_bound must be a finite non-negative number.")

    scaled_norm = t_max * norm_bound
    if scaled_norm == 0:
        return 0, 1

    candidates = []
    for m, theta in _EXPM_MULTIPLY_THETA.items():
        s = max(1, int(math.ceil(scaled_norm / theta)))
        candidates.append((m * s, m, s))
    _, m, s = min(candidates)
    return m, s


def expm_multiply_evol(
    hamiltonian: Any,
    initial_state: Tensor,
    t: Tensor,
    *,
    m: Optional[int] = None,
    s: Optional[int] = None,
    traceH: Optional[Tensor] = None,
    config: Optional[matrixfunc.TaylorConfig] = None,
) -> Tensor:
    """
    Evolve a state with a fixed-schedule scaling-and-Taylor exponential action.

    This is a JIT- and autodiff-compatible counterpart of SciPy's
    ``expm_multiply`` specialized to real-time evolution,
    ``exp(-1j * t * H) @ initial_state``. It accepts a dense or sparse matrix,
    a :class:`~tensorcircuit.quantum.LinearOperator`, or an MVP callable.

    Unlike SciPy, the Taylor degree ``m`` and scaling count ``s`` are static
    Python integers. The kernel deliberately does not estimate a norm or stop
    the Taylor series early, so it can be safely used under JIT and reverse-mode
    autodiff. Use :func:`estimate_expm_multiply_parameters` outside JIT to
    select a conservative schedule for a bounded time interval.

    :param hamiltonian: Hamiltonian matrix, LinearOperator, or MVP callable.
    :type hamiltonian: Any
    :param initial_state: One-dimensional initial state vector.
    :type initial_state: Tensor
    :param t: Real evolution time.
    :type t: Tensor
    :param m: Static Taylor degree. Must be non-negative.
    :type m: int
    :param s: Static number of scaling steps. Must be positive.
    :type s: int
    :param traceH: Optional trace of the Hamiltonian. Supplying it applies the
        trace shift exactly and can substantially reduce ``m * s``. Omitting it
        uses a zero shift and remains mathematically correct.
    :type traceH: Optional[Tensor]
    :param config: Optional shared :class:`~tensorcircuit.matrixfunc.TaylorConfig`.
        When supplied, ``m`` and ``s`` must be omitted; ``degree`` and
        ``scaling_steps`` replace them as static schedule values. ``traceH``
        remains a wrapper-level optional shift and is mapped to
        ``energy_shift=traceH / dimension``.
    :type config: Optional[matrixfunc.TaylorConfig]
    :return: The evolved state ``exp(-1j * t * H) @ initial_state``.
    :rtype: Tensor
    """
    if config is None:
        if m is None or s is None:
            raise ValueError("m and s must be provided when config is absent.")
        config = matrixfunc.TaylorConfig(degree=m, scaling_steps=s)
    else:
        if m is not None or s is not None:
            raise ValueError("config cannot be combined with m or s.")
        if not isinstance(config, matrixfunc.TaylorConfig):
            raise TypeError("config must be a TaylorConfig.")

    energy_shift = None
    if traceH is not None:
        state_size = backend.shape_tuple(initial_state)[0]
        energy_shift = backend.convert_to_tensor(traceH, dtype=dtypestr) / state_size

    return matrixfunc.exponential_action(
        hamiltonian,
        initial_state,
        -1.0j * backend.convert_to_tensor(t, dtype=dtypestr),
        config,
        energy_shift=energy_shift,
    )


@partial(
    arg_alias,
    alias_dict={"h": ["hamiltonian"], "psi0": ["initial_state"], "tlist": ["times"]},
)
def hamiltonian_evol(
    h: Tensor,
    psi0: Tensor,
    tlist: Tensor,
    callback: Optional[Callable[..., Any]] = None,
) -> Tensor:
    """
    Fast implementation of time independent Hamiltonian evolution using eigendecomposition.

    Unlike the other evolution methods in this module, this function evaluates
    ``exp(-t * H)`` and normalizes every output state. Real ``t`` therefore means
    imaginary-time evolution; use ``t=1j * time`` for real-time evolution.

    :param h: Time-independent Hamiltonian matrix
    :type h: Tensor
    :param psi0: Initial state vector
    :type psi0: Tensor
    :param tlist: Time points for evolution
    :type tlist: Tensor
    :param callback: Optional function to process state at each time point
    :type callback: Optional[Callable[..., Any]], optional
    :return: Evolution results at each time point. If callback is None, returns state vectors;
            otherwise returns callback results
    :rtype: Tensor

    :Example:

    >>> import tensorcircuit as tc
    >>> import numpy as np
    >>> # Define a simple 2-qubit Hamiltonian
    >>> h = tc.array_to_tensor([
    ...     [1.0, 0.0, 0.0, 0.0],
    ...     [0.0, -1.0, 2.0, 0.0],
    ...     [0.0, 2.0, -1.0, 0.0],
    ...     [0.0, 0.0, 0.0, 1.0]
    ... ])
    >>> # Initial state |00>
    >>> psi0 = tc.array_to_tensor([1.0, 0.0, 0.0, 0.0])
    >>> # Evolution times
    >>> times = tc.array_to_tensor([0.0, 0.5, 1.0])
    >>> # Evolve and get states
    >>> states = tc.experimental.hamiltonian_evol(h, psi0, times)
    >>> print(states.shape)  # (3, 4)


    Note:
        1. The Hamiltonian must be time-independent
        2. For time-dependent Hamiltonians, use ``evol_local`` or ``evol_global`` instead
        3. The evolution is performed in imaginary time by default (factor -t in exponential)
        4. The state is automatically normalized at each time point
    """
    psi0 = backend.cast(psi0, dtypestr)
    es, u = backend.eigh(h)
    u = backend.cast(u, dtypestr)
    eigenbasis_psi0 = backend.convert_to_tensor(
        backend.conj(backend.transpose(u)) @ backend.reshape(psi0, [-1, 1])
    )  # in case np.matrix...
    eigenbasis_psi0 = backend.reshape(eigenbasis_psi0, [-1])
    es = backend.cast(es, dtypestr)
    tlist = backend.cast(backend.convert_to_tensor(tlist), dtypestr)

    @backend.jit
    def _evol(t: Tensor) -> Tensor:
        evolved_eigenbasis = backend.exp(-t * es) * eigenbasis_psi0
        psi_exact = u @ backend.reshape(evolved_eigenbasis, [-1, 1])
        psi_exact = backend.reshape(psi_exact, [-1])
        psi_exact = psi_exact / backend.norm(psi_exact)
        if callback is None:
            return psi_exact
        return callback(psi_exact)

    return backend.stack([_evol(t) for t in tlist])


ed_evol = hamiltonian_evol


def _solve_ode(
    f: Callable[..., Tensor],
    s: Tensor,
    times: Tensor,
    args: Any,
    solver_kws: Dict[str, Any],
) -> Tensor:
    rtol = solver_kws.get("rtol", 1e-8)
    atol = solver_kws.get("atol", 1e-8)
    ode_backend = solver_kws.get("ode_backend", "jaxode")
    max_steps = solver_kws.get("max_steps", 4096)

    s = backend.cast(s, dtype=dtypestr)
    ts = backend.convert_to_tensor(times)
    if not backend.shape_tuple(ts):
        ts = backend.stack([backend.zeros_like(ts), ts])
    ts = backend.cast(ts, dtype=rdtypestr)

    if ode_backend == "jaxode":
        from jax.experimental.ode import odeint

        s1 = odeint(f, s, ts, rtol=rtol, atol=atol, mxstep=max_steps, *args)
        return s1

    import diffrax

    # Ignore complex warning
    warnings.simplefilter("ignore", category=UserWarning, append=True)

    solver = solver_kws.get("solver", "Tsit5")
    dt0 = solver_kws.get("dt0", 0.01)
    all_solvers = {
        "Dopri5": diffrax.Dopri5,
        "Tsit5": diffrax.Tsit5,
        "Dopri8": diffrax.Dopri8,
        "Kvaerno5": diffrax.Kvaerno5,
    }

    solver_obj = all_solvers[solver]()
    is_implicit = isinstance(solver_obj, diffrax.AbstractImplicitSolver)
    is_complex = "complex" in backend.dtype(s)

    if is_implicit and is_complex:
        # Implicit diffrax solvers require real-valued state.
        # Split complex state into real/imaginary parts and wrap the ODE.
        s_re = backend.real(s)
        s_im = backend.imag(s)
        s_real = backend.concat([s_re, s_im], axis=-1)
        n = s.shape[-1]

        def f_real(y_real: Tensor, t: Any, *fargs: Any) -> Tensor:
            y_complex = y_real[..., :n] + 1j * y_real[..., n:]
            dy_complex = f(y_complex, t, *fargs)
            return backend.concat(
                [backend.real(dy_complex), backend.imag(dy_complex)], axis=-1
            )

        term = diffrax.ODETerm(lambda t, y, fargs: f_real(y, t, *fargs))
        s1_real = diffrax.diffeqsolve(
            terms=term,
            solver=solver_obj,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=s_real,
            saveat=diffrax.SaveAt(ts=ts),
            args=args,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
        ).ys
        s1 = s1_real[..., :n] + 1j * s1_real[..., n:]
        return backend.cast(s1, dtype=dtypestr)

    # ODE
    term = diffrax.ODETerm(lambda t, y, args: f(y, t, *args))

    # solve ODE
    s1 = diffrax.diffeqsolve(
        terms=term,
        solver=solver_obj,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=s,
        saveat=diffrax.SaveAt(ts=ts),
        args=args,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    ).ys
    return s1


def ode_evol_local(
    hamiltonian: Callable[..., Tensor],
    initial_state: Tensor,
    times: Tensor,
    index: Sequence[int],
    callback: Optional[Callable[..., Tensor]] = None,
    *args: Any,
    **solver_kws: Any,
) -> Tensor:
    """
    ODE-based time evolution for a time-dependent Hamiltonian acting on a subsystem of qubits.
    This function solves the time-dependent Schrodinger equation using numerical ODE integration.
    The Hamiltonian is applied only to a specific subset of qubits (indices) in the system.
    The ode_backend parameter defaults to 'jaxode' (which uses ``jax.experimental.ode.odeint`` with a default solver
    of 'Dopri5'); if set to 'diffrax', it uses ``diffrax.diffeqsolve`` instead (with a default solver of 'Tsit5').

    Note: This function currently only supports the JAX backend.

    :param hamiltonian: A function that returns a dense Hamiltonian matrix for the specified
        subsystem size. The function signature should be ``hamiltonian(time, *args) -> Tensor``.
    :type hamiltonian: Callable[..., Tensor]
    :param initial_state: The initial quantum state vector of the full system.
    :type initial_state: Tensor
    :param times: Real time points for which to compute the evolution. A scalar
        denotes the final time of an interval starting at zero.
    :type times: Tensor
    :param index: Indices of qubits where the Hamiltonian is applied.
    :type index: Sequence[int]
    :param callback: Optional function to apply to the state at each time step.
    :type callback: Optional[Callable[..., Tensor]]
    :param args: Additional arguments to pass to the Hamiltonian function.
    :param solver_kws: Additional keyword arguments to pass to the ODE solver.

        - ``ode_backend='jaxode'`` (default) uses ``jax.experimental.ode.odeint``; ``ode_backend='diffrax'``
          uses ``diffrax.diffeqsolve``.

        - ``rtol`` (default: 1e-8) and ``atol`` (default: 1e-8) are used to determine how accurately you would
          like the numerical approximation to your equation.

        - The ``solver`` parameter accepts one of {'Tsit5' (default), 'Dopri5', 'Dopri8', 'Kvaerno5'}
          and only works when ``ode_backend='diffrax'``.

        - ``dt0`` (default: 0.01) specifies the initial step size and only works when ``ode_backend='diffrax'``.

        - ``max_steps`` (default: 4096)  The maximum number of steps to take before quitting the computation
          unconditionally and only works when ``ode_backend='diffrax'``.
    :type solver_kws: dict

    :return: Evolved quantum states at the specified time points. If callback is provided,
        returns the callback results; otherwise returns the state vectors.
    :rtype: Tensor
    """

    # TODO(@refraction-ray): support qudits (d != 2). The site count below uses
    # log2 and the body relies on `backend.reshape2` (hardcoded `[2]*n`), so
    # this path is qubit-only.

    n = int(np.log2(backend.shape_tuple(initial_state)[-1]) + 1e-7)
    l = len(index)

    def f(y: Tensor, t: Tensor, *args: Any) -> Tensor:
        y = backend.reshape2(y)
        y = Gate(y)
        h = -1.0j * hamiltonian(t, *args)
        if backend.is_sparse(h):
            h = backend.to_dense(h)
        h = backend.reshape2(h)
        h = Gate(h)
        edges = []
        for i in range(n):
            if i not in index:
                edges.append(y[i])
            else:
                j = index.index(i)
                edges.append(h[j])
                h[j + l] ^ y[i]
        y = contractor([y, h], output_edge_order=edges)
        return backend.reshape(y.tensor, [-1])

    s1 = _solve_ode(f, initial_state, times, args, solver_kws)

    if callback is None:
        return s1
    return backend.stack([callback(a_state) for a_state in s1])


def ode_evol_global(
    hamiltonian: Callable[..., Tensor],
    initial_state: Tensor,
    times: Tensor,
    callback: Optional[Callable[..., Tensor]] = None,
    *args: Any,
    mode: str = "hamiltonian",
    **solver_kws: Any,
) -> Tensor:
    """
    ODE-based time evolution for a time-dependent Hamiltonian acting on the entire system.
    This function solves the time-dependent Schrodinger equation using numerical ODE integration.
    The Hamiltonian is applied to the full system and should be provided in sparse matrix
    format for efficiency.
    The ode_backend parameter defaults to 'jaxode' (which uses ``jax.experimental.ode.odeint`` with a default solver
    of 'Dopri5'); if set to 'diffrax', it uses ``diffrax.diffeqsolve`` instead (with a default solver of 'Tsit5').

    Note: This function currently only supports the JAX backend.

    :param hamiltonian: Function defining the evolution. With ``mode="hamiltonian"``, the function
        returns a full-system Hamiltonian matrix with signature ``hamiltonian(time, *args) -> Tensor``.
        With ``mode="raw"``, the function returns the ODE right-hand side directly with signature
        ``hamiltonian(state, time, *args) -> Tensor``.
    :type hamiltonian: Callable[..., Tensor]
    :param initial_state: The initial quantum state vector.
    :type initial_state: Tensor
    :param times: Real time points for which to compute the evolution. A scalar
        denotes the final time of an interval starting at zero.
    :type times: Tensor
    :param callback: Optional function to apply to the state at each time step.
    :type callback: Optional[Callable[..., Tensor]]
    :param args: Additional arguments to pass to the Hamiltonian function.
    :type args: tuple | list
    :param solver_kws: Additional keyword arguments to pass to the ODE solver.

        - ``ode_backend='jaxode'`` (default) uses ``jax.experimental.ode.odeint``; ``ode_backend='diffrax'``
          uses ``diffrax.diffeqsolve``.

        - ``rtol`` (default: 1e-8) and ``atol`` (default: 1e-8) are used to determine how accurately you would
          like the numerical approximation to your equation.

        - The ``solver`` parameter accepts one of {'Tsit5' (default), 'Dopri5', 'Dopri8', 'Kvaerno5'}
          and only works when ``ode_backend='diffrax'``.

        - ``dt0`` (default: 0.01) specifies the initial step size and only works when ``ode_backend='diffrax'``.

        - ``max_steps`` (default: 4096)  The maximum number of steps to take before quitting the computation
          unconditionally and only works when ``ode_backend='diffrax'``.
    :type solver_kws: dict

    :return: Evolved quantum states at the specified time points. If callback is provided,
        returns the callback results; otherwise returns the state vectors.
    :rtype: Tensor
    """

    if mode == "raw":

        def f(y: Tensor, t: Tensor, *args: Any) -> Tensor:
            return hamiltonian(y, t, *args)

    elif mode == "hamiltonian":

        def f(y: Tensor, t: Tensor, *args: Any) -> Tensor:
            h = hamiltonian(t, *args)
            if callable(h) and not hasattr(h, "__matmul__"):
                h = aslinearoperator(
                    h,
                    shape=(initial_state.shape[0], initial_state.shape[0]),
                )
            return -1.0j * (h @ y)

    else:
        raise ValueError("mode must be 'hamiltonian' or 'raw'.")

    s1 = _solve_ode(f, initial_state, times, args, solver_kws)

    if callback is None:
        return s1
    return backend.stack([callback(a_state) for a_state in s1])


@partial(arg_alias, alias_dict={"h_fun": ["hamiltonian"], "t": ["times"]})
def evol_local(
    c: Circuit,
    index: Sequence[int],
    h_fun: Callable[..., Tensor],
    t: float,
    *args: Any,
    **solver_kws: Any,
) -> Circuit:
    """
    ode evolution of time dependent Hamiltonian on circuit of given indices
    [only jax backend support for now]

    :param c: Input circuit whose state is evolved.
    :type c: Circuit
    :param index: qubit sites to evolve
    :type index: Sequence[int]
    :param h_fun: h_fun should return a dense Hamiltonian matrix
        with input arguments ``time`` and ``*args``
    :type h_fun: Callable[..., Tensor]
    :param t: evolution time
    :type t: float
    :return: A new Circuit of the same type whose state is the ODE-evolved
        state at the final time point.
    :rtype: Circuit
    """
    s = c.state()
    # TODO(@refraction-ray): qubit-only; see `ode_evol_local` for the qudit
    # rework needed here (log2 site counting).
    n = int(np.log2(s.shape[-1]) + 1e-7)
    s1 = ode_evol_local(h_fun, s, t, index, None, *args, **solver_kws)
    return type(c)(n, inputs=s1[-1])


@partial(arg_alias, alias_dict={"h_fun": ["hamiltonian"], "t": ["times"]})
def evol_global(
    c: Circuit, h_fun: Callable[..., Tensor], t: float, *args: Any, **solver_kws: Any
) -> Circuit:
    """
    ode evolution of time dependent Hamiltonian on circuit of all qubits
    [only jax backend support for now]

    :param c: Input circuit whose state is evolved.
    :type c: Circuit
    :param h_fun: h_fun should return a **SPARSE** Hamiltonian matrix
        with input arguments ``time`` and ``*args``
    :type h_fun: Callable[..., Tensor]
    :param t: evolution time
    :type t: float
    :return: A new Circuit of the same type whose state is the ODE-evolved
        state at the final time point.
    :rtype: Circuit
    """
    s = c.state()
    n = c._nqubits
    s1 = ode_evol_global(h_fun, s, t, None, *args, **solver_kws)
    return type(c)(n, inputs=s1[-1])


def chebyshev_evol(
    hamiltonian: Any,
    initial_state: Tensor,
    t: float,
    spectral_bounds: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
    M: Optional[int] = None,
    *,
    config: Optional[matrixfunc.ChebyshevConfig] = None,
) -> Any:
    """
    Chebyshev evolution method by expanding the time evolution exponential operator
    in Chebyshev series.
    Note the state returned is not normalized. But the norm should be very close to 1 for
    sufficiently large ``k``, which can serve as an accuracy check of the final result.

    :param hamiltonian: Hamiltonian matrix, LinearOperator, or MVP callable
    :type hamiltonian: Any
    :param initial_state: Initial state vector
    :type initial_state: Tensor
    :param t: Time to evolve
    :type t: float
    :param spectral_bounds: Spectral bounds for the Hamiltonian (Emax, Emin)
    :type spectral_bounds: Tuple[float, float]
    :param k: Number of Chebyshev coefficients, a good estimate is k > t*(Emax-Emin)/2
    :type k: int
    :param M: Deprecated compatibility parameter. It is accepted for historical
        callers but is not used by the shared coefficient implementation; the
        internal Bessel recurrence length is selected from ``k``.
    :type M: int
    :param config: Optional shared :class:`~tensorcircuit.matrixfunc.ChebyshevConfig`.
        When supplied, the legacy ``spectral_bounds``, ``k``, and ``M`` arguments
        must be omitted. The config always uses ascending ``(emin, emax)`` bounds,
        while the legacy tuple remains ``(emax, emin)``.
    :type config: Optional[matrixfunc.ChebyshevConfig]
    :return: Evolved state ``exp(-1j * t * H) @ initial_state``.
    :rtype: Tensor
    """
    if config is None:
        if spectral_bounds is None or k is None:
            raise ValueError(
                "spectral_bounds and k must be provided when config is absent."
            )
        if M is not None:
            warnings.warn(
                "M is deprecated and ignored; the shared coefficient kernel "
                "chooses its internal recurrence length from k.",
                DeprecationWarning,
                stacklevel=2,
            )
        emax, emin = spectral_bounds
        config = matrixfunc.ChebyshevConfig(order=k, bounds=(emin, emax))
    else:
        if spectral_bounds is not None or k is not None or M is not None:
            raise ValueError("config cannot be combined with spectral_bounds, k, or M.")
        if not isinstance(config, matrixfunc.ChebyshevConfig):
            raise TypeError("config must be a ChebyshevConfig.")

    return matrixfunc.exponential_action(
        hamiltonian,
        initial_state,
        -1.0j * backend.convert_to_tensor(t, dtype=dtypestr),
        config,
    )


def estimate_k(t: float, spectral_bounds: Tuple[float, float]) -> int:
    """
    Estimate the Chebyshev expansion order for a time interval.

    :param t: Evolution time; its absolute magnitude determines the order.
    :type t: float
    :param spectral_bounds: Historical bounds in the order ``(Emax, Emin)``.
    :type spectral_bounds: Tuple[float, float]
    :return: Estimated number of Chebyshev terms.
    :rtype: int
    """
    E_max, E_min = spectral_bounds
    a = (E_max - E_min) / 2.0
    tau = abs(a * t)
    return max(int(1.1 * tau), int(tau + 20))


def estimate_M(t: float, spectral_bounds: Tuple[float, float], k: int) -> int:
    """
    Estimate the historical Bessel recurrence length.

    The returned value is retained for scripts that used the old Chebyshev
    interface. ``chebyshev_evol`` now ignores it and chooses its internal
    recurrence length from the shared configuration.

    :param t: Evolution time.
    :type t: float
    :param spectral_bounds: Historical bounds in the order ``(Emax, Emin)``.
    :type spectral_bounds: Tuple[float, float]
    :param k: Chebyshev expansion order.
    :type k: int
    :return: Historical recurrence-length recommendation.
    :rtype: int
    """
    E_max, E_min = spectral_bounds
    a = (E_max - E_min) / 2.0
    tau = a * t  # tau is now a scalar
    safety_factor = 15
    M = max(k, int(abs(tau))) + int(safety_factor * np.sqrt(abs(tau)))
    M = max(M, k + 30)
    return M


def estimate_spectral_bounds(
    h: Any,
    n_iter: int = 30,
    psi0: Optional[Any] = None,
    shape: Optional[Sequence[int]] = None,
) -> Tuple[float, float]:
    """
    Lanczos algorithm to estimate the spectral bounds of a Hamiltonian.
    Just for quick run before `chebyshev_evol`, non jit-able.

    :param h: Hamiltonian matrix, LinearOperator, or MVP callable.
    :type h: Any
    :param n_iter: iteration number.
    :type n_iter: int
    :param psi0: Optional initial state.
    :type psi0: Optional[Any]
    :param shape: Optional operator shape. Required when ``h`` is an MVP callable
        and ``psi0`` is not provided.
    :type shape: Optional[Sequence[int]]
    :return: Historical descending bounds ``(E_max, E_min)`` for ``chebyshev_evol``.
    :rtype: Tuple[float, float]
    """
    if n_iter < 1:
        raise ValueError("n_iter must be positive.")
    if psi0 is None:
        if shape is None:
            shape = getattr(h, "shape", None)
        if shape is None:
            raise ValueError("shape is required when psi0 is not provided.")
        dimension = int(shape[-1])
        psi0 = np.random.normal(size=[dimension])
    else:
        dimension = backend.shape_tuple(psi0)[0]

    emin, emax = matrixfunc.estimate_spectral_bounds(
        h,
        backend.cast(backend.convert_to_tensor(psi0), dtypestr),
        matrixfunc.KrylovConfig(max_dim=n_iter),
        dimension=dimension,
        padding=0.0,
    )
    return emax, emin
