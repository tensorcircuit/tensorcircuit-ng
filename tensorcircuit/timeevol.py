"""
Analog time evolution engines
"""

from typing import Any, Tuple, Optional, Callable, List, Sequence, Dict
from functools import partial
import warnings

import numpy as np

from .cons import backend, dtypestr, rdtypestr, contractor
from .gates import Gate
from .utils import arg_alias

Tensor = Any
Circuit = Any


def lanczos_iteration_scan(
    hamiltonian: Any, initial_vector: Any, subspace_dimension: int
) -> Tuple[Any, Any]:
    """
    Use Lanczos algorithm to construct orthogonal basis and projected Hamiltonian
    of Krylov subspace, using `tc.backend.scan` for JIT compatibility.

    :param hamiltonian: Sparse or dense Hamiltonian matrix
    :type hamiltonian: Tensor
    :param initial_vector: Initial quantum state vector
    :type initial_vector: Tensor
    :param subspace_dimension: Dimension of Krylov subspace
    :type subspace_dimension: int
    :return: Tuple containing (basis matrix, projected Hamiltonian)
    :rtype: Tuple[Tensor, Tensor]
    """
    state_size = backend.shape_tuple(initial_vector)[0]
    if backend.is_sparse(hamiltonian):
        hamiltonian = backend.sparse_csr_from_coo(hamiltonian)

    # Main scan body for the outer loop (iterating j)
    def lanczos_step(carry: Tuple[Any, ...], j: int) -> Tuple[Any, ...]:
        v, basis, alphas, betas = carry

        if backend.is_sparse(hamiltonian):
            w = backend.sparse_dense_matmul(hamiltonian, v)
        else:
            w = backend.matvec(hamiltonian, v)

        alpha = backend.real(backend.sum(backend.conj(v) * w))
        w = w - backend.cast(alpha, dtypestr) * v

        # Inner scan for re-orthogonalization (iterating k)
        # def ortho_step(inner_carry: Tuple[Any, Any], k: int) -> Tuple[Any, Any]:
        #     w_carry, j_val = inner_carry

        #     def do_projection() -> Any:
        #         # `basis` is available here through closure
        #         v_k = basis[:, k]
        #         projection = backend.sum(backend.conj(v_k) * w_carry)
        #         return w_carry - projection * v_k

        #     def do_nothing() -> Any:
        #         return w_carry

        #     # Orthogonalize against v_0, ..., v_j
        #     w_new = backend.cond(k <= j_val, do_projection, do_nothing)
        #     return (w_new, j_val)  # Return the new carry for the inner loop

        # # Pass `j` into the inner scan's carry
        # inner_init_carry = (w, j)
        # final_inner_carry = backend.scan(
        #     ortho_step, backend.arange(subspace_dimension), inner_init_carry
        # )
        # w_ortho = final_inner_carry[0]

        def ortho_step(w_carry: Any, elems_tuple: Tuple[Any, Any]) -> Any:
            k, j_from_elems = elems_tuple

            def do_projection() -> Any:
                v_k = basis[:, k]
                projection = backend.sum(backend.conj(v_k) * w_carry)
                return w_carry - projection * v_k

            def do_nothing() -> Any:
                return backend.cast(w_carry, dtype=dtypestr)

            w_new = backend.cond(k <= j_from_elems, do_projection, do_nothing)
            return w_new

        k_elems = backend.arange(subspace_dimension)
        j_elems = backend.tile(backend.reshape(j, [1]), [subspace_dimension])
        inner_elems = (k_elems, j_elems)
        w_ortho = backend.scan(ortho_step, inner_elems, w)

        beta = backend.norm(w_ortho)
        beta = backend.real(beta)

        # Update alphas and betas arrays
        new_alphas = backend.scatter(
            alphas, backend.reshape(j, [1, 1]), backend.reshape(alpha, [1])
        )
        new_betas = backend.scatter(
            betas, backend.reshape(j, [1, 1]), backend.reshape(beta, [1])
        )

        def update_state_fn() -> Tuple[Any, Any]:
            epsilon = 1e-15
            next_v = w_ortho / backend.cast(beta + epsilon, dtypestr)

            one_hot_update = backend.onehot(j + 1, subspace_dimension)
            one_hot_update = backend.cast(one_hot_update, dtype=dtypestr)

            # Create a mask to update only the (j+1)-th column
            mask = 1.0 - backend.reshape(one_hot_update, [1, subspace_dimension])
            new_basis = basis * mask + backend.reshape(
                next_v, [-1, 1]
            ) * backend.reshape(one_hot_update, [1, subspace_dimension])

            return next_v, new_basis

        def keep_state_fn() -> Tuple[Any, Any]:
            return v, basis

        next_v_carry, new_basis = backend.cond(
            j < subspace_dimension - 1, update_state_fn, keep_state_fn
        )

        return (next_v_carry, new_basis, new_alphas, new_betas)

    # Prepare initial state for the main scan
    v0 = initial_vector / backend.norm(initial_vector)

    init_basis = backend.zeros((state_size, subspace_dimension), dtype=dtypestr)
    init_alphas = backend.zeros((subspace_dimension,), dtype=rdtypestr)
    init_betas = backend.zeros((subspace_dimension,), dtype=rdtypestr)

    one_hot_0 = backend.onehot(0, subspace_dimension)
    one_hot_0 = backend.cast(one_hot_0, dtype=dtypestr)
    init_basis = init_basis + backend.reshape(v0, [-1, 1]) * backend.reshape(
        one_hot_0, [1, subspace_dimension]
    )

    init_carry = (v0, init_basis, init_alphas, init_betas)

    # Run the main scan
    final_carry = backend.scan(
        lanczos_step, backend.arange(subspace_dimension), init_carry
    )
    basis_matrix, alphas_tensor, betas_tensor = (
        final_carry[1],
        final_carry[2],
        final_carry[3],
    )

    betas_off_diag = betas_tensor[:-1]

    diag_part = backend.diagflat(alphas_tensor)
    if backend.shape_tuple(betas_off_diag)[0] > 0:
        off_diag_part = backend.diagflat(betas_off_diag, k=1)
        projected_hamiltonian = (
            diag_part + off_diag_part + backend.conj(backend.transpose(off_diag_part))
        )
    else:
        projected_hamiltonian = diag_part

    return basis_matrix, projected_hamiltonian


def lanczos_iteration(
    hamiltonian: Tensor, initial_vector: Tensor, subspace_dimension: int
) -> Tuple[Tensor, Tensor]:
    """
    Use Lanczos algorithm to construct orthogonal basis and projected Hamiltonian
    of Krylov subspace.

    :param hamiltonian: Sparse or dense Hamiltonian matrix
    :type hamiltonian: Tensor
    :param initial_vector: Initial quantum state vector
    :type initial_vector: Tensor
    :param subspace_dimension: Dimension of Krylov subspace
    :type subspace_dimension: int
    :return: Tuple containing (basis matrix, projected Hamiltonian)
    :rtype: Tuple[Tensor, Tensor]
    """
    # Initialize
    vector = initial_vector
    vector = backend.cast(vector, dtypestr)

    # Use list to store basis vectors
    basis_vectors: List[Any] = []

    # Store alpha and beta coefficients for constructing tridiagonal matrix
    alphas = []
    betas = []

    # Normalize initial vector
    vector_norm = backend.norm(vector)
    vector = vector / vector_norm

    # Add first basis vector
    basis_vectors.append(vector)

    if backend.is_sparse(hamiltonian):
        hamiltonian = backend.sparse_csr_from_coo(hamiltonian)

    # Lanczos iteration (fixed number of iterations for JIT compatibility)
    for j in range(subspace_dimension):
        # Calculate H|v_j>
        if backend.is_sparse(hamiltonian):
            w = backend.sparse_dense_matmul(hamiltonian, vector)
        else:
            w = backend.matvec(hamiltonian, vector)

        # Calculate alpha_j = <v_j|H|v_j>
        alpha = backend.real(backend.sum(backend.conj(vector) * w))
        alphas.append(alpha)

        # w = H|v_j> - alpha_j|v_j> - beta_{j-1}|v_{j-1}>
        # is not sufficient, require re-normalization
        w = w - backend.cast(alpha, dtypestr) * vector

        for k in range(j + 1):
            v_k = basis_vectors[k]
            projection = backend.sum(backend.conj(v_k) * w)
            w = w - projection * v_k

        # if j > 0:
        #     w = w - prev_beta * basis_vectors[-2]

        # Calculate beta_{j+1} = ||w||
        beta = backend.norm(w)
        betas.append(beta)

        # Use regularization technique to avoid division by zero error,
        # adding small epsilon value to ensure numerical stability
        epsilon = 1e-15
        norm_factor = 1.0 / (beta + epsilon)

        # Normalize w to get |v_{j+1}> (except for the last iteration)
        if j < subspace_dimension - 1:
            vector = w * backend.cast(norm_factor, dtypestr)
            basis_vectors.append(vector)

    # Construct final basis matrix
    basis_matrix = backend.stack(basis_vectors, axis=1)

    # Construct tridiagonal projected Hamiltonian
    # Use vectorized method to construct tridiagonal matrix at once
    alphas_tensor = backend.stack(alphas)
    # Only use first krylov_dim-1 beta values to construct off-diagonal
    betas_tensor = backend.stack(betas[:-1]) if len(betas) > 1 else backend.stack([])

    # Convert to correct data type
    alphas_tensor = backend.cast(alphas_tensor, dtype=dtypestr)
    if len(betas_tensor) > 0:
        betas_tensor = backend.cast(betas_tensor, dtype=dtypestr)

    # Construct diagonal and off-diagonal parts
    diag_part = backend.diagflat(alphas_tensor)
    if len(betas_tensor) > 0:
        off_diag_part = backend.diagflat(betas_tensor, k=1)
        projected_hamiltonian = (
            diag_part + off_diag_part + backend.transpose(off_diag_part)
        )
    else:
        projected_hamiltonian = diag_part

    return basis_matrix, projected_hamiltonian


def krylov_evol(
    hamiltonian: Tensor,
    initial_state: Tensor,
    times: Tensor,
    subspace_dimension: int,
    callback: Optional[Callable[[Any], Any]] = None,
    scan_impl: bool = False,
) -> Any:
    """
    Perform quantum state time evolution using Krylov subspace method.

    :param hamiltonian: Sparse or dense Hamiltonian matrix
    :type hamiltonian: Tensor
    :param initial_state: Initial quantum state
    :type initial_state: Tensor
    :param times: List of time points
    :type times: Tensor
    :param subspace_dimension: Krylov subspace dimension
    :type subspace_dimension: int
    :param callback: Optional callback function applied to quantum state at
                  each evolution time point, return some observables
    :type callback: Optional[Callable[[Any], Any]], optional
    :param scan_impl: whether use scan implementation, suitable for jit but may be slow on numpy
        defaults False, True not work for tensorflow backend + jit, due to stupid issue of tensorflow
        context separation and the notorious inaccesibletensor error
    :type scan_impl: bool, optional
    :return: List of evolved quantum states, or list of callback function results
        (if callback provided)
    :rtype: Any
    """
    # TODO(@refraction-ray): stable and efficient AD is to be investigated
    if not scan_impl:
        basis_matrix, projected_hamiltonian = lanczos_iteration(
            hamiltonian, initial_state, subspace_dimension
        )
    else:
        basis_matrix, projected_hamiltonian = lanczos_iteration_scan(
            hamiltonian, initial_state, subspace_dimension
        )
    initial_state = backend.cast(initial_state, dtypestr)
    # Project initial state to Krylov subspace: |psi_proj> = V_m^† |psi(0)>
    projected_state = backend.matvec(
        backend.conj(backend.transpose(basis_matrix)), initial_state
    )

    # Perform spectral decomposition of projected Hamiltonian: T_m = U D U^†
    eigenvalues, eigenvectors = backend.eigh(projected_hamiltonian)
    eigenvalues = backend.cast(eigenvalues, dtypestr)
    eigenvectors = backend.cast(eigenvectors, dtypestr)
    times = backend.convert_to_tensor(times)
    times = backend.cast(times, dtypestr)

    # Transform projected state to eigenbasis: |psi_coeff> = U^† |psi_proj>
    eigenvectors_projected_state = backend.matvec(
        backend.conj(backend.transpose(eigenvectors)), projected_state
    )

    # Calculate exp(-i*projected_H*t) * projected_state
    results = []
    for t in times:
        # Calculate exp(-i*eigenvalues*t)
        exp_diagonal = backend.exp(-1j * eigenvalues * t)

        # Evolve state in eigenbasis: |psi_evolved_coeff> = exp(-i*D*t) |psi_coeff>
        evolved_projected_coeff = exp_diagonal * eigenvectors_projected_state

        # Transform back to eigenbasis: |psi_evolved_proj> = U |psi_evolved_coeff>
        evolved_projected = backend.matvec(eigenvectors, evolved_projected_coeff)

        # Transform back to original basis: |psi(t)> = V_m |psi_evolved_proj>
        evolved_state = backend.matvec(basis_matrix, evolved_projected)

        # Apply callback function if provided
        if callback is not None:
            result = callback(evolved_state)
        else:
            result = evolved_state

        results.append(result)

    return backend.stack(results)


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
    By default, performs imaginary time evolution.

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
    >>> states = tc.experimental.hamiltonian_evol(times, h, psi0)
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
    utpsi0 = backend.convert_to_tensor(
        backend.transpose(u) @ backend.reshape(psi0, [-1, 1])
    )  # in case np.matrix...
    utpsi0 = backend.reshape(utpsi0, [-1])
    es = backend.cast(es, dtypestr)
    tlist = backend.cast(backend.convert_to_tensor(tlist), dtypestr)

    @backend.jit
    def _evol(t: Tensor) -> Tensor:
        ebetah_utpsi0 = backend.exp(-t * es) * utpsi0
        psi_exact = backend.conj(u) @ backend.reshape(ebetah_utpsi0, [-1, 1])
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

    ts = backend.convert_to_tensor(times)
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

    # ODE
    term = diffrax.ODETerm(lambda t, y, args: f(y, t, *args))

    # solve ODE
    s1 = diffrax.diffeqsolve(
        terms=term,
        solver=all_solvers[solver](),
        t0=times[0],
        t1=times[-1],
        dt0=dt0,
        y0=s,
        saveat=diffrax.SaveAt(ts=times),
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
    :param times: Time points for which to compute the evolution. Should be a 1D array of times.
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

        - ``t0`` (default: 0.01) specifies the initial step size and only works when ``ode_backend='diffrax'``.

        - ``max_steps`` (default: 4096)  The maximum number of steps to take before quitting the computation
          unconditionally and only works when ``ode_backend='diffrax'``.
    :type solver_kws: dict

    :return: Evolved quantum states at the specified time points. If callback is provided,
        returns the callback results; otherwise returns the state vectors.
    :rtype: Tensor
    """

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

    :param hamiltonian: A function that returns a sparse Hamiltonian matrix for the full system.
        The function signature should be ``hamiltonian(time, *args) -> Tensor``.
    :type hamiltonian: Callable[..., Tensor]
    :param initial_state: The initial quantum state vector.
    :type initial_state: Tensor
    :param times: Time points for which to compute the evolution. Should be a 1D array of times.
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

        - ``t0`` (default: 0.01) specifies the initial step size and only works when ``ode_backend='diffrax'``.

        - ``max_steps`` (default: 4096)  The maximum number of steps to take before quitting the computation
          unconditionally and only works when ``ode_backend='diffrax'``.
    :type solver_kws: dict

    :return: Evolved quantum states at the specified time points. If callback is provided,
        returns the callback results; otherwise returns the state vectors.
    :rtype: Tensor
    """

    def f(y: Tensor, t: Tensor, *args: Any) -> Tensor:
        h = -1.0j * hamiltonian(t, *args)
        return h @ y

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

    :param c: _description_
    :type c: Circuit
    :param index: qubit sites to evolve
    :type index: Sequence[int]
    :param h_fun: h_fun should return a dense Hamiltonian matrix
        with input arguments ``time`` and ``*args``
    :type h_fun: Callable[..., Tensor]
    :param t: evolution time
    :type t: float
    :return: _description_
    :rtype: Circuit
    """
    s = c.state()
    n = int(np.log2(s.shape[-1]) + 1e-7)
    if isinstance(t, float):
        t = backend.stack([0.0, t])
    s1 = ode_evol_local(h_fun, s, t, index, None, *args, **solver_kws)
    return type(c)(n, inputs=s1[-1])


@partial(arg_alias, alias_dict={"h_fun": ["hamiltonian"], "t": ["times"]})
def evol_global(
    c: Circuit, h_fun: Callable[..., Tensor], t: float, *args: Any, **solver_kws: Any
) -> Circuit:
    """
    ode evolution of time dependent Hamiltonian on circuit of all qubits
    [only jax backend support for now]

    :param c: _description_
    :type c: Circuit
    :param h_fun: h_fun should return a **SPARSE** Hamiltonian matrix
        with input arguments ``time`` and ``*args``
    :type h_fun: Callable[..., Tensor]
    :param t: _description_
    :type t: float
    :return: _description_
    :rtype: Circuit
    """
    s = c.state()
    n = c._nqubits
    if isinstance(t, float):
        t = backend.stack([0.0, t])
    s1 = ode_evol_global(h_fun, s, t, None, *args, **solver_kws)
    return type(c)(n, inputs=s1[-1])


def chebyshev_evol(
    hamiltonian: Any,
    initial_state: Tensor,
    t: float,
    spectral_bounds: Tuple[float, float],
    k: int,
    M: int,
) -> Any:
    """
    Chebyshev evolution method by expanding the time evolution exponential operator
    in Chebyshev series.
    Note the state returned is not normalized. But the norm should be very close to 1 for
    sufficiently large k and M, which can serve as a accuracy check of the final result.

    :param hamiltonian: Hamiltonian matrix (sparse or dense)
    :type hamiltonian: Any
    :param initial_state: Initial state vector
    :type initial_state: Tensor
    :param time: Time to evolve
    :type time: float
    :param spectral_bounds: Spectral bounds for the Hamiltonian (Emax, Emin)
    :type spectral_bounds: Tuple[float, float]
    :param k: Number of Chebyshev coefficients, a good estimate is k > t*(Emax-Emin)/2
    :type k: int
    :param M: Number of iterations to estimate Bessel function, a good estimate is given
        by `estimate_M` helper method.
    :type M: int
    :return: Evolved state
    :rtype: Tensor
    """
    # TODO(@refraction-ray): no support for tf backend as bessel function has no implementation
    E_max, E_min = spectral_bounds
    if E_max <= E_min:
        raise ValueError("E_max must be > E_min.")

    a = (E_max - E_min) / 2.0
    b = (E_max + E_min) / 2.0
    tau = a * t  # Rescaled time parameter

    if backend.is_sparse(hamiltonian):
        hamiltonian = backend.sparse_csr_from_coo(hamiltonian)

    def apply_h_norm(psi: Any) -> Any:
        """Applies the normalized Hamiltonian to a state."""
        return ((hamiltonian @ psi) - b * psi) / a

    # Handle edge case where no evolution is needed.
    if k == 0:
        # The phase factor still applies even for zero evolution of the series part.
        phase = backend.exp(-1j * b * t)
        return phase * backend.zeros_like(initial_state)

    # --- 2. Calculate Chebyshev Expansion Coefficients ---
    k_indices = backend.arange(k)
    bessel_vals = backend.special_jv(k, tau, M)

    # Prefactor is 1 for k=0 and 2 for k>0.
    prefactor = backend.ones([k])
    if k > 1:
        # Using concat for backend compatibility (vs. jax's .at[1:].set(2.0))
        prefactor = backend.concat(
            [backend.ones([1]), backend.ones([k - 1]) * 2.0], axis=0
        )

    ik_powers = backend.power(0 - 1j, k_indices)
    coeffs = prefactor * ik_powers * bessel_vals

    # --- 3. Iteratively build the result using a scan ---

    # Handle the simple case of k=1 separately.
    if k == 1:
        psi_unphased = coeffs[0] * initial_state
    else:  # k >= 2, use the scan operation.
        # Initialize the first two Chebyshev vectors and the initial sum.
        T0 = initial_state
        T1 = apply_h_norm(T0)
        initial_sum = coeffs[0] * T0 + coeffs[1] * T1

        # The carry for the scan holds the state needed for the next iteration:
        # (current vector T_k, previous vector T_{k-1}, and the running sum).
        initial_carry = (T1, T0, initial_sum)

        def scan_body(carry, i):  # type: ignore
            """The body of the scan operation."""
            Tk, Tkm1, current_sum = carry

            # Calculate the next Chebyshev vector using the recurrence relation.
            Tkp1 = 2 * apply_h_norm(Tk) - Tkm1

            # Add its contribution to the running sum.
            new_sum = current_sum + coeffs[i] * Tkp1

            # Return the updated carry for the next step. No intermediate output is needed.
            return (Tkp1, Tk, new_sum)

        # Run the scan over the remaining coefficients (from index 2 to k-1).
        final_carry = backend.scan(scan_body, backend.arange(2, k), initial_carry)

        # The final result is the sum accumulated in the last carry state.
        psi_unphased = final_carry[2]

    # --- 4. Final Step: Apply Phase Correction ---
    # This undoes the energy shift from the Hamiltonian normalization.
    phase = backend.exp(-1j * b * t)
    psi_final = phase * psi_unphased

    return psi_final


def estimate_k(t: float, spectral_bounds: Tuple[float, float]) -> int:
    """
    estimate k for chebyshev expansion

    :param t: time
    :type t: float
    :param spectral_bounds: spectral bounds (Emax, Emin)
    :type spectral_bounds: Tuple[float, float]
    :return: k
    :rtype: int
    """
    E_max, E_min = spectral_bounds
    a = (E_max - E_min) / 2.0
    tau = a * t  # tau is now a scalar
    return max(int(1.1 * tau), int(tau + 20))


def estimate_M(t: float, spectral_bounds: Tuple[float, float], k: int) -> int:
    """
    estimate M for Bessel function iterations

    :param t: time
    :type t: float
    :param spectral_bounds: spectral bounds (Emax, Emin)
    :type spectral_bounds: Tuple[float, float]
    :param k: k
    :type k: int
    :return: M
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
    h: Any, n_iter: int = 30, psi0: Optional[Any] = None
) -> Tuple[float, float]:
    """
    Lanczos algorithm to estimate the spectral bounds of a Hamiltonian.
    Just for quick run before `chebyshev_evol`, non jit-able.

    :param h: Hamiltonian matrix.
    :type h: Any
    :param n_iter: iteration number.
    :type n_iter: int
    :param psi0: Optional initial state.
    :type psi0: Optional[Any]
    :return: (E_max, E_min)
    """
    shape = h.shape
    D = shape[-1]
    if psi0 is None:
        psi0 = np.random.normal(size=[D])

    psi0 = backend.convert_to_tensor(psi0) / backend.norm(psi0)
    psi0 = backend.cast(psi0, dtypestr)

    # Lanczos
    alphas = []
    betas = []
    q_prev = backend.zeros(psi0.shape, dtype=psi0.dtype)
    q = psi0
    beta = 0

    for _ in range(n_iter):
        r = h @ q
        r = backend.convert_to_tensor(r)  # in case np.matrix
        r = backend.reshape(r, [-1])
        if beta != 0:
            r -= backend.cast(beta, dtypestr) * q_prev

        alpha = backend.real(backend.sum(backend.conj(q) * r))

        alphas.append(alpha)

        r -= backend.cast(alpha, dtypestr) * q

        q_prev = q
        beta = backend.norm(r)
        q = r / beta
        beta = backend.abs(beta)
        betas.append(beta)
        if beta < 1e-8:
            break

    alphas = backend.stack(alphas)
    betas = backend.stack(betas)
    T = (
        backend.diagflat(alphas)
        + backend.diagflat(betas[:-1], k=1)
        + backend.diagflat(betas[:-1], k=-1)
    )

    ritz_values, _ = backend.eigh(T)

    return backend.max(ritz_values), backend.min(ritz_values)
