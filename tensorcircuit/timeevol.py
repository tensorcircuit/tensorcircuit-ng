"""
Analog time evolution engines
"""

from typing import Any, Tuple, Optional, Callable, List, Sequence

from .cons import backend, dtypestr, rdtypestr, contractor
from .gates import Gate

Tensor = Any
Circuit = Any


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
    time_points: Tensor,
    subspace_dimension: int,
    callback: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """
    Perform quantum state time evolution using Krylov subspace method.

    :param hamiltonian: Sparse or dense Hamiltonian matrix
    :type hamiltonian: Tensor
    :param initial_state: Initial quantum state
    :type initial_state: Tensor
    :param time_points: List of time points
    :type time_points: Tensor
    :param subspace_dimension: Krylov subspace dimension
    :type subspace_dimension: int
    :param callback: Optional callback function applied to quantum state at
                  each evolution time point, return some observables
    :type callback: Optional[Callable[[Any], Any]], optional
    :return: List of evolved quantum states, or list of callback function results
        (if callback provided)
    :rtype: Any
    """
    # TODO(@refraction-ray): stable and efficient AD is to be investigated
    basis_matrix, projected_hamiltonian = lanczos_iteration(
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
    time_points = backend.convert_to_tensor(time_points)
    time_points = backend.cast(time_points, dtypestr)

    # Transform projected state to eigenbasis: |psi_coeff> = U^† |psi_proj>
    eigenvectors_projected_state = backend.matvec(
        backend.conj(backend.transpose(eigenvectors)), projected_state
    )

    # Calculate exp(-i*projected_H*t) * projected_state
    results = []
    for t in time_points:
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


def hamiltonian_evol(
    tlist: Tensor,
    h: Tensor,
    psi0: Tensor,
    callback: Optional[Callable[..., Any]] = None,
) -> Tensor:
    """
    Fast implementation of time independent Hamiltonian evolution using eigendecomposition.
    By default, performs imaginary time evolution.

    :param tlist: Time points for evolution
    :type tlist: Tensor
    :param h: Time-independent Hamiltonian matrix
    :type h: Tensor
    :param psi0: Initial state vector
    :type psi0: Tensor
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
    >>> # Initial state |00⟩
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
    utpsi0 = backend.reshape(
        backend.transpose(u) @ backend.reshape(psi0, [-1, 1]), [-1]
    )
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
        with input arguments time and *args
    :type h_fun: Callable[..., Tensor]
    :param t: evolution time
    :type t: float
    :return: _description_
    :rtype: Circuit
    """
    from jax.experimental.ode import odeint

    s = c.state()
    n = c._nqubits
    l = len(index)

    def f(y: Tensor, t: Tensor, *args: Any) -> Tensor:
        y = backend.reshape2(y)
        y = Gate(y)
        h = -1.0j * h_fun(t, *args)
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

    ts = backend.stack([0.0, t])
    ts = backend.cast(ts, dtype=rdtypestr)
    s1 = odeint(f, s, ts, *args, **solver_kws)
    return type(c)(n, inputs=s1[-1])


ode_evol_local = evol_local


def evol_global(
    c: Circuit, h_fun: Callable[..., Tensor], t: float, *args: Any, **solver_kws: Any
) -> Circuit:
    """
    ode evolution of time dependent Hamiltonian on circuit of all qubits
    [only jax backend support for now]

    :param c: _description_
    :type c: Circuit
    :param h_fun: h_fun should return a **SPARSE** Hamiltonian matrix
        with input arguments time and *args
    :type h_fun: Callable[..., Tensor]
    :param t: _description_
    :type t: float
    :return: _description_
    :rtype: Circuit
    """
    from jax.experimental.ode import odeint

    s = c.state()
    n = c._nqubits

    def f(y: Tensor, t: Tensor, *args: Any) -> Tensor:
        h = -1.0j * h_fun(t, *args)
        return backend.sparse_dense_matmul(h, y)

    ts = backend.stack([0.0, t])
    ts = backend.cast(ts, dtype=rdtypestr)
    s1 = odeint(f, s, ts, *args, **solver_kws)
    return type(c)(n, inputs=s1[-1])


ode_evol_global = evol_global
