"""
Chebyshev time evolution example with CLI options.

This script demonstrates the Chebyshev time evolution method with various options
for backend, matrix type, and JIT compilation.
"""

import argparse
from typing import Any, Tuple
import numpy as np
from scipy.linalg import expm

import tensorcircuit as tc

tc.set_dtype("complex128")


def create_heisenberg_hamiltonian(num_sites: int, sparse: bool = True) -> Any:
    """
    Create Heisenberg Hamiltonian for a 1D chain.

    Args:
        num_sites: Number of sites in the chain.
        sparse: Whether to create a sparse matrix.

    Returns:
        Hamiltonian matrix.
    """
    graph = tc.templates.graphs.Line1D(num_sites)
    return tc.quantum.heisenberg_hamiltonian(graph, sparse=sparse)


def create_initial_state(dim: int) -> Any:
    """
    Create initial state as equal superposition.

    Args:
        dim: Dimension of the Hilbert space.

    Returns:
        Normalized initial state.
    """
    psi0 = tc.backend.ones([dim])
    return psi0 / tc.backend.norm(psi0)


def estimate_spectral_bounds(hamiltonian: Any, n_iter: int = 40) -> Tuple[float, float]:
    """
    Estimate spectral bounds of the Hamiltonian.

    Args:
        hamiltonian: Hamiltonian matrix.
        n_iter: Number of iterations for Lanczos algorithm.

    Returns:
        Tuple of (E_max, E_min).
    """
    print(f"Estimating spectral bounds of Hamiltonian (iterations: {n_iter})...")
    # Ensure the initial vector is compatible with JAX backend
    e_max, e_min = tc.timeevol.estimate_spectral_bounds(hamiltonian, n_iter=n_iter)
    print(f"Estimated result: E_max = {e_max:.4f}, E_min = {e_min:.4f}")
    return float(e_max), float(e_min)


def compare_with_exact_evolution(
    hamiltonian: Any,
    initial_state: Any,
    chebyshev_state: Any,
    time: float,
) -> float:
    """
    Compare Chebyshev evolution result with exact evolution.

    Args:
        hamiltonian: Hamiltonian matrix.
        initial_state: Initial quantum state.
        chebyshev_state: State evolved with Chebyshev method.
        time: Evolution time.

    Returns:
        Fidelity between the two states.
    """
    # Exact evolution using matrix exponential
    if tc.backend.is_sparse(hamiltonian):
        h = tc.backend.to_dense(hamiltonian)
    else:
        h = hamiltonian
    psi_exact = expm(-1j * np.asarray(h) * time) @ np.asarray(initial_state)

    fidelity = np.abs(np.vdot(psi_exact, np.asarray(chebyshev_state))) ** 2
    return fidelity


def run_chebyshev_evolution(
    num_sites: int = 8,
    time: float = 500.0,
    backend_name: str = "numpy",
    sparse: bool = True,
    use_jit: bool = False,
) -> None:
    """
    Run Chebyshev time evolution with specified parameters.

    Args:
        num_sites: Number of sites in the system.
        time: Evolution time.
        backend_name: Backend to use (numpy, jax, tensorflow, pytorch).
        sparse: Whether to use sparse matrices.
        use_jit: Whether to use JIT compilation.
    """
    # Set backend
    tc.set_dtype("complex128")  # Ensure dtype is set after backend if needed
    tc.set_backend(backend_name)
    backend = tc.backend
    print(f"Using {backend_name} backend")

    # Create system
    dim = 2**num_sites
    graph = tc.templates.graphs.Line1D(num_sites)
    h_matrix = tc.quantum.heisenberg_hamiltonian(graph, sparse=sparse)
    print(f"Created Heisenberg Hamiltonian for {num_sites} sites")
    print(f"Matrix is {'sparse' if sparse else 'dense'}")

    # Create initial state
    psi0 = create_initial_state(dim)
    print("Created initial state (equal superposition)")

    # Estimate spectral bounds
    e_max, e_min = estimate_spectral_bounds(h_matrix, n_iter=40)

    # Prepare Chebyshev evolution function
    if use_jit:
        chebyshev_evol_jit = backend.jit(
            tc.timeevol.chebyshev_evol, static_argnums=(3, 4, 5)
        )
        chebyshev_function = chebyshev_evol_jit
        print("Using JIT compilation")
    else:
        chebyshev_function = tc.timeevol.chebyshev_evol
        print("Not using JIT compilation")

    # Perform Chebyshev evolution
    print("\nPerforming Chebyshev evolution...")
    print("--- Testing single time evolution ---")
    k_estimate = tc.timeevol.estimate_k(time, (e_max, e_min))
    m_estimate = tc.timeevol.estimate_M(time, (e_max, e_min), k=k_estimate)
    print(f"Required M (estimated): {m_estimate}")
    print(f"Required k (estimated): {k_estimate}")

    psi_cheby = chebyshev_function(
        h_matrix,
        psi0,
        t=time,
        spectral_bounds=(e_max + 0.1, e_min - 0.1),
        k=k_estimate,
        M=m_estimate,
    )

    norm = tc.backend.norm(psi_cheby)
    print(f"Norm of evolved state: {norm}")

    # Compare with exact evolution
    print("\nComparing with exact evolution...")
    fidelity = compare_with_exact_evolution(h_matrix, psi0, psi_cheby, time)
    print(f"Fidelity for t={time}: {fidelity:.8f}")


def main() -> None:
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Chebyshev time evolution example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python chebyshev_evol.py --num_sites 8 --time 500.0
  python chebyshev_evol.py --backend jax --jit
  python chebyshev_evol.py --dense --backend jax --jit
        """,
    )

    parser.add_argument(
        "--num_sites",
        type=int,
        default=8,
        help="Number of sites in the system (default: 8)",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=500.0,
        help="Evolution time (default: 500.0)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "jax", "tensorflow", "pytorch"],
        help="Backend selection (default: numpy)",
    )

    parser.add_argument(
        "--dense",
        dest="sparse",
        action="store_false",
        help="Use dense matrices instead of sparse",
    )

    parser.add_argument(
        "--sparse",
        dest="sparse",
        action="store_true",
        help="Use sparse matrices (default)",
    )

    parser.add_argument(
        "--jit",
        action="store_true",
        help="Enable JIT compilation (only works with JAX backend)",
    )

    parser.set_defaults(sparse=True)

    args = parser.parse_args()

    run_chebyshev_evolution(
        num_sites=args.num_sites,
        time=args.time,
        backend_name=args.backend,
        sparse=args.sparse,
        use_jit=args.jit,
    )


if __name__ == "__main__":
    main()
