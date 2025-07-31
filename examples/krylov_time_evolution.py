"""Comprehensive Krylov evolution analysis tool"""

import argparse
import sys
import time
import traceback
from typing import Any, Tuple, List, Dict, Union
import numpy as np
import tensorcircuit as tc
from tensorcircuit.timeevol import krylov_evol

tc.set_dtype("complex128")


def compute_magnetization(psi: Any, backend: Any = None) -> float:
    """Calculate total magnetization (sum of Z operators).

    This function can be used as a callback for krylov_evol.
    When used as a callback, it only receives the psi parameter.
    When called directly, it can receive both psi and backend parameters.
    """
    n = int(np.log2(psi.shape[0]))
    c = tc.Circuit(n, inputs=psi)
    return np.real(np.sum([(-1) ** i * c.expectation_ps(z=[i]) for i in range(n)]))


def exact_evolution(
    hamiltonian: Any, initial_state: Any, t: float, backend: Any
) -> Any:
    """Perform exact time evolution using matrix exponential."""
    # Calculate exp(-i*H*t)
    # For JAX backend, special handling of sparse matrices is needed
    try:
        exp_hamiltonian_t = backend.expm(-1j * hamiltonian * t)

        # Apply to initial state
        if backend.is_sparse(hamiltonian):
            evolved_state = backend.sparse_dense_matmul(
                exp_hamiltonian_t, initial_state
            )
        else:
            evolved_state = backend.tensordot(exp_hamiltonian_t, initial_state, axes=1)
        return evolved_state
    except Exception:
        # For JAX backend sparse matrix issues, try converting to dense matrix
        try:
            dense_hamiltonian = backend.to_dense(hamiltonian)
            exp_hamiltonian_t = backend.expm(-1j * dense_hamiltonian * t)
            evolved_state = backend.tensordot(exp_hamiltonian_t, initial_state, axes=1)
            return evolved_state
        except Exception as exc:
            # If still failing, re-raise the original exception
            raise exc


def compute_fidelity(psi1: Any, psi2: Any, backend: Any) -> float:
    """Calculate fidelity between two quantum states."""
    # Fidelity F = |<psi1|psi2>|^2
    # First ensure state vectors are normalized
    psi1_array = backend.numpy(psi1) if hasattr(backend, "numpy") else psi1
    psi2_array = backend.numpy(psi2) if hasattr(backend, "numpy") else psi2

    # Normalize state vectors
    norm1 = np.sqrt(np.sum(np.abs(psi1_array) ** 2))
    norm2 = np.sqrt(np.sum(np.abs(psi2_array) ** 2))

    if norm1 > 0:
        psi1_array = psi1_array / norm1
    if norm2 > 0:
        psi2_array = psi2_array / norm2

    # Calculate inner product
    inner_product = np.sum(np.conj(psi1_array) * psi2_array)
    # Fidelity should be the square of the absolute value of inner product
    fidelity = np.abs(inner_product) ** 2
    # Ensure fidelity is within reasonable range
    fidelity = np.clip(fidelity, 0.0, 1.0)
    return float(fidelity)


def parse_list(string: str) -> List[float]:
    """Parse comma-separated list of numbers."""
    try:
        # Remove spaces and split
        items = string.replace(" ", "").split(",")
        return [float(x) for x in items if x]  # Filter out empty strings
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid list format: {string}") from exc


def run_comprehensive_analysis(
    num_sites: int = 10,
    time_points: Union[List[float], None] = None,
    subspace_dims: Union[List[int], None] = None,
    hz: float = 0.5,
    hx: float = 0.3,
    hy: float = 0.2,
    verbose: bool = True,
    backend_name: str = "numpy",
    use_jit: bool = False,
    scan_impl: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run comprehensive analysis showing fidelity, magnetization error and runtime
    for different t and m values.

    Parameters:
    - num_sites: Number of sites in the system
    - time_points: List of time points
    - subspace_dims: List of Krylov subspace dimensions
    - hz, hx, hy: Magnetic field components
    - verbose: Whether to show detailed output
    - backend_name: Backend name
    - use_jit: Whether to use JIT compilation
    - scan_impl: Whether to use scan implementation in krylov_evol
    """
    if time_points is None:
        time_points = [1.0, 2.0, 5.0]

    if subspace_dims is None:
        subspace_dims = [20, 50, 100]

    # Set backend
    tc.set_backend(backend_name)
    backend = tc.backend

    # If JIT is enabled, create JIT version of krylov_evol function
    if use_jit:
        # Create wrapper function to avoid passing backend parameter to JIT
        def krylov_evol_wrapper(h, psi0, tlist, m):
            return krylov_evol(h, psi0, tlist, m, scan_impl=scan_impl)

        # JIT compile function (static parameter is m)
        krylov_evol_jit = backend.jit(krylov_evol_wrapper, static_argnums=(3,))
        # Use JIT version of function
        krylov_function = krylov_evol_jit
        jit_info = " (using JIT compilation)"
    else:
        # Use regular version of function
        krylov_function = lambda h, psi0, tlist, m: krylov_evol(
            h, psi0, tlist, m, scan_impl=scan_impl
        )
        jit_info = ""

    if verbose:
        print(
            f"Krylov evolution comprehensive analysis (using {backend_name} backend{jit_info})"
        )
        print("==================")
        print(f"System size: {num_sites} sites")
        print(f"Time points: {time_points}")
        print(f"Krylov subspace dimensions: {subspace_dims}")
        print(f"Magnetic field: hz={hz}, hx={hx}, hy={hy}")
        print("=" * 80)

    # Create Heisenberg Hamiltonian
    graph = tc.templates.graphs.Line1D(num_sites, pbc=False)
    heisenberg_ham = tc.quantum.heisenberg_hamiltonian(
        graph, hzz=1, hyy=1, hxx=1, hz=hz, hx=hx, hy=hy, sparse=True
    )

    # Convert to backend-agnostic sparse COO format
    # Handle differences between backends
    try:
        if backend_name in ["jax", "tensorflow", "pytorch"]:
            # For these backends, use sparse matrix directly
            hamiltonian_sparse = heisenberg_ham
        else:
            hamiltonian_sparse = backend.coo_sparse_matrix_from_numpy(heisenberg_ham)
    except Exception as exc:
        # If conversion fails, use original matrix
        if verbose:
            print(
                f"Warning: Unable to convert to sparse format ({str(exc)}), using original matrix"
            )
        hamiltonian_sparse = heisenberg_ham

    # For TensorFlow backend, convert to dense matrix to avoid issues
    if backend_name == "tensorflow":
        try:
            hamiltonian_sparse = backend.to_dense(hamiltonian_sparse)
            if verbose:
                print("Converted to dense matrix for TensorFlow compatibility")
        except Exception as exc:
            if verbose:
                print(f"Unable to convert to dense matrix: {str(exc)}")

    # For systems with 12 sites or fewer, try exact diagonalization
    if num_sites <= 12:
        try:
            if backend_name in ["jax", "tensorflow", "pytorch"]:
                # For these backends, use sparse matrix directly
                hamiltonian_dense = heisenberg_ham
            else:
                hamiltonian_dense = backend.to_dense(hamiltonian_sparse)
        except Exception as exc:
            # If conversion fails (e.g. out of memory), skip exact diagonalization
            hamiltonian_dense = None
            if verbose:
                print(
                    f"Warning: Unable to convert to dense matrix ({str(exc)}), skipping ED comparison"
                )
    else:
        hamiltonian_dense = None
        if verbose:
            print("Warning: System too large, skipping ED comparison")

    if verbose:
        hamiltonian_dim = (
            backend.sparse_shape(hamiltonian_sparse)[0]
            if backend.is_sparse(hamiltonian_sparse)
            else hamiltonian_sparse.shape[0]
        )
        print(f"Hilbert space dimension: {hamiltonian_dim}")

    # Create Neel initial state
    circuit = tc.Circuit(num_sites)
    circuit.x([2 * i for i in range(num_sites // 2)])
    initial_state = circuit.state()

    # Store results
    results: Dict = {}
    exact_results: Dict = {}

    # Evolve for each time point
    for t in time_points:
        if verbose:
            print(f"\nTime t = {t}")
            print("-" * 40)

        results[t] = {}

        # Exact evolution (if possible)
        if hamiltonian_dense is not None:
            try:
                start_time = time.time()
                exact_state = exact_evolution(
                    hamiltonian_dense, initial_state, t, backend
                )
                exact_time = time.time() - start_time

                exact_magnetization = compute_magnetization(exact_state, backend)
                exact_results[t] = {
                    "state": exact_state,
                    "magnetization": exact_magnetization,
                    "time": exact_time,
                }

                if verbose:
                    print(
                        f"  Exact evolution completed, time: {exact_time:.4f} seconds"
                    )
                    print(f"  Magnetization: {exact_magnetization.real:8.6f}")
            except Exception as exc:
                if verbose:
                    print(f"  Exact evolution failed: {str(exc)}")

        # Krylov evolution (different m values)
        reference_result = None
        reference_m = None

        for m in subspace_dims:

            start_time = time.time()
            krylov_result = krylov_function(
                hamiltonian_sparse,
                initial_state,
                [t],
                int(m),  # Ensure m is integer
            )
            print(krylov_result[0, 0])
            krylov_time = time.time() - start_time

            # Extract result (krylov_result is an array containing a single time point)
            if hasattr(krylov_result, "shape") and len(krylov_result.shape) > 1:
                evolved_state = krylov_result[
                    0
                ]  # Take the first (and only) time point result
            else:
                evolved_state = krylov_result

            # Calculate magnetization
            magnetization = compute_magnetization(evolved_state, backend)

            results[t][m] = {
                "state": evolved_state,
                "magnetization": magnetization,
                "time": krylov_time,
            }

            if verbose:
                print(
                    f"  m = {m:3d}: Time {krylov_time:.4f}s, Magnetization {magnetization.real:8.6f}"
                )

            # Set reference result (using largest m value, only when no exact result)
            # Fix: Only use Krylov result as reference when no exact result available
            if reference_result is None and t not in exact_results:
                reference_result = results[t][m]
                reference_m = m

        # Compare with exact results (if available)
        if t in exact_results and reference_result:
            exact_state = exact_results[t]["state"]
            exact_mag = exact_results[t]["magnetization"]
            krylov_state = reference_result["state"]
            krylov_mag = reference_result["magnetization"]

            # Calculate fidelity
            fidelity = compute_fidelity(exact_state, krylov_state, backend)

            # Calculate magnetization difference
            mag_diff = abs(exact_mag.real - krylov_mag.real)

            if verbose:
                print(f"  Comparison with exact result (m={reference_m}):")
                print(f"    Fidelity: {fidelity:.6f}")
                print(f"    Magnetization difference: {mag_diff:.2e}")

        # Convergence analysis
        # Fix: Prioritize using exact results as reference, if not available use largest m value
        reference_state = None
        reference_mag = None
        reference_label = "None"

        if t in exact_results:
            # Use exact result as reference
            reference_state = exact_results[t]["state"]
            reference_mag = exact_results[t]["magnetization"]
            reference_label = "Exact"
            if verbose:
                print(f"t={t:4.1f} (Reference: {reference_label}):")
                print(f"  Magnetization = {reference_mag.real:8.6f}")
        elif reference_result:
            # If no exact result, use largest m value as reference
            reference_state = reference_result["state"]
            reference_mag = reference_result["magnetization"]
            reference_label = f"m={reference_m}"
            if verbose:
                print(f"t={t:4.1f} (Reference: {reference_label}):")
                print(f"  Magnetization = {reference_mag.real:8.6f}")

        if reference_state is not None and verbose:
            # Compare with other m values
            print(f"  {'m':<6} {'Fidelity vs ref':<15} {'Mag Diff vs ref':<15}")
            print(f"  {'-'*6} {'-'*15} {'-'*15}")
            for m_value in subspace_dims:
                if m_value in results[t] and "state" in results[t][m_value]:
                    current_state = results[t][m_value]["state"]
                    current_mag = results[t][m_value]["magnetization"]

                    # Fidelity vs reference result
                    fidelity_vs_ref = compute_fidelity(
                        reference_state, current_state, backend
                    )

                    # Magnetization difference vs reference result
                    mag_diff_vs_ref = abs(current_mag.real - reference_mag.real)

                    print(
                        f"  {m_value:<6} {fidelity_vs_ref:<15.6f} {mag_diff_vs_ref:<15.2e}"
                    )
        elif verbose:
            print(f"t={t:4.1f}: No successful results for convergence analysis")

    return results, exact_results


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Krylov evolution comprehensive analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python comprehensive_analysis.py --num_sites 12 --time_points "5,15" --subspace_dims "50,100,300"
  python comprehensive_analysis.py --num_sites 10 --time_points "1,2,5" --subspace_dims "20,50,100"
  python comprehensive_analysis.py  # Use default parameters
        """,
    )
    parser.add_argument(
        "--num_sites",
        type=int,
        default=10,
        help="Number of sites in the system (default: 10)",
    )
    parser.add_argument(
        "--time_points",
        type=parse_list,
        default="1,2,5",
        help="Comma-separated list of time points (default: 0.5,1,2,5)",
    )
    parser.add_argument(
        "--subspace_dims",
        type=parse_list,
        default="20,50,100",
        help="Comma-separated list of Krylov subspace dimensions (default: 20,30,50,100)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=0.5,
        help="Longitudinal magnetic field (default: 0.5)",
    )
    parser.add_argument(
        "--hx",
        type=float,
        default=0.3,
        help="Transverse magnetic field x-component (default: 0.3)",
    )
    parser.add_argument(
        "--hy",
        type=float,
        default=0.2,
        help="Transverse magnetic field y-component (default: 0.2)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Quiet mode, only output final results"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "jax", "tensorflow", "pytorch"],
        help="Backend selection (default: numpy)",
    )
    parser.add_argument("--jit", action="store_true", help="Enable JIT compilation")
    parser.add_argument(
        "--scan_impl",
        action="store_true",
        help="Use scan implementation in krylov_evol",
    )

    args = parser.parse_args()

    # Add global exception handling
    try:
        # Run comprehensive analysis
        run_comprehensive_analysis(
            num_sites=args.num_sites,
            time_points=args.time_points,
            subspace_dims=[int(m) for m in args.subspace_dims],
            hz=args.hz,
            hx=args.hx,
            hy=args.hy,
            verbose=not args.quiet,
            backend_name=args.backend,
            use_jit=args.jit,
            scan_impl=args.scan_impl,
        )
    except KeyboardInterrupt:
        print("\nUser interrupted program execution")
        sys.exit(1)
    except Exception as exc:
        print(f"\nSerious error occurred: {str(exc)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
