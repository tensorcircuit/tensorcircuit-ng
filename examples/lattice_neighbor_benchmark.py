"""
An example script to benchmark neighbor-finding algorithms in CustomizeLattice.

This script demonstrates the performance difference between the KDTree-based
neighbor search and a baseline all-to-all distance matrix method.
As shown by the results, the KDTree approach offers a significant speedup,
especially when calculating for a large number of neighbor shells (large max_k).
"""

import timeit
from typing import Any, Dict, List


def run_benchmark() -> None:
    """
    Executes the benchmark test and prints the results in a formatted table.
    """
    # --- Benchmark Parameters ---
    # A list of lattice sizes (N = number of sites) to test
    site_counts: List[int] = [10, 50, 100, 200, 500, 1000, 1500, 2000]

    # Use a large k to better showcase the performance of KDTree in
    # finding multiple neighbor shells, as suggested by the maintainer.
    max_k: int = 2000

    # Reduce the number of runs to keep the total benchmark time reasonable,
    # especially with a large max_k.
    number_of_runs: int = 3
    # --------------------------

    results: List[Dict[str, Any]] = []

    print("=" * 75)
    print("Starting neighbor finding benchmark for CustomizeLattice...")
    print(f"Parameters: max_k={max_k}, number_of_runs={number_of_runs}")
    print("=" * 75)
    print(
        f"{'Sites (N)':>10} | {'KDTree Time (s)':>18} | {'Baseline Time (s)':>20} | {'Speedup':>10}"
    )
    print("-" * 75)

    for n_sites in site_counts:
        # Prepare the setup code for timeit.
        # This code generates a random lattice and is executed before timing begins.
        # We use a fixed seed to ensure the coordinates are the same for both tests.
        setup_code = f"""
import numpy as np
from tensorcircuit.templates.lattice import CustomizeLattice

np.random.seed(42)
coords = np.random.rand({n_sites}, 2)
ids = list(range({n_sites}))
lat = CustomizeLattice(dimensionality=2, identifiers=ids, coordinates=coords)
"""
        # Define the Python statements to be timed.
        stmt_kdtree = f"lat._build_neighbors(max_k={max_k})"
        stmt_baseline = f"lat._build_neighbors_by_distance_matrix(max_k={max_k})"

        try:
            # Execute the timing. timeit returns the total time for all runs.
            time_kdtree = (
                timeit.timeit(stmt=stmt_kdtree, setup=setup_code, number=number_of_runs)
                / number_of_runs
            )
            time_baseline = (
                timeit.timeit(
                    stmt=stmt_baseline, setup=setup_code, number=number_of_runs
                )
                / number_of_runs
            )

            # Calculate and store results, handling potential division by zero.
            speedup = time_baseline / time_kdtree if time_kdtree > 0 else float("inf")
            results.append(
                {
                    "n_sites": n_sites,
                    "time_kdtree": time_kdtree,
                    "time_baseline": time_baseline,
                    "speedup": speedup,
                }
            )
            print(
                f"{n_sites:>10} | {time_kdtree:>18.6f} | {time_baseline:>20.6f} | {speedup:>9.2f}x"
            )

        except Exception as e:
            print(f"An error occurred at N={n_sites}: {e}")
            break

    print("-" * 75)
    print("Benchmark complete.")


if __name__ == "__main__":
    run_benchmark()
