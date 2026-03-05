import time
import numpy as np
from tensorcircuit.templates.lattice import CustomizeLattice


def generate_random_lattice(num_sites, dimensionality=2):
    np.random.seed(42)
    coords = np.random.rand(num_sites, dimensionality)
    identifiers = list(range(num_sites))
    return CustomizeLattice(dimensionality, identifiers, coords)


def benchmark_kdtree(lattice, max_k=10, iterations=10):
    times = []
    for _ in range(iterations):
        lattice._reset_computations()
        start = time.perf_counter()
        lattice._build_neighbors(max_k=max_k, use_kdtree=True)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"Benchmark Results (sites={lattice.num_sites}, max_k={max_k}):")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Min time:     {min_time:.6f} seconds")
    print(f"Max time:     {max_time:.6f} seconds")
    return avg_time


if __name__ == "__main__":
    import sys

    num_sites = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    # Test for smaller lattice branch (num_sites <= 100)
    print("--- Small Lattice ---")
    lattice_small = generate_random_lattice(100)
    benchmark_kdtree(lattice_small)

    # Test for larger lattice branch (num_sites > 100)
    print("\n--- Large Lattice ---")
    lattice_large = generate_random_lattice(num_sites)
    benchmark_kdtree(lattice_large)
