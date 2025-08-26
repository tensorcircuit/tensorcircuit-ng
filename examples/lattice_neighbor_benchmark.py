"""
Benchmark: Compare neighbor-building time between KDTree and distance-matrix
methods in CustomizeLattice for varying lattice sizes.
"""

import argparse
import csv
import time
from typing import Iterable, List, Tuple, Optional
import logging

import numpy as np

# Silence verbose infos from the library during benchmarks

logging.basicConfig(level=logging.WARNING)

from tensorcircuit.templates.lattice import CustomizeLattice


def run_once(
    n: int, d: int, max_k: int, repeats: int, seed: int
) -> Tuple[float, float]:
    """Run one size point and return (time_kdtree, time_matrix)."""
    rng = np.random.default_rng(seed)
    ids = list(range(n))

    # Collect times for each repeat with different random coordinates
    kdtree_times: List[float] = []
    matrix_times: List[float] = []

    for _ in range(repeats):
        # Generate different coordinates for each repeat
        coords = rng.random((n, d), dtype=float)
        lat = CustomizeLattice(dimensionality=d, identifiers=ids, coordinates=coords)

        # KDTree path - single measurement
        t0 = time.perf_counter()
        lat._build_neighbors(max_k=max_k, use_kdtree=True)
        kdtree_times.append(time.perf_counter() - t0)

        # Distance-matrix path - single measurement
        t0 = time.perf_counter()
        lat._build_neighbors(max_k=max_k, use_kdtree=False)
        matrix_times.append(time.perf_counter() - t0)

    return float(np.mean(kdtree_times)), float(np.mean(matrix_times))


def parse_sizes(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def format_row(n: int, t_kdtree: float, t_matrix: float) -> str:
    speedup = (t_matrix / t_kdtree) if t_kdtree > 0 else float("inf")
    return f"{n:>8} | {t_kdtree:>12.6f} | {t_matrix:>14.6f} | {speedup:>7.2f}x"


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Neighbor-building time comparison")
    p.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[128, 256, 512, 1024, 2048],
        help="Comma-separated site counts to benchmark (default: 128,256,512,1024,2048)",
    )
    p.add_argument(
        "--dims", type=int, default=2, help="Lattice dimensionality (default: 2)"
    )
    p.add_argument(
        "--max-k", type=int, default=10, help="Max neighbor shells k (default: 6)"
    )
    p.add_argument(
        "--repeats", type=int, default=5, help="Repeats per measurement (default: 5)"
    )
    p.add_argument("--seed", type=int, default=42, help="PRNG seed (default: 42)")
    p.add_argument("--csv", type=str, default="", help="Optional CSV output path")
    args = p.parse_args(list(argv) if argv is not None else None)

    print("=" * 74)
    print(
        f"Benchmark CustomizeLattice neighbor-building | dims={args.dims} max_k={args.max_k} repeats={args.repeats}"
    )
    print("=" * 74)
    print(f"{'N':>8} | {'KDTree(s)':>12} | {'DistMatrix(s)':>14} | {'Speedup':>7}")
    print("-" * 74)

    rows: List[Tuple[int, float, float]] = []
    for n in args.sizes:
        t_kdtree, t_matrix = run_once(n, args.dims, args.max_k, args.repeats, args.seed)
        rows.append((n, t_kdtree, t_matrix))
        print(format_row(n, t_kdtree, t_matrix))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["N", "time_kdtree_s", "time_distance_matrix_s", "speedup"])
            for n, t_kdtree, t_matrix in rows:
                speedup = (t_matrix / t_kdtree) if t_kdtree > 0 else float("inf")
                w.writerow([n, f"{t_kdtree:.6f}", f"{t_matrix:.6f}", f"{speedup:.2f}"])

        print("-" * 74)
        print(f"Saved CSV to: {args.csv}")

    print("-" * 74)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
