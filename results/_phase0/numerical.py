"""Phase 0 Task 9: numerical validation matrix (final-remediation Task 9).

Aggregates the numerical correctness of all four BF16 contraction routes
(planar / grouped / region_fused / cutlass_4m_single) over actual-large shapes,
>=3 seeds, 3 adversarial dynamic-range levels, {C16BF, C32F} output dtypes, against
a c64 fp32 materialized reference. Produces a fail-closed numerical_validation.csv +
.json consumed by Task 10 (gonogo) and Task 11 (manifest).

Pure functions (compute_metrics, make_inputs, apply_policy, aggregate, writers) are
GPU-free and unit-tested first. GPU route collectors import existing helpers from
cublaslt.py / region_proto.py (zero changes to those modules).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np


def compute_metrics(out, ref, signal_floor: float = 0.5) -> dict:
    """Numerical correctness of ``out`` vs c64 fp32 materialized ``ref``.

    Returns JSON-serializable scalars:
    - relative_l2: ||out-ref||_2 / max(1, ||ref||_2)
    - max_abs:     max |out-ref|
    - max_rel:     max |out-ref| / max(|ref|, signal_floor)   (signal_floor avoids div-by-0)
    - nan_inf:     any non-finite in out
    - n_elems:     out.size
    """
    out = np.asarray(out)
    ref = np.asarray(ref)
    diff = out - ref
    nan_inf = bool(not np.all(np.isfinite(out)))
    denom = np.maximum(np.abs(ref), signal_floor)
    rel_l2 = float(np.linalg.norm(diff) / max(1.0, float(np.linalg.norm(ref))))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    max_rel = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
    return {
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nan_inf": nan_inf,
        "n_elems": int(out.size),
    }


_LEVELS = ("baseline", "mixed_scale", "cancellation")


def make_inputs(level, shape, seed, ref_dtype=np.complex64):
    """Generate (A, B) for C = A @ B at a given dynamic-range level.

    shape = (M, N, K) (matches cublaslt artifacts); A is (M,K), B is (K,N). Deterministic in seed.
    - baseline: real/imag ~ N(0,1)
    - mixed_scale: per-element Bernoulli(0.5) mix of N(0, 1e2^2) and N(0, 1e-2^2)
      -> dynamic range 1e4, exposes bf16 small-magnitude loss (spec §4.2)
    - cancellation: B rows paired +- (B[2j+1] = -B[2j]); reference C has near-zero
      elements -> amplifies max_rel denominator sensitivity (spec §4.3). Requires K even.
    """
    if level not in _LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {_LEVELS}")
    M, N, K = shape
    rng = np.random.default_rng(seed)

    def complex_normal(sz, sigma):
        return (rng.standard_normal(sz) + 1j * rng.standard_normal(sz)).astype(
            ref_dtype
        ) * sigma

    if level == "baseline":
        A = complex_normal((M, K), 1.0)
        B = complex_normal((K, N), 1.0)
    elif level == "mixed_scale":
        mask_a = rng.random((M, K)) < 0.5
        big_a = complex_normal((M, K), 1e2)
        small_a = complex_normal((M, K), 1e-2)
        A = np.where(mask_a, big_a, small_a).astype(ref_dtype)
        mask_b = rng.random((K, N)) < 0.5
        big_b = complex_normal((K, N), 1e2)
        small_b = complex_normal((K, N), 1e-2)
        B = np.where(mask_b, big_b, small_b).astype(ref_dtype)
    else:  # cancellation
        if K % 2 != 0:
            raise ValueError(f"cancellation requires even K, got K={K}")
        A = complex_normal((M, K), 1.0)
        half = complex_normal((K // 2, N), 1.0)
        B = np.empty((K, N), dtype=ref_dtype)
        B[0::2] = half
        B[1::2] = -half
    return A, B


# Per route x dtype policy (spec §5). A threshold of None means "not applicable /
# diagnostic only" (e.g. max_abs for region_fused/cutlass where output scale varies
# with dynamic range). nan_inf is always enforced.
POLICIES = {
    ("planar", "C16BF"): {"relative_l2": 1e-3, "max_abs": 1e-1, "max_rel": 5e-3},
    ("planar", "C32F"): {"relative_l2": 1e-4, "max_abs": 1e-2, "max_rel": 1e-3},
    ("grouped", "C16BF"): {"relative_l2": 1e-3, "max_abs": 1e-1, "max_rel": 5e-3},
    ("grouped", "C32F"): {"relative_l2": 1e-4, "max_abs": 1e-2, "max_rel": 1e-3},
    ("region_fused", "c64"): {"relative_l2": 1e-4, "max_abs": None, "max_rel": 1e-3},
    ("cutlass_4m_single", "C16BF"): {"relative_l2": 1e-3, "max_abs": None, "max_rel": 5e-3},
}


def apply_policy(route, dtype, metrics):
    """Apply the per route x dtype policy to a metrics dict.

    Returns (verdict, reason). verdict in {"PASS","FAIL",None}: None means a required
    metric was missing (cell incomplete). nan_inf=True forces FAIL regardless of values.
    """
    # nan_inf is enforced first, before the policy-key lookup, so that a non-finite
    # output fails for *any* route/dtype cell (test_apply_policy_nan_inf_fails_any_route
    # covers region_fused + C16BF, which has no policy row).
    if metrics.get("nan_inf"):
        return "FAIL", "nan_inf=True"
    key = (route, dtype)
    if key not in POLICIES:
        return None, f"no policy for {(route, dtype)}"
    pol = POLICIES[key]
    for field, thresh in pol.items():
        if thresh is None:
            continue  # diagnostic-only field
        val = metrics.get(field)
        if val is None:
            return None, f"missing metric {field}"
        if val >= thresh:
            return "FAIL", f"{field}={val:.2e} >= {thresh:.0e}"
    return "PASS", None


_ROUTES = ("planar", "grouped", "region_fused", "cutlass_4m_single")


def aggregate(rows, expected_counts, case_hashes, legit_not_run):
    """Fail-closed aggregation -> numerical_validation.json payload (spec §7).

    rows: list of cell dicts (route, dtype, shape, level, seed, + metrics).
    expected_counts: {(route, dtype): N_expected_rows}.
    case_hashes: {hash_name: value}; any value == "MISMATCH" -> INCONCLUSIVE.
    legit_not_run: human-readable reasons for legitimate NOT_RUN (e.g. region_fused
      actual-large fused compute-bound); listed in fail_closed_reasons but do NOT
      sink overall to INCONCLUSIVE.
    """
    fail_closed_reasons = list(legit_not_run)

    hash_mismatch = any(v == "MISMATCH" for v in case_hashes.values())
    if hash_mismatch:
        fail_closed_reasons.append("case-binding hash mismatch")

    per_route = []
    statuses = []
    for route in _ROUTES:
        # group rows by dtype for this route
        dtypes_for_route = sorted({r["dtype"] for r in rows if r["route"] == route})
        route_cells = [r for r in rows if r["route"] == route]
        verdicts = []
        for dtype in dtypes_for_route:
            expected = expected_counts.get((route, dtype), 0)
            present = sum(1 for r in route_cells if r["dtype"] == dtype)
            if present < expected:
                verdicts.append("UNKNOWN")
                continue
            for r in route_cells:
                if r["dtype"] != dtype:
                    continue
                v, _ = apply_policy(route, dtype, r)
                verdicts.append(v or "UNKNOWN")
        if not verdicts:
            criterion = "NOT_RUN"
        elif any(v == "FAIL" for v in verdicts):
            criterion = "FAIL"
        elif any(v == "UNKNOWN" for v in verdicts):
            criterion = "UNKNOWN"
        else:
            criterion = "PASS"
        statuses.append(criterion)
        per_route.append({"route": route, "criterion": criterion, "n_cells": len(route_cells)})

    if hash_mismatch or any(s == "UNKNOWN" for s in statuses):
        overall = "INCONCLUSIVE"
    elif any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif all(s in ("PASS", "NOT_RUN") for s in statuses) and any(s == "PASS" for s in statuses):
        overall = "PASS"
    else:
        overall = "INCONCLUSIVE"  # all NOT_RUN, nothing proven

    return {
        "schema_version": "numerical-validation-v1",
        "case_binding": case_hashes,
        "per_route": per_route,
        "overall_numerical_status": overall,
        "fail_closed_reasons": fail_closed_reasons,
    }


# ---------------------------------------------------------------------------
# Task 5: matrix constants + CSV/JSON writers (spec §6, §2)
# ---------------------------------------------------------------------------

OUT_DIR = "results/phase0"

SHAPES = [
    # (M, N, K) order — matches cublaslt_full_matrix.csv / cublaslt_planar_accuracy.csv
    (262144, 64, 4),
    (8388608, 2, 2),
    (4194304, 4, 4),
    (16384, 1024, 1024),
    (2097152, 8, 8),
    (524288, 32, 32),
    (262144, 64, 64),
    (1048576, 16, 16),
]
# real-gemm actual-large = aligned=1 subset (spec §2): M,N,K all 16-aligned.
REAL_GEMM_SHAPES = [(16384, 1024, 1024), (524288, 32, 32), (262144, 64, 64), (1048576, 16, 16)]
LEVELS = ("baseline", "mixed_scale", "cancellation")
SEEDS = (0, 1, 2)
DTYPES_BY_ROUTE = {
    "planar": ("C16BF", "C32F"),
    "grouped": ("C16BF", "C32F"),
    "region_fused": ("c64",),
    "cutlass_4m_single": ("C16BF",),
}

_CSV_COLUMNS = [
    "route", "M", "N", "K", "out_dtype", "dynamic_range_level", "seed",
    "relative_l2", "max_abs", "max_rel", "nan_inf", "n_elems",
    "policy_pass", "reference_dtype", "source_hash",
]


def source_hash(route, dtype, shape, level, seed):
    key = f"{route}|{dtype}|{shape}|{level}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def write_csv(path, rows):
    # Tolerant to partial rows (e.g. minimal test rows that only carry a subset
    # of fields); production collectors pass the full schema. Missing numeric
    # fields render as empty CSV cells rather than raising KeyError.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_COLUMNS)
        for r in rows:
            shape = r.get("shape")
            if shape is not None:
                M, N, K = shape
            else:
                M = r.get("M", 0)
                N = r.get("N", 0)
                K = r.get("K", 0)
            route = r.get("route", "")
            dtype = r.get("dtype", "")
            level = r.get("level", "")
            seed = r.get("seed", "")
            rel_l2 = r.get("relative_l2")
            max_abs = r.get("max_abs")
            max_rel = r.get("max_rel")
            sh = r.get("source_hash")
            if not sh:
                sh = source_hash(route, dtype, shape or (), level, seed)
            w.writerow([
                route, M, N, K, dtype, level, seed,
                f"{rel_l2:.6e}" if rel_l2 is not None else "",
                f"{max_abs:.6e}" if max_abs is not None else "",
                f"{max_rel:.6e}" if max_rel is not None else "",
                int(bool(r.get("nan_inf", False))),
                r.get("n_elems", 0),
                int(r.get("policy_pass", 0)),
                r.get("reference_dtype", "c64"),
                sh,
            ])


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
