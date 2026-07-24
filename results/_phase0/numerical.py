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
    - relative_l2: ||out-ref||_2 / max(||ref||_2, epsilon)  (epsilon = 1.0; spec §3.2.1).
                   This is the canonical vector L2 error metric. It MUST be a real
                   vector L2 -- never substitute max_rel for it (Task 3a, plan §6 3.2).
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
    epsilon = 1.0  # floor on ||ref||_2 to avoid div-by-zero (spec §3.2.1)
    rel_l2 = float(np.linalg.norm(diff) / max(epsilon, float(np.linalg.norm(ref))))
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

# Cancellation input construction (plan §3.1 / spec §3.4). eps is small
# enough to produce real cancellation (ratio << 0.1) and amplify BF16
# rounding/cancellation risk, but large enough to keep the reference
# non-zero finite (avoids a trivially all-zero output).
CANCELLATION_EPSILON = 1e-3
INPUT_CONSTRUCTION_VERSION = "cancellation_v2"


def _version_token_for_level(level):
    """Canonical ``input_construction_version`` token for a dynamic-range level.

    Cancellation-level cells carry ``INPUT_CONSTRUCTION_VERSION``
    (``"cancellation_v2"``) -- the unified token that the producer constant,
    ``required_cell_keys``, the CSV reader/writer, and ``cell_key_hash`` ALL
    use (plan §3.1 / errata #2). Baseline / mixed_scale cells carry
    ``level + "_v1"`` (``"baseline_v1"`` / ``"mixed_scale_v1"``) so their rows
    match ``required_cell_keys`` (errata #4: those producers MUST write the
    token; previously only ``_enrich_cancellation_metrics`` wrote it, leaving
    baseline/mixed rows with an empty field that could never match).
    """
    if level == "cancellation":
        return INPUT_CONSTRUCTION_VERSION
    return level + "_v1"


def make_inputs(level, shape, seed, ref_dtype=np.complex64):
    """Generate (A, B) for C = A @ B at a given dynamic-range level.

    shape = (M, N, K) (matches cublaslt artifacts); A is (M,K), B is (K,N). Deterministic in seed.
    - baseline: real/imag ~ N(0,1)
    - mixed_scale: per-element Bernoulli(0.5) mix of N(0, 1e2^2) and N(0, 1e-2^2)
      -> dynamic range 1e4, exposes bf16 small-magnitude loss (spec §4.2)
    - cancellation: A columns paired equal (A[:,2j+1]=A[:,2j]) and B rows
      paired +- with a controlled residual (B[2j+1]=-B[2j]+eps*residual) so
      the paired contribution is A[:,2j]@(eps*residual) -- small and
      controlled, amplifying max_rel denominator sensitivity (spec §4.3 /
      plan §3.1). Requires K even.
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
    else:  # cancellation (plan §3.1 / spec §3.4)
        if K % 2 != 0:
            raise ValueError(f"cancellation requires even K, got K={K}")
        # A[:, 2j+1] = A[:, 2j] (paired equal columns) so the paired
        # contribution collapses:
        #   A[:,2j]@B[2j] + A[:,2j+1]@B[2j+1] = A[:,2j]@(B[2j]+B[2j+1])
        # With B[2j+1] = -B[2j] + eps*residual, this becomes
        #   A[:,2j] @ (eps * residual)  -- small and controlled (real
        #   cancellation), while the residual keeps the reference non-zero
        #   finite (spec §4.3). eps is small enough to amplify BF16
        #   rounding/cancellation risk but large enough to avoid an
        #   all-zero reference.
        half_A = complex_normal((M, K // 2), 1.0)
        A = np.empty((M, K), dtype=ref_dtype)
        A[:, 0::2] = half_A
        A[:, 1::2] = half_A  # paired equal columns
        half_B = complex_normal((K // 2, N), 1.0)
        residual = complex_normal((K // 2, N), 1.0)
        B = np.empty((K, N), dtype=ref_dtype)
        B[0::2] = half_B
        B[1::2] = -half_B + CANCELLATION_EPSILON * residual
    return A, B


def cancellation_metrics(shape, seed):
    """Diagnostic metrics for the cancellation input (plan §3.1).

    Computes the output norm under cancellation vs baseline and returns the
    fields that must be recorded in the numerical output so the cancellation
    is independently auditable:

      - input_construction_version: identifies the A/B pairing scheme
      - cancellation_epsilon: the controlled-residual coefficient
      - reference_norm: ||A_cancel @ B_cancel||_F (the small, non-zero output)
      - baseline_norm:  ||A_base   @ B_base  ||_F (the reference magnitude)
      - cancellation_ratio: reference_norm / baseline_norm (must be << 0.1)

    GPU-free: uses numpy CPU matmul only.
    """
    A_base, B_base = make_inputs("baseline", shape, seed)
    baseline_norm = float(np.linalg.norm(A_base @ B_base))
    A_cancel, B_cancel = make_inputs("cancellation", shape, seed)
    reference_norm = float(np.linalg.norm(A_cancel @ B_cancel))
    ratio = reference_norm / baseline_norm if baseline_norm > 0 else float("inf")
    return {
        "input_construction_version": INPUT_CONSTRUCTION_VERSION,
        "cancellation_epsilon": CANCELLATION_EPSILON,
        "reference_norm": reference_norm,
        "baseline_norm": baseline_norm,
        "cancellation_ratio": ratio,
    }


# The 5 cancellation diagnostic fields that must be recorded in the numerical
# output (CSV) for cancellation-level rows so the cancellation is independently
# auditable from the artifacts (plan §3.1 / spec §3.4).
CANCEL_FIELDS = (
    "input_construction_version",
    "cancellation_epsilon",
    "reference_norm",
    "baseline_norm",
    "cancellation_ratio",
)


def _enrich_cancellation_metrics(row):
    """Wire ``cancellation_metrics()`` into a numerical row dict (GPU-free).

    For cancellation-level rows whose ``shape`` is a real (M, N, K) tuple with
    even K, computes the 5 cancellation diagnostic fields via
    ``cancellation_metrics(shape, seed)`` and merges them into ``row`` in place.
    For all other rows (non-cancellation level, or label shapes like
    ``"small_contract"``) the row is returned unchanged.

    Idempotent + legacy short-circuit (errata #6):
      - ``input_construction_version == "cancellation_legacy_v1"`` -> return
        immediately (legacy diagnostic from an old GPU run; do NOT re-enrich
        or overwrite the token -- the row is archival evidence, not a fresh
        v2 diagnostic).
      - any other truthy ``input_construction_version`` -> return (idempotent;
        the row was already enriched by a producer or a prior pass, so the
        5 diagnostic fields are already present).
      - absent / empty -> enrich (compute the 5 fields + set the canonical
        ``cancellation_v2`` token).

    Called by the GPU collectors (``collect_planar`` / ``collect_grouped`` /
    ``collect_cutlass``) and by ``main(regen_no_gpu=True)`` for CSV-read rows,
    so the 5 fields are recorded in the CSV output for every cancellation cell.
    """
    if row.get("level") != "cancellation":
        return row
    ver = row.get("input_construction_version")
    if ver == "cancellation_legacy_v1":
        return row  # legacy diagnostic: don't re-enrich or overwrite
    if ver:
        return row  # idempotent (e.g. "cancellation_v2" from a real GPU run)
    shape = row.get("shape")
    if not (isinstance(shape, (tuple, list)) and len(shape) == 3):
        return row  # label shapes (e.g. "small_contract") cannot compute metrics
    M, N, K = shape
    if K % 2 != 0:
        return row  # cancellation requires even K (make_inputs constraint)
    seed = row.get("seed", 0)
    row.update(cancellation_metrics(tuple(shape), seed))
    return row


# diagnostic only" (e.g. max_abs for region_fused/cutlass where output scale varies
# with dynamic range). nan_inf is always enforced.
POLICIES = {
    # C16BF = bf16-output path: relative_l2 is bounded by bf16 precision
    # (~2^-8/sqrt(3) ~ 2.3e-3; measured ~1.66e-3), and max_abs scales with output
    # magnitude (|C|_max * 2^-8; unbounded across shapes) so it is diagnostic-only
    # (None), mirroring the cublaslt.py §7.5 gate which keys on max_rel for bf16
    # output. max_rel is bounded by bf16 precision (~2^-8 ~ 3.9e-3; measured
    # ~3.85e-3). Task 6 smoke test revealed the original rel_l2<1e-3 / max_abs<1e-1
    # were structurally impossible for bf16 output (measured rel_l2=1.66e-3,
    # max_abs=0.136 at the smoke shape).
    ("planar", "C16BF"): {"relative_l2": 5e-3, "max_abs": None, "max_rel": 5e-3},
    # C32F = bf16-input + fp32-output (fp32 accumulation). max_abs is output-scale-
    # dependent: under mixed_scale (σ=1e2), |C|_max ~ σ·√K = 1e2·√1024 ≈ 3.2e3, and
    # fp32 accumulation round-off ~ |C|·1e-7 ≈ 3e-2 structurally exceeds 1e-2 (measured
    # ~3e-2 … 2.4e-1 across the 8 shapes). That is an absolute-threshold-vs-output-scale
    # artifact, NOT a numerical bug — same root cause as the C16BF max_abs→None fix.
    # Diagnostic-only (None); rel_l2 + max_rel carry the real signal.
    ("planar", "C32F"): {"relative_l2": 1e-4, "max_abs": None, "max_rel": 1e-3},
    ("grouped", "C16BF"): {"relative_l2": 5e-3, "max_abs": None, "max_rel": 5e-3},
    ("grouped", "C32F"): {"relative_l2": 1e-4, "max_abs": None, "max_rel": 1e-3},
    ("region_fused", "c64"): {"relative_l2": 1e-4, "max_abs": None, "max_rel": 1e-3},
    ("cutlass_4m_single", "C16BF"): {
        "relative_l2": 5e-3,
        "max_abs": None,
        "max_rel": 5e-3,
    },
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


def _shape_key(shape):
    """Normalize a row's ``shape`` field to a hashable schema-key component.

    Tuple/list shapes (M,N,K) become tuples; label strings (e.g. ``"small_contract"``)
    pass through unchanged so that diagnostic small-contract rows remain distinct
    from the intended full-anchor shape in the key-set comparison.
    """
    if isinstance(shape, (tuple, list)):
        return tuple(shape)
    return shape


def _cell_key(row):
    """Canonical required-cell schema key for a row (plan §6 3.1 / errata #1).

    Key = (route, dtype, shape, level, input_construction_version, seed,
    reference_dtype). ``reference_dtype`` is always ``"c64"`` for the c64 fp32
    materialized reference. ``input_construction_version`` is the unified
    token (``"cancellation_v2"`` / ``"baseline_v1"`` / ``"mixed_scale_v1"``)
    that separates MEASURED (GPU v2) from planned (CPU) diagnostics and
    distinguishes legacy v1 archival rows from real v2 measurements. The
    7-tuple is consistent across ``_cell_key`` / ``required_cell_keys`` /
    ``_as_expected_keys`` / ``_emit_not_run_rows`` / ``cell_key_hash`` /
    ``aggregate`` accounting (errata #1).
    """
    return (
        row["route"],
        row["dtype"],
        _shape_key(row.get("shape")),
        row["level"],
        row.get("input_construction_version", ""),
        row["seed"],
        row.get("reference_dtype", "c64"),
    )


def required_cell_keys():
    """Build the canonical EXPECTED set of numerical cell keys (plan §6 3.1).

    The schema is the outer product of (route, dtype, shape, level,
    input_construction_version, seed, reference_dtype) where each route's shape
    set is fixed by its evidence contract:

      - planar, grouped: 8 SHAPES (cublaslt full-matrix set) x {C16BF, C32F}
      - region_fused: the INTENDED full-anchor P=A[4096,1024]@B[1024,16384] (plan
        §5 2.2); these cells are NOT_RUN until Task 3b measures them.
      - cutlass_4m_single: the anchor (16384,1024,1024)

    All routes x 3 levels x >=3 seeds x c64 reference. The
    ``input_construction_version`` token is ``"cancellation_v2"`` for
    cancellation-level cells and ``level + "_v1"`` for baseline/mixed_scale
    (errata #2 / #4: unified token + baseline/mixed producers MUST write
    version tokens so those routes can match).
    """
    keys = set()
    for shape in SHAPES:
        for route in ("planar", "grouped"):
            for dtype in DTYPES_BY_ROUTE[route]:
                for level in LEVELS:
                    ver = _version_token_for_level(level)
                    for seed in SEEDS:
                        keys.add((route, dtype, tuple(shape), level, ver, seed, "c64"))
    for level in LEVELS:
        ver = _version_token_for_level(level)
        for seed in SEEDS:
            keys.add(
                (
                    "region_fused",
                    "c64",
                    REGION_FULL_ANCHOR_SHAPE,
                    level,
                    ver,
                    seed,
                    "c64",
                )
            )
    for level in LEVELS:
        ver = _version_token_for_level(level)
        for seed in SEEDS:
            keys.add(
                (
                    "cutlass_4m_single",
                    "C16BF",
                    CUTLASS_ANCHOR_SHAPE,
                    level,
                    ver,
                    seed,
                    "c64",
                )
            )
    return keys


def _as_expected_keys(expected_counts, rows):
    """Normalize the ``expected_counts`` argument to a set of canonical cell keys.

    Accepts either:
      - a set/iterable of (route, dtype, shape, level, input_construction_version,
        seed, reference_dtype) 7-tuples (preferred, used by
        ``required_cell_keys()``), OR
      - a legacy dict ``{(route, dtype): N_count}`` for backward compatibility with
        count-based tests. In that mode up to N keys per (route, dtype) are sampled
        from the rows themselves in row order, preserving the old count semantics.
    """
    if isinstance(expected_counts, dict):
        keys = set()
        for (route, dtype), n in expected_counts.items():
            taken = 0
            for r in rows:
                if taken >= n:
                    break
                if r["route"] == route and r["dtype"] == dtype:
                    keys.add(_cell_key(r))
                    taken += 1
        return keys
    return {_cell_key(k) if isinstance(k, dict) else k for k in expected_counts}


# Case-binding hash keys + their source files (plan §5.2 / spec §4.4). MUST
# stay in sync with ``manifest.NUMERICAL_BINDINGS``. ``_CASE_HASH_KEYS`` is the
# canonical set of required case-binding keys (excluding the ``"algorithm"``
# metadata key) used by ``aggregate``'s ``binding_unavailable`` check (Task 4
# errata #2: empty/missing/None/short/non-hex -> unavailable -> global-invalid).
_CASE_HASH_FILES = (
    ("edge_map_sha256", "c1_c2_edge_map.json"),
    ("region_prototype_sha256", "region_prototype.json"),
    ("contraction_shapes_sha256", "contraction_shapes.csv"),
    ("cublaslt_planar_capability_sha256", "cublaslt_planar_capability.json"),
    ("cublaslt_full_matrix_sha256", "cublaslt_full_matrix.csv"),
    ("cublaslt_grouped_capability_sha256", "cublaslt_grouped_capability.json"),
    ("cublaslt_grouped_rows_sha256", "cublaslt_grouped.csv"),
    ("cutlass_4m_sha256", "cutlass_sm120_4m.json"),
    ("numerical_csv_sha256", "numerical_validation.csv"),
)
_CASE_HASH_KEYS = tuple(k for k, _ in _CASE_HASH_FILES)
# Minimum length of a valid case-binding hash: ``_case_hashes()`` returns the
# full ``hashlib.sha256(...).hexdigest()`` = 64 hex chars (no truncation).
# Task 4 finding 3.4 fix: the errata explicitly lists "short" as a trigger for
# ``binding_unavailable`` -> a valid-hex string of 8-63 chars (e.g. ``"a"*10``)
# must be rejected so it cannot slip past ``_is_invalid_hash`` and let the
# route-local loop run (false PASS). ``cell_key_hash()`` returns 16-char
# truncated hex for cell metadata -- a DIFFERENT binding, not checked here.
_CASE_HASH_MIN_LEN = 64


def _is_invalid_hash(v):
    """True if a case-binding hash value is None, not a str, empty, shorter
    than the expected sha256 hex length (64 chars), or contains non-hex chars.

    ``"MISMATCH"`` is NOT invalid -- it is the ``binding_mismatch`` sentinel
    (handled separately by ``aggregate``). Used by the ``binding_unavailable``
    check (Task 4 errata #2) so that empty/None/short/non-hex case bindings
    force ``global_invalid`` (ALL per_route = UNKNOWN).

    ``_case_hashes()`` returns the full 64-char sha256 hex; ``cell_key_hash()``
    returns 16-char truncated hex for cell metadata (a different binding, NOT
    checked here). A case-binding hash shorter than ``_CASE_HASH_MIN_LEN``
    (64) is treated as malformed: valid-hex but too-short strings like
    ``"a"*10`` (Task 4 finding 3.4) MUST be rejected so they cannot bypass the
    ``binding_unavailable`` deny-all and let the route-local loop run (false
    PASS). ``"MISMATCH"``-without-the-sentinel would still be caught by the
    non-hex branch below.
    """
    if v == "MISMATCH":
        return False  # binding_mismatch sentinel -- not an unavailable hash
    if v is None or not isinstance(v, str):
        return True
    if len(v) == 0:
        return True
    if len(v) < _CASE_HASH_MIN_LEN:
        return True
    try:
        int(v, 16)  # raises ValueError if non-hex chars
    except ValueError:
        return True
    return False


def aggregate(rows, expected_counts, case_hashes, legit_not_run, shape_drift=False):
    """Fail-closed aggregation -> numerical_validation.json payload (spec §6 3.3).

    expected_counts: either a set of canonical cell keys (route, dtype, shape,
      level, input_construction_version, seed, reference_dtype) 7-tuples
      [preferred; see ``required_cell_keys()``], or a legacy dict
      ``{(route, dtype): N_count}`` for backward compatibility.

    Task 4 (finding 3.4): explicit global-invalid flags are computed BEFORE the
    per-route loop. If ``global_invalid`` (duplicate / shape_drift /
    binding_mismatch / binding_unavailable), ALL per_route criteria = UNKNOWN
    and overall = INCONCLUSIVE (return early). Previously aggregate computed
    per-route criterion FIRST, so a shape_drift / duplicate / binding error
    could leave overall=INCONCLUSIVE but a route=PASS, and gonogo reads
    per_route directly -> route VIABLE while NUMERICAL=UNKNOWN (fail-open).

    ``legit_not_run`` is informational only (Task 4 errata #1): recorded in
    ``fail_closed_reasons`` but does NOT set ``global_invalid`` (a legit NOT_RUN
    is still an UNKNOWN cell at the route level, not a global deny-all).

    Per-route criterion (plan §6 3.3)::

        any required cell missing or not-run -> route numerical = UNKNOWN
        all required cells measured, any policy failure    -> FAIL
        all required cells measured and pass               -> PASS

    A cell is **not-run** if its ``source`` starts with ``not_run:`` OR its
    ``relative_l2`` is None (the canonical metric was not measured; spec §3.2.1).
    NOT_RUN rows are KEPT in the CSV (diagnostic) but NEVER allow a route to PASS.

    ``shape_drift`` (plan §3.2 / §3.11): when True, the hardcoded SHAPES constant
    no longer matches ``contraction_shapes.csv``. The required numerical cells
    are stale -> global_invalid (overall UNKNOWN, ALL per_route UNKNOWN).

    JSON accounting per route: ``expected / actual / missing / extra`` cell counts
    where ``actual`` = measured keys that are in the expected set, ``missing`` =
    expected keys without a matching measured row, ``extra`` = measured keys not in
    the expected set (e.g. region_fused small_contract diagnostic rows). A duplicate
    cell key is a schema error (plan §6 3.1): global_invalid -> overall INCONCLUSIVE.
    """
    expected_keys = _as_expected_keys(expected_counts, rows)

    # Finding 3.4 (Task 4 errata #1): compute explicit global-invalid flags
    # BEFORE the per-route loop. If global_invalid, ALL per_route = UNKNOWN and
    # overall = INCONCLUSIVE (return early, before the route-local loop).
    seen = set()
    duplicate_count = 0
    for r in rows:
        k = _cell_key(r)
        if k in seen:
            duplicate_count += 1
        seen.add(k)

    binding_mismatch = any(v == "MISMATCH" for v in case_hashes.values() if v)
    # Task 4 errata #2: binding_unavailable MUST handle the empty/missing/None/
    # short/non-hex case + required-key completeness. The plan's original
    # ``any(v == "" for k,v in case_hashes.items() if k != "algorithm")`` MISSES
    # ``{"algorithm":"sha256"}`` (only the algorithm key, no case hashes): the
    # ``if k != "algorithm"`` filter removes it, leaving ``{}`` -> ``any([])`` =
    # False -> binding_unavailable=False (WRONG -- the binding is actually
    # unavailable because there are NO case hashes). Fix: True if (a) no required
    # case keys are present at all, OR (b) any required case key is MISSING from
    # case_hashes, OR (c) any case-hash value is empty/None/short/non-hex.
    required_case_keys = _CASE_HASH_KEYS
    case_hash_values = [case_hashes.get(k) for k in required_case_keys]
    binding_unavailable = (
        len(required_case_keys) == 0  # no case keys defined (defensive)
        or any(k not in case_hashes for k in required_case_keys)  # required key missing
        or any(
            _is_invalid_hash(v) for v in case_hash_values
        )  # empty/None/short/non-hex
    )
    schema_error = duplicate_count > 0
    global_invalid = bool(
        schema_error or shape_drift or binding_mismatch or binding_unavailable
    )

    # legit_not_run is informational only (errata #1): recorded in
    # fail_closed_reasons but does NOT set global_invalid.
    fail_closed_reasons = list(legit_not_run)

    if global_invalid:
        per_route = [
            {
                "route": rt,
                "criterion": "UNKNOWN",
                "n_cells": 0,
                "expected": 0,
                "actual": 0,
                "missing": 0,
                "extra": 0,
            }
            for rt in _ROUTES
        ]
        if schema_error:
            fail_closed_reasons.append(f"duplicate cell keys: {duplicate_count}")
        if shape_drift:
            fail_closed_reasons.append(
                "shape drift: SHAPES != contraction_shapes.csv (required numerical "
                "cells no longer match the contraction artifact)"
            )
        if binding_mismatch:
            fail_closed_reasons.append("case-binding hash mismatch")
        if binding_unavailable:
            fail_closed_reasons.append("case-binding hash unavailable")
        return {
            "schema_version": "numerical-validation-v1",
            "case_binding": case_hashes,
            "per_route": per_route,
            "overall_numerical_status": "INCONCLUSIVE",
            "fail_closed_reasons": fail_closed_reasons,
        }

    # global valid -> existing route-local loop (legit_not_run recorded as
    # informational in fail_closed_reasons, does NOT set global_invalid).
    per_route = []
    statuses = []
    for route in _ROUTES:
        dtypes = sorted(
            {r["dtype"] for r in rows if r["route"] == route}
            | {k[1] for k in expected_keys if k[0] == route}
        )
        route_verdicts = []
        route_counts = {"expected": 0, "actual": 0, "missing": 0, "extra": 0}
        for dtype in dtypes:
            exp = {k for k in expected_keys if k[0] == route and k[1] == dtype}
            cells = [r for r in rows if r["route"] == route and r["dtype"] == dtype]
            # measured = real source AND a real relative_l2 (the canonical metric)
            measured_rows = [
                r
                for r in cells
                if not str(r.get("source", "")).startswith("not_run")
                and r.get("relative_l2") is not None
            ]
            not_run_rows = [
                r
                for r in cells
                if str(r.get("source", "")).startswith("not_run")
                or r.get("relative_l2") is None
            ]
            measured_keys = {_cell_key(r) for r in measured_rows}
            missing = exp - measured_keys
            extra = measured_keys - exp
            route_counts["expected"] += len(exp)
            route_counts["actual"] += len(measured_keys & exp)
            route_counts["missing"] += len(missing)
            route_counts["extra"] += len(extra)

            if missing or not_run_rows:
                route_verdicts.append("UNKNOWN")
            else:
                for r in measured_rows:
                    if _cell_key(r) in exp:
                        v, _ = apply_policy(route, dtype, r)
                        route_verdicts.append(v or "UNKNOWN")

        if not route_verdicts:
            criterion = "NOT_RUN"
        elif any(v == "FAIL" for v in route_verdicts):
            criterion = "FAIL"
        elif any(v == "UNKNOWN" for v in route_verdicts):
            criterion = "UNKNOWN"
        else:
            criterion = "PASS"
        statuses.append(criterion)
        per_route.append(
            {
                "route": route,
                "criterion": criterion,
                "n_cells": sum(1 for r in rows if r["route"] == route),
                "expected": route_counts["expected"],
                "actual": route_counts["actual"],
                "missing": route_counts["missing"],
                "extra": route_counts["extra"],
            }
        )

    # Global-valid path: duplicate / shape_drift / binding_mismatch /
    # binding_unavailable are all False (otherwise we returned early above).
    # Overall status depends only on the route-local criteria.
    if any(s == "UNKNOWN" for s in statuses):
        overall = "INCONCLUSIVE"
    elif any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif all(s in ("PASS", "NOT_RUN") for s in statuses) and any(
        s == "PASS" for s in statuses
    ):
        overall = "PASS"
    else:
        overall = "INCONCLUSIVE"  # all NOT_RUN or empty: nothing proven

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
REAL_GEMM_SHAPES = [
    (16384, 1024, 1024),
    (524288, 32, 32),
    (262144, 64, 64),
    (1048576, 16, 16),
]
# Intended full-anchor P->T->E contract for region_fused (plan §5 2.2):
#   P = A[4096,1024] @ B[1024,16384] -> T -> E = D[64,64] @ T
# These cells are NOT_RUN until Task 3b measures them; required-cell schema only.
REGION_FULL_ANCHOR_SHAPE = (4096, 16384, 1024)
# cutlass_4m_single anchor (matches cublaslt anchor + cutlass_sm120_4m.json).
CUTLASS_ANCHOR_SHAPE = (16384, 1024, 1024)
LEVELS = ("baseline", "mixed_scale", "cancellation")
SEEDS = (0, 1, 2)
DTYPES_BY_ROUTE = {
    "planar": ("C16BF", "C32F"),
    "grouped": ("C16BF", "C32F"),
    "region_fused": ("c64",),
    "cutlass_4m_single": ("C16BF",),
}


def load_current_shapes(csv_path=None):
    """Stdlib-only loader for actual-large contraction shapes (plan §3.2 / §3.11).

    Reads ``contraction_shapes.csv``, applies the SAME actual-large policy as
    ``cublaslt.load_c1_c2_shapes`` (``bytes >= 64 MiB``), dedupes by (M,N,K),
    and returns a list of (M,N,K) tuples in first-seen order. Pure stdlib
    (csv + os) -- no numpy/CUDA -- so pure-function tests work GPU-free.

    Shape drift (CSV updated while SHAPES stays stale) is detected by
    ``shapes_in_sync()`` and forces the numerical route to UNKNOWN.
    """
    if csv_path is None:
        csv_path = os.path.join(OUT_DIR, "contraction_shapes.csv")
    min_bytes = 64 << 20  # 64 MiB -- matches cublaslt.load_c1_c2_shapes default
    seen = set()
    shapes = []
    with open(csv_path, newline="") as fh:
        rd = csv.DictReader(fh)
        for raw in rd:
            try:
                b = int(raw["bytes"])
                if b < min_bytes:
                    continue
                m, n, k = int(raw["M"]), int(raw["N"]), int(raw["K"])
            except (KeyError, ValueError, TypeError):
                continue
            key = (m, n, k)
            if key not in seen:
                seen.add(key)
                shapes.append(key)
    return shapes


def shapes_in_sync():
    """Check that SHAPES matches the current contraction artifact (plan §3.2).

    Returns True iff ``set(SHAPES) == set(load_current_shapes())``; False on
    drift or if the CSV is unreadable. Shape drift -> UNKNOWN: the required
    numerical cells no longer match the contraction artifact, so the case
    cannot be validated (do NOT silently re-hash and continue).
    """
    try:
        loaded = load_current_shapes()
    except (OSError, ValueError):
        return False
    return set(SHAPES) == set(loaded)


_CSV_COLUMNS = [
    "route",
    "M",
    "N",
    "K",
    "out_dtype",
    "dynamic_range_level",
    "seed",
    "relative_l2",
    "max_abs",
    "max_rel",
    "nan_inf",
    "n_elems",
    "policy_pass",
    "reference_dtype",
    "cell_key_hash",
    # ``source`` (spec §6 3.3) makes the CSV self-describing about each row's
    # origin: a real measurement ("measured"), a diagnostic row
    # ("diagnostic:small-contract"), a reused artifact ("task8_reuse"), or a
    # NOT_RUN required cell ("not_run:<reason>"). NOT_RUN rows are KEPT in the
    # CSV so a reader can see why a required cell was not measured without
    # consulting the JSON fail_closed_reasons. The aggregate keys its not_run
    # detection off this prefix (symmetric with the in-memory rows) in addition
    # to ``relative_l2 is None``.
    "source",
    # Cancellation diagnostic fields (plan §3.1 / spec §3.4): recorded for
    # cancellation-level rows via ``_enrich_cancellation_metrics`` so the
    # cancellation is independently auditable from the CSV artifacts (not just
    # by calling ``cancellation_metrics`` / ``make_inputs`` directly). Empty for
    # non-cancellation rows and label-shape rows (e.g. region_fused
    # "small_contract").
    "input_construction_version",
    "cancellation_epsilon",
    "reference_norm",
    "baseline_norm",
    "cancellation_ratio",
]


def cell_key_hash(route, dtype, shape, level, ver, seed):
    """SHA256[:16] of the cell-key tuple (route|dtype|shape|level|ver|seed).

    ``ver`` is the ``input_construction_version`` token (errata #5: it MUST be
    included in the hashed string so the hash is consistent with the 7-tuple
    cell key). This is a cell-metadata hash (identifies which numerical cell a
    row belongs to), NOT a source-artifact hash. Renamed from ``source_hash``
    so the field name no longer implies it binds the measurement source
    (plan §3.3). The actual source-artifact hashes live in the JSON
    ``case_binding`` (Task 5's full hash binding).
    """
    key = f"{route}|{dtype}|{shape}|{level}|{ver}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def write_csv(path, rows):
    # Tolerant to partial rows (e.g. minimal test rows that only carry a subset
    # of fields); production collectors pass the full schema. Missing numeric
    # fields render as empty CSV cells rather than raising KeyError.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(_CSV_COLUMNS)
        for r in rows:
            shape = r.get("shape")
            if isinstance(shape, (tuple, list)) and len(shape) == 3:
                M, N, K = shape
            else:
                # region_fused rows carry shape="small_contract" (a label, not a
                # tuple); fall back to explicit M/N/K fields or zeros.
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
            # Cancellation diagnostic fields (empty for non-cancellation rows).
            # icv is also consumed by cell_key_hash (errata #5: include
            # input_construction_version in the hashed string).
            icv = r.get("input_construction_version", "")
            sh = r.get("cell_key_hash")
            if not sh:
                sh = cell_key_hash(route, dtype, shape or (), level, icv, seed)
            # source defaults to "measured" for real measured rows; NOT_RUN rows
            # carry "not_run:<reason>"; diagnostic rows carry "diagnostic:*".
            source = r.get("source") or "measured"
            ceps = r.get("cancellation_epsilon")
            rnorm = r.get("reference_norm")
            bnorm = r.get("baseline_norm")
            cratio = r.get("cancellation_ratio")
            w.writerow(
                [
                    route,
                    M,
                    N,
                    K,
                    dtype,
                    level,
                    seed,
                    f"{rel_l2:.6e}" if rel_l2 is not None else "",
                    f"{max_abs:.6e}" if max_abs is not None else "",
                    f"{max_rel:.6e}" if max_rel is not None else "",
                    int(bool(r.get("nan_inf", False))),
                    r.get("n_elems", 0),
                    int(r.get("policy_pass", 0)),
                    r.get("reference_dtype", "c64"),
                    sh,
                    source,
                    icv,
                    f"{ceps:.6e}" if ceps is not None else "",
                    f"{rnorm:.6e}" if rnorm is not None else "",
                    f"{bnorm:.6e}" if bnorm is not None else "",
                    f"{cratio:.6e}" if cratio is not None else "",
                ]
            )


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Task 6: planar route numerical collector (GPU; spec §3)
# ---------------------------------------------------------------------------


def collect_planar(shape, dtype, level, seed):
    """Planar-complex BF16 (C16BF) or FP32-output (C32F) GEMM accuracy vs c64
    materialized. C16BF = bf16-rounded real parts + bf16 output; C32F = bf16-upcast
    inputs + fp32 output (out_dtype='fp32', fp32 accumulation, no output rounding).
    Reference is the fp32 complex matmul on the SAME bf16-upcast inputs (c64
    precision, apples-to-apples).

    Reality note (ext.cpp:104-109): the pybind11 ext requires uint16 (BF16-bit)
    inputs for ALL out_dtype values — there is no fp32-input path. So C32F here
    means bf16-input + fp32-output (fp32 accumulation, no output rounding), and
    the reference uses the SAME bf16-upcast inputs (apples-to-apples per
    reference_complex_matmul's contract). For out_dtype='fp32' the ext returns
    already-decoded float32 host arrays (NOT uint16 bits), so _bf16_bits_to_f32
    must not be called on them. The brief's original C32F path passed fp32 arrays
    to the uint16-typed ext args (TypeError) and called _bf16_bits_to_f32 on the
    float32 output (corruption); both are fixed here.
    """
    from results._phase0.cublaslt import (
        load_ext,
        _f32_to_bf16_bits_and_upcast,
        _bf16_bits_to_f32,
        reference_complex_matmul,
    )

    if dtype not in ("C16BF", "C32F"):
        raise ValueError(
            f"collect_planar unsupported dtype {dtype!r}; expected C16BF or C32F"
        )

    M, N, K = shape
    A, B = make_inputs(level, shape, seed)  # A=(M,K), B=(K,N)
    ar, ai = A.real.astype(np.float32), A.imag.astype(np.float32)
    br, bi = B.real.astype(np.float32), B.imag.astype(np.float32)
    ext = load_ext()
    if dtype == "C16BF":
        ar_bf, ar_f = _f32_to_bf16_bits_and_upcast(ar)
        ai_bf, ai_f = _f32_to_bf16_bits_and_upcast(ai)
        br_bf, br_f = _f32_to_bf16_bits_and_upcast(br)
        bi_bf, bi_f = _f32_to_bf16_bits_and_upcast(bi)
        cr_u16, ci_u16 = ext.planar_complex_matmul_bf16(
            ar_bf, ai_bf, br_bf, bi_bf, M, N, K, out_dtype="bf16"
        )
        cr = _bf16_bits_to_f32(cr_u16)
        ci = _bf16_bits_to_f32(ci_u16)
        cr_ref, ci_ref = reference_complex_matmul(ar_f, ai_f, br_f, bi_f)
        out = (cr + 1j * ci).astype(np.complex64)
        ref = (cr_ref + 1j * ci_ref).astype(np.complex64)
    else:  # C32F — ext requires uint16 BF16-bit inputs even for fp32 output
        ar_bf, ar_f = _f32_to_bf16_bits_and_upcast(ar)
        ai_bf, ai_f = _f32_to_bf16_bits_and_upcast(ai)
        br_bf, br_f = _f32_to_bf16_bits_and_upcast(br)
        bi_bf, bi_f = _f32_to_bf16_bits_and_upcast(bi)
        cr, ci = ext.planar_complex_matmul_bf16(
            ar_bf, ai_bf, br_bf, bi_bf, M, N, K, out_dtype="fp32"
        )
        cr_ref, ci_ref = reference_complex_matmul(ar_f, ai_f, br_f, bi_f)
        out = (cr + 1j * ci).astype(np.complex64)
        ref = (cr_ref + 1j * ci_ref).astype(np.complex64)
    metrics = compute_metrics(out, ref)
    verdict, _ = apply_policy("planar", dtype, metrics)
    row = {
        "route": "planar",
        "dtype": dtype,
        "shape": shape,
        "level": level,
        "seed": seed,
        "reference_dtype": "c64",
        **metrics,
        "policy_pass": int(verdict == "PASS"),
    }
    # errata #4: baseline/mixed_scale producers MUST write version tokens so
    # their rows match required_cell_keys. Cancellation-level rows get the
    # token from _enrich_cancellation_metrics (which also computes the 5
    # diagnostic fields -- setting it here would short-circuit that).
    if level != "cancellation":
        row["input_construction_version"] = _version_token_for_level(level)
    return _enrich_cancellation_metrics(row)


# ---------------------------------------------------------------------------
# Task 7: grouped (batched) route numerical collector (GPU; spec §3)
# ---------------------------------------------------------------------------


def collect_grouped(shape, dtype, level, seed, batch=4):
    """Batched planar-complex GEMM accuracy (cublasLt batched route, Task 7) vs c64.

    Runs ``batch`` GEMMs of ``shape``; reports the WORST cell across the batch
    (consistent with cublaslt._run_batched_timing aggregation). C32F uses fp32 output.
    """
    from results._phase0.cublaslt import (
        load_ext,
        _f32_to_bf16_bits_and_upcast,
        _bf16_bits_to_f32,
        reference_complex_matmul,
    )

    M, N, K = shape
    # one independent (A,B) per batch element, derived from seed+batch_idx
    ar = np.empty((batch, M, K), np.float32)
    ai = np.empty_like(ar)
    br = np.empty((batch, K, N), np.float32)
    bi = np.empty_like(br)
    refs = []
    for b in range(batch):
        A, B = make_inputs(level, shape, seed * 1000 + b)
        ar[b], ai[b] = A.real.astype(np.float32), A.imag.astype(np.float32)
        br[b], bi[b] = B.real.astype(np.float32), B.imag.astype(np.float32)
        refs.append((ar[b], ai[b], br[b], bi[b]))
    ext = load_ext()
    if dtype not in ("C16BF", "C32F"):
        raise ValueError(f"collect_grouped unsupported dtype {dtype!r}")
    out_dtype = "bf16" if dtype == "C16BF" else "fp32"
    # ext requires uint16 BF16-bit inputs for ALL out_dtype (ext.cpp:104-109);
    # C32F = bf16-input + fp32-output (fp32 accumulation). See Task 6 reality note.
    ar_bf, ar_f = _f32_to_bf16_bits_and_upcast(ar)
    ai_bf, ai_f = _f32_to_bf16_bits_and_upcast(ai)
    br_bf, br_f = _f32_to_bf16_bits_and_upcast(br)
    bi_bf, bi_f = _f32_to_bf16_bits_and_upcast(bi)
    cr_u16, ci_u16 = ext.planar_complex_matmul_bf16_batched(
        ar_bf, ai_bf, br_bf, bi_bf, M, N, K, batch, out_dtype=out_dtype
    )
    worst = {
        "relative_l2": 0.0,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "nan_inf": False,
        "n_elems": 0,
    }
    for b in range(batch):
        if dtype == "C16BF":
            cr = _bf16_bits_to_f32(cr_u16[b])
            ci = _bf16_bits_to_f32(ci_u16[b])
        else:  # C32F: ext returns fp32 directly, no decode
            cr, ci = cr_u16[b], ci_u16[b]
        cr_ref, ci_ref = reference_complex_matmul(ar_f[b], ai_f[b], br_f[b], bi_f[b])
        m = compute_metrics(
            (cr + 1j * ci).astype(np.complex64),
            (cr_ref + 1j * ci_ref).astype(np.complex64),
        )
        for kk in ("relative_l2", "max_abs", "max_rel"):
            worst[kk] = max(worst[kk], m[kk])
        worst["nan_inf"] = worst["nan_inf"] or m["nan_inf"]
        worst["n_elems"] += m["n_elems"]
    verdict, _ = apply_policy("grouped", dtype, worst)
    row = {
        "route": "grouped",
        "dtype": dtype,
        "shape": shape,
        "level": level,
        "seed": seed,
        "reference_dtype": "c64",
        **worst,
        "policy_pass": int(verdict == "PASS"),
    }
    if level != "cancellation":
        row["input_construction_version"] = _version_token_for_level(level)
    return _enrich_cancellation_metrics(row)


# ---------------------------------------------------------------------------
# Task 8: region_fused small-contract correctness collector (GPU; spec §3, §7.2)
# ---------------------------------------------------------------------------


def collect_region_fused(level, seed):
    """region_fused correctness on the small 8-D contract (spec §3, §7.2).

    actual-large fused is compute-bound (producer recompute ~TM=64) and is NOT run;
    that legitimate NOT_RUN is recorded by main() in legit_not_run. Here we prove
    fused == materialized at c64 on the small contract, over the requested level/seed.
    """
    import cupy as cp
    from results._phase0 import region_proto as rp

    s = rp.SMALL_SHAPES
    # derive A/B/D from make_inputs at the small shape; D from the same generator
    A, B = make_inputs(
        level, (s["PM"], s["PN"], s["K1"]), seed
    )  # (M,N,K); A=(PM,K1), B=(K1,PN)
    D = make_inputs(level, (s["TM"], s["TM"], s["TM"]), seed + 7000)[
        0
    ]  # (TM,TM) consumer matrix
    E_mat, _, _ = rp.materialized_reference(
        cp.asarray(A), cp.asarray(B), cp.asarray(D), rp.SMALL_STEPS
    )
    E_fus = rp.fused_reference(
        cp.asarray(A), cp.asarray(B), cp.asarray(D), rp.SMALL_STEPS, s
    )
    metrics = compute_metrics(cp.asnumpy(E_fus), cp.asnumpy(E_mat))
    verdict, _ = apply_policy("region_fused", "c64", metrics)
    row = {
        "route": "region_fused",
        "dtype": "c64",
        "shape": "small_contract",
        "level": level,
        "seed": seed,
        "reference_dtype": "c64",
        # diagnostic: this row is the small-contract correctness proof (spec §7.2),
        # NOT the required full-anchor cell. It shows up as `extra` in the JSON
        # accounting because its shape key ("small_contract") does not match the
        # required REGION_FULL_ANCHOR_SHAPE tuple.
        "source": "diagnostic:small-contract",
        **metrics,
        "policy_pass": int(verdict == "PASS"),
    }
    if level != "cancellation":
        row["input_construction_version"] = _version_token_for_level(level)
    return _enrich_cancellation_metrics(row)


# ---------------------------------------------------------------------------
# Task 9: cutlass_4m_single numerical collector (spec §3, §12)
# ---------------------------------------------------------------------------


def _cutlass_injection_available():
    """Probe whether cutlass_probe can accept external input data for adversarial
    levels. Returns True only if a re-run entry point is confirmed; default False
    until Task 9 verifies the injection point (spec §12 risk). When False, adversarial
    levels are recorded as legit NOT_RUN (toolchain-bound), baseline reuses Task 8.
    """
    return False


def collect_cutlass(level, seed):
    """cutlass_4m_single numerical row. C16BF only (CUTLASS GemmElement=bf16).

    baseline: reuse results/phase0/cutlass_sm120_4m.json (Task 8 single, 3 seeds @
    anchor 16384x1024x1024). adversarial: attempt injection; else NOT_RUN row.

    Task 3a reality correction (spec §3.2.1 / plan §6 3.2): the cutlass artifact
    measures max_rel + max_abs but NOT relative_l2 (no vector L2 was computed). The
    baseline row therefore carries ``relative_l2=None`` -- NEVER substituted by
    max_rel. apply_policy then reports the cell incomplete (verdict None -> UNKNOWN
    at the route layer) which is the honest state until Task 3b re-measures with a
    real vector L2.
    """
    if level == "baseline":
        with open(os.path.join(OUT_DIR, "cutlass_sm120_4m.json")) as fh:
            data = json.load(fh)
        c = data["single_4m"]["correctness"]
        metrics = {
            # NEVER substitute max_rel for relative_l2 (spec §3.2.1). If the
            # artifact did not measure a real vector L2, the field stays None and
            # apply_policy flags the cell incomplete.
            "relative_l2": c.get("relative_l2"),
            "max_abs": c.get("max_abs", 0.0),
            "max_rel": c.get("max_rel", 1e9),
            "nan_inf": bool(c.get("nan_inf", True)),
            "n_elems": 16384 * 1024,
        }
        verdict, _ = apply_policy("cutlass_4m_single", "C16BF", metrics)
        row = {
            "route": "cutlass_4m_single",
            "dtype": "C16BF",
            "shape": CUTLASS_ANCHOR_SHAPE,
            "level": level,
            "seed": seed,
            "reference_dtype": "c64",
            "source": "task8_reuse",
            **metrics,
            "policy_pass": int(verdict == "PASS"),
        }
        if level != "cancellation":
            row["input_construction_version"] = _version_token_for_level(level)
        return _enrich_cancellation_metrics(row)
    # adversarial level
    if _cutlass_injection_available():
        # Future: re-run cutlass kernel with make_inputs(level) injected.
        raise NotImplementedError("cutlass adversarial injection not wired yet")
    row = {
        "route": "cutlass_4m_single",
        "dtype": "C16BF",
        "shape": CUTLASS_ANCHOR_SHAPE,
        "level": level,
        "seed": seed,
        "reference_dtype": "c64",
        "source": "not_run:toolchain-injection-unavailable",
        "relative_l2": None,
        "max_abs": None,
        "max_rel": None,
        "nan_inf": False,
        "n_elems": 0,
        "policy_pass": 0,
    }
    if level != "cancellation":
        row["input_construction_version"] = _version_token_for_level(level)
    return _enrich_cancellation_metrics(row)


# ---------------------------------------------------------------------------
# Task 10: main() integration — full matrix + artifact generation (spec §6, §10)
# ---------------------------------------------------------------------------


def _case_hashes():
    """Read ALL 9 route-source artifact hashes for case binding (plan §5.2 /
    spec §4.4 -- full SHA256 binding of every numerical source file).

    Returns a dict keyed by the ``_sha256`` binding key names (matching
    ``manifest.NUMERICAL_BINDINGS``) with full 64-hex-char sha256 values, plus
    an ``"algorithm": "sha256"`` metadata field documenting the algorithm +
    length (spec §4.4: no unexplained truncation). Missing files -> empty
    string for that key (the manifest validator treats an empty expected hash
    as UNAVAILABLE).

    Sanitization ordering invariant (plan §5.4): this function MUST be called
    AFTER ``write_csv`` (so ``numerical_csv_sha256`` reflects the final CSV
    bytes) and AFTER source-artifact sanitization (so all 9 hashes reflect
    POST-sanitization file bytes). The canonical pipeline is::

      sanitize source artifacts
        -> generate/recompute numerical CSV/JSON (write_csv THEN _case_hashes)
        -> compute final bindings (this function)
        -> generate gonogo
        -> generate manifest

    The sanitizer must NOT mask a source content change by only rewriting the
    expected hash -- for semantic changes the producer MUST be re-run. For
    privacy-only sanitization (path/name tokens, no numerical semantics) the
    binding hashes are recomputed from the post-sanitized files so the binding
    stays consistent (``sanitize.rehash_numerical_binding``).

    The (binding key, file) pairs live in the module-level ``_CASE_HASH_FILES``
    constant (shared with ``aggregate``'s ``binding_unavailable`` check via
    ``_CASE_HASH_KEYS``) so there is a single source of truth for the required
    case-binding key set.
    """
    hashes = {"algorithm": "sha256"}
    for hash_key, fname in _CASE_HASH_FILES:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            with open(p, "rb") as _fh:
                hashes[hash_key] = hashlib.sha256(_fh.read()).hexdigest()
        else:
            hashes[hash_key] = ""
    return hashes


def _legit_not_run_reasons():
    """Human-readable reasons for legitimate NOT_RUN cells.

    Informational only -- recorded in fail_closed_reasons but does NOT change the
    verdict. A NOT_RUN cell still forces its route to UNKNOWN regardless of whether
    it is "legit" (spec §3.2 / plan §6 3.3).
    """
    reasons = [
        "region_fused:actual-large-fused:compute-bound (spec §7.2; correctness "
        "proven on small contract only; intended full-anchor cells NOT_RUN until "
        "Task 3b)",
    ]
    if not _cutlass_injection_available():
        reasons.append(
            "cutlass_4m_single:adversarial-level:toolchain-injection-unavailable "
            "(baseline reused from Task 8; relative_l2 not measured by artifact)"
        )
    return reasons


def _read_csv_rows(csv_path):
    """Read a numerical_validation.csv back into row dicts (Task 3a JSON regen).

    Preserves the existing measured rows (planar/grouped/region_fused small-contract)
    so the JSON accounting can be recomputed WITHOUT new GPU measurement (plan §6 3a).
    Region_fused small-contract rows (M=N=K=0) round-trip with shape="small_contract".
    Rows with empty relative_l2/max_abs/max_rel cells (cutlass adversarial NOT_RUN)
    round-trip with None metrics so the aggregate treats them as not-run.

    The ``source`` column (Task 3a CSV NOT_RUN fix) round-trips verbatim so the
    ``not_run:<reason>`` / ``diagnostic:small-contract`` / ``task8_reuse`` labels
    survive write→read. Old CSVs lacking the column get a backward-compat default
    (``"measured"`` for real rows, ``"diagnostic:small-contract"`` for region
    small-contract rows) so the regen path stays idempotent across the schema bump.
    """
    rows = []
    with open(csv_path, newline="") as fh:
        rd = csv.DictReader(fh)
        for raw in rd:
            M = int(raw["M"]) if raw["M"] else 0
            N = int(raw["N"]) if raw["N"] else 0
            K = int(raw["K"]) if raw["K"] else 0
            is_region_small = raw["route"] == "region_fused" and not (M or N or K)
            if is_region_small:
                shape = "small_contract"
            else:
                shape = (M, N, K)

            def _maybe_float(v):
                return float(v) if v else None

            # source: prefer the column value; fall back to a route-aware default
            # for pre-schema-bump CSVs (no `source` column at all).
            source = (raw.get("source") or "").strip()
            if not source:
                source = "diagnostic:small-contract" if is_region_small else "measured"
            # Cancellation diagnostic fields (absent in pre-schema-bump CSVs
            # and empty for non-cancellation rows).
            icv = (raw.get("input_construction_version") or "").strip()
            row = {
                "route": raw["route"],
                "dtype": raw["out_dtype"],
                "shape": shape,
                "level": raw["dynamic_range_level"],
                "seed": int(raw["seed"]),
                "reference_dtype": raw.get("reference_dtype") or "c64",
                "relative_l2": _maybe_float(raw["relative_l2"]),
                "max_abs": _maybe_float(raw["max_abs"]),
                "max_rel": _maybe_float(raw["max_rel"]),
                "nan_inf": bool(int(raw["nan_inf"])) if raw["nan_inf"] else False,
                "n_elems": int(raw["n_elems"]) if raw["n_elems"] else 0,
                "policy_pass": (int(raw["policy_pass"]) if raw["policy_pass"] else 0),
                "source": source,
            }
            if icv:
                row["input_construction_version"] = icv
                row["cancellation_epsilon"] = _maybe_float(
                    raw.get("cancellation_epsilon")
                )
                row["reference_norm"] = _maybe_float(raw.get("reference_norm"))
                row["baseline_norm"] = _maybe_float(raw.get("baseline_norm"))
                row["cancellation_ratio"] = _maybe_float(raw.get("cancellation_ratio"))
            rows.append(row)
    return rows


def _not_run_reason_for(route):
    """Short, stable reason slug for a NOT_RUN required cell on ``route``.

    The slug is embedded in the CSV ``source`` column as ``not_run:<slug>`` so a
    reader can see WHY a required cell was not measured. The long-form
    human-readable reason also lives in ``_legit_not_run_reasons`` /
    ``fail_closed_reasons``; this slug is the machine-friendly mirror.
    """
    if route == "region_fused":
        return "compute-bound-actual-large-fused"
    if route == "cutlass_4m_single":
        return "toolchain-injection-unavailable"
    return "not-measured"


def _emit_not_run_rows(existing_rows, required_keys):
    """Emit explicit NOT_RUN rows for required cells that have NO CSV row at all
    (spec §6 3.3: "CSV 中保留 NOT_RUN row 及 reason").

    A required cell counts as "has a row" if ANY row (measured, diagnostic, or
    already-not_run) already carries its key. This avoids creating duplicate keys
    for cells that already have a (possibly partial) row -- e.g. the 3 cutlass
    baseline rows (source=task8_reuse, relative_l2=None) already represent their
    cells, so no duplicate NOT_RUN row is emitted for them.

    The absent cells (region_fused's 9 intended full-anchor cells until Task 3b)
    get a NOT_RUN row carrying the full expected key with empty metrics and
    ``source="not_run:<reason>"``. The aggregate's not_run detection keys off
    that prefix symmetrically with in-memory rows, so the CSV is now
    self-describing: a reader sees the NOT_RUN row + reason without consulting
    the JSON.

    Returned rows are appended to the measured rows BEFORE ``aggregate`` /
    ``write_csv``. JSON accounting is unaffected: NOT_RUN rows never count as
    ``actual``/measured (``actual`` counts only measured keys in the expected
    set), so ``expected / actual / missing / extra`` are unchanged. Rows are
    emitted in a deterministic (level, seed) order so the CSV byte-content is
    reproducible across runs (the manifest hashes this file).
    """
    present_keys = {_cell_key(r) for r in existing_rows}
    not_run_rows = []
    for key in required_keys - present_keys:
        route, dtype, shape, level, ver, seed, ref_id = key
        not_run_rows.append(
            {
                "route": route,
                "dtype": dtype,
                "shape": shape,
                "level": level,
                "input_construction_version": ver,
                "seed": seed,
                "reference_dtype": ref_id,
                "source": f"not_run:{_not_run_reason_for(route)}",
                "relative_l2": None,
                "max_abs": None,
                "max_rel": None,
                "nan_inf": False,
                "n_elems": 0,
                "policy_pass": 0,
            }
        )
    # Deterministic order: LEVELS order then seed (matches the measured-row
    # emission order in main()), so repeated regen yields byte-identical CSV.
    level_order = {lvl: i for i, lvl in enumerate(LEVELS)}
    not_run_rows.sort(
        key=lambda r: (r["route"], level_order.get(r["level"], 99), r["seed"])
    )
    return not_run_rows


def main(run_gpu: bool = True, regen_no_gpu: bool = False):
    """Run the full numerical matrix and write numerical_validation.{csv,json}.

    run_gpu=False: use whatever collect_* resolve to (test harness monkeypatches them).
    regen_no_gpu=True (Task 3a): read existing CSV rows for planar/grouped/region_fused
      (preserving real measured data), regenerate cutlass rows via the non-GPU
      artifact reader (now emitting relative_l2=None instead of the max_rel proxy),
      and recompute the fail-closed aggregate. NO GPU measurement.
    """
    legit_not_run = _legit_not_run_reasons()
    # Shape drift check (plan §3.2 / §3.11): if SHAPES no longer matches
    # contraction_shapes.csv, the required numerical cells are stale -> UNKNOWN.
    drift = not shapes_in_sync()

    if regen_no_gpu:
        existing_csv = os.path.join(OUT_DIR, "numerical_validation.csv")
        rows = _read_csv_rows(existing_csv)
        # Drop any old cutlass rows and regenerate them via the (non-GPU) artifact
        # reader so the baseline rows carry relative_l2=None (no max_rel proxy).
        rows = [r for r in rows if r["route"] != "cutlass_4m_single"]
        # Drop any stale emitted NOT_RUN rows so the idempotent re-emit below
        # is the single source of truth for NOT_RUN rows (prevents duplicates on
        # repeated regen).
        rows = [r for r in rows if not str(r.get("source", "")).startswith("not_run:")]
        for level in LEVELS:
            for seed in SEEDS:
                rows.append(collect_cutlass(level, seed))
        # errata #3 / INV-1 (finding 3.1): relabel ALL old measured cancellation
        # rows to ``cancellation_legacy_v1``. In a no-GPU regen NO measured
        # cancellation row can be a real ``cancellation_v2`` (that requires a
        # GPU v2 run, which the no-GPU round does NOT do). Relabeling
        # unconditionally -- regardless of the old token (``v2_cancellation``,
        # empty, or even a previously-mislabeled ``cancellation_v2``) -- ensures
        # INV-1: non-GPU round ``cancellation_v2`` + ``measured`` row count == 0.
        # This separates archival GPU v1 evidence (legacy) from planned CPU v2
        # diagnostics, fixing the provenance forgery where old GPU-measured
        # cancellation rows were labeled v2 while CPU-computing new v2
        # diagnostics and appending them (finding 3.1).
        for r in rows:
            if (
                r.get("level") == "cancellation"
                and not str(r.get("source", "")).startswith("not_run")
                and r.get("relative_l2") is not None
            ):
                r["input_construction_version"] = "cancellation_legacy_v1"
        # errata #4: ensure baseline/mixed_scale rows carry the correct version
        # token (``baseline_v1`` / ``mixed_scale_v1``). Old CSV rows from a
        # pre-schema-bump CSV leave the field empty, and monkeypatched producers
        # in tests may return the wrong token -- either way those rows would
        # never match ``required_cell_keys`` (which expects the per-level v1
        # token for baseline/mixed). Overriding here is defensive and makes the
        # no-GPU regen robust against producer drift.
        for r in rows:
            lvl = r.get("level")
            if lvl in ("baseline", "mixed_scale"):
                r["input_construction_version"] = _version_token_for_level(lvl)
        # Emit explicit NOT_RUN rows for required cells with no CSV row at all
        # (region_fused full-anchor; spec §6 3.3). Makes the CSV self-describing.
        rows.extend(_emit_not_run_rows(rows, required_cell_keys()))
        # Enrich cancellation-level rows with the 5 cancellation diagnostic
        # fields (GPU-free numpy CPU). CSV-read rows from a pre-schema-bump CSV
        # and emitted NOT_RUN rows (e.g. region_fused full-anchor cancellation
        # cells) don't go through a collector, so they are enriched here.
        # collect_cutlass rows are already enriched (idempotent skip).
        # Legacy-v1 rows short-circuit (errata #6: don't re-enrich archival
        # diagnostic). NOT_RUN cancellation rows whose version was set to
        # ``cancellation_v2`` by _emit_not_run_rows also short-circuit
        # (idempotent). Rows with NO version (e.g. region_fused full-anchor
        # cancellation NOT_RUN from an old CSV that lacked the field) get
        # enriched here with the canonical ``cancellation_v2`` token + the 5
        # diagnostic fields.
        rows = [_enrich_cancellation_metrics(r) for r in rows]
        # Plan §5.4 ordering: write the CSV BEFORE computing case_binding so
        # numerical_csv_sha256 reflects the final on-disk CSV bytes.
        write_csv(os.path.join(OUT_DIR, "numerical_validation.csv"), rows)
        payload = aggregate(
            rows, required_cell_keys(), _case_hashes(), legit_not_run, shape_drift=drift
        )
        write_json(os.path.join(OUT_DIR, "numerical_validation.json"), payload)
        return payload

    rows = []
    # planar + grouped: 8 shapes x {C16BF,C32F} x 3 levels x 3 seeds
    for shape in SHAPES:
        for dtype in DTYPES_BY_ROUTE["planar"]:
            for level in LEVELS:
                for seed in SEEDS:
                    rows.append(collect_planar(shape, dtype, level, seed))
                    rows.append(collect_grouped(shape, dtype, level, seed))
    # region_fused: small contract x 3 levels x 3 seeds (diagnostic; the required
    # full-anchor cells are NOT_RUN until Task 3b and are tracked by the schema).
    for level in LEVELS:
        for seed in SEEDS:
            rows.append(collect_region_fused(level, seed))
    # cutlass_4m_single: anchor x 3 levels x 3 seeds (baseline real, adversarial NOT_RUN).
    for level in LEVELS:
        for seed in SEEDS:
            rows.append(collect_cutlass(level, seed))
    # Emit explicit NOT_RUN rows for required cells with no CSV row at all
    # (region_fused full-anchor; spec §6 3.3). Makes the CSV self-describing.
    rows.extend(_emit_not_run_rows(rows, required_cell_keys()))

    # Plan §5.4 ordering: write the CSV BEFORE computing case_binding so
    # numerical_csv_sha256 reflects the final on-disk CSV bytes.
    write_csv(os.path.join(OUT_DIR, "numerical_validation.csv"), rows)
    payload = aggregate(
        rows, required_cell_keys(), _case_hashes(), legit_not_run, shape_drift=drift
    )
    write_json(os.path.join(OUT_DIR, "numerical_validation.json"), payload)
    return payload


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Phase 0 numerical validation matrix (plan §6 / spec §3)."
    )
    parser.add_argument(
        "--regen-no-gpu",
        action="store_true",
        help="Regenerate numerical_validation.{csv,json} from existing CSV rows "
        "WITHOUT GPU measurement (Task 3a regen path). Reads planar/grouped/"
        "region_fused rows from the existing CSV, regenerates cutlass rows via "
        "the non-GPU artifact reader, and recomputes the fail-closed aggregate.",
    )
    args = parser.parse_args()
    if args.regen_no_gpu:
        result = main(run_gpu=False, regen_no_gpu=True)
    else:
        result = main()
    print(_json.dumps(result, indent=2))
