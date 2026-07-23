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
