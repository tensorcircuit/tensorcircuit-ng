# Region-fused Full-anchor Numerical Accuracy Policy (Continuous Local-Gate)

> **Status:** DRAFT v2 for reviewer B policy review (v1 was POLICY_NOT_ACCEPTED 2026-07-26). This file MUST be committed to the git repo (`tensorcircuit-ng/`) so it enters the reviewed trust chain; reviewer B issues `POLICY_ACCEPTED` bound to the policy commit SHA-256 + policy ID + frozen constants. No adjustment based on new results after freeze.
>
> **Location:** this file lives at `tensorcircuit-ng/docs/superpowers/specs/` (INSIDE the git-tracked repo) so `git log` / commit SHAs bind it. (v1 was at workspace-root `docs/` — outside the repo, unverifiable.)

## v2 changes (addressing B's 4 blockers)

1. **Continuity (was P1 #1):** replaced the discontinuous 100x-tolerance-jump dual-gate with a **single continuous local gate** `local_scaled_max = max_i |error_i| / max(|reference_i|, α·s)`. No high/low partition, no jump. (Dual-partition form kept only as an alternative with the continuity constraint `low_threshold = α × high_threshold`.)
2. **Diagnosis as hypothesis, not claim (was P1 #2):** the v1 diagnosis ("6/9 failures are small-magnitude per-element relative-error blow-up") is **unverified** — old artifacts did not record the `|reference_i|` of the max-error element, and the two existing producers use **different** `max_rel` definitions (see §9). It is rewritten as a **hypothesis to be verified by re-measurement**, and both old `max_rel` definitions are **formally deprecated**.
3. **In-repo + trust chain (was P1 #3):** spec is committed to the git repo. `POLICY_ACCEPTED` binds the policy commit SHA-256 + policy ID (`REGION_FUSED_FULL_ANCHOR_ACCURACY_v2`) + all constants. Each measurement artifact records the same `policy_hash`; consumers recompute the verdict from the frozen constants (NOT trusting producer-written `policy_pass`).
4. **Freeze manifest (was P1 #4):** a `policy_freeze_manifest.json` is created BEFORE measurement, binding policy SHA + implementation commit + profile versions + exact shape + exact seed list + metric schema version + GPU/run-env identity + retry/failed-run retention rules. Holdout seeds are **B-specified or deterministically derived from the frozen policy hash** (not chosen post-hoc); never swapped after a trial run.

## Math contract (frozen)

- `output` = fused kernel `E` (c64[TM,TN] = c64[64,1048576], 2^26 = 67,108,864 elements)
- `reference` = materialized oracle `E_mat` (same shape, same inputs/seed)
- `error = output - reference`
- `s = RMS(reference) = sqrt(mean(|reference_i|^2))` — global RMS of the reference
- `τ = α * s` — signal scale (α frozen)

### Numerical guarantees (FP64 / stable accumulation — required)

All metric computation MUST satisfy:
- **dtype**: `s` (RMS) accumulated in **FP64** (cast reference to float64 for the sum-of-squares; the fused/materialized outputs may be c64 but the metric math is FP64). No FP32 accumulation for the sum-of-squares.
- **complex modulus**: `|z| = sqrt(re(z)^2 + im(z)^2)` computed in FP64; `|z|^2 = re(z)^2 + im(z)^2` (no `|z| = |re| + |im|` or other surrogate).
- **shape equality**: `output.shape == reference.shape` else `UNKNOWN_SHAPE_MISMATCH` (fail-closed; not PASS).
- **non-empty**: `output.size > 0` else `UNKNOWN_EMPTY_ARRAY`.
- **finite**: all metrics must be **finite, non-negative, non-bool real** numbers; any non-finite/NaN/Inf/bool metric -> `FAIL_INVALID_METRIC` (fail-closed).
- **sum-of-squares**: use a numerically stable accumulation (e.g. `numpy.linalg.norm` which uses a stable algorithm, or a pairwise/Kahan sum if custom). DOCUMENT the accumulation method.

## Metrics

1. `global_rel_l2 = ||error||_2 / ||reference||_2` (canonical L2 ratio; FP64, stable accumulation)
2. `local_scaled_max = max_i ( |error_i| / max(|reference_i|, τ) )` (single continuous local gate; `τ = α·s`)
3. `nan_inf = not all(isfinite(output)) OR not all(isfinite(reference)) OR not all(isfinite(error)) OR not all(isfinite(computed_metrics))` (BOTH output AND reference AND error AND computed metrics checked)

## Gate (PASS iff ALL hold; else FAIL)

1. `nan_inf == False` (output + reference + error + metrics all finite)
2. `global_rel_l2 < global_rel_l2_threshold`
3. `local_scaled_max < eta`

Any required cell violating any gate -> route `FAIL`. All required cells pass -> numerical `PASS` -> pipeline derives `VIABLE` **only if capability is also OK** (see §6, conditional).

## Initial candidate constants (for B to approve/freeze)

| constant | candidate | meaning |
|---|---|---|
| `α` (signal scale) | `1e-3` | clamp denominator to `α·s`; no 100x jump (continuous) |
| `global_rel_l2_threshold` | `1e-4` | canonical global L2 ratio (unchanged) |
| `eta` (`local_scaled_max` threshold) | `1e-3` | per-element error cap, scaled by `max(|ref|, α·s)` |

**Reviewer B must finalize these constants** at freeze time. After `POLICY_ACCEPTED`, constants cannot be adjusted based on new measurement results (that would be post-hoc threshold-fishing — the exact P2 #6 failure mode).

### Alternative: keep dual-partition form (continuity-constrained)

If B prefers the dual-partition form, it MUST satisfy the continuity constraint:
`low_signal_max_abs_norm_threshold == α × high_signal_max_rel_threshold`.

With `α=1e-3`, `high_signal_max_rel_threshold=1e-3`, the low-signal threshold must be `1e-6` (not v1's `1e-4`). Alternatively, with `low_signal_max_abs_norm_threshold=1e-4`, continuity requires `α=0.1`. The single continuous `local_scaled_max` (this section) is simpler and avoids the constraint; recommended.

## Fail-closed semantics + deterministic priority

When multiple anomalies co-occur, apply deterministic priority **FAIL > UNKNOWN > PASS**. Retain ALL reason codes (a cell may have a list of reasons):

| condition | verdict | reason code |
|---|---|---|
| `nan_inf == True` | `FAIL` | `FAIL_NAN_INF` (checks output + reference + error + metrics) |
| `global_rel_l2 >= global_rel_l2_threshold` | `FAIL` | `FAIL_GLOBAL_REL_L2` |
| `local_scaled_max >= eta` | `FAIL` | `FAIL_LOCAL_SCALED_MAX` |
| metric is non-finite/negative/bool/non-real (any metric) | `FAIL` | `FAIL_INVALID_METRIC` |
| `output.shape != reference.shape` | `UNKNOWN` | `UNKNOWN_SHAPE_MISMATCH` |
| `output.size == 0` | `UNKNOWN` | `UNKNOWN_EMPTY_ARRAY` |
| all-zero reference (`s == 0`) -> metrics undefined (division by zero) | `UNKNOWN` | `UNKNOWN_ALL_ZERO_REFERENCE` |
| a required metric is missing/not computed | `UNKNOWN` | `UNKNOWN_MISSING_METRIC` |
| all gates hold | `PASS` | `PASS` |

Priority: collect ALL triggered reason codes; the verdict is the highest-priority (`FAIL` if any FAIL reason, else `UNKNOWN` if any UNKNOWN reason, else `PASS`). `UNKNOWN` is fail-closed (not PASS). The aggregate (per-route) treats UNKNOWN required cells as route-UNKNOWN (not VIABLE), consistent with the existing fail-closed aggregate.

### Edge cases (corrected from v1 per B)

- **"empty high-signal mask"** (v1 had this test): when `0 < α ≤ 1` and `s > 0`, `max|reference_i| ≥ s ≥ α·s = τ`, so at least one element satisfies `|reference_i| ≥ τ`. The high-signal mask is **mathematically never empty** (for the continuous form, there is no partition at all; for the dual form, B notes empty high-signal is impossible). v1's `test_dual_gate_empty_high_signal_mask_unknown` is **removed** — the test cannot be constructed.
- **"empty low-signal mask"** (dual form only): a legitimate configuration (e.g. all reference elements equal magnitude) may have an empty low-signal mask. The low-signal gate should **vacuously pass** (not `UNKNOWN`) in that case unless an independent profile contract requires low-signal coverage.
- **localized single-element error test (corrected per B's bound):** with `N=2^26`, `α=1e-3`, a single-element `|error|/|reference| = 0.5` yields `global_rel_l2 ≥ ~6.1e-8` (B's lower bound), NOT `1e-9` as v1 claimed. The test asserts `local_scaled_max >= eta -> FAIL` (catches the localized error), AND `global_rel_l2 < global_rel_l2_threshold` (the localized error is NOT caught by rel_l2 alone) — proving the local gate catches what rel_l2 misses. Use `global_rel_l2 ≈ 6e-8` (or just assert `< 1e-4`).

## Conditional VIABLE (corrected from v1 per B)

v1 stated "G2 verdict stands" unconditionally — too strong. The correct statement:

`region_fused = VIABLE` is a **conditional conclusion** that holds **only if ALL**:
1. **capability OK** (C2_REGION_KERNEL_FEASIBILITY=PASS, MEASURED) — the P1 #2 fix MUST have the region gate reading `runtime_peak_measurement_method` + `runtime_peak_sample_count` + nested `full_anchor_correctness.*`; the capability verdict is re-derived from the fixed gate on the new measurement (NOT inherited from the buggy G2 gate).
2. **numerical PASS** — all 9 frozen holdout cells pass the dual-gate (this policy).
3. **the full evidence package is re-submitted and accepted by reviewer B** (result review).

None of these may be skipped. `region_fused=VIABLE` is PENDING until all three hold.

## Counterexample / mutation tests (TDD RED, implemented with the policy)

The policy implementation MUST be validated by these mutation tests:

1. `test_policy_all_zero_reference_unknown` — reference all zeros (`s=0`) -> `UNKNOWN_ALL_ZERO_REFERENCE` (not PASS).
2. `test_policy_nan_inf_fail` — one NaN in output -> `FAIL_NAN_INF`.
3. `test_policy_reference_nan_fail` — NaN in REFERENCE (not output) -> `FAIL_NAN_INF` (both checked).
4. `test_policy_error_nan_fail` — NaN in `error` propagation -> `FAIL_NAN_INF` (error + metrics checked too).
5. `test_policy_global_rel_l2_fail` — `output = 2*reference` -> `global_rel_l2 ≈ 1.0` -> `FAIL_GLOBAL_REL_L2`.
6. `test_policy_local_scaled_max_localized_error_fail` — a SINGLE high-signal element with `|error|/max(|ref|,τ) = 0.5` but `global_rel_l2 ≈ 6e-8 < 1e-4` (localized error NOT caught by rel_l2) -> `FAIL_LOCAL_SCALED_MAX`. **(Key test: proves the local gate catches what rel_l2 misses.)**
7. `test_policy_shape_mismatch_unknown` — `output.shape != reference.shape` -> `UNKNOWN_SHAPE_MISMATCH`.
8. `test_policy_empty_array_unknown` — `output.size == 0` -> `UNKNOWN_EMPTY_ARRAY`.
9. `test_policy_missing_metric_unknown` — a required metric is None -> `UNKNOWN_MISSING_METRIC` (not PASS).
10. `test_policy_invalid_metric_fail` — a computed metric is non-finite/negative/bool -> `FAIL_INVALID_METRIC`.
11. `test_policy_multiple_reasons_priority` — multiple anomalies -> verdict is highest-priority (FAIL > UNKNOWN > PASS) AND ALL reason codes retained.
12. `test_policy_pass` — all metrics within thresholds -> `PASS`.

(Note: `test_dual_gate_empty_high_signal_mask_unknown` from v1 is **removed** — mathematically impossible to construct for `0<α≤1, s>0`.)

## Deprecated metrics (formal deprecation per B)

The two existing `max_rel` producer definitions are **deprecated** and MUST be removed/replaced by the dual-gate metrics in the re-measurement (do not co-exist):

1. `numerical.py:25-45` `compute_metrics`: `max_rel = max|error| / max(|ref|, 0.5)` (signal_floor=0.5) — **deprecated**. The `max_rel` field in `compute_metrics` output is informational only and MUST NOT be gated on. (Other routes' policies that currently gate `max_rel` — `cutlass` C16BF `max_rel<5e-3` — must be re-justified OR migrated to `local_scaled_max` in a separate policy decision; this spec does NOT touch them.)
2. `region_proto.py:933` `run_full_anchor_correctness`: `max_rel = max|error| / max(1, max|ref|)` (scalar denom) — **deprecated**. The `worst_max_rel` in `full_anchor_correctness` MUST be recomputed under the new policy's `local_scaled_max` definition (or the field removed if the consumer doesn't gate on it; the c2.py `accuracy_state` reads `worst_max_rel` per P1 #2 fix — so it MUST be the new `local_scaled_max` value, with `τ` recomputed from the cell's `s`).

**The v1 diagnosis is a hypothesis, not a verified claim:** the old artifacts did not record which `|reference_i|` produced the max-error element, so "6/9 failures are small-magnitude per-element blow-up" is **unverified**. The re-measurement MUST record, per cell, the `|reference_i|` at the max-error element (a `max_error_reference_abs` field) so the hypothesis can be checked against the data. If the hypothesis fails (the max-error elements are NOT low-signal), the continuous `local_scaled_max` still catches them (it's defined for all magnitudes) — the policy does not depend on the hypothesis being true; only the v1 *diagnosis* does.

## Trust chain (binding per B P1 #3)

1. **Spec in-repo:** this file is committed to `tensorcircuit-ng/` (the git-tracked repo). `git log` binds it.
2. **`POLICY_ACCEPTED`** issued by reviewer B, binding:
   - policy commit SHA-256 (the git commit containing this file)
   - policy ID: `REGION_FUSED_FULL_ANCHOR_ACCURACY_v2`
   - all frozen constants (α, global_rel_l2_threshold, eta)
3. **`policy_freeze_manifest.json`** (see §11) created BEFORE measurement, recording the policy SHA + constants.
4. **Each measurement artifact** records `policy_hash` (= policy commit SHA-256) + `policy_id`.
5. **New `review_subject`** explicitly binds the policy (records `policy_hash`).
6. **Consumers recompose** the verdict from the frozen constants — they do NOT trust a producer-written `policy_pass` boolean. The gate recomputes `global_rel_l2` + `local_scaled_max` from the recorded metrics (or from the raw output/error if recorded) + the frozen constants, and derives the verdict.

## Freeze manifest (`policy_freeze_manifest.json`) — created BEFORE measurement

The freeze manifest is a JSON artifact created BEFORE the re-measurement, binding everything B requires. It is committed (or recorded in run_context) BEFORE the measurement run. Schema:

```
{
  "schema_version": "policy-freeze-manifest-v1",
  "policy_id": "REGION_FUSED_FULL_ANCHOR_ACCURACY_v2",
  "policy_commit_sha256": "<git commit SHA-256 containing this spec file>",
  "policy_file_path": "docs/superpowers/specs/2026-07-26-region-fused-dual-gate-accuracy-policy.md",
  "constants": {
    "alpha": 1e-3,
    "global_rel_l2_threshold": 1e-4,
    "eta": 1e-3
  },
  "metric_schema_version": "dual-gate-v2",
  "metric_definitions": {
    "global_rel_l2": "||error||_2 / ||reference||_2 (FP64, stable accumulation)",
    "local_scaled_max": "max_i |error_i| / max(|reference_i|, alpha*s), s=RMS(reference)",
    "nan_inf": "not all(isfinite(output)) OR not all(isfinite(reference)) OR not all(isfinite(error)) OR not all(isfinite(metrics))"
  },
  "implementation": {
    "source_commit": "<git commit SHA-256 containing the dual-gate compute_metrics implementation>",
    "implementation_file": "results/_phase0/numerical.py (compute_metrics_dual_gate + apply_policy_region_fused)",
    "implementation_test_files": ["results/_phase0/numerical_test.py (policy mutation tests)"]
  },
  "profiles": {
    "required_input_profiles": ["baseline_v1", "mixed_scale_v1", "cancellation_v2"],
    "shape": [64, 1048576],
    "dtype": "complex64",
    "seeds": "<B-specified holdout seeds, OR deterministically derived from policy_commit_sha256>"
  },
  "run_env": {
    "gpu": "RTX 5070 Ti Laptop (sm_120, 12GB)",
    "conda_env": "tcng",
    "package_versions_captured_at_freeze": true
  },
  "retry_and_retention": {
    "max_retries_per_cell": 0,
    "failed_run_retention": "all failed runs retained with reason codes; no silent drop",
    "retry_only_on": "infra failure (OOM/timeout), NOT on policy FAIL"
  },
  "freeze_created_at": "<ISO datetime at freeze creation>",
  "frozen_by": "<agent/agent+controller>"
}
```

**Holdout seeds (per B):** B-specified, OR deterministically derived from `policy_commit_sha256` (e.g. `hash(policy_sha)[:N] mod large_prime` mapped to 3 seeds in a documented range). NEVER chosen after a trial run, never swapped. Document the derivation. (Initial candidate: derive from `sha256(policy_commit)[:8]` interpreted as 3 uint32 seeds in [0, 2^31); B may override.)

**No `976c7892` as measurement commit:** the measurement commit must contain the dual-gate implementation (which does NOT exist in `976c7892`). The actual measurement commit = the freeze manifest's `implementation.source_commit` (or a descendant containing the same implementation). `run_context.measurement.source_commit` MUST equal that, NOT the stale `20589967` (P1 #5 fix) and NOT `976c7892`.

## Process (per B's prescribed flow)

1. **This spec v2** committed to the git repo, submitted to reviewer B for policy review.
2. **B issues `POLICY_ACCEPTED`** (or revisions), binding policy commit SHA-256 + policy ID + constants.
3. **Implement dual-gate** `compute_metrics_dual_gate` + `apply_policy_region_fused` (consuming `local_scaled_max` + `global_rel_l2` + `nan_inf`) in `results/_phase0/numerical.py` + 12 mutation tests. Commit the implementation.
4. **Create `policy_freeze_manifest.json`** binding policy SHA + implementation commit + profile versions + shape + seed list + metric schema version + run-env + retry rules. Commit it. Holdout seeds derived (or B-specified) at this point — frozen.
5. **Re-run all `region_fused` accuracy cells** (3 levels × 3 holdout seeds = 9 cells) with the dual-gate metrics recorded (including `max_error_reference_abs` to verify the v1 hypothesis). `run_context.measurement.source_commit` = the implementation commit (not stale).
6. **Gate** (consumers recompute from frozen constants, NOT producer `policy_pass`): any required cell violating any gate -> route `FAIL`; all cells pass -> numerical `PASS` -> pipeline derives `VIABLE` **only if** capability also OK (P1 #2-fixed gate) AND full evidence re-accepted by B.
7. **New `review_subject`** covering the GPU phase, bound to the new clean commit X + the policy hash, NOT the old `bc6294a` (P1 #1 fix). Submit to B for result review.

If pre-approved policy + new holdout-seed measurement all pass + full evidence accepted by B, `region_fused=VIABLE` is legitimate and acceptable.

## Scope

- This policy applies to **`region_fused` c64 full-anchor numerical ONLY**.
- `planar` / `grouped` / `cutlass_4m_single` numerical policies are UNCHANGED for now.
  - `cutlass` C16BF currently gates `relative_l2<5e-3` + `max_rel<5e-3`. The deprecated `compute_metrics` `max_rel` (signal_floor=0.5) is still used by `cutlass`. **A separate policy decision** (out of scope for this spec) must re-justify or migrate `cutlass`'s `max_rel` gate to `local_scaled_max` — do NOT silently change `cutlass` here. The 1/9 `cutlass` FAIL stands under the current policy.
- The **5 P1 fail-open fixes** (c2.py region gate reads runtime fields + nested `full_anchor_correctness`; gonogo.py binding requires hash + verdict allowlist; numerical.py aggregate strict `source=="measured"`; run_context provenance; new GPU review subject) are SEPARATE remediation items, executed alongside this policy. The c2.py `accuracy_state` reads `full_anchor_correctness.worst_max_rel` (P1 #2 fix) — that field MUST be the new `local_scaled_max` value (or the gate must read the dual-gate metrics directly from the cell).

## Non-goals

- No change to the 3 CUDA kernels (direct/tiled/persistent) — they're correct (rel_l2 8.5e-7).
- No change to G2's MEASURED capability verdict as a *claim* — but the capability is re-derived under the P1 #2 fix (conditional, see §6); if the fixed gate still yields PASS on the new measurement, capability OK stands; else it's UNKNOWN/FAIL honestly.
- No Phase 1 authorization (this resolves region_fused numerical; Phase 1 is a separate decision).
- No pushing (branch stays local).