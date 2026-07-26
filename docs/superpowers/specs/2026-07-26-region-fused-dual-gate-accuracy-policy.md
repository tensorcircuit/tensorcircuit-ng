# Region-fused Full-anchor Numerical Accuracy Policy (Continuous Local-Gate)

> **Status:** DRAFT v3 for reviewer B policy review. v2 was POLICY_NOT_ACCEPTED (2026-07-26) for 4 consistency blockers. v3 fixes: (1) bool/real metric-type distinction; (2) NEW fields (no worst_max_rel overloading) + legacy max_rel scope-limited to region_fused decision chain; (3) Git SHA-1 commit identity (not SHA-256) + distinct policy_git_commit / policy_file_sha256 / implementation_git_commit / measurement_source_commit; (4) freeze manifest binds the actual kernel variant + full contract + D input version.
>
> This file is committed to `tensorcircuit-ng/` (the git-tracked repo, SHA-1) so `git log` binds it. Reviewer B issues `POLICY_ACCEPTED` bound to the policy git commit (SHA-1) + policy file SHA-256 + policy ID + frozen constants. No adjustment based on new results after freeze.

## Math contract (frozen)

- `output` = fused kernel `E` (c64[TM,TN] = c64[64,1048576], 2^26 = 67,108,864 elements)
- `reference` = materialized oracle `E_mat` (same shape, same inputs/seed)
- `error = output - reference`
- `s = RMS(reference) = sqrt(mean(|reference_i|^2))` — global RMS of the reference
- `τ = α * s` — signal scale (α frozen)

### Field-type distinction (P1 #1 fix, reviewer B v2)

**Numerical metrics** (`s`, `global_rel_l2`, `local_scaled_max`, `reference_rms`, `worst_local_scaled_max`, `local_scaled_argmax_reference_abs`): MUST be **finite, non-negative, non-bool real** numbers. Any non-finite, negative, bool, or non-real value -> `FAIL_INVALID_METRIC` (fail-closed).

**Status field** (`nan_inf`): MUST be strictly **bool** (`True`/`False`). Missing `nan_inf` OR non-bool value (`0`/`1`/`"false"`/`None`) -> fail-closed (treated as `nan_inf=True` -> `FAIL_NAN_INF`, since a missing/invalid finiteness state cannot be trusted as finite). A present `nan_inf=False` (proper bool) does NOT trigger `FAIL_INVALID_METRIC` — it is a valid status field, not a numerical metric.

This resolves the v2 contradiction (v2 line 29 "all metrics non-bool real" vs line 36 `nan_inf` bool).

### Numerical guarantees (FP64 / stable accumulation — required)

- **dtype**: `s` (RMS) and `global_rel_l2` accumulated in **FP64**. The complex64 (`c64`) inputs are NOT directly cast to float64 (that would drop imaginary parts); instead `|reference_i|^2 = re(ref)^2 + im(ref)^2` is computed element-wise (preserving both real/imaginary parts), then summed in FP64. Same for `|error_i|`.
- **complex modulus**: `|z| = sqrt(re(z)^2 + im(z)^2)`; `|z|^2 = re(z)^2 + im(z)^2` (no `|z| = |re| + |im|` or other surrogate). Computed in FP64.
- **shape equality**: `output.shape == reference.shape` else `UNKNOWN_SHAPE_MISMATCH` (fail-closed; not PASS).
- **non-empty**: `output.size > 0` else `UNKNOWN_EMPTY_ARRAY`.
- **accumulation method**: use a **blocked pairwise / scaled-sum-of-squares** accumulation in FP64 (e.g. compute `|ref_i|^2` as float64 element-wise, then sum via a numerically stable pairwise or Kahan reduction; NOT an unspecified `numpy.linalg.norm` which v2 over-claimed as stable). DOCUMENT the exact accumulation method used in the implementation. (If `numpy.linalg.norm` is used, justify it specifically for the implementation; do not blanket-assert it's "stable".)
- **finite metrics**: all numerical metrics must be **finite, non-negative, non-bool real** numbers; any non-finite/NaN/Inf/negative/bool/non-real -> `FAIL_INVALID_METRIC` (fail-closed).

## FIELD SCHEMA (P1 #2 fix, reviewer B v2) — NEW distinct fields, no overloading

The dual-gate metrics use **NEW distinct field names**. The old `worst_max_rel` field is NOT overloaded (it keeps its old semantics, deprecated but not redefined). Producers MUST emit the new fields; consumers (c2.py `accuracy_state`) MUST read the new fields; missing new field -> UNKNOWN (not aliased to an old field).

### Per-cell metric fields (produced + recorded in numerical artifacts)

| field | type | definition |
|---|---|---|
| `reference_rms` | numerical | `s = sqrt(mean(|reference_i|^2))`, FP64 |
| `global_rel_l2` | numerical | `\|\|error\|\|_2 / \|\|reference\|\|_2`, FP64 stable accumulation |
| `local_scaled_max` | numerical | `max_i \|error_i\| / max(\|reference_i\|, α·s)`, FP64 |
| `worst_local_scaled_max` | numerical | worst (max) of `local_scaled_max` across the 3 seeds for the cell (per-cell worst; this is the NEW field replacing the role v2 wrongly assigned to `worst_max_rel`) |
| `local_scaled_argmax_reference_abs` | numerical | `\|reference_i\|` at the `i` where `local_scaled_max` is attained (the NEW field replacing v2's ambiguous `max_error_reference_abs`; used to verify the v1 hypothesis about small-magnitude blow-up) |
| `nan_inf` | status (bool) | `not all(isfinite(output)) OR not all(isfinite(reference)) OR not all(isfinite(error)) OR not all(isfinite(numerical_metrics))` |
| `policy_id` | string | `"REGION_FUSED_FULL_ANCHOR_ACCURACY_v3"` (frozen at freeze) |
| `policy_file_sha256` | string | file SHA-256 of the frozen policy spec (64-hex) |
| `metric_schema_version` | string | `"dual-gate-v3"` |

### Consumer (c2.py accuracy_state) MUST read the new fields

`c2.py` `accuracy_state` (the P1 #2 fix reads nested `full_anchor_correctness.*`) MUST read `full_anchor_correctness.worst_local_scaled_max` + `full_anchor_correctness.global_rel_l2` + `full_anchor_correctness.nan_inf` (the NEW fields), NOT `worst_max_rel`. If any new field is missing -> `accuracy_state=MISSING` -> UNKNOWN (fail-closed). **`worst_max_rel` MUST NOT be used as an alias** for `worst_local_scaled_max`.

`full_anchor_correctness` (produced by `run_full_anchor_correctness` in `region_proto.py`) MUST emit the new fields (`reference_rms`, `global_rel_l2`, `local_scaled_max`, `worst_local_scaled_max`, `local_scaled_argmax_reference_abs`) per the new schema, alongside the (deprecated, unchanged-semantics) old `worst_relative_l2`/`worst_max_rel` fields which are retained for audit/history but NOT gated on.

### Legacy `max_rel` scope (P1 #2 fix, reviewer B v2)

The old `max_rel` definitions are **deprecated ONLY in the `region_fused` decision chain** (compute_metrics `max_rel` + region_proto `worst_max_rel` are no longer gated on for `region_fused`). **Other routes (`cutlass_4m_single` C16BF `max_rel<5e-3`, planar, grouped) keep their existing `max_rel` metric under a LEGACY-tagged schema** — explicitly marked legacy, NOT silently removed, NOT migrated here. Migrating `cutlass` to `local_scaled_max` is a **separate policy decision** (out of scope for this spec). The 1/9 `cutlass` FAIL under its current legacy policy stands.

This resolves the v2 conflict (v2 said "old max_rel cannot co-exist" but `cutlass` still uses it).

## Gate (continuous, scale-aware — B v2-accredited)

The continuous-gate formula passed B v2's technical review and the candidate constants are accepted as a pre-approval basis (B v2: "no need to adjust these three constants based on new data"):

```
PASS iff:
    nan_inf == False
    global_rel_l2 < global_rel_l2_threshold
    local_scaled_max < eta
```

where `local_scaled_max = max_i ( |error_i| / max(|reference_i|, α·s) )`, `s = RMS(reference)`, `τ = α·s`.

This is **continuous** (no high/low partition, no 100x jump): it's equivalent to a high-signal rtol=`η` and a low-signal atol=`η·α·s`, smoothly joined. It preserves localized catastrophic-error detection (a single high-signal element with large per-element error is caught, which `global_rel_l2` alone would dilute).

### Classification (corrected, NOT "else FAIL")

The cell verdict is classified per the **fail-closed table** below (NOT the v2 "PASS iff all hold; else FAIL" which contradicted the UNKNOWN rows). Apply deterministic priority **FAIL > UNKNOWN > PASS**: collect ALL triggered reason codes; the verdict is the highest-priority.

| condition | verdict | reason code |
|---|---|---|
| `nan_inf == True` (output + reference + error + numerical_metrics checked; OR `nan_inf` missing/non-bool per §1) | `FAIL` | `FAIL_NAN_INF` |
| any numerical metric non-finite/negative/bool/non-real | `FAIL` | `FAIL_INVALID_METRIC` |
| `global_rel_l2 >= global_rel_l2_threshold` | `FAIL` | `FAIL_GLOBAL_REL_L2` |
| `local_scaled_max >= eta` | `FAIL` | `FAIL_LOCAL_SCALED_MAX` |
| `output.shape != reference.shape` | `UNKNOWN` | `UNKNOWN_SHAPE_MISMATCH` |
| `output.size == 0` | `UNKNOWN` | `UNKNOWN_EMPTY_ARRAY` |
| all-zero reference (`s == 0`) -> metrics undefined | `UNKNOWN` | `UNKNOWN_ALL_ZERO_REFERENCE` |
| a required new metric field is missing/not computed | `UNKNOWN` | `UNKNOWN_MISSING_METRIC` |
| all gates hold | `PASS` | `PASS` |

Priority: `FAIL` if any FAIL reason, else `UNKNOWN` if any UNKNOWN reason, else `PASS`. ALL triggered reason codes retained (list, not single). `UNKNOWN` is fail-closed (not PASS).

### Edge cases (corrected per B v1+v2)

- **"empty high-signal mask"** (v1): when `0 < α ≤ 1` and `s > 0`, `max|reference_i| ≥ s ≥ α·s = τ`, so at least one element satisfies `|reference_i| ≥ τ`. Mathematically never empty. v1's `test_dual_gate_empty_high_signal_mask_unknown` is **removed** — the test cannot be constructed.
- **"empty low-signal mask"** (dual form only; the continuous form has no partition): a legitimate configuration may have all reference elements equal magnitude (empty low-signal). The low-signal gate vacuously passes (not `UNKNOWN`) in that case unless an independent profile contract requires low-signal coverage.
- **localized-error test bound (B v1):** with `N=2^26`, `α=1e-3`, single-element `|error|/|reference| = 0.5` yields `global_rel_l2 ≥ ~6.1e-8` (B's lower bound), NOT `1e-9`. The mutation test asserts `local_scaled_max >= eta -> FAIL` (catches the localized error), AND `global_rel_l2 < global_rel_l2_threshold` (the localized error is NOT caught by rel_l2 alone) — proving the local gate catches what rel_l2 misses. Use `global_rel_l2 ≈ 6e-8` (or just assert `< 1e-4`).

## Candidate constants (B v2-accredited, for freeze)

| constant | candidate | meaning |
|---|---|---|
| `α` (signal scale) | `1e-3` | clamp denominator to `α·s` |
| `global_rel_l2_threshold` | `1e-4` | canonical global L2 ratio (unchanged) |
| `eta` (`local_scaled_max` threshold) | `1e-3` | per-element error cap, scaled by `max(|ref|, α·s)` |

B v2: "no need to adjust these three constants based on new data." `POLICY_ACCEPTED` freezes them.

## Conditional VIABLE (corrected from v1/v2)

`region_fused = VIABLE` is a **conditional conclusion** holding **only if ALL**:
1. **capability OK** (C2_REGION_KERNEL_FEASIBILITY=PASS, MEASURED) — the P1 #2 fix MUST have the region gate reading `runtime_peak_measurement_method` + `runtime_peak_sample_count` + the NEW nested `full_anchor_correctness.{worst_local_scaled_max, global_rel_l2, nan_inf}` fields; the capability verdict is re-derived from the fixed gate on the new measurement (NOT inherited from the buggy G2 gate).
2. **numerical PASS** — all 9 frozen holdout cells pass the dual-gate (this policy), specifically for the **direct `fused_pte_kernel` variant** bound in the freeze manifest.
3. **full evidence package re-submitted and accepted by reviewer B** (result review).

None may be skipped. `region_fused=VIABLE` is PENDING until all three hold.

### Variant-scoping (per B v2 P1 #4)

The freeze manifest binds the **specific kernel variant** (direct `fused_pte_kernel` — the one numerical uses, see `numerical.py:1090`). The numeric PASS certifies **`region_fused/direct`**. The **persistent** and **tiled** variants need their own precision evidence before they can inherit the VIABLE conclusion. If the policy certifies only direct, the closeout MUST write `region_fused/direct = VIABLE` (not blanket `region_fused = VIABLE`); persistent requires its own measured numerical cells.

## Trust chain (P1 #3 fix, reviewer B v2 — Git SHA-1, not SHA-256)

The git repo (`tensorcircuit-ng/.git`) uses **SHA-1** (40-hex commit IDs). v2 wrongly required "Git commit SHA-256." v3 binds TWO distinct objects (file content hash is SHA-256; git commit identity is SHA-1):

```
{
  "policy_git_commit": "b97b63c64159c95fde83cb4abd579d7b08a45ee9 (40-hex git SHA-1)",
  "policy_file_sha256": "897E955A9BA57AD90CAC2E06CFAD658A11FD1DF9B3BC7E3EB5704E3A73452979 (64-hex file content)",
  "policy_file_path": "docs/superpowers/specs/2026-07-26-region-fused-dual-gate-accuracy-policy.md"
}
```

Distinct identities (do NOT conflate):
- `policy_git_commit` (40-hex SHA-1): the git commit containing the frozen policy file.
- `policy_file_sha256` (64-hex): the SHA-256 of the policy file's content (byte-exact, for content binding).
- `implementation_git_commit` (40-hex SHA-1): the git commit containing the dual-gate `compute_metrics_dual_gate` + `apply_policy_region_fused` implementation + tests.
- `implementation_file_sha256` (64-hex): SHA-256 of the implementation source file(s) content.
- `measurement_source_commit` (40-hex SHA-1): the **exact commit F checked out at measurement time** (the frozen measurement commit). This is the commit where the measurement run happened; it MUST equal the freeze manifest's `measurement_source_commit` (the commit F created by the freeze), NOT the earlier `implementation_git_commit` I. `run_context.measurement.source_commit` MUST record F (the runtime frozen commit), NOT I (implementation), NOT stale `20589967`.

**POLICY_ACCEPTED** (reviewer B) binds: `policy_git_commit` (SHA-1) + `policy_file_sha256` + `policy_id` (`REGION_FUSED_FULL_ANCHOR_ACCURACY_v3`) + all frozen constants + this freeze manifest schema.

**Each measurement artifact** records `policy_git_commit` (SHA-1) + `policy_file_sha256` + `policy_id` + `metric_schema_version`.

**`review_subject`** explicitly binds the policy (`policy_git_commit` + `policy_file_sha256`).

**Consumers recompute** the verdict from the frozen constants (NOT trusting a producer-written `policy_pass`): the gate recomputes `global_rel_l2` + `local_scaled_max` from the recorded `reference_rms` + raw error/reference (if recorded) + frozen `α`/thresholds, and derives the verdict.

## Counterexample / mutation tests (TDD RED, implemented with the policy)

The policy implementation MUST be validated by these mutation tests:

1. `test_policy_all_zero_reference_unknown` — reference all zeros (`s=0`) -> `UNKNOWN_ALL_ZERO_REFERENCE` (not PASS).
2. `test_policy_nan_inf_true_fail` — `nan_inf=True` (proper bool) -> `FAIL_NAN_INF`.
3. `test_policy_nan_inf_missing_fail` — `nan_inf` field missing -> fail-closed (`FAIL_NAN_INF`, treated True — a missing finiteness state cannot be trusted finite).
4. `test_policy_nan_inf_nonbool_fail` — `nan_inf=0` or `1` or `"false"` (non-bool) -> `FAIL_NAN_INF` (fail-closed; non-bool status field invalid).
5. `test_policy_reference_nan_fail` — NaN in REFERENCE (not output) -> `FAIL_NAN_INF` (both checked).
6. `test_policy_error_nan_fail` — NaN in `error` propagation -> `FAIL_NAN_INF` (error + metrics checked too).
7. `test_policy_invalid_numerical_metric_fail` — a numerical metric (`global_rel_l2`) is non-finite/negative/bool -> `FAIL_INVALID_METRIC` (NOT `FAIL_NAN_INF`; the status/metric distinction is tested).
8. `test_policy_global_rel_l2_fail` — `output = 2*reference` -> `global_rel_l2 ≈ 1.0` -> `FAIL_GLOBAL_REL_L2`.
9. `test_policy_local_scaled_max_localized_error_fail` — a SINGLE high-signal element with `|error|/max(|ref|,τ) = 0.5` but `global_rel_l2 ≈ 6e-8 < 1e-4` (localized error NOT caught by rel_l2) -> `FAIL_LOCAL_SCALED_MAX`. **(Key test: proves the local gate catches what rel_l2 misses.)**
10. `test_policy_shape_mismatch_unknown` — `output.shape != reference.shape` -> `UNKNOWN_SHAPE_MISMATCH`.
11. `test_policy_empty_array_unknown` — `output.size == 0` -> `UNKNOWN_EMPTY_ARRAY`.
12. `test_policy_missing_new_metric_field_unknown` — a required NEW field (`local_scaled_max` or `worst_local_scaled_max`) is None/missing -> `UNKNOWN_MISSING_METRIC` (NOT aliased to old `worst_max_rel`).
13. `test_policy_no_worst_max_rel_alias` — a fixture with ONLY old `worst_max_rel` (no new `worst_local_scaled_max`) -> `UNKNOWN_MISSING_METRIC` (c2.py must NOT read `worst_max_rel` as an alias for the new field).
14. `test_policy_multiple_reasons_priority` — multiple anomalies -> verdict is highest-priority (FAIL > UNKNOWN > PASS) AND ALL reason codes retained.
15. `test_policy_pass` — all metrics within thresholds, all new fields present, `nan_inf=False` (proper bool) -> `PASS`.

(Note: v1's `test_dual_gate_empty_high_signal_mask_unknown` is **removed** — mathematically impossible for `0<α≤1, s>0`.)

## Freeze manifest (`policy_freeze_manifest.json`) — committed BEFORE measurement (P1 #4 fix)

The freeze manifest is a JSON artifact created and **committed BEFORE the re-measurement**, binding everything B requires (including the actual kernel variant + full contract + D input version + seed derivation). It is NOT generated by the run — `run_context` records the runtime state but does NOT replace the freeze manifest (the freeze manifest is the pre-measurement contract; `run_context` is the runtime record).

Schema (v3):

```
{
  "schema_version": "policy-freeze-manifest-v1",
  "policy_id": "REGION_FUSED_FULL_ANCHOR_ACCURACY_v3",
  "policy_git_commit": "<40-hex git SHA-1 of the commit containing this spec>",
  "policy_file_sha256": "<64-hex SHA-256 of the spec file content>",
  "policy_file_path": "docs/superpowers/specs/2026-07-26-region-fused-dual-gate-accuracy-policy.md",
  "constants": {
    "alpha": 1e-3,
    "global_rel_l2_threshold": 1e-4,
    "eta": 1e-3
  },
  "metric_schema_version": "dual-gate-v3",
  "metric_definitions": {
    "reference_rms": "sqrt(mean(|reference_i|^2)) FP64",
    "global_rel_l2": "||error||_2 / ||reference||_2 (FP64 blocked pairwise/scaled sum-of-squares; |z|^2=re^2+im^2, no float64 cast of c64)",
    "local_scaled_max": "max_i |error_i| / max(|reference_i|, alpha*s) FP64",
    "worst_local_scaled_max": "max of local_scaled_max across seeds (per cell)",
    "local_scaled_argmax_reference_abs": "|reference_i| at local_scaled_max argmax",
    "nan_inf": "bool: not all(isfinite(output)) OR not all(isfinite(reference)) OR not all(isfinite(error)) OR not all(isfinite(numerical_metrics))"
  },
  "implementation": {
    "implementation_git_commit": "<40-hex git SHA-1 of the commit containing compute_metrics_dual_gate + apply_policy_region_fused>",
    "implementation_file_sha256": "<64-hex content SHA-256 of the impl source file(s)>",
    "implementation_file": "results/_phase0/numerical.py (compute_metrics_dual_gate + apply_policy_region_fused)",
    "implementation_test_files": ["results/_phase0/numerical_test.py (policy mutation tests)"]
  },
  "kernel_variant": {
    "variant": "direct",
    "kernel_name": "fused_pte_kernel",
    "kernel_source_path": "results/_phase0/cpp/region_proto.cu",
    "kernel_source_sha256": "<64-hex content SHA-256 of the kernel source at measurement time>",
    "kernel_blob_or_ptx_sha256": "<64-hex of compiled kernel blob/ptx if reproducible; else null + reason>",
    "reason": "G5 numerical uses the direct-recompute fused_pte_kernel (numerical.py:1090), NOT tiled/persistent. The numeric PASS certifies region_fused/direct only."
  },
  "contract": {
    "PM": 4096, "PN": 16384, "K1": 1024, "TM": 64, "TN": 1048576,
    "transform": "P=c64[PM,PN]=A@B -> T=transform(P)=c64[TM,TN] (8-D reshape->transpose->reshape, row-major) -> E=D@T=c64[TM,TN]",
    "transform_contract_version": "<transform steps version/c2_c2_edge_map hash>",
    "transform_contract_sha256": "<64-hex of the frozen transform steps contract>",
    "D_input_construction_version": "<baseline_v1 / mixed_scale_v1 / cancellation_v2>",
    "D_seed_offset": "<seed offset for D input construction, if any; documented>"
  },
  "profiles": {
    "required_input_profiles": ["baseline_v1", "mixed_scale_v1", "cancellation_v2"],
    "shape": [64, 1048576],
    "dtype": "complex64"
  },
  "seeds": {
    "derivation": "B-specified OR deterministic from policy_git_commit/policy_file_sha256 (NOT post-hoc, NOT 8-byte truncation). Use >=12 digest bytes, define endianness + range mapping + dedup.",
    "seed_list": "<3 holdout seeds, derived or B-specified; frozen here>"
  },
  "run_env": {
    "gpu": "RTX 5070 Ti Laptop (sm_120, 12GB)",
    "env_deps_sha256": "<64-hex SHA-256 of environment/dependency manifest (NOT hardcoded local conda env name)>",
    "package_versions_captured_at_freeze": true
  },
  "retry_and_retention": {
    "max_retries_per_cell": "0 (zero retries) OR a fixed count B approves; if fixed count, ALL attempts retained with their reason codes (no silent drop of any attempt)",
    "retry_only_on": "infra failure (OOM/timeout), NEVER on policy FAIL (a policy FAIL is a final measurement result)"
  },
  "freeze_created_at": "<ISO datetime at freeze creation>",
  "frozen_by": "<agent/agent+controller>",
  "measurement_source_commit": "<40-hex git SHA-1 F: the exact commit checked out at measurement time; recorded here pre-measurement, verified equal to run_context.measurement.source_commit post-measurement>"
}
```

### Seed derivation (per B v2 minor)

`sha256(policy_commit)[:8]` produces only one uint32 — insufficient for 3 seeds. v3: use **≥12 digest bytes** (e.g. `sha256(policy_git_commit || policy_file_sha256)[:12]` interpreted as 3 uint32 via documented endianness [little-endian], mapped to [0, 2^31) with dedup if collisions). **OR (preferred per B) B provides a nonce/seed at POLICY_ACCEPTED time** — avoiding commit-hash grinding entirely. Document the derivation in the freeze manifest. Seeds are frozen; never swapped after a trial run.

### kernel_variant binding (per B v2 P1 #4)

The freeze manifest MUST bind the actual kernel variant numerical uses (`direct` / `fused_pte_kernel` from `numerical.py:1090`), NOT blanket all 3. The numeric PASS certifies `region_fused/direct`; tiled/persistent need separate evidence. The closeout MUST write `region_fused/direct = VIABLE` (variant-scoped), not blanket `region_fused = VIABLE`.

### run_env (per B v2 minor)

Do NOT hardcode local conda env name (`tcng`) in tracked artifacts. Use an **environment/dependency manifest SHA-256** (e.g. a hash of the captured package-version list) for reproducibility, not env names.

## Process (per B's prescribed flow)

1. **This spec v3** committed to the git repo, submitted to reviewer B for policy review.
2. **B issues `POLICY_ACCEPTED`** binding `policy_git_commit` (SHA-1) + `policy_file_sha256` + `policy_id` + all constants. (B may provide the nonce/seed.)
3. **Implement dual-gate** `compute_metrics_dual_gate` + `apply_policy_region_fused` (consuming NEW fields `local_scaled_max` + `global_rel_l2` + `nan_inf`, NOT old `worst_max_rel`) + 15 mutation tests in `results/_phase0/numerical.py`. Update `region_proto.run_full_anchor_correctness` to emit the NEW `full_anchor_correctness` fields (`reference_rms`, `global_rel_l2`, `local_scaled_max`, `worst_local_scaled_max`, `local_scaled_argmax_reference_abs`). Update c2.py `accuracy_state` (P1 #2 fix) to read the NEW fields. Commit.
4. **Create + commit `policy_freeze_manifest.json`** BEFORE measurement, binding policy SHA-1 + policy file SHA-256 + impl commit + kernel variant (direct) + full contract (PM/PN/K1/TM/TN + transform + D version) + holdout seeds + env deps SHA + retry rules + `measurement_source_commit` (the upcoming measurement commit F, recorded pre-measurement). Freeze manifest is the pre-measurement contract.
5. **Re-run all `region_fused` accuracy cells** (3 levels × 3 holdout seeds = 9 cells) with the dual-gate metrics recorded (including `local_scaled_argmax_reference_abs` to verify the v1 hypothesis). `run_context.measurement.source_commit` MUST equal the freeze manifest's `measurement_source_commit` (the commits match exactly).
6. **Gate** (consumers recompute from frozen constants, NOT producer `policy_pass`): any required cell violating any gate -> route `FAIL`; all cells pass (for the direct variant) -> numerical `PASS` for `region_fused/direct` -> pipeline derives `region_fused/direct = VIABLE` **only if** capability also OK (P1 #2-fixed gate) AND full evidence re-accepted by B.
7. **New `review_subject`** covering the GPU phase, bound to the new clean commit X + the policy (`policy_git_commit` + `policy_file_sha256`), NOT the old `bc6294a` (P1 #1 fix). Submit to B for result review.

If pre-approved policy + new holdout-seed measurement all pass + full evidence accepted by B, `region_fused/direct = VIABLE` is legitimate and acceptable.

## Scope

- This policy applies to **`region_fused` c64 full-anchor numerical ONLY**, specifically the **direct `fused_pte_kernel` variant** (variant-scoped).
- `planar` / `grouped` / `cutlass_4m_single` numerical policies are UNCHANGED. Legacy `max_rel` is retained (explicitly tagged legacy) for `cutlass` C16BF `max_rel<5e-3`; migrating `cutlass` to `local_scaled_max` is a separate policy decision (out of scope). The 1/9 `cutlass` FAIL stands under the legacy policy.
- The **5 P1 fail-open fixes** (c2.py region gate reads runtime fields + NEW nested `full_anchor_correctness` fields; gonogo.py binding requires hash + verdict allowlist; numerical.py aggregate strict `source=="measured"`; run_context provenance; new GPU review subject) are SEPARATE remediation items, executed alongside this policy. The c2.py `accuracy_state` reads the NEW `full_anchor_correctness.{worst_local_scaled_max, global_rel_l2, nan_inf}` fields (P1 #2 fix, per the §2 schema).

## Non-goals

- No change to the 3 CUDA kernels (direct/tiled/persistent) — they're correct (rel_l2 8.5e-7).
- No change to G2's MEASURED capability verdict as a *claim* — but the capability is re-derived under the P1 #2 fix (conditional, see §6); the fixed gate re-derived on the new measurement determines capability OK.
- No Phase 1 authorization (this resolves region_fused/direct numerical; Phase 1 is a separate decision).
- No pushing (branch stays local).