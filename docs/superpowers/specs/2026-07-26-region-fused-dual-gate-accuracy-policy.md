# Region-Fused Full-Anchor Dual-Gate Accuracy Policy

Status: **DRAFT v5 — requires independent Reviewer B `POLICY_ACCEPTED` before any qualifying remeasurement.**

This policy applies only to the c64 full-anchor `region_fused/direct` path implemented by `fused_pte_kernel`. It does not authorize Phase 1 and does not change the accuracy policy of planar, grouped, CUTLASS, tiled, or persistent variants.

## 1. Motivation and frozen constants

The legacy elementwise `max_rel` divides by each reference element and is unstable for very small reference outputs. Removing it after seeing failures was not acceptable evidence. The replacement therefore uses a pre-reviewed, continuous local normalization together with the existing global relative-L2 check.

Reviewer B previously accepted these constants in principle; v5 does not change them:

- `alpha = 1e-3`
- `global_rel_l2_threshold = 1e-4`
- `eta = 1e-3`

All inequalities are strict. Equality with a threshold fails.

For reference `r`, output `o`, error `e = o - r`, and `n = r.size`:

```text
s = sqrt(sum_i |r_i|^2 / n)
global_rel_l2 = ||e||_2 / ||r||_2
local_scaled_max = max_i |e_i| / max(|r_i|, alpha * s)
```

Complex magnitudes use `|z|^2 = real(z)^2 + imag(z)^2`. Reductions use FP64 scaled/pairwise sum-of-squares; casting a c64 array to float64 and discarding the imaginary component is forbidden.

Per-cell pass condition:

```text
nan_inf is exactly False
AND reference size > 0
AND reference RMS s > 0
AND global_rel_l2 < 1e-4
AND local_scaled_max < 1e-3
```

NaN, Inf, negative, boolean, non-real, or missing numerical metrics fail closed. Empty, shape-mismatched, or all-zero-reference inputs are `UNKNOWN`, never `PASS`.

## 2. Measurement cells and field schema

One accuracy cell is `(input_profile, seed)` at the frozen full anchor:

- `P = A[4096,1024] @ B[1024,16384]`, c64
- `T = transform(P)`, c64 `[64,1048576]`
- `E = D[64,64] @ T`, c64 `[64,1048576]`
- candidate kernel: `region_fused/direct`, CUDA symbol `fused_pte_kernel`
- oracle: materialized c64 `P -> T -> E` using identical `A/B/D`

Required input profiles:

- `baseline_v1`
- `mixed_scale_v1`
- `cancellation_v2`

Every measured cell records these new fields; the old `worst_max_rel` remains legacy diagnostic data and is never an alias:

```json
{
  "reference_rms": "finite non-negative real",
  "global_rel_l2": "finite non-negative real",
  "local_scaled_max": "finite non-negative real",
  "local_scaled_argmax_reference_abs": "finite non-negative real or null only when the metric is unavailable",
  "nan_inf": "strict bool",
  "policy_id": "REGION_FUSED_FULL_ANCHOR_ACCURACY_v5",
  "policy_file_sha256": "64-hex SHA-256",
  "metric_schema_version": "dual-gate-v5"
}
```

`numerical_validation.csv` must persist these fields with sufficient precision for a lossless policy decision. `regen-no-gpu` must read them back and recompute the gate; it must not derive them from legacy columns or trust producer-written `policy_pass`.

## 3. Required calibration and holdout coverage

The known diagnostic failures from seeds `0,1,2` cannot be erased by switching to new seeds. They are frozen as calibration seeds and must be remeasured under the final policy implementation.

Reviewer B also supplies or approves three blind holdout seeds. The freeze manifest contains:

```json
{
  "calibration_seed_list": [0, 1, 2],
  "holdout_seed_list": "<three unique B-approved seeds>",
  "required_seed_list": "<the six unique seeds above, in frozen order>"
}
```

The qualifying matrix is therefore exactly:

```text
3 required profiles x 6 required seeds = 18 required cells
```

No seed may be removed, replaced, or retried because it failed the policy. Holdout seeds are non-negative integers in `[0, 2^31)`, distinct from one another and from `0,1,2`.

If B chooses deterministic derivation instead of directly providing the seeds, the manifest must define the complete SHA-256 input bytes, byte order, range mapping, collision handling, and use at least 12 digest bytes. B-provided seeds are preferred because they avoid commit-hash grinding.

## 4. Summary schema and consumers

Summary is a pure reduction over the already-recorded exact 18-cell set. A single-cell collector must not launch hidden extra seeds, depend on call order, or use a module-global cache.

The two maxima are tracked independently and may have different cell keys:

```json
{
  "summary_complete": true,
  "n_cells_expected": 18,
  "n_cells_measured": 18,
  "required_seed_list": "<six frozen seeds>",
  "required_input_profiles": ["baseline", "mixed_scale", "cancellation"],
  "worst_global_rel_l2": "max of global_rel_l2 over all 18 cells",
  "worst_global_rel_l2_cell_key": "<profile:version:seed>",
  "worst_local_scaled_max": "max of local_scaled_max over all 18 cells",
  "worst_local_scaled_max_cell_key": "<profile:version:seed>",
  "any_nan_inf": "OR of strict per-cell nan_inf",
  "policy_id": "REGION_FUSED_FULL_ANCHOR_ACCURACY_v5",
  "policy_file_sha256": "<64-hex>",
  "metric_schema_version": "dual-gate-v5"
}
```

Missing, duplicate, extra-as-substitute, invalid, or wrong-policy cells make `summary_complete` false and the route `UNKNOWN`. Extra diagnostic rows may be retained, but they cannot replace required cells or participate in the qualifying summary.

Consumers have two obligations:

1. `numerical.aggregate` recomputes every required cell using `global_rel_l2`, `local_scaled_max`, and `nan_inf`. It never trusts `policy_pass` and never falls back to `relative_l2`/`max_rel` for `region_fused`.
2. `c2.py` reads only the v5 `full_anchor_correctness` summary, verifies policy identity and exact 18-cell coverage, validates both independent maxima, and recomputes the two thresholds. `worst_max_rel` is not an alias.

Route result:

- any required cell `FAIL` -> numerical `FAIL`
- no failures but any required cell missing/unknown/invalid -> `UNKNOWN`
- all 18 required cells pass -> numerical `PASS`

`region_fused/direct = VIABLE` is possible only when this numerical result is `PASS`, capability is independently `OK`, all artifact bindings match, and Reviewer B accepts the measurement result. A producer self-report that disagrees with recomputation is `CONFLICT -> UNKNOWN`.

## 5. Known pre-policy diagnostic

The current artifact reports approximately:

```text
worst_global_rel_l2    = 8.50e-7
worst_local_scaled_max = 2.08e-3
```

Those values are useful diagnostics but are not qualifying v5 evidence: they cover only the old three baseline seeds and predate the v5 freeze. Because the local value exceeds `eta`, the old cells must be rerun as calibration cells; they may not be discarded in favor of holdouts.

## 6. Policy acceptance and freeze manifest

Reviewer B's `POLICY_ACCEPTED` token binds:

- `policy_git_commit`: 40-hex Git SHA-1 of the commit containing this exact spec
- `policy_file_sha256`: SHA-256 of the Git-blob bytes returned by `git show <policy_git_commit>:docs/superpowers/specs/2026-07-26-region-fused-dual-gate-accuracy-policy.md` (not platform-dependent working-tree line endings)
- `policy_id = REGION_FUSED_FULL_ANCHOR_ACCURACY_v5`
- `metric_schema_version = dual-gate-v5`
- all constants, formulas, coverage rules, and this freeze-manifest schema

The freeze manifest is committed before measurement and contains at least:

```json
{
  "schema_version": "policy-freeze-manifest-v2",
  "policy_id": "REGION_FUSED_FULL_ANCHOR_ACCURACY_v5",
  "policy_git_commit": "<B-accepted policy commit SHA-1>",
  "policy_file_sha256": "<B-accepted file SHA-256>",
  "policy_file_path": "docs/superpowers/specs/2026-07-26-region-fused-dual-gate-accuracy-policy.md",
  "metric_schema_version": "dual-gate-v5",
  "constants": {
    "alpha": 0.001,
    "global_rel_l2_threshold": 0.0001,
    "eta": 0.001
  },
  "implementation": {
    "implementation_git_commit": "<I: commit containing implementation and tests>",
    "implementation_file_sha256": "<SHA-256 of implementation source set>"
  },
  "kernel_variant": {
    "variant": "direct",
    "kernel_name": "fused_pte_kernel",
    "kernel_source_path": "results/_phase0/cpp/region_proto.cu",
    "kernel_source_sha256": "<64-hex>",
    "kernel_blob_or_ptx_sha256": "<64-hex or null plus reason>"
  },
  "contract": {
    "PM": 4096,
    "PN": 16384,
    "K1": 1024,
    "TM": 64,
    "TN": 1048576,
    "transform_contract_sha256": "<64-hex>",
    "D_seed_offset": 7000
  },
  "profiles": {
    "required_input_profiles": ["baseline_v1", "mixed_scale_v1", "cancellation_v2"]
  },
  "seeds": {
    "calibration_seed_list": [0, 1, 2],
    "holdout_seed_list": "<three frozen seeds>",
    "required_seed_list": "<six frozen seeds>"
  },
  "run_env": {
    "gpu": "<device and compute capability>",
    "env_deps_sha256": "<64-hex dependency-manifest hash>"
  },
  "retry_and_retention": {
    "max_retries_per_cell": "<B-approved fixed integer>",
    "retry_only_on": ["OOM", "timeout", "infrastructure failure"],
    "retain_all_attempts": true,
    "policy_failure_is_retriable": false
  }
}
```

Do not put the future freeze commit SHA in its own manifest. The non-self-referential flow is:

1. `I` is the clean implementation commit.
2. Commit the manifest on top of `I`; that commit is `F`. The manifest binds `I`, not `F`.
3. Verify `git show F:<manifest-path>` before measuring.
4. Run all 18 cells from checkout `F`.
5. Record `run_context.measurement.source_commit = F` at runtime.
6. Verify the run-context commit equals the checked-out `F` and that all artifact policy hashes equal the accepted policy.

The manifest must not contain `measurement_source_commit`; that would recreate the self-reference bug.

## 7. Required mutation and integration tests

At minimum, tests cover:

1. all-zero reference -> `UNKNOWN_ALL_ZERO_REFERENCE`
2. empty input and shape mismatch -> `UNKNOWN`
3. output/reference/error NaN or Inf -> `FAIL_NAN_INF`
4. missing or non-bool `nan_inf` -> fail closed
5. negative, bool, NaN, Inf, or non-real metric -> `FAIL_INVALID_METRIC`
6. global-L2 failure
7. localized high-signal error where global L2 passes but local gate fails
8. exact-threshold equality fails for both gates
9. old `worst_max_rel` alone cannot satisfy a v5 field
10. independent global/local worst values retain different cell keys
11. a non-default explicit seed set proves there is no hard-coded `0,1,2` collector cache
12. missing/duplicate required cell makes the summary incomplete
13. dual-gate fields survive CSV write/read without changing the verdict
14. aggregate ignores producer `policy_pass` and recomputes v5 metrics
15. C2 rejects missing policy identity, wrong schema, malformed seed lists, and any coverage other than exactly 18 required cells
16. the current pre-freeze artifact remains `UNKNOWN`, never conditionally accepted by a permissive test

## 8. Execution sequence

1. Commit v5 policy and implementation changes as clean commit `I`.
2. Submit only the policy to independent Reviewer B.
3. Stop unless B returns `POLICY_ACCEPTED` binding the policy commit and file hash.
4. Create and commit `policy_freeze_manifest.json` as `F`.
5. Run the exact 18-cell matrix from `F`, retaining every attempt.
6. Regenerate numerical, prototype, C2, gonogo, manifest, and closeout artifacts from the recorded measurements.
7. Create a new clean review subject binding the result commit, policy commit/hash, freeze manifest, and run context.
8. Submit the results to Reviewer B. Only B's result acceptance can change the pending external-review state.

## 9. Non-goals

- No threshold changes after any v5 measurement begins.
- No claim that tiled or persistent variants inherit direct-kernel accuracy evidence.
- No deletion or reinterpretation of the known `2.08e-3` diagnostic.
- No Phase 1 authorization.
- No push or remote publication requirement.
