# Phase 0 Non-GPU Rereview Closeout (v3)

**Date:** 2026-07-25
**Plan:** `docs/superpowers/plans/2026-07-24-phase0-nongpu-evidence-integrity-remediation-plan-v2.md`
**Spec:** `docs/superpowers/specs/2026-07-24-anti-cycle4-scope-reset-spec.md`
**Review Spec:** `docs/superpowers/specs/2026-07-24-phase0-nongpu-evidence-integrity-plan-v3-review-spec.md` (8 P1 findings)
**Branch:** `feat/contraction-algebra-tropical`
**Scope:** NON-GPU. v2 GPU measurement 未执行；full-anchor region 未执行；独立复审未执行。

## Honest terminal state

```
phase0_completion = INCONCLUSIVE
phase1_authorization = NOT_AUTHORIZED
planar = UNKNOWN
grouped = NOT_VIABLE
region_fused = UNKNOWN
cutlass_4m_single = UNKNOWN
self_verdict = PENDING_EXTERNAL_REVIEW
```

No route upgraded to PASS/VIABLE. No APPROVED/merge-ready token.

## Findings -> fix mapping (8 findings, v3 review spec)

| # | Finding (v3 spec) | Fix commit | Regression test(s) | Status |
|---|---|---|---|---|
| 4.1 | Cancellation migration trusts old labels (measured->v2) | `7286bdde` | `numerical_test.py` regen_no_gpu_zero_v2_measured_and_legacy_kept, legacy_does_not_satisfy_v2_required | FIXED -- all old measured cancellation -> legacy_v1; v2 measured == 0 |
| 4.2 | Baseline/mixed-scale producer missing version token | `7286bdde` | `numerical_test.py` emit_not_run_rows_handles_7_tuple, canonical_token_is_cancellation_v2 | FIXED -- baseline_v1 / mixed_scale_v1 / cancellation_v2 unified token |
| 4.3 | Region GateContract missing accuracy/resource states | `e5afadea` | `gonogo_test.py` region proto canonical tests; `c2_test.py` region layer tests | FIXED -- accuracy_state + resource_state in GateContract; missing/failed -> not PASS |
| 4.4 | Region case binding source undefined (no real data) | `e5afadea` | `c2_test.py` binding verification tests; `gonogo_test.py` region reader tests | FIXED -- binding from actual comparison (c2._binding_problems); proto self-report rejected |
| 4.5 | Numerical binding schema weak (empty key set -> PASS) | `f5ea1af7` | `numerical_test.py` / `manifest_test.py` hash length/missing/short/non-hex/None tests | FIXED -- 64-char hex validation; required key set verification; algorithm=sha256 required |
| 4.6 | CUTLASS blocker/source not in GateContract | `a01382f6` | `gonogo_test.py` CUTLASS native/fallback canonical tests; blocker source allowlist | FIXED -- blocker_state + blocker_source_state in CUTLASS_NATIVE contract; recognized source required |
| 4.7 | run-context v1 measurement migration missing; command not executable | `2a1a3a8d` | `manifest_test.py` provenance tests; run-context migration tests | FIXED -- v1 source_commit migrated to measurement role; aggregation command records real --regen-no-gpu |
| 4.8 | Review-subject validator doesn't recompute internal hashes | `d12f6644` | `review_subject_test.py` Git tree X recompute; `derived_status_test.py` positive/negative/e2e | FIXED -- 5 hashes recomputed from Git tree X + workspace docs; stale review detected |

## Invariant status

| Invariant | Description | Status |
|---|---|---|
| INV-1 | non-GPU round `cancellation_v2` MEASURED row count == 0 | VERIFIED (0 v2 measured rows) |
| INV-2 | exact v2 NOT_RUN key set matches expected v2 keys | VERIFIED (cell key equality) |
| INV-3 | all canonical readers call GateContract; no ad-hoc PASS branches | VERIFIED (grouped/CUTLASS/region via evaluate_gate) |
| INV-4 | GateContract is single executable semantic source | VERIFIED (normative_policy.json stores constants only) |
| INV-5 | doc references workspace-root-relative; absolute/escape rejected | VERIFIED (closeout_facts.validate_doc_references) |
| INV-6 | doc hashes validated against actual file content | VERIFIED (sha256 recompute in closeout_facts) |

## Re-aggregation result (this session)

Regenerated via producers in dependency order (no GPU):
1. numerical regen (no-GPU, legacy migration applied) -- `numerical_validation.csv` + `.json`
2. gonogo regen (v2 schema, canonical readers, GateContract) -- `gonogo.json` + `gonogo.md`
3. run_context build (v2, measurement role preserved) -- `run_context.json`
4. manifest build (provenance, hashes, criteria) -- `manifest.json`
5. test_report (stdlib wrapper) -- `test_report.json`
6. closeout_facts -- `closeout_facts.json`

C3_GROUPED restored to NOT_SUPPORTED (grouped artifact v2 schema reader returns canonical token).
CUTLASS criteria restored to canonical UNKNOWN (fallback attempted, native blocked, both computed via GateContract).

## Remaining (NOT this closeout's scope)

- GPU Task 2b (full-anchor region kernel), Task 3b (numerical re-measure) -- gated, pending user authorization
- Independent external review (Task 9 handoff Y)
- Final Phase 0 clean rerun + final `rereview_closeout.md` (after GPU tasks)
