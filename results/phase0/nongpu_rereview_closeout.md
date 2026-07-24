# Phase 0 Non-GPU Rereview Closeout

**Date:** 2026-07-24
**Plan:** `docs/superpowers/plans/2026-07-24-phase0-nongpu-remediation-plan.md`
**Spec:** `docs/superpowers/specs/2026-07-24-phase0-nongpu-rereview-spec.md` (11 findings §3.1-§3.11)
**Branch:** `feat/contraction-algebra-tropical`
**Base:** `83263a02` (Task 0 RED tests) -> **Head:** `cefbc056` (Task 8) -> no-GPU re-aggregation (this commit)
**Scope:** NON-GPU only. This is NOT the final Phase 0 `rereview_closeout.md` — the GPU tasks 2b/3b and the final clean rerun remain (2026-07-23 plan).

## Honest terminal state (unchanged by re-aggregation — verified)

```
phase0_completion = INCONCLUSIVE
phase1_authorization = NOT_AUTHORIZED
planar = NOT_VIABLE
grouped = NOT_VIABLE
region_fused = UNKNOWN
cutlass_4m_single = UNKNOWN
```

Re-aggregation applied all Task 1-8 gate fixes to the artifacts using EXISTING measured rows (no GPU). No route was upgraded to PASS/VIABLE (that would have been a false upgrade and an error).

## Findings -> fix mapping (11 findings, all GREEN)

| # | Finding (spec §3) | Fix commit | Test name(s) | Artifact / schema field | Command / gate | Status |
|---|---|---|---|---|---|---|
| 3.1 | C2 treats MODEL_ONLY peak as measured (false PASS) | `f7c71b30` | `c2_test.py` peak-evidence tests (MODEL_ONLY->UNKNOWN; delete evidence_class->UNKNOWN; fake MEASURED missing field->UNKNOWN; complete measured->PASS) | `c2_judgment.json` `recomputed.region_peak_gain_bytes` = `null` for MODEL_ONLY; `peak_evidence_class` gate in `_recompute_conditions` | `pytest results/_phase0/c2_test.py` | ✅ FIXED — region_peak_gain_bytes now `null` (was `1073741824`); C2_REGION_KERNEL UNKNOWN |
| 3.2 | Numerical manifest binding fail-open on 6 presence-only files | `fdfeceb0` | `manifest_test.py` 6 mutation tests (each source file -> MISMATCH) | `numerical_validation.json.case_binding` — 9 SHA256 bindings (edge_map, region_prototype, contraction_shapes, cublaslt_planar/full_matrix/grouped_capability/grouped_rows, cutlass_4m, numerical_csv) | `pytest results/_phase0/manifest_test.py` | ✅ FIXED — 9 hashes bound; mutation->MISMATCH->NUMERICAL UNKNOWN |
| 3.3 | REQUIRED_CRITERIA uses old C2 alias; completion false-GO | `713f4758` | `gonogo_test.py` + `verdict_schema_test.py` C2-layer-UNKNOWN->INCONCLUSIVE (4 C2 layers) | `verdict_schema.REQUIRED_CRITERIA = CRITERIA_NAMES` (12, no "C2" alias); `validate_criteria` + `rollup_c2_canonical` | `pytest results/_phase0/gonogo_test.py verdict_schema_test.py` | ✅ FIXED — C2 alias removed from gates; all 4 C2 layers in REQUIRED_CRITERIA |
| 3.4 | Cancellation input doesn't actually cancel | `7acc7d49` (+`7a08d65f` record fields) | `numerical_test.py` cancellation-ratio test (ratio<0.1, non-zero reference) | `numerical_validation.csv` cancellation fields (`input_construction_version`, `cancellation_epsilon`, `reference_norm`, `baseline_norm`, `cancellation_ratio`) | `pytest results/_phase0/numerical_test.py` | ✅ FIXED — paired A columns + eps*residual; ratio ~7e-4 |
| 3.5 | Capability readers return detail tokens -> real success can't be PASS (false negative) | `927888ac` (+`48fcb753` strict) | `gonogo_test.py` grouped SUPPORTED->PASS; region full-anchor->PASS | `gonogo._c3_grouped_status` (api_ok->PASS); `_region_proto_status` (reuse C2 peak gate + real-PTE check) | `pytest results/_phase0/gonogo_test.py` | ✅ FIXED — readers return canonical PASS for complete evidence |
| 3.6 | CUTLASS trusts self-reported capability (false PASS) | `927888ac` (+`48fcb753` strict path/runs/gate) | `gonogo_test.py` CUTLASS capability=PASS+runs=false->UNKNOWN/FAIL; wrong-path->UNKNOWN | `gonogo._cutlass_native_sm120/_sm80_fallback_criterion` recompute from kernel_path+runs+gate; `section.capability` diagnostic-only | `pytest results/_phase0/gonogo_test.py` | ✅ FIXED — recompute from evidence; no cross-promotion |
| 3.7 | Manifest presence map missing CUTLASS_SM80_FALLBACK_CAPABILITY | `fdfeceb0` | `manifest_test.py` delete cutlass artifact -> fallback NOT_RUN | `manifest.REQUIRED_ARTIFACTS` maps both CUTLASS criteria to `cutlass_sm120_4m.json` | `pytest results/_phase0/manifest_test.py` | ✅ FIXED — both criteria mapped; missing artifact downgrades both+numerical |
| 3.8 | C3 full-matrix algo/workspace constraints incomplete | `7b790990` | `gonogo_test.py` first_algo_id=-1->UNKNOWN; workspace>cap->UNKNOWN; no-algo workspace>0->UNKNOWN | `gonogo._c3_planar_full_matrix_status` checks (ok: first_algo_id>=0 + workspace<=cap; no-algo: workspace=0) | `pytest results/_phase0/gonogo_test.py` | ✅ FIXED — 3 checks added; current 128-cell still PASS |
| 3.9 | Sanitizer source hardcodes private names (AGENTS.md violation) | `e7074a67` | `sanitize_test.py` source-scan (no hardcoded names; fictional names; probe-source scan genuine) | `sanitize.py` `_dynamic_private_names()` from CONDA_PREFIX/CUDA_HOME/CUTLASS_ROOT/home/repo; `cutlass_probe.discover_paths()` fail-fast | `pytest results/_phase0/sanitize_test.py` + tracked-source grep | ✅ FIXED — no hardcoded private names; dynamic extraction; idempotent |
| 3.10 | blocking_artifacts wrong semantics (only C2, misses REGION_PROTOTYPE/NUMERICAL, lists grouped) | `cefbc056` | `gonogo_test.py` + `verdict_schema_test.py` blocking = C2+REGION_PROTOTYPE+NUMERICAL, not grouped | `verdict_schema._build_blocking_artifacts` two-rule; `CRITERION_BLOCKING_ARTIFACTS` map | `pytest results/_phase0/gonogo_test.py verdict_schema_test.py` | ✅ FIXED — `gonogo.json.blocking_artifacts` = [C2, REGION_PROTOTYPE, NUMERICAL] |
| 3.11 | Numerical shapes hardcoded, not bound to contraction artifact | `7acc7d49` | `numerical_test.py` load_current_shapes + set equality + drift->UNKNOWN | `numerical.load_current_shapes()` from `contraction_shapes.csv` (stdlib); `shapes_in_sync()` assert | `pytest results/_phase0/numerical_test.py` | ✅ FIXED — shapes derived from CSV; drift->INCONCLUSIVE |

## Re-aggregation result (this commit)

Regenerated via producers in dependency order (no GPU):
1. sanitize (dynamic, Task 7) applied to source artifacts
2. numerical regen from existing measured rows (Task 3 cancellation fields + Task 5 9-hash case_binding) — `numerical_validation.json` + `.csv`
3. C2 judgment regen with Task 2 MODEL_ONLY peak gate — `c2_judgment.json` (`region_peak_gain_bytes`: `1073741824` -> `null`), `c2_checkpoint_manifest.json`
4. gonogo regen (Task 1 schema v3 + Task 4 readers + Task 6 C3 checks + Task 8 blocking) — `gonogo.json` + `gonogo.md`
5. manifest regen (Task 5 full binding + validate_criteria + subset assert + Task 8 blocking) — `manifest.json`

## Gates (all PASS)

- `pytest -q results/_phase0/ -m "not gpu"`: **332 passed, 6 skipped, 3 deselected, 0 failed**
- `black --check --target-version py310 results/_phase0/`: 45 files clean
- `git diff --check`: clean
- Verdicts unchanged: phase0=INCONCLUSIVE, phase1=NOT_AUTHORIZED, C2_CANONICAL=UNKNOWN, CUTLASS_SM80_FALLBACK_CAPABILITY=PASS, region_peak_gain_bytes=null, blocking=[C2,REGION_PROTOTYPE,NUMERICAL]
- Numerical 9-hash case_binding present; overall_numerical_status=INCONCLUSIVE
- Tracked-artifact privacy scan: zero hits (untracked `??` scratch XLA dumps/HLOs out of scope, per dirty-tree protection)
- All 11 Task-0 RED tests: GREEN

## Remaining (NOT this closeout's scope)

- GPU Task 2b (full-anchor region kernel), Task 3b (numerical re-measure) — gated, pending user authorization (these resolve region_fused/cutlass UNKNOWN to real PASS/FAIL)
- Final Phase 0 clean rerun + final `rereview_closeout.md` (after GPU tasks)
