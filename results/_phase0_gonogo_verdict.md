**PROVISIONAL — pending full frontier run + coverability analysis; NOT a final go/no-go.**

# Phase 0 Go/No-Go Verdict

**Verdict: NO-GO** (provisional; see header — criterion 2 is *inconclusive*, not definitively false)

Reason: window exists but NOT single-consumer/tile-mappable — region fusion (spec §8.1) cannot cover it; open problem

Criteria:
- 1_window_exists: True
- 2_coverable: False
- 3_ceiling_real: True

## Provisional reasoning (2026-07-22)

Inputs: `--has y --coverable n --ratio 2.7` (non-interactive). Caveats per criterion:

- **Criterion 1 (window) — provisional YES.** Probe 2 smoke shows `peak/state` ratios of
  7.5×–29× (n=18/22, depth=3/10, state output). Probe 3 (n=18, d=10) classified
  state→materialized-unavoidable. This signal is *smoke-only* (the full frontier was not
  run) and partly reflects the trivially-unavoidable output floor for state output, so the
  "yes" is provisional pending the full frontier measurement.
- **Criterion 2 (coverable) — INCONCLUSIVE (recorded as `n`).** nsys is unavailable
  (sudo-gated; see `_phase0_setup_note.md`) and Probe 3's fusion column was dropped
  (`_phase0_fusion_nsys_note.md`), so single-consumer / tile-mappable judgment cannot be
  automated yet. `--coverable n` here means *inconclusive*, NOT a definitive false.
- **Criterion 3 (ceiling) — YES.** Probe 1 proxy on SM120 (4-real bf16 GEMM, TF32 off)
  gives bf16_TFLOPS / fp32_TFLOPS ≈ 2.7× (e.g. 35.6 vs 12.7 TFLOPS at M=2048;
  `_phase0_cublaslt_gap.txt`), which clears the 1.3× threshold.

**Next step:** re-run after the controller's full frontier measurement produces a
definitive criterion-1 signal AND a coverability analysis (nsys or substitute) resolves
criterion 2. Only then is a final go/no-go warranted.
