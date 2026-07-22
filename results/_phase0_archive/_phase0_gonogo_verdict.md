# Phase 0 Go/No-Go Verdict — SUPERSEDED (2026-07-22)

**Status: SUPERSEDED.** This file previously held a "GO for large-n" verdict. The 2026-07-22 review
(`docs/superpowers/2026-07-22-phase0-review-spec.md`) refuted that verdict's strong claims. The
correct current status is **INCONCLUSIVE / CONDITIONAL-GO-FOR-DEFERRED-PROBE-ONLY**.

## Why superseded

The earlier "GO for n≥24" rested on evidence the review showed was confounded:

- **C1 confounded.** The probed circuits have NO runtime parameters (`rz(0.7)`/cnot/H are all
  compile-time constants; `jax.jit(lambda: ...)` takes no args) → XLA can constant-fold. Combined with
  `peak_bytes_in_use` being a compile+runtime cumulative high-water mark read after compile+first-exec,
  `peak(default)≈peak(no-fusion)` only proves "two processes' lifetime peaks are close" — NOT that the
  ~1.1/4.3 GB is unavoidable runtime GEMM materialization that BF16 could halve. "Halve 1.1/4.3 GB" and
  "push boundary to n≈27–28" are unproven hypotheses, not conclusions.
- **C2 not proven.** 519 `dot_general` + cotengra binary tree is a heuristic, not proof of
  single-consumer / tile-mappable / acceptable recompute. C2 = UNKNOWN.
- **C3 proxy ≠ capability.** The real-BF16 square-GEMM ratio (~2.7×) is a Tensor-Core ceiling proxy,
  not a planar-complex cuBLASLt capability test.
- **Harness bug.** Fusion worker crash (`_phase0_fusion_window_probe.py:95-105`) emits crash JSON +
  exits 0 → orchestrator mislabels it `run`. (Same bug class as the Task-3 fix; final review missed it
  for Probe 3 because its smoke never triggered the crash path.)

## What still stands (weak, valid)

- A **phase-transition signal** at n≈22→24 in JIT peak (123 KB → 1.1 GB → 4.3 GB) — worth
  investigating, cause unattributed.
- SM120 real-BF16 GEMM ceiling ≈ 2.7× FP32 (TF32 off).
- Frameworks have no native complex-BF16 dtype (production needs pair rep / custom call / native ext).

## Current authoritative status

Per the review (§14):

```
Phase 0 cheap pre-screen: COMPLETE
Phase 0 canonical capability validation: INCOMPLETE
Large-n BF16 architecture: NOT YET GO
Deferred planar-cuBLASLt probe: AUTHORIZED
Phase 1: BLOCKED UNTIL RE-REVIEW
```

Next step = review §5–9 remediation (dynamic params, compile/runtime memory split, optimized
HLO + buffer assignment, real shape export, tile-mappability classification, planar cuBLASLt probe,
CUTLASS SM120 probe, four-state aggregator). Do not treat this file's earlier GO as authoritative.
