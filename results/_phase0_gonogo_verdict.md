# Phase 0 Go/No-Go Verdict (informed, 2026-07-22)

**Verdict: GO (for large-n workloads, n≥24 on 12GB) — NO-GO for small/medium-n (≤22).**
Not a blanket GO. The BF16 memory+speed window is **size-dependent**: it opens where XLA can no longer fuse the contraction away.

## Criteria (with the Phase-0 evidence)

### C1 — window exists: YES at n≥24, NO at n≤22
JIT (`jax.jit(expectation)`) peak_bytes_in_use, brickwork, fusion-disable A/B (ratio = nofusion/default; calibration proved the `--xla_disable_hlo_passes=fusion` flag engages):

| n | depth | peak (default) | peak (nofusion) | ratio | read |
|---|---|---|---|---|---|
| 22 | 10 | 123 KB | 126 KB | 1.02 | XLA fuses → tiny → no BF16 benefit |
| 22 | 16 | 189 KB | 192 KB | 1.01 | XLA fuses → tiny → no BF16 benefit |
| 24 | 10 | **1.107 GB** | 1.107 GB | 1.00 | phase transition — materialization unavoidable |
| 24 | 16 | 1.108 GB | 1.108 GB | 1.00 | unavoidable |
| 26 | 10 | **4.329 GB** | 4.329 GB | 1.00 | unavoidable; **and JIT runs (eager OOMs here)** |
| 26 | 16 | 4.329 GB | 4.329 GB | 1.00 | unavoidable |

- **Phase transition at n≈22→24**: JIT peak jumps ~9000× (123 KB → 1.1 GB → 4.3 GB). Below it XLA eliminates intermediates; above it XLA cannot — the intermediates are forced to materialize.
- **ratio 1.00 at n≥24**: the flag is proven to work (calibration changed peak 21504→24064 at n=10/d=3/norm), so bit-identical peak at n≥24 means fusion is *irrelevant to peak here* — the GBs are materialized GEMM library outputs fusion passes don't touch. **This is exactly the BF16 window the prior "dead-end" memory named as "the only theoretical window (F6 region)" — but it is NOT just the crash zone: JIT runs n=24–26 at 1–4 GB.**
- Caveat: peak_bytes_in_use is cumulative (compile+run); the ratio is fair (both arms cumulative), and a 9000× jump is not compile-scratch noise. The true run-peak may be somewhat lower, but the materialization is unambiguous.
- Frontier (eager, Probe 2) boundary: n=26 OOMs eager; n=24 runs (state peak ~4.8 GB at d=16). JIT extends the boundary (n=26 runs at 4.3 GB).

**C1 bottom line**: BF16 could halve ~1.1 GB (n=24) / ~4.3 GB (n=26) → could push the 12 GB boundary from n≈26 to n≈27–28. Real, user-relevant, JIT-reachable.

### C2 — coverable: YES-heuristic (not proven)
- n=24 expectation stablehlo = **519 `dot_general` contractions**. The materialized intermediates are GEMM-like (tile-mappable — exactly what cuBLASLt / Tensor Core handle).
- tc-ng contracts via a cotengra **binary tree** → each intermediate is single-consumer by construction (feeds one parent) → region fusion (spec §8.1) can in principle tile-fuse them.
- **Heuristic YES**. Not proven: tile-mappability in practice (register/shared-mem pressure, irregular shapes) is a Phase-1 implementation risk, not answerable in Phase 0 without a region-fusion prototype. nsys was unavailable (sudo-gated), so this is structural reasoning, not dynamic confirmation.

### C3 — Tensor Core ceiling real: YES (~2.7×)
Probe 1 proxy, SM120, TF32 off: bf16/fp32 GEMM ratio 2.65 / 3.62 / 3.27 at M=2k/4k/8k (all ≥ 1.3). Run-to-run TFLOPS variance ~40% but the ratio is stable (~8% band). K3's 4–5.7× was on square 4096²; these contraction-relevant shapes show ~2.7× — still comfortably above threshold.

## What this changes vs the prior memory (`tc-ng-bf16-leverage`)

The memory's "dead end, 勿再走" holds **for n≤22** (XLA fuses; BF16 has nothing to halve — confirmed). But it does **not** hold blanket: at n≥24 the M3 escape hatch ("forced materialization") activates and is JIT-reachable (not just the F6 crash zone). The memory *predicted* this was the only window but underestimated its reachability. **Refining the memory accordingly.**

## Recommended next step (per staged plan)

The deferred **Probe 1 libcublasLt binding** is now warranted for these large-n GEMM contractions: test planar-complex BF16 (`CUDA_C_16BF` + `PLANE_OFFSET` + FP32 accumulate) on SM120 against the 519-`dot_general` workload's real shapes. That is the decisive test of whether the C3 ceiling composes into end-to-end gain on the C1 window.

## Non-goals reaffirmed
- BF16 still gives nothing for small/medium-n variational QC (≤22) — XLA fuses. Do not pursue BF16 there.
- This verdict is about the BF16 *native executor* direction (the optimal-contraction spec), not the existing 4-M pair path (which the memory already closed).
