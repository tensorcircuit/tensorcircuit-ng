# Phase 0 BF16 Region-Fused v5 GPU Research Report

Date: 2026-07-27
Scope: `region_fused/direct`, CUDA `fused_pte_kernel`, c64 full anchor
Policy: `REGION_FUSED_FULL_ANCHOR_ACCURACY_v5` (`dual-gate-v5`)
Policy SHA-256: `3ecfa370409e2397319276b8aa1b64bf19a816b2e8e0fb478b51569bf383ced1`

## Result

The full 18-cell GPU matrix completed without OOM, timeout, infrastructure failure, or retry. Coverage was complete, all outputs were finite, and all cells carried the frozen policy identity.

The global relative-L2 gate passed in all 18 cells, but the local-scaled-max gate failed in all 18 cells. The accuracy verdict for `region_fused/direct` is therefore **FAIL**, and this route is **NOT_VIABLE** under the frozen v5 policy.

The memory result remains positive: the fused path avoided the 512 MiB `P` and 512 MiB `T` buffers. Measured allocator peaks were 1,778,384,896 bytes for the materialized path and 704,643,072 bytes for the fused path, a reduction of 1,073,741,824 bytes. Region fusion is a real memory lever, but this direct producer-recompute implementation does not meet the accepted accuracy contract.

## Accuracy results

Frozen thresholds:

- `global_rel_l2 < 1e-4`
- `local_scaled_max < 1e-3`
- `alpha = 1e-3`

| Input profile | Cells | Maximum global relative-L2 | Local-scaled-max range | Reference magnitude at local argmax | Verdict |
|---|---:|---:|---:|---:|---|
| baseline | 6 | 8.5113e-7 | 1.6523e-3 to 2.0868e-3 | 0.572 to 0.840 | FAIL |
| mixed_scale | 6 | 7.4816e-7 | 1.4297e-3 to 2.3253e-3 | 1.25e5 to 3.39e5 | FAIL |
| cancellation | 6 | 5.8129e-5 | 1.1744e-1 to 1.4666e-1 | 2.12e-4 to 5.45e-4 | FAIL |

Worst cell:

- Cell key: `cancellation:cancellation_v2:seed=542109305`
- `global_rel_l2 = 5.8128938069216e-5` - global gate PASS
- `local_scaled_max = 0.14666077359851473` - local gate FAIL
- `local_scaled_argmax_reference_abs = 0.0004882161863755533`
- `nan_inf = false`

## Interpretation

The cancellation profile confirms that low-amplitude outputs can produce very large localized normalized errors. However, that is not the complete explanation: baseline and mixed-scale also fail at high-signal output elements. The earlier hypothesis that elementwise failures were caused only by near-zero reference values is therefore falsified.

Global relative-L2 alone is insufficient for this kernel. It averages error across 67,108,864 output elements and passes even when a reproducible localized error exceeds the frozen local threshold. The dual-gate policy detected this distinction as intended.

The thresholds must not be relaxed after observing these results. Any alternative local policy or numerically different fused kernel is a new research subject and requires a new precommitted evaluation.

## Provenance

- Accepted policy commit: `30a0048b09f6f7f58d9fa72ea8eacbd161ca382a`
- Original candidate freeze commit: `fc8b2c1d522861beaa849d808b0fc8a9c6dab873`
- Actual measurement code commit: `09e69b9fe9542879a13f74fcca3f6e51a53e8253`
- Required seeds: `0, 1, 2, 1598166685, 542109305, 1463850203`
- GPU environment: RTX 5070 Ti Laptop, `sm_120`, 12 GB

`run_context.py` was executed before a seven-line `version_token` mapping was added to `region_proto.py`, so it originally recorded `fc8b2c1d`. The mapping only supplies `baseline_v1`, `mixed_scale_v1`, and `cancellation_v2` strings for summary cell keys. It does not change inputs, kernel execution, the materialized oracle, or metric calculations. `run_context.json` records this correction explicitly. This is acceptable for the research conclusion but remains a provenance deviation for a strict audit closeout.

Evidence SHA-256 values:

- `region_prototype.json`: `114b7b5daba88e15ab704c42f4925c4e97e4731c1246316eba6250a7760de19d`
- `region_prototype_accuracy.csv`: `31ae2a52ac8f17bc40f91cc1c7b3ac1c719d0e557d694781d8a86e8c9db4b256`
- `region_fused_v5_research_run.log`: `efeb4c3f470ead4640de9d11e81f8aa45fab23bb64c9953929c7c496c5c07076`

## Phase status

- `region_fused/direct` accuracy: **FAIL**
- `region_fused/direct`: **NOT_VIABLE**
- Region-fusion memory leverage: **CONFIRMED**
- Phase 0: **INCONCLUSIVE**
- Phase 1: **NOT_AUTHORIZED**
