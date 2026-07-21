# Probe 2 (frontier) — smoke run note

**Matrix:** `smoke` (8 configs, all brickwork / jax).
**Date:** 2026-07-22.
**Machine:** 12GB RTX 5070 Ti Laptop (sm_120), WSL2 tcng env.

## Raw output

```
# Probe 2 frontier, matrix=smoke, configs=8
circuit    n   depth  output       backend  outcome  peak_alloc_B  peak/state  ms
---------  --  -----  -----------  -------  -------  ------------  ----------  -----
brickwork  18  3      state        jax      run      15702528      7.49        26.2
brickwork  18  3      expectation  jax      run      16825600      8.02        1.1
brickwork  18  10     state        jax      run      58958848      28.11       88.5
brickwork  18  10     expectation  jax      run      52131584      24.86       1.1
brickwork  22  3      state        jax      run      237756160     7.09        31.7
brickwork  22  3      expectation  jax      run      201839872     6.02        2.4
brickwork  22  10     state        jax      run      970542592     28.92       119.4
brickwork  22  10     expectation  jax      run      480753664     14.33       2.3

# frontier boundary (output, backend) -> max_run_n / min_fail_n:
  expectation  jax      max_run_n=22  min_fail_n=1000000000
  state        jax      max_run_n=22  min_fail_n=1000000000
=== phase0_frontier done ===
```

## Key observations

- **All 8 configs ran** (no OOM, no crash). Smoke matrix stays well inside the 12GB ceiling
  (largest peak ≈ 970 MB at n=22 depth=10 state).
- **state peak_ratio vs state-vector size: 7.5×–29×.** State-vector itself is *not* the
  bottleneck; the contraction's intermediate tensors dominate. The peak grows steeply with
  `depth` (7.5× at depth=3 → 29× at depth=10 for the same n) — so depth is the primary
  memory driver for the brickwork family, not n.
- **expectation ran at every config.** wall ≈ 1–2 ms vs state's 26–119 ms — XLA *does*
  fuse expectation into a much cheaper graph (no full state materialization on the hot
  path), but `peak/state` is still > 1 (6×–25×) because the contraction intermediates
  still allocate before the scalar terminal reduces them. Expectation is not free in
  peak memory under eager execution.
- **No `oom` boundary reached in smoke.** Frontier table shows `min_fail_n=1e9` for both
  outputs — full matrix (Task 7, controller-run) is required to find the n/depth where
  state or expectation tips into OOM.

## Implementation deviations from the task brief

Two genuine bugs in the brief were fixed (documented in the task-3 report):

1. `c.expectation(("z", [0]))` → `c.expectation((tc.gates.z(), [0]))`.
   tc-ng `Circuit.expectation` takes `(tc.gates.X(), [qubit])` tuples, not `(str, list)`.
2. `poller.start()` / `poller.stop()` → `with poller: ...`.
   `GpuSmiPoller` (bench_bf16_gpu.py) only exposes `__enter__` / `__exit__`;
   it has no public `start`/`stop` methods.
