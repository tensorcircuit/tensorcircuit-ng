# Noise and QEC

Use this file for noise APIs, detector sampling, and stabilizer/QEC workflows.

## Noise API rules

- `circuit_with_noise(...)` applies gate noise but not readout error automatically; pass `readout_error=noise_conf.readout_error` when sampling if readout confusion is part of the experiment.
- `NoiseConf.add_noise("readout", ...)` expects per-qubit probabilities of the form `[p(0|0), p(1|1)]`, not a full confusion matrix.
- Prefer `Circuit.expectation(..., noise_conf=..., nmc=...)` when the noisy expectation API already covers the use case.
- `thermalrelaxationchannel` is a single-qubit channel. Do not attach it directly to multi-qubit gates without building a compatible multi-qubit noise model.

## Stabilizer and detector sampling

- For repeated noisy QEC trials, embed stochastic channels in the circuit and call `sample_detectors(shots=N)` once.
- Do not emulate repeated trials by rebuilding circuits with explicit sampled Pauli gates; that produces one deterministic error pattern copied across shots.
- `detector_instruction(...)` and `observable_instruction(...)` index the absolute measurement record with zero-based positions, not Stim-style negative `rec(...)` offsets.
- Use `sample_measurements` for raw readout and `sample_detectors` when the QIR includes detector or observable annotations.

## Post-selection workflow

- Run the full syndrome and decode pipeline for all shots, then post-select with a detector mask afterward.
- For CSS-style deterministic reference circuits, raw detector samples are usually already the signal of interest and do not need extra reference-sample correction.
