# Noise Modeling and Mitigation

1.  **Readout Error Handling**:
    - The `circuit_with_noise(c, noise_conf)` function internally applies general quantum noise (Kraus channels specified in `nc`) to gates, but it does **not** automatically apply readout error configuration (`readout_error`) encoded in `NoiseConf`.
    - **Protocol**: You must **explicitly** pass the readout error when calling `circuit.sample()`. Example:
      ```python
      c_noisy.sample(..., readout_error=noise_conf.readout_error)
      ```
    - Failing to do so will result in noiseless measurements even if `readout_error` is present in `NoiseConf`.

2.  **Readout Error Format**:
    - When adding readout noise via `NoiseConf.add_noise("readout", errors)`, the expected format for each qubit's error is `[p(0|0), p(1|1)]` (a list of two probabilities), **not** the full $2 \times 2$ confusion matrix.
    - **Pitfall**: Passing a full matrix like `[[0.9, 0.1], [0.1, 0.9]]` can cause unexpected `TypeError` during internal matrix construction (e.g., `1 - list` error).
    - **Protocol**: Specify readout error as `[0.9, 0.9]` which implies $p(0|0)=0.9$ and $p(1|1)=0.9$.

3.  **Circuit Expectation with Noise**:
    - The `Circuit.expectation` method supports `noise_conf` as a keyword argument (e.g., `c.expectation(..., noise_conf=conf, nmc=1000)`). This is often cleaner than calling `tc.noisemodel.expectation_noisfy(c, ...)` directly.

4.  **Multi-Qubit Thermal Noise**:
    - The `thermalrelaxationchannel` returns single-qubit Kraus operators. To apply thermal noise to multi-qubit gates (like CNOT), you generally cannot simply pass the single-qubit channel to `add_noise("cnot", ...)` because of dimension mismatch.
