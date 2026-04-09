# StabilizerTCircuit & ZX Subsystem — QEC Simulation Patterns

## 1. Vectorized multi-shot QEC simulation via `sample_detectors`

**The key pattern**: embed noise as probabilistic channels, annotate with DETECTOR/OBSERVABLE, then call `sample_detectors(shots=N)` once.  Do **not** build a new circuit per trial.

```python
# WRONG — one circuit per trial, Python loop over trials
for _ in range(trials):
    error_pattern = sample_errors(p)         # Python-side sampling
    stc = build_circuit()
    _apply_explicit_paulis(stc, error_pattern)  # errors become gates → same result for all shots
    result = stc.sample_measurements(shots=1)

# RIGHT — one circuit, one JAX call for all shots
stc = build_circuit_with_channels(p)        # noise = DEPOLARIZE1 channels inside the circuit
det, obs = stc.sample_detectors(shots=N, separate_observables=True)
accepted = np.all(det == 0, axis=1)         # post-select in NumPy
```

**Why**: `DEPOLARIZE1` channels are compiled into the ZX graph.  `ChannelSampler` draws independent error bits per shot at inference time.  Setting `shots=N` on a circuit with explicit Pauli *gates* returns N identical deterministic outcomes (same error pattern every shot) — not N independent trials.

---

## 2. Adding noise channels

```python
stc.t(q)
if p > 0:
    stc.depolarizing(q, p)   # DEPOLARIZE1 with px=py=pz=p/3
```

`depolarizing(q, p)` (single float) sets `px=py=pz=p/3`.  To set components individually: `depolarizing(q, px=..., py=..., pz=...)`.  Guard with `p > 0`; a zero-probability channel is harmless but wastes compiler work.

---

## 3. DETECTOR and OBSERVABLE_INCLUDE annotations

### Indexing convention — 0-based absolute, NOT Stim-style negative

`detector_instruction([i])` and `observable_instruction([i])` take **0-based absolute indices** into the measurement record (ordered by `measure_instruction` calls).  This is different from Stim's `rec(-1)` negative convention.

```python
meas_idx = 0
for support in z_checks:
    stc.cnot(q, anc); stc.measure_instruction(anc)
    stc.detector_instruction([meas_idx])   # refers to this measurement
    meas_idx += 1

stc.measure_instruction(logical_qubit)
stc.observable_instruction([meas_idx], observable_index=0)  # probe bit
```

### Semantics

- **DETECTOR([i])** fires (= 1) when `measurement[i] XOR noiseless_baseline` is 1.  For syndrome ancillas that should measure 0, the detector fires exactly when an error is detected.
- **OBSERVABLE_INCLUDE([i], observable_index=k)** marks measurement `i` as logical observable `k`.  The observable output is `measurement[i] XOR noiseless_baseline`.  For a probe that maps `|T†⟩ → |0⟩` in a noiseless circuit, observable = 1 means a logical error.

---

## 4. `sample_detectors` output structure

```python
det_samples, obs_samples = stc.sample_detectors(
    shots=N, separate_observables=True
)
# det_samples : (N, num_detectors)  — syndrome detector outcomes
# obs_samples : (N, num_observables) — logical observable outcomes
```

- `use_detector_reference_sample=True` / `use_observable_reference_sample=True`: XORs a noiseless reference sample into the output.  Only meaningful when the noiseless circuit is deterministic (always the case for CSS syndrome ancillas).
- Default (`use_*_reference_sample=False`): raw outcomes are returned.  For syndrome ancillas starting in `|0⟩`, raw = deviation from 0 = syndrome, so the default is usually correct.

---

## 5. Conditional post-selection replaces conditional circuit execution

The old pattern ran a two-stage circuit (syndrome → conditionally decode) to avoid wasted work on rejected trials.  With `sample_detectors`, always run the full circuit (encode → noise → syndrome → decode → probe) and filter afterwards:

```python
accepted = np.all(det_samples == 0, axis=1)   # (N,) bool mask
accept_rate = float(accepted.mean())
logical_error_rate = float(obs_samples[accepted, 0].mean())
```

Running the decode/probe on rejected shots is harmless and far cheaper than a Python conditional per trial.

---

## 6. Internal call chain (for debugging)

```
sample_detectors(shots=N)
  └─ _compile(sample_detectors=True)          # builds ZX graph with det/obs nodes
  └─ _sample_batches(shots, batch_size=1000)
       └─ ChannelSampler.sample(batch_size)   # draws (batch_size, num_error_bits)
       └─ sample_program(compiled_program, f_params, key)
            └─ sample_component(...)          # autoregressive per component, JAX jit
```

`_sample_batches` loops over `ceil(shots / batch_size)` JAX calls.  For GPU, set `batch_size` explicitly to saturate VRAM; on CPU the default (1000) is fine.

---

## 7. `sample_measurements` vs `sample_detectors`

| | `sample_measurements(shots=N)` | `sample_detectors(shots=N)` |
|---|---|---|
| Compiles with | `sample_detectors=False` | `sample_detectors=True` |
| Returns | `(N, num_measurements)` raw bits | `(N, num_detectors [+ observables])` |
| Requires | none | DETECTOR / OBSERVABLE_INCLUDE in QIR |
| Use for | direct readout | QEC syndrome + logical observable |

Both share the same `_sample_batches` / `sample_program` JAX backend — performance is identical.
