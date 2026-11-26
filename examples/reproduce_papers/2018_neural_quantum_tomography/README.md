# W-state tomography with restricted Boltzmann machines

This example reproduces the 20-qubit W-state experiments in Figure 1 of
[Torlai et al.](https://arxiv.org/abs/1703.05334). The positive-state experiment
measures how shot count affects the learned populations and overlap. The
random-phase experiment compares a Z-only fit with a fit that also sees XX and
XY outcomes on adjacent pairs.

## Run and verify

Use TensorCircuit-NG from this checkout with JAX, Optax, NumPy, SciPy, and
Matplotlib. From the repository root, after installing the project in your
environment, run

```sh
example=examples/reproduce_papers/2018_neural_quantum_tomography
export PYTHONPATH=.
export JAX_PLATFORMS=cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python "$example/verify_results.py" --preflight
python "$example/main.py" --output rerun
python "$example/verify_results.py" --output rerun
```

On PowerShell, set the same environment variables with `$env:NAME = 'value'`
and set `$example = 'examples/reproduce_papers/2018_neural_quantum_tomography'`.
Omit `--output` to use this example's `outputs/` directory. Run
`python "$example/verify_results.py"` to check the supplied archive. The preflight
checks all 20 encoded singleton outcomes through the sampler and NPZ round trip,
20,000 additional Z draws, small-system XX/XY probabilities and signs, and a
phase gradient and natural-gradient update against dense NumPy references.

## Measurement and training

- The positive experiment uses nested prefixes of 50, 1,000, and 20,000 shots
  per seed. For the phase-augmented state, the 39 settings are Z followed by XX
  and XY on each of the 19 neighboring pairs. Each setting has 6,400 shots
  (249,600 total per seed). Target angles are sampled uniformly on
  $[-\pi,\pi)$. The targets generate the measurements and enter the final
  evaluation. The fit uses the saved shot streams.
- A bond-two MPS generates each measurement with TensorCircuit's single-shot
  sampler. JIT and vmap share the dynamic MPS tensors across fixed-size shot
  blocks. TensorCircuit's H and S-dagger gates implement the basis changes.
  Saved outcomes are uint32, with site one at the most significant used bit.
- Each RBM has 20 visible and 20 hidden units and models the full bit space.
  The amplitude model starts with zero biases and normal weights of standard
  deviation 0.1. It uses 200,000 SGD updates at rate 0.1, with 128 observations
  drawn with replacement from the assigned Z stream per update. Each chain
  starts at an observed bit string and takes one batched block Gibbs step
  to produce the model sample (CD-1).
- The phase RBM starts with normal hidden biases and weights of standard
  deviation 0.1. Its visible biases start from adjacent phase differences
  inferred from the measured XX/XY parities conditioned on zeros at the other
  sites. This W-specific initialization reuses the phase training shots.
  With the amplitude frozen, each observed outcome has four local
  computational-basis configurations. Duplicate outcomes are grouped per
  setting and weighted by their shot counts in the gradient and uncentered
  empirical Fisher. Each phase step uses all 38 rotated settings. The phase fit
  takes 48 JAX natural-gradient steps with
  diagonal damping 0.01 and a Fisher trust radius of 0.20. The fixed schedule
  reports the final iterate for every seed.
- The Z-only control uses the same learned amplitudes and sets every phase to
  zero. The fits and initialization use only the stated training budgets.
  Blocked $2^{20}$-state normalization runs after training, for evaluation.
  The overlap is $O=\lvert\langle\Psi_W\mid\psi\rangle\rvert$, in the paper's
  convention.
  The population plot shows exact unconditional probabilities of the fitted
  RBM at the W components, averaged over seeds. Their sum is the probability
  of a single excitation and can be below one.

## CPU results

| Seed | Positive, 50 shots | Positive, 1,000 shots | Positive, 20,000 shots | Phase, Z only | Phase, all 39 bases |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9238 | 0.9864 | 0.9866 | 0.1290 | 0.9475 |
| 1 | 0.9168 | 0.9903 | 0.9875 | 0.1382 | 0.9719 |
| 2 | 0.8521 | 0.9899 | 0.9881 | 0.2595 | 0.9625 |

All entries are absolute overlaps. The 20,000-shot models assign 0.9802–0.9854
probability to the single-excitation sector. The all-bases phase fits improve on
the paired Z-only controls and fall short of the paper's reported phase overlap
of 0.997.

A first run of 100,000 amplitude updates gave 20,000-shot overlaps of
0.9437, 0.9466, and 0.9428 and all-bases phase overlaps of 0.8136, 0.7856, and
0.6163. Doubling the updates lowered the measured training Z negative
log-likelihood for every fit on exactly the same observations. At 20,000 shots,
it moved from 3.1141, 3.1110, and 3.1182 to 3.0270, 3.0316, and 3.0259 nats per
observation. The 20,000-shot overlap slightly exceeds the 1,000-shot overlap for
seed 0 and remains slightly below it for seeds 1 and 2.

The phase initialization in the final schedule gives overlaps of 0.9829,
0.9869, and 0.9832. The phase fit lowers the all-bases training NLL from 4.2845,
4.2863, and 4.2869 to 4.2832, 4.2857, and 4.2858 nats per observation, respectively.
Float64 replay shows lower normalized NLL at every phase step in both schedules.
The final overlap is lower than the initialization overlap for all three seeds.
The figures report the fixed final iterates.

The full run took **365.6 s** and reached **708.7 MiB** peak resident memory
on a Windows CPU (Intel Core i7-13700HX). Wall time runs from launching a fresh
Python process through process exit, including imports, JAX compilation,
sampling, training, synchronization, and saving the figures and data. Peak
memory is the experiment process's Windows peak working set. Preflight,
independent verification, and gallery generation run separately. The measured
environment used Python 3.11.16, JAX 0.7.0, Optax 0.2.8, NumPy 2.2.6, SciPy
1.17.1, and Matplotlib 3.11.0.

## Outputs

- [Overview](outputs/result.png) shows overlap versus budget, site
  populations, and the Z-only versus all-bases overlap.
- [Relative phases](outputs/phases.png) shows target and learned phases
  relative to site one for every seed.
- [Metrics](outputs/metrics.json) contains each seed's overlaps,
  single-excitation mass, populations, and relative-phase errors.
- [Archive](outputs/raw_data.npz) stores all measurement outcomes and final
  RBM parameters. Its leading axis is ordered by seed (0, 1, 2).
  `amplitude_parameters` stores the three positive budgets followed by the
  shared random-phase/Z-only amplitude fit. The parameter order is visible
  biases, hidden biases, then the row-major hidden-by-visible weights.
