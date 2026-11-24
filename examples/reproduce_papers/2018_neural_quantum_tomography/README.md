# W-state tomography with restricted Boltzmann machines

This example illustrates Figure 1 of [Torlai et al.](https://arxiv.org/abs/1703.05334).
The positive W-state fits show how the number of measurements affects overlap
and the populations of the single-excitation configurations. The random-phase
fits compare Z-only reconstruction with Z plus measurements on adjacent XX and
XY pairs. Z probabilities alone are insensitive to the relative phases.

Install TensorCircuit-NG with the JAX extra and ensure SciPy and Matplotlib are
available. The supplied results were generated on CPU. The exact software
versions are recorded in [metrics.json](outputs/metrics.json).

From this folder:

```sh
python main.py --smoke
python verify_results.py
python main.py --output rerun
python verify_results.py rerun
```

The four-qubit smoke run exercises state preparation, measurement, and fitting.
The full run covers 45 positive-state fits and nine random-phase fits at
8, 10, and 12 qubits using seeds 0, 1, and 2, with nine paired Z-only controls.
By default, the main script regenerates the files in `outputs/`. An explicit
`--output` writes to a separate directory. Relative output paths are resolved
from the example folder.

- [Overview plot](outputs/result.png) shows overlap $O = |\langle\Psi|\psi\rangle|$,
  single-excitation probabilities, and fidelity $F = O^2$. Error bars are
  sample standard deviations over three seeds. Shot budgets in the positive
  sweep are nested prefixes of one stream per size and seed.
- [Phase plot](outputs/phases.png) compares phases relative to the first
  single-excitation component for seed 0 at each size. Each multi-basis fit
  uses 6,400 shots per basis across $2N-1$ bases. The control reuses the same
  Z measurements and fits only the amplitudes.
- The NumPy verification script computes overlap, fidelity, normalization,
  leakage, and relative phases from the saved parameters. It also checks the
  nine control fidelities and confirms coverage of the full experiment grid.

The example uses exact enumeration of all bit strings, L-BFGS-B, and a phase
warm start fitted to empirical probabilities. These adaptations make the
fixed small systems tractable. The target state is used to generate simulated
samples and to evaluate the fit. Both RBMs are unrestricted, and the loss uses
measurement frequencies. The benchmark covers the small-system mechanisms
of Figure 1. The paper's Monte Carlo training and larger W states are outside
this example's scope.
