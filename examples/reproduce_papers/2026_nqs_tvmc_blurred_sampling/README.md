# NQS-tVMC with blurred sampling: Figure 5(a,b)

This example reproduces the single-spin and four-spin benchmarks in
[Wan, Wiersema, and Zhang, PRX **16**, 031059 (2026)](https://journals.aps.org/prx/abstract/10.1103/jrn5-gv19).
The authors' [Fig4.ipynb](https://github.com/therooler/nqs_blurred_sampling/blob/1db5fffab7bbcbc3a4ea035280fab1af72daf322/paper/Fig4.ipynb)
corresponds to the published Figure 5. No system-size reduction is made.

NQS-tVMC applies the time-dependent variational principle (TDVP) to a
neural-network wave function, estimating its metric and force by Monte
Carlo. Blurred sampling adds postprocessing and importance reweighting to
recover contributions missed near wave-function nodes. Both methods here
use the same ansatz, initial state, sample budget, integrator and regularizer.

## Run

In an environment containing TensorCircuit-NG, JAX, NumPy, SciPy and Matplotlib:

```bash
python examples/reproduce_papers/2026_nqs_tvmc_blurred_sampling/main.py
python examples/reproduce_papers/2026_nqs_tvmc_blurred_sampling/validate.py
```

Default sampling seeds are the authors' 100 and 1500 for panels (a) and
(b); `--seed` overrides both. Outputs are saved relative to the script. `physics.py`
contains the RBM, local sampler and estimators; `main.py` integrates and
plots them. All evolution kernels use `tc.backend`, including JIT, vmap
and scan. NumPy handles storage, reporting and plotting. `validate.py`
uses TC autodiff and independent `tc.Circuit`/Pauli constructions.
Dense operators and full-state sums are used only for reference evolution,
measurement and validation; they never supply the MC evolution estimator.
Repeated observed samples are losslessly grouped with their counts.

## Physical and numerical conventions

| | Figure 5(a) | Figure 5(b) |
|---|---|---|
| System | H=Y, initial (1,1) | Open 2x2 Heisenberg square, total Sz=0 |
| Ansatz | Two complex amplitudes | psi(x)=2 cosh(b + W.x), four complex weights, hidden bias, no visible biases |
| Quench | Fixed H | S=Pauli/2; vertical J: +1 to -1, horizontal J=+1 |
| Observable | Pauli Z; exact -sin(2t) | Single vertical bond S0.S1 |
| Integrator | Adaptive Dormand–Prince RK5(4) | Heun, physical dt=0.001 |
| Samples | 8192 independent chains | 16384 independent chains |
| Blur | q=0.5 | q=0.5, uniformly selected nonzero off-diagonal H connection |

The four-spin bonds are (0,1), (2,3), (0,2), (1,3). The RBM evolves in
the Marshall basis, with the fixed Z0 Z3 transformation applied for
physical-basis circuit validation. Each force evaluation follows four
local proposals per chain; initialization uses 128 proposals.

The following source conventions matter for interpreting the plot:

- Panel (b) uses the notebook's single vertical bond and explicit
  `t_physical/4` axis. The paper's written sum of the two vertical bonds
  is twice the single-bond result for the symmetric exact trajectory.
- Panel (a) follows published Table II's adaptive integration and sample
  count; the notebook instead uses fixed dt=0.001 and 16384 samples.
  Here rtol=1e-4, atol=1e-8, and dt is bounded by [1e-5, 0.01]. The Z=-1
  minimum is the computational ket(1), although the caption calls it ket(0).
- Schmitt regularization uses relative cutoffs 1e-14 and 1e-10 and nominal
  SNR cutoff 2. Both author drivers sum force rows already divided by N
  but use their standard deviation: their SNR is N times the usual
  mean-based SNR. We explicitly retain this convention (cutoff 2/N for
  mean-based rows). Changing it moves the ordinary tVMC breakdown.
- The included 792-byte `initial_state.npz` was independently regenerated
  from the notebook's seed-1500 SR recipe: complex normal initialization
  of scale 0.001, 1024 samples, SGD 0.01, SR shift 1e-4, at most 1000
  steps, stopping when estimated energy is within 0.001 of the exact
  ground energy. NetKet 3.21.0, Flax 0.12.3 and JAX 0.9.1 stopped after
  39 steps. Exact preparation energy error is 0.003746999 and infidelity
  0.001314468. These parameters and reference amplitudes are bundled;
  NetKet is not required to run this example. Both samplers and the exact
  reference start from this identical state. The notebook's exact-plot
  call instead defaults to seed 100; its seed-1500 checkpoint was absent.
- Local proposals, warmup and random streams differ from NetKet's sampler.
  The blur excludes zero matrix elements rather than depending on padded
  connection counts. Individual stochastic trajectories are not bitwise
  reproductions of the authors' run.

## Reweighting at nodes

For a blurred sample y, with source degree d(x), use

```text
R(y) = (1-q)|psi(y)|² + q sum_{x != y, H_yx != 0} |psi(x)|²/d(x)
a = psi/sqrt(R), d = partial_θ psi/sqrt(R), h = Hpsi/sqrt(R)
z = mean(|a|²), mu = mean(conj(a)d)/z, E = mean(conj(a)h)/z
c = d-a mu, e = h-a E
S_ij = mean(conj(c_i)c_j)/z, F_i = mean(conj(c_i)e)/z
Re(S) θ_dot = Im(F)
```

Direct amplitude derivatives preserve finite S/F contributions at psi=0,
without division by psi, amplitude floors or discarded samples. Equation
(11)'s covariance correction is N mean(w)²/(N-1), with w=|psi|²/R. It
multiplies both S and F and cancels in the unregularized TDVP solve;
`validate.py` checks the corrected estimator explicitly. Finite-sample
SNIS and regularized TDVP updates are not claimed to be unbiased.

## Results and validation

The default curves recover ordinary tVMC's failure and blurred tVMC's
agreement with exact evolution. In a matched seed-1500 comparison,
ordinary four-spin tVMC first exceeds absolute correlation error 0.05 at
figure time 0.72525; the original author driver gives 0.7275. Blurred
sampling's maximum correlation error is approximately 4.4e-6, versus
9.3e-6 for the original driver. Separate seeds 100, 200 and 300 gave
4.1e-6 to 4.8e-6. The single-spin blurred error is 1.8e-15. Our standard
four-spin trajectory remains near zero after the second maximum, whereas
the published failed trajectory later resumes oscillating; this
post-failure behavior depends on sampling and is not matched pointwise.
The small blurred residual includes finite sampling, integration
and ansatz projection error; the initial preparation error is measured
separately above. Failure onset and subsequent standard trajectories vary
with the sampling seed.

`outputs/results.json` records each actual run's errors, versions,
synchronized first-step and warm-step timings, trajectory time and peak
RSS. The runs used a Linux CPU server, four logical CPUs per job, Python
3.12.3, TC-NG 1.9.1, JAX/JAXlib 0.9.1, NumPy 2.4.3, SciPy 1.17.1 and
Matplotlib 3.10.8. Four GB of RAM is recommended; no GPU is needed.
The two four-spin trajectories took about 239 and 241 seconds after the
timed warmup steps; the main process peaked at 766 MiB RSS.

`validate.py` checks local Hpsi, TC circuit states/observables/evolution,
RBM derivatives against autodiff, and lossless sample aggregation. It
checks S/F against exact amplitude inner products at a strict single-spin
node and an RBM cosh node, then checks MC and Eq. (11) estimates using 32
independent replicas of 4096 samples, within five replica standard errors.
The ordinary estimator demonstrably misses a finite contribution at the
strict node. Results are saved to `outputs/validation.json`.
