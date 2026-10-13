# Entanglement growth from entangled states

L=12 reproduction of Figures 1(d), 1(e), and 3 of [Zhang, Li and Zhang, arXiv:2510.08344](https://arxiv.org/abs/2510.08344). A thermal XXZ quench prepares entangled states; subsequent Hamiltonian and random-circuit dynamics distinguish entanglement generation from redistribution. Bipartition-averaged entropy (BAEE) measures the reservoir beyond half-chain entropy (HCEE).

## Run

```bash
python -m pip install -e '.[jax]' matplotlib cotengra
cd examples/reproduce_papers/2025_entanglement_growth
python verify.py
python main.py
```

`main.py` generates both committed outputs, `outputs/summary.npz` and `outputs/result.png`. Parameters at the top of the script use L=12, 72 independent disorder samples, seed 20260917, five circuit realizations per sample, depth 2000, and all 462 distinct equal bipartitions. Figure 3 is evaluated on its displayed interval, τ=0,…,50, in steps of 0.5. The summary stores means, one-SEM errors, time grids and these run parameters. `python main.py --plot-only` redraws the summary. Output paths are relative to the script, regardless of the working directory. The default uses one worker; `python main.py --workers 4` distributes independent disorder samples across four processes, with roughly 2 GB RAM budgeted per worker. When using several workers, limit BLAS threads in the execution environment to avoid nested parallelism.

The JAX engine uses TensorCircuit's `K.jit`, `K.vmap` and `K.scan` for charge-block entropy and circuit trajectories. CPU spectral preprocessing retains `eigh`, complex Schur decomposition and extended-precision phase reduction through the TC NumPy backend. No GPU is required.

## Numerical conventions

- Open boundaries, total Sz=0, S=σ/2, Jz=0.5; site 0 is the most significant bit. Hamiltonians use `tc.quantum.PauliStringSum2COO`. Independent uniform fields have W=0.5 for preparation and thermal quench, and W=5 for MBL, AL and Floquet. Fig. 3 follows the same preparation evolution that generates the Fig. 1 initial states. AL and free fermions use Jz=0; free fermions have zero fields.
- Entropies use log base 2 and charge-block `K.svd`. HCEE keeps the first L/2 sites; BAEE counts each equal bipartition/complement pair once. Circuit realizations are averaged within each disorder sample. Growth and reservoir SEMs use paired differences.
- Each circuit step applies one gate to a uniformly sampled adjacent bond. The gate is exp[−i(α(SxSx+SySy)+βSzSz)], constructed with TC rotations rxx(α/2), ryy(α/2), rzz(β/2):

| Protocol | α | β |
|---|---|---|
| thermal | π/2 | π |
| A | 0 | π |
| B | π/2 | 0 |
| C | π | 0 |
| D | π | π/2 |
| SWAP | π | π |

- Thermal/MBL/AL are measured at t=10¹², free fermions are averaged over t=201,…,300, and circuits over depths 1901,…,2000. The SWAP curve uses its exact asymptotic prediction, the initial BAEE.
- Floquet evolution uses exp(−i Hz) exp(−0.4i Hxy) for 3×10¹¹ periods, with Jz=1 in Hz and Jz=0 and zero fields in Hxy. There is no Trotter or MPS truncation. Finite-precision eigenvalues still limit very-long-time phases; extended-precision phase reduction does not remove that error.

## Results and verification

The generated L=12 curves show interior maxima for MBL and Floquet entanglement growth and an initially growing, then decreasing BAEE−HCEE reservoir, consistent with the paper's qualitative conclusions. The system is smaller than the paper’s L=16, so finite-size and sampling effects affect the detailed curves. Dashed lines mark the complex-Haar mean in the fixed-charge sector at the simulated L.

`verify.py` checks Hamiltonians, evolution and Floquet ordering against independent Kronecker products, matrix exponentials and direct powers at L=4,6. JAX checks at L=4,6,8,12 compare gates with full TC circuits and entropy batches with full-space SVD and `tc.quantum.entanglement_entropy` (accounting for the latter’s density-matrix regularization), covering all equal cuts, SWAP invariance, and circuit final states and late-window averages. A Bell-pair check fixes the entropy convention in bits.
