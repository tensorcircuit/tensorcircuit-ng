"""
15-to-1 magic state distillation via the [[15,1,3]] Reed-Muller code.

Reference: Bravyi & Kitaev, quant-ph/0403025.

────────────────────────────────────────────────────────────────────
WHY MAGIC STATE DISTILLATION?
────────────────────────────────────────────────────────────────────
Clifford gates (H, CNOT, S, …) can be implemented fault-tolerantly
by most quantum error-correcting codes.  The T gate (π/8 rotation)
cannot — it requires either a non-transversal implementation or a
separate resource called a "magic state" |T⟩ = T|+⟩.

Distillation takes 15 noisy copies of |T⟩ and outputs 1 higher-fidelity
copy, suppressing the logical error rate from O(p) to O(p³).  This
improvement comes from the fact that the code used has distance d=3:
it can detect (but not correct) up to 2 errors, so the first
undetectable error patterns have weight 3 and their probability is O(p³).

────────────────────────────────────────────────────────────────────
CODE STRUCTURE  (CSS, asymmetric)
────────────────────────────────────────────────────────────────────
The [[15,1,3]] code is a CSS code built from two classical Reed-Muller
codes L1 ⊂ L2 (BK eq. 26).  Qubits are indexed by the 15 nonzero
4-bit vectors x ∈ {1,…,15}, so qubit j ↔ x = j+1.

  L1 = punctured RM(1,4): the [15,4,8] simplex code.
       4 generators: each is a linear form x_i  (i = 0..3), i.e. the
       set of 4-bit vectors whose i-th bit is 1.  Each row has weight 8.
       This gives H_X (4 × 15).

  L2 = punctured RM(2,4): the [15,11,4] punctured Hamming-dual code.
       10 generators: the 4 linear forms above  PLUS  the 6 quadratic
       forms x_i · x_j (0 ≤ i < j ≤ 3), which select qubits where both
       bit i AND bit j of (j+1) are 1.  Weight-4 rows.
       This gives H_Z (10 × 15).

CSS construction:
  X-type stabilizers ← rows of H_X   (4 generators)
  Z-type stabilizers ← rows of H_Z   (10 generators)
  Code parameters: n=15, k = 15 − rank(H_X) − rank(H_Z) = 15−4−10 = 1
                   d_X = 7 (minimum weight in null(H_X) \\ rowspan(H_Z))
                   d_Z = 3 (minimum weight in null(H_Z) \\ rowspan(H_X))
                   d   = min(d_X, d_Z) = 3  →  [[15,1,3]]

CSS commutativity requirement: H_X · H_Z^T = 0 (mod 2).
  Proof sketch: each linear form dotted with each quadratic form has
  even overlap (the product x_i · x_i · x_j = x_j contributes pairs),
  and two linear forms x_i, x_j dot to give weight 4, which is even.

────────────────────────────────────────────────────────────────────
TRANSVERSAL T AND LOGICAL T†
────────────────────────────────────────────────────────────────────
T⊗15 implements logical T†.  The argument uses the weight distribution
of codewords in null(H_Z):

  null(H_Z) = C_0 ∪ C_1    (disjoint cosets)

  C_0 = rowspan(H_X) — the "logical-0 support":
        weights ∈ {0, 8}  (all ≡ 0 mod 8)
        Phase from T⊗15: e^{iπ·w/4} = e^{0} = 1  for every w ∈ {0,8}

  C_1 = x_L ⊕ C_0 — the "logical-1 support" (x_L = logical-X representative):
        weights ∈ {7, 15}  (all ≡ 7 mod 8)
        Phase from T⊗15: e^{i7π/4} = e^{-iπ/4} for every w ∈ {7,15}

  T⊗15 |+_L⟩ = T⊗15 · (1/√|null|) Σ_c |c⟩
              = (1/√|null|) [Σ_{c∈C_0} 1·|c⟩ + Σ_{c∈C_1} e^{-iπ/4}·|c⟩]
              = (|0_L⟩ + e^{-iπ/4}|1_L⟩) / √2
              = T†|+⟩_L  =  |T†_L⟩   ✓

The output of the distillation is therefore |T†⟩; applying T gives
|+⟩, and then H gives |0⟩ — a deterministic 0 measurement for
error-free runs.

────────────────────────────────────────────────────────────────────
ENCODER / DECODER FOR |+_L⟩
────────────────────────────────────────────────────────────────────
RREF of H_Z reveals the code's free-variable structure:
  pivot columns : [0,1,2,3,4,5,7,8,9,11]  (10 cols, one per Z-stabilizer)
  free  columns : [6,10,12,13,14]           (5 cols, one per logical/gauge DOF)

A codeword c ∈ null(H_Z) is uniquely determined by its values on the
5 free columns.  The 5-dimensional null space is spanned by basis vectors
v_{6}, v_{10}, v_{12}, v_{13}, v_{14}, where v_f is the null vector
whose free-column pattern is the unit vector e_f.

|+_L⟩ = uniform superposition over ALL 2^5 = 32 null(H_Z) vectors:
         |+_L⟩ = (1/√32) Σ_{s ∈ {0,1}^5} |v_{s_6,s_{10},s_{12},s_{13},s_{14}}⟩

Encoder (from |0⟩^15):
  Step 1: H on every free qubit  → (1/√32) Σ_s |s_6,s_10,s_12,s_13,s_14,0,...⟩
  Step 2: CNOT(free→pivot) per RREF row → maps each s to the corresponding
           null vector by setting pivot qubits to XOR of their free-column
           contributions, giving exactly |+_L⟩.

Decoder (inverse):
  Step 1: Reverse CNOTs  → returns pivot qubits to |0⟩, free qubits hold s
  Step 2: H on every free qubit  → Fourier-transforms back to standard basis

After decode, the 5 free qubits hold the logical+gauge information.
The *logical* degree of freedom is the joint parity of all 5 free qubits
(because the logical-X operator in the free-qubit subspace is [1,1,1,1,1]).
This is explained in detail in the output probe section below.

────────────────────────────────────────────────────────────────────
OUTPUT PROBE AND WHY q6 ALONE IS NOT ENOUGH
────────────────────────────────────────────────────────────────────
After decoding |T†_L⟩, we do NOT get a product state with T†|+⟩ at q6.
Here is the derivation:

1. The 4 H_X generators in the free-qubit basis [6,10,12,13,14] are:
     h^0=[1,1,1,0,1], h^1=[1,1,0,1,1], h^2=[1,0,1,1,1], h^3=[0,1,1,1,1]
   All have EVEN parity (weight 4 each).  Their span is exactly the set of
   all 16 even-parity 5-bit strings = C_0 in the free-qubit picture.
   The odd-parity strings (all 16 of them) form C_1.

   Therefore: the phase e^{iπ|v_s|/4} depends only on parity(s):
     parity(s) = 0  →  s ∈ C_0  →  phase = 1
     parity(s) = 1  →  s ∈ C_1  →  phase = e^{-iπ/4}

   And parity(s) = s_6 ⊕ s_{10} ⊕ s_{12} ⊕ s_{13} ⊕ s_{14}.

2. After decode (reverse CNOTs + H^⊗5 on free qubits), the Fourier
   transform of a function that depends only on the 5-bit parity
   concentrates at exactly two outputs (t=00000 and t=11111):

     Σ_s (-1)^{t·s} e^{iπ|v_s|/4}
       = 16 · (1+e^{-iπ/4}) / 2   if t = 00000
       = 16 · (1-e^{-iπ/4}) / 2   if t = 11111
       = 0                          otherwise

   Dividing by √32 (normalization), the decoded state is:

     A₀|00000⟩ + A₁|11111⟩  in free qubits [6,10,12,13,14]

   where  A₀ = (1+e^{-iπ/4})/2
          A₁ = (1-e^{-iπ/4})/2

3. Note: A₀|0⟩ + A₁|1⟩ = H · T†|+⟩.
   (Verify: H·(|0⟩+e^{-iπ/4}|1⟩)/√2 gives coefficients A₀,A₁ above.)

   The logical information is encoded in the GHZ-like state — q6 and
   gauge qubits are ENTANGLED.  Measuring q6 alone gives a mixed result
   (~73% correct), not the deterministic 0 we need.

4. PROBE PROCEDURE (fan-out CNOTs + H + T + H + measure on q6):

   a. CNOT(q6→q10), CNOT(q6→q12), CNOT(q6→q13), CNOT(q6→q14):
      |00000⟩ → |00000⟩      (q6=0, no flips)
      |11111⟩ → |10000⟩      (q6=1, each gauge qubit: 1⊕1=0)
      Result: gauge qubits disentangled, q6 holds A₀|0⟩+A₁|1⟩ = H·T†|+⟩

   b. H on q6:   H·(H·T†|+⟩) = T†|+⟩ = (|0⟩+e^{-iπ/4}|1⟩)/√2

   c. T on q6:   T·T†|+⟩ = |+⟩ = (|0⟩+|1⟩)/√2

   d. H on q6:   H|+⟩ = |0⟩

   e. Measure → 0  (deterministically, no logical error)
"""

import numpy as np
import tensorcircuit as tc
from tensorcircuit.zx.stabilizertcircuit import StabilizerTCircuit

tc.set_backend("jax")


# ============================================================
# 1.  [[15,1,3]] stabilizer generator matrices
# ============================================================


def build_hx():
    """Return H_X as a (4 × 15) numpy array over GF(2).

    H_X generates the X-type stabilizers.  Row i is the linear form x_i:
    it marks qubit j whenever bit i of (j+1) is 1.  Equivalently, row i
    is the characteristic vector of the set {x ∈ {1..15} : x_i = 1}.

    All 4 rows have weight 8 (exactly half of the 15 qubits).
    CSS commutativity: H_X · H_Z^T = 0 (mod 2) — each linear form has
    even overlap with every quadratic form and with every other linear form.
    """
    return np.array(
        [[(x >> i) & 1 for x in range(1, 16)] for i in range(4)], dtype=np.int32
    )


def build_hz():
    """Return H_Z as a (10 × 15) numpy array over GF(2).

    H_Z generates the Z-type stabilizers.  It is partitioned as:

      Rows 0–3  : same 4 linear forms x_i as H_X  (weight 8 each).
                  These rows lie inside rowspan(H_X) and inside null(H_Z),
                  which ensures the transversal-T weight argument holds:
                  all codewords in C_0 = rowspan(H_X) have weight ≡ 0 mod 8.

      Rows 4–9  : 6 quadratic forms x_i · x_j  (0 ≤ i < j ≤ 3).
                  Qubit j is in the support iff both bit i AND bit j
                  of (j+1) are 1.  Each row has weight 4.

    The 10 rows together give rank 10 over GF(2), so
      k = 15 − 4 − 10 = 1  logical qubit.

    Weight distribution of null(H_Z):
      C_0 = rowspan(H_X) has weights {0, 8}  (≡ 0 mod 8)
      C_1 = logical-X coset has weights {7, 15}  (≡ 7 mod 8)
    This weight-mod-8 property is the triorthogonality condition that
    makes T⊗15 implement logical T† rather than an unstructured map.
    """
    H_X = build_hx()
    quad_rows = [
        [int(((x >> i) & 1) and ((x >> j) & 1)) for x in range(1, 16)]
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    return np.vstack([H_X, np.array(quad_rows, dtype=np.int32)])


def get_checks():
    """Return (x_checks, z_checks) as lists of qubit-index lists.

    x_checks : 4  entries  (H_X rows → X-type stabilizers, weight 8)
    z_checks : 10 entries  (H_Z rows → Z-type stabilizers, weight 8 or 4)

    These are the supports used to build ancilla-based parity measurements.
    """
    x_checks = [list(np.where(row)[0]) for row in build_hx()]
    z_checks = [list(np.where(row)[0]) for row in build_hz()]
    return x_checks, z_checks


# ============================================================
# 2.  Encoder / decoder for |+_L⟩
# ============================================================

# RREF of H_Z over GF(2).  Computing the row-reduced echelon form reveals
# which columns are "pivot" (determined by a stabilizer) and which are "free"
# (independent logical/gauge degrees of freedom).
#
#   pivot columns : [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]  (10 cols = rank(H_Z))
#   free  columns : [6, 10, 12, 13, 14]               (5 cols = null-space dim)
#
# Each free column f defines one basis vector of null(H_Z): the 15-bit vector
# v_f that has 1 at position f and 1 at each pivot position p that appears in
# the same RREF row as f.  The CNOT pairs below encode exactly this structure:
# each (ctrl=free, tgt=pivot) pair records that pivot qubit tgt depends on
# free qubit ctrl in the codeword.
_FREE_COLS = [6, 10, 12, 13, 14]
_CNOT_PAIRS = [  # (control=free, target=pivot) in forward order
    (6, 0),
    (10, 0),
    (12, 0),
    (6, 1),
    (10, 1),
    (13, 1),
    (6, 2),
    (10, 2),
    (14, 2),
    (6, 3),
    (12, 3),
    (13, 3),
    (6, 4),
    (12, 4),
    (14, 4),
    (6, 5),
    (13, 5),
    (14, 5),
    (10, 7),
    (12, 7),
    (13, 7),
    (10, 8),
    (12, 8),
    (14, 8),
    (10, 9),
    (13, 9),
    (14, 9),
    (12, 11),
    (13, 11),
    (14, 11),
]

# The logical qubit's "home" free column.  This is the first free column of
# the H_Z RREF and serves as the measurement target in the output probe.
_LOGICAL_QUBIT = 6


def encode_logical_plus(stc):
    """Prepare |+_L⟩ on data qubits 0..14 (all must start in |0⟩).

    |+_L⟩ is the uniform superposition over all 32 = 2^5 codewords in
    null(H_Z).  The encoding uses two steps:

    Step 1 — H on ALL 5 free qubits [6, 10, 12, 13, 14].
      Creates (1/√32) Σ_{s ∈ {0,1}^5} |s_6, s_10, s_12, s_13, s_14, 0,…⟩.

      H must be applied to ALL 5 free qubits, not just the logical one (q6),
      because:
        • 4 of the 5 free qubits correspond to gauge degrees of freedom
          (the X-stabilizer group in free-qubit space spans all even-parity
          5-bit strings).
        • Applying H to all 5 simultaneously projects each X-stabilizer
          to its +1 eigenvalue, placing the state inside the code space.
        • Omitting any single free qubit leaves the corresponding X-stabilizer
          in a random +1/−1 eigenstate, causing non-zero X-syndromes.

    Step 2 — CNOTs (free → pivot) from the RREF of H_Z.
      Each CNOT(f, p) sets pivot qubit p to be the XOR of all free qubits f
      that appear in the same RREF row.  This ensures H_Z · |codeword⟩ = 0
      for every basis state in the superposition, i.e. all states are valid
      Z-stabilizer eigenstates with eigenvalue +1.
    """
    for q in _FREE_COLS:
        stc.h(q)
    for ctrl, tgt in _CNOT_PAIRS:
        stc.cnot(ctrl, tgt)


def decode_logical(stc):
    """Inverse of encode_logical_plus.

    Reverses the encoding to map from the code space back to the
    free-qubit representation:

    Step 1 — Reverse CNOTs (applied in reverse order).
      Undoes the pivot-setting step.  After this, pivot qubits return to |0⟩
      and the 5 free qubits carry the logical+gauge information as a
      superposition of 5-bit strings.

    Step 2 — H on ALL 5 free qubits.
      Fourier-transforms the free-qubit state back to the standard basis.
      For |+_L⟩ (uniform over 32 strings), H^⊗5 returns |00000⟩ — all
      five free qubits in |0⟩.

      For |T†_L⟩ (which arises after T⊗15 on |+_L⟩), the H^⊗5 step
      produces the GHZ-like state A₀|00000⟩ + A₁|11111⟩.  The full
      derivation is in the module docstring; see also the probe section
      in _make_circuit.
    """
    for ctrl, tgt in reversed(_CNOT_PAIRS):
        stc.cnot(ctrl, tgt)
    for q in _FREE_COLS:
        stc.h(q)


# ============================================================
# 3.  Syndrome measurement helpers
# ============================================================


def measure_x_parity(stc, anc, support):
    """Measure the X-parity ⊗_{i ∈ support} X_i using ancilla qubit anc.

    Circuit: H(anc) — CNOT(anc→q) for q in support — H(anc) — measure(anc).

    Derivation: the ancilla starts in |0⟩.
      After H: |+⟩ = (|0⟩+|1⟩)/√2 — ancilla is now an X-basis probe.
      Each CNOT(anc, q) controlled on the ancilla performs X_q when anc=|1⟩.
      By the Hadamard conjugation identity, this is equivalent to controlled-X
      propagating *from* data *to* ancilla in the X basis, accumulating the
      X-parity of the support qubits into the ancilla phase.
      The final H on the ancilla converts the phase kickback into a bit value:
        ancilla = 0  ↔  +1 eigenstate of ⊗ X_i  (no error detected)
        ancilla = 1  ↔  −1 eigenstate (X-syndrome triggered)
    """
    stc.h(anc)
    for q in support:
        stc.cnot(anc, q)
    stc.h(anc)
    stc.measure_instruction(anc)


def measure_z_parity(stc, anc, support):
    """Measure the Z-parity ⊗_{i ∈ support} Z_i using ancilla qubit anc.

    Circuit: CNOT(q→anc) for q in support — measure(anc).

    Derivation: the ancilla starts in |0⟩.
      Each CNOT(q, anc) fans out the Z value of qubit q to the ancilla:
      after all CNOTs, the ancilla holds XOR of all data qubit values in
      the support.  Measuring in the Z basis reads out this parity directly.
        ancilla = 0  ↔  even Z-parity (no Z-type error detectable here)
        ancilla = 1  ↔  odd Z-parity (Z-syndrome triggered)
    """
    for q in support:
        stc.cnot(q, anc)
    stc.measure_instruction(anc)


# ============================================================
# 4.  Circuit builder
# ============================================================


def _make_circuit(p):
    """Build the distillation circuit for vectorized multi-shot sampling.

    Instead of injecting explicit Pauli gates per trial, this circuit uses
    DEPOLARIZE1(p) probabilistic noise channels after each T gate.  At
    sample time, the ChannelSampler draws independent error realizations for
    each shot, so one compiled circuit serves all N shots in a single
    sample_detectors(shots=N) call.

    Circuit stages:
      1. Encode  |+_L⟩ on data qubits 0..14.
      2. T⊗15 with DEPOLARIZE1(p) noise channels on each data qubit.
      3. 14 syndrome measurements (4 X-type + 10 Z-type) with DETECTOR
         annotations.  Measurement record indices are 0-based in order of
         measure_instruction calls; DETECTOR([i]) fires when measurement i
         deviates from its noiseless value (0 for every syndrome).
      4. Decode: reverse the encoding to extract the logical state.
      5. Probe: fan-out CNOTs + H + T + H on qubit 6, then measure.
         OBSERVABLE_INCLUDE([14]) marks the probe as the logical observable.

    After sample_detectors(shots=N, separate_observables=True):
      det_samples : (N, 14) array — 1 if that syndrome fired (error detected)
      obs_samples : (N,  1) array — 1 if undetected logical error occurred

    Post-select on det_samples == 0 to compute accept rate and logical error rate.
    """
    x_checks, z_checks = get_checks()
    num_x = len(x_checks)  # 4  (H_X rows)
    num_z = len(z_checks)  # 10 (H_Z rows)
    n_total = 15 + num_x + num_z  # 29 qubits total

    stc = StabilizerTCircuit(n_total)

    # ── Stage 1: Prepare |+_L⟩ ──────────────────────────────────────────
    encode_logical_plus(stc)

    # ── Stage 2: Transversal T with depolarizing noise ───────────────────
    # DEPOLARIZE1(p) = X with prob p/3, Y with prob p/3, Z with prob p/3.
    # The noise channel is compiled into the ZX graph; the ChannelSampler
    # samples independent error bits per shot at inference time.
    for q in range(15):
        stc.t(q)
        if p > 0:
            stc.depolarizing(q, p)

    # ── Stage 3: Syndrome extraction with DETECTOR annotations ──────────
    # Ancilla layout: qubits 15..18 for X-checks, 19..28 for Z-checks.
    # Measurement index meas_idx counts measure_instruction calls (0-based).
    # DETECTOR([meas_idx]) compares measurement meas_idx to its noiseless
    # expected value (0), so the detector fires exactly when the syndrome is 1.
    meas_idx = 0
    for i, support in enumerate(x_checks):
        measure_x_parity(stc, 15 + i, support)
        stc.detector_instruction([meas_idx])
        meas_idx += 1
    for i, support in enumerate(z_checks):
        measure_z_parity(stc, 15 + num_x + i, support)
        stc.detector_instruction([meas_idx])
        meas_idx += 1

    # ── Stage 4: Decode ──────────────────────────────────────────────────
    # After decoding the ideal |T†_L⟩, free qubits [6,10,12,13,14] hold
    # the GHZ-like state A₀|00000⟩ + A₁|11111⟩ (see module docstring).
    decode_logical(stc)

    # ── Stage 5: Probe ───────────────────────────────────────────────────
    # Fan-out CNOTs from q6 disentangle gauge qubits: |11111⟩ → |10000⟩.
    # q6 then holds A₀|0⟩+A₁|1⟩ = H·T†|+⟩.
    # Sequence H → T → H maps H·T†|+⟩ → T†|+⟩ → |+⟩ → |0⟩.
    for gauge_q in [q for q in _FREE_COLS if q != _LOGICAL_QUBIT]:
        stc.cnot(_LOGICAL_QUBIT, gauge_q)
    stc.h(_LOGICAL_QUBIT)
    stc.t(_LOGICAL_QUBIT)
    stc.h(_LOGICAL_QUBIT)
    stc.measure_instruction(_LOGICAL_QUBIT)
    # OBSERVABLE_INCLUDE: probe bit = 0 means correct distillation.
    # The observable flips (= 1) exactly when an undetected logical error occurs.
    stc.observable_instruction([meas_idx], observable_index=0)

    return stc


# ============================================================
# 5.  Vectorized simulation
# ============================================================


def simulate_15to1_stc(p=0.01, shots=10000):
    """Simulate the 15-to-1 protocol for N shots in a single vectorized call.

    Compiles one circuit with probabilistic noise channels and calls
    sample_detectors(shots=N).  The ChannelSampler draws independent error
    patterns for each shot, making this equivalent to N independent trials
    while keeping all computation inside one JAX execution.

    Post-processing:
      - Accepted shots : all 14 syndrome detectors fire 0 (no error detected).
      - Logical errors : observable = 1 among accepted shots.

    For small p, BK theory predicts:
      accept_rate        ≈ 1 − 15p     (each T gate adds ~p chance of error)
      logical error rate ∝ 35 p³       (first undetectable patterns have weight 3)

    Args:
      p     : depolarizing error probability per T gate (0 ≤ p ≤ 1)
      shots : number of independent distillation rounds to simulate

    Returns a dict with accept_rate and logical_proxy_error_rate.
    """
    stc = _make_circuit(p)
    det_samples, obs_samples = stc.sample_detectors(
        shots=shots, separate_observables=True, batch_size=10000
    )
    det_samples = np.array(det_samples)  # (shots, 14)
    obs_samples = np.array(obs_samples)  # (shots, 1)

    # Accept iff all 14 syndrome detectors fire 0 (no detectable error).
    accepted = np.all(det_samples == 0, axis=1)
    n_accepts = int(accepted.sum())

    accepted_obs = obs_samples[accepted, 0]
    logical_error_rate = float(accepted_obs.mean()) if n_accepts > 0 else None

    return {
        "p": p,
        "shots": shots,
        "accepts": n_accepts,
        "accept_rate": float(accepted.mean()),
        # At p=0 this must be exactly 0.0.  For small p, scales as ~35p³.
        "logical_proxy_error_rate": logical_error_rate,
        "num_total_qubits": 29,
        "num_data_qubits": 15,
        "num_ancilla_qubits": 14,
        "num_T_gates": 15,
    }


# ============================================================
# 6.  Main
# ============================================================


def main():
    print("=== 15-to-1 magic state distillation via [[15,1,3]] Reed-Muller code ===")
    print("Code structure (CSS, L1 ⊂ L2 from BK quant-ph/0403025):")
    print("  H_X : 4 rows  (X-stabs, L1 = punctured RM(1,4) linear forms, weight 8)")
    print("  H_Z : 10 rows (Z-stabs, L2 = linear + quadratic forms of RM(2,4))")
    print("  k = 15 - 4 - 10 = 1,  d_Z = 3,  d_X = 7,  d = 3  →  [[15,1,3]]")
    print("  T^⊗15 implements logical T†  (C_1 weights ≡ 7 mod 8)")
    print()
    print("Resource count per round:")
    print("  total qubits   = 29  (15 data + 14 ancilla)")
    print("  ancilla qubits = 14  (4 X-checks + 10 Z-checks)")
    print("  T gates        = 15")
    print()

    # ── Sanity check: p=0 must give perfect accept rate and zero logical errors.
    print("--- p=0 sanity check ---")
    sanity = simulate_15to1_stc(p=0.0, shots=200)
    assert sanity["accept_rate"] == 1.0, (
        f"p=0 accept_rate={sanity['accept_rate']} != 1.0; "
        "encoding or syndrome circuit is broken"
    )
    assert sanity["logical_proxy_error_rate"] == 0.0, (
        f"p=0 logical_error_rate={sanity['logical_proxy_error_rate']} != 0.0; "
        "output probe or decoder is broken"
    )
    print(
        f"  accept_rate={sanity['accept_rate']},  "
        f"logical_error_rate={sanity['logical_proxy_error_rate']}  OK"
    )
    print()

    # ── Monte Carlo sweep over physical error rates.
    # Expected trends per BK theory:
    #   accept_rate        decreases as ~1 - 15p  (more errors get detected)
    #   logical_error_rate scales as ~35p³         (cubic error suppression)
    for p in [0.001, 0.005, 0.01, 0.02]:
        result = simulate_15to1_stc(p=p, shots=1000000)
        print(result)


if __name__ == "__main__":
    main()
