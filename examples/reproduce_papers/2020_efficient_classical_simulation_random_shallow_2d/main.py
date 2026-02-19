"""
Reproduction of "Efficient classical simulation of random shallow 2D quantum circuits"
Link: https://arxiv.org/abs/2001.00021

Description:
This script implements the Spatial Evolution Block Decimation (SEBD) algorithm
described in the paper. It simulates random shallow 2D quantum circuits by mapping
the 2D circuit to a 1D dynamical process where one spatial dimension is treated as time.
This allows for efficient approximate simulation using Matrix Product States (MPS).

The script benchmarks the SEBD method against exact state vector simulation for small
grids and demonstrates its capability on larger grids.
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use numpy backend for broad compatibility and simplicity in manual tensor ops
K = tc.set_backend("numpy")


def generate_random_2d_circuit(rows, cols, depth, seed=42):
    """
    Generates a random shallow 2D quantum circuit on a grid.
    Architecture:
    - Qubits are arranged in a (rows x cols) grid.
    - Gates are applied in layers.
    - Each layer consists of random single-qubit rotations followed by
      nearest-neighbor entangling gates (e.g., CZ).
    - Entangling gates pattern follows a brickwork-like structure in 2D.
    """
    np.random.seed(seed)
    n_qubits = rows * cols
    c = tc.Circuit(n_qubits)

    # Helper to get qubit index from (r, c)
    def q(r, col):
        return r * cols + col

    # Apply gates
    # Note: TensorCircuit applies gates in order.
    # We should ensure single qubit gates are applied first in each layer if that's the model.

    for d in range(depth):
        # Single qubit gates layer
        for i in range(n_qubits):
            # Random SU(2) ~ Rz Ry Rz
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            lam = np.random.uniform(0, 2 * np.pi)
            c.rz(i, theta=phi)
            c.ry(i, theta=theta)
            c.rz(i, theta=lam)

        # Entangling gates (CZ)
        # Alternate patterns

        # For simplicity and to match the paper more closely (brickwork)
        # The paper typically uses 2-qubit gates on edges of a graph.
        # Let's ensure we are consistent.
        # The current implementation alternates layers.

        # Horizontal gates
        for r in range(rows):
            # Apply to (col, col+1)
            # Staggered:
            # If d is even: (0,1), (2,3)...
            # If d is odd: (1,2), (3,4)...
            start_c = d % 2
            for col in range(start_c, cols - 1, 2):
                c.cz(q(r, col), q(r, col + 1))

        # Vertical gates
        for col in range(cols):
            start_r = d % 2
            for r in range(start_r, rows - 1, 2):
                c.cz(q(r, col), q(r + 1, col))

    return c


def exact_probability(c, bitstring):
    """
    Computes the exact probability of measuring the given bitstring.
    """
    if c._nqubits > 20:
        logger.warning("Exact simulation for >20 qubits is slow/impossible. Skipping.")
        return 0.0

    # Calculate probability
    probs = c.probability()
    index = int(bitstring, 2)
    return probs[index]


def compress_mps(mps, max_bond):
    """
    Compresses the MPS using SVD truncation.
    MPS is a list of tensors with shape (Up, Down, Phys).
    We perform a sweep to canonicalize and truncate.
    """
    rows = len(mps)

    # Right canonicalization (QR) from bottom to top (rows-1 -> 0)
    for i in range(rows - 1, 0, -1):
        tensor = mps[i]  # (u, d, p)
        u_dim, d_dim, p_dim = tensor.shape

        # Reshape to combine d and p, separating u
        # T (u, d, p) -> Transpose to (d, p, u) -> Reshape (d*p, u)
        t_p = K.transpose(tensor, (1, 2, 0))
        t_flat = K.reshape(t_p, (d_dim * p_dim, u_dim))

        # QR decomposition
        # t_flat = Q @ R
        # Q: (d*p, k), R: (k, u)
        q, r = K.qr(t_flat)
        k_dim = r.shape[0]

        # New tensor for site i is Q
        # Reshape Q back to (d, p, k) -> Transpose to (k, d, p) (Up, Down, Phys)
        q_reshaped = K.reshape(q, (d_dim, p_dim, k_dim))
        mps[i] = K.transpose(q_reshaped, (2, 0, 1))

        # Contract R into site i-1
        # mps[i-1] (u_prev, d_prev, p_prev)
        # d_prev connects to u (dim 1 of R)
        # We need to contract d_prev with u
        # R is (k, u) -> Transpose to (u, k)
        r_t = K.transpose(r, (1, 0))
        prev = mps[i - 1]

        # Contract prev axis 1 (d_prev) with r_t axis 0 (u)
        # Result: (u_prev, p_prev, k)
        new_prev = K.tensordot(prev, r_t, axes=[[1], [0]])
        # Transpose to (u_prev, k, p_prev)
        mps[i - 1] = K.transpose(new_prev, (0, 2, 1))

    # Left canonicalization (SVD) from 0 to rows-1
    for i in range(rows - 1):
        tensor = mps[i]  # (u, d, p)
        u_dim, d_dim, p_dim = tensor.shape

        # Group u and p, separate d
        # T (u, d, p) -> Transpose (u, p, d) -> Reshape (u*p, d)
        t_p = K.transpose(tensor, (0, 2, 1))
        t_flat = K.reshape(t_p, (u_dim * p_dim, d_dim))

        # SVD
        u, s, v, _ = K.svd(t_flat)

        # Truncate
        if max_bond is not None and s.shape[0] > max_bond:
            u = u[:, :max_bond]
            s = s[:max_bond]
            v = v[:max_bond, :]

        k_dim = s.shape[0]

        # New tensor for site i is U
        # Reshape U to (u, p, k) -> Transpose to (u, k, p)
        u_reshaped = K.reshape(u, (u_dim, p_dim, k_dim))
        mps[i] = K.transpose(u_reshaped, (0, 2, 1))

        # Absorb S and V into next site
        # SV = S @ V -> (k, d_next)
        sv = K.tensordot(K.diagflat(s), v, axes=[[1], [0]])

        next_t = mps[i + 1]  # (u_next, d_next, p_next)
        # u_next matches d
        # Contract SV axis 1 with next_t axis 0
        new_next = K.tensordot(sv, next_t, axes=[[1], [0]])
        # Result (k, d_next, p_next)
        mps[i + 1] = new_next

    return mps


def sebd_probability(c, rows, cols, bitstring, max_bond=None):
    """
    Computes the probability of `bitstring` using SEBD.
    """
    # Map qubit index to (r, c)
    q_to_rc = {}
    for r in range(rows):
        for col in range(cols):
            q_to_rc[r * cols + col] = (r, col)

    # Initialize sites with state |0>
    sites = {}
    for i in range(rows * cols):
        sites[i] = {
            "tensor": K.cast(K.convert_to_tensor(np.array([1.0, 0.0])), "complex128"),
            "directions": [],
        }

    # Process gates to build PEPS
    for gate in c._qir:
        idx = gate["index"]
        func = gate["gatef"]
        param = gate.get("parameters", {})

        if len(idx) == 1:
            q = idx[0]
            # logger.info(f"Gate full: {gate}")
            mat = func(**param).tensor
            # logger.info(f"Gate on {q}: {func.n} params {param}")
            t = sites[q]["tensor"]
            # Contract: new_T_i... = M_ij * old_T_j...
            # M (2_new, 2_old). T (2_old, bonds...)
            sites[q]["tensor"] = K.tensordot(mat, t, axes=[[1], [0]])

        elif len(idx) == 2:
            q1, q2 = idx[0], idx[1]
            r1, c1 = q_to_rc[q1]
            r2, c2 = q_to_rc[q2]

            mat = func(**param).tensor
            mat = K.reshape(mat, (2, 2, 2, 2))

            # Note: mat indices are (out1, out2, in1, in2)
            # We want to contract in1 with T1, in2 with T2.
            # And produce out1 for T1, out2 for T2.

            # SVD split
            # We group (out1, in1) for q1 and (out2, in2) for q2.
            # Permute to (out1, in1, out2, in2)
            mat_p = K.transpose(mat, (0, 2, 1, 3))
            mat_flat = K.reshape(mat_p, (4, 4))

            u, s, v, _ = K.svd(mat_flat)
            # Full rank (max 4)
            k_dim = s.shape[0]

            u = K.reshape(u, (2, 2, k_dim))  # (out1, in1, k)
            v = K.reshape(v, (k_dim, 2, 2))  # (k, out2, in2)

            s_sqrt = K.sqrt(s)
            u = u * K.reshape(s_sqrt, (1, 1, k_dim))
            v = v * K.reshape(s_sqrt, (k_dim, 1, 1))

            # Determine direction
            dr, dc = r2 - r1, c2 - c1
            dir1, dir2 = "Generic", "Generic"
            if dc == 1:
                dir1, dir2 = "Right", "Left"
            elif dc == -1:
                dir1, dir2 = "Left", "Right"
            elif dr == 1:
                dir1, dir2 = "Down", "Up"
            elif dr == -1:
                dir1, dir2 = "Up", "Down"

            # Update T1
            t1 = sites[q1]["tensor"]
            # T1 shape: (Phys_in, Bonds...)
            # u shape: (Phys_out, Phys_in, Bond_k)
            # Contract T1 Phys_in (0) with u Phys_in (1)
            t1_new = K.tensordot(u, t1, axes=[[1], [0]])  # (Phys_out, Bond_k, Bonds...)
            sites[q1]["tensor"] = t1_new
            sites[q1]["directions"].insert(0, dir1)

            # Update T2
            t2 = sites[q2]["tensor"]
            # T2 shape: (Phys_in, Bonds...)
            # v shape: (Bond_k, Phys_out, Phys_in)
            # Contract T2 Phys_in (0) with v Phys_in (2)
            t2_new = K.tensordot(v, t2, axes=[[2], [0]])  # (Bond_k, Phys_out, Bonds...)
            # We want (Phys_out, Bond_k, Bonds...)
            # Current axes: 0->k, 1->Phys_out, 2...->Bonds
            # Transpose: 1, 0, 2...
            perm = [1, 0] + list(range(2, len(t2_new.shape)))
            t2_new = K.transpose(t2_new, perm)
            sites[q2]["tensor"] = t2_new
            sites[q2]["directions"].insert(0, dir2)

    # Project onto final bitstring
    # The bitstring determines the final state <b1 b2 ... bn|
    # which is contracted with the physical legs of the PEPS.

    # IMPORTANT:
    # The gates applied were Unitaries U acting on |0>.
    # sites[i]['tensor'] represents U|0>_i ... (with entanglements)
    # The final state is |psi> = PEPS.
    # We want amplitude <b1...bn|psi>.
    # So we project each site onto <bi|.

    # However, we must be careful with indices.
    # Initial state |0> (Tensor shape (2,)). Index 0 is Physical.
    # Single qubit gates M (2, 2) applied: M_ij T_j -> T'_i. Index 0 is Physical.
    # Two qubit gates SVD:
    # q1: u (2_out, 2_in, k). T1 (2_in, ...). -> T1' (2_out, k, ...). Index 0 is Physical.
    # q2: v (k, 2_out, 2_in). T2 (2_in, ...). -> T2' (k, 2_out, ...).
    # Then we permuted T2' to (2_out, k, ...). Index 0 is Physical.

    # So for ALL tensors, index 0 is the physical output index (ket).
    # To compute <b|psi>, we contract <b| (bra) with physical index 0.
    # <0| = [1, 0], <1| = [0, 1].
    # In code: vec[b] = 1.0. This is the vector representation of <b|?
    # Yes, if we treat T as a vector in the basis.

    bits = [int(x) for x in bitstring]
    for i in range(rows * cols):
        b = bits[i]
        vec = np.array([0.0, 0.0])
        vec[b] = 1.0
        # Conjugate? <b| is conjugate of |b>. But basis vectors are real.
        vec = K.cast(K.convert_to_tensor(vec), "complex128")
        t = sites[i]["tensor"]

        # Contract axis 0
        sites[i]["tensor"] = K.tensordot(vec, t, axes=[[0], [0]])  # (bonds...)

    # Check if projection worked correctly.
    # If the state was |0> (tensor [1,0]), and b=0 (vec [1,0]). Dot is 1.
    # If b=1 (vec [0,1]). Dot is 0.
    # This logic is correct.

    # Standardize tensors
    grid = [[None for _ in range(rows)] for _ in range(cols)]

    def standardize_tensor(site_data):
        t = site_data["tensor"]
        dirs = site_data["directions"]
        groups = {"Up": [], "Down": [], "Left": [], "Right": []}
        for idx, d in enumerate(dirs):
            if d in groups:
                groups[d].append(idx)

        perm = groups["Up"] + groups["Down"] + groups["Left"] + groups["Right"]
        t = K.transpose(t, perm)

        shape = t.shape
        curr = 0
        new_shape = []
        for g in ["Up", "Down", "Left", "Right"]:
            dim = 1
            for _ in groups[g]:
                dim *= shape[curr]
                curr += 1
            new_shape.append(dim)
        return K.reshape(t, tuple(new_shape))

    for c_idx in range(cols):
        for r in range(rows):
            grid[c_idx][r] = standardize_tensor(sites[r * cols + c_idx])

    # SEBD Contraction (Boundary MPS)
    # MPS sites r=0..rows-1
    # mps[r] tensor shape: (Up, Down, Right)
    # Initially, we have a dummy boundary MPS.
    # The first column contraction takes T[r,0] (shape: U, D, 1, R)
    # and effectively sets MPS[r] = T[r,0].
    # So we can initialize MPS as "identity" that passes T through.
    # OR we just handle the first column separately.

    # Initialization:
    # mps[r] should be (1, 1, 1) if we assume virtual bonds are 1.
    # mps[r] connects to T[r,0].Left.
    # T[r,0].Left has dim 1.
    # So mps[r].Right has dim 1.
    # Correct.

    mps = [K.ones((1, 1, 1), dtype="complex128") for _ in range(rows)]

    for c_idx in range(cols):
        # Contract column into MPS
        new_mps = []
        for r in range(rows):
            m = mps[r]  # (v_u, v_d, p_l)
            t = grid[c_idx][r]  # (c_u, c_d, c_l, c_r)

            # Contract p_l with c_l
            # NOTE: p_l connects to c_l.
            res = K.tensordot(m, t, axes=[[2], [2]])
            # res: (v_u, v_d, c_u, c_d, c_r)

            # Group (v_u, c_u) -> U'
            # Group (v_d, c_d) -> D'
            # c_r -> Right

            # Permute to (v_u, c_u, v_d, c_d, c_r)
            res = K.transpose(res, (0, 2, 1, 3, 4))
            s = res.shape

            # Fuse vertical bonds
            # This is the standard PEPS-to-MPS step.
            new_t = K.reshape(res, (s[0] * s[1], s[2] * s[3], s[4]))
            new_mps.append(new_t)

        mps = new_mps
        if max_bond is not None:
            mps = compress_mps(mps, max_bond)

    # Contract final MPS
    # The final MPS represents the state after processing all columns.
    # It should be contracted with the Right boundary (vacuum).
    # Since we have no gates to the right, we just close the open legs (Left of the vacuum, Right of the grid).
    # But wait, the current MPS tensors have shape (Up, Down, Right).
    # The 'Right' index connects to the grid?
    # No, the 'Right' index IS the grid output to the right.
    # If there are no more columns, we contract these with Vacuum <0| ?
    # Or simply close them if they are dummy?

    # Actually, in our construction:
    # mps[r] came from contracting Old MPS (Left) with Grid Tensor (Left).
    # Grid Tensor has Right index.
    # So New MPS has Right index.
    # If this is the last column, the 'Right' index corresponds to the boundary condition on the right.
    # If open boundary, we project onto <0|? Or if it's temporal, it's open.
    # But for a circuit, the "Right" boundary is just time=T?
    # No, we mapped Time to a spatial dimension.
    # The PEPS is the amplitude.
    # The "Right" boundary corresponds to the last column of the circuit?
    # Wait, we iterated columns c=0..cols-1.
    # The 'Right' legs of the last column are dangling.
    # But in 'standardize_tensor', if no 'Right' direction, we added dim 1.
    # For the last column, there are no gates connecting to the right.
    # So 'Right' dim should be 1.
    # So the final MPS tensors have shape (u, d, 1).
    # This means the 'Right' index is effectively scalar.
    # So we just contract the vertical chain.

    # BUT, we initialized MPS with ONES.
    # mps = [K.ones((1, 1, 1)) ...].
    # This means we started with <Phi| where Phi is product of |+> (sum of all states)?
    # No, indices are virtual bonds.
    # A bond dimension 1 with value 1 means a product state?
    # If we are computing <b|U|0>, the boundaries of the PEPS are:
    # Left (First col): No gates to left.
    # Right (Last col): No gates to right.
    # Top (First row): No gates above.
    # Bottom (Last row): No gates below.

    # Our `standardize_tensor` handles "missing" directions by dimension 1.
    # This implies a boundary condition.
    # When we contract dimension 1 with dimension 1, it's just a product.
    # This effectively closes the boundary with a "cap" of value 1.
    # Is this correct?
    # Yes, for open boundaries in tensor networks, this is standard.

    # So why is the result 1.0?
    # This happens if the whole contraction evaluates to 1.
    # If we have Unitary Grid contracted with <0|s?
    # No, we have random bitstring <b|.

    # Wait! I found a potential bug in `standardize_tensor`.
    # shape = t.shape
    # curr = 0
    # for g in ...:
    #    dim = 1
    #    for _ in groups[g]:
    #        dim *= shape[curr]
    #        curr += 1
    #    new_shape.append(dim)

    # If groups[g] is empty, dim=1. curr does not increment.
    # This is correct.
    # But `t` was permuted before:
    # perm = groups['Up'] + ...
    # So `t` indices ARE ordered by groups.
    # So this logic is correct.

    # What about `t` passed to it?
    # In `sebd_probability`, we updated `t` with gate tensors.
    # `directions` keeps track of indices 1..N.
    # Index 0 is Physical (contracted).
    # So `t` has N indices.
    # `directions` has N elements.
    # This seems correct.

    # Let's debug `compress_mps`.
    # It modifies `mps` in place.
    # SVD truncation.
    # If we don't truncate (max_bond large), it should be exact.

    # Wait, `mps` contraction loop.
    # res = ones((1, 1))
    # for r:
    #   m (u, d)
    #   res (1, u_prev).
    #   res @ m -> (1, d).
    #   This contracts res[0, k] * m[k, j].
    #   This effectively multiplies matrices along the chain.
    #   This is correct for contracting a chain.
    #   The first `u` (from mps[0]) must match `d` of "virtual top boundary"?
    #   mps[0] shape (1, d0, 1). u=1.
    #   res has shape (1, 1).
    #   So we contract index 0 of mps[0] (dim 1) with index 1 of res (dim 1).
    #   This works.

    # What if the tensors are all normalized such that they sum to 1?
    # Unitary evolution preserves norm.
    # But we project onto specific bitstring.
    # The sum of probabilities should be 1.
    # The amplitude squared for ONE bitstring should be small.

    # Is it possible that `K.tensordot` with `axes=[[0], [0]]` is wrong?
    # vec (2,). t (2, bonds...).
    # axes=[[0], [0]].
    # Sum over index 0.
    # vec[0]*t[0, ...] + vec[1]*t[1, ...].
    # This is <b|t>. Correct.

    # Maybe `generate_random_2d_circuit` isn't doing what we think?
    # We printed the gates. They look real.

    # I suspect the initial state of the MPS or Grid contraction logic.
    # mps = [ones(1, 1, 1)].
    # This corresponds to a left boundary vector |L> = product of |1> states (unnormalized?) on the virtual bonds?
    # No, the bond is dimension 1.
    # It basically feeds "1" into the left side of the grid.
    # Is this correct?
    # The grid tensors T have shape (U, D, L, R).
    # If col=0, L=1.
    # We contract MPS[r].P (which is 1) with T[r].L (which is 1).
    # It's 1 * T[...].
    # This is correct for open boundary.

    # Let's verify `c_large` contraction.
    # It gives 0.99995.
    # This suggests that REGARDLESS of the circuit, we get ~1.
    # This usually means we are computing the Norm of the state (which is 1).
    # How could we be computing the norm?
    # Norm = <psi|psi>.
    # We are computing <b|psi>.
    # Unless... we contracted with Identity somewhere instead of projection?

    # Look at projection again.
    # vec[b] = 1.0. vec is [1, 0] or [0, 1].
    # This is a projection.

    # Is it possible that `t` has the physical index somewhere else?
    # We established index 0 is Phys.

    # Wait...
    # When we create `t = tensordot(u, t1, axes=[[1], [0]])`.
    # u (2_out, 2_in, k).
    # t1 (2_in, ...).
    # u index 1 matches t1 index 0.
    # Result indices: (2_out, k, ...).
    # Phys is at 0.

    # When we create `t = tensordot(mat, t, axes=[[1], [0]])`.
    # mat (2_new, 2_old).
    # t (2_old, ...).
    # Result (2_new, ...).
    # Phys is at 0.

    # What if we have NO gates on a qubit?
    # t is [1, 0].
    # bit is 0. vec is [1, 0].
    # dot is 1.
    # bit is 1. vec is [0, 1].
    # dot is 0.
    # So for Identity circuit, prob of |0...0> is 1. Prob of |1...1> is 0.
    # This is correct.

    # What if we have H gate?
    # t -> [1/sq2, 1/sq2].
    # bit 0 -> 1/sq2. Prob 0.5.

    # So logic holds.

    # Why is it failing for the random circuit?
    # Is it possible that `generate_random_2d_circuit` logic is flawed such that everything cancels out?
    # We saw "rz", "ry", "rz", "cz".
    # Random parameters.
    # It shouldn't cancel.

    # Let's check `get_gate_tensors` or `sebd_probability`'s loop over gates.
    # for gate in c._qir:
    # Maybe we are missing gates?
    # c._qir is the standard way.

    # WAIT!
    # I see `sites = {}`.
    # `sites[i] = {'tensor': ...}`.
    # When we update `sites[q]`, we replace the dictionary entry.
    # BUT, what if `q1` and `q2` are the same reference?
    # `sites[q1]` returns a dict.
    # We modify `sites[q1]['tensor']`.
    # This is fine.

    # Let's look at `standardize_tensor` again.
    # Permutation.
    # perm = groups['Up'] + groups['Down'] + ...
    # groups['Up'] contains INDICES from `directions`.
    # `directions` is a list of strings.
    # enumerate(directions).
    # Example: directions=['Right', 'Up'].
    # groups['Right']=[0]. groups['Up']=[1].
    # perm = [1, 0] + [] + [].
    # Tensor indices: (Phys(contracted), Right, Up) -> (Right, Up).
    # We want (Up, Down, Left, Right).
    # We permute to (Up, Right).
    # Reshape to (dim_u, 1, 1, dim_r).
    # Correct.

    # Wait... tensor indices were (Right, Up) because directions were ['Right', 'Up'].
    # Axis 0 corresponds to 'Right'. Axis 1 corresponds to 'Up'.
    # Because 'Right' was inserted LATEST (at 0).
    # 'Up' was inserted EARLIER (at 0, then pushed to 1).
    # So `directions[0]` is the axis 0 of `t` (after Phys contraction).
    # Yes.

    # BUT `standardize_tensor` takes `t`.
    # `t` axes match `directions`.
    # So `perm` logic must map `t` axes to (Up, Down, Left, Right).
    # `perm` is a list of indices.
    # `transpose` uses `perm` to reorder.
    # If we want Up to be first. We need the index of 'Up' in `t`.
    # That is exactly what `groups['Up']` contains.
    # So `perm = groups['Up'] + ...` puts Up indices first.
    # This is correct.

    # I am running out of ideas.
    # Maybe the problem is the `K.tensordot` implementation in numpy backend?
    # Or `K.reshape`.

    # Let's try to verify with a simple case.
    # 2x1 grid. 2 qubits.
    # Gate H on 0. CNOT 0->1.
    # Target |00>: 0.5. |11>: 0.5. |01>: 0.

    # Maybe I can add a small test case in `main` to debug.

    res = K.ones((1, 1), dtype="complex128")  # Shape (1, 1)
    for r in range(rows):
        m = mps[r]  # (u, d, 1)
        u_dim = m.shape[0]
        d_dim = m.shape[1]
        m = K.reshape(m, (u_dim, d_dim))
        res = K.matmul(res, m)  # (1, u) @ (u, d) -> (1, d)

    # Final res should be (1, 1)
    prob_amp = K.reshape(res, (1,))[0]

    # Add a print for debug
    # print(f"Final amp: {prob_amp}")

    return np.abs(K.numpy(prob_amp)) ** 2


def test_sebd_simple():
    """
    Test SEBD on a simple entangled circuit (Bell pair).
    """
    logger.info("Running simple SEBD test (Bell pair)...")
    # 2 qubits. 2x1 grid? Or 1x2?
    # Grid: rows=2, cols=1.
    # Qubits: (0,0) and (1,0).

    # Circuit
    c = tc.Circuit(2)
    c.h(0)
    c.cnot(0, 1)

    # Target |00> -> 0.5. |11> -> 0.5.

    rows, cols = 2, 1

    p00 = sebd_probability(c, rows, cols, "00")
    p11 = sebd_probability(c, rows, cols, "11")
    p01 = sebd_probability(c, rows, cols, "01")

    logger.info(f"P(00) = {p00}")
    logger.info(f"P(11) = {p11}")
    logger.info(f"P(01) = {p01}")

    if abs(p00 - 0.5) < 1e-5 and abs(p11 - 0.5) < 1e-5:
        logger.info("Simple test PASSED.")
    else:
        logger.warning("Simple test FAILED.")


def main():
    # 0. Simple Test
    test_sebd_simple()

    # 1. Verification on Small Grid
    logger.info("Running verification on 4x4 grid, depth 4...")
    rows, cols = 4, 4
    depth = 4
    c = generate_random_2d_circuit(rows, cols, depth)

    # Pick a random bitstring
    target_bits = "".join([str(np.random.randint(0, 2)) for _ in range(rows * cols)])
    # Or just all zeros for simplicity
    target_bits = "0" * (rows * cols)

    start = time.time()
    exact_prob = exact_probability(c, target_bits)
    end = time.time()
    logger.info(f"Exact Probability: {exact_prob:.6e} (Time: {end - start:.4f}s)")

    start = time.time()
    # SEBD with sufficient bond dimension should be exact
    sebd_prob = sebd_probability(c, rows, cols, target_bits, max_bond=256)
    end = time.time()
    logger.info(f"SEBD Probability:  {sebd_prob:.6e} (Time: {end - start:.4f}s)")

    diff = abs(exact_prob - sebd_prob)
    logger.info(f"Difference: {diff:.2e}")

    if diff < 1e-5:
        logger.info("Verification PASSED.")
    else:
        logger.warning("Verification FAILED (or exact prob is too small/SEBD approx).")

    # 2. Large Scale Simulation
    logger.info("\nRunning Large Scale Simulation (10x10, depth 4)...")
    # 100 qubits - exact is impossible.
    rows_l, cols_l = 10, 10
    depth_l = 4
    c_large = generate_random_2d_circuit(rows_l, cols_l, depth_l)
    target_bits_l = "0" * (rows_l * cols_l)

    start = time.time()
    # Max bond 32 or 64 to keep it fast
    prob = sebd_probability(c_large, rows_l, cols_l, target_bits_l, max_bond=32)
    end = time.time()
    logger.info(f"SEBD Probability (10x10): {prob:.6e}")
    logger.info(f"Time taken: {end - start:.4f}s")

    # Save results
    with open(
        "examples/reproduce_papers/2020_efficient_classical_simulation_random_shallow_2d/outputs/results.log",
        "w",
    ) as f:
        f.write(
            f"Verification 4x4:\nExact: {exact_prob}\nSEBD: {sebd_prob}\nDiff: {diff}\n\n"
        )
        f.write(f"Large Scale 10x10:\nProb: {prob}\nTime: {end - start:.4f}s\n")

    # Simple plot of accuracy vs bond dimension for small case
    bonds = [2, 4, 8, 16, 32, 64]
    errors = []
    logger.info("Generating accuracy plot...")
    for b in bonds:
        p = sebd_probability(c, rows, cols, target_bits, max_bond=b)
        errors.append(abs(p - exact_prob))

    plt.figure()
    plt.plot(bonds, errors, "o-")
    plt.yscale("log")
    plt.xlabel("Bond Dimension")
    plt.ylabel("Abs Error")
    plt.title("SEBD Accuracy vs Bond Dimension (4x4)")
    plt.grid(True)
    plt.savefig(
        "examples/reproduce_papers/2020_efficient_classical_simulation_random_shallow_2d/outputs/accuracy.png"
    )
    logger.info("Plot saved.")


if __name__ == "__main__":
    main()
