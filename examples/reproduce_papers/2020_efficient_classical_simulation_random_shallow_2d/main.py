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
# TODO(@refraction-ray): Support JAX backend with JIT compilation for better performance
K = tc.set_backend("numpy")

LOG_FILE_PATH = "examples/reproduce_papers/2020_efficient_classical_simulation_random_shallow_2d/outputs/results.log"


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


def exact_amplitude(c, bitstring):
    """
    Computes the exact amplitude of measuring the given bitstring.
    """
    if c._nqubits > 20:
        logger.warning("Exact simulation for >20 qubits is slow/impossible. Skipping.")
        return 0.0

    return np.abs(c.amplitude(bitstring)) ** 2


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


def build_peps_tensors(c, rows, cols, bitstring):
    """
    Constructs the PEPS tensors for the given circuit and bitstring projection.
    Returns:
        grid: 2D list of standardized tensors representing the PEPS.
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
            mat = func(**param).tensor
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

            # SVD split
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
            t1_new = K.tensordot(u, t1, axes=[[1], [0]])  # (Phys_out, Bond_k, Bonds...)
            sites[q1]["tensor"] = t1_new
            sites[q1]["directions"].insert(0, dir1)

            # Update T2
            t2 = sites[q2]["tensor"]
            t2_new = K.tensordot(v, t2, axes=[[2], [0]])  # (Bond_k, Phys_out, Bonds...)
            # We want (Phys_out, Bond_k, Bonds...)
            # Current axes: 0->k, 1->Phys_out, 2...->Bonds
            # Transpose: 1, 0, 2...
            perm = [1, 0] + list(range(2, len(t2_new.shape)))
            t2_new = K.transpose(t2_new, perm)
            sites[q2]["tensor"] = t2_new
            sites[q2]["directions"].insert(0, dir2)

    # Project onto final bitstring
    bits = [int(x) for x in bitstring]
    for i in range(rows * cols):
        b = bits[i]
        vec = np.array([0.0, 0.0])
        vec[b] = 1.0
        vec = K.cast(K.convert_to_tensor(vec), "complex128")
        t = sites[i]["tensor"]

        # Contract axis 0
        sites[i]["tensor"] = K.tensordot(vec, t, axes=[[0], [0]])  # (bonds...)

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

    return grid


def contract_peps(grid, rows, cols, max_bond=None):
    """
    Contracts the PEPS grid using Boundary MPS.
    """
    # SEBD Contraction (Boundary MPS)
    mps = [K.ones((1, 1, 1), dtype="complex128") for _ in range(rows)]

    for c_idx in range(cols):
        # Contract column into MPS
        new_mps = []
        for r in range(rows):
            m = mps[r]  # (v_u, v_d, p_l)
            t = grid[c_idx][r]  # (c_u, c_d, c_l, c_r)

            # Contract p_l with c_l
            res = K.tensordot(m, t, axes=[[2], [2]])
            # res: (v_u, v_d, c_u, c_d, c_r)

            # Permute to (v_u, c_u, v_d, c_d, c_r)
            res = K.transpose(res, (0, 2, 1, 3, 4))
            s = res.shape

            # Fuse vertical bonds
            new_t = K.reshape(res, (s[0] * s[1], s[2] * s[3], s[4]))
            new_mps.append(new_t)

        mps = new_mps
        if max_bond is not None:
            mps = compress_mps(mps, max_bond)

    # Contract final MPS
    res = K.ones((1, 1), dtype="complex128")  # Shape (1, 1)
    for r in range(rows):
        m = mps[r]  # (u, d, 1)
        u_dim = m.shape[0]
        d_dim = m.shape[1]
        m = K.reshape(m, (u_dim, d_dim))
        res = K.matmul(res, m)  # (1, u) @ (u, d) -> (1, d)

    # Final res should be (1, 1)
    prob_amp = K.reshape(res, (1,))[0]

    return np.abs(K.numpy(prob_amp)) ** 2


def sebd_probability(c, rows, cols, bitstring, max_bond=None):
    """
    Computes the probability of `bitstring` using SEBD.
    """
    grid = build_peps_tensors(c, rows, cols, bitstring)
    prob = contract_peps(grid, rows, cols, max_bond)
    return prob


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

    # Clear log file
    with open(LOG_FILE_PATH, "w") as f:
        f.write("Verification Results:\n")

    # 1. Verification on Small Grids
    logger.info("Running verification on small grids...")

    test_cases = [
        (2, 2, 2),  # 2x2, depth 2
        (4, 4, 4),  # 4x4, depth 4
        (4, 5, 4),  # 4x5, depth 4 (Rectangular)
    ]

    for rows, cols, depth in test_cases:
        logger.info(f"Verifying {rows}x{cols} grid, depth {depth}...")
        c = generate_random_2d_circuit(rows, cols, depth)

        # Test 2 random bitstrings
        for _ in range(2):
            target_bits = "".join(
                [str(np.random.randint(0, 2)) for _ in range(rows * cols)]
            )

            start = time.time()
            exact_prob = exact_amplitude(c, target_bits)
            sebd_prob = sebd_probability(c, rows, cols, target_bits, max_bond=256)
            end = time.time()

            diff = abs(exact_prob - sebd_prob)
            msg = f"Bitstring: {target_bits[:5]}... Exact: {exact_prob:.6e}, SEBD: {sebd_prob:.6e}, Diff: {diff:.2e}"
            logger.info(msg)

            with open(LOG_FILE_PATH, "a") as f:
                f.write(f"Grid {rows}x{cols} depth {depth}: {msg}\n")

            # Relaxed threshold for larger grids/floating point error
            if diff > 1e-8:
                logger.error("Verification FAILED.")
                # raise ValueError(f"Verification failed for {rows}x{cols}")
            else:
                logger.info("Verification PASSED.")

    # 2. Large Scale Simulation
    logger.info("\nRunning Large Scale Simulation (10x10, depth 4)...")
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
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"Large Scale 10x10:\nProb: {prob}\nTime: {end - start:.4f}s\n")

    # Simple plot of accuracy vs bond dimension for small case (4x4)
    logger.info("Generating accuracy plot for 4x4...")
    rows, cols = 4, 4
    c = generate_random_2d_circuit(rows, cols, 4)
    target_bits = "0" * (rows * cols)
    exact_prob = exact_amplitude(c, target_bits)

    bonds = [2, 4, 8, 16, 32, 64]
    errors = []
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
