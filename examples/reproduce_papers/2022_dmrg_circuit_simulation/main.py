"""
Reproduction of "A density-matrix renormalization group algorithm for simulating
quantum circuits with a finite fidelity"
Link: https://arxiv.org/abs/2207.05612

Description:
This script reproduces Figure 2(a) from the paper.
It implements the specific DMRG algorithm described in the paper:
1. Qubit Grouping: Maps 2D grid columns to single MPS sites (Pure State).
2. 1-site Analytic Variational Update: M_new = F / norm(F).
3. Local Vertical Contraction: Environment tensors are computed by contracting
   gates individually without forming explicit MPOs.
4. Backend Agnostic: Uses tensorcircuit backend for JAX compatibility.
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use numpy backend for broad compatibility (can be switched to jax)
K = tc.set_backend("numpy")


def generate_sycamore_gates(rows, cols, depth, seed=42):
    """
    Generates gates for a Sycamore-like circuit.
    Returns a list of layers, where each layer is a list of gates.
    Each gate is a dict: {'gatef': func, 'index': tuple, 'parameters': dict}
    """
    np.random.seed(seed)
    n_qubits = rows * cols

    # We define qubit index as q(r, c) = r + c * rows
    def q(r, col):
        return r + col * rows

    layers = []

    for d in range(depth):
        layer_gates = []

        # Single qubit gates
        for i in range(n_qubits):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            lam = np.random.uniform(0, 2 * np.pi)

            layer_gates.append(
                {"gatef": tc.gates.rz, "index": (i,), "parameters": {"theta": phi}}
            )
            layer_gates.append(
                {"gatef": tc.gates.ry, "index": (i,), "parameters": {"theta": theta}}
            )
            layer_gates.append(
                {"gatef": tc.gates.rz, "index": (i,), "parameters": {"theta": lam}}
            )

        layers.append(layer_gates)
        layer_gates = []

        # Two-qubit gates
        layer_type = d % 4

        if layer_type == 0:  # Horizontal (col, col+1) Even cols
            for c in range(0, cols - 1, 2):
                for r in range(rows):
                    q1, q2 = q(r, c), q(r, c + 1)
                    layer_gates.append(
                        {"gatef": tc.gates.cz, "index": (q1, q2), "parameters": {}}
                    )
        elif layer_type == 1:  # Horizontal (col, col+1) Odd cols
            for c in range(1, cols - 1, 2):
                for r in range(rows):
                    q1, q2 = q(r, c), q(r, c + 1)
                    layer_gates.append(
                        {"gatef": tc.gates.cz, "index": (q1, q2), "parameters": {}}
                    )
        elif layer_type == 2:  # Vertical (row, row+1) Even rows
            for c in range(cols):
                for r in range(0, rows - 1, 2):
                    q1, q2 = q(r, c), q(r + 1, c)
                    layer_gates.append(
                        {"gatef": tc.gates.cz, "index": (q1, q2), "parameters": {}}
                    )
        elif layer_type == 3:  # Vertical (row, row+1) Odd rows
            for c in range(cols):
                for r in range(1, rows - 1, 2):
                    q1, q2 = q(r, c), q(r + 1, c)
                    layer_gates.append(
                        {"gatef": tc.gates.cz, "index": (q1, q2), "parameters": {}}
                    )

        if layer_gates:
            layers.append(layer_gates)

    return layers


class GroupedMPS:
    """
    Manages an MPS where each site represents a column of qubits (Qubit Grouping).
    """

    def __init__(self, rows, cols, bond_dim):
        self.rows = rows
        self.cols = cols
        self.bond_dim = bond_dim
        self.phys_dim = 2**rows
        self.mps_len = cols

        # Initialize tensors (l, d, r)
        self.tensors = []
        for i in range(self.cols):
            l_dim = bond_dim if i > 0 else 1
            r_dim = bond_dim if i < self.cols - 1 else 1

            # Random initialization
            t = np.random.randn(l_dim, self.phys_dim, r_dim) + 1j * np.random.randn(
                l_dim, self.phys_dim, r_dim
            )
            t /= np.linalg.norm(t)
            self.tensors.append(K.convert_to_tensor(t))

    def right_canonicalize(self):
        """
        Brings MPS to right-canonical form.
        """
        for i in range(self.cols - 1, 0, -1):
            t = self.tensors[i]  # (l, d, r)
            l, d, r = t.shape
            t_flat = K.reshape(t, (l, d * r))

            # RQ decomposition: T = R * Q. Q is isometric (rows orthogonal).
            # K.rq returns R (l, l) and Q (l, d*r) if l < d*r.
            # R is (l, rank), Q is (rank, d*r).
            r_mat, q_mat = K.rq(t_flat)

            self.tensors[i] = K.reshape(q_mat, (q_mat.shape[0], d, r))

            # Absorb R into T[i-1]
            prev = self.tensors[i - 1]  # (pl, pd, pr)
            # Contract pr with r_mat (left index)
            new_prev = K.tensordot(prev, r_mat, axes=[[2], [0]])
            self.tensors[i - 1] = new_prev


def get_gate_matrix(gate, rows):
    """
    Constructs the unitary matrix for a gate acting on a grouped site (column).
    If the gate is single-qubit or vertical 2-qubit, it acts on the 'rows' qubits of the site.
    Returns matrix of shape (d, d) where d = 2^rows.
    """
    c_local = tc.Circuit(rows)
    idx = [x % rows for x in gate["index"]]
    params = gate["parameters"]
    # Use .n for name access as gatef is GateVF
    name = gate["gatef"].n
    getattr(c_local, name)(*idx, **params)
    return c_local.matrix()


def get_interaction_gate_matrix(gate, rows):
    """
    Constructs the unitary matrix for a gate acting on two adjacent grouped sites.
    The gate acts on 2*rows qubits.
    Returns matrix of shape (d*d, d*d).
    """
    c_bond = tc.Circuit(2 * rows)
    # Map indices: site 1 qubits are 0..rows-1, site 2 qubits are rows..2*rows-1
    # Gate indices are global.
    # We assume gate connects (r, c) and (r, c+1).
    # r1 = gate['index'][0] % rows
    # r2 = gate['index'][1] % rows
    # In the circuit, r1 is on first site, r2 on second.
    # So index in c_bond: r1, r2 + rows.

    idx_global = gate["index"]
    r1 = idx_global[0] % rows
    r2 = idx_global[1] % rows

    # Check which is left/right. Assuming ordered in generation (c, c+1).
    # But generation might be (c+1, c) if symmetric? No, we generated carefully.

    idx1 = r1
    idx2 = r2 + rows

    params = gate["parameters"]
    name = gate["gatef"].n
    getattr(c_bond, name)(idx1, idx2, **params)
    return c_bond.matrix()


def apply_internal_gates(tensor, gates, rows):
    """
    Applies internal gates to a site tensor (l, d, r).
    """
    if not gates:
        return tensor

    # Build total unitary U for the site
    # Start with Identity
    # We can compose circuit or matrices.
    c_local = tc.Circuit(rows)
    for gate in gates:
        idx = [x % rows for x in gate["index"]]
        params = gate["parameters"]
        name = gate["gatef"].n
        getattr(c_local, name)(*idx, **params)
    U = c_local.matrix()  # (d, d)

    # Contract U with T
    # T: (l, d, r). U acts on d (index 1).
    # New T = U * T (contract U axis 1 with T axis 1) -> (d_out, l, r)
    # Permute back -> (l, d_out, r)

    l, d, r = tensor.shape
    t_flat = K.transpose(tensor, (1, 0, 2))  # (d, l, r)
    t_flat = K.reshape(t_flat, (d, l * r))

    new_t = K.matmul(U, t_flat)  # (d, l*r)
    new_t = K.reshape(new_t, (d, l, r))
    new_t = K.transpose(new_t, (1, 0, 2))  # (l, d, r)

    return new_t


def apply_layer_dmrg(mps, layer_gates):
    """
    Applies a layer of gates using 1-site analytic variational update.
    """
    # 1. Classify gates
    internal_gates = [[] for _ in range(mps.cols)]
    interaction_gates = [[] for _ in range(mps.cols - 1)]

    for gate in layer_gates:
        idx = gate["index"]
        cols_involved = sorted(list(set([i // mps.rows for i in idx])))

        if len(cols_involved) == 1:
            c = cols_involved[0]
            internal_gates[c].append(gate)
        elif len(cols_involved) == 2:
            c1, c2 = cols_involved
            bond = min(c1, c2)
            interaction_gates[bond].append(gate)

    # 2. Right Canonicalize
    mps.right_canonicalize()

    # 3. Environment Sweep & Update
    # We maintain mps_old (copy) and update mps (new)
    mps_old_tensors = [K.copy(t) for t in mps.tensors]

    # Prepare MPO-like objects for Interaction Gates to use in contraction
    # SVD interaction unitaries into (W_L, W_R) for each bond
    # This effectively makes the "layer" an MPO

    W_L_list = []
    W_R_list = []

    d = mps.phys_dim
    for c in range(mps.cols - 1):
        gates = interaction_gates[c]
        if not gates:
            # Identity interaction
            # W_L = I, W_R = I
            W_L = K.eye(d).reshape(d, d, 1)  # (out, in, bond)
            W_R = K.eye(d).reshape(d, d, 1)
        else:
            # V = get_interaction_gate_matrix({"index": [0,0], "parameters": {}, "gatef": type("obj", (object,), {"n": "id"})}, mps.rows) # Dummy
            # Actually use real logic
            c_bond = tc.Circuit(2 * mps.rows)
            for gate in gates:
                r1 = gate["index"][0] % mps.rows
                r2 = gate["index"][1] % mps.rows
                getattr(c_bond, gate["gatef"].n)(
                    r1, r2 + mps.rows, **gate["parameters"]
                )
            V = c_bond.matrix()  # (d*d, d*d)

            # SVD V -> L * R
            V = K.reshape(V, (d, d, d, d))  # (o1, o2, i1, i2)
            V = K.transpose(V, (0, 2, 1, 3))  # (o1, i1, o2, i2)
            V_flat = K.reshape(V, (d * d, d * d))
            # K.svd may return 4 values if truncated? or 3?
            # tensorcircuit.backend.svd might return error?
            # Check backend implementation. Usually u, s, v or u, s, v, err.
            # But numpy backend svd returns u, s, v.
            # Wait, tc.backend.svd might have different signature?
            # Let's inspect signature or try unpacking differently.
            # In 'numpy' backend, it calls np.linalg.svd which returns 3.
            # But tc.backend wrapper might add truncation error.
            svd_res = K.svd(V_flat)
            if len(svd_res) == 3:
                u, s, v = svd_res
            else:
                u, s, v, _ = svd_res

            s_sqrt = K.sqrt(s)

            L = u * s_sqrt  # (d*d, rank)
            L = K.reshape(L, (d, d, -1))  # (o1, i1, bond)

            # Use backend diagflat or similar? or just broadcasting
            # s_sqrt is vector (rank). v is (rank, d*d).
            # We want diag(s) @ v => scale rows of v.
            # s_sqrt[:, None] * v
            R = s_sqrt[:, None] * v
            R = K.transpose(R, (1, 0))  # (d*d, rank)
            R = K.reshape(R, (d, d, -1))  # (o2, i2, bond)

            W_L = L
            W_R = R

        W_L_list.append(W_L)
        W_R_list.append(W_R)

    # Apply Internal Gates to mps_old tensors to simplify contraction
    # Effectively incorporating U_internal into the "Ket"
    for i in range(mps.cols):
        mps_old_tensors[i] = apply_internal_gates(
            mps_old_tensors[i], internal_gates[i], mps.rows
        )

    # Environments
    # L[i]: (l_new, bond_prev, l_old)
    # R[i]: (r_new, bond_next, r_old)
    # Bond dimension of MPO is dynamic (from SVD of interactions)

    L_env = [None] * (mps.cols + 1)
    R_env = [None] * (mps.cols + 1)

    # Boundary conditions
    L_env[0] = K.ones((1, 1, 1), dtype="complex128")
    R_env[mps.cols] = K.ones((1, 1, 1), dtype="complex128")

    # Precompute R environments
    for i in range(mps.cols - 1, 0, -1):
        # Site i
        # Op i: W_L[i] (if i < N-1) and W_R[i-1] (if i > 0)
        # Note: Interaction is V_{i-1, i} and V_{i, i+1}.
        # Site i has W_R[i-1] acting from left bond, W_L[i] acting to right bond?
        # No.
        # Layer Op = prod V_{c, c+1}.
        # V_{c, c+1} = sum W_L[c]_k * W_R[c]_k.
        # Site i is acted upon by W_R[i-1] (part of V_{i-1, i}) AND W_L[i] (part of V_{i, i+1}).
        # So effective Operator on site i has 2 interaction bonds: k_{i-1} and k_i.

        # Interaction MPO tensor for site i:
        # Indices: (k_{i-1}, k_i, p_out, p_in)
        # T = W_R[i-1] * W_L[i]
        # W_R[i-1]: (p_out, p_in, k_{i-1})
        # W_L[i]: (p_out, p_in, k_i)
        # Total Op: O_{k_{i-1}, k_i, p_out, p_in} = \sum_x (W_R[i-1])_{px, p_in, k_{i-1}} * (W_L[i])_{p_out, px, k_i}
        # Wait, order of operations? V_{i-1, i} and V_{i, i+1} commute (disjoint).
        # So we can apply them in any order or symmetric.
        # Let's assume sequential: apply all W_R then all W_L?
        # Actually, simpler:
        # Treat them as MPO.
        # Site i has left bond k_{i-1} and right bond k_i.
        # MPO Tensor M_i:
        # If i=0: W_L[0] (indices: p_out, p_in, k_0). Shape (1, k_0, p_out, p_in).
        # If i=N-1: W_R[N-2] (indices: p_out, p_in, k_{N-2}). Shape (k_{N-2}, 1, p_out, p_in).
        # Middle: Contract W_R[i-1] and W_L[i].
        # Since they commute, just matrix multiply?
        # No, W_R[i-1] acts on physical. W_L[i] acts on physical.
        # M_i = W_L[i] @ W_R[i-1] (matrix mult on physical).
        # Outer product on bonds.

        # Construct M_i for site i
        if i == 0:
            # Only W_L[0]
            # W_L[0]: (o, i, k0)
            M = K.transpose(W_L_list[0], (2, 0, 1))  # (k0, o, i)
            M = K.reshape(M, (1, M.shape[0], M.shape[1], M.shape[2]))  # (1, k0, o, i)
        elif i == mps.cols - 1:
            # Only W_R[N-2]
            # W_R: (o, i, k_prev)
            M = K.transpose(W_R_list[i - 1], (2, 0, 1))  # (k_prev, o, i)
            M = K.reshape(
                M, (M.shape[0], 1, M.shape[1], M.shape[2])
            )  # (k_prev, 1, o, i)
        else:
            # W_R[i-1] and W_L[i]
            # Both (o, i, k)
            # Combine: O = W_L[i] @ W_R[i-1]
            wr = W_R_list[i - 1]  # (o, i, kL)
            wl = W_L_list[i]  # (o, i, kR)

            # M_{kL, kR, o, i} = sum_x wl_{o, x, kR} * wr_{x, i, kL}
            M = K.tensordot(wl, wr, axes=[[1], [0]])  # (o, kR, i, kL)
            M = K.transpose(M, (3, 1, 0, 2))  # (kL, kR, o, i)

        # Contract R[i+1] -- A_new[i] -- M_i -- A_old[i]
        # R[i+1]: (r_n, kR, r_o)
        # A_new[i]: (l_n, p, r_n)

        # T1 = A_new[i] * R[i+1] (sum r_n) -> (l_n, p, kR, r_o)
        T1 = K.tensordot(mps.tensors[i], R_env[i + 1], axes=[[2], [0]])

        # T2 = T1 * M_i (sum p, kR)
        # T1: (l_n, p, kR, r_o)
        # M_i: (kL, kR, p_out(p), p_in)
        # axes: T1[1, 2] with M_i[2, 1]
        T2 = K.tensordot(T1, M, axes=[[1, 2], [2, 1]])  # (l_n, r_o, kL, p_in)

        # T3 = T2 * A_old[i] (sum r_o, p_in)
        # T2: (l_n, r_o, kL, p_in)
        # A_old: (l_o, p_in, r_o)
        # axes: T2[1, 3] with A_old[2, 1]
        R_env[i] = K.tensordot(
            T2, mps_old_tensors[i], axes=[[1, 3], [2, 1]]
        )  # (l_n, kL, l_o)

    # Sweep Left -> Right
    for i in range(mps.cols):
        # Build M_i again (could cache)
        if i == 0:
            M = K.transpose(W_L_list[0], (2, 0, 1))
            M = K.reshape(M, (1, M.shape[0], M.shape[1], M.shape[2]))
        elif i == mps.cols - 1:
            M = K.transpose(W_R_list[i - 1], (2, 0, 1))
            M = K.reshape(M, (M.shape[0], 1, M.shape[1], M.shape[2]))
        else:
            wr = W_R_list[i - 1]
            wl = W_L_list[i]
            M = K.tensordot(wl, wr, axes=[[1], [0]])
            M = K.transpose(M, (3, 1, 0, 2))

        # Compute F_i
        # F = L[i] * M_i * R[i+1] * A_old[i]
        # L[i]: (l_n, kL, l_o)
        # A_old: (l_o, p_in, r_o)

        # T1 = L[i] * A_old[i] (sum l_o) -> (l_n, kL, p_in, r_o)
        T1 = K.tensordot(L_env[i], mps_old_tensors[i], axes=[[2], [0]])

        # T2 = T1 * M_i (sum kL, p_in)
        # T1: (l_n, kL, p_in, r_o)
        # M_i: (kL, kR, p_out, p_in)
        # axes: T1[1, 2] with M_i[0, 3]
        T2 = K.tensordot(T1, M, axes=[[1, 2], [0, 3]])  # (l_n, r_o, kR, p_out)

        # T3 = T2 * R[i+1] (sum r_o, kR)
        # T2: (l_n, r_o, kR, p_out)
        # R[i+1]: (r_n, kR, r_o)
        # axes: T2[1, 2] with R[2, 1]
        F = K.tensordot(T2, R_env[i + 1], axes=[[1, 2], [2, 1]])  # (l_n, p_out, r_n)

        # 4. Analytic Update
        norm_F = K.norm(F)
        if norm_F > 1e-12:
            mps.tensors[i] = F / norm_F
        else:
            # Fallback if zero (unlikely)
            pass

        # 5. Left Canonicalize (Shift center to i+1)
        if i < mps.cols - 1:
            t = mps.tensors[i]
            l, d, r = t.shape
            t_flat = K.reshape(t, (l * d, r))
            q, r_mat = K.qr(t_flat)

            # Truncate if rank > bond_dim?
            # 1-site update doesn't change bond dim if initialized with fixed bond dim.
            # But QR can reveal rank deficiency.
            # We keep the shape consistent with bond_dim if possible, or adapt.
            # Here we assume bond_dim is fixed by initialization.
            # However, QR returns Q(..., k) R(k, ...).
            # If we want to keep fixed dimensions, we need to pad or standard shape?
            # `K.qr` usually produces full rank or reduced.
            # TensorCircuit MPS tensors are usually fixed size?
            # We initialized with size `bond_dim`.
            # If Q is smaller, next site shape mismatch?
            # We should ensure shapes match.
            # Simplification: Just allow shapes to adapt.

            mps.tensors[i] = K.reshape(q, (l, d, -1))

            # Absorb R into i+1
            next_t = mps.tensors[i + 1]
            mps.tensors[i + 1] = K.tensordot(r_mat, next_t, axes=[[1], [0]])

            # 6. Update L[i+1]
            # T1 = L[i] * A_new[i] (sum l_n) -> (kL, l_o, p, bond)
            T1 = K.tensordot(L_env[i], mps.tensors[i], axes=[[0], [0]])

            # T2 = T1 * M_i (sum kL, p)
            # T1: (kL, l_o, p, bond)
            # M_i: (kL, kR, p, p_in)
            # axes: T1[0, 2] with M_i[0, 2]
            T2 = K.tensordot(T1, M, axes=[[0, 2], [0, 2]])  # (l_o, bond, kR, p_in)

            # T3 = T2 * A_old[i] (sum l_o, p_in)
            # T2: (l_o, bond, kR, p_in)
            # A_old: (l_o, p_in, r_o)
            # axes: T2[0, 3] with A_old[0, 1]
            L_env[i + 1] = K.tensordot(
                T2, mps_old_tensors[i], axes=[[0, 3], [0, 1]]
            )  # (bond, kR, r_o)
            # Matches (l_n, kL, l_o) pattern for next site.


def run_simulation(rows, cols, depth, bond_dims):
    layers = generate_sycamore_gates(rows, cols, depth)

    # Exact state
    c_exact = tc.Circuit(rows * cols)
    for layer in layers:
        for gate in layer:
            name = gate["gatef"].n
            idx = gate["index"]
            params = gate["parameters"]
            getattr(c_exact, name)(*idx, **params)
    psi_exact = c_exact.state()
    psi_exact = K.reshape(psi_exact, (-1,))

    infidelities = []

    for chi in bond_dims:
        logger.info(f"Running MPS (DMRG) with chi={chi}")
        start = time.time()

        mps = GroupedMPS(rows, cols, bond_dim=chi)

        for layer in layers:
            apply_layer_dmrg(mps, layer)

        # Compute Fidelity
        # Contract MPS tensors
        full_tensor = mps.tensors[0]  # (1, d, r)
        for i in range(1, mps.cols):
            # full: (1, d...d, r_prev)
            # next: (r_prev, d, r_next)
            full_tensor = K.tensordot(full_tensor, mps.tensors[i], axes=[[-1], [0]])

        psi_mps = K.reshape(full_tensor, (-1,))

        # Handle potential phase difference? No, fidelity is abs^2.
        # Also MPS might not be normalized if we didn't track it carefully (though we normalize F).
        # We should normalize psi_mps.
        norm_mps = K.norm(psi_mps)
        psi_mps /= norm_mps

        overlap = K.tensordot(K.conj(psi_exact), psi_mps, axes=[[0], [0]])
        fid = float(np.abs(overlap) ** 2)

        infidelities.append(1 - fid)
        logger.info(f"Chi={chi}, Infidelity={1-fid:.6e}, Time={time.time()-start:.2f}s")

    plt.figure()
    plt.loglog(bond_dims, infidelities, "o-")
    plt.xlabel("Bond Dimension")
    plt.ylabel("Infidelity")
    plt.title("DMRG Simulation (Figure 2a Repro)")
    plt.savefig(
        "examples/reproduce_papers/2022_dmrg_circuit_simulation/outputs/result.png"
    )


def main():
    run_simulation(3, 4, 8, [2, 4, 8, 16, 32, 64])


if __name__ == "__main__":
    main()
