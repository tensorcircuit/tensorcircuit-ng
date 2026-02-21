"""
Reproduction of "A density-matrix renormalization group algorithm for simulating
quantum circuits with a finite fidelity"
Link: https://arxiv.org/abs/2207.05612

Method:
- Step-wise DMRG with JAX backend.
- Qubit Grouping: Columns of the grid are mapped to single MPS sites.
- Blocked Layer Application: Circuit is applied in blocks of depth K.
- Variational Optimization: Find |psi_new> ~ U_block |psi_old>.
- Canonical Form: Explicit QR/LQ decomposition during sweeps.
"""

import logging
import os
import numpy as np
import tensorcircuit as tc

# Using tc.backend instead of direct jax usage where possible
K = tc.set_backend("jax")
import jax
import jax.numpy as jnp # Still needed for some specific logic, but aliased to K where appropriate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("tensorcircuit").setLevel(logging.ERROR)

class GateManager:
    def __init__(self, layers, rows, cols):
        self.layers = layers
        self.rows = rows
        self.cols = cols

    def get_internal_gates(self, col_idx, layer_idx):
        layer = self.layers[layer_idx]
        return [g for g in layer if all((x // self.rows) == col_idx for x in g["index"])]

    def get_bond_gate(self, bond_idx, layer_idx):
        if bond_idx < 0 or bond_idx >= self.cols - 1:
            return None
        layer = self.layers[layer_idx]
        for g in layer:
            cols = [x // self.rows for x in g["index"]]
            if min(cols) == bond_idx and max(cols) == bond_idx + 1:
                return g
        return None

def get_gate_matrix(gate):
    num_qubits = len(gate["index"])
    c = tc.Circuit(num_qubits)
    getattr(c, gate["gatef"].n)(*range(num_qubits), **gate["parameters"])
    return c.matrix()

def apply_gate_to_tensor_indices(T, gate, indices, rows, dagger=False):
    num_q = len(indices)
    U = get_gate_matrix(gate)
    U = jnp.reshape(U, (2,) * (2 * num_q))

    if dagger:
        # For U^dagger, we want to apply U* and contract on the OUTPUT indices of U.
        # But mathematically <psi| U^dagger = (U |psi>)^dagger.
        # Here we are applying gate to T.
        # If T represents a ket state: T' = U T. Contract U(in) with T(phys).
        # If dagger: T' = U^dagger T. U^dagger = (U^T)*.
        # So we use conj(U). And we contract U(out) with T(phys).
        # U indices: (out_0..out_n, in_0..in_n)
        # out indices are 0..num_q-1. in indices are num_q..2*num_q-1.

        # Original code used: U_op = conj(transpose(U)). Then contracted T with U_op's first half (which was original 'in').
        # This effectively applied U^dagger.

        # Reviewer comment:
        # "In your implementation, U_axes is hardcoded to list(range(num_q)). This contracts the state with the output indices, effectively applying U^T instead of U."
        # "When dagger=False, you must contract on the in indices. When dagger=True, you should apply U^dagger = U^*, contracting on the out indices."

        # Let's verify `tensordot(T, U_op, axes=[T_axes, U_axes])`.
        # T_axes are the physical indices of T.

        # Case dagger=False:
        # U_op = U.
        # We want to contract T (state) with INPUT of U.
        # INPUT of U is indices `num_q` to `2*num_q - 1`.
        # Reviewer says my code used `list(range(num_q))`, which are OUTPUT indices.
        # Correct. So I was applying U^T (since T_i U_ji -> T_j, effectively contracting output).
        # Wait. U_{out, in}. Contraction: \sum_{out} T_{out} U_{out, in} = res_{in}. This is vector-matrix multiplication from left: v U.
        # If T is ket (column vector), we want U T. \sum_{in} U_{out, in} T_{in} = res_{out}.
        # So we should contract U's INPUT with T.
        # U's input is `range(num_q, 2*num_q)`.

        U_op = jnp.conj(U) # U^dagger
        U_axes = list(range(num_q)) # Contract on 'out' (which becomes 'in' for U^dagger acting from left?)
        # Wait. U_{ij}. U^dagger_{ji} = U^*_{ij}.
        # (U^dagger v)_k = \sum_l U^dagger_{kl} v_l = \sum_l U^*_{lk} v_l.
        # We need to contract v_l with l index of U^*.
        # l is the INPUT of U (second index).
        # But U^* has same shape as U.
        # So we contract with INPUT of U^*.

        # Let's use the reviewer's snippet directly to be safe.
        # "When dagger=True, you should apply U^\dagger = U^*, contracting on the out indices."
        # Wait. If I use U^*, and contract on 'out' indices (0..num_q).
        # \sum_{out} U^*_{out, in} v_{out}.
        # This matches \sum_l U^*_{lk} v_l IF 'out' corresponds to l (the row index of U, which is output).
        # Yes. U_{row, col}. row=output. col=input.
        # So contracting on output is correct for U^dagger?

        # Let's trace:
        # U_{out, in}.
        # Want (U^\dagger v)_{in'} = \sum_{in} (U^\dagger)_{in', in} v_{in}.
        # (U^\dagger)_{in', in} = (U^T)_{in', in}^* = U_{in, in'}^*.
        # So \sum_{in} U_{in, in'}^* v_{in}.
        # We contract v (T) with the FIRST index of U (out).
        # And we get the SECOND index (in) as the new physical index.

        # Reviewer snippet for dagger=True:
        # U_op = jnp.conj(U)
        # U_axes = list(range(num_q)) # Contract with 'out'
        # This matches my derivation.

        U_op = jnp.conj(U)
        U_axes = list(range(num_q))
    else:
        # Case dagger=False.
        # Want U v.
        # (U v)_{out} = \sum_{in} U_{out, in} v_{in}.
        # Contract v (T) with SECOND index of U (in).
        # Result is FIRST index (out).

        # Reviewer snippet for dagger=False:
        # U_op = U
        # U_axes = list(range(num_q, 2 * num_q)) # Contract with 'in'

        U_op = U
        U_axes = list(range(num_q, 2 * num_q))

    T_axes = list(indices)
    T = jnp.tensordot(T, U_op, axes=[T_axes, U_axes])

    # After tensordot, the new axes (from U) are appended at the end.
    # If dagger=False: remaining axes are 0..num_q (out).
    # If dagger=True: remaining axes are num_q..2*num_q (in).
    # We need to move them back to T_axes positions.

    sources = list(range(len(T.shape) - num_q, len(T.shape)))
    T = jnp.moveaxis(T, sources, T_axes)
    return T

class EnvManager:
    def __init__(self, mps_old, mps_new, layer_block, gate_manager, rows, cols):
        self.mps_old = mps_old
        self.mps_new = mps_new
        self.layers = layer_block
        self.block_gm = GateManager(layer_block, rows, cols)
        self.rows = rows
        self.cols = cols
        self.L = {}
        self.R = {}
        self.num_layers = len(layer_block)
        self.recompute_all_R()

    def recompute_all_R(self):
        dtype = self.mps_old[0].dtype
        # shape: (b_old, b_new) + (g0...gK-1)
        shape = (1, 1) + (1,) * (2 * self.num_layers)
        self.R[self.cols - 1] = jnp.ones(shape, dtype=dtype)
        for i in range(self.cols - 2, -1, -1):
            self.R[i] = self.update_R(self.R[i + 1], i + 1)

    def get_L(self, i):
        if i == 0:
            shape = (1, 1) + (1,) * (2 * self.num_layers)
            return jnp.ones(shape, dtype=self.mps_old[0].dtype)
        if i not in self.L:
            self.L[i] = self.update_L(self.get_L(i - 1), i - 1)
        return self.L[i]

    def get_R(self, i):
        if i == self.cols - 1:
             dtype = self.mps_old[0].dtype
             shape = (1, 1) + (1,) * (2 * self.num_layers)
             return jnp.ones(shape, dtype=dtype)
        if i not in self.R:
             self.R[i] = self.update_R(self.get_R(i + 1), i + 1)
        return self.R[i]

    def update_L(self, L_prev, site_idx):
        A_new = self.mps_new[site_idx]
        A_old = self.mps_old[site_idx]

        # Contract L, A_new*, A_old
        # L_prev: (b_new, b_old, g...)
        # A_new: (b_l, p, b_r)
        # Contract b_new (0) with b_l (0)
        T = jnp.tensordot(L_prev, jnp.conj(A_new), axes=[[0], [0]])
        # T: b_old, g..., p_new, b_new_out

        # Contract b_old (0) with A_old (b_l, 0)
        T = jnp.tensordot(T, A_old, axes=[[0], [0]])
        # T: g..., p_new, b_new_out, p_old, b_old_out

        ndim = len(T.shape)
        # T: g... (0..num_g-1), p_new (num_g), b_new (num_g+1), p_old (num_g+2), b_old (num_g+3)
        num_g = ndim - 4
        b_new_idx = num_g + 1
        b_old_idx = num_g + 3
        p_new_idx = num_g
        p_old_idx = num_g + 2

        perm = [b_new_idx, b_old_idx, p_new_idx, p_old_idx] + list(range(num_g))
        T = jnp.transpose(T, perm)

        shape_phys = (2,) * self.rows
        shape_rest = T.shape[4:]
        T = jnp.reshape(T, (T.shape[0], T.shape[1]) + shape_phys + shape_phys + shape_rest)

        phys_new_start = 2
        phys_old_start = 2 + self.rows
        g_start = 2 + 2 * self.rows

        for k in range(self.num_layers):
            idx_ri = g_start
            idx_ro = g_start + 1

            # 1. Internal Gates
            gates_int = self.block_gm.get_internal_gates(site_idx, k)
            for g in gates_int: # Apply forwards
                indices = [x % self.rows for x in g['index']]
                T = apply_gate_to_tensor_indices(T, g, [phys_old_start + x for x in indices], self.rows, dagger=False)

            # 2. Left Gate (Consume g legs of bond i-1)
            gate_L = self.block_gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None

            if gate_L:
                r = gate_L["index"][1] % self.rows
                # Contract L[ri] with T[phys_old]
                T = jnp.trace(T, axis1=idx_ri, axis2=phys_old_start + r)
                # idx_ri, phys_old removed. idx_ro shifts to idx_ri.

                curr_ro = idx_ro - 2 # Shifted by -2 (idx_ri and phys_old)
                # Move curr_ro to phys_old (become new psi)
                dest = phys_old_start + r
                T = jnp.moveaxis(T, curr_ro, dest)

            else:
                 # Squeeze dummy gL
                 T = jnp.squeeze(T, axis=(idx_ri, idx_ro))

            # 3. Right Gate (Create g legs for bond i)
            gate_R = self.block_gm.get_bond_gate(site_idx, k) if site_idx < self.cols - 1 else None

            if gate_R:
                r = gate_R["index"][0] % self.rows
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)
                # U(lo, ro, li, ri) -> out_i, out_i+1, in_i, in_i+1

                # Contract T[phys_old] with U[li] (in_i)
                T = jnp.tensordot(T, U, axes=[[phys_old_start + r], [2]])

                # T: ..., lo, ro, ri.
                # lo (out_i) is new phys_old. Move to phys_old_start + r.
                T = jnp.moveaxis(T, -3, phys_old_start + r)

                # ro, ri are new legs (g) at END.
                # Swap last two axes to match optimize_site expectation (ri, ro)
                T = jnp.swapaxes(T, -1, -2)
            else:
                 # Expand dummy g legs at END
                 T = jnp.expand_dims(T, axis=-1)
                 T = jnp.expand_dims(T, axis=-1)

        # Final Trace
        for r in range(self.rows):
            T = jnp.trace(T, axis1=2, axis2=2+self.rows-r)

        return T

    def update_R(self, R_next, site_idx):
        A_new = self.mps_new[site_idx]
        A_old = self.mps_old[site_idx]

        T = jnp.tensordot(jnp.conj(A_new), R_next, axes=[[2], [1]])
        T = jnp.tensordot(T, A_old, axes=[[2], [2]])

        ndim = len(T.shape)
        num_g = ndim - 4
        perm = [ndim-2, 0, 1, ndim-1] + list(range(2, 2+num_g))
        T = jnp.transpose(T, perm)

        shape_phys = (2,) * self.rows
        shape_rest = T.shape[4:]
        T = jnp.reshape(T, (T.shape[0], T.shape[1]) + shape_phys + shape_phys + shape_rest)

        phys_new_start = 2
        phys_old_start = 2 + self.rows
        g_start = 2 + 2 * self.rows

        for k in range(self.num_layers): # Forward loop
            # Always consume from g_start because processed pairs are appended to the end
            idx_ri = g_start
            idx_ro = g_start + 1

            # 1. Internal Gates
            gates_int = self.block_gm.get_internal_gates(site_idx, k)
            for g in gates_int: # Apply forwards
                indices = [x % self.rows for x in g['index']]
                T = apply_gate_to_tensor_indices(T, g, [phys_old_start + x for x in indices], self.rows, dagger=False)

            # 2. Right Gate (Consume g legs of bond i+1)
            gate_R = self.block_gm.get_bond_gate(site_idx, k) if site_idx < self.cols - 1 else None
            if gate_R:
                r = gate_R["index"][0] % self.rows
                # Contract T[ri] with phys_old (psi)
                T = jnp.trace(T, axis1=idx_ri, axis2=phys_old_start + r)
                # idx_ri, phys_old removed. idx_ro shifts -2 (idx_ri > phys_old, idx_ro > idx_ri).

                curr_ro = idx_ro - 2
                # Move curr_ro to phys_old position
                dest = phys_old_start + r
                T = jnp.moveaxis(T, curr_ro, dest)
            else:
                T = jnp.squeeze(T, axis=(idx_ri, idx_ro))

            # 3. Left Gate (Create g legs for bond i)
            gate_L = self.block_gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None
            if gate_L:
                r = gate_L["index"][1] % self.rows
                U = get_gate_matrix(gate_L).reshape(2, 2, 2, 2)

                # Contract T[phys_old] with U[in_R] (ri, 3).
                T = jnp.tensordot(T, U, axes=[[phys_old_start+r], [3]])

                # T: ..., lo, ro, li.
                # ro (1) is out_R. It becomes new phys_old.
                # lo (0), li (2) are g legs (on i-1).

                # Move ro to phys_old.
                T = jnp.moveaxis(T, -2, phys_old_start+r)

                # Remaining: lo (idx -2), li (idx -1).
                # These are appended to the END.

                # Swap last two axes to match optimize_site expectation (li, lo)
                T = jnp.swapaxes(T, -1, -2)
            else:
                 # Expand dummy legs at END
                 T = jnp.expand_dims(T, axis=-1)
                 T = jnp.expand_dims(T, axis=-1)

        for r in range(self.rows):
            T = jnp.trace(T, axis1=2, axis2=2+self.rows-r)

        return T

    def optimize_site(self, i, L, R):
        A_old = self.mps_old[i]
        # A_old: (b_l, p, b_r)

        # 1. Contract L and A_old
        # L: (b_phi, b_psi, g_L...) -> we contract b_psi (1) with A_old (0)
        # A_old: (b_psi, p, b_psi_R)
        T = jnp.tensordot(L, A_old, axes=[[1], [0]])
        # T: (b_phi, g_L..., p, b_psi_R)

        # 2. Contract T and R
        # R: (b_psi, b_phi, g_R...) -> we contract b_psi (0) with T's last (b_psi_R)
        # T: (b_phi, g_L..., p, b_psi_R)
        T = jnp.tensordot(T, R, axes=[[len(T.shape)-1], [0]])
        # T: (b_phi, g_L..., p, b_phi_R, g_R...)

        # Reshape p to individual qubits
        shape_phys = (2,) * self.rows
        # Identify axes
        # b_L_n: 0
        # g_L: 1 .. 2K
        # p: 2K+1
        # b_R_n: 2K+2
        # g_R: 2K+3 .. 4K+2

        ndim = len(T.shape)
        num_gL = 2 * self.num_layers
        p_idx = 1 + num_gL

        # Move p to end to easily reshape and manipulate
        perm = [0, p_idx, p_idx+1] + list(range(1, p_idx)) + list(range(p_idx+2, ndim))
        # perm: [b_L_n, p, b_R_n, g_L..., g_R...]
        T = jnp.transpose(T, perm)

        # Reshape p
        shape_rest = T.shape[3:]
        T = jnp.reshape(T, (T.shape[0],) + shape_phys + (T.shape[2],) + shape_rest)

        # Indices:
        # 0: b_L_n
        # 1..rows: q0..qN
        # rows+1: b_R_n
        # rows+2..: g_L..., g_R...

        gL_start = self.rows + 2
        gR_start = gL_start + num_gL

        q_start = 1

        for k in range(self.num_layers):
            idx_ri_L = gL_start
            idx_ro_L = gL_start + 1

            gate_L = self.block_gm.get_bond_gate(i - 1, k) if i > 0 else None
            if gate_L:
                r = gate_L['index'][1] % self.rows
                # Contract q_r (q_start+r) with idx_ri_L
                T = jnp.trace(T, axis1=q_start+r, axis2=idx_ri_L)

                curr_ro = idx_ro_L - 2
                # Move idx_ro_L to q_start+r (become new q)
                T = jnp.moveaxis(T, curr_ro, q_start+r)

                # gL block reduced by 2.
                gR_start -= 2

            else:
                 # Squeeze dummy gL
                 T = jnp.squeeze(T, axis=(idx_ri_L, idx_ro_L))
                 gR_start -= 2

            gates_int = self.block_gm.get_internal_gates(i, k)
            for g in reversed(gates_int):
                indices = [x % self.rows for x in g['index']]
                # apply gate to q_indices
                T = apply_gate_to_tensor_indices(T, g, [q_start + x for x in indices], self.rows, dagger=False)

            idx_ri_R = gR_start
            idx_ro_R = gR_start + 1

            gate_R = self.block_gm.get_bond_gate(i, k) if i < self.cols - 1 else None
            if gate_R:
                r = gate_R['index'][0] % self.rows
                # Contract q_r with idx_ri_R
                T = jnp.trace(T, axis1=q_start+r, axis2=idx_ri_R)

                # idx_ro_R becomes q_r
                curr_ro = idx_ro_R - 2
                T = jnp.moveaxis(T, curr_ro, q_start+r)

                # gR block reduced by 2.
            else:
                 T = jnp.squeeze(T, axis=(idx_ri_R, idx_ro_R))

        # Final T: (b_L_n, q0..qN, b_R_n)
        # Reshape q to p
        T = jnp.reshape(T, (T.shape[0], -1, T.shape[-1]))
        return T

    def run_dmrg(self, sweeps=2):
        fidelities = []
        for sweep in range(sweeps):
            for i in range(self.cols):
                L = self.get_L(i)
                R = self.get_R(i)
                E = self.optimize_site(i, L, R)

                if i < self.cols - 1:
                    d = E.shape[1]
                    b_L = E.shape[0]
                    b_R = E.shape[2]
                    E_mat = jnp.reshape(E, (-1, b_R))
                    Q, R_mat = jnp.linalg.qr(E_mat)
                    new_dim = Q.shape[1]
                    self.mps_new[i] = jnp.reshape(Q, (b_L, -1, new_dim))
                    next_A = self.mps_new[i+1]
                    self.mps_new[i+1] = jnp.tensordot(R_mat, next_A, axes=[[1], [0]])
                else:
                    norm = jnp.linalg.norm(E)
                    self.mps_new[i] = E / norm

                if i < self.cols - 1:
                    if (i + 1) in self.L: del self.L[i + 1]

            for i in range(self.cols - 1, -1, -1):
                L = self.get_L(i)
                R = self.get_R(i)
                E = self.optimize_site(i, L, R)

                if i > 0:
                    b_L = E.shape[0]
                    E_mat = jnp.reshape(E, (b_L, -1))
                    Q_prime, R_prime = jnp.linalg.qr(E_mat.T)
                    L_mat = R_prime.T
                    Q_mat = Q_prime.T
                    self.mps_new[i] = jnp.reshape(Q_mat, (-1, E.shape[1], E.shape[2]))
                    prev_A = self.mps_new[i-1]
                    self.mps_new[i-1] = jnp.tensordot(prev_A, L_mat, axes=[[2], [0]])
                else:
                    norm = jnp.linalg.norm(E)
                    self.mps_new[i] = E / norm

                if i > 0:
                    if (i - 1) in self.R: del self.R[i - 1]

        # Use the squared norm of the final environment tensor (at site 0) as fidelity
        f = float(norm ** 2)
        fidelities.append(f)
        logger.info(f"Sweep {sweep}: Fidelity = {f}")

        return fidelities

def generate_sycamore_circuit(rows, cols, depth):
    layers = []
    l0 = []
    for i in range(rows * cols):
        l0.append({"gatef": tc.gates.h, "index": [i], "parameters": {}})
    layers.append(l0)
    import random
    random.seed(42)
    for _ in range(depth):
        l = []
        for i in range(rows * cols):
            l.append({"gatef": tc.gates.rz, "index": [i], "parameters": {"theta": random.random()}})
        for r in range(rows):
            for c_ in range(cols):
                idx = c_ * rows + r
                if c_ < cols - 1:
                    idx_next = (c_ + 1) * rows + r
                    l.append({"gatef": tc.gates.cz, "index": [idx, idx_next], "parameters": {}})
                if r < rows - 1:
                    idx_next = c_ * rows + (r + 1)
                    l.append({"gatef": tc.gates.cz, "index": [idx, idx_next], "parameters": {}})
        layers.append(l)
    return layers

def main():
    rows = 4
    cols = 4
    total_depth = 6
    block_size = 2
    layers = generate_sycamore_circuit(rows, cols, total_depth)
    gm = GateManager(layers, rows, cols)
    d = 2**rows
    bond_dim = 16
    mps_tensors = []
    # Initialize random MPS with bond dimension to prevent lock
    key = jax.random.PRNGKey(0)
    for i in range(cols):
        b_l = bond_dim if i > 0 else 1
        b_r = bond_dim if i < cols - 1 else 1
        shape = (b_l, d, b_r)
        key, subkey = jax.random.split(key)
        tensor = jax.random.normal(subkey, shape, dtype=jnp.complex64)
        tensor = tensor / jnp.linalg.norm(tensor)
        mps_tensors.append(tensor)

    print("Initial state: Random MPS (Rank {})".format(bond_dim), flush=True)

    # Initialize actual |0> MPS for physics
    mps_phys = []
    for i in range(cols):
        shape = (1, d, 1)
        t = jnp.zeros(shape, dtype=jnp.complex64)
        t = t.at[0, 0, 0].set(1.0)
        mps_phys.append(t)

    final_fids = []

    mps_current = mps_phys

    for start in range(0, len(layers), block_size):
        end = min(start + block_size, len(layers))
        block = layers[start:end]
        print(f"Processing layers {start} to {end}", flush=True)

        mps_new = []
        for i in range(cols):
            old_t = mps_current[i]
            # expand bond dims
            b_l = bond_dim if i > 0 else 1
            b_r = bond_dim if i < cols - 1 else 1
            new_t = jnp.zeros((b_l, d, b_r), dtype=jnp.complex64)
            # Copy old into top-left corner
            old_bl, _, old_br = old_t.shape
            new_t = new_t.at[:old_bl, :, :old_br].set(old_t)
            # Add small noise to allow gradient to explore?
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, new_t.shape, dtype=jnp.complex64) * 1e-3
            new_t += noise
            new_t = new_t / jnp.linalg.norm(new_t)
            mps_new.append(new_t)

        # Transform mps_new to right-canonical form before starting DMRG
        for i in range(cols - 1, 0, -1):
            t = mps_new[i]
            b_L = t.shape[0]
            t_mat = jnp.reshape(t, (b_L, -1))
            # LQ decomposition via QR of transpose
            Q_prime, R_prime = jnp.linalg.qr(t_mat.T)
            L_mat = R_prime.T
            Q_mat = Q_prime.T
            mps_new[i] = jnp.reshape(Q_mat, (-1, t.shape[1], t.shape[2]))
            mps_new[i-1] = jnp.tensordot(mps_new[i-1], L_mat, axes=[[2], [0]])

        mps_new[0] = mps_new[0] / jnp.linalg.norm(mps_new[0])

        em = EnvManager(mps_current, mps_new, block, gm, rows, cols)
        fids = em.run_dmrg(sweeps=2)
        print(f"Block Fidelities: {fids}", flush=True)
        final_fids.extend(fids)
        mps_current = em.mps_new # Result becomes old for next step

    import matplotlib.pyplot as plt
    import os
    plt.plot(final_fids)
    plt.xlabel("Sweeps (Cumulative)")
    plt.ylabel("Block Fidelity")
    plt.title("DMRG Simulation (Step-wise)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "result.png"))

if __name__ == "__main__":
    main()
