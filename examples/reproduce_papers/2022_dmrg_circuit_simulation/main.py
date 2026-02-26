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
import jax
import tensorcircuit as tc

# Using tc.backend instead of direct jax usage where possible
K = tc.set_backend("jax")
import jax.numpy as jnp  # Still needed for some specific logic, but aliased to K where appropriate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("tensorcircuit").setLevel(logging.ERROR)


# --- Helper functions for backend compatibility ---
def moveaxis(a, source, destination):
    ndim = len(a.shape)
    source = [source] if isinstance(source, int) else source
    destination = [destination] if isinstance(destination, int) else destination

    source = [s % ndim for s in source]
    destination = [d % ndim for d in destination]

    perm = [None] * ndim
    for s, d in zip(source, destination):
        perm[d] = s

    remaining = [i for i in range(ndim) if i not in source]
    idx = 0
    for i in range(ndim):
        if perm[i] is None:
            perm[i] = remaining[idx]
            idx += 1

    return K.transpose(a, perm)


def swapaxes(a, axis1, axis2):
    ndim = len(a.shape)
    perm = list(range(ndim))
    axis1 = axis1 % ndim
    axis2 = axis2 % ndim
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return K.transpose(a, perm)


def squeeze(tensor, axis=None):
    shape = tensor.shape
    if axis is None:
        new_shape = [s for s in shape if s != 1]
    else:
        if isinstance(axis, int):
            axis = [axis]
        axis = [a % len(shape) for a in axis]
        new_shape = [shape[i] for i in range(len(shape)) if i not in axis]
    return K.reshape(tensor, new_shape)


def expand_dims(tensor, axis):
    shape = list(tensor.shape)
    if axis < 0:
        axis += len(shape) + 1
    shape.insert(axis, 1)
    return K.reshape(tensor, tuple(shape))


# ------------------------------------------------


class GateManager:
    def __init__(self, layers, rows, cols):
        self.layers = layers
        self.rows = rows
        self.cols = cols

    def get_internal_gates(self, col_idx, layer_idx):
        layer = self.layers[layer_idx]
        return [
            g for g in layer if all((x // self.rows) == col_idx for x in g["index"])
        ]

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
    U = K.reshape(U, (2,) * (2 * num_q))

    if dagger:
        perm = list(range(num_q, 2 * num_q)) + list(range(num_q))
        U_op = K.conj(K.transpose(U, perm))
        # After transpose, 'in' indices are at front (0..num_q-1)
        U_axes = list(range(num_q))
    else:
        U_op = U
        # Contract with 'in' indices (num_q..2*num_q-1)
        U_axes = list(range(num_q, 2 * num_q))

    T_axes = list(indices)
    T = K.tensordot(T, U_op, axes=[T_axes, U_axes])

    # Move the new axes (from U_op output) back to the original physical positions
    sources = list(range(len(T.shape) - num_q, len(T.shape)))
    T = moveaxis(T, sources, T_axes)
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
        self.R[self.cols - 1] = K.ones(shape, dtype=dtype)
        for i in range(self.cols - 2, -1, -1):
            self.R[i] = self.update_R(self.R[i + 1], i + 1)

    def get_L(self, i):
        if i == 0:
            shape = (1, 1) + (1,) * (2 * self.num_layers)
            return K.ones(shape, dtype=self.mps_old[0].dtype)
        if i not in self.L:
            self.L[i] = self.update_L(self.get_L(i - 1), i - 1)
        return self.L[i]

    def get_R(self, i):
        if i == self.cols - 1:
            dtype = self.mps_old[0].dtype
            shape = (1, 1) + (1,) * (2 * self.num_layers)
            return K.ones(shape, dtype=dtype)
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
        T = K.tensordot(L_prev, K.conj(A_new), axes=[[0], [0]])
        # T: b_old, g..., p_new, b_new_out

        # Contract b_old (0) with A_old (b_l, 0)
        T = K.tensordot(T, A_old, axes=[[0], [0]])
        # T: g..., p_new, b_new_out, p_old, b_old_out

        ndim = len(T.shape)
        # T: g... (0..num_g-1), p_new (num_g), b_new (num_g+1), p_old (num_g+2), b_old (num_g+3)
        num_g = ndim - 4
        b_new_idx = num_g + 1
        b_old_idx = num_g + 3
        p_new_idx = num_g
        p_old_idx = num_g + 2

        perm = [b_new_idx, b_old_idx, p_new_idx, p_old_idx] + list(range(num_g))
        T = K.transpose(T, perm)

        shape_phys = (2,) * self.rows
        shape_rest = T.shape[4:]
        T = K.reshape(
            T, (T.shape[0], T.shape[1]) + shape_phys + shape_phys + shape_rest
        )

        phys_old_start = 2 + self.rows
        g_start = 2 + 2 * self.rows

        for k in range(self.num_layers):
            idx_ri = g_start
            idx_ro = g_start + 1

            # 1. Internal Gates
            gates_int = self.block_gm.get_internal_gates(site_idx, k)
            for g in gates_int:  # Apply forwards
                indices = [x % self.rows for x in g["index"]]
                T = apply_gate_to_tensor_indices(
                    T, g, [phys_old_start + x for x in indices], self.rows, dagger=False
                )

            # 2. Left Gate (Consume g legs of bond i-1)
            gate_L = (
                self.block_gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None
            )

            if gate_L:
                r = gate_L["index"][1] % self.rows
                # Contract L[ri] with T[phys_old]
                T = K.trace(T, axis1=idx_ri, axis2=phys_old_start + r)
                # idx_ri, phys_old removed. idx_ro shifts to idx_ri.

                curr_ro = idx_ro - 2  # Shifted by -2 (idx_ri and phys_old)
                # Move curr_ro to phys_old (become new psi)
                dest = phys_old_start + r
                T = moveaxis(T, curr_ro, dest)

            else:
                # Squeeze dummy gL
                T = squeeze(T, axis=(idx_ri, idx_ro))

            # 3. Right Gate (Create g legs for bond i)
            gate_R = (
                self.block_gm.get_bond_gate(site_idx, k)
                if site_idx < self.cols - 1
                else None
            )

            if gate_R:
                r = gate_R["index"][0] % self.rows
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)
                # U(lo, ro, li, ri) -> out_i, out_i+1, in_i, in_i+1

                # Contract T[phys_old] with U[li] (in_i)
                T = K.tensordot(T, U, axes=[[phys_old_start + r], [2]])

                # T: ..., lo, ro, ri.
                # lo (out_i) is new phys_old. Move to phys_old_start + r.
                T = moveaxis(T, -3, phys_old_start + r)

                # ro, ri are new legs (g) at END.
                # Swap last two axes to match optimize_site expectation (ri, ro)
                T = swapaxes(T, -1, -2)
            else:
                # Expand dummy g legs at END
                T = expand_dims(T, axis=-1)
                T = expand_dims(T, axis=-1)

        # Final Trace
        for r in range(self.rows):
            T = K.trace(T, axis1=2, axis2=2 + self.rows - r)

        return T

    def update_R(self, R_next, site_idx):
        A_new = self.mps_new[site_idx]
        A_old = self.mps_old[site_idx]

        T = K.tensordot(K.conj(A_new), R_next, axes=[[2], [1]])
        T = K.tensordot(T, A_old, axes=[[2], [2]])

        ndim = len(T.shape)
        num_g = ndim - 4
        perm = [ndim - 2, 0, 1, ndim - 1] + list(range(2, 2 + num_g))
        T = K.transpose(T, perm)

        shape_phys = (2,) * self.rows
        shape_rest = T.shape[4:]
        T = K.reshape(
            T, (T.shape[0], T.shape[1]) + shape_phys + shape_phys + shape_rest
        )

        phys_old_start = 2 + self.rows
        g_start = 2 + 2 * self.rows

        for k in range(self.num_layers):  # Forward loop
            # Always consume from g_start because processed pairs are appended to the end
            idx_ri = g_start
            idx_ro = g_start + 1

            # 1. Internal Gates
            gates_int = self.block_gm.get_internal_gates(site_idx, k)
            for g in gates_int:  # Apply forwards
                indices = [x % self.rows for x in g["index"]]
                T = apply_gate_to_tensor_indices(
                    T, g, [phys_old_start + x for x in indices], self.rows, dagger=False
                )

            # 2. Right Gate (Consume g legs of bond i+1)
            gate_R = (
                self.block_gm.get_bond_gate(site_idx, k)
                if site_idx < self.cols - 1
                else None
            )
            if gate_R:
                r = gate_R["index"][0] % self.rows
                # Contract T[ri] with phys_old (psi)
                T = K.trace(T, axis1=idx_ri, axis2=phys_old_start + r)
                # idx_ri, phys_old removed. idx_ro shifts -2 (idx_ri > phys_old, idx_ro > idx_ri).

                curr_ro = idx_ro - 2
                # Move curr_ro to phys_old position
                dest = phys_old_start + r
                T = moveaxis(T, curr_ro, dest)
            else:
                T = squeeze(T, axis=(idx_ri, idx_ro))

            # 3. Left Gate (Create g legs for bond i)
            gate_L = (
                self.block_gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None
            )
            if gate_L:
                r = gate_L["index"][1] % self.rows
                U = get_gate_matrix(gate_L).reshape(2, 2, 2, 2)

                # Contract T[phys_old] with U[in_R] (ri, 3).
                T = K.tensordot(T, U, axes=[[phys_old_start + r], [3]])

                # T: ..., lo, ro, li.
                # ro (1) is out_R. It becomes new phys_old.
                # lo (0), li (2) are g legs (on i-1).

                # Move ro to phys_old.
                T = moveaxis(T, -2, phys_old_start + r)

                # Remaining: lo (idx -2), li (idx -1).
                # These are appended to the END.

                # Swap last two axes to match optimize_site expectation (li, lo)
                T = swapaxes(T, -1, -2)
            else:
                # Expand dummy legs at END
                T = expand_dims(T, axis=-1)
                T = expand_dims(T, axis=-1)

        for r in range(self.rows):
            T = K.trace(T, axis1=2, axis2=2 + self.rows - r)

        return T

    def optimize_site(self, i, L, R):
        A_old = self.mps_old[i]
        # A_old: (b_l, p, b_r)

        # 1. Contract L and A_old
        # L: (b_phi, b_psi, g_L...) -> we contract b_psi (1) with A_old (0)
        # A_old: (b_psi, p, b_psi_R)
        T = K.tensordot(L, A_old, axes=[[1], [0]])
        # T: (b_phi, g_L..., p, b_psi_R)

        # 2. Contract T and R
        # R: (b_psi, b_phi, g_R...) -> we contract b_psi (0) with T's last (b_psi_R)
        # T: (b_phi, g_L..., p, b_psi_R)
        T = K.tensordot(T, R, axes=[[len(T.shape) - 1], [0]])
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
        perm = (
            [0, p_idx, p_idx + 1] + list(range(1, p_idx)) + list(range(p_idx + 2, ndim))
        )
        # perm: [b_L_n, p, b_R_n, g_L..., g_R...]
        T = K.transpose(T, perm)

        # Reshape p
        shape_rest = T.shape[3:]
        T = K.reshape(T, (T.shape[0],) + shape_phys + (T.shape[2],) + shape_rest)

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
                r = gate_L["index"][1] % self.rows
                # Contract q_r (q_start+r) with idx_ri_L
                T = K.trace(T, axis1=q_start + r, axis2=idx_ri_L)

                curr_ro = idx_ro_L - 2
                # Move idx_ro_L to q_start+r (become new q)
                T = moveaxis(T, curr_ro, q_start + r)

                # gL block reduced by 2.
                gR_start -= 2

            else:
                # Squeeze dummy gL
                T = squeeze(T, axis=(idx_ri_L, idx_ro_L))
                gR_start -= 2

            gates_int = self.block_gm.get_internal_gates(i, k)
            for g in gates_int:
                indices = [x % self.rows for x in g["index"]]
                # apply gate to q_indices
                T = apply_gate_to_tensor_indices(
                    T, g, [q_start + x for x in indices], self.rows, dagger=False
                )

            idx_ri_R = gR_start
            idx_ro_R = gR_start + 1

            gate_R = self.block_gm.get_bond_gate(i, k) if i < self.cols - 1 else None
            if gate_R:
                r = gate_R["index"][0] % self.rows
                # Contract q_r with idx_ri_R
                T = K.trace(T, axis1=q_start + r, axis2=idx_ri_R)

                # idx_ro_R becomes q_r
                curr_ro = idx_ro_R - 2
                T = moveaxis(T, curr_ro, q_start + r)

                # gR block reduced by 2.
            else:
                T = squeeze(T, axis=(idx_ri_R, idx_ro_R))

        # Final T: (b_L_n, q0..qN, b_R_n)
        # Reshape q to p
        T = K.reshape(T, (T.shape[0], -1, T.shape[-1]))
        return T

    def run_dmrg(self, sweeps=2):
        fidelities = []
        for sweep in range(sweeps):
            for i in range(self.cols):
                L = self.get_L(i)
                R = self.get_R(i)
                E = self.optimize_site(i, L, R)

                if i < self.cols - 1:
                    b_L = E.shape[0]
                    b_R = E.shape[2]
                    E_mat = K.reshape(E, (-1, b_R))
                    Q, R_mat = K.qr(E_mat)
                    new_dim = Q.shape[1]
                    self.mps_new[i] = K.reshape(Q, (b_L, -1, new_dim))
                    next_A = self.mps_new[i + 1]
                    self.mps_new[i + 1] = K.tensordot(R_mat, next_A, axes=[[1], [0]])
                else:
                    norm = K.norm(E)
                    self.mps_new[i] = E / norm

                if i < self.cols - 1:
                    if (i + 1) in self.L:
                        del self.L[i + 1]

            for i in range(self.cols - 1, -1, -1):
                L = self.get_L(i)
                R = self.get_R(i)
                E = self.optimize_site(i, L, R)

                if i > 0:
                    b_L = E.shape[0]
                    E_mat = K.reshape(E, (b_L, -1))
                    Q_prime, R_prime = K.qr(
                        K.transpose(E_mat)
                    )  # K.qr handles rectangular
                    # Note: K.qr(X) -> Q, R such that Q R = X
                    # We want LQ: X = L Q. X^T = Q^T L^T = Q' R'.
                    # Q' R' = X^T => R'^T Q'^T = X.
                    # L_mat = R_prime.T, Q_mat = Q_prime.T
                    L_mat = K.transpose(R_prime)
                    Q_mat = K.transpose(Q_prime)
                    self.mps_new[i] = K.reshape(Q_mat, (-1, E.shape[1], E.shape[2]))
                    prev_A = self.mps_new[i - 1]
                    self.mps_new[i - 1] = K.tensordot(prev_A, L_mat, axes=[[2], [0]])
                else:
                    norm = K.norm(E)
                    self.mps_new[i] = E / norm

                if i > 0:
                    if (i - 1) in self.R:
                        del self.R[i - 1]

        # Use the squared norm of the final environment tensor (at site 0) as fidelity
        f = float(norm**2)
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
        # 1. Rz layer
        l_rz = []
        for i in range(rows * cols):
            l_rz.append(
                {
                    "gatef": tc.gates.rz,
                    "index": [i],
                    "parameters": {"theta": random.random()},
                }
            )
        layers.append(l_rz)

        # 2. Vertical CZ layer (Internal)
        l_v = []
        for r in range(rows - 1):
            for c_ in range(cols):
                idx = c_ * rows + r
                idx_next = c_ * rows + (r + 1)
                l_v.append(
                    {
                        "gatef": tc.gates.cz,
                        "index": [idx, idx_next],
                        "parameters": {},
                    }
                )
        if l_v:
            layers.append(l_v)

        # 3. Horizontal CZ layers (Bond) - Must be serialized per row
        for r in range(rows):
            l_h = []
            for c_ in range(cols - 1):
                idx = c_ * rows + r
                idx_next = (c_ + 1) * rows + r
                l_h.append(
                    {
                        "gatef": tc.gates.cz,
                        "index": [idx, idx_next],
                        "parameters": {},
                    }
                )
            if l_h:
                layers.append(l_h)

    return layers


def main():
    rows = 4
    cols = 4
    total_depth = 8
    block_size = 2
    layers = generate_sycamore_circuit(rows, cols, total_depth)
    gm = GateManager(layers, rows, cols)
    d = 2**rows

    # Reproduction of Figure 2(a): Infidelity vs Bond Dimension
    bond_dims = [2, 4, 8, 16]
    infidelities = []

    import matplotlib.pyplot as plt

    for bond_dim in bond_dims:
        print(f"\\nRunning simulation with bond_dim={bond_dim}...", flush=True)
        key = jax.random.PRNGKey(0)

        # Initialize actual |0> MPS for physics
        mps_phys = []
        for i in range(cols):
            shape = (1, d, 1)
            t = jnp.zeros(shape, dtype=jnp.complex64)
            t = t.at[0, 0, 0].set(1.0)
            mps_phys.append(t)

        mps_current = mps_phys

        block_fidelities = []

        for start in range(0, len(layers), block_size):
            end = min(start + block_size, len(layers))
            block = layers[start:end]
            # print(f"  Processing layers {start} to {end}", flush=True)

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
                # Add small noise to allow gradient to explore
                key, subkey = jax.random.split(key)
                noise = (
                    jax.random.normal(subkey, new_t.shape, dtype=jnp.complex64) * 1e-3
                )
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
                mps_new[i - 1] = jnp.tensordot(mps_new[i - 1], L_mat, axes=[[2], [0]])

            mps_new[0] = mps_new[0] / jnp.linalg.norm(mps_new[0])

            em = EnvManager(mps_current, mps_new, block, gm, rows, cols)
            fids = em.run_dmrg(sweeps=4)
            final_fid = fids[-1]
            block_fidelities.append(final_fid)
            mps_current = em.mps_new  # Result becomes old for next step

        # Calculate Total Fidelity of the circuit
        total_fid = float(jnp.prod(jnp.array(block_fidelities)))
        infidelity = max(1.0 - total_fid, 1e-7)
        print(f"  Total Infidelity for chi={bond_dim}: {infidelity}", flush=True)
        infidelities.append(infidelity)

    plt.figure()
    plt.plot(bond_dims, infidelities, marker="o", linestyle="-")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Bond Dimension (chi)")
    plt.ylabel("Infidelity (1 - F)")
    plt.title("DMRG Infidelity Scaling (Figure 2a Reproduction)")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "result.png"))


if __name__ == "__main__":
    main()
