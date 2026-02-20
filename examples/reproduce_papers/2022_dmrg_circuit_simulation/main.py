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
import jax.numpy as jnp
import tensorcircuit as tc
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("tensorcircuit").setLevel(logging.ERROR)

K = tc.set_backend("jax")

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
        perm = list(range(num_q, 2 * num_q)) + list(range(num_q))
        U_op = jnp.conj(jnp.transpose(U, perm))
    else:
        U_op = U
    T_axes = list(indices)
    U_axes = list(range(num_q))
    T = jnp.tensordot(T, U_op, axes=[T_axes, U_axes])
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
        return self.R[i]

    def update_L(self, L_prev, site_idx):
        A_new = self.mps_new[site_idx]
        A_old = self.mps_old[site_idx]

        # Contract L, A_new*, A_old
        T = jnp.tensordot(L_prev, jnp.conj(A_new), axes=[[1], [0]])
        T = jnp.tensordot(T, A_old, axes=[[0], [0]])

        ndim = len(T.shape)
        num_g = ndim - 4
        perm = [ndim-3, ndim-1, ndim-4, ndim-2] + list(range(num_g))
        T = jnp.transpose(T, perm)

        shape_phys = (2,) * self.rows
        shape_rest = T.shape[4:]
        T = jnp.reshape(T, (T.shape[0], T.shape[1]) + shape_phys + shape_phys + shape_rest)

        phys_new_start = 2
        phys_old_start = 2 + self.rows
        g_start = 2 + 2 * self.rows

        for k in range(self.num_layers):
            idx_ri = g_start + 2*k
            idx_ro = g_start + 2*k + 1

            gate_L = self.block_gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None

            if gate_L:
                r = gate_L["index"][1] % self.rows
                U = get_gate_matrix(gate_L).reshape(2, 2, 2, 2)

                # U(lo, ro, li, ri).
                # L provides li (Input/idx_ri), lo (Output/idx_ro).
                # Contract L[li] with U[li] (2).
                # Contract T[phys_old] with U[ri] (3).
                # Result U[lo], U[ro].
                # Match U[lo] with L[lo] (idx_ro).
                # U[ro] becomes new phys_old.

                T = jnp.tensordot(T, U, axes=[[idx_ri, phys_old_start + r], [2, 3]])
                T = jnp.trace(T, axis1=idx_ro - 1, axis2=len(T.shape)-2) # idx_ro (shifted), lo (-2)

                # ro is new phys_old.
                T = jnp.moveaxis(T, -1, phys_old_start + r)

                # Restore dummies
                T = jnp.expand_dims(T, axis=idx_ri)
                T = jnp.expand_dims(T, axis=idx_ro)

            else:
                pass

            # Internal Gates
            gates_int = self.block_gm.get_internal_gates(site_idx, k)
            for g in reversed(gates_int):
                indices = [x % self.rows for x in g['index']]
                T = apply_gate_to_tensor_indices(T, g, [phys_old_start + x for x in indices], self.rows, dagger=False)

            # Right Gate
            gate_R = self.block_gm.get_bond_gate(site_idx, k) if site_idx < self.cols - 1 else None

            if gate_R:
                r = gate_R["index"][0] % self.rows
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)

                # U(lo, ro, li, ri).
                # Contract T[phys_old] with U[li].
                T = jnp.tensordot(T, U, axes=[[phys_old_start + r], [2]])
                # T: ..., lo, ro, ri
                # lo is new phys_old.
                T = jnp.moveaxis(T, -3, phys_old_start + r)

                # ro, ri are new legs for R.
                # ro (Output to R) -> idx_ro.
                # ri (Input from R) -> idx_ri.

                T = jnp.squeeze(T, axis=(idx_ri, idx_ro))
                T = jnp.moveaxis(T, [-1, -2], [idx_ro, idx_ri]) # ri->idx_ri, ro->idx_ro

            else:
                pass

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

        for k in range(self.num_layers - 1, -1, -1):
            idx_ri = g_start + 2*k
            idx_ro = g_start + 2*k + 1

            gate_R = self.block_gm.get_bond_gate(site_idx, k) if site_idx < self.cols - 1 else None
            if gate_R:
                r = gate_R["index"][0] % self.rows
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)

                # U(lo, ro, li, ri).
                # R (T) provides ri (Input), ro (Output).
                # Contract T[idx_ri] with U[ro] (1).
                # Contract T[idx_ro] with U[ri] (3).
                # Contract T[phys_old] with U[li] (2).

                # CORRECTED AXES:
                T = jnp.tensordot(T, U, axes=[[phys_old_start+r, idx_ri, idx_ro], [2, 1, 3]])
                # T: ..., lo (new phys_old).
                T = jnp.moveaxis(T, -1, phys_old_start+r)

                T = jnp.expand_dims(T, axis=idx_ri)
                T = jnp.expand_dims(T, axis=idx_ro)
            else:
                pass

            gates_int = self.block_gm.get_internal_gates(site_idx, k)
            for g in reversed(gates_int):
                indices = [x % self.rows for x in g['index']]
                T = apply_gate_to_tensor_indices(T, g, [phys_old_start + x for x in indices], self.rows, dagger=False)

            gate_L = self.block_gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None
            if gate_L:
                r = gate_L["index"][1] % self.rows
                U = get_gate_matrix(gate_L).reshape(2, 2, 2, 2)

                # Create Legs. U(lo, ro, li, ri).
                # Contract phys_old (ri) with U[ri].
                # Expose li (Input), lo (Output).

                T = jnp.tensordot(T, U, axes=[[phys_old_start+r], [3]])
                T = jnp.moveaxis(T, -2, phys_old_start+r)
                # T: ..., lo, ro, li.
                # lo (Output to L) -> idx_ro.
                # li (Input from L) -> idx_ri.
                # ro is phys_old.

                # Replace g slots with li, lo
                T = jnp.squeeze(T, axis=(idx_ri, idx_ro))
                T = jnp.moveaxis(T, [-1, -2], [idx_ri, idx_ro]) # li->idx_ri, lo->idx_ro
            else:
                pass

        for r in range(self.rows):
            T = jnp.trace(T, axis1=2, axis2=2+self.rows-r)

        return T

    def optimize_site(self, i, L, R):
        A_old = self.mps_old[i]
        T = jnp.tensordot(L, A_old, axes=[[0], [0]])

        ndim = len(T.shape)
        shape_phys = (2,) * self.rows
        shape_rest = T.shape[1:-2]

        num_gL = len(shape_rest)
        perm = [0, ndim-2, ndim-1] + list(range(1, 1+num_gL))
        T = jnp.transpose(T, perm)
        T = jnp.reshape(T, (T.shape[0],) + shape_phys + (T.shape[1+self.rows],) + shape_rest)

        phys_start = 1
        gL_start = 1 + self.rows + 1

        for k in range(self.num_layers):
            idx_ri = gL_start + 2*k
            idx_ro = gL_start + 2*k + 1

            gate_L = self.block_gm.get_bond_gate(i - 1, k) if i > 0 else None
            if gate_L:
                r = gate_L['index'][1] % self.rows
                U = get_gate_matrix(gate_L).reshape(2, 2, 2, 2)

                T = jnp.tensordot(T, U, axes=[[phys_start+r, idx_ri], [3, 2]])
                T = jnp.trace(T, axis1=idx_ro-1, axis2=len(T.shape)-2)
                T = jnp.moveaxis(T, -1, phys_start+r)
                T = jnp.expand_dims(T, axis=idx_ri)
                T = jnp.expand_dims(T, axis=idx_ro)
            else:
                pass

            gates_int = self.block_gm.get_internal_gates(i, k)
            for g in reversed(gates_int):
                indices = [x % self.rows for x in g['index']]
                U = get_gate_matrix(g).reshape((2,)*2*len(indices))
                T_axes = [phys_start + x for x in indices]
                U_axes = list(range(len(indices), 2*len(indices)))
                T = jnp.tensordot(T, U, axes=[T_axes, U_axes])
                src = list(range(len(T.shape)-len(indices), len(T.shape)))
                T = jnp.moveaxis(T, src, T_axes)

            gate_R = self.block_gm.get_bond_gate(i, k) if i < self.cols - 1 else None
            if gate_R:
                r = gate_R['index'][0] % self.rows
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)
                T = jnp.tensordot(T, U, axes=[[phys_start+r], [2]])
                T = jnp.moveaxis(T, -3, phys_start+r)
                T = jnp.squeeze(T, axis=(idx_ri, idx_ro))
                T = jnp.moveaxis(T, [-1, -2], [idx_ro, idx_ri])
            else:
                pass

        T_axes = [1+self.rows]
        R_axes = [0]
        gR_start = 2
        for k in range(self.num_layers):
            idx_ri = gL_start + 2*k
            idx_ro = gL_start + 2*k + 1
            idx_ri_R = gR_start + 2*k
            idx_ro_R = gR_start + 2*k + 1

            T_axes.append(idx_ro) # T output matches R input (ri)
            T_axes.append(idx_ri) # T input matches R output (ro)
            R_axes.append(idx_ri_R)
            R_axes.append(idx_ro_R)

        E = jnp.tensordot(T, R, axes=[T_axes, R_axes])
        return E

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

            L_final = self.update_L(self.get_L(self.cols - 1), self.cols - 1)
            f = jnp.abs(jnp.reshape(L_final, (-1,))[0])**2
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
    mps_tensors = []
    for i in range(cols):
        shape = (1, d, 1)
        tensor = jnp.zeros(shape, dtype=jnp.complex64)
        tensor = tensor.at[0, 0, 0].set(1.0)
        mps_tensors.append(tensor)
    print("Initial state |0>", flush=True)
    final_fids = []
    for start in range(0, len(layers), block_size):
        end = min(start + block_size, len(layers))
        block = layers[start:end]
        print(f"Processing layers {start} to {end}", flush=True)
        mps_new = [x for x in mps_tensors]
        em = EnvManager(mps_tensors, mps_new, block, gm, rows, cols)
        fids = em.run_dmrg(sweeps=2)
        print(f"Block Fidelities: {fids}", flush=True)
        final_fids.extend(fids)
        mps_tensors = em.mps_new
    import matplotlib.pyplot as plt
    import os
    plt.plot(final_fids)
    plt.xlabel("Sweeps (Cumulative)")
    plt.ylabel("Block Fidelity")
    plt.title("DMRG Simulation (Step-wise)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "fidelity.png"))

if __name__ == "__main__":
    main()
