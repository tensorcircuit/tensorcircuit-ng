"""
Reproduction of "A density-matrix renormalization group algorithm for simulating
quantum circuits with a finite fidelity"
Link: https://arxiv.org/abs/2207.05612

Method:
- Single-site DMRG with JAX backend.
- Qubit Grouping: Columns of the grid are mapped to single MPS sites.
- Vertical Contraction: Environment tensors L and R carry "gate legs".
- Analytic Update: New site tensor is computed by contracting L, R, and the circuit column.
"""

import logging
import jax
import jax.numpy as jnp
import tensorcircuit as tc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("tensorcircuit").setLevel(logging.ERROR)

# Use JAX backend
K = tc.set_backend("jax")


class GateManager:
    """Manages the circuit layers and identifies gates."""

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


def apply_gate_to_tensor(T, gate, rows, dagger=False):
    """
    Applies a gate to the physical indices of tensor T.
    T: (b, q0, q1, ..., qR-1, ...)
    gate: Dictionary with 'index' (global indices).
    dagger: If True, apply U^dag.
    """
    indices = [x % rows for x in gate["index"]]
    num_q = len(indices)
    U = get_gate_matrix(gate)

    # Reshape U to (2, 2..., 2, 2...) (Outs, Ins)
    shape = (2,) * (2 * num_q)
    U = jnp.reshape(U, shape)

    if dagger:
        # U_dag: Conj and Transpose (In <-> Out)
        # U indices: 0..n-1 (Out), n..2n-1 (In)
        # Permute to (In, Out)
        perm = list(range(num_q, 2 * num_q)) + list(range(num_q))
        U_op = jnp.conj(jnp.transpose(U, perm))
    else:
        U_op = U

    # Contract T with U_op.
    # U_op inputs are 0..n-1.
    # T indices are 1+r.

    T_axes = [1 + r for r in indices]
    U_axes = list(range(num_q))

    T = jnp.tensordot(T, U_op, axes=[T_axes, U_axes])

    # New indices are appended at the end.
    # We need to move them back to 1+r.
    # Sources: last num_q indices.
    # Destinations: 1+r.

    sources = list(range(len(T.shape) - num_q, len(T.shape)))
    destinations = [1 + r for r in indices]

    T = jnp.moveaxis(T, sources, destinations)
    return T


class EnvManager:
    def __init__(self, mps_tensors, gate_manager, rows, cols):
        self.mps_tensors = mps_tensors
        self.gm = gate_manager
        self.rows = rows
        self.cols = cols
        self.L = {}
        self.R = {}
        self.mps_dim = 2**rows
        self.num_layers = len(gate_manager.layers)

        self.recompute_all_R()

    def recompute_all_R(self):
        dtype = self.mps_tensors[0].dtype
        shape = (1,) + (1,) * (2 * self.num_layers)
        self.R[self.cols - 1] = jnp.ones(shape, dtype=dtype)

        for i in range(self.cols - 2, -1, -1):
            self.R[i] = self.update_R(self.R[i + 1], i + 1)

    def get_L(self, i):
        if i == 0:
            shape = (1,) + (1,) * (2 * self.num_layers)
            return jnp.ones(shape, dtype=self.mps_tensors[0].dtype)
        if i not in self.L:
            self.L[i] = self.update_L(self.get_L(i - 1), i - 1)
        return self.L[i]

    def get_R(self, i):
        return self.R[i]

    def update_L(self, L_prev, site_idx):
        A = self.mps_tensors[site_idx]
        T = jnp.tensordot(L_prev, jnp.conj(A), axes=[[0], [0]])

        ndim = len(T.shape)
        perm = [ndim - 1, ndim - 2] + list(range(ndim - 2))
        T = jnp.transpose(T, perm)

        shape_phys = (2,) * self.rows
        shape_rest = T.shape[2:]
        T = jnp.reshape(T, (T.shape[0],) + shape_phys + shape_rest)

        for k in range(self.num_layers - 1, -1, -1):
            # 1. Right Gate (Create new L legs)
            gate_R = (
                self.gm.get_bond_gate(site_idx, k) if site_idx < self.cols - 1 else None
            )

            if gate_R:
                r = gate_R["index"][0] % self.rows
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)
                U_dag = jnp.conj(jnp.transpose(U, (2, 3, 0, 1)))
                T = jnp.tensordot(T, U_dag, axes=[[1 + r], [2]])
                T = jnp.moveaxis(T, -3, 1 + r)
            else:
                T = jnp.expand_dims(T, axis=[-1])
                T = jnp.expand_dims(T, axis=[-1])

            # 2. Internal Gates
            gates_int = self.gm.get_internal_gates(site_idx, k)
            for g in reversed(gates_int):
                T = apply_gate_to_tensor(T, g, self.rows, dagger=True)

            # 3. Left Gate (Consume L_prev legs)
            gate_L = self.gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None

            if gate_L:
                g_start = 1 + self.rows
                idx_out = g_start + 2 * k
                idx_in = g_start + 2 * k + 1
                r = gate_L["index"][1] % self.rows
                U = get_gate_matrix(gate_L).reshape(2, 2, 2, 2)
                U_dag = jnp.conj(jnp.transpose(U, (2, 3, 0, 1)))
                T = jnp.tensordot(T, U_dag, axes=[[1 + r, idx_out, idx_in], [3, 2, 0]])
                T = jnp.moveaxis(T, -1, 1 + r)
            else:
                g_start = 1 + self.rows
                idx_out = g_start + 2 * k
                idx_in = g_start + 2 * k + 1
                T = jnp.squeeze(T, axis=(idx_out, idx_in))

        # 4. Final Contraction with |0>
        s = (
            [slice(None)]
            + [0] * self.rows
            + [slice(None)] * (len(T.shape) - 1 - self.rows)
        )
        T = T[tuple(s)]

        perm = [0]
        for k in range(self.num_layers):
            idx = 1 + 2 * (self.num_layers - 1 - k)
            perm.append(idx)
            perm.append(idx + 1)
        T = jnp.transpose(T, perm)
        return T

    def update_R(self, R_next, site_idx):
        A = self.mps_tensors[site_idx]
        T = jnp.tensordot(jnp.conj(A), R_next, axes=[[2], [0]])
        T = jnp.reshape(T, (T.shape[0],) + (2,) * self.rows + T.shape[2:])

        for k in range(self.num_layers - 1, -1, -1):
            # 1. Left Gate (Create new R legs)
            gate_L = self.gm.get_bond_gate(site_idx - 1, k) if site_idx > 0 else None

            if gate_L:
                r = gate_L["index"][1] % self.rows
                U = get_gate_matrix(gate_L).reshape(2, 2, 2, 2)
                U_dag = jnp.conj(jnp.transpose(U, (2, 3, 0, 1)))
                T = jnp.tensordot(T, U_dag, axes=[[1 + r], [3]])
                T = jnp.moveaxis(T, -3, 1 + r)
            else:
                T = jnp.expand_dims(T, axis=[-1])
                T = jnp.expand_dims(T, axis=[-1])

            # 2. Internal Gates
            gates_int = self.gm.get_internal_gates(site_idx, k)
            for g in reversed(gates_int):
                T = apply_gate_to_tensor(T, g, self.rows, dagger=True)

            # 3. Right Gate (Consume R_next legs)
            gate_R = (
                self.gm.get_bond_gate(site_idx, k) if site_idx < self.cols - 1 else None
            )

            if gate_R:
                r = gate_R["index"][0] % self.rows
                g_start = 1 + self.rows
                idx_ro = g_start + 2 * k + 1
                idx_ri = g_start + 2 * k
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)
                U_dag = jnp.conj(jnp.transpose(U, (2, 3, 0, 1)))
                T = jnp.tensordot(T, U_dag, axes=[[1 + r, idx_ro, idx_ri], [2, 3, 1]])
                T = jnp.moveaxis(T, -1, 1 + r)
            else:
                g_start = 1 + self.rows
                idx_ro = g_start + 2 * k + 1
                idx_ri = g_start + 2 * k
                T = jnp.squeeze(T, axis=(idx_ri, idx_ro))

        # 4. Final slice |0>
        s = (
            [slice(None)]
            + [0] * self.rows
            + [slice(None)] * (len(T.shape) - 1 - self.rows)
        )
        T = T[tuple(s)]

        perm = [0]
        for k in range(self.num_layers):
            idx = 1 + 2 * (self.num_layers - 1 - k)
            perm.append(idx)
            perm.append(idx + 1)
        T = jnp.transpose(T, perm)
        return T

    def optimize_site(self, i, L, R):
        T = jnp.expand_dims(L, axis=list(range(1, 1 + self.rows)))
        shape_q = (2,) * self.rows
        shape_L_rest = L.shape[1:]
        T_full = jnp.zeros((L.shape[0],) + shape_q + shape_L_rest, dtype=L.dtype)
        s = [slice(None)] + [0] * self.rows + [slice(None)] * len(shape_L_rest)
        T_full = T_full.at[tuple(s)].set(L)
        T = T_full

        for k in range(self.num_layers):
            # 1. Left Gate (Consume L legs)
            gate_L = self.gm.get_bond_gate(i - 1, k) if i > 0 else None
            if gate_L:
                r = gate_L["index"][1] % self.rows
                idx_ri = 1 + self.rows
                idx_ro = 1 + self.rows + 1
                T = jnp.trace(T, axis1=1 + r, axis2=idx_ri)
                T = jnp.moveaxis(T, idx_ro - 2, 1 + r)
            else:
                idx_ri = 1 + self.rows
                idx_ro = 1 + self.rows + 1
                T = jnp.squeeze(T, axis=(idx_ri, idx_ro))

            # 2. Internal Gates
            gates_int = self.gm.get_internal_gates(i, k)
            for g in gates_int:
                T = apply_gate_to_tensor(T, g, self.rows, dagger=False)

            # 3. Right Gate (Connect to R)
            gate_R = self.gm.get_bond_gate(i, k) if i < self.cols - 1 else None

            if gate_R:
                r = gate_R["index"][0] % self.rows
                U = get_gate_matrix(gate_R).reshape(2, 2, 2, 2)
                T = jnp.tensordot(T, U, axes=[[1 + r], [2]])
                T = jnp.moveaxis(T, -3, 1 + r)
            else:
                T = jnp.expand_dims(T, axis=[-1])
                T = jnp.expand_dims(T, axis=[-1])

        g_start_T = 1 + self.rows
        g_start_R = 1
        T_axes = []
        R_axes = []

        for k in range(self.num_layers):
            idx_ro_T = g_start_T + 2 * k
            idx_ri_T = g_start_T + 2 * k + 1
            idx_ri_R = g_start_R + 2 * k
            idx_ro_R = g_start_R + 2 * k + 1
            T_axes.append(idx_ro_T)
            T_axes.append(idx_ri_T)
            R_axes.append(idx_ro_R)
            R_axes.append(idx_ri_R)

        E = jnp.tensordot(T, R, axes=[T_axes, R_axes])
        E = jnp.reshape(E, (E.shape[0], -1, E.shape[-1]))

        norm = jnp.linalg.norm(E)
        E = E / norm
        return E, norm

    def run_dmrg(self, sweeps=10):
        fidelities = []

        for sweep in range(sweeps):
            # Left to Right
            for i in range(self.cols):
                L = self.get_L(i)
                R = self.get_R(i)
                E, _ = self.optimize_site(i, L, R)
                self.mps_tensors[i] = E
                if i < self.cols - 1:
                    if (i + 1) in self.L:
                        del self.L[i + 1]

            # Right to Left
            for i in range(self.cols - 1, -1, -1):
                L = self.get_L(i)
                R = self.get_R(i)
                E, _ = self.optimize_site(i, L, R)
                self.mps_tensors[i] = E
                if i > 0:
                    if (i - 1) in self.R:
                        del self.R[i - 1]
                    self.R[i - 1] = self.update_R(self.R[i], i)

            L_final = self.update_L(self.get_L(self.cols - 1), self.cols - 1)
            f = jnp.linalg.norm(L_final) ** 2
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
            l.append(
                {
                    "gatef": tc.gates.rz,
                    "index": [i],
                    "parameters": {"theta": random.random()},
                }
            )
        for r in range(rows):
            for c_ in range(cols):
                idx = c_ * rows + r
                if c_ < cols - 1:
                    idx_next = (c_ + 1) * rows + r
                    l.append(
                        {
                            "gatef": tc.gates.cz,
                            "index": [idx, idx_next],
                            "parameters": {},
                        }
                    )
                if r < rows - 1:
                    idx_next = c_ * rows + (r + 1)
                    l.append(
                        {
                            "gatef": tc.gates.cz,
                            "index": [idx, idx_next],
                            "parameters": {},
                        }
                    )
        layers.append(l)

    return layers


def main():
    rows = 4
    cols = 4
    depth = 4

    layers = generate_sycamore_circuit(rows, cols, depth)
    gm = GateManager(layers, rows, cols)

    bond_dim = 10
    d = 2**rows
    mps_tensors = []

    key = jax.random.PRNGKey(0)
    for i in range(cols):
        key, subkey = jax.random.split(key)
        shape = (bond_dim, d, bond_dim)
        if i == 0:
            shape = (1, d, bond_dim)
        if i == cols - 1:
            shape = (bond_dim, d, 1)

        tensor = jax.random.normal(subkey, shape) + 1j * jax.random.normal(
            subkey, shape
        )
        tensor = tensor / jnp.linalg.norm(tensor)
        mps_tensors.append(tensor)

    em = EnvManager(mps_tensors, gm, rows, cols)

    logger.info("Starting DMRG...")
    fids = em.run_dmrg(sweeps=2)

    print("Final Fidelities:", fids)

    import matplotlib.pyplot as plt
    import os

    plt.plot(fids)
    plt.xlabel("Sweeps")
    plt.ylabel("Fidelity")
    plt.title("DMRG Simulation of Quantum Circuit")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(base_dir, "fidelity.png"))


if __name__ == "__main__":
    main()
