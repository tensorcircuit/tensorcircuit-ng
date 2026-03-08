"""
Reproduction of "Quantum Convolutional Neural Networks"
Link: https://arxiv.org/abs/1810.03787
Description:
This script reproduces Figure 2(c) from the paper using TensorCircuit-NG.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import quimb.tensor as qtn
import quimb as qu
import cotengra
import tensornetwork as tn

import tensorcircuit as tc

tc.set_backend("jax")

opt = cotengra.ReusableHyperOptimizer(
    parallel=4,
    minimize="combo",
    max_time=48,
    max_repeats=128,
    progbar=True,
)
tc.set_contractor("custom", optimizer=opt, preprocessing=True)


def build_cluster_dense(n_qubits, h1, h2, j=1.0):
    ls = []
    weights = []
    for i in range(n_qubits - 2):
        term = [0] * n_qubits
        term[i] = 3
        term[i + 1] = 1
        term[i + 2] = 3
        ls.append(term)
        weights.append(-j)
    for i in range(n_qubits):
        term = [0] * n_qubits
        term[i] = 1
        ls.append(term)
        weights.append(-h1)
    for i in range(n_qubits - 1):
        term = [0] * n_qubits
        term[i] = 1
        term[i + 1] = 1
        ls.append(term)
        weights.append(-h2)

    return tc.quantum.PauliStringSum2COO(ls, weights, numpy=True)


def get_dmrg_ground_state_dense(n_qubits, h1, h2):
    h_mat = build_cluster_dense(n_qubits, h1, h2)
    _, eigvecs = sla.eigsh(h_mat, k=1, which="SA")
    return eigvecs[:, 0]


def get_dmrg_ground_state_quimb(N, h1, h2, J=1.0, bond_dim=16):
    H = None

    # ZXZ terms
    for i in range(N - 2):
        ops = [np.eye(2)] * N
        ops[i] = qu.pauli("Z")
        ops[i + 1] = qu.pauli("X")
        ops[i + 2] = qu.pauli("Z")

        term_mpo = qtn.MPO_product_operator(ops)
        if H is None:
            H = -J * term_mpo
        else:
            H = H + term_mpo * (-J)

    # X terms
    for i in range(N):
        ops = [np.eye(2)] * N
        ops[i] = qu.pauli("X")

        term_mpo = qtn.MPO_product_operator(ops)
        H = H + term_mpo * (-h1)

    # XX terms
    if h2 != 0.0:
        for i in range(N - 1):
            ops = [np.eye(2)] * N
            ops[i] = qu.pauli("X")
            ops[i + 1] = qu.pauli("X")

            term_mpo = qtn.MPO_product_operator(ops)
            H = H + term_mpo * (-h2)

    dmrg = qtn.DMRG1(H, bond_dims=[bond_dim])
    dmrg.solve(tol=1e-5, verbosity=0)

    qop = tc.quantum.quimb2qop(dmrg.state)
    return qop


def evaluate_qcnn_and_sop(inputs, n_qubits, depth, is_mps=False):
    """JIT-compatible evaluation for a single state."""
    if is_mps:
        # inputs is a list of tensors, each (left, phys, right)
        # Squeeze out the size-1 dummy bond dimensions at the endpoints
        # so no ignore_edges are needed in the QuOperator.
        squeezed = []
        for i, t in enumerate(inputs):
            if i == 0:
                t = t[0, :, :]  # (1, d, bond) -> (d, bond)
            if i == len(inputs) - 1:
                t = t[:, :, 0]  # (bond, d, 1) -> (bond, d)
            squeezed.append(t)

        nodes = [tn.Node(t) for t in squeezed]

        # Connect bonds: first node is (phys, right), middle are (left, phys, right),
        # last is (left, phys)
        for i in range(n_qubits - 1):
            r_axis = len(nodes[i].edges) - 1  # right bond is last axis
            tn.connect(nodes[i][r_axis], nodes[i + 1][0])

        # Physical edges: axis 0 for first node, axis 1 for middle/last nodes
        out_edges = []
        for i, n in enumerate(nodes):
            if i == 0:
                out_edges.append(n[0])  # (phys, right)
            else:
                out_edges.append(n[1])  # (left, phys, ...) or (left, phys)

        qop = tc.quantum.QuOperator(out_edges, [])
        c = tc.Circuit(n_qubits, mps_inputs=qop)
    else:
        # Dense input vector
        c = tc.Circuit(n_qubits, inputs=inputs)

    # 1. SOP calculation
    sop_len = n_qubits // 3
    if sop_len % 2 == 0:
        sop_len += 1
    start_idx = (n_qubits - sop_len) // 2
    ops_list = [[tc.gates.z(), [start_idx]]]
    for idx in range(1, sop_len - 1, 2):
        ops_list.append([tc.gates.x(), [start_idx + idx]])
    ops_list.append([tc.gates.z(), [start_idx + sop_len - 1]])
    sop_val = tc.backend.abs(tc.backend.real(c.expectation(*ops_list, reuse=False)))

    # 2. QCNN circuit logic
    current_qubits = list(range(n_qubits))
    for _ in range(depth):
        # Disentangler
        for i in range(len(current_qubits) - 1):
            c.cz(current_qubits[i], current_qubits[i + 1])
        # Pooling
        next_qubits = []
        for i in range(0, len(current_qubits) - 2, 3):
            q1, q2, q3 = current_qubits[i], current_qubits[i + 1], current_qubits[i + 2]
            c.H(q1)
            c.H(q3)
            c.toffoli(q1, q3, q2)
            c.H(q1)
            c.H(q3)
            c.H(q1)
            c.cz(q1, q2)
            c.H(q1)
            c.H(q3)
            c.cz(q3, q2)
            c.H(q3)
            next_qubits.append(q2)
        rem = len(current_qubits) % 3
        if rem > 0:
            next_qubits.extend(current_qubits[-rem:])
        # Re-entangler
        for i in range(len(next_qubits) - 1):
            c.cz(next_qubits[i], next_qubits[i + 1])
        current_qubits = next_qubits

    # Fully connected layer
    for i in range(len(current_qubits) - 1):
        c.cz(current_qubits[i], current_qubits[i + 1])

    if len(current_qubits) >= 1:
        mid_idx = len(current_qubits) // 2
        q_mid = current_qubits[mid_idx]
        qcnn_val = tc.backend.abs(
            tc.backend.real(c.expectation_ps(x=[q_mid], reuse=False))
        )
    else:
        qcnn_val = 0.0

    return qcnn_val, sop_val


if __name__ == "__main__":
    depths = [1, 2, 3]
    h1_val = 0.5
    h2_vals = np.linspace(-1.5, 1.5, 31)

    # --- Small System (N=11) ---
    N_small = 11
    print(f"\n[Small N={N_small}] Pre-calculating Ground States...")
    states_small = []
    for h2_v in h2_vals:
        states_small.append(get_dmrg_ground_state_dense(N_small, h1_val, h2_v))
    states_small = tc.backend.convert_to_tensor(np.array(states_small))

    results_small = {}
    sop_results_small = None

    for d_val in depths:
        print(
            f"[Small N={N_small}] Evaluating Depth {d_val} using vmap (JIT accelerated)..."
        )
        eval_jit_small = tc.backend.jit(
            lambda x: evaluate_qcnn_and_sop(x, N_small, d_val, False)
        )
        qcnn_batch, sop_batch = tc.backend.vmap(eval_jit_small)(states_small)
        results_small[d_val] = np.array(qcnn_batch)
        sop_results_small = np.array(sop_batch)

    # --- Large System (N=45) ---
    N_large = 45
    print(f"\n[Large N={N_large}] Pre-calculating Ground States (MPS)...")
    mps_tensors_batch = []
    for h2_v in h2_vals:
        print(f"  Solving h2={h2_v:.2f}...", end="\r")
        qop = get_dmrg_ground_state_quimb(N_large, h1_val, h2_v, bond_dim=16)
        nodes, _, _ = tc.quantum.extract_tensors_from_qop(qop)

        # Standardize to (left, phys, right) rank-3
        std_tensors = []
        for i, n in enumerate(nodes):
            p_axis = [j for j, e in enumerate(n.edges) if e in qop.out_edges][0]
            other_axes = [j for j in range(len(n.edges)) if j != p_axis]
            l_axis, r_axis = -1, -1
            for ax in other_axes:
                e = n.edges[ax]
                neighbor_idx = i - 1 if i > 0 else -1
                if neighbor_idx >= 0 and (
                    e.node1 == nodes[neighbor_idx] or e.node2 == nodes[neighbor_idx]
                ):
                    l_axis = ax
                neighbor_idx = i + 1 if i < N_large - 1 else -1
                if neighbor_idx >= 0 and (
                    e.node1 == nodes[neighbor_idx] or e.node2 == nodes[neighbor_idx]
                ):
                    r_axis = ax

            perm = []
            if l_axis != -1:
                perm.append(l_axis)
            perm.append(p_axis)
            if r_axis != -1:
                perm.append(r_axis)

            t = n.tensor.transpose(perm)
            if l_axis == -1:
                t = t[np.newaxis, :]
            if r_axis == -1:
                t = t[..., np.newaxis]
            std_tensors.append(t)
        mps_tensors_batch.append(std_tensors)
    print("\n  Solving complete.")

    results_large = {}
    sop_results_large = None
    for d_val in depths:
        print(
            f"[Large N={N_large}] Evaluating Depth {d_val} using naive for loop (JIT accelerated)..."
        )
        eval_jit_large = tc.backend.jit(
            lambda tensors: evaluate_qcnn_and_sop(tensors, N_large, d_val, True)
        )

        q_list, s_list = [], []
        for tensors in mps_tensors_batch:
            q_val, s_val = eval_jit_large(tensors)
            q_list.append(q_val)
            s_list.append(s_val)

        results_large[d_val] = np.array(q_list)
        sop_results_large = np.array(s_list)
    print("\n  Large system evaluation complete.")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["C0", "C1", "C2"]

    for i, (N, results, sop_res, title) in enumerate(
        [
            (N_small, results_small, sop_results_small, f"Exact N={N_small}"),
            (N_large, results_large, sop_results_large, f"DMRG N={N_large}"),
        ]
    ):
        ax = axes[i]
        for idx, d_val in enumerate(depths):
            res_arr = results[d_val]
            if np.max(res_arr) > 1e-6:
                res_arr /= np.max(res_arr)
            ax.plot(
                h2_vals,
                res_arr,
                label=f"QCNN D={d_val}",
                marker="o",
                color=colors[idx],
                markersize=4,
            )

        sop_arr = np.array(sop_res)
        if np.max(sop_arr) > 1e-6:
            sop_arr /= np.max(sop_arr)
        ax.plot(
            h2_vals,
            sop_arr,
            label="SOP",
            marker="x",
            color="C3",
            linestyle="--",
            alpha=0.7,
        )

        ax.axvline(0.423, color="black", linestyle=":", label="Critical Point")
        ax.set_xlabel("$h_2 / J$")
        ax.set_ylabel("Normalized Signal")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(prop={"size": 9})

    plt.tight_layout()
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "result.png", dpi=300)
    print(f"\nPlot saved to {out_dir / 'result.png'}")
