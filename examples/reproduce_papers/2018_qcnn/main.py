"""Reproduction of "Quantum Convolutional Neural Networks"
Link: https://arxiv.org/abs/1810.03787
Description:
This script reproduces Figure 2(c) from the paper using TensorCircuit-NG.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import quimb.tensor as qtn
import quimb as qu

import tensorcircuit as tc

tc.set_backend("jax")


def build_cluster_dense(n_qubits, h1, h2, j=1.0):
    i_mat = sp.eye(2)
    x_mat = sp.csr_matrix([[0, 1], [1, 0]])
    z_mat = sp.csr_matrix([[1, 0], [0, -1]])

    def op_at(op, idx, n_q):
        ops = [i_mat] * n_q
        ops[idx] = op
        res = ops[0]
        for idx2 in range(1, n_q):
            res = sp.kron(res, ops[idx2])
        return res

    def op_at_2(op1, op2, idx1, idx2, n_q):
        ops = [i_mat] * n_q
        ops[idx1] = op1
        ops[idx2] = op2
        res = ops[0]
        for idx3 in range(1, n_q):
            res = sp.kron(res, ops[idx3])
        return res

    def op_at_3(op1, op2, op3, idx1, idx2, idx3, n_q):
        ops = [i_mat] * n_q
        ops[idx1] = op1
        ops[idx2] = op2
        ops[idx3] = op3
        res = ops[0]
        for m in range(1, n_q):
            res = sp.kron(res, ops[m])
        return res

    h_mat = sp.csr_matrix((2**n_qubits, 2**n_qubits), dtype=float)
    for i in range(n_qubits - 2):
        h_mat -= j * op_at_3(z_mat, x_mat, z_mat, i, i + 1, i + 2, n_qubits)
    for i in range(n_qubits):
        h_mat -= h1 * op_at(x_mat, i, n_qubits)
    for i in range(n_qubits - 1):
        h_mat -= h2 * op_at_2(x_mat, x_mat, i, i + 1, n_qubits)

    return h_mat


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

    # Compress explicitly checking for 0 terms? Actually, quimb compress can fail if bond_dim
    # is large but matrix is exact zeros. We can just avoid compressing H here because DMRG1
    # will use whatever H has. 45 is perfectly fine without compression. Let's just remove compress.
    # H.compress(max_bond=bond_dim * 2)

    dmrg = qtn.DMRG1(H, bond_dims=[bond_dim])
    dmrg.solve(tol=1e-5, verbosity=0)

    qop = tc.quantum.quimb2qop(dmrg.state)
    return qop


def exact_qcnn_circuit(n_qubits, depth, inputs):
    if isinstance(inputs, np.ndarray):
        c = tc.Circuit(n_qubits, inputs=inputs)
    else:
        c = tc.Circuit(n_qubits, mps_inputs=inputs)

    current_qubits = list(range(n_qubits))

    for _ in range(depth):
        # 1. Translationally invariant convolution layer
        for i in range(len(current_qubits) - 1):
            c.cz(current_qubits[i], current_qubits[i + 1])

        # 2. Pooling layer (Toffoli and conditional phase flips)
        next_qubits = []
        for i in range(0, len(current_qubits) - 2, 3):
            q1 = current_qubits[i]
            q2 = current_qubits[i + 1]
            q3 = current_qubits[i + 2]

            # CZs are already applied above, so we skip them here

            # Toffoli with controls in X basis
            c.H(q1)
            c.H(q3)
            c.toffoli(q1, q3, q2)
            c.H(q1)
            c.H(q3)

            # Phase flips when adjacent measurement is X=-1
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
        current_qubits = next_qubits

    return c, current_qubits


def calculate_sop(circ, N):
    sop_len = N // 3
    if sop_len % 2 == 0:
        sop_len += 1

    start_idx = (N - sop_len) // 2
    ops_list = []
    ops_list.append([tc.gates.z(), [start_idx]])
    for idx in range(1, sop_len - 1, 2):
        ops_list.append([tc.gates.x(), [start_idx + idx]])
    ops_list.append([tc.gates.z(), [start_idx + sop_len - 1]])

    val = circ.expectation(*ops_list, reuse=False)
    return abs(float(val.real))


def calculate_qcnn(circ, rem_qubits):
    # Fully connected layer applies CZ gates
    for i in range(len(rem_qubits) - 1):
        circ.cz(rem_qubits[i], rem_qubits[i + 1])

    if len(rem_qubits) >= 3:
        mid_idx = len(rem_qubits) // 2
        q2 = rem_qubits[mid_idx]

        # Followed by an X projection
        val = circ.expectation(
            [tc.gates.x(), [q2]],
            reuse=False,
        )
        return abs(float(val.real))

    if len(rem_qubits) >= 1:
        mid_idx = len(rem_qubits) // 2
        val = circ.expectation_ps(x=[rem_qubits[mid_idx]], reuse=False)
        return abs(float(val.real))
    return 0.0


if __name__ == "__main__":
    import cotengra

    opt = cotengra.ReusableHyperOptimizer(
        methods=["greedy"],
        parallel=False,
        minimize="combo",
        max_time=0.5,
        max_repeats=8,
        progbar=False,
    )
    tc.set_contractor("custom", optimizer=opt, preprocessing=True)

    depths = [1, 2, 3]
    h1_val = 0.5
    h2_vals = np.linspace(0.0, 1.0, 15)

    # For N=11 (exact dense)
    N_small = 11
    results_small = {d: [] for d in depths}
    sop_results_small = []

    for h2_v in h2_vals:
        state_dense = get_dmrg_ground_state_dense(N_small, h1_val, h2_v)

        # SOP
        circ_sop = tc.Circuit(N_small, inputs=state_dense)
        sop_val = calculate_sop(circ_sop, N_small)
        sop_results_small.append(sop_val)

        # QCNN
        for d_val in depths:
            circ, rem_qubits = exact_qcnn_circuit(N_small, d_val, state_dense)
            val = calculate_qcnn(circ, rem_qubits)
            results_small[d_val].append(val)
            print(f"[Small N={N_small}] Depth {d_val}, h2 {h2_v:.2f}, val {val:.4f}")

    # For N=45 (large DMRG/MPS)
    N_large = 45
    results_large = {d: [] for d in depths}
    sop_results_large = []

    for h2_v in h2_vals:
        print(f"[Large N={N_large}] Solving ground state for h2={h2_v:.2f}...")
        qop = get_dmrg_ground_state_quimb(N_large, h1_val, h2_v, bond_dim=16)

        # SOP
        circ_sop = tc.Circuit(N_large, mps_inputs=qop)
        sop_val = calculate_sop(circ_sop, N_large)
        sop_results_large.append(sop_val)

        # QCNN
        for d_val in depths:
            circ, rem_qubits = exact_qcnn_circuit(N_large, d_val, qop)
            val = calculate_qcnn(circ, rem_qubits)
            results_large[d_val].append(val)
            print(f"[Large N={N_large}] Depth {d_val}, h2 {h2_v:.2f}, val {val:.4f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # N=11 plot
    ax = axes[0]
    colors = ["C0", "C1", "C2"]
    for idx, d_val in enumerate(depths):
        res_arr = np.array(results_small[d_val])
        if np.max(res_arr) > 0:
            res_arr /= np.max(res_arr)
        ax.plot(h2_vals, res_arr, label=f"Depth {d_val}", marker="o", color=colors[idx])

    sop_arr_small = np.array(sop_results_small)
    if np.max(sop_arr_small) > 0:
        sop_arr_small /= np.max(sop_arr_small)
    ax.plot(h2_vals, sop_arr_small, label="SOP", marker="x", color="C3", linestyle="--")

    ax.axvline(0.423, color="r", linestyle=":", label="Critical Point")
    ax.set_xlabel("$h_2 / J$")
    ax.set_ylabel("Normalized Value")
    ax.set_title(f"1D SPT Phase Transition (N={N_small}, Exact)")
    ax.legend()

    # N=45 plot
    ax = axes[1]
    for idx, d_val in enumerate(depths):
        res_arr = np.array(results_large[d_val])
        if np.max(res_arr) > 0:
            res_arr /= np.max(res_arr)
        ax.plot(h2_vals, res_arr, label=f"Depth {d_val}", marker="o", color=colors[idx])

    sop_arr_large = np.array(sop_results_large)
    if np.max(sop_arr_large) > 0:
        sop_arr_large /= np.max(sop_arr_large)
    ax.plot(h2_vals, sop_arr_large, label="SOP", marker="x", color="C3", linestyle="--")

    ax.axvline(0.423, color="r", linestyle=":", label="Critical Point")
    ax.set_xlabel("$h_2 / J$")
    ax.set_ylabel("Normalized Value")
    ax.set_title(f"1D SPT Phase Transition (N={N_large}, DMRG)")
    ax.legend()

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
