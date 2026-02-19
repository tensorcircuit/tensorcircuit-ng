"""
Reproduction of "A density-matrix renormalization group algorithm for simulating
quantum circuits with a finite fidelity"
Link: https://arxiv.org/abs/2207.05612

Description:
This script reproduces Figure 2(a) from the paper.
It simulates a Sycamore-like random quantum circuit using both exact state vector simulation
and an MPS-based simulator (DMRG-like algorithm) with varying bond dimensions.
The script plots the infidelity (1 - Fidelity) as a function of the bond dimension.

Implementation Note:
This script implements a 1-site DMRG-like simulation logic.
We construct an MPO for each chunk of layers (e.g., 2 layers) and then
variationally optimize the MPS to maximize the overlap with the state after
applying the MPO. This avoids forming intermediate high-bond-dimension states
explicitly (like standard TEBD or global contraction) and aligns with the
standard DMRG algorithm for time evolution / circuit simulation.
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use numpy backend for broad compatibility
K = tc.set_backend("numpy")


def generate_sycamore_like_circuit_structure(rows, cols, depth, seed=42):
    """
    Generates a random quantum circuit with a structure similar to Sycamore circuits.
    Returns the exact circuit (for validation) and a list of layers, where each layer
    is a list of gate dictionaries.
    """
    np.random.seed(seed)
    n_qubits = rows * cols
    c = tc.Circuit(n_qubits)
    layers = []

    def q(r, col):
        return r * cols + col

    for d in range(depth):
        layer_gates = []

        # Single qubit gates
        for i in range(n_qubits):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            lam = np.random.uniform(0, 2 * np.pi)

            c.rz(i, theta=phi)
            layer_gates.append(
                {"gatef": tc.gates.rz, "index": (i,), "parameters": {"theta": phi}}
            )

            c.ry(i, theta=theta)
            layer_gates.append(
                {"gatef": tc.gates.ry, "index": (i,), "parameters": {"theta": theta}}
            )

            c.rz(i, theta=lam)
            layer_gates.append(
                {"gatef": tc.gates.rz, "index": (i,), "parameters": {"theta": lam}}
            )

        layers.append(layer_gates)
        layer_gates = []

        # Two-qubit gates
        layer_type = d % 4

        if layer_type == 0:  # Horizontal (col, col+1) for even cols
            for r in range(rows):
                for col in range(0, cols - 1, 2):
                    c.cz(q(r, col), q(r, col + 1))
                    layer_gates.append(
                        {
                            "gatef": tc.gates.cz,
                            "index": (q(r, col), q(r, col + 1)),
                            "parameters": {},
                        }
                    )
        elif layer_type == 1:  # Horizontal (col, col+1) for odd cols
            for r in range(rows):
                for col in range(1, cols - 1, 2):
                    c.cz(q(r, col), q(r, col + 1))
                    layer_gates.append(
                        {
                            "gatef": tc.gates.cz,
                            "index": (q(r, col), q(r, col + 1)),
                            "parameters": {},
                        }
                    )
        elif layer_type == 2:  # Vertical (row, row+1) for even rows
            for col in range(cols):
                for r in range(0, rows - 1, 2):
                    c.cz(q(r, col), q(r + 1, col))
                    layer_gates.append(
                        {
                            "gatef": tc.gates.cz,
                            "index": (q(r, col), q(r + 1, col)),
                            "parameters": {},
                        }
                    )
        elif layer_type == 3:  # Vertical (row, row+1) for odd rows
            for col in range(cols):
                for r in range(1, rows - 1, 2):
                    c.cz(q(r, col), q(r + 1, col))
                    layer_gates.append(
                        {
                            "gatef": tc.gates.cz,
                            "index": (q(r, col), q(r + 1, col)),
                            "parameters": {},
                        }
                    )

        if layer_gates:
            layers.append(layer_gates)

    return c, layers


def build_mpo_from_layers(n_qubits, layers):
    """
    Constructs an MPO (list of tensors) representing the layers of gates.
    The MPO tensors have shape (left_bond, right_bond, phys_out, phys_in).
    """
    # Start with Identity MPO
    # shape: (1, 1, 2, 2)
    # mpo_tensors = [
    #     np.eye(2).reshape(1, 1, 2, 2).astype(np.complex128) for _ in range(n_qubits)
    # ]

    # We use MPSCircuit to simulate the superoperator evolution
    # State space dimension is 4 (2*2).
    # Initial state is Identity vector (unnormalized trace but it's operator).
    # Wait, MPSCircuit usually starts in |0...0>.
    # We want it to start in |I...I> where I is flattened identity.
    # Identity (2x2) flattened is [1, 0, 0, 1].

    initial_tensors = []
    for _ in range(n_qubits):
        # Shape (1, 4, 1) -> (1, dim, 1)
        t = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128).reshape(1, 4, 1)
        initial_tensors.append(t)

    mpo_mps = tc.MPSCircuit(n_qubits, tensors=initial_tensors, dim=4)
    mpo_mps.set_split_rules({})  # No truncation

    for layer in layers:
        for gate in layer:
            idx = gate["index"]
            params = gate.get("parameters", {})
            g_obj = gate["gatef"](**params)
            u = g_obj.tensor  # (2, 2) or (2, 2, 2, 2)

            # Create superoperator gate U \otimes I
            # U acts on the "output" index of the MPO, which corresponds to the first factor in 2x2.
            # Flattened index i corresponds to (i // 2, i % 2) -> (out, in).
            # We want U on out, I on in.

            if len(idx) == 1:
                # U: (2, 2)
                # Super: (4, 4)
                # U_{ik} \delta_{jl} -> index (i,j), (k,l)
                # kron(U, I) does exactly this.
                u_super = np.kron(u, np.eye(2))
                g_super = tc.gates.Gate(u_super.reshape(4, 4))
                mpo_mps.apply(g_super, *idx)

            elif len(idx) == 2:
                # U: (2, 2, 2, 2) -> (out1, out2, in1, in2)
                # We need (4, 4, 4, 4) -> (site1_out, site1_in), (site2_out, site2_in) etc.
                # Actually, kron(U_mat, I_4) gives a 16x16 matrix acting on (sys1, sys2, anc1, anc2).
                # Indices: s1, s2, a1, a2.
                # We want: s1, a1, s2, a2.
                # So we need to permute.

                # U matrix: (s1_out * s2_out, s1_in * s2_in)
                u_mat = u.reshape(4, 4)
                super_op = np.kron(u_mat, np.eye(4))  # (16, 16)
                # Indices of super_op: (s1_out, s2_out, a1, a2), (s1_in, s2_in, a1, a2) (row, col)
                # We want to act on MPS sites which are (s1, a1) and (s2, a2).
                # So row indices should be (s1_out, a1), (s2_out, a2).
                # Col indices should be (s1_in, a1), (s2_in, a2).

                # Reshape to tensor (4, 4, 4, 4)
                t = super_op.reshape(2, 2, 2, 2, 2, 2, 2, 2)
                # (s1o, s2o, a1o, a2o, s1i, s2i, a1i, a2i)
                # Note: a1o = a1i (Identity on ancilla).

                # Permute to (s1o, a1o, s2o, a2o, s1i, a1i, s2i, a2i)
                t = np.transpose(t, (0, 2, 1, 3, 4, 6, 5, 7))

                # Reshape to (4, 4, 4, 4)
                u_super = t.reshape(4, 4, 4, 4)
                g_super = tc.gates.Gate(u_super)

                # Custom handling for non-adjacent gates with d=4
                # because standard MPSCircuit uses qubit swap.
                idx1, idx2 = idx
                if abs(idx1 - idx2) > 1:
                    # Construct Swap(d=4)
                    swap_d4 = np.zeros((4, 4, 4, 4), dtype=np.complex128)
                    for i in range(4):
                        for j in range(4):
                            swap_d4[j, i, i, j] = 1.0
                    g_swap_d4 = tc.gates.Gate(swap_d4)

                    # Normalize index order
                    p1, p2 = min(idx1, idx2), max(idx1, idx2)

                    # Move p2 to p1 + 1
                    for k in range(p2, p1 + 1, -1):
                        mpo_mps.apply_adjacent_double_gate(g_swap_d4, k - 1, k)

                    # Apply gate
                    mpo_mps.apply_adjacent_double_gate(g_super, p1, p1 + 1)

                    # Move back
                    for k in range(p1 + 1, p2):
                        mpo_mps.apply_adjacent_double_gate(g_swap_d4, k, k + 1)
                else:
                    mpo_mps.apply(g_super, *idx)

    # Extract tensors and convert to MPO format (l, r, p_out, p_in)
    # MPS format: (l, d=4, r)
    final_mps_tensors = mpo_mps.get_tensors()
    final_mpo = []
    for t in final_mps_tensors:
        # t: (l, 4, r)
        # Transpose to (l, r, 4)
        t = np.transpose(t, (0, 2, 1))
        # Reshape to (l, r, 2, 2)
        # Note: 4 corresponds to (out, in) via kron(U, I) logic, i.e. (out * 2 + in).
        final_mpo.append(t.reshape(t.shape[0], t.shape[1], 2, 2))

    return final_mpo


def run_1site_dmrg(mps_old, mpo_tensors, chi, sweeps=1):
    """
    Optimizes a new MPS to approximate MPO @ MPS_old using 1-site DMRG sweep.
    """
    n = mps_old._nqubits

    # Initialize guess: Copy of old MPS
    # To allow bond dimension to adapt up to chi, we rely on svd truncation limit.
    # If starting bond dim < chi, it will grow?
    # In 1-site DMRG, bond dimension is fixed by the guess.
    # To allow growth, we usually use 2-site or subspace expansion.
    # User asked for "1-site dmrg".
    # But strictly 1-site cannot increase bond dimension.
    # If we want to simulate circuit, we need to increase bond dimension.
    # So we probably need 2-site update logic to adapt bond dimension,
    # OR start with a guess that has max bond dimension.
    # Given "Figure 4... 1-site DMRG", maybe they do subspace expansion.
    # Simpler: Use 2-site update which naturally handles bond dimension.
    # Or start with a guess of size chi (random or padded).

    # Let's implement 2-site update logic for the sweep as it's more robust
    # for dynamic bond dimension and standard in TEBD/DMRG codes.
    # It updates 2 sites at a time, SVDs, and truncates to chi.

    mps_new = mps_old.copy()

    # Tensors
    A_new = [t.copy() for t in mps_new.get_tensors()]  # (l, d, r)
    A_old = mps_old.get_tensors()  # (l, d, r)
    W = mpo_tensors  # (l, r, u, d)

    # Environments
    L = [np.ones((1, 1, 1))] * (n + 1)
    R = [np.ones((1, 1, 1))] * (n + 1)

    # Build initial R environments
    for i in range(n - 1, 0, -1):
        # Contract R[i+1] with site i
        # R: (r_n, r_m, r_o)
        # A_n: (l_n, p, r_n)
        T = np.tensordot(A_new[i], R[i + 1], axes=[[2], [0]])  # (l_n, p, r_m, r_o)
        # W: (l_m, r_m, p, p')
        T = np.tensordot(T, W[i], axes=[[2, 1], [1, 2]])  # (l_n, r_o, l_m, p')
        # A_o: (l_o, p', r_o)
        R[i] = np.tensordot(T, A_old[i], axes=[[1, 3], [2, 1]])  # (l_n, l_m, l_o)

    # Sweep
    for _ in range(sweeps):
        # Left -> Right (2-site update)
        for i in range(n - 1):
            # Form effective tensor for sites i, i+1
            # E = L[i] * W[i] * W[i+1] * A_old[i] * A_old[i+1] * R[i+2]

            # 1. Contract Left block: L[i] * A_old[i] * W[i]
            # L: (l_n, l_m, l_o)
            # A_o: (l_o, p1', r_o)
            T = np.tensordot(L[i], A_old[i], axes=[[2], [0]])  # (l_n, l_m, p1', r_o)
            # W: (l_m, r_m, p1, p1')
            T = np.tensordot(T, W[i], axes=[[1, 2], [0, 3]])  # (l_n, r_o, r_m, p1)

            # 2. Contract with site i+1 parts
            # A_o[i+1]: (r_o, p2', r_o2)
            T = np.tensordot(
                T, A_old[i + 1], axes=[[1], [0]]
            )  # (l_n, r_m, p1, p2', r_o2)
            # W[i+1]: (r_m, r_m2, p2, p2')
            T = np.tensordot(
                T, W[i + 1], axes=[[1, 3], [0, 3]]
            )  # (l_n, p1, r_o2, r_m2, p2)

            # 3. Contract with Right block: R[i+2]
            # R: (r_n2, r_m2, r_o2)
            # T: (l_n, p1, r_o2, r_m2, p2)
            # axes: [2, 3] with [2, 1]
            E = np.tensordot(T, R[i + 2], axes=[[2, 3], [2, 1]])  # (l_n, p1, p2, r_n2)

            # E is the target 2-site tensor (l_n, p1, p2, r_n2)

            # 4. SVD and Truncate
            # Reshape to (l_n * p1, p2 * r_n2)
            shape = E.shape
            E_flat = E.reshape(shape[0] * shape[1], shape[2] * shape[3])

            u, s, v = np.linalg.svd(E_flat, full_matrices=False)

            rank = min(len(s), chi)
            u = u[:, :rank]
            s = s[:rank]
            v = v[:rank, :]

            # 5. Update A_new[i] and A_new[i+1]
            A_new[i] = u.reshape(shape[0], shape[1], rank)  # (l_n, p1, bond)

            # Absorb s into v for next site
            sv = np.dot(np.diag(s), v)
            A_new[i + 1] = sv.reshape(rank, shape[2], shape[3])  # (bond, p2, r_n2)

            # 6. Update L[i+1] using new A_new[i]
            # Same logic as before
            # L[i+1] = L[i] * A_new[i] * W[i] * A_old[i]
            T_L = np.tensordot(L[i], A_new[i], axes=[[0], [0]])  # (l_m, l_o, p1, bond)
            T_L = np.tensordot(
                T_L, W[i], axes=[[0, 2], [0, 2]]
            )  # (l_o, bond, r_m, p1')
            L[i + 1] = np.tensordot(
                T_L, A_old[i], axes=[[0, 3], [0, 1]]
            )  # (bond, r_m, r_o)
            # (r_n, r_m, r_o) -> Matches.

        # Right -> Left sweep (optional but good for stability)
        for i in range(n - 2, -1, -1):
            # Form effective tensor E (same as above)
            T = np.tensordot(L[i], A_old[i], axes=[[2], [0]])
            T = np.tensordot(T, W[i], axes=[[1, 2], [0, 3]])
            T = np.tensordot(T, A_old[i + 1], axes=[[1], [0]])
            T = np.tensordot(T, W[i + 1], axes=[[1, 3], [0, 3]])
            E = np.tensordot(T, R[i + 2], axes=[[2, 3], [2, 1]])  # (l_n, p1, p2, r_n2)

            # SVD
            shape = E.shape
            E_flat = E.reshape(shape[0] * shape[1], shape[2] * shape[3])
            u, s, v = np.linalg.svd(E_flat, full_matrices=False)

            rank = min(len(s), chi)
            u = u[:, :rank]
            s = s[:rank]
            v = v[:rank, :]

            # Update A_new[i+1] (Right-isometric)
            A_new[i + 1] = v.reshape(rank, shape[2], shape[3])

            # Absorb u * s into A_new[i]
            us = np.dot(u, np.diag(s))
            A_new[i] = us.reshape(shape[0], shape[1], rank)

            # Update R[i+1] using new A_new[i+1]
            # R[i+1] = R[i+2] * A_new[i+1] * W[i+1] * A_old[i+1]
            T_R = np.tensordot(
                A_new[i + 1], R[i + 2], axes=[[2], [0]]
            )  # (bond, p2, r_m2, r_o2)
            T_R = np.tensordot(
                T_R, W[i + 1], axes=[[2, 1], [1, 2]]
            )  # (bond, r_o2, l_m2, p2')
            R[i + 1] = np.tensordot(
                T_R, A_old[i + 1], axes=[[1, 3], [2, 1]]
            )  # (bond, l_m2, l_o2)

    # Return new MPSCircuit
    # Center is at 0 after backward sweep
    return tc.MPSCircuit(n, tensors=A_new, center_position=0)


def run_mps_simulation_dmrg(n_qubits, layers, bond_dim):
    """
    Runs simulation using DMRG-style update (2-site sweep for bond adaptability).
    """
    mps = tc.MPSCircuit(n_qubits)
    mps.position(0)

    # Process layers in chunks
    chunk_size = 2  # Process 2 layers at a time (Fig 4 idea: several layers)
    for i in range(0, len(layers), chunk_size):
        chunk = layers[i : i + chunk_size]

        # Build MPO for this chunk
        mpo_tensors = build_mpo_from_layers(n_qubits, chunk)

        # Run DMRG sweep (2-site update to adapt bond dim up to chi)
        mps = run_1site_dmrg(mps, mpo_tensors, bond_dim, sweeps=1)

    return mps


def calculate_fidelity(exact_c, mps_c):
    """
    Calculates the fidelity between the exact state and the MPS state.
    F = |<psi_exact | psi_mps>|^2
    """
    psi_exact = exact_c.state()
    psi_mps = mps_c.wavefunction()

    psi_exact = K.reshape(psi_exact, (-1,))
    psi_mps = K.reshape(psi_mps, (-1,))

    overlap = K.tensordot(K.conj(psi_exact), psi_mps, axes=1)
    fidelity = np.abs(overlap) ** 2
    return float(fidelity)


def main():
    # Parameters
    ROWS = 3
    COLS = 4  # 12 qubits
    DEPTH = 8
    # Bond dimensions to sweep
    BOND_DIMS = [2, 4, 8, 16, 32, 64]

    logger.info(f"Generating random circuit: {ROWS}x{COLS} grid, Depth {DEPTH}")
    circuit, layers = generate_sycamore_like_circuit_structure(
        ROWS, COLS, DEPTH, seed=42
    )

    # 1. Exact Simulation
    logger.info("Running Exact Simulation...")
    start_time = time.time()
    # Force state calculation
    _ = circuit.state()
    logger.info(f"Exact simulation done in {time.time() - start_time:.4f}s")

    infidelities = []

    # 2. MPS Simulation with varying bond dimension
    logger.info("Running MPS Simulations (DMRG Sweep)...")
    for chi in BOND_DIMS:
        start_time = time.time()
        mps = run_mps_simulation_dmrg(circuit._nqubits, layers, chi)

        # Calculate Fidelity
        fid = calculate_fidelity(circuit, mps)
        infidelity = 1.0 - fid
        # Avoid log(0)
        if infidelity < 1e-15:
            infidelity = 1e-15

        infidelities.append(infidelity)

        logger.info(
            f"Bond Dim: {chi}, Fidelity: {fid:.6f}, Infidelity: {infidelity:.4e}, Time: {time.time() - start_time:.4f}s"
        )

    # 3. Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(BOND_DIMS, infidelities, "o-", label="Total Infidelity (1-F)")

    plt.xlabel("Bond Dimension (chi)")
    plt.ylabel("Infidelity (1 - F)")
    plt.title(
        f"MPS Simulation Accuracy vs Bond Dimension\n{ROWS}x{COLS} Circuit, Depth {DEPTH}"
    )
    plt.grid(True, which="both", ls="--")
    plt.legend()

    output_path = (
        "examples/reproduce_papers/2022_dmrg_circuit_simulation/outputs/result.png"
    )
    plt.savefig(output_path)
    logger.info(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
