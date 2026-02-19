"""
Reproduction of "A density-matrix renormalization group algorithm for simulating
quantum circuits with a finite fidelity"
Link: https://arxiv.org/abs/2207.05612

Description:
This script reproduces Figure 2(a) from the paper.
It simulates a Sycamore-like random quantum circuit using both exact state vector simulation
and an MPS-based simulator (DMRG-like algorithm) with varying bond dimensions.
The script plots the infidelity (1 - Fidelity) as a function of the bond dimension.
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


def generate_sycamore_like_circuit(rows, cols, depth, seed=42):
    """
    Generates a random quantum circuit with a structure similar to Sycamore circuits.
    Since real Sycamore gates are specific, we use a simplified model:
    - 2D Grid connectivity
    - Layers of random single-qubit gates
    - Layers of two-qubit gates (CZ or similar) in a specific pattern
    """
    np.random.seed(seed)
    n_qubits = rows * cols
    c = tc.Circuit(n_qubits)

    def q(r, col):
        return r * cols + col

    for d in range(depth):
        # Single qubit gates
        for i in range(n_qubits):
            # Random single qubit gate (e.g., Rx, Ry, Rz)
            # Simplification: Use a general random unitary or specific rotations
            # Sycamore uses sqrt(X), sqrt(Y), sqrt(W)
            # We'll use random rotations for generic RQC behavior
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            lam = np.random.uniform(0, 2 * np.pi)
            c.rz(i, theta=phi)
            c.ry(i, theta=theta)
            c.rz(i, theta=lam)

        # Two-qubit gates
        # Sycamore uses a specific pattern (ABCD...)
        # We will use a simple alternating pattern
        # Layer A: horizontal even
        # Layer B: horizontal odd
        # Layer C: vertical even
        # Layer D: vertical odd
        # We cycle through these patterns based on depth

        layer_type = d % 4

        if layer_type == 0:  # Horizontal (col, col+1) for even cols
            for r in range(rows):
                for col in range(0, cols - 1, 2):
                    c.cz(q(r, col), q(r, col + 1))
        elif layer_type == 1:  # Horizontal (col, col+1) for odd cols
            for r in range(rows):
                for col in range(1, cols - 1, 2):
                    c.cz(q(r, col), q(r, col + 1))
        elif layer_type == 2:  # Vertical (row, row+1) for even rows
            for col in range(cols):
                for r in range(0, rows - 1, 2):
                    c.cz(q(r, col), q(r + 1, col))
        elif layer_type == 3:  # Vertical (row, row+1) for odd rows
            for col in range(cols):
                for r in range(1, rows - 1, 2):
                    c.cz(q(r, col), q(r + 1, col))

    return c


def run_mps_simulation(c, bond_dim):
    """
    Runs the simulation using MPSCircuit with a maximum bond dimension.
    """
    # Create MPSCircuit from the circuit operations
    # We need to re-apply the gates to an MPSCircuit
    # Or we can just build the MPSCircuit directly in the generation function.
    # But to ensure exactly the same circuit, we can iterate over the qir of the tc.Circuit
    n = c._nqubits
    mps = tc.MPSCircuit(n)

    # Set truncation rules
    # We use `max_singular_values` to control the bond dimension (chi)
    mps.set_split_rules({"max_singular_values": bond_dim})

    for gate in c._qir:
        index = gate["index"]
        params = gate.get("parameters", {})

        # Construct the gate on MPS
        # tc.Circuit stores 'gatef' which returns a Gate object when called with params
        g_obj = gate["gatef"](**params)

        # Apply to MPS
        mps.apply(g_obj, *index)

    return mps


def calculate_fidelity(exact_c, mps_c):
    """
    Calculates the fidelity between the exact state and the MPS state.
    F = |<psi_exact | psi_mps>|^2
    """
    # Get exact state vector
    psi_exact = exact_c.state()

    # Get MPS state vector (converted to full tensor)
    # Note: For large N, this will OOM. We should keep N small (e.g. <= 20).
    psi_mps = mps_c.wavefunction()

    # Compute overlap
    # exact state is (2^N,) or (1, 2^N)
    # mps state is (1, 2^N) usually

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
    # For 12 qubits, full bond dimension is 2^6=64.
    # So we should see perfect fidelity at 64, and error below it.
    BOND_DIMS = [2, 4, 8, 16, 32, 64]

    logger.info(f"Generating random circuit: {ROWS}x{COLS} grid, Depth {DEPTH}")
    circuit = generate_sycamore_like_circuit(ROWS, COLS, DEPTH, seed=42)

    # 1. Exact Simulation
    logger.info("Running Exact Simulation...")
    start_time = time.time()
    # Force state calculation
    _ = circuit.state()
    logger.info(f"Exact simulation done in {time.time() - start_time:.4f}s")

    infidelities = []

    # 2. MPS Simulation with varying bond dimension
    logger.info("Running MPS Simulations...")
    for chi in BOND_DIMS:
        start_time = time.time()
        mps = run_mps_simulation(circuit, chi)

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
