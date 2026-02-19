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
This script implements a "layerwise DMRG" logic. Instead of truncating the MPS
after every 2-qubit gate (TEBD approach), we apply a full layer of gates to the
MPS (allowing the bond dimension to grow) and then perform a compression sweep
to truncate the MPS back to the target bond dimension. This variational-like
compression is closer to the DMRG-style algorithm described in the paper.
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


def run_mps_simulation_layerwise(n_qubits, layers, bond_dim):
    """
    Runs the simulation using MPSCircuit with layerwise DMRG-like compression.
    """
    mps = tc.MPSCircuit(n_qubits)

    # We want to manually control truncation
    # First, we set no truncation for gate application
    mps.set_split_rules({})  # Infinite bond dimension during application

    for layer in layers:
        # 1. Apply all gates in the layer
        # This will increase the bond dimension significantly
        for gate in layer:
            index = gate["index"]
            params = gate.get("parameters", {})
            g_obj = gate["gatef"](**params)
            mps.apply(g_obj, *index)

        # 2. Perform compression sweep (DMRG-style logic)
        # We sweep from left to right (and/or right to left) and truncate
        # the bonds to the target dimension `bond_dim`.

        # We use standard SVD-based compression (sweeping) which is optimal for minimizing 2-norm error
        # This is effectively what DMRG does when optimizing overlap for a fixed bond dimension.

        # Sweep Left -> Right
        # First ensure we are at the beginning
        mps.position(0)
        for i in range(n_qubits - 1):
            mps.reduce_dimension(
                i, center_left=False, split={"max_singular_values": bond_dim}
            )

        # Sweep Right -> Left (to ensure canonicalization and further optimization)
        # We are at n_qubits - 1 now.
        for i in range(n_qubits - 2, -1, -1):
            mps.reduce_dimension(
                i, center_left=True, split={"max_singular_values": bond_dim}
            )

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
    logger.info("Running MPS Simulations (Layerwise DMRG)...")
    for chi in BOND_DIMS:
        start_time = time.time()
        mps = run_mps_simulation_layerwise(circuit._nqubits, layers, chi)

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
