"""
Variational Quantum Eigensolver (VQE) for the Toric Code Model

This module implements VQE optimization for finding ground states of the
generalized 2D toric code Hamiltonian with open boundary conditions and provides comparison
among different ansätze: FLDC, GLDC, and FDC.

"""

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from itertools import product
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO_tf


def build_toric_hamiltonian(
    Lx: int, Ly: int, h: float, hx: float = 0.0, hz: float = 0.0
) -> Dict[str, float]:
    """
    Build the generalized 2D toric code Hamiltonian with open boundary conditions.

    The Hamiltonian is: H = -(1-h)∑A_v - (1-h)∑B_p - h∑(hx*X_i + hz*Z_i)
    where A_v are vertex operators (X products) and B_p are plaquette operators (Z products).

    Args:
        Lx: Number of plaquettes in x-direction
        Ly: Number of plaquettes in y-direction
        h: Magnetic field parameter controlling topological phase (0 to 1)
        hx: X-direction field strength (default: 0.0)
        hz: Z-direction field strength (default: 0.0)

    Returns:
        Dictionary mapping Pauli strings to coefficients {pauli_string: weight}
    """
    H = {}

    # Qubit layout: horizontal edges followed by vertical edges
    num_horizontal = (Lx + 1) * Ly
    num_vertical = Lx * (Ly + 1)
    num_qubits = num_horizontal + num_vertical

    print(f"System size: {Lx}x{Ly} lattice, {num_qubits} qubits")

    # Vertex operators A_v (X products on edges around each vertex)
    for i, j in product(range(Lx + 1), range(Ly + 1)):
        connected_edges = []

        # Top vertical edge
        if i > 0:
            edge_idx = (i - 1) * (Ly + 1) + j + num_horizontal
            connected_edges.append(edge_idx)

        # Bottom vertical edge
        if i < Lx:
            edge_idx = i * (Ly + 1) + j + num_horizontal
            connected_edges.append(edge_idx)

        # Left horizontal edge
        if j > 0:
            edge_idx = i * Ly + (j - 1)
            connected_edges.append(edge_idx)

        # Right horizontal edge
        if j < Ly:
            edge_idx = i * Ly + j
            connected_edges.append(edge_idx)

        if connected_edges:
            pauli_str = ["I"] * num_qubits
            for idx in connected_edges:
                pauli_str[idx] = "X"
            pauli_str = "".join(pauli_str)

            # Adjust weight for boundary vertices
            weight = 1.0
            if (i == 0 or i == Lx) and (j == 0 or j == Ly):
                weight = 0.5  # Corner vertices
            elif i == 0 or i == Lx or j == 0 or j == Ly:
                weight = 0.75  # Edge vertices

            H[pauli_str] = -weight * (1 - h)

    # Plaquette operators B_p (Z products on edges around each plaquette)
    for i, j in product(range(Lx), range(Ly)):
        top = i * Ly + j
        bottom = (i + 1) * Ly + j
        left = num_horizontal + i * (Ly + 1) + j
        right = left + 1

        pauli_str = ["I"] * num_qubits
        for idx in [top, bottom, left, right]:
            pauli_str[idx] = "Z"
        pauli_str = "".join(pauli_str)

        H[pauli_str] = -(1 - h)

    # External field terms
    for q in range(num_qubits):
        if hx != 0:
            x_str = ["I"] * num_qubits
            x_str[q] = "X"
            H["".join(x_str)] = -hx * h

        if hz != 0:
            z_str = ["I"] * num_qubits
            z_str[q] = "Z"
            H["".join(z_str)] = -hz * h

    print(f"Hamiltonian built: {len(H)} terms")
    return H


def get_plaquette_qubits(i: int, j: int, Lx: int, Ly: int) -> Dict[str, int]:
    """
    Get qubit indices for the four edges of plaquette (i,j).

    Args:
        i, j: Plaquette coordinates
        Lx, Ly: Lattice dimensions

    Returns:
        Dictionary with keys 'top', 'bottom', 'left', 'right' mapping to qubit indices
    """
    num_horizontal = (Lx + 1) * Ly

    return {
        "top": int(i * Ly + j),
        "bottom": int((i + 1) * Ly + j),
        "left": int(num_horizontal + i * (Ly + 1) + j),
        "right": int(num_horizontal + i * (Ly + 1) + j + 1),
    }


def building_block(circuit: tc.Circuit, params: tf.Variable, q1: int, q2: int) -> None:
    """
    Apply a parameterized two-qubit gate sequence.

    Gate sequence: RZ-RY-RZ-RZ-RXX-RYY-RZZ-RZ-RY-RZ-RZ

    Args:
        circuit: Quantum circuit
        params: 11 parameters for the gate sequence
        q1, q2: Qubit indices
    """
    circuit.RZ(q1, theta=params[0])
    circuit.RY(q2, theta=params[1])
    circuit.RZ(q1, theta=params[2])
    circuit.RZ(q2, theta=params[3])
    circuit.RXX(q1, q2, theta=params[4])
    circuit.RYY(q1, q2, theta=params[5])
    circuit.RZZ(q1, q2, theta=params[6])
    circuit.RZ(q1, theta=params[7])
    circuit.RY(q2, theta=params[8])
    circuit.RZ(q1, theta=params[9])
    circuit.RZ(q2, theta=params[10])


def fldc_claw_ansatz(
    params: tf.Variable, Lx: int, Ly: int, layers: int = 1
) -> tc.Circuit:
    """
    Fixed Layer Depth Claw (FLDC) ansatz for the toric code.

    Applies gate sequences to (left,bottom), (top,bottom), (right,bottom) pairs
    for each plaquette in sequence.

    Args:
        params: Variational parameters
        Lx, Ly: Lattice dimensions
        layers: Number of ansatz layers

    Returns:
        Quantum circuit
    """
    num_qubits = (Lx + 1) * Ly + Lx * (Ly + 1)
    circuit = tc.Circuit(num_qubits)

    param_idx = 0
    for _ in range(layers):
        for i in range(Lx):
            for j in range(Ly):
                qubits = get_plaquette_qubits(i, j, Lx, Ly)

                for pair in [
                    (qubits["left"], qubits["bottom"]),
                    (qubits["top"], qubits["bottom"]),
                    (qubits["right"], qubits["bottom"]),
                ]:
                    building_block(
                        circuit, params[param_idx : param_idx + 11], pair[0], pair[1]
                    )
                    param_idx += 11

    return circuit


def gldc_claw_ansatz(
    params: tf.Variable, Lx: int, Ly: int, layers: int = 1
) -> tc.Circuit:
    """
    Grouped Layer Depth Claw (GLDC) ansatz for the toric code.

    Groups gates by pair type: all (left,bottom), then all (top,bottom),
    then all (right,bottom).

    Args:
        params: Variational parameters
        Lx, Ly: Lattice dimensions
        layers: Number of ansatz layers, if not 1 (not FDC), set to num_qubits

    Returns:
        Quantum circuit
    """
    num_qubits = (Lx + 1) * Ly + Lx * (Ly + 1)

    if layers != 1:
        layers = num_qubits

    circuit = tc.Circuit(num_qubits)

    param_idx = 0
    for _ in range(layers):
        # Left-bottom pairs
        for i in range(Lx):
            for j in range(Ly):
                qubits = get_plaquette_qubits(i, j, Lx, Ly)
                building_block(
                    circuit,
                    params[param_idx : param_idx + 11],
                    qubits["left"],
                    qubits["bottom"],
                )
                param_idx += 11

        # Top-bottom pairs
        for i in range(Lx):
            for j in range(Ly):
                qubits = get_plaquette_qubits(i, j, Lx, Ly)
                building_block(
                    circuit,
                    params[param_idx : param_idx + 11],
                    qubits["top"],
                    qubits["bottom"],
                )
                param_idx += 11

        # Right-bottom pairs
        for i in range(Lx):
            for j in range(Ly):
                qubits = get_plaquette_qubits(i, j, Lx, Ly)
                building_block(
                    circuit,
                    params[param_idx : param_idx + 11],
                    qubits["right"],
                    qubits["bottom"],
                )
                param_idx += 11

    return circuit


def fdc_claw_ansatz(params: tf.Variable, Lx: int, Ly: int) -> tc.Circuit:
    """Fixed Depth Claw (FDC) ansatz - single layer GLDC."""
    return gldc_claw_ansatz(params, Lx, Ly, layers=1)


def build_sparse_hamiltonian(
    hamiltonian_dict: Dict[str, float],
) -> PauliStringSum2COO_tf:
    """
    Convert Hamiltonian dictionary to sparse matrix representation.

    Args:
        hamiltonian_dict: Hamiltonian as {pauli_string: coefficient}

    Returns:
        Sparse Hamiltonian in COO format
    """
    pauli_strings = list(hamiltonian_dict.keys())
    coefficients = list(hamiltonian_dict.values())

    pauli_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    pauli_sequences = [[pauli_map[p] for p in ps] for ps in pauli_strings]

    return PauliStringSum2COO_tf(pauli_sequences, weight=coefficients)


def energy_expectation(state: tf.Tensor, hamiltonian_dict: Dict[str, float]) -> float:
    """
    Calculate energy expectation value <ψ|H|ψ>.

    Args:
        state: Quantum state vector
        hamiltonian_dict: Hamiltonian dictionary

    Returns:
        Energy expectation value
    """
    sparse_hamiltonian = build_sparse_hamiltonian(hamiltonian_dict)

    H_psi = tc.backend.sparse_dense_matmul(
        sparse_hamiltonian, tc.backend.reshape(state, [-1, 1])
    )
    expectation = tc.backend.sum(tc.backend.conj(state) * H_psi[:, 0])

    return tc.backend.real(expectation)


def toric_energy(
    c: tc.Circuit, h: float, hx: float, hz: float, Lx: int, Ly: int
) -> float:
    """Calculate toric code energy for a given circuit state."""
    H = build_toric_hamiltonian(Lx=Lx, Ly=Ly, h=h, hx=hx, hz=hz)
    return tc.backend.real(energy_expectation(c.state(), H))


def vqe_toric_fldc(
    params: tf.Variable,
    Lx: int,
    Ly: int,
    h: float,
    hx: float,
    hz: float,
    num_layers: int = 1,
) -> float:
    """VQE loss function for FLDC ansatz."""
    c = fldc_claw_ansatz(params, Lx, Ly, layers=num_layers)
    return toric_energy(c, h, hx, hz, Lx, Ly)


def vqe_toric_gldc(
    params: tf.Variable,
    Lx: int,
    Ly: int,
    h: float,
    hx: float,
    hz: float,
    num_layers: int = 3,
) -> float:
    """VQE loss function for GLDC ansatz."""
    c = gldc_claw_ansatz(params, Lx, Ly, layers=num_layers)
    return toric_energy(c, h, hx, hz, Lx, Ly)


def vqe_toric_fdc(
    params: tf.Variable,
    Lx: int,
    Ly: int,
    h: float,
    hx: float,
    hz: float,
    num_layers: int = 1,
) -> float:
    """VQE loss function for FDC ansatz. num_layers is unused."""
    c = fdc_claw_ansatz(params, Lx, Ly)
    return toric_energy(c, h, hx, hz, Lx, Ly)


def train_vqe(
    vqe_vag,
    Lx: int,
    Ly: int,
    maxiter: int,
    h: float,
    hx: float,
    hz: float,
    num_params: int,
    num_layers: int = 1,
    seed: int = 0,
    lr: float = 1e-2,
) -> np.ndarray:
    """
    Train VQE model.

    Args:
        vqe_vag: JIT-compiled value_and_grad function
        Lx, Ly: Lattice dimensions
        maxiter: Maximum iterations
        h, hx, hz: Hamiltonian parameters
        num_params: Number of variational parameters
        num_layers: Number of ansatz layers (for FLDC, set to 1 for FDC and set to num_qubits for GLDC)
        seed: Random seed
        lr: Learning rate

    Returns:
        Array of energy values during training
    """
    energies = []
    params = tf.Variable(
        tf.random.uniform(
            shape=[num_params, 1],
            minval=0.0,
            maxval=2 * np.pi,
            dtype=getattr(tf, tc.rdtypestr),
            seed=seed,
        )
    )

    optimizer = tf.keras.optimizers.Adam(lr)

    for i in range(maxiter):
        energy, grad = vqe_vag(params, Lx, Ly, h, hx, hz, num_layers=num_layers)
        optimizer.apply_gradients([(grad, params)])
        energies.append(energy)

        if i % 200 == 0:
            print(f"Iteration {i}: Energy = {energy:.6f}")

    return np.array(energies)


def run_vqe_trials(
    ansatz_type: str,
    Lx: int,
    Ly: int,
    maxiter: int,
    h: float,
    hx: float,
    hz: float,
    fldc_num_layers: int = 1,
    trials: int = 100,
    avg_percent: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """
    Run multiple VQE trials and return averaged results.

    Args:
        ansatz_type: 'fldc', 'gldc', or 'fdc'
        Lx, Ly: Lattice dimensions
        maxiter: Maximum iterations per trial
        h, hx, hz: Hamiltonian parameters
        fldc_num_layers: Number of FLDC ansatz layers
        trials: Number of random trials
        avg_percent: Fraction of best trials to average

    Returns:
        (best_energy_avg, avg_energy_trajectory)
    """
    # Setup based on ansatz type

    num_qubits = (Lx + 1) * Ly + Lx * (Ly + 1)

    if ansatz_type == "fldc":
        vqe_func = vqe_toric_fldc
        num_params = Lx * Ly * 11 * 3 * fldc_num_layers
    elif ansatz_type == "gldc":
        vqe_func = vqe_toric_gldc
        num_params = Lx * Ly * 11 * 3 * num_qubits
    elif ansatz_type == "fdc":
        vqe_func = vqe_toric_fdc
        num_params = Lx * Ly * 11 * 3
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")

    vqe_vag = tc.backend.jit(tc.backend.value_and_grad(vqe_func), static_argnums=(1, 2))

    final_energies = []
    avg_energies = np.zeros(maxiter)

    for trial in range(trials):
        print(f"\nTrial {trial+1}/{trials}")
        energies = train_vqe(
            vqe_vag, Lx, Ly, maxiter, h, hx, hz, num_params, fldc_num_layers, seed=trial
        )
        final_energies.append(energies[-1])
        avg_energies += energies / trials
        print(f"Final energy: {energies[-1]:.6f}")

    # Average best trials
    final_energies = np.array(final_energies)
    best_indices = np.argsort(final_energies)[: int(trials * avg_percent)]
    best_energy = final_energies[best_indices].mean()

    return best_energy, avg_energies


def compute_exact_ground_energy(
    Lx: int, Ly: int, h: float, hx: float, hz: float
) -> float:
    """
    Compute exact ground state energy using sparse diagonalization.

    Args:
        Lx, Ly: Lattice dimensions
        h, hx, hz: Hamiltonian parameters

    Returns:
        Ground state energy
    """
    H = build_toric_hamiltonian(Lx=Lx, Ly=Ly, h=h, hx=hx, hz=hz)
    sparse_H = build_sparse_hamiltonian(H)

    indices = sparse_H.indices.numpy()
    values = sparse_H.values.numpy()
    shape = sparse_H.dense_shape.numpy()

    coo_mat = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape)
    csr_mat = coo_mat.tocsr()

    eigenvalues, _ = eigs(csr_mat, k=1, which="SR")
    return eigenvalues[0].real


def main():
    """Main execution function."""
    # Configuration
    tc.set_backend("tensorflow")
    tc.set_dtype("complex128")

    print(f"Backend: {tc.backend.name}")
    print(f"Dtype: {tc.dtypestr} (real: {tc.rdtypestr})")

    # Parameters
    Lx, Ly = 2, 2
    fldc_num_layers = 3
    maxiter = 1000
    # For each ansatz, there are 100 independent VQE trials
    # And the data are averaged over the best half of the 100 training trajectories starting from different initializations.
    # It's time consuming to run GLDC ansatz with 100 trials; consider reducing trials for testing.
    # And GLDC ansatz requires NUM_QUBITS layers FDC structure, which may lead to very long training time.
    trials = 100
    avg_percent = 0.5
    h_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    num_qubits = (Lx + 1) * Ly + Lx * (Ly + 1)

    # Storage for results
    results = {"ed": [], "fldc": [], "gldc": [], "fdc": []}

    # Run experiments
    for h in h_values:
        print(f"\n{'='*60}")
        print(f"Training for h={h:.1f}")
        print(f"{'='*60}")

        # Exact diagonalization
        ground_energy = compute_exact_ground_energy(Lx, Ly, h, h, h)
        print(f"Exact ground energy: {ground_energy:.6f}")
        results["ed"].append(ground_energy)

        # VQE with different ansätze
        for ansatz in ["fldc", "gldc", "fdc"]:
            print(f"\n--- {ansatz.upper()} Ansatz ---")
            best_energy, avg_traj = run_vqe_trials(
                ansatz,
                Lx,
                Ly,
                maxiter,
                h,
                h,
                h,
                fldc_num_layers=fldc_num_layers if ansatz != "fdc" else 1,
                trials=trials,
                avg_percent=avg_percent,
            )
            results[ansatz].append(best_energy)

            # Save first trajectory for plotting
            if h == h_values[0]:
                results[f"{ansatz}_traj"] = (avg_traj - ground_energy) / num_qubits

    # Convert to arrays
    for key in results:
        if not key.endswith("_traj"):
            results[key] = np.array(results[key])

    # Plot error convergence
    plt.figure(figsize=(10, 6))
    for ansatz in ["fdc", "fldc", "gldc"]:
        if f"{ansatz}_traj" in results:
            plt.plot(results[f"{ansatz}_traj"], label=ansatz.upper())
    plt.xlabel("Iteration")
    plt.ylabel("Error per qubit")
    plt.title(f"Training Error for h={h_values[0]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_error.png", dpi=300)
    plt.show()

    # Plot energy landscape
    plt.figure(figsize=(10, 6))
    h_array = np.array(h_values)
    plt.plot(h_array, results["ed"] / num_qubits, "o-", label="Exact", linewidth=2)
    plt.plot(h_array, results["fdc"] / num_qubits, "s-", label="FDC")
    plt.plot(h_array, results["fldc"] / num_qubits, "^-", label="FLDC")
    plt.plot(h_array, results["gldc"] / num_qubits, "d-", label="GLDC")
    plt.xlabel("Magnetic field h")
    plt.ylabel("Energy per qubit (E/N)")
    plt.title("Ground State Energy vs Magnetic Field")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_landscape.png", dpi=300)
    plt.show()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
