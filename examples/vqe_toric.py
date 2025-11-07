"""
Variational Quantum Eigensolver (VQE) for the Toric Code Model

This module implements VQE optimization for finding ground states of the
generalized 2D toric code Hamiltonian with open boundary conditions and provides comparison
among different ansätze: FLDC, GLDC, and FDC.

"""

from itertools import product
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO


def build_toric_hamiltonian(
    Lx: int, Ly: int, h: float, hx: float = 0.0, hz: float = 0.0
) -> Dict[str, float]:
    r"""
    Build the generalized 2D toric code Hamiltonian with open boundary conditions.

    The Hamiltonian is: H = - (1 - h) \sum A_v - (1 - h) \sum B_p - h \sum (hx * X_i + hz * Z_i)
    where A_v are vertex operators (X products) and B_p are plaquette operators (Z products).

    Plaquette Operators at Boundaries:

    In the toric code with open boundary conditions, each plaquette operator B_p acts on the
    four edges surrounding a single plaquette. For interior plaquettes, all four edges exist.
    However, at the boundaries, the treatment remains consistent:

    - Each plaquette (i,j) is defined for i in [0, Lx-1] and j in [0, Ly-1]
    - The four edges (top, bottom, left, right) are always well-defined within the lattice
    - Boundary plaquettes still have four edges, but some of these edges may be at the physical boundary
    - No special weighting is applied to boundary plaquettes (unlike vertex operators, which are weighted as 0.5 or 0.75)

    The qubit indices for each edge are calculated as:
    - Horizontal edges: indexed sequentially along rows
    - Vertical edges: indexed after all horizontal edges
    - Top edge: i * Ly + j
    - Bottom edge: (i + 1) * Ly + j
    - Left edge: num_horizontal + i * (Ly + 1) + j
    - Right edge: left + 1

    :param Lx: Number of plaquettes in x-direction
    :type Lx: int
    :param Ly: Number of plaquettes in y-direction
    :type Ly: int
    :param h: Magnetic field parameter controlling topological phase (0 to 1)
    :type h: float
    :param hx: X-direction field strength (default: 0.0)
    :type hx: float
    :param hz: Z-direction field strength (default: 0.0)
    :type hz: float
    :return: Hamiltonian as a dictionary mapping Pauli strings to coefficients
    :rtype: Dict[str, float]
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

    :param i: Plaquette x-coordinate
    :type i: int
    :param j: Plaquette y-coordinate
    :type j: int
    :param Lx: Lattice width
    :type Lx: int
    :param Ly: Lattice height
    :type Ly: int
    :return: Dictionary with keys 'top', 'bottom', 'left', 'right' mapping to qubit indices
    :rtype: Dict[str, int]
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

    :param circuit: Quantum circuit
    :type circuit: tc.Circuit
    :param params: 11 parameters for the gate sequence
    :type params: tf.Variable
    :param q1: Qubit index 1
    :type q1: int
    :param q2: Qubit index 2
    :type q2: int
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


def building_block_su4(
    circuit: tc.Circuit, params: tf.Variable, q1: int, q2: int
) -> None:
    """
    Apply a parameterized two-qubit SU(4) gate.

    :param circuit: Quantum circuit
    :type circuit: tc.Circuit
    :param params: 15 parameters for the SU(4) gate
    :type params: tf.Variable
    :param q1: Qubit index 1
    :type q1: int
    :param q2: Qubit index 2
    :type q2: int
    """
    circuit.SU4(q1, q2, theta=params)
    # su4_gate = tc.gates.su4_gate(params)
    # circuit.any(su4_gate, q1, q2)


def fldc_claw_ansatz(
    params: tf.Variable, Lx: int, Ly: int, layers: int = 1
) -> tc.Circuit:
    """
    Finite local depth circuit (FLDC) ansatz for the toric code.

    Applies gate sequences to (left,bottom), (top,bottom), (right,bottom) pairs
    for each plaquette in sequence.

    :param params: Variational parameters
    :type params: tf.Variable
    :param Lx: Lattice width
    :type Lx: int
    :param Ly: Lattice height
    :type Ly: int
    :param layers: Number of ansatz layers
    :type layers: int
    :return: Quantum circuit
    :rtype: tc.Circuit
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
                    building_block_su4(
                        circuit, params[param_idx : param_idx + 15], pair[0], pair[1]
                    )
                    param_idx += 15

    return circuit


def gldc_claw_ansatz(
    params: tf.Variable, Lx: int, Ly: int, layers: int = 1
) -> tc.Circuit:
    """
    Global linear depth circuit (GLDC) ansatz for the toric code.

    Groups gates by pair type: all (left,bottom), then all (top,bottom),
    then all (right,bottom).

    :param params: Variational parameters
    :type params: tf.Variable
    :param Lx: Lattice width
    :type Lx: int
    :param Ly: Lattice height
    :type Ly: int
    :param layers: Number of ansatz layers, if not 1 (not FDC), set to num_qubits (which means full GLDC)
    :type layers: int
    :return: Quantum circuit
    :rtype: tc.Circuit
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
                building_block_su4(
                    circuit,
                    params[param_idx : param_idx + 15],
                    qubits["left"],
                    qubits["bottom"],
                )
                param_idx += 15

        # Top-bottom pairs
        for i in range(Lx):
            for j in range(Ly):
                qubits = get_plaquette_qubits(i, j, Lx, Ly)
                building_block_su4(
                    circuit,
                    params[param_idx : param_idx + 15],
                    qubits["top"],
                    qubits["bottom"],
                )
                param_idx += 15

        # Right-bottom pairs
        for i in range(Lx):
            for j in range(Ly):
                qubits = get_plaquette_qubits(i, j, Lx, Ly)
                building_block_su4(
                    circuit,
                    params[param_idx : param_idx + 15],
                    qubits["right"],
                    qubits["bottom"],
                )
                param_idx += 15

    return circuit


def fdc_claw_ansatz(params: tf.Variable, Lx: int, Ly: int) -> tc.Circuit:
    """Finite depth circuit (FDC) ansatz - single layer GLDC."""
    return gldc_claw_ansatz(params, Lx, Ly, layers=1)


def build_sparse_hamiltonian(hamiltonian_dict: Dict[str, float]) -> PauliStringSum2COO:
    """
    Convert Hamiltonian dictionary to sparse matrix representation.

    :param hamiltonian_dict: Hamiltonian as {pauli_string: coefficient}
    :type hamiltonian_dict: Dict[str, float]

    :return: Sparse Hamiltonian in COO format
    :rtype: PauliStringSum2COO
    """
    pauli_strings = list(hamiltonian_dict.keys())
    coefficients = list(hamiltonian_dict.values())

    pauli_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    pauli_sequences = [[pauli_map[p] for p in ps] for ps in pauli_strings]

    return PauliStringSum2COO(pauli_sequences, weight=coefficients)


def energy_calc(c: tc.Circuit, H: PauliStringSum2COO) -> float:
    """Calculate energy expectation value for a given circuit and Hamiltonian."""
    return tc.templates.measurements.operator_expectation(c, H)


def vqe_func(
    params: tf.Variable,
    ansatz_type: str,
    H: PauliStringSum2COO,
    Lx: int,
    Ly: int,
    h: float,
    hx: float,
    hz: float,
    num_layers: int = 1,
) -> float:
    """VQE loss function for FLDC/GLDC/FDC ansatz."""
    if ansatz_type == "fldc":
        c = fldc_claw_ansatz(params, Lx, Ly, layers=num_layers)
    elif ansatz_type == "gldc":
        c = gldc_claw_ansatz(params, Lx, Ly, layers=num_layers)
    elif ansatz_type == "fdc":
        c = fdc_claw_ansatz(params, Lx, Ly)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    return energy_calc(c, H)


def train_vqe(
    vqe_vag,
    ansatz_type: str,
    H: PauliStringSum2COO,
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

    :param vqe_vag: JIT-compiled value_and_grad function
    :param Lx: Lattice width
    :param Ly: Lattice height
    :param maxiter: Maximum iterations
    :param h: Hamiltonian parameter
    :param hx: Hamiltonian parameter
    :param hz: Hamiltonian parameter
    :param num_params: Number of variational parameters
    :param num_layers: Number of ansatz layers (for FLDC, set to 1 for FDC and set to num_qubits for GLDC)
    :param seed: Random seed
    :param lr: Learning rate
    :return: Array of energy values during training
    :rtype: np.ndarray
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
        energy, grad = vqe_vag(
            params, ansatz_type, H, Lx, Ly, h, hx, hz, num_layers=num_layers
        )
        optimizer.apply_gradients([(grad, params)])
        energies.append(energy)

        if i % 200 == 0:
            print(f"Iteration {i}: Energy = {energy:.6f}")

    return np.array(energies)


def run_vqe_trials(
    ansatz_type: str,
    H: PauliStringSum2COO,
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

    :param ansatz_type: 'fldc', 'gldc', or 'fdc'
    :param Lx: Lattice width
    :param Ly: Lattice height
    :param maxiter: Maximum iterations per trial
    :param h: Hamiltonian parameter
    :param hx: Hamiltonian parameter
    :param hz: Hamiltonian parameter
    :param fldc_num_layers: Number of FLDC ansatz layers
    :param trials: Number of random trials
    :param avg_percent: Fraction of best trials to average
    :return: (best_energy_avg, avg_energy_trajectory)
    :rtype: Tuple[float, np.ndarray]
    """

    # Setup based on ansatz type
    num_qubits = (Lx + 1) * Ly + Lx * (Ly + 1)

    if ansatz_type == "fldc":
        num_params = Lx * Ly * 15 * 3 * fldc_num_layers
    elif ansatz_type == "gldc":
        num_params = Lx * Ly * 15 * 3 * num_qubits
    elif ansatz_type == "fdc":
        num_params = Lx * Ly * 15 * 3
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")

    vqe_vag = tc.backend.jit(
        tc.backend.value_and_grad(vqe_func), static_argnums=(1, 2, 3, 4)
    )

    final_energies = []
    avg_energies = np.zeros(maxiter)

    for trial in range(trials):
        print(f"\nTrial {trial+1}/{trials}")
        energies = train_vqe(
            vqe_vag,
            ansatz_type=ansatz_type,
            H=H,
            Lx=Lx,
            Ly=Ly,
            maxiter=maxiter,
            h=h,
            hx=hx,
            hz=hz,
            num_params=num_params,
            num_layers=fldc_num_layers,
            seed=trial,
        )
        final_energies.append(energies[-1])
        avg_energies += energies / trials
        print(f"Final energy: {energies[-1]:.6f}")

    # Average best trials
    final_energies = np.array(final_energies)
    best_indices = np.argsort(final_energies)[: int(trials * avg_percent)]
    best_energy = final_energies[best_indices].mean()

    return best_energy, avg_energies


def compute_exact_ground_energy(sparse_H: PauliStringSum2COO) -> float:
    """
    Compute exact ground state energy using sparse diagonalization.

    :param sparse_H: Sparse Hamiltonian in COO format
    :type sparse_H: PauliStringSum2COO
    :return: Exact ground state energy
    :rtype: float
    """
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

        H = build_toric_hamiltonian(Lx=Lx, Ly=Ly, h=h, hx=h, hz=h)
        sparse_H = build_sparse_hamiltonian(H)

        # Exact diagonalization
        ground_energy = compute_exact_ground_energy(sparse_H)
        print(f"Exact ground energy: {ground_energy:.6f}")
        results["ed"].append(ground_energy)

        # VQE with different ansätze
        for ansatz in ["fldc", "gldc", "fdc"]:
            print(f"\n--- {ansatz.upper()} Ansatz ---")
            best_energy, avg_traj = run_vqe_trials(
                ansatz,
                sparse_H,
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

    h_array = np.array(h_values)

    # Define a color map for consistency
    # Using Matplotlib default color cycle (C0: blue, C1: orange, C2: green, C3: red)
    color_map = {"Exact": "C0", "FDC": "C1", "FLDC": "C2", "GLDC": "C3"}

    fig, ax_main = plt.subplots(figsize=(10, 6))

    # Plot main graph: Energy per qubit vs h
    ax_main.plot(
        h_array,
        results["ed"] / num_qubits,
        "o-",
        label="Exact",
        linewidth=2,
        color=color_map["Exact"],
    )
    ax_main.plot(
        h_array, results["fdc"] / num_qubits, "s-", label="FDC", color=color_map["FDC"]
    )
    ax_main.plot(
        h_array,
        results["fldc"] / num_qubits,
        "^-",
        label="FLDC",
        color=color_map["FLDC"],
    )
    ax_main.plot(
        h_array,
        results["gldc"] / num_qubits,
        "d-",
        label="GLDC",
        color=color_map["GLDC"],
    )

    ax_main.set_xlabel("Magnetic field h")
    ax_main.set_ylabel("Energy per qubit (E/N)")
    ax_main.set_title("Ground State Energy vs Magnetic Field")
    ax_main.legend()
    ax_main.grid(False)

    # Plot inset graph: Error convergence for h=0.1
    # To control the position of the inset, you can adjust these values to change the position and size of the inset
    inset_pos = [0.72, 0.63, 0.25, 0.25]  # [left x, bottom y, width, height]
    ax_inset = fig.add_axes(inset_pos)

    for ansatz in ["fdc", "fldc", "gldc"]:
        if f"{ansatz}_traj" in results:
            ansatz_upper = ansatz.upper()
            ax_inset.plot(
                results[f"{ansatz}_traj"],
                label=ansatz_upper,
                color=color_map[ansatz_upper],
            )

    ax_inset.set_xlabel("Iteration", fontsize=10)  # 使用稍小的字体
    ax_inset.set_ylabel("Error per qubit", fontsize=10)
    ax_inset.set_title(f"Training Error (h={h_values[0]})", fontsize=11)
    ax_inset.grid(False)
    ax_inset.tick_params(axis="both", which="major", labelsize=8)

    # plt.savefig('combined_energy_plot.png', dpi=300)
    plt.show()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
