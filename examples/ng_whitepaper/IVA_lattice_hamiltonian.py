"""
IIIA_lattice_hamiltonian.py

This script demonstrates the advanced lattice and Hamiltonian generation capabilities
of TensorCircuit-NG, specifically designed for the whitepaper.

It covers three main aspects:
1.  **Lattice Generation**: Creating standard lattices (Square, Triangular, Kagome) and visualizing them.
2.  **Lattice Customization**: Creating a custom lattice with defects by manipulating site coordinates.
3.  **Automatic Differentiation (AD)**: Optimizing a geometric parameter (lattice constant) of a
    Rydberg atom array to minimize the ground state energy. This showcases the
    end-to-end differentiability of the lattice generation and Hamiltonian construction pipeline.
"""

import time
import matplotlib.pyplot as plt
import optax
import tensorcircuit as tc

# Use JAX backend for automatic differentiation and JIT compilation
tc.set_backend("jax")
tc.set_dtype("complex128")
K = tc.backend


def demo_lattice_generation():
    """
    Demonstrates the creation of various standard lattices.
    """
    print("\n--- Part 1: Standard Lattice Generation ---")

    # 1. Square Lattice - Precompute neighbors for visualization
    square = tc.templates.lattice.SquareLattice(
        size=(4, 4), pbc=True, precompute_neighbors=1
    )
    print(f"Square Lattice: {square.num_sites} sites")

    # 2. Triangular Lattice
    triangular = tc.templates.lattice.TriangularLattice(
        size=(4, 4), pbc=False, precompute_neighbors=1
    )
    print(f"Triangular Lattice: {triangular.num_sites} sites")

    # 3. Kagome Lattice
    kagome = tc.templates.lattice.KagomeLattice(
        size=(2, 2), pbc=False, precompute_neighbors=1
    )
    print(f"Kagome Lattice: {kagome.num_sites} sites")

    # Visualization
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    square.show(ax=axes[0], show_bonds_k=1)
    axes[0].set_title("Square Lattice (4x4)")

    triangular.show(ax=axes[1], show_bonds_k=1)
    axes[1].set_title("Triangular Lattice (4x4)")

    kagome.show(ax=axes[2], show_bonds_k=1)
    axes[2].set_title("Kagome Lattice (2x2, PBC)")

    plt.tight_layout()
    plt.savefig("lattice_types_demo.pdf")
    print("Saved lattice visualization to 'lattice_types_demo.pdf'")
    plt.close()


def demo_custom_lattice():
    """
    Demonstrates how to create a custom lattice by modifying an existing one.
    We create a square lattice and remove a site to simulate a defect.
    """
    print("\n--- Part 2: Custom Lattice with Defect ---")

    # Start with a standard 4x4 square lattice
    # Using pbc=False to match original demo
    base_lattice = tc.templates.lattice.SquareLattice(size=(4, 4), pbc=False)

    # Convert to CustomizeLattice to allow structural modifications
    custom_lattice = tc.templates.lattice.CustomizeLattice.from_lattice(base_lattice)

    # Let's remove site at index 5 (arbitrary choice for a "hole")
    hole_index = 5
    print(f"Removing site index {hole_index} to create a defect.")

    # Get the identifier for the site we want to remove
    hole_identifier = custom_lattice.get_identifier(hole_index)

    # Use the remove_sites API
    print(f"Original sites: {custom_lattice.num_sites}")
    custom_lattice.remove_sites([hole_identifier])
    print(f"New sites: {custom_lattice.num_sites}")

    # Visualize the defect
    # Visualize the defect
    # Create figure and axis explicitly to avoid blocking plt.show() in .show() method
    fig, ax = plt.subplots(figsize=(6, 6))
    custom_lattice.show(ax=ax, show_indices=True, show_bonds_k=1)
    ax.set_title("Custom Lattice with a Point Defect")

    plt.savefig("lattice_defect_demo.pdf")
    print("Saved defect lattice visualization to 'lattice_defect_demo.pdf'")
    plt.close(fig)


def demo_ad_optimization():
    """
    Demonstrates Automatic Differentiation (AD) through the lattice and Hamiltonian pipeline.
    We optimize the lattice constant 'a' of a Triangular lattice to minimize
    the ground state energy of a Rydberg Hamiltonian.
    """
    print("\n--- Part 3: AD Optimization of Lattice Constant ---")

    # Configuration
    # We use a small triangular cluster.
    nx, ny = 2, 4

    # Physical parameters for Rydberg Hamiltonian
    # H = Omega/2 * sum(X) - Delta/2 * sum(Z) + sum(V_ij * n_i * n_j)
    omega = 1.0
    delta = 1.0
    c6 = 10.0  # Interaction strength coefficient V_ij = C6 / R^6

    print(f"System: Triangular Lattice {nx}x{ny} ({nx*ny} sites)")
    print(f"Hamiltonian: Rydberg (Omega={omega}, Delta={delta}, C6={c6})")

    def get_ground_state_energy(a):
        """
        Computes the ground state energy for a given lattice constant 'a'.
        This function is designed to be fully differentiable and JIT-compatible.
        """
        lattice_constant = K.abs(a)  # Ensure positivity

        # 1. Construct Lattice with differentiable constant 'a'
        lattice = tc.templates.lattice.TriangularLattice(
            size=(nx, ny), lattice_constant=lattice_constant, pbc=False
        )

        # 2. Generate Rydberg Hamiltonian using the library function
        ham_sparse = tc.templates.hamiltonians.rydberg_hamiltonian(
            lattice, omega=omega, delta=delta, c6=c6
        )

        # 3. Convert to dense for eigendecomposition
        # Note: For larger systems, we would use sparse solvers or VQE,
        # but dense eigh is robust for AD on small systems.
        H_total = K.to_dense(ham_sparse)

        return H_total

    # Wrapper to optimize towards a target energy value
    # This creates a well-defined minimum for 'a' regularizing the problem.
    target_energy = -20.0

    def get_loss_val(a):
        H_mat = get_ground_state_energy(a)
        # Use K.eigh for JAX, which is differentiable
        evals = K.eigh(H_mat)[0]
        ground_energy = evals[0]

        # Loss is MSE between ground energy and target
        return (ground_energy - target_energy) ** 2, ground_energy

    # Create value and gradient function
    # has_aux=True allows returning the actual energy as auxiliary data
    # vg_func = K.jit(K.value_and_grad(get_loss_val, has_aux=True))

    # Optimization Loop
    # We optimize 'a' directly
    a_init = 1.5
    a_param = K.convert_to_tensor(a_init)

    optimizer = optax.adam(learning_rate=0.005)
    opt_state = optimizer.init(a_param)

    history_a = []
    history_e = []

    print(f"Initial lattice constant: {a_init}")

    steps = 100
    for i in range(steps):
        time_start = time.time()

        # Redefine wrapper for optimization variable
        def loss_wrapper(a):
            loss, aux = get_loss_val(a)
            return loss, aux

        vg_wrapper = K.jit(K.value_and_grad(loss_wrapper, has_aux=True))

        (loss, real_energy), grads = vg_wrapper(a_param)
        time_end = time.time()

        current_a = a_param
        history_a.append(K.numpy(current_a).item())
        history_e.append(K.numpy(real_energy).item())

        if i % 10 == 0:
            print(
                f"Step {i}: Loss = {loss:.6f}, Energy = {real_energy:.6f}, "
                f"a = {current_a:.6f}, grad = {grads:.6f} "
                f"({time_end - time_start:.4f}s)"
            )

        updates, opt_state = optimizer.update(grads, opt_state)
        a_param = optax.apply_updates(a_param, updates)

    final_a = a_param
    print(f"Final lattice constant: {final_a:.6f}")
    print(f"Final Energy: {real_energy:.6f} (Target: {target_energy})")

    # Plot optimization trajectory
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Lattice Constant (a)", color=color)
    ax1.plot(history_a, color=color, marker="o", markersize=3)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Ground State Energy", color=color)
    ax2.plot(history_e, color=color, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("AD Optimization of Lattice Constant (Rydberg Array)")
    fig.tight_layout()
    plt.savefig("lattice_optimization_trajectory.pdf")
    print("Saved optimization trajectory to 'lattice_optimization_trajectory.pdf'")
    plt.close()


if __name__ == "__main__":
    demo_lattice_generation()
    demo_custom_lattice()
    demo_ad_optimization()
