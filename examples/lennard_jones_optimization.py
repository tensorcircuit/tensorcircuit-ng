"""
Lennard-Jones Potential Optimization Example

This script demonstrates how to use TensorCircuit's differentiable lattice geometries
to optimize crystal structure. It finds the equilibrium lattice constant that minimizes
the total Lennard-Jones potential energy of a 2D square lattice.

This example showcases a key capability of the differentiable lattice system:
making geometric parameters (like lattice constants) fully differentiable and
optimizable using automatic differentiation. This enables variational material design
where crystal structures can be optimized to minimize physical energy functions.
"""

import optax
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc


tc.set_dtype("float64")  # Use tc for universal control
K = tc.set_backend("jax")


def calculate_potential(log_a, epsilon=0.5, sigma=1.0):
    """
    Calculate the total Lennard-Jones potential energy for a given logarithm of the lattice constant (log_a).
    This version creates the lattice inside the function to demonstrate truly differentiable geometry.
    """
    lattice_constant = K.exp(log_a)

    # Create lattice with the differentiable parameter
    size = (4, 4)  # Smaller size for demonstration
    lattice = tc.templates.lattice.SquareLattice(
        size, lattice_constant=lattice_constant, pbc=True
    )
    d = lattice.distance_matrix

    d_safe = K.where(d > 1e-9, d, K.convert_to_tensor(1e-9))

    term12 = K.power(sigma / d_safe, 12)
    term6 = K.power(sigma / d_safe, 6)
    potential_matrix = 4 * epsilon * (term12 - term6)

    num_sites = lattice.num_sites
    # Zero out self-interactions (diagonal elements)
    eye_mask = K.eye(num_sites, dtype=potential_matrix.dtype)
    potential_matrix = potential_matrix * (1 - eye_mask)

    potential_energy = K.sum(potential_matrix) / 2.0

    return potential_energy


# Create value and grad function for optimization
value_and_grad_fun = K.jit(K.value_and_grad(calculate_potential))

optimizer = optax.adam(learning_rate=0.01)

log_a = K.convert_to_tensor(K.log(K.convert_to_tensor(2.0)))

opt_state = optimizer.init(log_a)

history = {"a": [], "energy": []}

print("Starting optimization of lattice constant...")
for i in range(200):
    energy, grad = value_and_grad_fun(log_a)

    history["a"].append(K.exp(log_a))
    history["energy"].append(energy)

    updates, opt_state = optimizer.update(grad, opt_state)
    log_a = optax.apply_updates(log_a, updates)

    if (i + 1) % 20 == 0:
        current_a = K.exp(log_a)
        print(
            f"Iteration {i+1}/200: Total Energy = {energy:.4f}, Lattice Constant = {current_a:.4f}"
        )

final_a = K.exp(log_a)
final_energy = calculate_potential(log_a)

print("\nOptimization finished!")
print(f"Final optimized lattice constant: {final_a:.6f}")
print(f"Corresponding minimum total energy: {final_energy:.6f}")

# Vectorized calculation for the potential curve
a_vals = np.linspace(0.8, 1.5, 200)
log_a_vals = K.log(K.convert_to_tensor(a_vals))

# Use vmap to create a vectorized version of the potential function
vmap_potential = K.vmap(lambda la: calculate_potential(la))
potential_curve = vmap_potential(log_a_vals)

plt.figure(figsize=(10, 6))
plt.plot(a_vals, potential_curve, label="Lennard-Jones Potential", color="blue")
plt.scatter(
    history["a"],
    history["energy"],
    color="red",
    s=20,
    zorder=5,
    label="Optimization Steps",
)
plt.scatter(
    final_a,
    final_energy,
    color="green",
    s=100,
    zorder=6,
    marker="*",
    label="Final Optimized Point",
)

plt.title("Lennard-Jones Potential Optimization")
plt.xlabel("Lattice Constant (a)")
plt.ylabel("Total Potential Energy")
plt.legend()
plt.grid(True)
plt.show()
