import optax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import tensorcircuit as tc


jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")


def calculate_potential(log_a, base_distance_matrix, epsilon=0.5, sigma=1.0):
    """
    Calculate the total Lennard-Jones potential energy for a given logarithm of the lattice constant (log_a).
    """
    lattice_constant = jnp.exp(log_a)
    d = base_distance_matrix * lattice_constant
    d_safe = jnp.where(d > 1e-9, d, 1e-9)

    term12 = (sigma / d_safe) ** 12
    term6 = (sigma / d_safe) ** 6
    potential_matrix = 4 * epsilon * (term12 - term6)

    num_sites = d.shape[0]
    potential_matrix = potential_matrix * (
        1 - K.eye(num_sites, dtype=potential_matrix.dtype)
    )

    potential_energy = K.sum(potential_matrix) / 2.0

    return potential_energy


# Pre-calculate the base distance matrix (for lattice_constant=1.0)
size = (10, 10)
lat_base = tc.templates.lattice.SquareLattice(size, lattice_constant=1.0, pbc=True)
base_distance_matrix = lat_base.distance_matrix

# Create a lambda function to pass the base distance matrix to the potential function
potential_fun_for_grad = lambda log_a: calculate_potential(log_a, base_distance_matrix)
value_and_grad_fun = K.jit(K.value_and_grad(potential_fun_for_grad))

optimizer = optax.adam(learning_rate=0.01)

log_a = K.convert_to_tensor(jnp.log(1.1))

opt_state = optimizer.init(log_a)

history = {"a": [], "energy": []}

print("Starting optimization of lattice constant...")
for i in range(200):
    energy, grad = value_and_grad_fun(log_a)

    history["a"].append(jnp.exp(log_a))
    history["energy"].append(energy)

    if jnp.isnan(grad):
        print(f"Gradient became NaN at iteration {i+1}. Stopping optimization.")
        print(f"Current energy: {energy}, Current log_a: {log_a}")
        break

    updates, opt_state = optimizer.update(grad, opt_state)
    log_a = optax.apply_updates(log_a, updates)

    if (i + 1) % 20 == 0:
        current_a = jnp.exp(log_a)
        print(
            f"Iteration {i+1}/200: Total Energy = {energy:.4f}, Lattice Constant = {current_a:.4f}"
        )

final_a = jnp.exp(log_a)
final_energy = calculate_potential(K.convert_to_tensor(log_a), base_distance_matrix)

if not jnp.isnan(final_energy):
    print("\nOptimization finished!")
    print(f"Final optimized lattice constant: {final_a:.6f}")
    print(f"Corresponding minimum total energy: {final_energy:.6f}")

    # Vectorized calculation for the potential curve
    a_vals = np.linspace(0.8, 1.5, 200)
    log_a_vals = np.log(a_vals)

    # Use vmap to create a vectorized version of the potential function
    vmap_potential = jax.vmap(lambda la: calculate_potential(la, base_distance_matrix))
    potential_curve = vmap_potential(K.convert_to_tensor(log_a_vals))

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
else:
    print("\nOptimization failed. Final energy is NaN.")
