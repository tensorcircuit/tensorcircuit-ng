"""
Parallel VQE with jax.pmap
==========================

This script demonstrates how to parallelize VQE observable measurements using `jax.pmap`.
We use the TensorCircuit JAX backend and mock 8 CPU devices to simulate a multi-device environment.

The Transverse Field Ising Model (TFIM) Hamiltonian terms are distributed across the available devices.
Each device computes the expectation value for its assigned terms, and the results are aggregated.
"""

import os

# Mock 8 CPU devices for demonstration
# This must be set before jax is initialized
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

# Setup TensorCircuit
tc.set_backend("jax")
tc.set_dtype("complex64")

# Configuration
N = 10  # Number of qubits
J = 1.0  # Ising interaction
H = 1.0  # Transverse field
lr = 0.02
steps = 200
layers = 3  # Number of ansatz layers

# Ensure we have multiple devices
n_devices = jax.local_device_count()
print(f"Number of JAX devices: {n_devices}")


def get_hamiltonian_terms(n, j_coupling, h_field):
    """
    Generate Pauli strings and weights for TFIM H = -J sum Z_i Z_{i+1} - H sum X_i
    Returns a list of tuples: (weight, [structure])
    """
    terms = []
    # ZZ interaction
    for i in range(n - 1):
        interaction = []
        for k in range(n):
            if k == i or k == i + 1:
                interaction.append(3)  # Z
            else:
                interaction.append(0)  # I
        terms.append((-j_coupling, interaction))

    # X field
    for i in range(n):
        field = []
        for k in range(n):
            if k == i:
                field.append(1)  # X
            else:
                field.append(0)  # I
        terms.append((-h_field, field))

    return terms


def distribute_terms(terms, n_dev):
    """
    Distribute Hamiltonian terms evenly across devices.
    Returns:
        weights_sharded: (n_dev, terms_per_dev)
        structures_sharded: (n_dev, terms_per_dev, n_qubits)
    """
    n_terms = len(terms)
    # Pad to make divisible by n_dev
    remainder = n_terms % n_dev
    if remainder != 0:
        padding = n_dev - remainder
        terms.extend([(0.0, [0] * N)] * padding)  # Pad with 0*Identity

    n_terms_padded = len(terms)
    terms_per_dev = n_terms_padded // n_dev

    weights = [t[0] for t in terms]
    structures = [t[1] for t in terms]

    weights_np = np.array(weights, dtype=np.float32)
    structures_np = np.array(structures, dtype=np.int32)

    # Reshape for pmap: (n_devices, terms_per_device, ...)
    w_sharded = weights_np.reshape(n_dev, terms_per_dev)
    s_sharded = structures_np.reshape(n_dev, terms_per_dev, N)

    return w_sharded, s_sharded


# Prepare Hamiltonian Data
all_terms = get_hamiltonian_terms(N, J, H)
weights_sharded, structures_sharded = distribute_terms(all_terms, n_devices)

# Replicate structures/weights to devices
# pmap expects the leading dimension of arguments to match device count
# However, if we pass them as fixed arguments (not mapped), every device gets the same copy.
# Here we want each device to get a *different* chunk, so we treat them as mapped inputs.


# Ansatz
def ansatz(params, n):
    c = tc.Circuit(n)
    k = 0
    for _ in range(layers):
        for i in range(n):
            c.rx(i, theta=params[k])
            k += 1
        for i in range(n):
            c.rz(i, theta=params[k])
            k += 1
        for i in range(n - 1):
            c.cnot(i, i + 1)
    return c


# Function to compute expectation of a CHUCK of terms on ONE device
# We need to compute Sum_i w_i <H_i>
def compute_chunk_expectation(params, sub_weights, sub_structures):
    # params: same for all terms (replicated via pmap logic or broadcast)
    # sub_weights: (M,)
    # sub_structures: (M, N)

    c = ansatz(params, N)

    # We can use tc.backend.vmap inside the device to compute expectations for M terms efficiently
    # Or simply loop if M is small. For vectorization, let's use vmap over the chunk terms.

    def term_expectation(w, s):
        # s is [0, 3, 0, ...] integers (0:I, 1:X, 2:Y, 3:Z)
        # parameterized_measurements is vmap-compatible
        return w * tc.templates.measurements.parameterized_measurements(
            c, s, onehot=True
        )

    # vmap over the chunk of terms assigned to this device
    expectations = jax.vmap(term_expectation)(sub_weights, sub_structures)

    return jnp.sum(expectations)


# Parallelized Loss Function
# We map over the devices.
# 'params' is broadcasted (None in in_axes if passed directly, or we replicate it)
# 'sub_weights', 'sub_structures' are mapped (0 in in_axes)

# Note: jax.pmap requires the first dimension to be the device dimension.
# We will replicate params to shape (n_devices, n_params) so we can map over it too
# (simpler for gradient aggregation usually, though broadcast works too).
# Let's broadcast params for simplicity.

# Parallelized Loss Function
# We map over the devices.
# 'params' is broadcasted (None in in_axes)
# 'sub_weights', 'sub_structures' are mapped (0 in in_axes)

parallel_expectation = jax.pmap(compute_chunk_expectation, in_axes=(None, 0, 0))


def loss_fn(params, w_sharded, s_sharded):
    # 1. Compute partial sums on each device
    # params is single array, we use in_axes=(None, 0, 0)
    # This runs the computations in parallel
    partial_sums = parallel_expectation(params, w_sharded, s_sharded)

    # 2. Sum up results from all devices
    total_energy = jnp.sum(partial_sums)
    return total_energy


# Gradient and Update
# We can JIT the update step which includes the pmap call.
# Since loss_fn calls pmap, differentiating it involves differentiating through pmap.
# JAX handles this by computing gradients on each device and then reducing them (all-reduce).


@jax.jit
def update(params, opt_state, w_sharded, s_sharded):
    grads = jax.grad(loss_fn)(params, w_sharded, s_sharded)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def main():
    print("Initializing...")
    # Parameters
    # layers * (2*N) params.
    n_params = layers * 2 * N
    params = jax.random.normal(jax.random.PRNGKey(42), (n_params,))

    # Send sharded data to devices (efficiently)
    # jax.device_put_sharded helps if we had specific device arrays,
    # but passing numpy arrays to pmap with in_axis=0 handles distribution automatically.

    # Optimizer
    global optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    print("Starting optimization loop...")
    start_time = time.time()

    losses = []
    for i in range(steps):
        params, opt_state = update(
            params, opt_state, weights_sharded, structures_sharded
        )
        if i % 20 == 0:
            current_loss = loss_fn(params, weights_sharded, structures_sharded)
            losses.append(current_loss)
            print(f"Step {i}: Energy = {current_loss.item():.6f}")

    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.4f}s")
    print(f"Final Energy: {losses[-1]:.6f}")

    # Compare with exact solution (for small N)
    if N <= 12:
        print("Computing exact ground state energy...")
        from tensorcircuit.quantum import PauliStringSum2COO
        import scipy.sparse.linalg

        # Use existing all_terms to construct Hamiltonian
        ls = [t[1] for t in all_terms]
        ws = [t[0] for t in all_terms]

        ham = PauliStringSum2COO(ls, weight=ws, numpy=True)
        e0, _ = scipy.sparse.linalg.eigsh(ham, k=1, which="SA")
        print(f"Exact Energy: {e0[0]:.6f}")


if __name__ == "__main__":
    main()
