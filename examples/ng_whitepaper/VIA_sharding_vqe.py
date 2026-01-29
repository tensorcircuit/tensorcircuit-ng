"""
Parallel VQE with jax.sharding
==============================

This script demonstrates how to parallelize VQE observable measurements using `jax.sharding` and `jax.jit`.
This utilizes JAX's GSPMD (General Single Program Multiple Data) capability.
We mock 8 CPU devices to simulate a multi-device environment.

The TFIM H terms are sharded across devices. The computation is written as if it were global,
and JAX automatically distributes the workload.
"""

import os

# Mock 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

# Sharding Definition
devices = jax.local_devices()
n_devices = len(devices)
print(f"Number of JAX devices: {n_devices}")

mesh = Mesh(devices, axis_names=("i",))
# We will shard along the first dimension (terms dimension)
sharding_w = NamedSharding(mesh, PartitionSpec("i"))
sharding_s = NamedSharding(mesh, PartitionSpec("i", None))

# Configuration
N = 10
J = 1.0
H = 1.0
lr = 0.02
steps = 200
layers = 3


def get_hamiltonian_terms(n, j_coupling, h_field):
    terms = []
    # ZZ interaction
    for i in range(n - 1):
        interaction = []
        for k in range(n):
            if k == i or k == i + 1:
                interaction.append(3)
            else:
                interaction.append(0)
        terms.append((-j_coupling, interaction))

    # X field
    for i in range(n):
        field = []
        for k in range(n):
            if k == i:
                field.append(1)
            else:
                field.append(0)
        terms.append((-h_field, field))
    return terms


# Prepare Data
all_terms = get_hamiltonian_terms(N, J, H)

# Pad to be divisible by n_devices for even sharding (optional but good for balance)
n_terms = len(all_terms)
remainder = n_terms % n_devices
if remainder != 0:
    padding = n_devices - remainder
    all_terms.extend([(0.0, [0] * N)] * padding)

weights = np.array([t[0] for t in all_terms], dtype=np.float32)
structures = np.array([t[1] for t in all_terms], dtype=np.int32)

# Shard the data
# device_put with sharding spec distributes the data across devices
weights_sharded = jax.device_put(weights, sharding_w)
structures_sharded = jax.device_put(structures, sharding_s)

print(f"Weights sharding: {weights_sharded.sharding}")
print(f"Structures sharding: {structures_sharded.sharding}")


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


# Function to compute expectation of MANY terms
# We use vmap to vectorize over the terms provided
def compute_expectations(params, w, s):
    # params: (n_params)
    # w: (n_terms) sharded
    # s: (n_terms, n_qubits) sharded

    c = ansatz(params, N)

    def term_expt(wi, si):
        return wi * tc.templates.measurements.parameterized_measurements(
            c, si, onehot=True
        )

    # vmap over terms
    # JAX compiler will see that w and s are sharded and distribute this vmap loop across devices
    expt = jax.vmap(term_expt)(w, s)
    return jnp.sum(expt)


def loss_fn(params, w, s):
    return compute_expectations(params, w, s)


# Optimizer
optimizer = optax.adam(lr)


@jax.jit
def update(params, opt_state, w, s):
    loss, grads = jax.value_and_grad(loss_fn)(params, w, s)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def main():
    n_params = layers * 2 * N
    params = jax.random.normal(jax.random.PRNGKey(42), (n_params,))
    opt_state = optimizer.init(params)

    print("Starting optimization...")
    start_time = time.time()

    for i in range(steps):
        params, opt_state, loss = update(
            params, opt_state, weights_sharded, structures_sharded
        )

        if i % 20 == 0:
            print(f"Step {i}: Energy = {loss:.6f}")

    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.4f}s")
    print(f"Final Energy: {loss:.6f}")

    # Compare with exact solution (for small N)
    if N <= 12:
        print("Computing exact ground state energy...")
        from tensorcircuit.quantum import PauliStringSum2COO
        import scipy.sparse.linalg

        # Use existing all_terms
        ls = [t[1] for t in all_terms]
        ws = [t[0] for t in all_terms]

        ham = PauliStringSum2COO(ls, weight=ws, numpy=True)
        e0, _ = scipy.sparse.linalg.eigsh(ham, k=1, which="SA")
        print(f"Exact Energy: {e0[0]:.6f}")


if __name__ == "__main__":
    main()
