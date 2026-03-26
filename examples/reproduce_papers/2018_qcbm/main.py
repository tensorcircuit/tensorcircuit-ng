"""Reproduction of "Differentiable Learning of Quantum Circuit Born Machine"
Link: https://arxiv.org/abs/1804.04168
Description:
This script reproduces Figure 6 from the paper using TensorCircuit-NG.
It trains a QCBM to model a mixture of two Gaussians using MMD loss.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

# Set JAX backend
tc.set_backend("jax")

# Parameters from the paper
n = 10  # number of qubits
n_states = 2**n
n_layers = 5  # Typical depth for this problem
learning_rate = 0.01
steps = 2000

# Target Distribution: Mixture of two Gaussians
# mu1 = 2/7 * 1024, mu2 = 5/7 * 1024, sigma = 128
mu1 = 2 / 7 * n_states
mu2 = 5 / 7 * n_states
sigma = 128


def target_pdf(x):
    p1 = jnp.exp(-0.5 * ((x - mu1) / sigma) ** 2)
    p2 = jnp.exp(-0.5 * ((x - mu2) / sigma) ** 2)
    p = p1 + p2
    return p / jnp.sum(p)


x_axis = jnp.arange(n_states)
p_target = target_pdf(x_axis)


# QCBM Ansatz
def qcbm_ansatz(params):
    c = tc.Circuit(n)
    # params shape: (n_layers + 1, n, 2)
    # Initial rotation layer
    for i in range(n):
        c.rz(i, theta=params[0, i, 0])
        c.rx(i, theta=params[0, i, 1])

    # Entangling layers
    for l in range(n_layers):
        # CNOT layer (circular)
        for i in range(n):
            c.cnot(i, (i + 1) % n)
        # Rotation layer
        for i in range(n):
            c.rz(i, theta=params[l + 1, i, 0])
            c.rx(i, theta=params[l + 1, i, 1])
    return c


# MMD Loss with Multi-scale Gaussian Kernels
# Bandwidths: 0.25, 10, 100, 1000 (standard for QCBM MMD)
bandwidths = jnp.array([0.25, 10, 100, 1000])


@jax.jit
def gaussian_kernel(x, y, sigma):
    # x, y are bitstrings converted to integers
    # Reshape for broadcasting: x (N, 1), y (1, M)
    dist_sq = (x[:, None] - y[None, :]) ** 2
    return jnp.exp(-dist_sq / (2 * sigma**2))


@jax.jit
def mmd_loss(params):
    # QCBM state vector
    c = qcbm_ansatz(params)
    probs = c.probability()  # All 2^n probabilities

    states = jnp.arange(n_states)

    def compute_mmd(s):
        # K(x, x') * p(x) * p(x')
        k_xx = gaussian_kernel(states, states, s)
        term_xx = jnp.dot(probs, jnp.dot(k_xx, probs))

        # K(x, y) * p(x) * p_target(y)
        k_xy = gaussian_kernel(states, states, s)
        term_xy = jnp.dot(probs, jnp.dot(k_xy, p_target))

        # K(y, y') * p_target(y) * p_target(y')
        k_yy = jnp.dot(p_target, jnp.dot(k_xx, p_target))

        return term_xx - 2 * term_xy + k_yy

    losses = jax.vmap(compute_mmd)(bandwidths)
    return jnp.mean(losses)


# Optimization
params = jnp.array(np.random.uniform(0, 2 * np.pi, size=(n_layers + 1, n, 2)))
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

loss_history = []


@jax.jit
def update(params, opt_state):
    val, grad = jax.value_and_grad(mmd_loss)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, val


print("Starting training...")
for i in range(steps):
    params, opt_state, val = update(params, opt_state)
    loss_history.append(float(val))
    if i % 100 == 0:
        print(f"Step {i}, MMD Loss: {val:.6f}")

# Final probabilities
c_final = qcbm_ansatz(params)
p_final = tc.backend.numpy(c_final.probability())

# Plotting results
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Fig 6(a): MMD Loss
ax1.plot(loss_history)
ax1.set_yscale("log")
ax1.set_xlabel("Adam Steps")
ax1.set_ylabel("MMD Loss")
ax1.set_title("MMD Loss Convergence")

# Fig 6(b): Histogram vs Target
bin_size = 16
n_bins = n_states // bin_size
x_binned = jnp.arange(n_bins) * bin_size + bin_size / 2
p_final_binned = p_final.reshape(n_bins, bin_size).sum(axis=1)
p_target_binned = p_target.reshape(n_bins, bin_size).sum(axis=1)

ax2.bar(
    x_binned,
    p_final_binned / bin_size,
    width=bin_size,
    alpha=0.6,
    color="green",
    label="QCBM",
)
ax2.plot(x_axis, p_target, "k--", label="Target PDF")
ax2.set_xlabel("x")
ax2.set_ylabel("p(x)")
ax2.set_title("Learned Distribution")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "result.png"))
print(f"Results saved to {os.path.join(output_dir, 'result.png')}")
