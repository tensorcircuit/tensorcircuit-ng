"""
Parallel QML with jax.pmap
==========================

This script demonstrates data parallelism for QML using `jax.pmap`.
We perform binary classification (digits 1 vs 9) on MNIST.
The data batch is split across 8 mocked CPU devices.
Gradients are computed locally on each device and averaged using `jax.lax.pmean`.
"""

import os

# Mock 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import time
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

# Configuration
N_QUBITS = 10
N_CLASSES = 1  # Binary classification output (sigmoid)
BATCH_SIZE = 128  # Total batch size (must be divisible by n_devices)
EPOCHS = 10
LR = 0.01

n_devices = jax.local_device_count()
print(f"Number of JAX devices: {n_devices}")

assert BATCH_SIZE % n_devices == 0, "Batch size must be divisible by device count"
BATCH_PER_DEVICE = BATCH_SIZE // n_devices


# --- Data Loading (1 vs 9) ---
def load_mnist_1_9():
    print("Loading MNIST data (1 vs 9)...")
    # Load via TF (CPU)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def filter_1_9(x, y):
        # 1 -> 0, 9 -> 1
        mask = (y == 1) | (y == 9)
        x = x[mask]
        y = y[mask]
        y = np.where(y == 1, 0, 1)  # Relabel: 1->0, 9->1
        return x, y

    x_train, y_train = filter_1_9(x_train, y_train)
    x_test, y_test = filter_1_9(x_test, y_test)

    # Flatten and Pad
    x_train = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32)

    # Pad to 2^N_QUBITS = 1024
    dim = 2**N_QUBITS
    if x_train.shape[1] < dim:
        pad = dim - x_train.shape[1]
        x_train = np.pad(x_train, ((0, 0), (0, pad)))
        x_test = np.pad(x_test, ((0, 0), (0, pad)))

    # Normalize for amplitude encoding
    x_train = x_train / (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-8)
    x_test = x_test / (np.linalg.norm(x_test, axis=1, keepdims=True) + 1e-8)

    return x_train, y_train, x_test, y_test


# --- Model ---
def qmodel(params, x):
    # x: (1024,)
    c = tc.Circuit(N_QUBITS, inputs=x)

    # Simple layered ansatz
    k = 0
    weights = params["weights"]

    for i in range(N_QUBITS):
        c.ry(i, theta=weights[k])
        k += 1

    # Entangling
    for i in range(N_QUBITS - 1):
        c.cnot(i, i + 1)

    for i in range(N_QUBITS):
        c.rx(i, theta=weights[k])
        k += 1

    # Readout from first qubit
    # Expectation of Z0
    preds = c.expectation_ps(z=[0])

    # Map to probability or logit
    # If we use binary cross entropy with logits we need (N, 1)
    # expectation is in [-1, 1]. Let's map to logits via some scaling + bias
    scale = params["scale"]
    bias = params["bias"]

    return preds * scale + bias


# --- Parallel Update ---

# Replicate optimizer
optimizer = optax.adam(LR)


def loss_fn(params, x, y):
    # x: (B, 1024), y: (B,)
    # vmap over batch
    logits = jax.vmap(qmodel, in_axes=(None, 0))(params, x)
    # Take real part for binary cross entropy
    logits = jnp.real(logits)
    # y is 0 or 1
    # optax.sigmoid_binary_cross_entropy takes logits and labels
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    return jnp.mean(loss)


# The update function runs on each device
def update_step(params, opt_state, x_batch, y_batch):
    # Compute gradient on the local batch
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)

    # Average gradients across all devices
    grads = jax.lax.pmean(grads, axis_name="i")
    loss = jax.lax.pmean(loss, axis_name="i")

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# parallel_update takes:
# params: replicated on all devices
# opt_state: replicated
# x_batch, y_batch: sharded across devices (leading dimension = n_devices)
parallel_update = jax.pmap(update_step, axis_name="i")


# --- Main ---
def main():
    x_train, y_train, x_test, _ = load_mnist_1_9()
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    # Init params
    n_weights = 2 * N_QUBITS

    init_params = {
        "weights": jax.random.normal(jax.random.PRNGKey(0), (n_weights,)),
        "scale": jnp.array(1.0),
        "bias": jnp.array(0.0),
    }

    # Replicate params for each device
    # Shape: (n_devices, ...)
    params_replicated = jax.device_put_replicated(init_params, jax.local_devices())

    # Initialize optimizer state
    opt_state = optimizer.init(init_params)
    opt_state_replicated = jax.device_put_replicated(opt_state, jax.local_devices())

    print("Starting training...")
    n_train = len(x_train)
    # Trim to divisible by BATCH_SIZE
    n_train = (n_train // BATCH_SIZE) * BATCH_SIZE
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    steps_per_epoch = n_train // BATCH_SIZE

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        # Shuffle (numpy) and reshape for pmap
        perms = np.random.permutation(n_train)
        x_shuffled = x_train[perms]
        y_shuffled = y_train[perms]

        # Reshape to (steps, n_devices, batch_per_device, features)
        x_reshaped = x_shuffled.reshape(
            steps_per_epoch, n_devices, BATCH_PER_DEVICE, -1
        )
        y_reshaped = y_shuffled.reshape(steps_per_epoch, n_devices, BATCH_PER_DEVICE)

        t0 = time.time()
        for step in range(steps_per_epoch):
            x_step = x_reshaped[step]
            y_step = y_reshaped[step]

            # Update
            params_replicated, opt_state_replicated, loss_val = parallel_update(
                params_replicated, opt_state_replicated, x_step, y_step
            )

            # loss_val is (n_devices,) but they should be identical (pmean)
            epoch_loss += loss_val[0]

        epoch_loss /= steps_per_epoch
        t1 = time.time()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Time = {t1-t0:.2f}s")

    print("Training complete.")


if __name__ == "__main__":
    main()
