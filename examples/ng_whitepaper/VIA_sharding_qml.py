"""
Parallel QML with jax.sharding
==============================

This script demonstrates data parallelism for QML using `jax.sharding` and `jax.jit`.
We perform binary classification (digits 1 vs 9) on MNIST.
The input data batch is sharded across 8 mocked CPU devices.
JAX automatically parallelizes the computation (GSPMD).
"""

import os

# Mock 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import time
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

# Configuration
N_QUBITS = 10
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01

n_devices = jax.local_device_count()
print(f"Number of JAX devices: {n_devices}")

mesh = Mesh(jax.local_devices(), axis_names=("batch",))
# Shard first dimension (batch) across devices
sharding = NamedSharding(mesh, PartitionSpec("batch"))


# --- Data Loading (1 vs 9) ---
def load_mnist_1_9():
    print("Loading MNIST data (1 vs 9)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def filter_1_9(x, y):
        mask = (y == 1) | (y == 9)
        x = x[mask]
        y = y[mask]
        y = np.where(y == 1, 0, 1)
        return x, y

    x_train, y_train = filter_1_9(x_train, y_train)
    x_test, y_test = filter_1_9(x_test, y_test)

    x_train = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32)

    dim = 2**N_QUBITS
    if x_train.shape[1] < dim:
        pad = dim - x_train.shape[1]
        x_train = np.pad(x_train, ((0, 0), (0, pad)))
        x_test = np.pad(x_test, ((0, 0), (0, pad)))

    x_train = x_train / (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-8)
    x_test = x_test / (np.linalg.norm(x_test, axis=1, keepdims=True) + 1e-8)

    return x_train, y_train, x_test, y_test


# --- Model ---
def qmodel(params, x):
    c = tc.Circuit(N_QUBITS, inputs=x)
    k = 0
    weights = params["weights"]
    for i in range(N_QUBITS):
        c.ry(i, theta=weights[k])
        k += 1
    for i in range(N_QUBITS - 1):
        c.cnot(i, i + 1)
    for i in range(N_QUBITS):
        c.ry(i, theta=weights[k])
        k += 1
    preds = c.expectation_ps(z=[0])
    scale = params["scale"]
    bias = params["bias"]
    return preds * scale + bias


# --- Update ---
optimizer = optax.adam(LR)


def loss_fn(params, x, y):
    # vmap over batch
    # if x is sharded, this vmap is automatically parallelized
    logits = jax.vmap(qmodel, in_axes=(None, 0))(params, x)
    logits = jnp.real(logits)
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    return jnp.mean(loss)


def update_step_impl(params, opt_state, x, y):
    # x, y are sharded arrays
    # params, opt_state are replicated (by default if scalar w.r.t sharded dim)
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


update_step = jax.jit(update_step_impl, in_shardings=(None, None, sharding, sharding))


def main():
    x_train, y_train, _, _ = load_mnist_1_9()
    n_train = len(x_train)
    # Trim
    n_train = (n_train // BATCH_SIZE) * BATCH_SIZE
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    init_params = {
        "weights": jax.random.normal(jax.random.PRNGKey(0), (2 * N_QUBITS,)),
        "scale": jnp.array(1.0),
        "bias": jnp.array(0.0),
    }
    opt_state = optimizer.init(init_params)

    print("Starting training...")
    steps_per_epoch = n_train // BATCH_SIZE

    for epoch in range(EPOCHS):
        perms = np.random.permutation(n_train)
        x_shuffled = x_train[perms]
        y_shuffled = y_train[perms]

        epoch_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            start = step * BATCH_SIZE
            end = start + BATCH_SIZE

            # Direct passing of numpy arrays; JAX handles sharding and transfer automatically with optimized pipelining
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Update
            params, opt_state, loss = update_step(
                init_params, opt_state, x_batch, y_batch
            )
            loss.block_until_ready()

            # Since update_step returns new params, we must update init_params
            init_params = params
            epoch_loss += loss

        epoch_loss /= steps_per_epoch
        t1 = time.time()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Time = {t1-t0:.2f}s")

    print("Training complete.")


if __name__ == "__main__":
    main()
