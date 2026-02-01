"""
Hybrid Quantum-Classical Neural Network with Flax and TensorCircuit
====================================================================

This example demonstrates how to combine TensorCircuit (with JAX backend) and Flax
to construct a hybrid quantum-classical neural network for MNIST binary classification.

The architecture is:
Input (784) -> Dense (10) -> PQC (10 qubits, SU4 gates) -> Dense (1) -> Sigmoid

"""

import time
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import tensorcircuit as tc

# Set backend to JAX
tc.set_backend("jax")
tc.set_dtype("complex64")


def get_mnist_data():
    """
    Load and preprocess MNIST data for binary classification (0 vs 1).
    """
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Filter for 0 and 1
    train_mask = (y_train == 0) | (y_train == 1)
    test_mask = (y_test == 0) | (y_test == 1)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Normalize and flatten
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    y_train = y_train.astype(np.float32)  # 0 or 1
    y_test = y_test.astype(np.float32)

    return (x_train, y_train), (x_test, y_test)


# Define PQC
n_qubits = 10
n_layers = 2


def pqc_circuit(inputs, weights):
    """
    PQC with SU4 gates.
    inputs: shape [n_qubits] (from previous classical layer)
    weights: shape [n_layers, n_qubits - 1, 15] (SU4 parameters)
    """
    c = tc.Circuit(n_qubits)

    # Encoding
    for i in range(n_qubits):
        c.ry(i, theta=inputs[i])

    # Variational layers (SU4 brick-wall)
    # weights shape: [n_layers, n_pairs, 15]
    # n_pairs for linear topology is n_qubits - 1

    weights_reshaped = weights.reshape((n_layers, n_qubits - 1, 15))

    for l in range(n_layers):
        # Even pairs
        for i in range(0, n_qubits - 1, 2):
            c.su4(i, i + 1, theta=weights_reshaped[l, i])

        # Odd pairs
        for i in range(1, n_qubits - 1, 2):
            c.su4(i, i + 1, theta=weights_reshaped[l, i])

    # Measurement
    outputs = [c.expectation_ps(z=[i]) for i in range(n_qubits)]
    return jnp.stack(outputs)


# Vectorize PQC
# inputs: [Batch, n_qubits], weights: [Shape]
# vmap over batch dimension (0) for inputs, but share weights (None) which corresponds to in_axes=(0, None)
pqc_vmap = jax.vmap(pqc_circuit, in_axes=(0, None))
K_pqc = tc.backend.jit(pqc_vmap)


class HybridModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Classical Layer 1: 784 -> 10
        x = nn.Dense(features=10)(x)
        x = jnp.pi * nn.tanh(x)  # Map to [-pi, pi] for angle encoding

        # weights shape [n_layers, n_qubits - 1, 15]
        su4_weights = self.param(
            "su4_weights",
            nn.initializers.normal(stddev=0.1),
            (n_layers, n_qubits - 1, 15),
        )

        # Run PQC
        # x shape is [Batch, 10]
        # K_pqc expects (batch_inputs, weights)
        x = K_pqc(x, su4_weights)  # Output shape [Batch, 10]

        # Classical Layer 2: 10 -> 1
        x = nn.Dense(features=1)(x)

        return x  # Logits


def main():
    # Data
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    print(f"Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")

    # Model initialization
    model = HybridModel()
    dummy_input = jnp.ones((1, 784))
    key = jax.random.PRNGKey(42)
    params = model.init(key, dummy_input)

    # Optimizer
    learning_rate = 0.001
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    # Loss function (Binary Cross Entropy with logits)
    def loss_fn(params, x, y):
        logits = model.apply(params, x)
        # Cast to float (taking real part if complex) because optax expects float
        logits = logits.real.astype(jnp.float32)
        # y is 0 or 1. logits -> sigmoid -> prob
        return optax.sigmoid_binary_cross_entropy(logits, y[:, None]).mean()

    # Generic Update Step
    @jax.jit
    def update_step(params, opt_state, batch_x, batch_y):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Evaluation
    @jax.jit
    def accuracy(params, x, y):
        logits = model.apply(params, x)
        predicted_class = (logits > 0).astype(jnp.float32)
        return jnp.mean(predicted_class == y[:, None])

    # Training Loop
    epochs = 5
    batch_size = 64
    steps_per_epoch = x_train.shape[0] // batch_size

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()

        # Shuffle (basic, not perfect)
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]

        for i in range(steps_per_epoch):
            batch_x = x_train[i * batch_size : (i + 1) * batch_size]
            batch_y = y_train[i * batch_size : (i + 1) * batch_size]

            params, opt_state, loss = update_step(params, opt_state, batch_x, batch_y)
            epoch_loss += loss

        train_acc = accuracy(
            params, x_train[:1000], y_train[:1000]
        )  # Estimate on subset
        test_acc = accuracy(params, x_test, y_test)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/steps_per_epoch:.4f} | Train Acc: {train_acc:.4f} \
            | Test Acc: {test_acc:.4f} | Time: {time.time()-start_time:.2f}s"
        )


if __name__ == "__main__":
    main()
