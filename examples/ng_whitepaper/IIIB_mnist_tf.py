"""
MNIST 1 vs 9 Binary Classification with TensorFlow + TensorCircuit KerasLayer

Network: Dense[784, 10] → PQC (10-qubit, 4 SU4 layers) → Dense[10, 1] → Sigmoid
"""

import numpy as np
import tensorflow as tf
import tensorcircuit as tc

# Use TensorFlow backend
K = tc.set_backend("tensorflow")

# Configuration
n_qubits = 10
n_layers = 4
n_su4_per_layer = n_qubits - 1  # 9 SU4 gates per layer
su4_params = 15  # Each SU4 gate has 15 parameters

# Dataset preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0


def filter_pair(x, y, a, b):
    """Filter dataset for binary classification of digits a and b."""
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = (y == a).astype(np.float32)  # 1 for digit a, 0 for digit b
    return x, y


# Filter for digits 1 and 9
x_train, y_train = filter_pair(x_train, y_train, 1, 9)
x_test, y_test = filter_pair(x_test, y_test, 1, 9)

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

# Use subset for faster training
n_train = min(500, len(x_train))
n_test = min(100, len(x_test))

x_train_subset = x_train[:n_train]
y_train_subset = y_train[:n_train]
x_test_subset = x_test[:n_test]
y_test_subset = y_test[:n_test]


def quantum_circuit(inputs, weights):
    """
    Parameterized Quantum Circuit with SU4 gates.

    Args:
        inputs: Input angles for RY gates, shape [n_qubits]
        weights: SU4 gate parameters, shape [n_layers, n_su4_per_layer, 15]

    Returns:
        Expectation values of RZ on each qubit, shape [n_qubits]
    """
    c = tc.Circuit(n_qubits)

    # Input layer: RY rotations
    for i in range(n_qubits):
        c.ry(i, theta=inputs[i])

    # SU4 layers
    for layer in range(n_layers):
        for i in range(n_qubits - 1):
            c.su4(i, i + 1, theta=weights[layer, i])

    # Output: RZ expectation values (Z basis)
    outputs = K.stack([K.real(c.expectation_ps(z=[i])) for i in range(n_qubits)])
    return outputs


# Wrap quantum circuit as KerasLayer
quantum_layer = tc.KerasLayer(
    quantum_circuit,
    weights_shape=[(n_layers, n_su4_per_layer, su4_params)],
)


# Build hybrid model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(n_qubits, input_shape=(784,)),
        quantum_layer,
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Build model by passing sample input
sample_input = tf.zeros([1, 784])
_ = model(sample_input)
model.summary()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

# Training
print("\nStarting training...")
history = model.fit(
    x_train_subset,
    y_train_subset,
    epochs=30,
    batch_size=32,
    validation_data=(x_test_subset, y_test_subset),
    verbose=1,
)

# Final evaluation
print("\nFinal Evaluation:")
loss, accuracy = model.evaluate(x_test_subset, y_test_subset)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
