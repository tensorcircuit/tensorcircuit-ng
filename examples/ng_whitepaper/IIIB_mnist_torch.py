"""
MNIST 1 vs 9 Binary Classification with PyTorch + TensorCircuit TorchLayer

Network: Dense[784, 10] → PQC (10-qubit, 4 SU4 layers) → Dense[10, 1] → Sigmoid
"""

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import tensorcircuit as tc

# Use JAX backend for quantum circuit (faster JIT)
K = tc.set_backend("jax")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Convert to torch tensors
x_train_torch = torch.tensor(x_train, device=device)
y_train_torch = torch.tensor(y_train, device=device)
x_test_torch = torch.tensor(x_test, device=device)
y_test_torch = torch.tensor(y_test, device=device)


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


# Wrap quantum circuit as TorchLayer
quantum_layer = tc.TorchLayer(
    quantum_circuit,
    weights_shape=[n_layers, n_su4_per_layer, su4_params],
    use_vmap=True,  # Vectorize the circuit execution over the batch dimension
    use_interface=True,  # Use jax backend for quantum part
    use_jit=True,  # Just-in-time compile the circuit for performance
    enable_dlpack=True,  # Use DLPack for zero-copy tensor transfer between JAX and PyTorch
)


class HybridModel(nn.Module):
    """Hybrid quantum-classical model for binary classification."""

    def __init__(self):
        super().__init__()
        # Classical input layer
        self.fc_in = nn.Linear(784, n_qubits)
        # Quantum layer
        self.quantum = quantum_layer
        # Classical output layer
        self.fc_out = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Classical preprocessing
        x = self.fc_in(x)  # [batch, 784] -> [batch, 10]
        # Quantum processing
        x = self.quantum(x)  # [batch, 10] -> [batch, 10]
        # Classical postprocessing
        x = self.fc_out(x)  # [batch, 10] -> [batch, 1]
        x = self.sigmoid(x)
        return x


# Create model
model = HybridModel().to(device)
print(model)

# Training setup
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training parameters
n_epochs = 30
batch_size = 32
n_train = 500  # Use subset for faster training


print("\nStarting training...")
for epoch in range(n_epochs):
    model.train()

    # Random batch sampling
    indices = np.random.choice(n_train, batch_size, replace=False)
    x_batch = x_train_torch[indices]
    y_batch = y_train_torch[indices].unsqueeze(1)

    # Forward pass
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)

    # Backward pass
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test_torch[:100])
            test_preds = (test_outputs > 0.5).float()
            test_acc = (test_preds == y_test_torch[:100].unsqueeze(1)).float().mean()
        print(
            f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Test Acc: {test_acc.item():.4f}"
        )

print("\nTraining completed!")

# Final evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(x_test_torch[:100])
    test_preds = (test_outputs > 0.5).float()
    final_acc = (test_preds == y_test_torch[:100].unsqueeze(1)).float().mean()
    print(f"Final Test Accuracy: {final_acc.item():.4f}")
