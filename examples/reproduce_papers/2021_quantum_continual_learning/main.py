"""Reproduction of "Quantum Continual Learning Overcoming Catastrophic Forgetting"
Link: https://arxiv.org/abs/2108.02786
Description:
This script reproduces Figure 2(b) from the paper using TensorCircuit-NG.
It demonstrates the catastrophic forgetting phenomena in a quantum classifier
trained sequentially on MNIST (0, 9) and permuted MNIST (0, 9).
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorcircuit as tc

# Set TensorCircuit backend to JAX
tc.set_backend("jax")

# 0. Robust Path Handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Experimental Setup & Data Preparation
n_qubits = 8
n_layers = 10
n_classes = 2
img_size = 16  # 16x16 = 256 pixels = 2^8
n_train_samples = 512  # Adjusted for easy batching
n_test_samples = 256
batch_size = 64


def load_mnist_binary(d1=0, d2=9):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Filter for two digits
    train_mask = (y_train == d1) | (y_train == d2)
    test_mask = (y_test == d1) | (y_test == d2)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Map labels to 0 and 1
    y_train = (y_train == d2).astype(int)
    y_test = (y_test == d2).astype(int)

    # Downsample to 16x16 using linear interpolation
    def downsample(images):
        return jax.image.resize(images, (images.shape[0], img_size, img_size), "linear")

    x_train = downsample(x_train)
    x_test = downsample(x_test)

    # Normalize images for amplitude encoding
    def normalize_amplitudes(x):
        x = x.reshape(x.shape[0], -1)
        # Use JAX for normalization to be consistent
        norms = jnp.linalg.norm(x, axis=1, keepdims=True)
        # Avoid division by zero
        norms = jnp.where(norms == 0, 1.0, norms)
        return x / norms

    x_train = normalize_amplitudes(x_train)
    x_test = normalize_amplitudes(x_test)

    return (
        jnp.array(x_train[:n_train_samples]),
        jnp.array(y_train[:n_train_samples]),
    ), (jnp.array(x_test[:n_test_samples]), jnp.array(y_test[:n_test_samples]))


# Load and prepare data
(x_train_old, y_train_old), (x_test_old, y_test_old) = load_mnist_binary()

# Create permuted MNIST for Task 2
# Fix permutation seed for reproducibility
np.random.seed(42)
permutation = np.random.permutation(img_size**2)
x_train_new = x_train_old[:, permutation]
x_test_new = x_test_old[:, permutation]
y_train_new = y_train_old
y_test_new = y_test_old


# 2. Quantum Circuit (VQC)
def vqc_circuit(params, inputs):
    # params shape: (n_layers, n_qubits, 2)
    # Group layers in pairs to handle alternating entanglement with scan
    params_scan = jnp.reshape(params, (n_layers // 2, 2, n_qubits, 2))

    def double_layer(state, ps):
        # ps shape: (2, n_qubits, 2)
        c = tc.Circuit(n_qubits, inputs=state)
        # Layer 1 (Even)
        for j in range(n_qubits):
            c.rx(j, theta=ps[0, j, 0])
            c.rz(j, theta=ps[0, j, 1])
        for j in range(0, n_qubits - 1, 2):
            c.cnot(j, j + 1)
        # Layer 2 (Odd)
        for j in range(n_qubits):
            c.rx(j, theta=ps[1, j, 0])
            c.rz(j, theta=ps[1, j, 1])
        for j in range(1, n_qubits - 1, 2):
            c.cnot(j, j + 1)
        c.cnot(n_qubits - 1, 0)
        return c.state()

    # Initial state preparation
    c0 = tc.Circuit(n_qubits, inputs=inputs)
    s0 = c0.state()

    # Scan over layers to reduce JIT staging overhead
    sf = tc.backend.scan(double_layer, params_scan, s0)

    # Final measurement
    cf = tc.Circuit(n_qubits, inputs=sf)
    res = cf.expectation((tc.gates.z(), [0]))
    # Mapping outcome from [-1, 1] to probability [0, 1]
    return (tc.backend.real(res) + 1.0) / 2.0


vqc_vmap = tc.backend.vmap(vqc_circuit, vectorized_argnums=1)


def cross_entropy_loss(params, x, y):
    preds = vqc_vmap(params, x)
    # Clip predictions to avoid log(0) or log(1)
    preds = jnp.clip(preds, 1e-7, 1 - 1e-7)
    loss = -jnp.mean(y * jnp.log(preds) + (1 - y) * jnp.log(1 - preds))
    return loss


@jax.jit
def get_accuracy(params, x, y):
    preds = vqc_vmap(params, x)
    labels_pred = (preds > 0.5).astype(int)
    return jnp.mean(labels_pred == y)


vg_loss = jax.value_and_grad(cross_entropy_loss)


@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = vg_loss(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# 3. Sequential Training
params = jnp.array(np.random.normal(scale=0.1, size=(n_layers, n_qubits, 2)))
optimizer = optax.adam(0.01)
opt_state = optimizer.init(params)

# History for plotting
history_old = []
history_new = []


def run_training(params, opt_state, x_train, y_train, task_name, epochs=50):
    print(f"\nTraining {task_name}...")
    num_batches = n_train_samples // batch_size
    for epoch in range(epochs):
        # Shuffle each epoch for mini-batching
        idx = np.random.permutation(n_train_samples)
        epoch_loss = 0
        for i in range(num_batches):
            batch_idx = idx[i * batch_size : (i + 1) * batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_loss += loss

        if epoch % 5 == 0:
            acc = get_accuracy(params, x_test_old, y_test_old)
            print(
                f"Epoch {epoch}, Avg Loss: {epoch_loss/num_batches:.4f}, Task 1 Acc: {acc:.4f}"
            )
    return params, opt_state


# Task 1: Learn original MNIST 0/9
params, opt_state = run_training(params, opt_state, x_train_old, y_train_old, "Task 1")

# Task 2: Learn Permuted MNIST and track forgetting
print("\nSequential Training Task 2 (Permuted MNIST) and tracking Task 1...")
# Reset optimizer state for new task if desired, or continue from previous
opt_state_new = optimizer.init(params)

num_batches = n_train_samples // batch_size
for epoch in range(100):
    idx = np.random.permutation(n_train_samples)
    for i in range(num_batches):
        batch_idx = idx[i * batch_size : (i + 1) * batch_size]
        x_batch = x_train_new[batch_idx]
        y_batch = y_train_new[batch_idx]
        params, opt_state_new, loss = train_step(
            params, opt_state_new, x_batch, y_batch
        )

    # Track accuracies on both tasks
    acc_old = float(get_accuracy(params, x_test_old, y_test_old))
    acc_new = float(get_accuracy(params, x_test_new, y_test_new))
    history_old.append(acc_old)
    history_new.append(acc_new)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}, Task 2 Acc: {acc_new:.4f}, Task 1 Acc: {acc_old:.4f} (Forgetting)"
        )

# 4. Plotting & Saving Results
plt.figure(figsize=(7, 5))
plt.plot(history_new, history_old, "o-", markersize=4, label="Forgetting Curve")
plt.xlabel(r"$\gamma_{new}$ (Accuracy on New Task)")
plt.ylabel(r"$\gamma_{old}$ (Accuracy on Old Task)")
plt.title("Catastrophic Forgetting in Quantum Classifier")
plt.grid(True, linestyle="--", alpha=0.6)
result_path = os.path.join(OUTPUT_DIR, "result.png")
plt.savefig(result_path, dpi=300)
plt.close()

print(f"\nReproduction complete. Results saved to {result_path}.")
