"""
MNIST Quantum Machine Learning with TensorCircuit
=================================================

This script demonstrates a Quantum Machine Learning (QML) pipeline for MNIST classification.
It uses:
- TensorCircuit with JAX backend and JIT compilation.
- Deep quantum circuit using `jax.lax.scan` for efficiency.
- SU(4) gates in a ladder layout for expressivity.
- Amplitude encoding of 28x28 images (padded to 1024 states = 10 qubits).
- Readout of the first 4 qubits (16 states) mapped to 10 classes.
"""

import time
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import optax
import cotengra
import tensorcircuit as tc

# --- configuration ---
# Ensure TF uses CPU to avoid memory conflict with JAX
tf.config.set_visible_devices([], "GPU")

tc.set_backend("jax")
tc.set_dtype("complex64")
conopt = cotengra.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel=16,
    minimize="combo",
    max_time=60,
    max_repeats=512,
    progbar=True,
)
tc.set_contractor("custom", optimizer=conopt)

# Parameters
N_QUBITS = 10
N_CLASSES = 10
IMAGE_DIM = 28 * 28  # 784 features
TC_DIM = 2**N_QUBITS  # 1024 features
BATCH_SIZE = 15000
EPOCHS = 20  # Reduced for demonstration
LEARNING_RATE = 0.01
LAYERS = 40  # Depth of the circuit
SEED = 42

# --- Data Loading ---


def load_mnist_data():
    """
    Loads MNIST data (Train + Test merged), preprocesses for amplitude encoding.
    Returns:
        x_all, y_all, x_test, y_test (JAX arrays on GPU)
    """
    print("Loading MNIST dataset...")
    # Use TF to load data (CPU only)
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Concatenate all data for training
    # x_all_raw = np.concatenate([x_train_raw, x_test_raw], axis=0)
    # y_all_raw = np.concatenate([y_train_raw, y_test_raw], axis=0)

    # Just use train data
    x_all_raw = x_train_raw
    y_all_raw = y_train_raw

    # Flatten: (N, 28, 28) -> (N, 784)
    x_all_flat = x_all_raw.reshape(x_all_raw.shape[0], -1).astype(np.float32)
    # We also keep x_test for validation
    x_test_flat = x_test_raw.reshape(x_test_raw.shape[0], -1).astype(np.float32)

    # Pad to 1024 features (2^10)
    # 1024 - 784 = 240
    pad_dim = TC_DIM - IMAGE_DIM
    x_all_padded = np.pad(x_all_flat, ((0, 0), (0, pad_dim)), mode="constant")
    x_test_padded = np.pad(x_test_flat, ((0, 0), (0, pad_dim)), mode="constant")

    # L2 Normalize for amplitude encoding
    norm_all = np.linalg.norm(x_all_padded, axis=1, keepdims=True)
    x_all_norm = x_all_padded / (norm_all + 1e-8)

    norm_test = np.linalg.norm(x_test_padded, axis=1, keepdims=True)
    x_test_norm = x_test_padded / (norm_test + 1e-8)

    # Labels to int64 flattened
    y_all = y_all_raw.flatten().astype(np.int64)
    y_test = y_test_raw.flatten().astype(np.int64)

    print(f"Data Loaded: All {x_all_norm.shape}, Test {x_test_norm.shape}")

    # Convert to JAX arrays (pushes to GPU)
    return (
        jnp.array(x_all_norm),
        jnp.array(y_all),
        jnp.array(x_test_norm),
        jnp.array(y_test),
    )


# --- QModel Definition ---


def qmodel(params, x):
    """
    Quantum Model using Scan.
    Args:
        params: Dictionary {'weights': ..., 'alpha': ...}
                weights: {'even': (layers, ...), 'odd': (layers, ...)}
        x: Input vector (amplitude encoded state)
    Returns:
        logits: (10,)
    """
    # Unpack parameters
    weights = params["weights"]
    alpha = params["alpha"]

    # 1. Amplitude Encoding (State Preparation)
    # x is (1024,) real vector
    c = tc.Circuit(N_QUBITS, inputs=x)

    # helper for one layer scan
    def one_layer(state, layer_params):
        # state: quantum state vector
        # layer_params: {'even': (n_even, 15), 'odd': (n_odd, 15)}

        c_layer = tc.Circuit(N_QUBITS, inputs=state)

        # Even Layer: (0,1), (2,3), ...
        # qubits 0..9 -> pairs (0,1), (2,3), (4,5), (6,7), (8,9) -> 5 pairs
        for i in range(0, N_QUBITS - 1, 2):
            c_layer.su4(i, i + 1, theta=layer_params["even"][i // 2])

        # Odd Layer: (1,2), (3,4), ...
        # pairs (1,2), (3,4), (5,6), (7,8) -> 4 pairs
        for i in range(1, N_QUBITS - 1, 2):
            c_layer.su4(i, i + 1, theta=layer_params["odd"][(i - 1) // 2])

        return c_layer.state(), None

    # Scan over layers
    final_state, _ = jax.lax.scan(one_layer, c.state(), weights)

    # Readout
    # Requirement: "Front four qubits probability amplitude output judgment head"
    # We trace out qubits 4, 5, 6, 7, 8, 9
    rho = tc.quantum.reduced_density_matrix(
        final_state, cut=[i for i in range(4, N_QUBITS)]
    )

    # Get diagonal elements (probabilities)
    probs = tc.backend.diagonal(rho)

    # Real part
    probs = tc.backend.real(probs)

    # Take first 10 probabilities
    # 4 qubits -> 16 states. We need 10.
    probs_10 = probs[:N_CLASSES]

    # Use alpha * log(probs) as logits
    logits = alpha * tc.backend.log(probs_10 + 1e-8)

    return logits


# --- Training ---


def create_params(seed):
    key = jax.random.PRNGKey(seed)

    # N=10: even=5 pairs, odd=4 pairs
    n_even = N_QUBITS // 2
    n_odd = (N_QUBITS - 1) // 2

    k1, k2 = jax.random.split(key)

    weights = {
        "even": jax.random.normal(k1, (LAYERS, n_even, 15)) * 0.1,
        "odd": jax.random.normal(k2, (LAYERS, n_odd, 15)) * 0.1,
    }

    params = {"weights": weights, "alpha": jnp.array(1.0)}
    return params


def loss_fn(params, x, y):
    logits = qmodel(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)


def compute_accuracy(params, x, y):
    batch_qmodel = jax.vmap(qmodel, in_axes=(None, 0))
    logits = batch_qmodel(params, x)
    predicted = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted == y)


@jax.jit
def update(params, opt_state, x, y):
    def batch_loss(p, bx, by):
        losses = jax.vmap(loss_fn, in_axes=(None, 0, 0))(p, bx, by)
        return jnp.mean(losses)

    loss, grads = jax.value_and_grad(batch_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def evaluate(params, x, y):
    acc = compute_accuracy(params, x, y)
    losses = jax.vmap(loss_fn, in_axes=(None, 0, 0))(params, x, y)
    loss = jnp.mean(losses)
    return acc, loss


def evaluate_batch(params, x, y):
    # t0 = time.time()
    n_samples = x.shape[0]
    n_batches = int(np.ceil(n_samples / BATCH_SIZE))

    total_loss = 0.0
    total_acc = 0.0

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, n_samples)
        bx = x[start:end]
        by = y[start:end]

        acc, loss = evaluate(params, bx, by)
        b_size = end - start
        total_loss += loss * b_size
        total_acc += acc * b_size

    # t1 = time.time()
    # print(f"Evaluation Time: {t1-t0:.4f}s")
    return total_acc / n_samples, total_loss / n_samples


# --- Main ---

if __name__ == "__main__":
    # Load Data (All on GPU)
    x_all, y_all, x_test, y_test = load_mnist_data()

    # Initialize Params
    params = create_params(SEED)

    # Optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    print(f"Starting Training: {EPOCHS} epochs, Batch Size {BATCH_SIZE}")

    n_train = x_all.shape[0]
    steps_per_epoch = n_train // BATCH_SIZE

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Shuffle indices
        perms = np.random.permutation(n_train)
        x_shuffled = x_all[perms]
        y_shuffled = y_all[perms]

        epoch_loss = 0.0
        batch_times = []

        for step in range(steps_per_epoch):
            start = step * BATCH_SIZE
            end = start + BATCH_SIZE
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            t0_batch = time.time()
            params, opt_state, loss = update(params, opt_state, x_batch, y_batch)
            loss.block_until_ready()
            t1_batch = time.time()
            batch_times.append(t1_batch - t0_batch)

            epoch_loss += loss

        epoch_loss /= steps_per_epoch
        epoch_time = time.time() - start_time

        avg_batch_time = np.mean(batch_times)

        # Validation on Test set
        val_acc, val_loss = evaluate_batch(params, x_test, y_test)

        print(
            f"Epoch {epoch+1:03d} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        print(f"  > Avg Batch Time: {avg_batch_time:.4f}s")

    print(f"\nTraining Complete.")
