"""
Reproduction of "Data re-uploading for a universal quantum classifier"
Link: https://arxiv.org/abs/1907.02085

Description:
This script reproduces Figure 6 from the paper using TensorCircuit.
It implements a single-qubit quantum classifier using data re-uploading.
The task is to classify points inside/outside a circle.
"""

import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import optax
import jax
import tensorcircuit as tc

# Set backend to JAX for better performance
K = tc.set_backend("jax")


def generate_circle_data(n_samples: int):
    """
    Generate 2D data points inside/outside a circle.
    Returns:
        X: (n_samples, 2) array of coordinates in [-1, 1]
        Y: (n_samples,) array of labels (0 or 1)
    """
    # Use fixed seed for reproducibility
    np.random.seed(42)
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    # Circle radius sqrt(2/pi) covers half the area of the square [-1, 1]x[-1, 1] (area 4)
    # Area of circle = pi * r^2. Area of square = 4.
    # To have balanced classes, pi * r^2 = 2 => r^2 = 2/pi => r = sqrt(2/pi) ~ 0.798
    radius = np.sqrt(2 / np.pi)
    Y = np.sum(X**2, axis=1) < radius**2
    return X, Y.astype(int)


def clf_circuit(params, x, n_layers):
    """
    Quantum circuit for classification.
    params: (n_layers, 4)
    x: (2,)
    """
    c = tc.Circuit(1)
    for i in range(n_layers):
        # params[i] -> [w1, b1, w0, b0]
        # ansatz: Rz(w1*x1 + b1) Ry(w0*x0 + b0)
        # Note: x is [x0, x1]
        theta_z = params[i, 0] * x[1] + params[i, 1]
        theta_y = params[i, 2] * x[0] + params[i, 3]
        c.rz(0, theta=theta_z)
        c.ry(0, theta=theta_y)
    return c


def predict_point(params, xi, n_layers):
    c = clf_circuit(params, xi, n_layers)
    # Probability of state |1>
    # TC z expectation is <Z> = P(0) - P(1) = 1 - 2P(1)
    # So P(1) = (1 - <Z>) / 2
    z_exp = c.expectation_ps(z=[0])
    p1 = (1.0 - z_exp) / 2.0
    return p1


def loss(params, x, y, n_layers):
    """
    Calculate the weighted fidelity loss.
    params: flat parameters from scipy (or reshaped)
    x: (n_samples, 2)
    y: (n_samples,)
    """
    probs_1 = K.vmap(predict_point, vectorized_argnums=1)(params, x, n_layers)
    loss_val = K.mean((y - probs_1) ** 2)
    return K.real(loss_val)


def main():
    n_samples = 200
    X, Y = generate_circle_data(n_samples)

    # Convert data to backend tensors once
    X_tc = K.convert_to_tensor(X)
    Y_tc = K.convert_to_tensor(Y)

    # Different number of layers to test
    layers_list = [1, 2, 4]

    plt.figure(figsize=(15, 5))

    for idx, n_layers in enumerate(layers_list):
        print(f"Training with {n_layers} layers...")

        # Initial parameters
        # shape: (n_layers, 4)
        # Initialize randomly
        param_shape = (n_layers, 4)
        init_params = np.random.normal(0, 1, size=param_shape)
        params = K.convert_to_tensor(init_params)

        # Use optax.lbfgs as requested
        solver = optax.lbfgs(learning_rate=1.0)
        opt_state = solver.init(params)

        # Jitted update step for L-BFGS
        @jax.jit
        def update_step(params, opt_state, x, y):
            loss_val, grads = jax.value_and_grad(loss)(params, x, y, n_layers)
            updates, opt_state = solver.update(
                grads,
                opt_state,
                params,
                value=loss_val,
                grad=grads,
                value_fn=partial(loss, x=x, y=y, n_layers=n_layers),
            )
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        start_time = time.time()
        loss_history = []
        # L-BFGS often converges faster in fewer steps, but needs more computation per step (line search)
        # We'll use fewer iterations compared to Adam (e.g., 50 or 100)
        for _ in range(50):
            params, opt_state, loss_val = update_step(params, opt_state, X_tc, Y_tc)
            loss_history.append(loss_val)

        end_time = time.time()
        final_loss = loss_history[-1]
        print(
            f"Optimization finished in {end_time - start_time:.2f}s. Loss: {final_loss}"
        )

        opt_params = params

        # Visualization
        plt.subplot(1, 3, idx + 1)

        # Generate grid
        grid_size = 50
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Predict on grid
        @K.jit
        def predict_batch(p, x_in):
            # reusing predict_point
            return K.vmap(predict_point, vectorized_argnums=1)(p, x_in, n_layers)

        # Convert grid_points to backend
        grid_points_tc = K.convert_to_tensor(grid_points)

        probs_grid = predict_batch(opt_params, grid_points_tc)
        probs_grid_np = K.numpy(probs_grid).reshape(grid_size, grid_size).real

        # Plot contour
        plt.contourf(xx, yy, probs_grid_np, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.6)

        # Plot data points
        plt.scatter(
            X[Y == 0, 0],
            X[Y == 0, 1],
            c="blue",
            s=20,
            edgecolors="k",
            label="Class 0",
        )
        plt.scatter(
            X[Y == 1, 0],
            X[Y == 1, 1],
            c="red",
            s=20,
            edgecolors="k",
            label="Class 1",
        )

        plt.title(f"Layers: {n_layers}\nLoss: {final_loss:.4f}")
        if idx == 0:
            plt.legend()

    plt.tight_layout()
    output_path = "examples/reproduce_papers/2019_Data_re_uploading/outputs/result.png"
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
