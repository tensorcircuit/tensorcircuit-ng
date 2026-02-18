"""
Reproduction of "Data re-uploading for a universal quantum classifier"
Link: https://arxiv.org/abs/1907.02085

Description:
This script reproduces Figure 6 from the paper using TensorCircuit.
It implements a single-qubit quantum classifier using data re-uploading.
The task is to classify points inside/outside a circle.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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


def loss(params, x, y, n_layers):
    """
    Calculate the weighted fidelity loss.
    params: flat parameters from scipy (or reshaped)
    x: (n_samples, 2)
    y: (n_samples,)
    """

    # We use vmap to compute expectation values for all samples in parallel
    def predict_point(xi):
        c = clf_circuit(params, xi, n_layers)
        # Probability of state |1>
        # TC z expectation is <Z> = P(0) - P(1) = 1 - 2P(1)
        # So P(1) = (1 - <Z>) / 2
        z_exp = c.expectation_ps(z=[0])
        p1 = (1.0 - z_exp) / 2.0
        return p1

    probs_1 = K.vmap(predict_point)(x)

    loss_val = K.mean((y - probs_1) ** 2)
    return K.real(loss_val)


def main():
    n_samples = 200
    X, Y = generate_circle_data(n_samples)

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

        # Scipy interface
        # We need to wrap loss to fix x, y, n_layers
        def loss_wrapper(p, x_in, y_in):
            return loss(p, x_in, y_in, n_layers)

        # Create the scipy compatible function
        loss_scipy = tc.interfaces.scipy_optimize_interface(
            loss_wrapper, shape=param_shape, gradient=True, jit=True
        )

        start_time = time.time()
        res = minimize(
            loss_scipy,
            init_params.flatten(),
            args=(X, Y),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000},
        )
        end_time = time.time()
        print(f"Optimization finished in {end_time - start_time:.2f}s. Loss: {res.fun}")

        opt_params = res.x.reshape(param_shape)

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
            def predict_point(xi):
                c = clf_circuit(p, xi, n_layers)
                z_exp = c.expectation_ps(z=[0])
                return (1.0 - z_exp) / 2.0

            return K.vmap(predict_point)(x_in)

        # Convert grid_points and opt_params to backend
        opt_params_tc = K.convert_to_tensor(opt_params)
        grid_points_tc = K.convert_to_tensor(grid_points)

        probs_grid = predict_batch(opt_params_tc, grid_points_tc)
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

        plt.title(f"Layers: {n_layers}\nLoss: {res.fun:.4f}")
        if idx == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig("examples/reproduce_papers/data_reuploading_results.png")
    print("Results saved to examples/reproduce_papers/data_reuploading_results.png")


if __name__ == "__main__":
    main()
