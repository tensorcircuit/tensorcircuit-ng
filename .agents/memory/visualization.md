# Visualization

1.  **Non-Blocking Plots**:
    - Library plotting functions (e.g., `Lattice.show`) must accept an optional `ax` (Matplotlib Axis) argument.
    - **Protocol**: Draw on the provided `ax` and avoid calling `plt.show()` inside library functions. This allows users to embed plots in subplots and manage the figure lifecycle (saving/closing) themselves.
