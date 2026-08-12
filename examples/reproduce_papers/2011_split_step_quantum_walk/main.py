"""Reproduction of "Topological phenomena in quantum walks"
Link: https://arxiv.org/abs/1112.1882
Description:
This script reproduces Figure 6 from the paper using TensorCircuit-NG.
It compares a topological interface with an interface between equal phases.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_POSITIONS = 128
POSITIONS = np.arange(-N_POSITIONS // 2, N_POSITIONS // 2)
STEP_VALUES = (0, 30, 60)
THETA1 = -np.pi / 2.0
THETA2_LEFT = 3.0 * np.pi / 4.0


def rotation_y(theta):
    """Return the spin rotation used by the split-step walk."""

    half_theta = theta / 2.0
    return np.array(
        [
            [np.cos(half_theta), -np.sin(half_theta)],
            [np.sin(half_theta), np.cos(half_theta)],
        ],
        dtype=np.complex64,
    )


def block_rotation(angles):
    """Embed one position-dependent spin rotation in the walker space."""

    operator = np.zeros((2 * N_POSITIONS, 2 * N_POSITIONS), dtype=np.complex64)
    for position_index, angle in enumerate(angles):
        start = 2 * position_index
        operator[start : start + 2, start : start + 2] = rotation_y(angle)
    return operator


def translation_up():
    """Translate spin-up components one site to the right."""

    operator = np.zeros((2 * N_POSITIONS, 2 * N_POSITIONS), dtype=np.complex64)
    for position_index in range(N_POSITIONS):
        target = (position_index + 1) % N_POSITIONS
        operator[2 * target, 2 * position_index] = 1.0
        operator[2 * position_index + 1, 2 * position_index + 1] = 1.0
    return operator


def translation_down():
    """Translate spin-down components one site to the left."""

    operator = np.zeros((2 * N_POSITIONS, 2 * N_POSITIONS), dtype=np.complex64)
    for position_index in range(N_POSITIONS):
        target = (position_index - 1) % N_POSITIONS
        operator[2 * target + 1, 2 * position_index + 1] = 1.0
        operator[2 * position_index, 2 * position_index] = 1.0
    return operator


def split_step_unitary(theta2_right):
    """Construct the inhomogeneous split-step unitary from Figure 6."""

    theta2 = 0.5 * (THETA2_LEFT + theta2_right)
    theta2 += 0.5 * (theta2_right - THETA2_LEFT) * np.tanh(POSITIONS / 3.0)
    first_rotation = block_rotation(np.full(N_POSITIONS, THETA1))
    second_rotation = block_rotation(theta2)
    return translation_down() @ second_rotation @ translation_up() @ first_rotation


def initial_state():
    """Prepare a spin-up walker at the interface position."""

    state = np.zeros(2 * N_POSITIONS, dtype=np.complex64)
    state[2 * (N_POSITIONS // 2)] = 1.0
    return tc.backend.convert_to_tensor(state)


def evolve_trajectory(state, unitary):
    """Evolve the walker and record the requested time slices."""

    recorded = [state]
    previous_step = 0
    for step in STEP_VALUES[1:]:
        state = tc.backend.scan(
            lambda current, _: apply_unitary(unitary, current),
            tc.backend.zeros((step - previous_step,)),
            state,
        )
        recorded.append(state)
        previous_step = step
    return tc.backend.stack(recorded)


def apply_unitary(unitary, state):
    """Apply a matrix to a backend vector through a two-dimensional product."""

    column = tc.backend.reshape(state, [-1, 1])
    return tc.backend.reshape(tc.backend.matmul(unitary, column), [-1])


def evolve(unitary):
    """Run the JIT-compiled Floquet ED evolution."""

    state = initial_state()
    runner = tc.backend.jit(evolve_trajectory)
    states = runner(state, unitary)
    states = np.asarray(tc.backend.numpy(states))
    states = states.reshape(len(STEP_VALUES), N_POSITIONS, 2)
    return np.sum(np.abs(states) ** 2, axis=2)


def plot_results(topological, trivial):
    """Plot the probability profiles corresponding to Figure 6."""

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), sharex=True, sharey=True)
    profiles = (
        (topological, "Different winding numbers"),
        (trivial, "Same winding number"),
    )
    for row, (probabilities, title) in enumerate(profiles):
        for column, (step, probability) in enumerate(zip(STEP_VALUES, probabilities)):
            axes[row, column].plot(
                POSITIONS, probability, color="#1769aa", linewidth=1.5
            )
            axes[row, column].axvline(0.0, color="black", linestyle=":", linewidth=0.8)
            axes[row, column].set_title(f"{title}\nstep {step}", fontsize=10)
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Probability")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 1].set_xlabel("Position")
    axes[1, 2].set_xlabel("Position")
    axes[0, 0].set_ylim(0.0, 0.55)
    axes[0, 0].set_xlim(-20.0, 20.0)
    fig.suptitle("Split-step quantum walk at an inhomogeneous interface", y=1.01)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "result.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    """Run the topological and trivial-interface quantum walks."""

    topological = evolve(tc.backend.convert_to_tensor(split_step_unitary(np.pi / 4.0)))
    trivial = evolve(
        tc.backend.convert_to_tensor(split_step_unitary(11.0 * np.pi / 8.0))
    )
    plot_results(topological, trivial)


if __name__ == "__main__":
    main()
