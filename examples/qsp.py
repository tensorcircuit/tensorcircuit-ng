"""
Quantum Signal Processing (QSP) example.

This script provides simple helper functions for constructing and fitting
QSP unitaries and runs a small fitting demo when executed directly.
"""

from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import minimize

import tensorcircuit as tc


def qsp_unitary(phases: np.ndarray, x: float) -> np.ndarray:
    """Return the QSP unitary U(x; phases) for a single signal x.

    :param phases: Array of QSP phase angles.
    :param x: Signal value in [-1, 1].
    :returns: 2x2 QSP unitary matrix.
    """
    x = np.clip(x, -1.0, 1.0)
    y = np.sqrt(max(0.0, 1.0 - x * x))
    unitary = np.array(
        [[np.exp(1j * phases[0]), 0.0], [0.0, np.exp(-1j * phases[0])]],
        dtype=complex,
    )
    signal_rotation = np.array([[x, -1j * y], [-1j * y, x]], dtype=complex)
    for phi in phases[1:]:
        unitary = (
            unitary
            @ signal_rotation
            @ np.diag([np.exp(1j * phi), np.exp(-1j * phi)])
        )
    return unitary


def qsp_polynomial(phases: np.ndarray, x: float) -> float:
    """Return the real QSP polynomial P(x) = Re[U_{00}(x)].

    :param phases: Array of QSP phase angles.
    :param x: Signal value in [-1, 1].
    :returns: Real part of the (0,0) element of the QSP unitary.
    """
    return np.real(qsp_unitary(phases, x)[0, 0])


def build_qsp_circuit(phases: np.ndarray, x: float) -> tc.Circuit:
    """Build a QSP circuit with given phases and signal value.

    :param phases: Array of QSP phase angles.
    :param x: Signal value in [-1, 1].
    :returns: tensorcircuit Circuit object.
    """
    x = np.clip(x, -1.0, 1.0)
    circuit = tc.Circuit(1)
    theta = 2.0 * np.arccos(x)
    for i, phi in enumerate(reversed(phases)):
        circuit.rz(0, theta=-2.0 * phi)
        if i < len(phases) - 1:
            circuit.rx(0, theta=theta)
    return circuit


def qsp_circuit_state(phases: np.ndarray, x: float) -> np.ndarray:
    """Return the final state of the QSP circuit.

    :param phases: Array of QSP phase angles.
    :param x: Signal value in [-1, 1].
    :returns: Final quantum state vector.
    """
    return build_qsp_circuit(phases, x).state()


def fit_qsp_phases(
    target: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    degree: int,
    x_samples: Optional[np.ndarray] = None,
    initial_phases: Optional[np.ndarray] = None,
) -> object:
    """Optimize QSP phases to approximate target(x) on [-1, 1].

    :param target: Target function to approximate.
    :param degree: Polynomial degree (number of phases = degree + 1).
    :param x_samples: Sample points for fitting. Defaults to 81 uniform points.
    :param initial_phases: Initial phase values. Defaults to small random values.
    :returns: scipy OptimizeResult object with optimized phases in result.x.
    """
    if x_samples is None:
        x_samples = np.linspace(-1.0, 1.0, 81)
    x_samples = np.asarray(x_samples)
    if initial_phases is None:
        initial_phases = np.random.randn(degree + 1) * 0.1

    def loss(phases: np.ndarray) -> float:
        values = np.array([qsp_polynomial(phases, x) for x in x_samples])
        return np.mean((values - target(x_samples)) ** 2)

    result = minimize(
        loss,
        initial_phases,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * len(initial_phases),
        options={"maxiter": 1000},
    )
    return result


def fit_qsp_phases_variational(
    target: Callable[[float], float],
    degree: int,
    x_samples: Optional[np.ndarray] = None,
    initial_phases: Optional[np.ndarray] = None,
    lr: float = 0.01,
) -> np.ndarray:
    """Optimize QSP phases with circuit-based loss evaluation.

    :param target: Target function to approximate.
    :param degree: Polynomial degree (number of phases = degree + 1).
    :param x_samples: Sample points for fitting. Defaults to 81 uniform points.
    :param initial_phases: Initial phase values. Defaults to small random values.
    :param lr: Learning rate placeholder kept for API compatibility.
    :returns: Optimized phase array.
    """
    del lr
    if x_samples is None:
        x_samples = np.linspace(-1.0, 1.0, 81)
    x_samples = np.asarray(x_samples)
    if initial_phases is None:
        initial_phases = np.random.randn(degree + 1) * 0.1

    def loss(phases: np.ndarray) -> float:
        total_loss = 0.0
        for x in x_samples:
            state = build_qsp_circuit(phases, x).state()
            approximation = np.real(state[0])
            total_loss += (approximation - target(x)) ** 2
        return total_loss / len(x_samples)

    result = minimize(
        loss,
        initial_phases,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * len(initial_phases),
        options={"maxiter": 500},
    )
    return result.x


def main() -> None:
    """Run a minimal QSP fitting demonstration."""
    target = lambda x: x
    sample_points = np.linspace(-1.0, 1.0, 21)
    result = fit_qsp_phases(target, degree=5, x_samples=sample_points)
    print("QSP fit success:", result.success)
    print("QSP fit loss:", float(result.fun))
    print("QSP phases:", result.x)


if __name__ == "__main__":
    main()