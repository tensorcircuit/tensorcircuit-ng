"""
Quantum Signal Processing (QSP) example.

This script provides simple helper functions for constructing and fitting
QSP unitaries and runs classical and circuit-based fitting demos when
executed directly.
"""

from typing import Callable, Optional, Union

import numpy as np
import optax
from scipy.optimize import minimize

import tensorcircuit as tc

MinimizeResult = tuple[np.ndarray, float, bool, int]


def _real_tensor(value: Union[float, np.ndarray, object]) -> object:
    return tc.backend.cast(tc.backend.convert_to_tensor(value), tc.rdtypestr)


def _complex_tensor(value: Union[complex, np.ndarray, object]) -> object:
    return tc.backend.cast(tc.backend.convert_to_tensor(value), tc.dtypestr)


def _phase_matrix(phi: object) -> object:
    phase = tc.backend.exp(1.0j * _complex_tensor(phi))
    zero = _complex_tensor(0.0)
    return tc.backend.stack(
        [
            tc.backend.stack([phase, zero], axis=0),
            tc.backend.stack([zero, tc.backend.conj(phase)], axis=0),
        ],
        axis=0,
    )


def _signal_rotation(x: object) -> object:
    x_tensor = tc.backend.clip(_real_tensor(x), -1.0, 1.0)
    x_complex = _complex_tensor(x_tensor)
    y_tensor = tc.backend.sqrt(tc.backend.relu(1.0 - x_tensor * x_tensor))
    off_diagonal = -1.0j * _complex_tensor(y_tensor)
    return tc.backend.stack(
        [
            tc.backend.stack([x_complex, off_diagonal], axis=0),
            tc.backend.stack([off_diagonal, x_complex], axis=0),
        ],
        axis=0,
    )


def _qsp_circuit_value(phases: np.ndarray, x: object) -> object:
    state = build_qsp_circuit(phases, x).state()
    return tc.backend.real(state[0])


def _minimize_with_scipy(
    loss: Callable[[np.ndarray], object],
    initial_parameters: np.ndarray,
    maxiter: int,
    use_gradient: bool,
) -> object:
    optimized_loss = tc.backend.jit(loss)
    if tc.backend.name == "numpy":

        def numpy_loss(phases: np.ndarray) -> float:
            return float(tc.backend.numpy(optimized_loss(phases)))

        loss_interface = numpy_loss
        jac = False
    else:
        loss_interface = tc.interfaces.scipy_interface(
            optimized_loss,
            shape=tuple(initial_parameters.shape),
            gradient=use_gradient,
        )
        jac = use_gradient

    return minimize(
        loss_interface,
        initial_parameters,
        method="L-BFGS-B",
        jac=jac,
        bounds=[(-np.pi, np.pi)] * len(initial_parameters),
        options={"maxiter": maxiter},
    )


def _minimize_with_optax(
    loss: Callable[[np.ndarray], object],
    initial_parameters: np.ndarray,
    maxiter: int,
) -> MinimizeResult:
    optimized_loss = tc.backend.jit(loss)
    clipped_initial_parameters = np.clip(
        initial_parameters / np.pi, -0.999999, 0.999999
    )
    raw_parameters = tc.backend.convert_to_tensor(
        np.arctanh(clipped_initial_parameters)
    )

    def bounded_loss(raw_phases: object) -> object:
        phases = np.pi * tc.backend.tanh(raw_phases)
        return optimized_loss(phases)

    value_and_grad = tc.backend.jit(tc.backend.value_and_grad(bounded_loss))
    optimizer = optax.lbfgs(learning_rate=1.0)
    optimizer_state = optimizer.init(raw_parameters)

    @tc.backend.jit
    def update_step(
        parameters: object,
        state: object,
    ) -> tuple[object, object, object, object]:
        loss_value, gradients = value_and_grad(parameters)
        updates, state = optimizer.update(
            gradients,
            state,
            parameters,
            value=loss_value,
            grad=gradients,
            value_fn=bounded_loss,
        )
        parameters = optax.apply_updates(parameters, updates)
        return parameters, state, loss_value, gradients

    loss_value = None
    gradients = None
    for _ in range(maxiter):
        raw_parameters, optimizer_state, loss_value, gradients = update_step(
            raw_parameters, optimizer_state
        )

    if loss_value is None or gradients is None:
        loss_value, gradients = value_and_grad(raw_parameters)

    bounded_parameters = np.pi * tc.backend.tanh(raw_parameters)
    gradient_norm = float(tc.backend.numpy(tc.backend.norm(gradients)))
    return (
        np.asarray(tc.backend.numpy(bounded_parameters), dtype=np.float64),
        float(tc.backend.numpy(loss_value)),
        bool(gradient_norm <= 1e-6),
        maxiter,
    )


def minimize_backend_loss(
    loss: Callable[[np.ndarray], object],
    initial_parameters: np.ndarray,
    maxiter: int,
    use_gradient: bool,
) -> MinimizeResult:
    """Minimize a scalar loss with backend-aware SciPy wrapping."""
    if tc.backend.name == "jax":
        return _minimize_with_optax(loss, initial_parameters, maxiter)
    result = _minimize_with_scipy(loss, initial_parameters, maxiter, use_gradient)
    return (
        np.asarray(result.x, dtype=np.float64),
        float(result.fun),
        bool(result.success),
        int(getattr(result, "nit", maxiter)),
    )


def _print_qsp_fit_summary(
    label: str,
    phases: np.ndarray,
    x_samples: np.ndarray,
    target_values: np.ndarray,
    optimizer_loss: Optional[float] = None,
    success: Optional[bool] = None,
) -> None:
    x_tensor = _real_tensor(np.asarray(x_samples, dtype=np.float64))
    fitted_values = tc.backend.vmap(
        lambda sample: qsp_polynomial(phases, sample), vectorized_argnums=0
    )(x_tensor)
    fitted_values = np.asarray(tc.backend.numpy(fitted_values), dtype=np.float64)
    reconstruction_loss = float(np.mean((fitted_values - target_values) ** 2))
    if success is not None:
        print(f"{label} success: {success}")
    if optimizer_loss is not None:
        print(f"{label} optimizer loss: {optimizer_loss:.6e}")
    print(f"{label} reconstruction loss: {reconstruction_loss:.6e}")
    print(f"{label} phases: {phases}")


def qsp_unitary(phases: np.ndarray, x: float) -> object:
    """Return the QSP unitary U(x; phases) for a single signal x.

    :param phases: Array of QSP phase angles.
    :param x: Signal value in [-1, 1].
    :returns: 2x2 QSP unitary matrix.
    """
    phases_tensor = _real_tensor(phases)
    signal_rotation = _signal_rotation(x)
    unitary = _phase_matrix(phases_tensor[0])
    for index in range(1, int(phases_tensor.shape[0])):
        unitary = unitary @ signal_rotation @ _phase_matrix(phases_tensor[index])
    return unitary


def qsp_polynomial(phases: np.ndarray, x: float) -> object:
    """Return the real QSP polynomial P(x) = Re[U_{00}(x)].

    :param phases: Array of QSP phase angles.
    :param x: Signal value in [-1, 1].
    :returns: Real part of the (0,0) element of the QSP unitary.
    """
    return tc.backend.real(qsp_unitary(phases, x)[0, 0])


def build_qsp_circuit(phases: np.ndarray, x: float) -> tc.Circuit:
    """Build a QSP circuit with given phases and signal value.

    :param phases: Array of QSP phase angles.
    :param x: Signal value in [-1, 1].
    :returns: tensorcircuit Circuit object.
    """
    circuit = tc.Circuit(1)
    theta = 2.0 * tc.backend.acos(_real_tensor(x))
    for index in range(len(phases) - 1, -1, -1):
        circuit.rz(0, theta=-2.0 * phases[index])
        if index > 0:
            circuit.rx(0, theta=theta)
    return circuit


def _fit_qsp_phases(
    target: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    degree: int,
    evaluator: Callable[[np.ndarray, object], object],
    x_samples: Optional[np.ndarray],
    initial_phases: Optional[np.ndarray],
    maxiter: int,
) -> MinimizeResult:
    if x_samples is None:
        n_samples = max(20, degree * 10)
        x_samples = np.linspace(-1.0, 1.0, n_samples)
    x_samples = np.asarray(x_samples, dtype=np.float64)
    if initial_phases is None:
        initial_phases = np.random.randn(degree + 1) * 0.1
    target_values = np.asarray([target(float(x)) for x in x_samples], dtype=np.float64)
    x_tensor = _real_tensor(x_samples)
    target_tensor = _real_tensor(target_values)
    use_gradient = tc.backend.name != "numpy"

    def loss(phases: np.ndarray) -> object:
        values = tc.backend.vmap(
            lambda sample: evaluator(phases, sample), vectorized_argnums=0
        )(x_tensor)
        difference = values - target_tensor
        return tc.backend.mean(difference * difference)

    return minimize_backend_loss(
        loss,
        initial_phases,
        maxiter=maxiter,
        use_gradient=use_gradient,
    )


def fit_qsp_phases(
    target: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    degree: int,
    x_samples: Optional[np.ndarray] = None,
    initial_phases: Optional[np.ndarray] = None,
    maxiter: int = 1000,
) -> MinimizeResult:
    """Optimize QSP phases to approximate target(x) on [-1, 1].

    :param target: Target function to approximate.
    :param degree: Polynomial degree (number of phases = degree + 1).
    :param x_samples: Sample points for fitting. Defaults to max(20, degree * 10)
        uniform points.
    :param initial_phases: Initial phase values. Defaults to small random values.
    :returns: Tuple of optimized phases, final loss, success flag, and iterations.
    """
    return _fit_qsp_phases(
        target,
        degree,
        qsp_polynomial,
        x_samples,
        initial_phases,
        maxiter,
    )


def fit_qsp_phases_variational(
    target: Callable[[float], float],
    degree: int,
    x_samples: Optional[np.ndarray] = None,
    initial_phases: Optional[np.ndarray] = None,
    maxiter: int = 500,
) -> np.ndarray:
    """Optimize QSP phases with vectorized circuit-based loss evaluation.

    :param target: Target function to approximate.
    :param degree: Polynomial degree (number of phases = degree + 1).
    :param x_samples: Sample points for fitting. Defaults to max(20, degree * 10)
        uniform points.
    :param initial_phases: Initial phase values. Defaults to small random values.
    :returns: Optimized phase array.
    """
    phases, _, _, _ = _fit_qsp_phases(
        target,
        degree,
        _qsp_circuit_value,
        x_samples,
        initial_phases,
        maxiter,
    )
    return phases


def main() -> None:
    """Run the QSP classical and circuit-based demonstrations."""
    tc.set_backend("jax")
    target = lambda x: x
    target_description = "f(x) = x"
    sample_points = np.linspace(-1.0, 1.0, 21)
    target_values = np.asarray(
        [target(float(x)) for x in sample_points], dtype=np.float64
    )

    print(f"Target function: {target_description}")
    print("Running classical QSP fitting demo.")
    phases, optimizer_loss, success, _ = fit_qsp_phases(
        target, degree=5, x_samples=sample_points
    )
    _print_qsp_fit_summary(
        "Classical solver",
        phases,
        sample_points,
        target_values,
        optimizer_loss=optimizer_loss,
        success=success,
    )

    print("Running circuit-based quantum-solver QSP demo.")
    quantum_phases = fit_qsp_phases_variational(
        target=target,
        degree=5,
        x_samples=sample_points,
    )
    _print_qsp_fit_summary(
        "Quantum solver",
        quantum_phases,
        sample_points,
        target_values,
    )


if __name__ == "__main__":
    main()
