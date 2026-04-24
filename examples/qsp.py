"""
Quantum Signal Processing (QSP) example.

This script provides simple helper functions for constructing and fitting
QSP unitaries and runs a circuit-based fitting demo when executed directly.
"""

from typing import Any, Callable, Union

import numpy as np
import optax
from scipy.optimize import minimize

import tensorcircuit as tc

MinimizeResult = tuple[Any, Any, bool, int]
RNG = np.random.default_rng(42)


def _real_tensor(value: Union[float, np.ndarray, object]) -> Any:
    return tc.backend.cast(tc.backend.convert_to_tensor(value), tc.rdtypestr)


def _minimize_with_scipy(
    loss: Callable[[np.ndarray], object],
    initial_parameters: np.ndarray,
    maxiter: int,
    use_gradient: bool,
) -> object:
    optimized_loss = tc.backend.jit(loss)
    if tc.backend.name == "numpy":

        def numpy_loss(phases: np.ndarray) -> object:
            return optimized_loss(phases)

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

    def optimizer_update_step(
        gradients: object,
        state: object,
        parameters: object,
        loss_value: object,
        value_fn: Callable[[object], object],
    ) -> tuple[object, object]:
        return optimizer.update(
            gradients,
            state,
            parameters,
            value=loss_value,
            grad=gradients,
            value_fn=value_fn,
        )

    optimizer_update = tc.backend.jit(
        optimizer_update_step,
        static_argnums=(4,),
    )

    for _ in range(maxiter):
        loss_value, gradients = value_and_grad(raw_parameters)
        updates, optimizer_state = optimizer_update(
            gradients,
            optimizer_state,
            raw_parameters,
            loss_value,
            bounded_loss,
        )
        raw_parameters = optax.apply_updates(raw_parameters, updates)

    bounded_parameters = np.pi * tc.backend.tanh(raw_parameters)
    gradient_norm = tc.backend.numpy(tc.backend.norm(gradients))
    return (
        bounded_parameters,
        tc.backend.numpy(loss_value),
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
        _real_tensor(result.x),
        result.fun,
        bool(result.success),
        int(getattr(result, "nit", maxiter)),
    )


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


def fit_qsp_phases(
    target: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    x_samples: np.ndarray,
    initial_phases: np.ndarray,
    maxiter: int,
) -> MinimizeResult:
    x_samples = np.asarray(x_samples, dtype=tc.rdtypestr)
    x_tensor = _real_tensor(x_samples)
    target_tensor = _real_tensor(target(x_samples))
    sample_count = _real_tensor(tc.backend.shape_tuple(x_tensor)[0])
    use_gradient = tc.backend.name != "numpy"

    def qsp_circuit_value(phases: Any, x: object) -> object:
        state = build_qsp_circuit(phases, x).state()
        return tc.backend.real(state[0])

    def loss(phases: np.ndarray) -> object:
        values = tc.backend.vmap(
            lambda sample: qsp_circuit_value(phases, sample), vectorized_argnums=0
        )(x_tensor)
        difference = values - target_tensor
        return tc.backend.tensordot(difference, difference, 1) / sample_count

    return minimize_backend_loss(
        loss,
        initial_phases,
        maxiter=maxiter,
        use_gradient=use_gradient,
    )


def main() -> None:
    """Run the circuit-based QSP demonstration."""
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    target = lambda x: x
    degree = 5
    maxiter = 500
    target_description = "f(x) = x"
    sample_points = np.linspace(-1.0, 1.0, 21, dtype=tc.rdtypestr)
    initial_phases = RNG.normal(loc=0.0, scale=0.1, size=degree + 1)

    print(f"Target function: {target_description}")
    print("Running circuit-based quantum-solver QSP demo.")
    quantum_phases, optimizer_loss, success, _ = fit_qsp_phases(
        target,
        x_samples=sample_points,
        initial_phases=initial_phases,
        maxiter=maxiter,
    )
    quantum_phases_np = tc.backend.numpy(quantum_phases)
    print(f"Quantum solver success: {success}")
    print(f"Quantum solver optimizer loss: {optimizer_loss:.6e}")
    print(f"Quantum solver phases: {quantum_phases_np}")


if __name__ == "__main__":
    main()
