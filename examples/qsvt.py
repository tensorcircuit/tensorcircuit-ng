"""
Quantum Singular Value Transformation (QSVT) example.

This script contains a small QSVT example, running a basic identity-target
demo when executed directly.
"""

from typing import Any, Callable, Optional

import numpy as np

# Imported from examples/qsp.py when running examples from this directory.
from qsp import fit_qsp_phases

import tensorcircuit as tc


def _real_tensor(value: object) -> Any:
    return tc.backend.cast(tc.backend.convert_to_tensor(value), tc.rdtypestr)


def _complex_tensor(value: object) -> Any:
    return tc.backend.cast(tc.backend.convert_to_tensor(value), tc.dtypestr)


def _system_qubit_count(matrix: object) -> int:
    shape = tc.backend.shape_tuple(_complex_tensor(matrix))
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("matrix must be a square matrix")
    dimension = shape[0]
    qubits = int(np.log2(dimension))
    if 2**qubits != dimension:
        raise ValueError("matrix dimension must be a power of two")
    return qubits


def block_encode(matrix: object) -> object:
    """Block encode a matrix A with ||A|| <= 1."""
    matrix_tensor = _complex_tensor(matrix)
    matrix_dag = tc.backend.adjoint(matrix_tensor)
    identity = tc.backend.eye(
        tc.backend.shape_tuple(matrix_tensor)[0], dtype=tc.dtypestr
    )
    sqrt_right = tc.backend.sqrtmh(identity - matrix_tensor @ matrix_dag, psd=True)
    sqrt_left = tc.backend.sqrtmh(identity - matrix_dag @ matrix_tensor, psd=True)
    top = tc.backend.concat([matrix_tensor, sqrt_right], axis=1)
    bottom = tc.backend.concat([sqrt_left, -matrix_dag], axis=1)
    return tc.backend.concat([top, bottom], axis=0)


def spectral_samples(matrix: np.ndarray) -> np.ndarray:
    spectrum = np.linalg.eigvalsh(np.asarray(matrix, dtype=tc.dtypestr))
    return np.asarray(np.real_if_close(spectrum), dtype=tc.rdtypestr)


def _transform_qsp_phases_to_qsvt(phases: object) -> Any:
    """Convert QSP phases to the reflection-convention QSVT phases."""
    qsp_phases = _real_tensor(phases)
    phase_count = tc.backend.shape_tuple(qsp_phases)[0]
    # The fitted QSP phases follow the single-signal convention, while the QSVT
    # circuit below alternates U and Udag reflections. These endpoint and middle
    # shifts align the two phase conventions before building the QSVT circuit.
    updates = np.full(phase_count, np.pi / 2, dtype=tc.rdtypestr)
    updates[0] = 3 * np.pi / 4 - (3 + phase_count % 4) * np.pi / 2
    updates[1:-1] = np.pi / 2
    updates[-1] = -np.pi / 4
    return qsp_phases + _real_tensor(updates)


def build_qsvt_circuit(
    phases: np.ndarray,
    unitary_matrix: object,
    total_qubits: int,
    inputs: Optional[object] = None,
) -> tc.Circuit:
    """Build the QSVT circuit using tensorcircuit."""
    phases_tensor = _real_tensor(phases)
    degree = tc.backend.shape_tuple(phases_tensor)[0] - 1
    unitary_tensor = _complex_tensor(unitary_matrix)
    unitary_dag = tc.backend.adjoint(unitary_tensor)
    circuit = tc.Circuit(total_qubits, inputs=inputs)
    unitary_qubits = tuple(range(total_qubits))
    for index in range(degree, -1, -1):
        circuit.rz(0, theta=-2.0 * phases_tensor[index])
        if index > 0:
            if (degree + index) % 2 == 1:
                circuit.any(*unitary_qubits, unitary=unitary_dag, name="Udag")
            else:
                circuit.any(*unitary_qubits, unitary=unitary_tensor, name="U")
    return circuit


def circuit_error(
    phases: object,
    matrix: object,
    target_function: Callable[[float], float],
) -> Any:
    """Validate optimized phases with the circuit realization only."""
    matrix_tensor = _complex_tensor(matrix)
    system_dimension = tc.backend.shape_tuple(matrix_tensor)[0]
    total_qubits = _system_qubit_count(matrix_tensor) + 1
    unitary_tensor = block_encode(matrix_tensor)
    eigvals, eigvecs = tc.backend.eigh(matrix_tensor)
    transformed_eigvals = _complex_tensor(target_function(tc.backend.real(eigvals)))
    target_tensor = (
        eigvecs @ tc.backend.diagflat(transformed_eigvals) @ tc.backend.adjoint(eigvecs)
    )
    target_norm = tc.backend.norm(target_tensor)
    circuit_matrix = build_qsvt_circuit(phases, unitary_tensor, total_qubits).matrix()
    approx_circ = circuit_matrix[:system_dimension, :system_dimension]
    return tc.backend.numpy(tc.backend.norm(approx_circ - target_tensor) / target_norm)


def fit_phases(
    target_function: Callable[[float], float],
    degree: int,
    x_samples: np.ndarray,
    max_steps: int = 1000,
    initial_phases: Optional[np.ndarray] = None,
) -> Any:
    """Optimize QSVT phases using the circuit-based QSP solver."""
    if initial_phases is None:
        initial_phases = np.linspace(-0.2, 0.2, degree + 1, dtype=tc.rdtypestr)
    qsp_phases, _, _, _ = fit_qsp_phases(
        target_function,
        x_samples=x_samples,
        initial_phases=initial_phases,
        maxiter=max_steps,
    )
    return _transform_qsp_phases_to_qsvt(qsp_phases)


def _example_target_matrix() -> np.ndarray:
    diagonal_entries = np.linspace(-0.75, 0.75, 8, dtype=tc.rdtypestr)
    return np.diag(diagonal_entries).astype(tc.dtypestr)


def main() -> None:
    """Run a multi-qubit QSVT demonstration."""
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    matrix = _example_target_matrix()
    target = lambda x: x
    sample_points = spectral_samples(matrix)

    print("Target polynomial: f(x) = x")
    print(f"Target matrix shape: {matrix.shape}")
    print("Running multi-qubit QSVT identity-target demo with the quantum solver.")
    phases = fit_phases(target, degree=3, x_samples=sample_points, max_steps=200)
    normalized_circuit_error = circuit_error(phases, matrix, target)
    phases_np = tc.backend.numpy(phases)
    print(f"QSVT phases: {phases_np}")
    print(f"QSVT circuit error: {normalized_circuit_error:.6e}")


if __name__ == "__main__":
    main()
