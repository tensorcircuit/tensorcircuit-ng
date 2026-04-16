"""
Quantum Singular Value Transformation (QSVT) example.

This script contains a small QSVT solver and runs a basic identity-target
demo when executed directly.
"""

from typing import Callable, Optional, Tuple

import numpy as np

from qsp import fit_qsp_phases_variational

import tensorcircuit as tc


def _complex_tensor(value: object) -> object:
    return tc.backend.cast(tc.backend.convert_to_tensor(value), tc.dtypestr)


def _system_qubit_count(matrix: object) -> int:
    shape = tuple(matrix.shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("matrix must be a square matrix")
    dimension = shape[0]
    qubits = int(np.log2(dimension))
    if 2**qubits != dimension:
        raise ValueError("matrix dimension must be a power of two")
    return qubits


def _phase_rotation_for_system(phi: object, system_dimension: int) -> object:
    phase = tc.backend.exp(1.0j * _complex_tensor(phi))
    zero = _complex_tensor(0.0)
    ancilla_rotation = tc.backend.stack(
        [
            tc.backend.stack([phase, zero], axis=0),
            tc.backend.stack([zero, tc.backend.conj(phase)], axis=0),
        ],
        axis=0,
    )
    return tc.backend.kron(
        ancilla_rotation,
        tc.backend.eye(system_dimension, dtype=tc.dtypestr),
    )


def block_encode(matrix: object) -> object:
    """Block encode a matrix A with ||A|| <= 1."""
    matrix_tensor = _complex_tensor(matrix)
    matrix_dag = tc.backend.adjoint(matrix_tensor)
    identity = tc.backend.eye(matrix_tensor.shape[0], dtype=tc.dtypestr)
    sqrt_right = tc.backend.sqrtmh(identity - matrix_tensor @ matrix_dag, psd=True)
    sqrt_left = tc.backend.sqrtmh(identity - matrix_dag @ matrix_tensor, psd=True)
    top = tc.backend.concat([matrix_tensor, sqrt_right], axis=1)
    bottom = tc.backend.concat([sqrt_left, -matrix_dag], axis=1)
    return tc.backend.concat([top, bottom], axis=0)


def _leading_basis_inputs(system_dimension: int) -> object:
    return tc.backend.eye(2 * system_dimension, dtype=tc.dtypestr)[:system_dimension]


def spectral_samples(matrix: np.ndarray) -> np.ndarray:
    spectrum = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.complex128))
    return np.asarray(np.real_if_close(spectrum), dtype=np.float64)


def target_matrix(
    matrix: np.ndarray,
    target_function: Callable[[float], float],
) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(np.asarray(matrix, dtype=np.complex128))
    transformed_eigvals = np.asarray(
        [target_function(float(np.real(eigval))) for eigval in eigvals],
        dtype=np.complex128,
    )
    return eigvecs @ np.diag(transformed_eigvals) @ eigvecs.conj().T


def _transform_qsp_phases_to_qsvt(phases: np.ndarray) -> np.ndarray:
    """Convert QSP phases to the reflection-convention QSVT phases."""
    qsp_phases = np.asarray(phases, dtype=np.float64)
    updates = np.empty_like(qsp_phases)
    updates[0] = 3 * np.pi / 4 - (3 + len(qsp_phases) % 4) * np.pi / 2
    updates[1:-1] = np.pi / 2
    updates[-1] = -np.pi / 4
    return qsp_phases + updates


def _transition_unitary(
    degree: int,
    index: int,
    unitary: object,
    unitary_dag: object,
) -> object:
    if (degree + index) % 2 == 1:
        return unitary_dag
    return unitary


def build_qsvt_unitary(
    phases: np.ndarray,
    unitary: object,
    system_dimension: int,
) -> object:
    """Build the QSVT unitary with alternating U and U†."""
    phases_tensor = _complex_tensor(phases)
    unitary_tensor = _complex_tensor(unitary)
    unitary_dag = tc.backend.adjoint(unitary_tensor)
    degree = int(phases_tensor.shape[0]) - 1
    result_unitary = _phase_rotation_for_system(phases_tensor[0], system_dimension)
    for index in range(1, degree + 1):
        gate = _transition_unitary(degree, index, unitary_tensor, unitary_dag)
        result_unitary = (
            result_unitary
            @ gate
            @ _phase_rotation_for_system(phases_tensor[index], system_dimension)
        )
    return result_unitary


def build_qsvt_circuit(
    phases: np.ndarray,
    unitary_matrix: object,
    total_qubits: int,
    inputs: Optional[object] = None,
) -> tc.Circuit:
    """Build the QSVT circuit using tensorcircuit."""
    degree = len(phases) - 1
    unitary_tensor = _complex_tensor(unitary_matrix)
    unitary_dag = tc.backend.adjoint(unitary_tensor)
    circuit = tc.Circuit(total_qubits, inputs=inputs)
    unitary_qubits = tuple(range(total_qubits))
    for index in range(degree, -1, -1):
        circuit.rz(0, theta=-2.0 * phases[index])
        if index > 0:
            gate = _transition_unitary(degree, index, unitary_tensor, unitary_dag)
            if (degree + index) % 2 == 1:
                circuit.any(*unitary_qubits, unitary=gate, name="Udag")
            else:
                circuit.any(*unitary_qubits, unitary=gate, name="U")
    return circuit


def matrix_block(
    phases: np.ndarray,
    unitary_matrix: object,
    system_dimension: int,
) -> object:
    unitary = build_qsvt_unitary(phases, unitary_matrix, system_dimension)
    basis_inputs = _leading_basis_inputs(system_dimension)
    outputs = tc.backend.vmap(
        lambda input_state: unitary @ input_state, vectorized_argnums=0
    )(basis_inputs)
    return tc.backend.transpose(outputs[:, :system_dimension])


def circuit_block(
    phases: np.ndarray,
    unitary_matrix: object,
    system_dimension: int,
    total_qubits: int,
) -> object:
    basis_inputs = _leading_basis_inputs(system_dimension)
    columns = tc.backend.vmap(
        lambda input_state: build_qsvt_circuit(
            phases, unitary_matrix, total_qubits, inputs=input_state
        ).state()[:system_dimension],
        vectorized_argnums=0,
    )(basis_inputs)
    return tc.backend.transpose(columns)


def validate_phases(
    phases: np.ndarray,
    unitary_a: np.ndarray,
    target_f_a: np.ndarray,
    system_dimension: int,
    total_qubits: int,
) -> Tuple[float, float]:
    """Validate optimized phases by comparing matrix and circuit results."""
    unitary_tensor = _complex_tensor(unitary_a)
    target_tensor = _complex_tensor(target_f_a)
    target_norm = tc.backend.norm(target_tensor)
    approx_mat = matrix_block(phases, unitary_tensor, system_dimension)
    matrix_error = tc.backend.numpy(
        tc.backend.norm(approx_mat - target_tensor) / target_norm
    )
    approx_circ = circuit_block(phases, unitary_tensor, system_dimension, total_qubits)
    circuit_error = tc.backend.numpy(
        tc.backend.norm(approx_circ - target_tensor) / target_norm
    )
    return float(matrix_error), float(circuit_error)


def fit_phases(
    target_function: Callable[[float], float],
    degree: int,
    x_samples: np.ndarray,
    max_steps: int = 1000,
    initial_phases: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimize QSVT phases using the circuit-based QSP solver."""
    if initial_phases is None:
        initial_phases = np.linspace(-0.2, 0.2, degree + 1, dtype=np.float64)
    qsp_phases = fit_qsp_phases_variational(
        target_function,
        degree=degree,
        x_samples=np.asarray(x_samples, dtype=np.float64),
        initial_phases=initial_phases,
        maxiter=max_steps,
    )
    return _transform_qsp_phases_to_qsvt(qsp_phases)


def _example_target_matrix() -> np.ndarray:
    diagonal_entries = np.linspace(-0.75, 0.75, 8, dtype=np.float64)
    return np.diag(diagonal_entries).astype(np.complex128)


def main() -> None:
    """Run a multi-qubit QSVT demonstration."""
    tc.set_backend("jax")
    matrix = _example_target_matrix()
    target = lambda x: x
    sample_points = spectral_samples(matrix)
    system_qubits = _system_qubit_count(matrix)
    system_dimension = 2**system_qubits
    total_qubits = system_qubits + 1

    print("Target polynomial: f(x) = x")
    print(f"Target matrix shape: {matrix.shape}")
    print("Target matrix:")
    with np.printoptions(precision=3, suppress=True):
        print(matrix)
    print("Running multi-qubit QSVT identity-target demo with the quantum solver.")
    phases = fit_phases(target, degree=3, x_samples=sample_points, max_steps=200)
    expected_matrix = target_matrix(matrix, target)
    matrix_error, circuit_error = validate_phases(
        phases,
        block_encode(matrix),
        expected_matrix,
        system_dimension,
        total_qubits,
    )
    print(f"QSVT phases: {phases}")
    print(f"QSVT matrix error: {float(matrix_error):.6e}")
    print(f"QSVT circuit error: {float(circuit_error):.6e}")


if __name__ == "__main__":
    main()
