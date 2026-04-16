"""
Quantum Singular Value Transformation (QSVT) example.

This script contains a small QSVT solver and runs a basic identity-target
demo when executed directly.
"""

from typing import Callable, Tuple

import numpy as np
from scipy.optimize import minimize

import tensorcircuit as tc


def hermitian_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute the Hermitian square root of a positive semi-definite matrix.

    :param matrix: A positive semi-definite matrix.
    :returns: The Hermitian square root of the matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 0, None)
    sqrt_matrix = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    if np.max(np.abs(np.imag(sqrt_matrix))) < 1e-10:
        sqrt_matrix = np.real(sqrt_matrix)
    return sqrt_matrix


def block_encode(matrix: np.ndarray) -> np.ndarray:
    """Block encode a 2x2 matrix A with ||A|| <= 1.

    Returns the unitary:
    U = [[A, sqrt(I - A A†)], [sqrt(I - A† A), -A†]]

    :param matrix: A 2x2 matrix with operator norm <= 1.
    :returns: 4x4 block-encoded unitary matrix.
    """
    identity = np.eye(2, dtype=complex)
    sqrt_right = hermitian_sqrt(identity - matrix @ matrix.conj().T)
    sqrt_left = hermitian_sqrt(identity - matrix.conj().T @ matrix)
    return np.block([[matrix, sqrt_right], [sqrt_left, -matrix.conj().T]])


class QSVTSolver:
    """Solver for Quantum Singular Value Transformation.

    :param degree: Polynomial degree (number of phases = degree + 1).
    :param matrix: Input 2x2 matrix for the transformation.
    :param target_function: Target matrix function to approximate.
    :param solver_type: Optimization method, 'classical' or 'variational'.
    """

    def __init__(
        self,
        degree: int,
        matrix: np.ndarray,
        target_function: Callable[[np.ndarray], np.ndarray],
        solver_type: str = "classical",
    ) -> None:
        self.degree = degree
        self.matrix = matrix
        self.target_function = target_function
        self.solver_type = solver_type

    def build_qsvt_unitary(self, phases: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        """Build the QSVT unitary with alternating U and U†."""
        identity_2 = np.eye(2, dtype=complex)
        unitary_dag = unitary.conj().T
        degree = len(phases) - 1

        def rotation(phi: float) -> np.ndarray:
            return np.kron(np.diag([np.exp(-1j * phi), np.exp(1j * phi)]), identity_2)

        result_unitary = rotation(phases[0])
        for k in range(1, degree + 1):
            if degree % 2 == 1:
                gate = unitary if k % 2 == 1 else unitary_dag
            else:
                gate = unitary_dag if k % 2 == 1 else unitary
            result_unitary = gate @ result_unitary
            result_unitary = rotation(phases[k]) @ result_unitary
        return result_unitary

    def build_qsvt_circuit(
        self, phases: np.ndarray, unitary_matrix: np.ndarray
    ) -> tc.Circuit:
        """Build the QSVT circuit using tensorcircuit."""
        import torch

        degree = len(phases) - 1
        unitary_dag = unitary_matrix.conj().T
        if not isinstance(unitary_matrix, torch.Tensor):
            unitary_matrix = torch.tensor(unitary_matrix, dtype=torch.complex128)
        if not isinstance(unitary_dag, torch.Tensor):
            unitary_dag = torch.tensor(unitary_dag, dtype=torch.complex128)
        circuit = tc.Circuit(2)
        for k in range(0, degree + 1):
            circuit.rz(0, theta=2.0 * phases[k])
            if k < degree:
                if (degree - k - 1) % 2 == 0:
                    circuit.any(0, 1, unitary=unitary_matrix, name="U")
                else:
                    circuit.any(0, 1, unitary=unitary_dag, name="Udag")
        return circuit

    def validate_phases(
        self,
        phases: np.ndarray,
        unitary_a: np.ndarray,
        target_f_a: np.ndarray,
    ) -> Tuple[float, float]:
        """Validate optimized phases by comparing matrix and circuit results."""
        unitary_mat = self.build_qsvt_unitary(phases, unitary_a)
        approx_mat = unitary_mat[0:2, 0:2]
        err_classical = np.linalg.norm(approx_mat - target_f_a, "fro") / np.linalg.norm(
            target_f_a, "fro"
        )
        circuit = self.build_qsvt_circuit(phases, unitary_a)
        approx_circ = circuit.matrix()[0:2, 0:2]
        err_circuit = np.linalg.norm(approx_circ - target_f_a, "fro") / np.linalg.norm(
            target_f_a, "fro"
        )
        return err_classical, err_circuit

    def fit_phases(self, max_steps: int = 1000, lr: float = 0.01) -> np.ndarray:
        """Optimize QSVT phases to approximate the target function."""
        unitary_a = block_encode(self.matrix)
        target_f_a = self.target_function(self.matrix)
        if self.solver_type == "classical":

            def loss(phases: np.ndarray) -> float:
                unitary = self.build_qsvt_unitary(phases, unitary_a)
                approx_f_a = unitary[0:2, 0:2]
                return (
                    np.linalg.norm(approx_f_a - target_f_a, "fro") ** 2
                    / np.linalg.norm(target_f_a, "fro") ** 2
                )

            result = minimize(
                loss,
                np.random.randn(self.degree + 1) * 0.1,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * (self.degree + 1),
                options={"maxiter": 10000},
            )
            return result.x
        if self.solver_type == "variational":
            import torch

            phases = [
                torch.tensor(
                    np.random.randn() * 0.1, dtype=torch.float64, requires_grad=True
                )
                for _ in range(self.degree + 1)
            ]
            optimizer = torch.optim.Adam(phases, lr=lr)
            unitary_a_torch = torch.tensor(unitary_a, dtype=torch.complex128)
            unitary_a_dag_torch = unitary_a_torch.conj().T
            target_f_a_torch = torch.tensor(target_f_a, dtype=torch.complex128)
            degree = self.degree
            for step in range(max_steps):
                optimizer.zero_grad()
                circuit = tc.Circuit(2)
                for k in range(0, degree + 1):
                    circuit.rz(0, theta=2.0 * phases[k])
                    if k < degree:
                        if (degree - k - 1) % 2 == 0:
                            circuit.any(0, 1, unitary=unitary_a_torch, name="U")
                        else:
                            circuit.any(0, 1, unitary=unitary_a_dag_torch, name="Udag")
                unitary = circuit.matrix()
                approx_f_a = unitary[0:2, 0:2]
                loss = (
                    torch.norm(approx_f_a - target_f_a_torch, p="fro") ** 2
                    / torch.norm(target_f_a_torch, p="fro") ** 2
                )
                loss.backward()
                optimizer.step()
                if step % 100 == 0:
                    print(f"QSVT variational step {step}, loss={loss.item():.6f}")
            return np.array([phase.detach().cpu().numpy() for phase in phases])
        raise ValueError(f"Unknown solver_type: {self.solver_type}")


def main() -> None:
    """Run a minimal QSVT demonstration."""
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex) / np.sqrt(2.0)
    solver = QSVTSolver(degree=3, matrix=matrix, target_function=lambda a: a)
    phases = solver.fit_phases(max_steps=200)
    classical_error, circuit_error = solver.validate_phases(
        phases, block_encode(matrix), matrix
    )
    print("QSVT phases:", phases)
    print("QSVT classical error:", float(classical_error))
    print("QSVT circuit error:", float(circuit_error))


if __name__ == "__main__":
    main()