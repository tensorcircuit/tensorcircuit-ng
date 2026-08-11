"""
Convert a TensorCircuit circuit to TenCirPauli Pauli propagation.

This is a short integration preview. For the complete propagation VQE,
truncation semantics, gradient checks, and benchmark study, see:
https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/pauli_propagation_julia_vqe
"""

from __future__ import annotations

import tencirpauli as tcp
import tensorcircuit as tc


def main() -> None:
    """
    Use TensorCircuit for circuit construction and native propagation.
    """
    tc.set_backend("numpy")
    tc.set_dtype("complex128")

    c = tc.Circuit(4)
    c.h(0)
    c.ry(1, theta=0.21)
    c.cnot(0, 1)
    c.rz(2, theta=-0.17)
    c.cnot(2, 3)

    circuit = tcp.PropagationCircuit.from_circuit(c)
    observable = tcp.PauliOperator.from_terms(4, [("ZZII", 1.0), ("IIZZ", 0.5)])
    result = circuit.value_and_grad(observable)

    print(f"propagated expectation: {result.value:.8f}")
    print(f"gate-angle gradient norm: {float((result.gradient**2).sum() ** 0.5):.3e}")


if __name__ == "__main__":
    main()
