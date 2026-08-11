"""
Convert a TensorCircuit U(1) circuit to TenCirPauli execution.

This is a short integration preview for fixed-particle-number circuits. For
the complete 60-qubit comparison and benchmark study, see:
https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/u1_vqe_60q
"""

from __future__ import annotations

import math

import tencirpauli as tcp
import tensorcircuit as tc

NQUBITS = 60
PARTICLES = 2


def _hamiltonian() -> tcp.PauliOperator:
    terms: list[tuple[str, float]] = []
    for left in range(NQUBITS - 1):
        right = left + 1
        xx = ["I"] * NQUBITS
        yy = ["I"] * NQUBITS
        xx[left] = xx[right] = "X"
        yy[left] = yy[right] = "Y"
        terms.extend((("".join(xx), 0.5), ("".join(yy), 0.5)))
    return tcp.PauliOperator.from_terms(NQUBITS, terms)


def main() -> None:
    """
    Build the circuit with TensorCircuit and execute it natively.
    """
    tc.set_backend("numpy")
    tc.set_dtype("complex128")

    c = tc.U1Circuit(NQUBITS, k=PARTICLES, filled=[0, 1])
    for layer in range(3):
        for left in range(layer % 2, NQUBITS - 1, 2):
            c.iswap(left, left + 1, theta=0.08 + 0.01 * layer)
        for qubit in range(NQUBITS):
            c.rz(qubit, theta=0.01 * (qubit + layer))

    native = tcp.U1Circuit.from_circuit(c)
    result = native.value_and_grad(_hamiltonian())

    print(f"sector dimension: {math.comb(NQUBITS, PARTICLES)}")
    print(f"energy: {result.value:.8f}")
    print(f"gradient norm: {float((result.gradient**2).sum() ** 0.5):.3e}")


if __name__ == "__main__":
    main()
