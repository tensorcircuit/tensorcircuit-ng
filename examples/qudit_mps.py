r"""
This script performs a sanity check for qudit (d>2) circuits using the MPSCircuit class
by explicitly applying unitary gates.

It constructs a small qutrit (d=3) circuit and applies gates exclusively through the
unitary method of MPSCircuit, using built-in qudit gate matrices from tensorcircuit.quditgates.

The same circuit is built using QuditCircuit to provide a dense reference statevector.
The resulting statevectors from both approaches are compared to verify correctness
of MPS evolution for qudits.

The tested 3-qutrit circuit consists of the following gates:
1) Hadamard (Hd) on wire 0
2) Generalized Pauli-X (Xd) on wire 1
3) Controlled-sum (CSUM) on wires (0, 1)
4) Controlled-phase (CPHASE) on wires (1, 2)
5) RZ rotation on wire 2

Numerical closeness between the MPS and dense states is asserted.
"""

# import os
# import sys
#
# base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if base_dir not in sys.path:
#     sys.path.insert(0, base_dir)

import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

from tensorcircuit.quditgates import (
    x_matrix_func,
    h_matrix_func,
    csum_matrix_func,
    cphase_matrix_func,
    rz_matrix_func,
)


def build_qutrit_circuit_mps(n=3, d=3, theta=np.pi / 7):
    """
    Build an MPSCircuit of size n with local dimension d=3 and apply qudit gates via the unitary method.

    The circuit applies the following gates in order:
    - Hadamard (Hd) on qudit 0
    - Generalized Pauli-X (Xd) on qudit 1
    - Controlled-sum (CSUM) on qudits (0, 1)
    - Controlled-phase (CPHASE) on qudits (1, 2)
    - RZ rotation with angle theta on qudit 2 (acting nontrivially on a chosen level pair)

    :param int n: Number of qudits in the circuit.
    :param int d: Local dimension of each qudit (default is 3 for qutrits).
    :param float theta: Rotation angle for the RZ gate.
    :return: The constructed MPSCircuit with applied gates.
    :rtype: tc.MPSCircuit
    """
    omega = np.exp(2j * np.pi / d)

    mps = tc.MPSCircuit(n, dim=d)

    Hd = h_matrix_func(d, omega)
    mps.unitary(0, unitary=Hd, name="H_d", dim=d)  # <-- pass dim=d

    Xd = x_matrix_func(d)
    mps.unitary(1, unitary=Xd, name="X_d", dim=d)  # <-- pass dim=d

    CSUM = csum_matrix_func(d)  # (d*d, d*d)
    mps.unitary(0, 1, unitary=CSUM, name="CSUM", dim=d)  # <-- pass dim=d

    CPHASE = cphase_matrix_func(d, omega=omega)  # (d*d, d*d)
    mps.unitary(1, 2, unitary=CPHASE, name="CPHASE", dim=d)  # <-- pass dim=d

    RZ2 = rz_matrix_func(d, theta=theta, j=1)  # (d, d)
    mps.unitary(2, unitary=RZ2, name="RZ_lvl1", dim=d)  # <-- pass dim=d

    return mps


def build_qutrit_circuit_dense(n=3, d=3, theta=np.pi / 7):
    """
    Build the same qudit circuit using QuditCircuit with high-level named qudit gates.

    This circuit serves as a dense reference for cross-checking against the MPSCircuit.

    :param int n: Number of qudits in the circuit.
    :param int d: Local dimension of each qudit (default is 3 for qutrits).
    :param float theta: Rotation angle for the RZ gate.
    :return: The constructed QuditCircuit with applied gates.
    :rtype: tc.QuditCircuit
    """
    qc = tc.QuditCircuit(n, dim=d)

    qc.h(0)  # H_d
    qc.x(1)  # X_d

    qc.csum(0, 1)
    qc.cphase(1, 2)

    qc.rz(2, theta=tc.num_to_tensor(theta), j=1)

    return qc


def main():
    """
    Construct both MPSCircuit and QuditCircuit for a 3-qutrit circuit and verify their statevectors match.

    This function checks that the maximum absolute difference between the MPS and dense statevectors
    is below a small numerical tolerance, ensuring correctness of MPS evolution for qudits.
    """
    n, d = 3, 3
    theta = np.pi / 7

    # Build circuits
    mps = build_qutrit_circuit_mps(n=n, d=d, theta=theta)
    qc = build_qutrit_circuit_dense(n=n, d=d, theta=theta)

    np.testing.assert_allclose(mps.wavefunction(), qc.wavefunction())
    print("OK: MPSCircuit matches QuditCircuit for d=3 with unitary-applied gates.")


if __name__ == "__main__":
    main()
