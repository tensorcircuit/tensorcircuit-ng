"""
Small PySCF -> TenCirPauli -> TensorCircuit/JAX VQE bridge example.

This is a short integration preview. For the complete chemistry workflow,
correctness checks, and benchmark study, see:
https://github.com/tensorcircuit/TenCirPauli/blob/main/examples/quantum_chemistry_pyscf.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tencirpauli.integrations.pyscf import from_molecule
from tencirpauli.integrations.tensorcircuit import backend_mvp
import tensorcircuit as tc

NLAYERS = 2


def main() -> None:
    """
    Run a small H2 VQE with TensorCircuit/JAX for circuit construction.
    """
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    from pyscf import gto

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    fermion_hamiltonian = from_molecule(molecule)
    pauli_hamiltonian = fermion_hamiltonian.map_fermions("jordan_wigner")
    apply_hamiltonian = backend_mvp(
        pauli_hamiltonian.backend_mvp_plan(), backend=tc.backend
    )

    def state(parameters: jax.Array) -> jax.Array:
        circuit = tc.Circuit(pauli_hamiltonian.nqubits)
        circuit.x(0)
        circuit.x(1)
        for layer in range(NLAYERS):
            for qubit in range(pauli_hamiltonian.nqubits):
                circuit.ry(qubit, theta=parameters[layer, qubit])
            for qubit in range(pauli_hamiltonian.nqubits - 1):
                circuit.cnot(qubit, qubit + 1)
        return circuit.state()

    def energy(parameters: jax.Array) -> jax.Array:
        wavefunction = state(parameters)
        return jnp.real(jnp.vdot(wavefunction, apply_hamiltonian(wavefunction)))

    parameters = jnp.zeros((NLAYERS, pauli_hamiltonian.nqubits), dtype=jnp.float64)
    value_and_grad = jax.jit(jax.value_and_grad(energy))
    for _ in range(20):
        value, gradient = value_and_grad(parameters)
        parameters = parameters - 0.08 * gradient

    print(f"H2 energy: {float(value):.8f}")
    print(f"mapped Pauli terms: {pauli_hamiltonian.term_count}")


if __name__ == "__main__":
    main()
