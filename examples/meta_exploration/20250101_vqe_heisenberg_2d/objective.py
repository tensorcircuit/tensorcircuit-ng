import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO, PauliStringSum2Dense
import jax
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

# Enforce JAX 64-bit precision for high-precision ground state search if needed
jax.config.update('jax_enable_x64', True)
tc.set_dtype("complex128")
K = tc.set_backend("jax")

n = 12
shape = (3, 4)

def get_lattice_edges():
    edges = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            # vertical edge
            if i + 1 < shape[0]:
                edges.append((i * shape[1] + j, (i + 1) * shape[1] + j))
            # horizontal edge
            if j + 1 < shape[1]:
                edges.append((i * shape[1] + j, i * shape[1] + j + 1))
    return edges

edges = get_lattice_edges()

def get_hamiltonian_terms():
    paulis = []
    weights = []
    for e in edges:
        for p in [1, 2, 3]:  # X=1, Y=2, Z=3
            pauli = [0] * n
            pauli[e[0]] = p
            pauli[e[1]] = p
            paulis.append(pauli)
            weights.append(-1.0)  # Ferromagnetic coupling: J = 1 (energy H = -J * sum(S.S))
    return paulis, weights

paulis, weights = get_hamiltonian_terms()
coo_hamiltonian = PauliStringSum2COO(
    tc.backend.convert_to_tensor(paulis),
    tc.backend.convert_to_tensor(weights, dtype="complex128")
)

def exact_diagonalization():
    print('Computing ED Energy...')
    dense = PauliStringSum2Dense(
        tc.backend.convert_to_tensor(paulis),
        tc.backend.convert_to_tensor(weights, dtype="complex128")
    )
    dense_np = tc.backend.numpy(dense)
    vals, vecs = sla.eigsh(sp.csr_matrix(dense_np), k=1, which='SA')
    return np.real(vals[0])

ED_ENERGY = exact_diagonalization()
print(f"Computed exact diagonalization ground state energy: {ED_ENERGY}")

def evaluate(circuit_fn, params):
    """
    Evaluates the energy of the given circuit using sparse expectation.

    Args:
        circuit_fn: A function that takes `params` and returns a `tc.Circuit` instance.
        params: The parameters to feed into the circuit.

    Returns:
        float: The expected energy of the circuit.
    """
    c = circuit_fn(params)
    # Using real to handle complex values cleanly (even though energy is real)
    energy = tc.backend.real(tc.templates.measurements.sparse_expectation(c, coo_hamiltonian))
    return energy
