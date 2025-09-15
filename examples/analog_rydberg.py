"""
Analog circuit simulation for Rydberg array
"""

import numpy as np
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex128")

# Define simulation parameters
# a compatible set of units: mus, mum, MHz
nqubits = 12
omega = 2.0 * np.pi
delta = 0.0
c6 = 8e8
evolution_time = 2.0

# Define the 1D lattice
chain = tc.templates.lattice.ChainLattice([nqubits], lattice_constant=10, pbc=False)

# Instantiate the AnalogCircuit
ac = tc.AnalogCircuit(nqubits)
ac.set_solver_options(
    ode_backend="diffrax", max_steps=20000
)  # more efficient and stable than the default one by jax
# 1. Create the sparsely excited state
ac.x([i for i in range(nqubits) if i % 4 == 0])

rydberg_hmatrix = tc.templates.hamiltonians.rydberg_hamiltonian(
    chain, omega=omega, delta=delta, c6=c6
)


# 2. Define the time-dependent Rydberg Hamiltonian
def rydberg_hamiltonian_func(t):
    # In this example, the Hamiltonian is time-independent, but it could be a function of t
    return rydberg_hmatrix


# 3. Add the analog evolution block
ac.add_analog_block(rydberg_hamiltonian_func, time=evolution_time)

# 4. apply some digital gates if needed, say for random measurements

# for i in range(nqubits):
#     j = np.random.choice(3)
#     if j == 1:
#         ac.h(i)
#     elif j == 2:
#         ac.rx(i, theta=-np.pi/4)

# 5. Sample from the final state in the computational basis
state = ac.state()
np.testing.assert_allclose(tc.backend.norm(state), 1, atol=1e-3)
sample = ac.sample(batch=1024, allow_state=True, format="count_dict_bin")
print("\nSampled bitstrings:\n", sample)
