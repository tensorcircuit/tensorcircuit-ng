import time
import optax
import tensorcircuit as tc
from tensorcircuit.templates.lattice import SquareLattice, get_compatible_layers
from tensorcircuit.templates.hamiltonians import heisenberg_hamiltonian

# Use JAX for high-performance, especially on GPU.
K = tc.set_backend("jax")
tc.set_dtype("complex64")
# On Windows, cotengra's multiprocessing can cause issues.
tc.set_contractor("cotengra-8192-8192", parallel=False)


def run_vqe():
    n, m, nlayers = 4, 4, 6
    lattice = SquareLattice(size=(n, m), pbc=True, precompute_neighbors=1)
    h = heisenberg_hamiltonian(lattice, j_coupling=[1.0, 1.0, 0.8])  # Jx, Jy, Jz
    nn_bonds = lattice.get_neighbor_pairs(k=1, unique=True)
    gate_layers = get_compatible_layers(nn_bonds)

    def singlet_init(circuit):
        # A good initial state for Heisenberg ground state search
        nq = circuit._nqubits
        for i in range(0, nq - 1, 2):
            j = (i + 1) % nq
            circuit.X(i)
            circuit.H(i)
            circuit.cnot(i, j)
            circuit.X(j)
        return circuit

    def vqe_forward(param):
        """
        Defines the VQE ansatz and computes the energy expectation.
        The ansatz consists of nlayers of RZZ, RXX, and RYY entangling layers.
        """
        c = tc.Circuit(n * m)
        c = singlet_init(c)

        for i in range(nlayers):
            for layer in gate_layers:
                for j, k in layer:
                    c.rzz(int(j), int(k), theta=param[i, 0])
            for layer in gate_layers:
                for j, k in layer:
                    c.rxx(int(j), int(k), theta=param[i, 1])
            for layer in gate_layers:
                for j, k in layer:
                    c.ryy(int(j), int(k), theta=param[i, 2])

        return tc.templates.measurements.operator_expectation(c, h)

    vgf = K.jit(K.value_and_grad(vqe_forward))
    param = tc.backend.implicit_randn(stddev=0.02, shape=[nlayers, 3])
    optimizer = optax.adam(learning_rate=3e-3)
    opt_state = optimizer.init(param)

    @K.jit
    def train_step(param, opt_state):
        """A single training step, JIT-compiled for maximum speed."""
        loss_val, grads = vgf(param)
        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        return param, opt_state, loss_val

    print("Starting VQE optimization...")
    for i in range(1000):
        time0 = time.time()
        param, opt_state, loss = train_step(param, opt_state)
        time1 = time.time()
        if i % 10 == 0:
            print(
                f"Step {i:4d}: Loss = {loss:.6f} \t (Time per step: {time1 - time0:.4f}s)"
            )

    print("Optimization finished.")
    print(f"Final Loss: {loss:.6f}")


if __name__ == "__main__":
    run_vqe()
