"""
Comparison of different time evolution methods for the Heisenberg model

This example compares four different time evolution methods on an L=8 Heisenberg chain:
1. Exact diagonalization (ed_evol)
2. Krylov subspace method (krylov_evol)
3. ODE-based global evolution (ode_evol_global)
4. ODE-based local evolution (ode_evol_local)

The comparison is done for real-time evolution, and verifies that all methods
produce consistent results.
"""

import tensorcircuit as tc

# Set backend to JAX for ODE-based methods
K = tc.set_backend("jax")
tc.set_dtype("complex128")


def create_heisenberg_hamiltonian(n, sparse=True):
    """Create Heisenberg Hamiltonian for n sites."""
    g = tc.templates.graphs.Line1D(n, pbc=False)
    # Standard Heisenberg Hamiltonian: H = Σ (XX + YY + ZZ)
    h = tc.quantum.heisenberg_hamiltonian(g, hzz=1.0, hxx=1.0, hyy=1.0, sparse=sparse)
    return h


def create_initial_state(n):
    r"""
    Create initial Neel state
    :math:`\vert \uparrow\downarrow\uparrow\downarrow\uparrow\downarrow\uparrow\downarrow\rangle`
    for n sites.
    """
    c = tc.Circuit(n)
    # Apply X gates to odd sites to create Neel state
    c.x([i for i in range(1, n, 2)])
    return c.state()


def fidelity(state1, state2):
    # Normalize states
    state1 = state1 / K.norm(state1)
    state2 = state2 / K.norm(state2)
    # Calculate inner product
    inner_product = K.sum(K.conj(state1) * state2)
    # Return fidelity
    return K.abs(inner_product) ** 2


def time_evolution_comparison():
    """Compare different time evolution methods for L=8 Heisenberg model."""
    n = 8  # Number of sites
    print(f"Comparing time evolution methods for L={n} Heisenberg chain")

    # Create Hamiltonians - dense for ED and local ODE, sparse for others
    h_sparse = create_heisenberg_hamiltonian(n, sparse=True)
    h_dense = create_heisenberg_hamiltonian(n, sparse=False)
    print(f"Sparse Hamiltonian shape: {getattr(h_sparse, 'shape', 'N/A')}")
    print(f"Dense Hamiltonian shape: {h_dense.shape}")

    # Create initial state (Neel state)
    psi0 = create_initial_state(n)
    print(f"Initial state shape: {psi0.shape}")

    # Time points for evolution
    times = K.convert_to_tensor([0.0, 1.0, 3.0])
    # For real-time evolution, we need to multiply by -1j
    real_times = 1j * times

    print("\n1. Exact diagonalization (ed_evol):")
    states_ed = tc.timeevol.ed_evol(h_dense, psi0, real_times)
    print(f"Final state shape: {states_ed.shape}")

    print("\n2. Krylov subspace method (krylov_evol):")
    # Use a subspace dimension of 20
    states_krylov = tc.timeevol.krylov_evol(
        h_sparse, psi0, times, subspace_dimension=100
    )
    print(f"Final state shape: {states_krylov.shape}")

    print("\n3. ODE-based global evolution (ode_evol_global):")

    # Define time-dependent Hamiltonian function (constant in this case)
    def h_fun(t, *args):
        return h_sparse

    # Evolve using ode_evol_global
    states_global = tc.timeevol.ode_evol_global(h_fun, psi0, times)
    print(f"Final state shape: {states_global.shape}")

    print("\n4. ODE-based local evolution (ode_evol_local):")

    # For comparison with local evolution, we use the dense Hamiltonian
    def h_fun_local(t, *args):
        return h_dense

    # Index list for all sites
    index_list = list(range(n))

    # Evolve using ode_evol_local
    states_local = tc.timeevol.ode_evol_local(h_fun_local, psi0, times, index_list)
    print(f"Final state shape: {states_local.shape}")

    # Compare results using fidelity
    print("\nComparison of state fidelities:")
    print("Time\tED-Krylov\tED-Global\tED-Local")
    print("----\t---------\t---------\t--------")

    tolerance = 1e-5  # Slightly relaxed tolerance
    print(f"\nConsistency check (tolerance = {tolerance}):")

    for i, t in enumerate(times):
        # Calculate fidelity between states (|<psi1|psi2>|^2)

        # Compare ED with Krylov
        fid_krylov = fidelity(states_ed[i], states_krylov[i])
        diff_krylov = abs(1.0 - fid_krylov)
        print(
            f"t={t.real}: |1 - F(ED, Krylov)| = {diff_krylov:.2e}",
            "✓" if diff_krylov < tolerance else "✗",
        )

        # Compare ED with Global ODE
        fid_global = fidelity(states_ed[i], states_global[i])
        diff_global = abs(1.0 - fid_global)
        print(
            f"t={t.real}: |1 - F(ED, Global)| = {diff_global:.2e}",
            "✓" if diff_global < tolerance else "✗",
        )

        # Compare ED with Local ODE
        fid_local = fidelity(states_ed[i], states_local[i])
        diff_local = abs(1.0 - fid_local)
        print(
            f"t={t.real}: |1 - F(ED, Local)| = {diff_local:.2e}",
            "✓" if diff_local < tolerance else "✗",
        )

    return {
        "ed": states_ed,
        "krylov": states_krylov,
        "global_ode": states_global,
        "local_ode": states_local,
    }


if __name__ == "__main__":
    results = time_evolution_comparison()
    print("\nTime evolution comparison completed!")
