"""
This script demonstrates the simulation of Fermionic Gaussian States (FGS) using TensorCircuit.
It features an integrated workflow to automatically discover the topological phase transition
of a Kitaev chain by differentiating through the ground state entanglement entropy.

Highlights:
1.  **FGS Simulation**: Efficient simulation of non-interacting fermions (Gaussian states).
2.  **Automatic Differentiation (AD)**: Differentiating through the eigensolver (`eigh`) to
    optimize physical parameters and find phase boundaries.
3.  **Entanglement Asymmetry**: Computing Renyi entanglement asymmetry to characterize
    symmetry breaking in different phases.
"""

import numpy as np
import matplotlib.pyplot as plt
import optax
import jax.numpy as jnp
import tensorcircuit as tc

# Set backend to JAX for AD capabilities
tc.set_backend("jax")
tc.set_dtype("complex128")
K = tc.backend


def get_kitaev_hamiltonian(L, t, delta, mu):
    """
    Constructs the quadratic Hamiltonian matrix for a Kitaev chain.

    H = -t sum (c_i^dag c_{i+1} + h.c.) + Delta sum (c_i c_{i+1} + h.c.) - mu sum c_i^dag c_i

    Args:
        L (int): System size.
        t (float): Hopping parameter.
        delta (float): Superconducting pairing parameter.
        mu (float): Chemical potential.

    Returns:
        Tensor: The Hamiltonian matrix (2L x 2L) in the Majorana basis format used by FGS.
    """
    # Initialize zero matrix
    hc = jnp.zeros((2 * L, 2 * L), dtype=jnp.complex128)

    # Add Hopping and Pairing terms
    # FGS helper methods return the matrix term for the Hamiltonian
    for i in range(L - 1):
        # Hopping: -t * (c_i^dag c_{i+1} + h.c.)
        hc += tc.FGSSimulator.hopping(-t, i, i + 1, L)

        # Pairing: Delta * (c_i c_{i+1} + h.c.)
        hc += tc.FGSSimulator.sc_pairing(delta, i, i + 1, L)

    # Add Chemical Potential
    # -mu * c_i^dag c_i
    for i in range(L):
        hc += tc.FGSSimulator.chemical_potential(-mu, i, L)

    epsilon = 1e-6
    hc += tc.FGSSimulator.hopping(epsilon, 0, L - 1, L)
    return hc


def get_entanglement_entropy(params, L):
    """
    Calculates the half-chain entanglement entropy for the ground state
    of the Kitaev Hamiltonian with given parameters.

    Args:
        params (tuple): (t, delta, mu)
        L (int): System size

    Returns:
        float: Von Neumann entropy of the half-chain.
    """
    t, delta, mu = params

    # 1. Build Hamiltonian (differentiable w.r.t mu)
    hc = get_kitaev_hamiltonian(L, t, delta, mu)

    # 2. Solve for Ground State (FGS)
    # The FGSSimulator diagonalizes hc to find the ground state alpha matrix.
    # This step involves eigh and is differentiable in JAX.
    sim = tc.FGSSimulator(L, hc=hc)

    # 3. Calculate Entropy
    # We want the entropy of the first half (A). We compute it by tracing out the rest (B).
    # Subsystem A: sites 0 to L//2 - 1
    # Subsystem B: sites L//2 to L - 1 (to be traced out)
    subsystem_b = list(range(L // 2, L))
    ee = sim.entropy(subsystems_to_trace_out=subsystem_b)

    return K.real(ee)


def detection_workflow():
    """
    Main workflow to detect phase transition.
    """
    L = 200
    t = 1.0
    delta = 1.0
    # Theoretical critical points are at mu = +/- 2t for this model.
    # We will search for the one at +2.0.

    print(f"\n=== Automated Topological Phase Transition Detection (L={L}) ===")
    print("Model: Kitaev Chain (Hopping t=1.0, Pairing Delta=1.0)")
    print(
        "Objective: Find critical Chemical Potential (mu) via AD on Entanglement Entropy."
    )

    # --- Part 1: Phase Scanning & Visualization ---
    print("\n--- Step 1: Scanning Phase Diagram ---")
    mus = np.linspace(0.0, 4.0, 80)
    mu_tensor = K.convert_to_tensor(mus)

    def get_quantities(mu_val):
        # 1. Calculate EE and its gradient via AD
        ee_func = lambda m: get_entanglement_entropy((t, delta, m), L)
        ee, grad_ee = K.value_and_grad(ee_func)(mu_val)

        # 2. Calculate Renyi Entanglement Asymmetry
        hc = get_kitaev_hamiltonian(L, t, delta, mu_val)
        sim = tc.FGSSimulator(L, hc=hc)
        subsystem_b = list(range(L // 2, L))
        asym = sim.renyi_entanglement_asymmetry(
            n=2, subsystems_to_trace_out=subsystem_b
        )

        return ee, grad_ee, asym

    # Use vmap to efficiently calculate all quantities in parallel
    v_get_quantities = K.jit(K.vmap(get_quantities))

    # Returns tuple of arrays (ees, grad_ees, asyms)
    ees, d_ees_ad, asyms = v_get_quantities(mu_tensor)
    print(f"Scanned {len(mus)} points")

    # Convert to numpy for plotting
    ees_np = K.numpy(K.real(ees))
    d_ees_np = K.numpy(K.real(d_ees_ad))
    asyms_np = K.numpy(K.real(asyms))

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot 1: Entanglement Entropy
    color1 = "tab:blue"
    ax1.set_xlabel(r"Chemical Potential $\mu$")
    ax1.set_ylabel(r"Ent. Entropy $S_A$", color=color1)
    ax1.plot(mus, ees_np, color=color1, linewidth=2, label="$S_A$")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Susceptibility (AD Gradient)
    ax2 = ax1.twinx()
    color2 = "tab:red"
    # Offset the secondary axis needed?
    # Let's put Asymmetry on the same axis as Gradient or a 3rd axis?
    # A 3rd axis is tricky in standard matplotlib.
    # Let's normalize or just plot both on right axis if ranges are compatible, or use relative scales.
    # Actually, Asymmetry is order 1, Gradient can be large near transition.

    # Let's try to plot Asymmetry on a separate subplot or just keep it simple.
    # User asked for "visualized in the figure".
    # Let's make ax2 for Gradient and Asymmetry with legend.

    ax2.set_ylabel(
        r"Susceptibility $|dS/d\mu|$ & Asymmetry $\Delta S_A^{(2)}$", color="black"
    )
    lns2 = ax2.plot(
        mus,
        np.abs(d_ees_np),
        color=color2,
        linestyle="--",
        alpha=0.7,
        label=r"Susceptibility $|dS/d\mu|$ (AD)",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Plot 3: Renyi Asymmetry
    color3 = "tab:green"
    lns3 = ax2.plot(
        mus,
        asyms_np,
        color=color3,
        linestyle="-.",
        linewidth=2,
        label=r"Renyi Asymmetry $\Delta S_A^{(2)}$",
    )

    # Combine legends
    lns = ax1.get_lines() + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    plt.axvline(2.0, color="gray", linestyle=":", label="Theoretical Critical Point")
    plt.title(
        f"Topological Phase Transition (L={L})\nDetected via EE Gradient (AD) & Symmetry Breaking"
    )
    fig.tight_layout()
    plt.savefig("fgs_phase_transition.pdf")
    print("Saved phase diagram to 'fgs_phase_transition.pdf'")
    plt.close()

    # --- Part 2: Automatic Discovery of Critical Point ---
    print("\n--- Step 2: Automatic Optimization for Critical Point ---")

    # We optimize to maximize the Entanglement Entropy directly.
    # The EE peaks at the quantum critical point.

    @K.jit
    def loss_func(mu_val):
        # Minimize negative entropy -> Maximize entropy
        ee = get_entanglement_entropy((t, delta, mu_val), L)
        return -K.real(ee)

    # Start optimization from the topological phase side
    mu_param = K.convert_to_tensor(1.0)

    # Use optax for gradient-based optimization
    optimizer = optax.adam(learning_rate=0.03)
    opt_state = optimizer.init(mu_param)

    history_mu = []
    best_mu = mu_param

    print(f"Initial Guess: mu = {mu_param:.4f}")
    for i in range(80):
        # Value and Grad of the specific loss function
        loss_val, grads = K.value_and_grad(loss_func)(mu_param)

        updates, opt_state = optimizer.update(grads, opt_state)
        mu_param = optax.apply_updates(mu_param, updates)
        history_mu.append(K.numpy(mu_param).item())

        best_mu = mu_param

        if i % 10 == 0:
            print(f"Iter {i}: mu = {mu_param:.4f}, Loss = {loss_val:.6f}")

    print(f"Run completed.")
    print(f"Optimized Critical Point (Max Entropy): mu = {best_mu:.4f}")
    print(f"Error from Theoretical (2.0): {abs(best_mu - 2.0):.4f}")


if __name__ == "__main__":
    detection_workflow()
