"""
Reproduction of "Differentiable Maximum Likelihood Noise Estimation for Quantum Error Correction"
Link: https://arxiv.org/abs/2602.19722

Description:
This script reproduces the core idea of differentiable noise estimation using tensor networks.
It simulates a rotated surface code (scalable distance d) with bit-flip errors.
The goal is to estimate the error probability 'p' given a set of observed syndromes.
We construct the tensor network for the likelihood P(syndrome | p) using CopyNodes to handle hyperedges,
and use JAX for automatic differentiation to maximize the likelihood (minimize NLL).

Implementation Details:
- We use the Walsh-Hadamard Transform (WHT) to efficiently handle parity checks.
- Qubits are represented by CopyNodes (error variables e_j).
- Stabilizers are represented by CopyNodes (dual variables u_s) in the Fourier basis.
- Edges connecting Qubits and Stabilizers carry a Hadamard matrix H = [[1, 1], [1, -1]].
- The syndrome m_s modifies the input to the Stabilizer CopyNode: [1, (-1)**m_s].
- This structure avoids dense tensor construction and allows efficient path finding and JIT reuse.
"""

import os
import matplotlib.pyplot as plt
import tensornetwork as tn
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import tensorcircuit as tc

# Set backend
tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")

# -----------------------------------------------------------------------------
# 1. Surface Code Parameters and Helper Functions
# -----------------------------------------------------------------------------


def generate_surface_code(d):
    """
    Generate the standard rotated surface code topology of distance d (d must be odd).
    Returns: data_qubits_count, z_checks, x_checks

    data_qubits_count: integer, total number of data qubits.
    z_checks: list of lists, each sublist contains indices of qubits involved in a Z-stabilizer.
    x_checks: list of lists, each sublist contains indices of qubits involved in an X-stabilizer.
    """
    assert d % 2 != 0, "Distance d must be odd (e.g., 3, 5, 7)"
    data_qubits_count = d * d
    z_checks = []
    x_checks = []

    # r and c represent the top-left logical coordinate of a plaquette
    # Coordinate system:
    # Qubits are at (r, c) where r, c in [0, d-1]
    # Qubit index = r * d + c

    # Plaquettes are centered at (r+0.5, c+0.5) roughly.
    # We iterate possible check positions.
    for r in range(-1, d):
        for c in range(-1, d):
            # 1. Collect valid data qubits covered by this plaquette
            # A plaquette at (r,c) covers qubits at (r,c), (r,c+1), (r+1,c), (r+1,c+1)
            qubits = []
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                qr, qc = r + dr, c + dc
                if 0 <= qr < d and 0 <= qc < d:
                    qubits.append(qr * d + qc)

            # Ignore invalid corners (weight <= 1 is not a stabilizer in bulk/boundary logic usually,
            # but for rotated surface code, weight-2 boundaries are crucial.
            # Standard rotated surface code has no weight-1 stabilizers.
            if len(qubits) <= 1:
                continue

            # 2. Checkerboard coloring: Even sum is Z-check, Odd sum is X-check
            is_z_check = (r + c) % 2 == 0

            if is_z_check:
                # Z-check detects X errors.
                # In standard rotated surface code, Z checks are top/bottom boundaries?
                # Let's visualize d=3.
                # (0,0) is Z? r=0,c=0 -> sum=0 even -> Z.
                # Qubits: (0,0), (0,1), (1,0), (1,1). Weight 4. Center.
                #
                # Boundary conditions:
                # Z-checks generally have Rough Left/Right boundaries? No, Z-logical is Top-Bottom.
                # X-logical is Left-Right.
                #
                # If d=3:
                # r=-1, c=0 (sum -1 odd -> X)
                # r=-1, c=1 (sum 0 even -> Z). c=1 != -1 and != 2. Valid. Qubits: (0,1), (0,2). Top boundary.
                #
                if c != -1 and c != d - 1:
                    z_checks.append(qubits)
            else:
                # X-check detects Z errors.
                if r != -1 and r != d - 1:
                    x_checks.append(qubits)

    return data_qubits_count, z_checks, x_checks


def get_syndrome(error_configuration, checks=None):
    """
    Compute syndrome for a given error configuration.
    """
    if checks is None:
        checks = ACTIVE_CHECKS

    syndrome = []
    for stab in checks:
        parity = 0
        for q in stab:
            parity += error_configuration[q]
        syndrome.append(parity % 2)
    return jnp.array(syndrome, dtype=jnp.int32)


# -----------------------------------------------------------------------------
# 2. Tensor Network Construction (Differentiable & Efficient WHT)
# -----------------------------------------------------------------------------


def surface_code_likelihood(p, observed_syndrome, checks=None, num_qubits=None):
    """
    Compute the likelihood P(syndrome | p) using a tensor network with WHT.
    """
    if checks is None:
        checks = ACTIVE_CHECKS
    if num_qubits is None:
        num_qubits = NUM_QUBITS

    nodes = []

    # 1. Error Probability Tensors (Source for Qubits)
    # Shape (2,): [Prob(No Error), Prob(Error)] = [1-p, p]
    prob_vec = jnp.stack([1 - p, p])

    # 2. Hadamard Matrix for edges
    hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128)

    # 3. Qubit Nodes (CopyNodes)
    qubit_nodes = []
    for q in range(num_qubits):
        # Find which stabilizers involve this qubit
        involved_stabs = [i for i, stab in enumerate(checks) if q in stab]
        degree = len(involved_stabs) + 1  # +1 for the error source

        # Create CopyNode for variable x_q
        cn = tn.CopyNode(degree, 2, name=f"Q{q}")
        qubit_nodes.append(cn)

        # Connect Error Probability Source
        en = tn.Node(prob_vec, name=f"E{q}")
        nodes.append(en)
        en[0] ^ cn[0]  # Leg 0

    # 4. Stabilizer Nodes (Dual CopyNodes in Fourier Basis)
    stabilizer_nodes = []
    for i, stab in enumerate(checks):
        degree = len(stab) + 1  # +1 for syndrome factor

        # Create CopyNode for dual variable u_i
        sn = tn.CopyNode(degree, 2, name=f"S{i}")
        stabilizer_nodes.append(sn)

        # Syndrome Factor: [1, (-1)**m]
        m = observed_syndrome[i]
        factor = jnp.stack([1.0, (-1.0) ** m]).astype(jnp.complex128)
        syn_node = tn.Node(factor, name=f"Syn{i}")
        nodes.append(syn_node)

        # Connect Syndrome Factor to Stabilizer Node (Leg 0)
        syn_node[0] ^ sn[0]

    # 5. Connect Qubits and Stabilizers with Hadamard Edges
    qubit_leg_counters = [1] * num_qubits
    stab_leg_counters = [1] * len(checks)

    for i, stab in enumerate(checks):
        sn = stabilizer_nodes[i]
        for q_idx in stab:
            cn = qubit_nodes[q_idx]

            # Insert Hadamard Node on the edge
            h_node = tn.Node(hadamard, name=f"H_{i}_{q_idx}")
            nodes.append(h_node)

            # Connect: Stabilizer -> Hadamard -> Qubit
            sn[stab_leg_counters[i]] ^ h_node[0]
            h_node[1] ^ cn[qubit_leg_counters[q_idx]]

            stab_leg_counters[i] += 1
            qubit_leg_counters[q_idx] += 1

    # Collect all nodes
    all_nodes = nodes + qubit_nodes + stabilizer_nodes

    # Contract
    result_node = tc.contractor(all_nodes)

    return jnp.abs(result_node.tensor)


def make_loss_fn(checks, num_qubits):
    def loss_fn_inner(p_logit, observed_syndrome):
        p = jax.nn.sigmoid(p_logit)
        likelihood = surface_code_likelihood(p, observed_syndrome, checks, num_qubits)
        return -jnp.log(likelihood + 1e-10)

    return loss_fn_inner


# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------


def main():
    print("--- Differentiable Surface Code Noise Estimation (WHT Method) ---")

    # Ground Truth
    p_true = 0.05
    print(f"True Error Probability: {p_true}")

    # Generate Synthetic Data
    n_samples = 3000
    print(f"Generating {n_samples} synthetic syndrome samples...")

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    random_probs = jax.random.uniform(subkey, (n_samples, NUM_QUBITS))
    errors = (random_probs < p_true).astype(jnp.int32)

    syndromes = []
    for i in range(n_samples):
        syn = get_syndrome(errors[i], ACTIVE_CHECKS)
        syndromes.append(syn)
    syndromes = jnp.array(syndromes)

    # Optimization
    p_init = 0.1
    logit_init = inverse_sigmoid(p_init)

    optimizer = optax.adam(learning_rate=0.02)
    opt_state = optimizer.init(logit_init)
    params = logit_init

    loss_history = []
    p_history = []

    print("\nStarting Optimization (Maximum Likelihood)...")
    pbar = tqdm(range(100))

    # Pre-JIT the batch step
    @jax.jit
    def step_batch(current_params, batch_syndromes):
        def single_sample_loss(s):
            return loss_fn(current_params, s)

        losses = jax.vmap(single_sample_loss)(batch_syndromes)
        return jnp.mean(losses)

    step_batch_grad = jax.value_and_grad(step_batch)

    for _ in pbar:
        loss_val, grads = step_batch_grad(params, syndromes)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        current_p = jax.nn.sigmoid(params)
        loss_history.append(loss_val)
        p_history.append(current_p)

        pbar.set_description(f"Loss: {loss_val:.4f}, p_est: {current_p:.4f}")

    print(f"\nFinal Estimated p: {p_history[-1]:.4f}")
    print(f"True p: {p_true}")

    # Save Output Plot
    os.makedirs("outputs", exist_ok=True)

    # Ensure directory exists for the full path if needed,
    # but 'outputs' relative to current dir is fine as script is run from repo root usually.
    # The meta.yaml uses specific path, but here we just need to save to where we are running.
    # Assuming running from repo root.
    output_dir = "examples/reproduce_papers/2026_diff_qec_surface/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "convergence.png")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="NLL Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(p_history, label="Estimated p")
    plt.axhline(p_true, color="r", linestyle="--", label="True p")
    plt.xlabel("Iteration")
    plt.ylabel("p")
    plt.legend()

    plt.savefig(output_path)
    print(f"Convergence plot saved to '{output_path}'")


def inverse_sigmoid(y):
    return jnp.log(y / (1 - y))


if __name__ == "__main__":
    # Initialize Topology
    DISTANCE = 5  # Configurable: 3, 5, 7...
    DATA_QUBITS_COUNT, Z_CHECKS, X_CHECKS = generate_surface_code(DISTANCE)

    print(f"Generated Surface Code (d={DISTANCE}):")
    print(f"Data Qubits: {DATA_QUBITS_COUNT}")
    print(f"Z-Checks (detect X errors): {len(Z_CHECKS)}")
    print(f"X-Checks (detect Z errors): {len(X_CHECKS)}")

    # For demonstration, we simulate X errors detected by Z checks.
    ACTIVE_CHECKS = Z_CHECKS
    NUM_QUBITS = DATA_QUBITS_COUNT
    # Initialize JIT-compiled functions for the active configuration
    loss_fn = make_loss_fn(ACTIVE_CHECKS, NUM_QUBITS)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    main()
