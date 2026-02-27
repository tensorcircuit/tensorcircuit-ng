"""
Reproduction of "Differentiable Maximum Likelihood Noise Estimation for Quantum Error Correction"
Link: https://arxiv.org/abs/2602.19722

Description:
This script reproduces the core idea of differentiable noise estimation using tensor networks.
It simulates a rotated surface code (d=3) with bit-flip errors.
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

# Rotated Surface Code d=3
# Data Qubits: 9
# Stabilizers (Z-type, detecting X errors):
# s0: Z(0) Z(1) Z(3) Z(4)
# s1: Z(1) Z(2) Z(4) Z(5)
# s2: Z(3) Z(4) Z(6) Z(7)
# s3: Z(4) Z(5) Z(7) Z(8)

DATA_QUBITS = 9
STABILIZERS = [
    [0, 1, 3, 4],  # s0
    [1, 2, 4, 5],  # s1
    [3, 4, 6, 7],  # s2
    [4, 5, 7, 8],  # s3
]


def get_syndrome(error_configuration):
    """
    Compute syndrome for a given error configuration (binary array of length 9).
    1 means error, 0 means no error.
    Syndrome bit i is parity of errors on qubits in stabilizer i.
    """
    syndrome = []
    for stab in STABILIZERS:
        parity = 0
        for q in stab:
            parity += error_configuration[q]
        syndrome.append(parity % 2)
    return jnp.array(syndrome, dtype=jnp.int32)


# -----------------------------------------------------------------------------
# 2. Tensor Network Construction (Differentiable & Efficient WHT)
# -----------------------------------------------------------------------------


def surface_code_likelihood(p, observed_syndrome):
    """
    Compute the likelihood P(syndrome | p) using a tensor network with WHT.
    p: Error probability (scalar).
    observed_syndrome: array of shape (4,), values 0 or 1.
    """
    nodes = []

    # 1. Error Probability Tensors (Source for Qubits)
    # Shape (2,): [Prob(No Error), Prob(Error)] = [1-p, p]
    prob_vec = jnp.stack([1 - p, p])

    # 2. Hadamard Matrix for edges
    # H = [[1, 1], [1, -1]] / sqrt(2)? No, we are doing classical parity check.
    # The identity is: sum_{x_i} delta(sum x_i = m) prod P(x_i)
    # = sum_{x_i} 1/2 sum_{u=0,1} (-1)^{u (sum x_i - m)} prod P(x_i)
    # = 1/2 sum_u (-1)^{-um} prod_i (sum_{x_i} (-1)^{u x_i} P(x_i))
    # Note the factor 1/2. Since we want Likelihood, normalization matters less for optimization,
    # but let's stick to the structure. The standard implementation puts H on the edges.
    # H_unnormalized = [[1, 1], [1, -1]]
    # This transforms the basis.

    hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128)

    # 3. Qubit Nodes (CopyNodes)
    # Degree = (number of stabilizers acting on this qubit) + 1 (for error source)
    qubit_nodes = []
    for q in range(DATA_QUBITS):
        involved_stabs = [i for i, stab in enumerate(STABILIZERS) if q in stab]
        degree = len(involved_stabs) + 1  # +1 for the error source

        # Create CopyNode for variable x_q
        cn = tn.CopyNode(degree, 2, name=f"Q{q}")
        qubit_nodes.append(cn)

        # Connect Error Probability Source
        en = tn.Node(prob_vec, name=f"E{q}")
        nodes.append(en)
        en[0] ^ cn[0]  # Leg 0

    # 4. Stabilizer Nodes (Dual CopyNodes in Fourier Basis)
    # Degree = (number of qubits in this stabilizer) + 1 (for syndrome factor)
    stabilizer_nodes = []
    for i, stab in enumerate(STABILIZERS):
        degree = len(stab) + 1  # +1 for syndrome factor

        # Create CopyNode for dual variable u_i
        sn = tn.CopyNode(degree, 2, name=f"S{i}")
        stabilizer_nodes.append(sn)

        # Syndrome Factor: [1, (-1)**m]
        # This injects the syndrome constraint.
        # If m=0: [1, 1] (sum over u: 1 + ...)
        # If m=1: [1, -1] (sum over u: 1 - ...)
        # Wait, the formula is (-1)^{-um}. Since u,m in {0,1}, -um = um mod 2.
        # So factor is (-1)^{m} if u=1.
        m = observed_syndrome[i]
        factor = jnp.stack([1.0, (-1.0) ** m]).astype(jnp.complex128)
        syn_node = tn.Node(factor, name=f"Syn{i}")
        nodes.append(syn_node)

        # Connect Syndrome Factor to Stabilizer Node (Leg 0)
        syn_node[0] ^ sn[0]

    # 5. Connect Qubits and Stabilizers with Hadamard Edges
    # We need to track used legs on both sides.
    qubit_leg_counters = [1] * DATA_QUBITS
    stab_leg_counters = [1] * len(STABILIZERS)

    for i, stab in enumerate(STABILIZERS):
        sn = stabilizer_nodes[i]
        for q_idx in stab:
            cn = qubit_nodes[q_idx]

            # Insert Hadamard Node on the edge
            h_node = tn.Node(hadamard, name=f"H_{i}_{q_idx}")
            nodes.append(h_node)

            # Connect: Stabilizer -> Hadamard -> Qubit
            # sn leg -> H[0]
            # H[1] -> cn leg
            sn[stab_leg_counters[i]] ^ h_node[0]
            h_node[1] ^ cn[qubit_leg_counters[q_idx]]

            stab_leg_counters[i] += 1
            qubit_leg_counters[q_idx] += 1

    # Collect all nodes
    all_nodes = nodes + qubit_nodes + stabilizer_nodes

    # Contract
    result_node = tc.contractor(all_nodes)

    # The result is unnormalized likelihood (due to 1/2 factors etc).
    # But since we maximize likelihood, constant factors don't change the argmax p.
    # To get exact probability, we would need to divide by 2^(num_stabilizers).
    return jnp.abs(result_node.tensor)  # Return magnitude (should be real positive)


# JIT and Grad
def loss_fn(p_logit, observed_syndrome):
    p = jax.nn.sigmoid(p_logit)
    likelihood = surface_code_likelihood(p, observed_syndrome)
    # Add a small epsilon to avoid log(0)
    return -jnp.log(likelihood + 1e-10)


loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))


# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------


def main():
    print("--- Differentiable Surface Code Noise Estimation (WHT Method) ---")

    # Ground Truth
    p_true = 0.05
    print(f"True Error Probability: {p_true}")

    # Generate Synthetic Data
    n_samples = 20
    print(f"Generating {n_samples} synthetic syndrome samples...")

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    random_probs = jax.random.uniform(subkey, (n_samples, DATA_QUBITS))
    errors = (random_probs < p_true).astype(jnp.int32)

    syndromes = []
    for i in range(n_samples):
        syn = get_syndrome(errors[i])
        syndromes.append(syn)
    syndromes = jnp.array(syndromes)

    # Optimization
    p_init = 0.1
    logit_init = inverse_sigmoid(p_init)

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(logit_init)
    params = logit_init

    loss_history = []
    p_history = []

    print("\nStarting Optimization (Maximum Likelihood)...")
    pbar = tqdm(range(100))
    for _ in pbar:
        # Batch Optimization
        def step_batch(current_params, batch_syndromes):
            def single_sample_loss(s):
                return loss_fn(current_params, s)

            losses = jax.vmap(single_sample_loss)(batch_syndromes)
            return jnp.mean(losses)

        step_batch_grad = jax.value_and_grad(step_batch)
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

    plt.savefig("outputs/convergence.png")
    print("Convergence plot saved to 'outputs/convergence.png'")


def inverse_sigmoid(y):
    return jnp.log(y / (1 - y))


if __name__ == "__main__":
    main()
