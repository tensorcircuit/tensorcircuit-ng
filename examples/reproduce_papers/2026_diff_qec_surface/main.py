"""
Reproduction of "Differentiable Maximum Likelihood Noise Estimation for Quantum Error Correction"
Link: https://arxiv.org/abs/2602.19722

Description:
This script reproduces the core idea of differentiable noise estimation using tensor networks.
It simulates a rotated surface code (d=3) with bit-flip errors.
The goal is to estimate the error probability 'p' given a set of observed syndromes.
We construct the tensor network for the likelihood P(syndrome | p) using CopyNodes to handle hyperedges,
and use JAX for automatic differentiation to maximize the likelihood (minimize NLL).
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
# Data qubits (X): 0, 1, 2, 3, 4, 5, 6, 7, 8 (3x3 grid)
# Stabilizers (Z-checks and X-checks):
# Z-checks (detect X errors): measured in Z basis
# X-checks (detect Z errors): measured in X basis
# For simplicity, we focus on X-errors only (detected by Z-stabilizers).
# This is equivalent to the repetition code limit or CSS code with only one type of error for demonstration.

# Let's use a standard layout for d=3 rotated surface code.
# Qubits are on vertices. Plaquettes are stabilizers.
#
# d=3 means code distance 3.
# 9 data qubits.
#
# Layout:
# .   Z0  .   Z1
# q0--q1--q2
# |   |   |
# q3--q4--q5
# |   |   |
# q6--q7--q8
# .   Z2  .   Z3
#
# Wait, standard rotated surface code:
#      q0 -- Z0 -- q1
#      |     |     |
#      X0 -- q2 -- X1
#      |     |     |
#      q3 -- Z1 -- q4
#
# Let's simplify: define stabilizers by the qubits they act on.
# For d=3 rotated surface code:
# Data qubits: 0..8 (9 qubits)
# Stabilizers:
# Z0: 0, 1, 3, 4
# Z1: 1, 2, 4, 5
# Z2: 3, 4, 6, 7
# Z3: 4, 5, 7, 8
# (This is a bulk 2x2 plaquette layout)
# X stabilizers would be on the vertices of the dual lattice.
# To keep it simple and runnable on CPU, let's consider a Repetition Code or small Surface Code part.
# The paper mentions "d=5 surface codes".
# Let's implement a d=3 Repetition Code first to ensure differentiability, then scale if possible.
# Actually, the prompt asks for "surface code". Let's try a small 2x2 plaquette patch (d=3).

# Code definition: Rotated Surface Code d=3
# Data Qubits: 9
# Stabilizers (Z-type, detecting X errors):
# s0: Z(0) Z(1) Z(3) Z(4)
# s1: Z(1) Z(2) Z(4) Z(5)
# s2: Z(3) Z(4) Z(6) Z(7)
# s3: Z(4) Z(5) Z(7) Z(8)
#
# We assume Independent Identical Distribution (I.I.D) X-error on each data qubit with probability p.
# We want to compute P(syndrome | p).
# Syndrome s = (m0, m1, m2, m3) where mi is the measurement outcome of stabilizer i (+1 or -1, mapped to 0 or 1).
# In the absence of errors, all Z stabilizers are +1.
# An X error on qubit j flips the adjacent Z stabilizers.

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
# 2. Tensor Network Construction (Differentiable)
# -----------------------------------------------------------------------------


def surface_code_likelihood(p, observed_syndrome):
    """
    Compute the likelihood P(syndrome | p) using a tensor network.
    p: Error probability (scalar).
    observed_syndrome: array of shape (4,), values 0 or 1.
    """
    # Nodes:
    # 1. Error tensors E_j for each qubit j.
    #    E_j is a rank-1 tensor (vector) of dimension 2.
    #    Index 0: No error (prob 1-p)
    #    Index 1: Error (prob p)
    #    Actually, we need to treat the error as a variable summed over.
    #    E_j = [sqrt(1-p), sqrt(p)] if we work with amplitudes, or [(1-p), p] for probabilities.
    #    Let's use probabilities directly for this classical error model (incoherent).
    #    E_j = [1-p, p]

    # 2. CopyNodes (Delta tensors) for each qubit.
    #    Each qubit participates in 2 or 4 stabilizers.
    #    We need to copy the "error status" of the qubit to all stabilizers it participates in.
    #    Degree = (number of stabilizers acting on this qubit) + 1 (from the error source).

    # 3. Stabilizer tensors S_i.
    #    S_i checks the parity of connected errors matches the observed syndrome bit m_i.
    #    S_i_{e_a, e_b, e_c, e_d} = 1 if (e_a + e_b + e_c + e_d) % 2 == m_i else 0.

    nodes = []

    # Error Probability Tensors (Source)
    # Shape (2,): [Prob(No Error), Prob(Error)]
    # Note: For differentiability, we use p directly.
    # We want log-likelihood, but let's compute likelihood first.
    # P(syndrome) = sum_{errors} P(errors) * delta(syndrome(errors) == observed)
    # P(errors) = prod_j (p if e_j=1 else 1-p)
    # This factorizes.

    prob_vec = jnp.stack([1 - p, p])

    # 1. Create qubit CopyNodes and attach Error Factors
    qubit_nodes = []
    for q in range(DATA_QUBITS):
        # Find which stabilizers involve this qubit
        involved_stabs = [i for i, stab in enumerate(STABILIZERS) if q in stab]
        degree = len(involved_stabs) + 1  # +1 for the error source

        # Create CopyNode
        # In this classical probability model, the "bond dimension" is 2 (error or no error).
        cn = tn.CopyNode(degree, 2, name=f"Q{q}")
        qubit_nodes.append(cn)

        # Create Error Factor Node
        en = tn.Node(prob_vec, name=f"E{q}")
        nodes.append(en)

        # Connect Error Factor to the first leg of CopyNode
        en[0] ^ cn[0]

    # 2. Create Stabilizer Tensors
    # These are parity checks.
    # Tensor S has shape (2, 2, 2, 2) for a weight-4 stabilizer.
    # S[a,b,c,d] = 1 if (a+b+c+d)%2 == m else 0
    stabilizer_nodes = []
    for i, stab in enumerate(STABILIZERS):
        k = len(stab)
        shape = [2] * k
        # Create the parity tensor
        # We can construct it by iterating all $2^k$ configurations
        # or smartly. For small k=4, iteration is fine.
        indices = jnp.indices(shape)
        # indices is a tuple of k arrays, each shape (2,2,2,2)
        # sum them up
        idx_sum = sum(indices)
        target_parity = observed_syndrome[i]

        # Parity check: (sum % 2) == target
        tensor_val = ((idx_sum % 2) == target_parity).astype(
            jnp.float64
        )  # Use float for differentiability flow?
        # Actually, the tensor values are constant 0 or 1.
        # Differentiability comes from the inputs (prob_vec).

        sn = tn.Node(tensor_val, name=f"S{i}")
        stabilizer_nodes.append(sn)
        nodes.append(sn)

        # Connect Stabilizer to Qubits
        # Handled in the next loop

    # Re-loop to connect, using a counter
    qubit_leg_counters = [1] * DATA_QUBITS  # Start at 1 because 0 is used

    for i, stab in enumerate(STABILIZERS):
        sn = stabilizer_nodes[i]
        for leg_idx, q_idx in enumerate(stab):
            cn = qubit_nodes[q_idx]

            # Connect stabilizer leg `leg_idx` to qubit copy node leg `counter`
            sn[leg_idx] ^ cn[qubit_leg_counters[q_idx]]

            qubit_leg_counters[q_idx] += 1

    # Collect all nodes including CopyNodes
    all_nodes = nodes + qubit_nodes

    # Contract
    # The result should be a scalar (likelihood)
    result_node = tc.contractor(all_nodes)

    return result_node.tensor


# JIT and Grad
# We want to minimize NLL: -log(Likelihood)
def loss_fn(p_logit, observed_syndrome):
    # Parameterize p with sigmoid to keep it in [0, 1]
    p = jax.nn.sigmoid(p_logit)
    likelihood = surface_code_likelihood(p, observed_syndrome)
    # Add a small epsilon to avoid log(0)
    return -jnp.log(likelihood + 1e-10)


loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))


# -----------------------------------------------------------------------------
# 3. Main Execution: Data Generation and Optimization
# -----------------------------------------------------------------------------


def main():
    print("--- Differentiable Surface Code Noise Estimation ---")

    # Ground Truth
    p_true = 0.05
    print(f"True Error Probability: {p_true}")

    # Generate Synthetic Data
    # We generate N samples of error configurations, calculate their syndromes.
    n_samples = 20
    print(f"Generating {n_samples} synthetic syndrome samples...")

    key = jax.random.PRNGKey(42)
    # Generate random errors: shape (N, 9)
    # 1 with prob p_true, 0 with prob 1-p_true
    key, subkey = jax.random.split(key)
    random_probs = jax.random.uniform(subkey, (n_samples, DATA_QUBITS))
    errors = (random_probs < p_true).astype(jnp.int32)

    syndromes = []
    for i in range(n_samples):
        syn = get_syndrome(errors[i])
        syndromes.append(syn)
    syndromes = jnp.array(syndromes)

    # Optimization
    # Initial guess: p=0.1 (logit approx -2.2)
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
        # Compute total loss over the batch
        # We can sum the loss for each sample

        # To make it efficient, we could vmap, but let's loop first to be safe with TN contraction
        # Or better: accumulate gradients.

        # Actually, let's define a batch_loss function

        # Note: 'loss_and_grad' is for a single sample.
        # Let's average gradients over the batch.

        def step_batch(current_params, batch_syndromes):
            def single_sample_loss(s):
                return loss_fn(current_params, s)

            # Use vmap here!
            losses = jax.vmap(single_sample_loss)(batch_syndromes)
            return jnp.mean(losses)

        # JIT the batch step
        step_batch_grad = jax.value_and_grad(step_batch)

        # Compute
        loss_val, grads = step_batch_grad(params, syndromes)

        # Update
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
