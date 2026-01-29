"""
MIPT probability histogram demo.
"""

from functools import partial
import time
import jax
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")


def delete2(pick, plist):
    # pick = 0, 1 : return plist[pick]/(plist[0]+plist[1])
    # pick = 2: return 1
    indicator = (K.sign(1.5 - pick) + 1) / 2  # 0,1 : 1, 2: 0
    p = 0
    p += 1 - indicator
    p += indicator / (plist[0] + plist[1]) * (plist[0] * (1 - pick) + plist[1] * pick)
    return p


@partial(K.jit, static_argnums=(2, 3))
def circuit_output(random_matrix, status, n, d, p):
    random_matrix = K.reshape(random_matrix, [d, n, 4, 4])
    status = K.reshape(status, [d, n])

    def one_step(carry, x):
        inputs, prob_accum = carry
        rm_j, st_j = x

        # Apply Unitaries
        if inputs is None:
            c = tc.Circuit(n)
        else:
            c = tc.Circuit(n, inputs=inputs)

        for i in range(0, n, 2):
            c.unitary(i, (i + 1) % n, unitary=rm_j[i])
        for i in range(1, n, 2):
            c.unitary(i, (i + 1) % n, unitary=rm_j[i])

        inputs = c.state()
        c = tc.Circuit(n, inputs=inputs)

        # Apply Measurements
        current_prob_log = 0.0
        for i in range(n):
            pick, plist = c.general_kraus(
                [
                    K.sqrt(p) * K.convert_to_tensor(np.array([[1.0, 0], [0, 0]])),
                    K.sqrt(p) * K.convert_to_tensor(np.array([[0, 0], [0, 1.0]])),
                    K.sqrt(1 - p) * K.eye(2),
                ],
                i,
                status=st_j[i],
                with_prob=True,
            )
            current_prob_log += K.log(delete2(pick, plist) + 1e-14)
            inputs = c.state()
            c = tc.Circuit(n, inputs=inputs)

        inputs = c.state()
        inputs /= K.norm(inputs)

        return (inputs, prob_accum + current_prob_log), None

    # Initial carry: (inputs=None, prob_accum=0.0)
    # However, scan requires concrete initial state.
    # tc.Circuit(n).state() gives the |0...0> state.
    init_inputs = tc.Circuit(n).state()
    init_carry = (init_inputs, 0.0)

    # Scan over d steps
    # xs = (random_matrix, status) which are already shaped [d, ...]
    (final_inputs, final_prob_accum), _ = (  # pylint: disable=unused-variable
        jax.lax.scan(one_step, init_carry, (random_matrix, status))
    )

    return final_prob_accum


@partial(K.jit, static_argnums=(2, 3))
def batch_simulate(random_matrices, statuses, n, d, p):
    # vmap over batch dimension
    return K.vmap(partial(circuit_output, n=n, d=d, p=p), vectorized_argnums=(0, 1))(
        random_matrices, statuses
    )


def main():
    n = 12
    d = 24
    batch_size = 20000
    p_list = [0.02, 0.6]

    plt.figure(figsize=(10, 6))

    # PT distribution reference
    # For Porter-Thomas, P(u) = e^-u where u = p / <p> * N (dim of Hilbert space)
    # But here we look at bitstring probabilities P.
    # Standard PT: x = P * D. P(x) = exp(-x).
    x_pt = np.linspace(0, 10, 100)
    plt.plot(x_pt, np.exp(-x_pt), "k--", label="Porter-Thomas ($e^{-x}$)", linewidth=2)

    for p in p_list:
        print(f"Simulating for p={p}...")

        # Initialize random unitary matrices
        rm = [stats.unitary_group.rvs(4) for _ in range(d * n * batch_size)]
        rm = [r / np.linalg.det(r) for r in rm]  # SU(4) check?
        rm = np.stack(rm).reshape([batch_size, d * n, 4, 4])
        rm = K.convert_to_tensor(rm)

        # Initialize random status for sampling
        st = np.random.uniform(size=[batch_size, d * n])
        st = K.convert_to_tensor(st)

        # Run Batch
        start_run = time.time()
        log_probs = batch_simulate(rm, st, n, d, p)
        log_probs = np.array(log_probs)
        end_run = time.time()
        print(f"Batch Execution time for p={p}: {end_run - start_run:.4f} s")

        # Process probs
        # P = exp(log_prob)
        probs = np.exp(log_probs)

        # Normalize to mean 1 for PT comparison
        # x = p / <p>
        mean_p = np.mean(probs)
        x = probs / mean_p

        # Histogram
        plt.hist(
            x,
            bins=50,
            density=True,
            alpha=0.6,
            label=f"p={p}",
            histtype="step",
            linewidth=2,
            range=(0, 10),
        )

    plt.yscale("log")
    plt.xlabel(r"$P / \langle P \rangle$")
    plt.ylabel("Probability Density")
    plt.ylim(1e-4, 2)
    plt.title("Bitstring Probability Distribution vs PT (n=12, d=10)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    save_path = "mipt_prob_histogram.pdf"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
