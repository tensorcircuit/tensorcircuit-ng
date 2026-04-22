"""
Example demonstrating how to use cotengra's strip_exponent = True
to avoid overflow and underflow in large tensor network contractions.
"""

import numpy as np
import tensornetwork as tn
import tensorcircuit as tc

# Set backend to JAX for JIT and potential large scale simulation
tc.set_backend("jax")
tc.set_dtype("complex128")


def run_underflow_demonstration():
    """
    Demonstrate basic scalar underflow handling with strip_exponent.
    """
    # Each node is a 0.1 scalar. 0.1^400 = 10^-400, which underflows float64 (~1e-308)
    nodes = [
        tn.Node(tc.backend.convert_to_tensor(0.1, tc.rdtypestr)) for _ in range(400)
    ]

    print("1. Standard contraction (expecting underflow to 0.0):")
    tc.set_contractor("cotengra")  # default strip_exponent=False
    res_node = tc.cons.contractor(nodes)
    print(f"Result: {res_node.tensor}")

    print("\n2. Contraction with strip_exponent=True (underflow protection):")
    tc.set_contractor("cotengra", strip_exponent=True)
    res_node, exponent = tc.cons.contractor(nodes)
    print(f"Scaled result: {res_node.tensor}")
    print(f"Log10 exponent: {exponent}")
    # Total value is res_node.tensor * 10^exponent


def run_overflow_demonstration():
    """
    Demonstrate circuit expectation overflow handling with strip_exponent.
    """
    n = 100
    c = tc.Circuit(n)
    # Each qubit gets a 2.0 * I gate. Total growth 2^200 (approx 10^60)
    matrix = np.array([[2.0, 0], [0, 2.0]])
    for i in range(n):
        c.any(i, unitary=matrix)

    print(
        "\n3. Large circuit expectation with strip_exponent=True (overflow protection):"
    )
    tc.set_contractor("cotengra", strip_exponent=True)

    # Low-level expectation_before gives us the nodes to contract
    nodes = c.expectation_before([tc.gates.z(), [0]], reuse=False)
    res_node, exponent = tc.cons.contractor(nodes)
    print(f"Scaled expectation: {res_node.tensor}")
    print(f"Log10 exponent: {exponent}")

    # Expected value: 2^200 = 10^(200 * log10(2)) = 10^60.206
    print(f"Expected Log10 exponent: {200 * np.log10(2.0)}")


if __name__ == "__main__":
    run_underflow_demonstration()
    run_overflow_demonstration()
