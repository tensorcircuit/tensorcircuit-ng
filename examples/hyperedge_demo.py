"""
Demonstration of hyperedge support using cotengra in TensorCircuit.
"""

import time
import numpy as np
import tensornetwork as tn
import tensorcircuit as tc


def hyperedge_demo():
    print("Demonstrating hyperedge contraction with cotengra...")

    # 1. Single Hyperedge Example
    # Three tensors A, B, C connected by a single hyperedge (CopyNode)
    # Result should be sum_i A_i * B_i * C_i

    dim = 2
    a = tn.Node(np.array([1.0, 2.0]), name="A")
    b = tn.Node(np.array([1.0, 2.0]), name="B")
    c = tn.Node(np.array([1.0, 2.0]), name="C")
    cn = tn.CopyNode(3, dim, name="CN")

    a[0] ^ cn[0]
    b[0] ^ cn[1]
    c[0] ^ cn[2]

    nodes = [a, b, c, cn]

    # Set contractor to cotengra
    tc.set_contractor("cotengra")

    res = tc.contractor(nodes)
    print("Single Hyperedge Result:", res.tensor)
    expected = 1 * 1 * 1 + 2 * 2 * 2
    print(f"Expected: {expected}")
    assert np.allclose(res.tensor, expected)

    # 2. Large Scale Hyperedge Example
    # Demonstrate memory and time efficiency with a large number of legs
    print("\nDemonstrating large scale hyperedge (20 legs)...")
    num_legs = 20
    dim = 2

    # Create 20 random tensors connected to a single CopyNode
    input_tensors = [
        tn.Node(np.random.rand(dim), name=f"T{i}") for i in range(num_legs)
    ]
    cn_large = tn.CopyNode(num_legs, dim, name="CN_Large")

    for i, t in enumerate(input_tensors):
        t[0] ^ cn_large[i]

    large_nodes = input_tensors + [cn_large]

    start_time = time.time()
    res_large = tc.contractor(large_nodes)
    end_time = time.time()

    print(f"Contracted {num_legs} legs in {end_time - start_time:.4f} seconds.")
    print("Large Hyperedge Result shape:", res_large.tensor.shape)

    # Verification: Explicitly calculate the sum
    # result = sum_k (prod_i T_i[k])

    # Transpose input tensors to shape (num_legs, dim)
    tensor_matrix = np.stack([t.tensor for t in input_tensors])
    # Product along the tensor axis (0) for each dimension index
    prod_along_legs = np.prod(tensor_matrix, axis=0)
    expected_sum = np.sum(prod_along_legs)

    print(f"Computed: {res_large.tensor}")
    print(f"Expected: {expected_sum}")
    assert np.allclose(res_large.tensor, expected_sum)


if __name__ == "__main__":
    hyperedge_demo()
