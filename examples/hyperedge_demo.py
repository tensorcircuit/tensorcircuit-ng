"""
Demonstration of hyperedge support using cotengra in TensorCircuit.
"""

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
    try:
        tc.set_contractor("cotengra")
    except ImportError:
        print("cotengra not installed, skipping demo")
        return

    res = tc.contractor(nodes)
    print("Single Hyperedge Result:", res.tensor)
    expected = 1*1*1 + 2*2*2
    print(f"Expected: {expected}")
    assert np.allclose(res.tensor, expected)

    # 2. Chained Hyperedge Example
    # A-CN1-B, CN1-CN2, C-CN2-D
    # Effectively A, B, C, D share an index

    a = tn.Node(np.array([1.0, 2.0]), name="A")
    b = tn.Node(np.array([1.0, 2.0]), name="B")
    c = tn.Node(np.array([1.0, 2.0]), name="C")
    d = tn.Node(np.array([1.0, 2.0]), name="D")

    cn1 = tn.CopyNode(3, dim, name="CN1")
    cn2 = tn.CopyNode(3, dim, name="CN2")

    a[0] ^ cn1[0]
    b[0] ^ cn1[1]
    cn1[2] ^ cn2[0] # Link between hyperedges
    c[0] ^ cn2[1]
    d[0] ^ cn2[2]

    nodes = [a, b, c, d, cn1, cn2]
    res = tc.contractor(nodes)
    print("Chained Hyperedge Result:", res.tensor)
    expected = 1*1*1*1 + 2*2*2*2
    print(f"Expected: {expected}")
    assert np.allclose(res.tensor, expected)

if __name__ == "__main__":
    hyperedge_demo()
