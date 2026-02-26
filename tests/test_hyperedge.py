import numpy as np
import tensornetwork as tn
import tensorcircuit as tc
import pytest

# Ensure cotengra is available, otherwise skip tests
try:
    import cotengra
    has_cotengra = True
except ImportError:
    has_cotengra = False

@pytest.mark.skipif(not has_cotengra, reason="cotengra not installed")
def test_single_hyperedge():
    # A(i), B(i), C(i)
    dim = 2
    a = tn.Node(np.array([1.0, 2.0]), name="A")
    b = tn.Node(np.array([1.0, 2.0]), name="B")
    c = tn.Node(np.array([1.0, 2.0]), name="C")
    cn = tn.CopyNode(3, dim, name="CN")

    a[0] ^ cn[0]
    b[0] ^ cn[1]
    c[0] ^ cn[2]

    nodes = [a, b, c, cn]
    tc.set_contractor("cotengra")
    res = tc.contractor(nodes)
    assert np.allclose(res.tensor, 9.0)

@pytest.mark.skipif(not has_cotengra, reason="cotengra not installed")
def test_chained_hyperedge():
    # A(i), B(i), C(i), D(i)
    # Connected via two CopyNodes: A-CN1-B, CN1-CN2, C-CN2-D
    dim = 2
    a = tn.Node(np.array([1.0, 2.0]), name="A")
    b = tn.Node(np.array([1.0, 2.0]), name="B")
    c = tn.Node(np.array([1.0, 2.0]), name="C")
    d = tn.Node(np.array([1.0, 2.0]), name="D")

    cn1 = tn.CopyNode(3, dim, name="CN1")
    cn2 = tn.CopyNode(3, dim, name="CN2")

    a[0] ^ cn1[0]
    b[0] ^ cn1[1]
    cn1[2] ^ cn2[0] # Link
    c[0] ^ cn2[1]
    d[0] ^ cn2[2]

    nodes = [a, b, c, d, cn1, cn2]
    tc.set_contractor("cotengra")
    res = tc.contractor(nodes)
    # sum i A_i B_i C_i D_i = 1+16 = 17
    assert np.allclose(res.tensor, 17.0)

@pytest.mark.skipif(not has_cotengra, reason="cotengra not installed")
def test_dangling_hyperedge():
    # A(i), B(i), Output(i)
    dim = 2
    a = tn.Node(np.array([1.0, 2.0]), name="A")
    b = tn.Node(np.array([1.0, 2.0]), name="B")
    cn = tn.CopyNode(3, dim, name="CN")

    a[0] ^ cn[0]
    b[0] ^ cn[1]
    # cn[2] is dangling

    nodes = [a, b, cn]
    tc.set_contractor("cotengra")
    res = tc.contractor(nodes) # Should return a tensor of shape (2,)

    # Expected: C_i = A_i * B_i => [1, 4]
    assert np.allclose(res.tensor, np.array([1.0, 4.0]))

@pytest.mark.skipif(not has_cotengra, reason="cotengra not installed")
def test_tensorcircuit_circuit_hyperedge_support():
    # While TC circuit doesn't typically create CopyNodes directly in gates,
    # ensuring the contractor works with general graphs is key.
    # This test just ensures normal circuit simulation still works with cotengra
    # (which implies the new logic handles regular nodes correctly too).
    c = tc.Circuit(2)
    c.H(0)
    c.CNOT(0, 1)

    tc.set_contractor("cotengra")
    state = c.state()
    # Bell state |00> + |11>
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    # The phase might vary? No, standard gates are deterministic.
    # But H gate normalization 1/sqrt(2).
    # |0> -> (|0>+|1>)/rt2 -> |00> + |11> / rt2.
    # Check absolute values
    assert np.allclose(np.abs(state), np.abs(expected))
