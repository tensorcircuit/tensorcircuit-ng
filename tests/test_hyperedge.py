import numpy as np
import tensornetwork as tn
import tensorcircuit as tc
import pytest


@pytest.fixture
def contractor_setup(request):
    """
    Fixture to set up the contractor and clean up afterwards.
    Default to cotengra, but can be parametrized.
    """
    contractor_name = getattr(request, "param", "cotengra")
    if contractor_name == "cotengra":
        try:
            import cotengra
        except ImportError:
            pytest.skip("cotengra not installed")

    tc.set_contractor(contractor_name)
    yield contractor_name
    # Reset to default
    tc.set_contractor("greedy")


@pytest.mark.parametrize("contractor_setup", ["cotengra", "greedy"], indirect=True)
def test_single_hyperedge(contractor_setup):
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

    if contractor_setup == "greedy":
        # Standard greedy contractor doesn't support CopyNodes natively in path finding?
        # Actually, TN's default might handle them if they are just nodes.
        # But if we rely on hyperedge optimization, greedy might fail or produce suboptimal paths.
        # Let's see if it runs. If not, we might expect failure or just skip.
        # TensorNetwork's CopyNode is just a node with a delta tensor.
        # So it should work mathematically, just not optimized as hyperedge.
        pass

    res = tc.contractor(nodes)
    assert np.allclose(res.tensor, 9.0)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_chained_hyperedge(contractor_setup):
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
    cn1[2] ^ cn2[0]  # Link
    c[0] ^ cn2[1]
    d[0] ^ cn2[2]

    nodes = [a, b, c, d, cn1, cn2]
    res = tc.contractor(nodes)
    # sum i A_i B_i C_i D_i = 1+16 = 17
    assert np.allclose(res.tensor, 17.0)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_dangling_hyperedge(contractor_setup):
    # A(i), B(i), Output(i)
    dim = 2
    a = tn.Node(np.array([1.0, 2.0]), name="A")
    b = tn.Node(np.array([1.0, 2.0]), name="B")
    cn = tn.CopyNode(3, dim, name="CN")

    a[0] ^ cn[0]
    b[0] ^ cn[1]
    # cn[2] is dangling

    nodes = [a, b, cn]
    res = tc.contractor(nodes)  # Should return a tensor of shape (2,)

    # Expected: C_i = A_i * B_i => [1, 4]
    assert np.allclose(res.tensor, np.array([1.0, 4.0]))


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_tensorcircuit_circuit_hyperedge_support(contractor_setup):
    # While TC circuit doesn't typically create CopyNodes directly in gates,
    # ensuring the contractor works with general graphs is key.
    c = tc.Circuit(2)
    c.H(0)
    c.CNOT(0, 1)

    state = c.state()
    # Bell state |00> + |11>
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(np.abs(state), np.abs(expected))
