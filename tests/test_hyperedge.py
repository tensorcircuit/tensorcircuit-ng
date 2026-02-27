import pytest
import numpy as np
import tensornetwork as tn
import tensorcircuit as tc


@pytest.fixture
def contractor_setup(request):
    """
    Fixture to set up the contractor and clean up afterwards.
    Default to cotengra, but can be parametrized.
    Support passing (method, kwargs) as param.
    """
    param = getattr(request, "param", "cotengra")
    if isinstance(param, str):
        method = param
        kwargs = {}
    else:
        method, kwargs = param

    opt = None
    if method == "cotengra":
        try:
            import cotengra

            opt = cotengra.ReusableHyperOptimizer(
                methods=["greedy", "kahypar"],
                parallel=True,
                minimize="combo",
                max_time=30,
                max_repeats=64,
                progbar=True,
            )
            tc.set_contractor("custom", optimizer=opt, **kwargs)
        except ImportError:
            pytest.skip("cotengra not installed")
    else:
        tc.set_contractor(method, **kwargs)

    yield method

    if opt is not None and hasattr(opt, "close"):
        opt.close()
    # Reset to default
    tc.set_contractor("greedy")


@pytest.fixture(params=["numpy", "jax", "tensorflow", "pytorch"])
def backend_setup(request):
    backend_name = request.param
    try:
        tc.set_backend(backend_name)
    except ImportError:
        pytest.skip(f"{backend_name} not installed")
    yield backend_name
    tc.set_backend("numpy")


@pytest.mark.parametrize("contractor_setup", ["cotengra", "greedy"], indirect=True)
def test_single_hyperedge(contractor_setup, backend_setup):
    # A(i), B(i), C(i)
    dim = 2
    a = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="A")
    b = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="B")
    c = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="C")
    cn = tn.CopyNode(3, dim, name="CN")

    a[0] ^ cn[0]
    b[0] ^ cn[1]
    c[0] ^ cn[2]

    nodes = [a, b, c, cn]

    res = tc.contractor(nodes)
    np.testing.assert_allclose(tc.backend.numpy(res.tensor), 9.0, atol=1e-5)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_chained_hyperedge(contractor_setup, backend_setup):
    # A(i), B(i), C(i), D(i)
    # Connected via two CopyNodes: A-CN1-B, CN1-CN2, C-CN2-D
    dim = 2
    a = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="A")
    b = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="B")
    c = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="C")
    d = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="D")

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
    np.testing.assert_allclose(tc.backend.numpy(res.tensor), 17.0, atol=1e-5)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_dangling_hyperedge(contractor_setup, backend_setup):
    # A(i), B(i), Output(i)
    dim = 2
    a = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="A")
    b = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="B")
    cn = tn.CopyNode(3, dim, name="CN")

    a[0] ^ cn[0]
    b[0] ^ cn[1]
    # cn[2] is dangling

    nodes = [a, b, cn]
    res = tc.contractor(nodes)  # Should return a tensor of shape (2,)

    # Expected: C_i = A_i * B_i => [1, 4]
    np.testing.assert_allclose(
        tc.backend.numpy(res.tensor), np.array([1.0, 4.0]), atol=1e-5
    )


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_tensorcircuit_circuit_hyperedge_support(contractor_setup, backend_setup):
    c = tc.Circuit(2)
    c.H(0)
    c.CNOT(0, 1)

    state = c.state()
    # Bell state |00> + |11>
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(
        np.abs(tc.backend.numpy(state)), np.abs(expected), atol=1e-5
    )


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_hyperedge_output_reordering(contractor_setup, backend_setup):
    dim = 2
    a = tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="A")
    b = tn.Node(tc.gates.num_to_tensor(np.array([3.0, 4.0])), name="B")

    cn1 = tn.CopyNode(2, dim, name="CN1")
    cn2 = tn.CopyNode(2, dim, name="CN2")

    a[0] ^ cn1[0]
    b[0] ^ cn2[0]

    nodes = [a, b, cn1, cn2]
    output_edge_order = [cn2[1], cn1[1]]

    res = tc.contractor(nodes, output_edge_order=output_edge_order)

    expected = np.outer(np.array([3.0, 4.0]), np.array([1.0, 2.0]))

    np.testing.assert_allclose(tc.backend.numpy(res.tensor), expected, atol=1e-5)
    assert res.tensor.shape == (2, 2)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_large_hyperedge_chain(contractor_setup, backend_setup):
    # A chain of 12 nodes connected by CopyNodes
    n_nodes = 12
    dim = 2

    # n0(a) -- cn0 -- n1(a, b) -- cn1 -- n2(b, c) -- ... -- n11(k)
    nodes = []
    # n0
    nodes.append(tn.Node(tc.gates.num_to_tensor(np.array([1.0, 2.0])), name="n0"))
    for i in range(1, n_nodes - 1):
        nodes.append(tn.Node(tc.gates.num_to_tensor(np.eye(dim)), name=f"n{i}"))
    # n11
    nodes.append(
        tn.Node(tc.gates.num_to_tensor(np.array([1.0, 3.0])), name=f"n{n_nodes-1}")
    )

    copy_nodes = []

    # cn0 connects n0[0] and n1[0]
    cn0 = tn.CopyNode(2, dim, name="cn0")
    nodes[0][0] ^ cn0[0]
    nodes[1][0] ^ cn0[1]
    copy_nodes.append(cn0)

    for i in range(1, n_nodes - 1):
        # n_i[1] connects to n_{i+1}[0]
        cn = tn.CopyNode(2, dim, name=f"cn{i}")
        nodes[i][1] ^ cn[0]
        nodes[i + 1][0] ^ cn[1]
        copy_nodes.append(cn)

    res = tc.contractor(nodes + copy_nodes)

    expected = 1 * 1 + 2 * 3  # = 7
    np.testing.assert_allclose(tc.backend.numpy(res.tensor), expected, atol=1e-5)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_large_star_hyperedge(contractor_setup, backend_setup):
    n_nodes = 35
    dim = 2
    nodes = []
    for i in range(n_nodes):
        nodes.append(
            tn.Node(tc.gates.num_to_tensor(np.array([1.0, 1.1])), name=f"n{i}")
        )

    cn = tn.CopyNode(n_nodes, dim, name="CN_star")
    for i in range(n_nodes):
        nodes[i][0] ^ cn[i]

    res = tc.contractor(nodes + [cn])

    expected = 1.0**35 + 1.1**35
    np.testing.assert_allclose(tc.backend.numpy(res.tensor), expected, atol=1e-4)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_hyperedge_jit(jaxb, contractor_setup):
    import jax

    @jax.jit
    def f(v1, v2):
        a = tn.Node(v1)
        b = tn.Node(v2)
        cn = tn.CopyNode(3, 2)
        a[0] ^ cn[0]
        b[0] ^ cn[1]
        nodes = [a, b, cn]
        res = tc.contractor(nodes)
        return res.tensor[0]

    v1 = jax.numpy.array([1.0, 2.0])
    v2 = jax.numpy.array([3.0, 4.0])
    res = f(v1, v2)
    np.testing.assert_allclose(res, 3.0, atol=1e-5)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_hyperedge_ad(jaxb, contractor_setup):
    import jax

    def f(v1, v2):
        a = tn.Node(v1)
        b = tn.Node(v2)
        cn = tn.CopyNode(3, 2)
        a[0] ^ cn[0]
        b[0] ^ cn[1]
        nodes = [a, b, cn]
        res = tc.contractor(nodes)
        return tc.backend.real(res.tensor[0])

    v1 = jax.numpy.array([1.0, 2.0])
    v2 = jax.numpy.array([3.0, 4.0])

    gv1 = jax.grad(f, argnums=0)(v1, v2)
    # df/dv1 = [b[0], 0] = [3.0, 0]
    np.testing.assert_allclose(gv1, jax.numpy.array([3.0, 0.0]), atol=1e-5)


@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_cotengra_path_reuse(caplog, contractor_setup):
    import logging

    if tc.backend.name != "numpy":
        pytest.skip("Path reuse test simplified for numpy")

    try:
        import cotengra
    except ImportError:
        pytest.skip("cotengra not installed")

    def run_contraction():
        # Use a star topology with 7 nodes (6 outer + 1 central CopyNode)
        # to ensure len(nodes) > 5 and algebraic path is triggered.
        nodes = []
        for i in range(6):
            nodes.append(tn.Node(np.array([1.0, 1.1])))
        cn = tn.CopyNode(6, 2)
        for i in range(6):
            nodes[i][0] ^ cn[i]
        return tc.contractor(nodes + [cn])

    # First run should trigger search
    with caplog.at_level(logging.INFO):
        run_contraction()
    assert "the contraction path is given as" in caplog.text

    # We need to access the optimizer to check its state.
    opt = tc.contractor.keywords["optimizer"]
    assert isinstance(opt, cotengra.ReusableHyperOptimizer)

    initial_searches = len(opt.deltas) if hasattr(opt, "deltas") else 0

    # Second run should NOT trigger a new full search (cotengra should hit its internal cache)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        run_contraction()
    assert "the contraction path is given as" in caplog.text

    final_searches = len(opt.deltas) if hasattr(opt, "deltas") else 0
    assert final_searches == initial_searches


@pytest.mark.parametrize(
    "contractor_setup", [("cotengra", {"use_primitives": True})], indirect=True
)
def test_hyperedge_partial_contraction(contractor_setup):
    # Test contracting a subset of nodes in a larger network
    a = tn.Node(np.array([1.0, 2.0]))  # [i]
    b = tn.Node(np.eye(2) * np.array([3, 4]))  # [i, j]
    c = tn.Node(np.eye(2) * np.array([5, 6]))  # [j, k]
    d = tn.Node(np.array([7.0, 8.0]))  # [k]

    a[0] ^ b[0]
    e_bc = b[1] ^ c[0]
    e_cd = c[1] ^ d[0]

    # 1. Contract [A, B]
    res_ab = tc.contractor([a, b])
    # res_ab should be rank 1, edges[0] should be e_bc
    assert len(res_ab.edges) == 1
    assert res_ab.edges[0] == e_bc
    assert e_bc.node1 == res_ab or e_bc.node2 == res_ab

    # 2. Contract [res_ab, c]
    res_abc = tc.contractor([res_ab, c])
    assert len(res_abc.edges) == 1
    assert res_abc.edges[0] == e_cd

    # 3. Final
    res = tc.contractor([res_abc, d])
    # Expected: sum_i,j,k A_i B_ij C_jk D_k = 1*3*5*7 + 2*4*6*8 = 105 + 384 = 489
    np.testing.assert_allclose(res.tensor, 489.0)
