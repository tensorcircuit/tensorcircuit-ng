import pytest
from pytest_lazyfixture import lazy_fixture as lf
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


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_circuit_diagonal_gate(contractor_setup, backend):
    n = 3
    c1 = tc.Circuit(n)
    c2 = tc.Circuit(n)

    for i in range(n):
        c1.h(i)
        c2.h(i)

    diag = tc.backend.convert_to_tensor(np.array([1, -1, 1j, -1j, 1, 1, -1, -1]))

    # Apply using dense diagonal representation
    diag_matrix = tc.backend.diagflat(diag)
    c1.any(*range(n), unitary=diag_matrix)

    # Apply using hyperedge CopyNode representation
    c2.diagonal(*range(n), diag=diag)

    for i in range(n):
        c1.rx(i, theta=0.2)
        c2.rx(i, theta=0.2)

    np.testing.assert_allclose(c1.state(), c2.state(), atol=1e-5)

    # Test expectation values
    exp1 = c1.expectation([tc.gates.z(), [0]], [tc.gates.y(), [1]])
    exp2 = c2.expectation([tc.gates.z(), [0]], [tc.gates.y(), [1]])
    np.testing.assert_allclose(exp1, exp2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_circuit_diagonal_qir(contractor_setup, backend):
    n = 2
    c = tc.Circuit(n)
    c.h(0)
    diag = tc.backend.convert_to_tensor(np.array([1.0, 1j, -1j, -1.0]))
    c.diagonal(0, 1, diag=diag)
    c.x(1)

    qir = c.to_qir()
    c2 = tc.Circuit.from_qir(qir)
    np.testing.assert_allclose(c.state(), c2.state(), atol=1e-5)

    c_inv = c.inverse()
    c_all = tc.Circuit(n)
    c_all.append_from_qir(c.to_qir())
    c_all.append_from_qir(c_inv.to_qir())

    # state should be |00>
    expected = np.zeros(4)
    expected[0] = 1.0
    np.testing.assert_allclose(np.abs(c_all.state()), expected, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_circuit_rzm_gate(contractor_setup, backend):
    n = 4
    theta = 1.2

    # 1. BaseCircuit implementation via exact TensorNetwork MPS (rzm)
    c_mps = tc.Circuit(n)
    for i in range(n):
        c_mps.h(i)
    c_mps.rzm(*range(n), theta=theta)
    for i in range(n):
        c_mps.rx(i, theta=0.3)

    s_mps = c_mps.state()

    # 2. Dense Matrix baseline comparison
    c_dense = tc.Circuit(n)
    for i in range(n):
        c_dense.h(i)

    # Construct dense R_{ZZ...Z} matrix
    diag = np.ones(2**n, dtype=np.complex128) * np.cos(theta / 2)
    z_str = np.array([(-1) ** bin(i).count("1") for i in range(2**n)])
    diag -= 1j * np.sin(theta / 2) * z_str

    diag_tensor = tc.backend.convert_to_tensor(diag)
    diag_matrix = tc.backend.diagflat(diag_tensor)
    c_dense.any(*range(n), unitary=diag_matrix)

    for i in range(n):
        c_dense.rx(i, theta=0.3)

    s_dense = c_dense.state()

    np.testing.assert_allclose(s_mps, s_dense, atol=1e-5)

    exp_mps = c_mps.expectation([tc.gates.z(), [0]], [tc.gates.x(), [1]])
    exp_dense = c_dense.expectation([tc.gates.z(), [0]], [tc.gates.x(), [1]])
    np.testing.assert_allclose(exp_mps, exp_dense, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_circuit_cmz_gate(contractor_setup, backend):
    n = 5

    # 1. BaseCircuit implementation via exact TensorNetwork MPS (cmz)
    c_mps = tc.Circuit(n)
    for i in range(n):
        c_mps.h(i)
    c_mps.cmz(*range(n))
    for i in range(n):
        c_mps.rx(i, theta=0.5)

    s_mps = c_mps.state()

    # 2. Dense Matrix baseline comparison (C...CZ)
    c_dense = tc.Circuit(n)
    for i in range(n):
        c_dense.h(i)

    diag = np.ones(2**n, dtype=np.complex128)
    diag[-1] = -1.0  # Only the last element |11...1> gets a -1 phase

    diag_tensor = tc.backend.convert_to_tensor(diag)
    diag_matrix = tc.backend.diagflat(diag_tensor)
    c_dense.any(*range(n), unitary=diag_matrix)

    for i in range(n):
        c_dense.rx(i, theta=0.5)

    s_dense = c_dense.state()

    np.testing.assert_allclose(s_mps, s_dense, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_circuit_ad_rzm(contractor_setup, backend):

    def test_grad_mps(theta):
        c = tc.Circuit(3)
        for i in range(3):
            c.h(i)
        c.rzm(0, 1, 2, theta=theta)
        return tc.backend.real(c.expectation([tc.gates.z(), [0]]))

    def test_grad_dense(theta):
        c = tc.Circuit(3)
        for i in range(3):
            c.h(i)
        theta_t = tc.backend.cast(theta, tc.dtypestr)
        c_val = tc.backend.cos(theta_t / 2)
        s_val = tc.backend.sin(theta_t / 2)
        z_str = tc.backend.cast(
            tc.backend.convert_to_tensor([1, -1, -1, 1, -1, 1, 1, -1]), tc.dtypestr
        )
        diag = (
            tc.backend.cast(c_val, tc.dtypestr)
            * tc.backend.ones([8], dtype=tc.dtypestr)
            - 1j * tc.backend.cast(s_val, tc.dtypestr) * z_str
        )
        diag_tensor = tc.backend.convert_to_tensor(diag)
        diag_matrix = tc.backend.diagflat(diag_tensor)
        c.any(0, 1, 2, unitary=diag_matrix)
        return tc.backend.real(c.expectation([tc.gates.z(), [0]]))

    grad_mps = tc.backend.grad(test_grad_mps)
    grad_dense = tc.backend.grad(test_grad_dense)
    g1 = grad_mps(tc.backend.convert_to_tensor(1.2))
    g2 = grad_dense(tc.backend.convert_to_tensor(1.2))

    np.testing.assert_allclose(g1, g2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_dmcircuit_rzm_cmz(contractor_setup, backend):
    n = 3
    c_dm = tc.DMCircuit(n)
    c_pure = tc.Circuit(n)

    for i in range(n):
        c_dm.h(i)
        c_pure.h(i)

    c_dm.rzm(*range(n), theta=1.5)
    c_pure.rzm(*range(n), theta=1.5)

    c_dm.cmz(*range(n))
    c_pure.cmz(*range(n))

    np.testing.assert_allclose(
        c_dm.densitymatrix(),
        c_pure.state()[:, None] @ np.conj(c_pure.state()[None, :]),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("jaxb"), lf("tfb")])
@pytest.mark.parametrize("contractor_setup", ["cotengra"], indirect=True)
def test_qir_fallback(contractor_setup, backend):
    n = 3
    c = tc.Circuit(n)
    c.h(0)
    c.rzm(0, 1, 2, theta=1.0)
    c.cmz(0, 1)

    qir = c.to_qir()
    c2 = tc.Circuit.from_qir(qir, circuit_params={"nqubits": n})
    np.testing.assert_allclose(c.state(), c2.state(), atol=1e-5)
