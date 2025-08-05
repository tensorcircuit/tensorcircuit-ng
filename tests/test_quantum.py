# pylint: disable=invalid-name

import os
import sys
from functools import partial

import numpy as np
import pytest
import tensornetwork as tn
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from tensorcircuit import quantum as qu

# Note that the first version of this file is adpated from source code of tensornetwork: (Apache2)
# https://github.com/google/TensorNetwork/blob/master/tensornetwork/quantum/quantum_test.py

# tc.set_contractor("greedy")
atol = 1e-5  # relax jax 32 precision
decimal = 5


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_constructor(backend):
    psi_tensor = np.random.rand(2, 2)
    psi_node = tn.Node(psi_tensor)

    op = qu.quantum_constructor([psi_node[0]], [psi_node[1]])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    assert len(op.out_edges) == 1
    assert len(op.in_edges) == 1
    assert op.out_edges[0] is psi_node[0]
    assert op.in_edges[0] is psi_node[1]

    op = qu.quantum_constructor([psi_node[0], psi_node[1]], [])
    assert not op.is_scalar()
    assert op.is_vector()
    assert not op.is_adjoint_vector()
    assert len(op.out_edges) == 2
    assert len(op.in_edges) == 0
    assert op.out_edges[0] is psi_node[0]
    assert op.out_edges[1] is psi_node[1]

    op = qu.quantum_constructor([], [psi_node[0], psi_node[1]])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert op.is_adjoint_vector()
    assert len(op.out_edges) == 0
    assert len(op.in_edges) == 2
    assert op.in_edges[0] is psi_node[0]
    assert op.in_edges[1] is psi_node[1]

    with pytest.raises(ValueError):
        op = qu.quantum_constructor([], [], [psi_node])

    _ = psi_node[0] ^ psi_node[1]
    op = qu.quantum_constructor([], [], [psi_node])
    assert op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    assert len(op.out_edges) == 0
    assert len(op.in_edges) == 0


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_checks(backend):
    node1 = tn.Node(np.random.rand(2, 2))
    node2 = tn.Node(np.random.rand(2, 2))
    _ = node1[1] ^ node2[0]

    # extra dangling edges must be explicitly ignored
    with pytest.raises(ValueError):
        _ = qu.QuVector([node1[0]])

    # correctly ignore the extra edge
    _ = qu.QuVector([node1[0]], ignore_edges=[node2[1]])

    # in/out edges must be dangling
    with pytest.raises(ValueError):
        _ = qu.QuVector([node1[0], node1[1], node2[1]])


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_from_tensor(backend):
    psi_tensor = np.random.rand(2, 2)

    op = qu.QuOperator.from_tensor(psi_tensor, [0], [1])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    np.testing.assert_almost_equal(op.eval(), psi_tensor, decimal=decimal)

    op = qu.QuVector.from_tensor(psi_tensor, [0, 1])
    assert not op.is_scalar()
    assert op.is_vector()
    assert not op.is_adjoint_vector()
    np.testing.assert_almost_equal(op.eval(), psi_tensor, decimal=decimal)

    op = qu.QuAdjointVector.from_tensor(psi_tensor, [0, 1])
    assert not op.is_scalar()
    assert not op.is_vector()
    assert op.is_adjoint_vector()
    np.testing.assert_almost_equal(op.eval(), psi_tensor, decimal=decimal)

    op = qu.QuScalar.from_tensor(1.0)
    assert op.is_scalar()
    assert not op.is_vector()
    assert not op.is_adjoint_vector()
    assert op.eval() == 1.0


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_identity(backend):
    E = qu.identity((2, 3, 4), dtype=np.float64)
    for n in E.nodes:
        assert isinstance(n, tn.CopyNode)
    twentyfour = E.trace()
    for n in twentyfour.nodes:
        assert isinstance(n, tn.CopyNode)
    assert twentyfour.eval() == 24

    tensor = np.random.rand(2, 2)
    psi = qu.QuVector.from_tensor(tensor)
    E = qu.identity((2, 2), dtype=np.float64)
    np.testing.assert_allclose((E @ psi).eval(), psi.eval(), atol=atol)

    np.testing.assert_allclose(
        (psi.adjoint() @ E @ psi).eval(), psi.norm().eval(), atol=atol
    )

    op = qu.QuOperator.from_tensor(tensor, [0], [1])
    op_I = op.tensor_product(E)
    op_times_4 = op_I.partial_trace([1, 2])
    np.testing.assert_allclose(op_times_4.eval(), 4 * op.eval(), atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_tensor_product(backend):
    psi = qu.QuVector.from_tensor(np.random.rand(2, 2))
    psi_psi = psi.tensor_product(psi)
    assert len(psi_psi.subsystem_edges) == 4
    np.testing.assert_almost_equal(
        psi_psi.norm().eval(), psi.norm().eval() ** 2, decimal=decimal
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_matmul(backend):
    mat = np.random.rand(2, 2)
    op = qu.QuOperator.from_tensor(mat, [0], [1])
    res = (op @ op).eval()
    np.testing.assert_allclose(res, mat @ mat, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_mul(backend):
    mat = np.eye(2)
    scal = np.float64(0.5)
    op = qu.QuOperator.from_tensor(mat, [0], [1])
    scal_op = qu.QuScalar.from_tensor(scal)

    res = (op * scal_op).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    res = (scal_op * op).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    res = (scal_op * scal_op).eval()
    np.testing.assert_almost_equal(res, 0.25, decimal=decimal)

    res = (op * np.float64(0.5)).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    res = (np.float64(0.5) * op).eval()
    np.testing.assert_allclose(res, mat * 0.5, atol=atol)

    with pytest.raises(ValueError):
        _ = op * op

    with pytest.raises(ValueError):
        _ = op * mat


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_expectations(backend):
    psi_tensor = np.random.rand(2, 2, 2) + 1.0j * np.random.rand(2, 2, 2)
    op_tensor = np.random.rand(2, 2) + 1.0j * np.random.rand(2, 2)

    psi = qu.QuVector.from_tensor(psi_tensor)
    op = qu.QuOperator.from_tensor(op_tensor, [0], [1])

    op_3 = op.tensor_product(qu.identity((2, 2), dtype=psi_tensor.dtype))
    res1 = (psi.adjoint() @ op_3 @ psi).eval()

    rho_1 = psi.reduced_density([1, 2])  # trace out sites 2 and 3
    res2 = (op @ rho_1).trace().eval()

    np.testing.assert_almost_equal(res1, res2, decimal=decimal)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_projector(backend):
    psi_tensor = np.random.rand(2, 2)
    psi_tensor /= np.linalg.norm(psi_tensor)
    psi = qu.QuVector.from_tensor(psi_tensor)
    P = psi.projector()
    np.testing.assert_allclose((P @ psi).eval(), psi_tensor, atol=atol)

    np.testing.assert_allclose((P @ P).eval(), P.eval(), atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_nonsquare_quop(backend):
    op = qu.QuOperator.from_tensor(np.ones([2, 2, 2, 2, 2]), [0, 1, 2], [3, 4])
    op2 = qu.QuOperator.from_tensor(np.ones([2, 2, 2, 2, 2]), [0, 1], [2, 3, 4])
    np.testing.assert_allclose(
        (op @ op2).eval(), 4 * np.ones([2, 2, 2, 2, 2, 2]), atol=atol
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_expectation_local_tensor(backend):
    op = qu.QuOperator.from_local_tensor(
        np.array([[1.0, 0.0], [0.0, 1.0]]), space=[2, 2, 2, 2], loc=[1]
    )
    state = np.zeros([2, 2, 2, 2])
    state[0, 0, 0, 0] = 1.0
    psi = qu.QuVector.from_tensor(state)
    psi_d = psi.adjoint()
    np.testing.assert_allclose((psi_d @ op @ psi).eval(), 1.0, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_rm_state_vs_mps(backend):
    @partial(tc.backend.jit, jit_compile=False, static_argnums=(1, 2))
    def entanglement1(param, n, nlayers):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, param, nlayers)
        w = c.wavefunction()
        rm = qu.reduced_density_matrix(w, int(n / 2))
        return qu.entropy(rm)

    @partial(tc.backend.jit, jit_compile=False, static_argnums=(1, 2))
    def entanglement2(param, n, nlayers):
        c = tc.Circuit(n)
        c = tc.templates.blocks.example_block(c, param, nlayers)
        w = c.get_quvector()
        rm = w.reduced_density([i for i in range(int(n / 2))])
        return qu.entropy(rm)

    param = tc.backend.ones([6, 6])
    rm1 = entanglement1(param, 6, 3)
    rm2 = entanglement2(param, 6, 3)
    np.testing.assert_allclose(rm1, rm2, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_trace_product(backend):
    o = np.ones([2, 2])
    h = np.eye(2)
    np.testing.assert_allclose(qu.trace_product(o, h), 2, atol=atol)
    oq = qu.QuOperator.from_tensor(o)
    hq = qu.QuOperator.from_tensor(h)
    np.testing.assert_allclose(qu.trace_product(oq, hq), 2, atol=atol)
    np.testing.assert_allclose(qu.trace_product(oq, h), 2, atol=atol)
    np.testing.assert_allclose(qu.trace_product(o, hq), 2, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_free_energy(backend):
    rho = np.array([[1.0, 0], [0, 0]])
    h = np.array([[-1.0, 0], [0, 1]])
    np.testing.assert_allclose(qu.free_energy(rho, h, 0.5), -1, atol=atol)
    np.testing.assert_allclose(qu.renyi_free_energy(rho, h, 0.5), -1, atol=atol)
    hq = qu.QuOperator.from_tensor(h)
    np.testing.assert_allclose(qu.free_energy(rho, hq, 0.5), -1, atol=atol)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_measurement_counts(backend):
    state = np.ones([4])
    ct, cs = qu.measurement_counts(state, format="count_tuple")
    np.testing.assert_allclose(ct.shape[0], 4, atol=atol)
    np.testing.assert_allclose(tc.backend.sum(cs), 8192, atol=atol)
    state = np.ones([2, 2])
    ct, cs = qu.measurement_counts(state, format="count_tuple")
    np.testing.assert_allclose(ct.shape[0], 2, atol=atol)
    np.testing.assert_allclose(tc.backend.sum(cs), 8192, atol=atol)
    state = np.array([1.0, 1.0, 0, 0])
    print(qu.measurement_counts(state))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_extract_from_measure(backend):
    np.testing.assert_allclose(
        qu.spin_by_basis(2, 1), np.array([1, -1, 1, -1]), atol=atol
    )
    state = tc.array_to_tensor(np.array([0.6, 0.4, 0, 0]))
    np.testing.assert_allclose(
        qu.correlation_from_counts([0, 1], state), 0.2, atol=atol
    )
    np.testing.assert_allclose(qu.correlation_from_counts([1], state), 0.2, atol=atol)

    samples_int = tc.array_to_tensor(np.array([0, 0, 3, 3, 3]), dtype="int32")
    r = qu.correlation_from_samples([0, 1], samples_int, n=2)
    np.testing.assert_allclose(r, 1, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_heisenberg_ham(backend):
    g = tc.templates.graphs.Line1D(6)
    h = tc.quantum.heisenberg_hamiltonian(g, sparse=False)
    e, _ = tc.backend.eigh(h)
    np.testing.assert_allclose(e[0], -11.2111, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_reduced_density_from_density(backend):
    n = 6
    w = np.random.normal(size=[2**n]) + 1.0j * np.random.normal(size=[2**n])
    w /= np.linalg.norm(w)
    rho = np.reshape(w, [-1, 1]) @ np.reshape(np.conj(w), [1, -1])
    dm1 = tc.quantum.reduced_density_matrix(w, cut=[0, 2])
    dm2 = tc.quantum.reduced_density_matrix(rho, cut=[0, 2])
    np.testing.assert_allclose(dm1, dm2, atol=1e-5)

    # with p
    n = 5
    w = np.random.normal(size=[2**n]) + 1.0j * np.random.normal(size=[2**n])
    w /= np.linalg.norm(w)
    p = np.random.normal(size=[2**3])
    p = tc.backend.softmax(p)
    p = tc.backend.cast(p, "complex128")
    rho = np.reshape(w, [-1, 1]) @ np.reshape(np.conj(w), [1, -1])
    dm1 = tc.quantum.reduced_density_matrix(w, cut=[1, 2, 3], p=p)
    dm2 = tc.quantum.reduced_density_matrix(rho, cut=[1, 2, 3], p=p)
    np.testing.assert_allclose(dm1, dm2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_mutual_information(backend):
    n = 5
    w = np.random.normal(size=[2**n]) + 1.0j * np.random.normal(size=[2**n])
    w /= np.linalg.norm(w)
    rho = np.reshape(w, [-1, 1]) @ np.reshape(np.conj(w), [1, -1])
    dm1 = tc.quantum.mutual_information(w, cut=[1, 2, 3])
    dm2 = tc.quantum.mutual_information(rho, cut=[1, 2, 3])
    np.testing.assert_allclose(dm1, dm2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_expectation_quantum(backend):
    c = tc.Circuit(3)
    c.ry(0, theta=0.4)
    c.cnot(0, 1)
    exp1 = c.expectation([tc.gates.z(), [0]], [tc.gates.z(), [2]], reuse=False)
    qv = c.quvector()
    exp2 = tc.expectation([tc.gates.z(), [0]], [tc.gates.z(), [2]], ket=qv)
    np.testing.assert_allclose(exp1, exp2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_ee(backend):
    c = tc.Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    s = c.state()
    np.testing.assert_allclose(
        tc.quantum.entanglement_entropy(s, [0, 1]), np.log(2.0), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_negativity(backend, highp):
    c = tc.DMCircuit(2)
    c.h(0)
    c.cnot(0, 1)
    c.depolarizing(0, px=0.1, py=0.1, pz=0.1)
    dm = c.state()
    np.testing.assert_allclose(
        tc.quantum.log_negativity(dm, [0], base="2"), 0.485427, atol=1e-5
    )
    np.testing.assert_allclose(
        tc.quantum.partial_transpose(tc.quantum.partial_transpose(dm, [0]), [0]),
        dm,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        tc.quantum.entanglement_negativity(dm, [1]), 0.2, atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_tn2qop(backend):
    nwires = 6
    dtype = np.complex64
    # only obc is supported, even if you supply nwires Jx terms
    Jx = np.array([1.0 for _ in range(nwires - 1)])  # strength of xx interaction (OBC)
    Bz = np.array([-1.0 for _ in range(nwires)])  # strength of transverse field
    tn_mpo = tn.matrixproductstates.mpo.FiniteTFI(Jx, Bz, dtype=dtype)
    qu_mpo = tc.quantum.tn2qop(tn_mpo)
    h1 = qu_mpo.eval_matrix()
    g = tc.templates.graphs.Line1D(nwires, pbc=False)
    h2 = tc.quantum.heisenberg_hamiltonian(
        g, hzz=0, hxx=1, hyy=0, hz=1, hx=0, hy=0, sparse=False, numpy=True
    )
    np.testing.assert_allclose(h1, h2, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_qb2qop(backend):
    try:
        import quimb
    except ImportError:
        pytest.skip("quimb is not installed")
    nwires = 6
    qb_mpo = quimb.tensor.tensor_builder.MPO_ham_ising(nwires, 4, 2, cyclic=True)
    qu_mpo = tc.quantum.quimb2qop(qb_mpo)
    h1 = qu_mpo.eval_matrix()
    g = tc.templates.graphs.Line1D(nwires, pbc=True)
    h2 = tc.quantum.heisenberg_hamiltonian(
        g, hzz=1, hxx=0, hyy=0, hz=0, hx=-1, hy=0, sparse=False, numpy=True
    )
    np.testing.assert_allclose(h1, h2, atol=1e-5)

    # in out edge order test
    builder = quimb.tensor.tensor_builder.SpinHam1D()
    # new version quimb breaking API change: SpinHam1D -> SpinHam
    builder += 1, "Y"
    builder += 1, "X"
    H = builder.build_mpo(3)
    h = tc.quantum.quimb2qop(H)
    m1 = h.eval_matrix()
    g = tc.templates.graphs.Line1D(3, pbc=False)
    m2 = tc.quantum.heisenberg_hamiltonian(
        g, hzz=0, hxx=0, hyy=0, hz=0, hy=0.5, hx=0.5, sparse=False, numpy=True
    )
    np.testing.assert_allclose(m1, m2, atol=1e-5)

    # test mps case

    s1 = quimb.tensor.tensor_builder.MPS_rand_state(3, 4)
    s2 = tc.quantum.quimb2qop(s1)
    m1 = s1.to_dense()
    m2 = s2.eval_matrix()
    np.testing.assert_allclose(m1, m2, atol=1e-5)


# TODO(@Charlespkuer): Add more conversion functions for other packages
@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_tenpy2qop(backend):
    """
    Tests the conversion from TeNPy MPO/MPS to TensorCircuit QuOperator.
    This test verifies the numerical correctness against a precisely matched
    Hamiltonian and state vector.
    """
    try:
        from tenpy.models.tf_ising import TFIChain
        from tenpy.networks.mps import MPS
    except ImportError:
        pytest.skip("TeNPy is not installed")

    nwires = 4
    Jx = 1.0
    Bz = -1.0
    model_params = {"L": nwires, "J": Jx, "g": Bz, "bc_MPS": "finite"}
    model = TFIChain(model_params)

    # MPO conversion
    tenpy_mpo = model.H_MPO
    qu_mpo = tc.quantum.tenpy2qop(tenpy_mpo)
    h_actual = qu_mpo.eval_matrix()
    g = tc.templates.graphs.Line1D(nwires, pbc=False)
    h_expected = tc.quantum.heisenberg_hamiltonian(
        g, hzz=0, hxx=-Jx, hyy=0, hz=Bz, hx=0, hy=0, sparse=False, numpy=True
    )
    np.testing.assert_allclose(h_actual, h_expected, atol=1e-5)

    # MPS conversion
    psi_tenpy = MPS.from_product_state(
        model.lat.mps_sites(), [0, 1] * (nwires // 2), bc=model.lat.bc_MPS
    )
    qu_mps = tc.quantum.tenpy2qop(psi_tenpy)
    psi_actual = np.reshape(qu_mps.eval_matrix(), -1)
    psi_expected = np.zeros(2**nwires)
    psi_expected[0b1010] = 1.0
    np.testing.assert_allclose(psi_actual, psi_expected, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_qop2tenpy(backend):
    # MPS conversion
    nwires_mps = 4
    chi_mps = 2
    mps_tensors = (
        [np.random.rand(1, 2, chi_mps)]
        + [np.random.rand(chi_mps, 2, chi_mps) for _ in range(nwires_mps - 2)]
        + [np.random.rand(chi_mps, 2, 1)]
    )
    mps_nodes = [tn.Node(t) for t in mps_tensors]
    for i in range(nwires_mps - 1):
        mps_nodes[i][2] ^ mps_nodes[i + 1][0]
    qop_mps = tc.QuOperator(
        [], [n[1] for n in mps_nodes], mps_nodes, [mps_nodes[0][0], mps_nodes[-1][2]]
    )
    tenpy_mps = tc.quantum.qop2tenpy(qop_mps)
    mat = qop_mps.eval_matrix()
    vec_from_qop = tc.backend.reshape(mat, [-1])
    full_wavefunction_tensor = tenpy_mps.get_theta(0, tenpy_mps.L)
    vec_from_tenpy = np.ravel(full_wavefunction_tensor.to_ndarray())
    np.testing.assert_allclose(vec_from_qop, vec_from_tenpy, atol=1e-5)

    # MPO conversion
    nwires_mpo = 4
    chi_mpo = 3
    t_left = np.random.rand(1, 2, 2, chi_mpo)
    t_bulk = [np.random.rand(chi_mpo, 2, 2, chi_mpo) for _ in range(nwires_mpo - 2)]
    t_right = np.random.rand(chi_mpo, 2, 2, 1)
    mpo_tensors = [t_left] + t_bulk + [t_right]
    mpo_nodes = [tn.Node(t) for t in mpo_tensors]
    for i in range(nwires_mpo - 1):
        mpo_nodes[i][3] ^ mpo_nodes[i + 1][0]
    qop_mpo = tc.QuOperator(
        [n[2] for n in mpo_nodes],
        [n[1] for n in mpo_nodes],
        mpo_nodes,
        [mpo_nodes[0][0], mpo_nodes[-1][3]],
    )
    tenpy_mpo = tc.quantum.qop2tenpy(qop_mpo)
    mat_from_qop = qop_mpo.eval_matrix()

    mpo_contract_nwires = tenpy_mpo.L
    canonical_order = ["wL", "wR", "p", "p*"]
    contraction_nodes = []
    for i in range(mpo_contract_nwires):
        W_tenpy_array = tenpy_mpo.get_W(i)
        existing_labels = [
            label for label in canonical_order if label in W_tenpy_array._labels
        ]
        W_transposed = W_tenpy_array.itranspose(existing_labels)
        contraction_nodes.append(tn.Node(W_transposed.to_ndarray()))

    for i in range(mpo_contract_nwires - 1):
        contraction_nodes[i][1] ^ contraction_nodes[i + 1][0]

    chi_left = contraction_nodes[0].shape[0]
    chi_right = contraction_nodes[-1].shape[1]
    left_bc_vec = np.zeros(chi_left)
    left_bc_vec[tenpy_mpo.IdL] = 1.0
    right_bc_vec = np.zeros(chi_right)
    right_bc_vec[tenpy_mpo.IdR] = 1.0
    left_node = tn.Node(left_bc_vec)
    right_node = tn.Node(right_bc_vec)
    contraction_nodes[0][0] ^ left_node[0]
    contraction_nodes[-1][1] ^ right_node[0]

    output_edges = [node[3] for node in contraction_nodes]
    input_edges = [node[2] for node in contraction_nodes]
    result_node = tn.contractors.auto(
        contraction_nodes + [left_node, right_node],
        output_edge_order=output_edges + input_edges,
    )
    mat_from_tenpy = tc.backend.reshape(
        result_node.tensor, (2**mpo_contract_nwires, 2**mpo_contract_nwires)
    )

    np.testing.assert_allclose(mat_from_qop, mat_from_tenpy, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_tenpy_roundtrip(backend):
    try:
        import tensornetwork as tn
        from tenpy.networks.mps import MPS
        from tenpy.networks.mpo import MPO
        from tenpy.networks.site import Site
        from tenpy.linalg.charges import LegCharge
        from tenpy.linalg import np_conserved as npc
    except ImportError:
        pytest.skip("TeNPy or TensorNetwork is not installed")

    nwires_mpo = 3
    chi_mpo = 4
    phys_dim = 2
    sites = [Site(LegCharge.from_trivial(phys_dim), "q") for _ in range(nwires_mpo)]
    t_left = np.random.rand(1, chi_mpo, phys_dim, phys_dim)
    Ws = [
        npc.Array.from_ndarray(
            t_left,
            labels=["wL", "wR", "p", "p*"],
            legcharges=[LegCharge.from_trivial(s) for s in t_left.shape],
        )
    ]
    for _ in range(nwires_mpo - 2):
        t_bulk = np.random.rand(chi_mpo, chi_mpo, phys_dim, phys_dim)
        Ws.append(
            npc.Array.from_ndarray(
                t_bulk,
                labels=["wL", "wR", "p", "p*"],
                legcharges=[LegCharge.from_trivial(s) for s in t_bulk.shape],
            )
        )
    t_right = np.random.rand(chi_mpo, 1, phys_dim, phys_dim)
    Ws.append(
        npc.Array.from_ndarray(
            t_right,
            labels=["wL", "wR", "p", "p*"],
            legcharges=[LegCharge.from_trivial(s) for s in t_right.shape],
        )
    )
    mpo_original = MPO(sites, Ws, IdL=0, IdR=chi_mpo - 1)

    qop_mpo = tc.quantum.tenpy2qop(mpo_original)
    mpo_roundtrip = tc.quantum.qop2tenpy(qop_mpo)

    try:
        mat_original = mpo_original.to_matrix()
        mat_roundtrip = mpo_roundtrip.to_matrix()
    except AttributeError:
        # Inlined logic for mpo_original
        contraction_nodes_orig = []
        canonical_order_orig = ["wL", "wR", "p", "p*"]
        for i in range(mpo_original.L):
            W_tenpy_array = mpo_original.get_W(i)
            W_transposed = W_tenpy_array.itranspose(canonical_order_orig)
            contraction_nodes_orig.append(tn.Node(W_transposed.to_ndarray()))
        for i in range(mpo_original.L - 1):
            contraction_nodes_orig[i][1] ^ contraction_nodes_orig[i + 1][0]
        chi_left_orig = contraction_nodes_orig[0].shape[0]
        chi_right_orig = contraction_nodes_orig[-1].shape[1]
        left_bc_vec_orig = np.zeros(chi_left_orig)
        left_idx_orig = 0 if chi_left_orig == 1 else mpo_original.IdL
        left_bc_vec_orig[left_idx_orig] = 1.0
        right_bc_vec_orig = np.zeros(chi_right_orig)
        right_idx_orig = 0 if chi_right_orig == 1 else mpo_original.IdR
        right_bc_vec_orig[right_idx_orig] = 1.0
        left_node_orig = tn.Node(left_bc_vec_orig)
        right_node_orig = tn.Node(right_bc_vec_orig)
        contraction_nodes_orig[0][0] ^ left_node_orig[0]
        contraction_nodes_orig[-1][1] ^ right_node_orig[0]
        output_edges_orig = [node[3] for node in contraction_nodes_orig]
        input_edges_orig = [node[2] for node in contraction_nodes_orig]
        result_node_orig = tn.contractors.auto(
            contraction_nodes_orig + [left_node_orig, right_node_orig],
            output_edge_order=output_edges_orig + input_edges_orig,
        )
        D_orig = np.prod(mpo_original.dim)
        mat_original = tc.backend.reshape(result_node_orig.tensor, (D_orig, D_orig))

        # Inlined logic for mpo_roundtrip
        contraction_nodes_rt = []
        canonical_order_rt = ["wL", "wR", "p", "p*"]
        for i in range(mpo_roundtrip.L):
            W_tenpy_array_rt = mpo_roundtrip.get_W(i)
            W_transposed_rt = W_tenpy_array_rt.itranspose(canonical_order_rt)
            contraction_nodes_rt.append(tn.Node(W_transposed_rt.to_ndarray()))
        for i in range(mpo_roundtrip.L - 1):
            contraction_nodes_rt[i][1] ^ contraction_nodes_rt[i + 1][0]
        chi_left_rt = contraction_nodes_rt[0].shape[0]
        chi_right_rt = contraction_nodes_rt[-1].shape[1]
        left_bc_vec_rt = np.zeros(chi_left_rt)
        left_idx_rt = 0 if chi_left_rt == 1 else mpo_roundtrip.IdL
        left_bc_vec_rt[left_idx_rt] = 1.0
        right_bc_vec_rt = np.zeros(chi_right_rt)
        right_idx_rt = 0 if chi_right_rt == 1 else mpo_roundtrip.IdR
        right_bc_vec_rt[right_idx_rt] = 1.0
        left_node_rt = tn.Node(left_bc_vec_rt)
        right_node_rt = tn.Node(right_bc_vec_rt)
        contraction_nodes_rt[0][0] ^ left_node_rt[0]
        contraction_nodes_rt[-1][1] ^ right_node_rt[0]
        output_edges_rt = [node[3] for node in contraction_nodes_rt]
        input_edges_rt = [node[2] for node in contraction_nodes_rt]
        result_node_rt = tn.contractors.auto(
            contraction_nodes_rt + [left_node_rt, right_node_rt],
            output_edge_order=output_edges_rt + input_edges_rt,
        )
        D_rt = np.prod(mpo_roundtrip.dim)
        mat_roundtrip = tc.backend.reshape(result_node_rt.tensor, (D_rt, D_rt))

    np.testing.assert_allclose(mat_original, mat_roundtrip, atol=1e-5)

    nwires_mps = 4
    chi_mps = 5
    sites_mps = [Site(LegCharge.from_trivial(phys_dim), "q") for _ in range(nwires_mps)]
    b_left = np.random.rand(1, phys_dim, chi_mps)
    Bs = [
        npc.Array.from_ndarray(
            b_left,
            labels=["vL", "p", "vR"],
            legcharges=[LegCharge.from_trivial(s) for s in b_left.shape],
        )
    ]
    for _ in range(nwires_mps - 2):
        b_bulk = np.random.rand(chi_mps, phys_dim, chi_mps)
        Bs.append(
            npc.Array.from_ndarray(
                b_bulk,
                labels=["vL", "p", "vR"],
                legcharges=[LegCharge.from_trivial(s) for s in b_bulk.shape],
            )
        )
    b_right = np.random.rand(chi_mps, phys_dim, 1)
    Bs.append(
        npc.Array.from_ndarray(
            b_right,
            labels=["vL", "p", "vR"],
            legcharges=[LegCharge.from_trivial(s) for s in b_right.shape],
        )
    )
    SVs = [np.ones([1])]
    for B in Bs[:-1]:
        sv_dim = B.get_leg("vR").ind_len
        SVs.append(np.ones(sv_dim))
    SVs.append(np.ones([1]))
    mps_original = MPS(sites_mps, Bs, SVs)

    qop_mps = tc.quantum.tenpy2qop(mps_original)
    mps_roundtrip = tc.quantum.qop2tenpy(qop_mps)

    try:
        vec_original = mps_original.to_ndarray()
        vec_roundtrip = mps_roundtrip.to_ndarray()
    except AttributeError:
        full_theta_orig = mps_original.get_theta(0, mps_original.L)
        state_tensor_orig = full_theta_orig.to_ndarray()
        vec_original = state_tensor_orig.reshape(-1)

        full_theta_rt = mps_roundtrip.get_theta(0, mps_roundtrip.L)
        state_tensor_rt = full_theta_rt.to_ndarray()
        vec_roundtrip = state_tensor_rt.reshape(-1)

    np.testing.assert_allclose(vec_original, vec_roundtrip, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_qop2quimb(backend):
    # MPO Conversion
    nwires_mpo = 4
    chi_mpo = 3
    phys_dim = 2

    t_left = np.random.rand(1, phys_dim, phys_dim, chi_mpo)
    t_bulk = [
        np.random.rand(chi_mpo, phys_dim, phys_dim, chi_mpo)
        for _ in range(nwires_mpo - 2)
    ]
    t_right = np.random.rand(chi_mpo, phys_dim, phys_dim, 1)
    mpo_tensors = [t_left] + t_bulk + [t_right]
    mpo_nodes = [tn.Node(t) for t in mpo_tensors]

    for i in range(nwires_mpo - 1):
        mpo_nodes[i][3] ^ mpo_nodes[i + 1][0]

    qop_mpo = tc.QuOperator(
        out_edges=[n[2] for n in mpo_nodes],
        in_edges=[n[1] for n in mpo_nodes],
        ref_nodes=mpo_nodes,
        ignore_edges=[mpo_nodes[0][0], mpo_nodes[-1][3]],
    )

    quimb_mpo = tc.quantum.qop2quimb(qop_mpo)

    mat_from_qop = qop_mpo.eval_matrix()

    ket_inds_mpo = [f"k{i}" for i in range(nwires_mpo)]
    bra_inds_mpo = [f"b{i}" for i in range(nwires_mpo)]
    mat_from_quimb = quimb_mpo.to_dense(ket_inds_mpo, bra_inds_mpo)

    np.testing.assert_allclose(mat_from_qop, mat_from_quimb, atol=1e-5)

    # MPS Conversion
    nwires_mps = 5
    chi_mps = 8

    mps_tensors = (
        [np.random.rand(1, phys_dim, chi_mps)]
        + [np.random.rand(chi_mps, phys_dim, chi_mps) for _ in range(nwires_mps - 2)]
        + [np.random.rand(chi_mps, phys_dim, 1)]
    )
    mps_nodes = [tn.Node(t) for t in mps_tensors]

    for i in range(nwires_mps - 1):
        mps_nodes[i][2] ^ mps_nodes[i + 1][0]

    qop_mps = tc.QuOperator(
        out_edges=[n[1] for n in mps_nodes],
        in_edges=[],
        ref_nodes=mps_nodes,
        ignore_edges=[mps_nodes[0][0], mps_nodes[-1][2]],
    )

    quimb_mps = tc.quantum.qop2quimb(qop_mps)

    mat_from_qop = qop_mps.eval_matrix()
    vec_from_qop = tc.backend.reshape(mat_from_qop, [-1])

    ket_inds_mps = [f"k{i}" for i in range(nwires_mps)]
    vec_from_quimb = quimb_mps.to_dense(ket_inds_mps).flatten()

    vec_from_qop /= np.linalg.norm(vec_from_qop)
    vec_from_quimb /= np.linalg.norm(vec_from_quimb)

    assert abs(abs(np.vdot(vec_from_qop, vec_from_quimb)) - 1.0) < 1e-5


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_quimb_roundtrip(backend):
    try:
        import quimb.tensor as qtn
    except ImportError:
        pytest.skip("quimb is not installed")
    # MPO roundtrip test
    nwires_mpo = 4
    mpo_original = qtn.MPO_ham_ising(nwires_mpo)
    qop_mpo = tc.quantum.quimb2qop(mpo_original)
    mpo_roundtrip = tc.quantum.qop2quimb(qop_mpo)
    ket_inds = [f"k{i}" for i in range(nwires_mpo)]
    bra_inds = [f"b{i}" for i in range(nwires_mpo)]
    mat_original = mpo_original.to_dense(ket_inds, bra_inds)
    mat_roundtrip = mpo_roundtrip.to_dense(ket_inds, bra_inds)
    np.testing.assert_allclose(mat_original, mat_roundtrip, atol=1e-5)

    # MPS roundtrip test
    nwires_mps = 5
    bond_dim = 8
    mps_original = qtn.MPS_rand_state(nwires_mps, bond_dim)
    mps_original.normalize()
    qop_mps = tc.quantum.quimb2qop(mps_original)
    mps_roundtrip = tc.quantum.qop2quimb(qop_mps)
    ket_inds_mps = [f"k{i}" for i in range(nwires_mps)]
    vec_original = mps_original.to_dense(ket_inds_mps).flatten()
    vec_roundtrip = mps_roundtrip.to_dense(ket_inds_mps).flatten()
    assert abs(abs(np.vdot(vec_original, vec_roundtrip)) - 1.0) < 1e-5


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_qop2tn(backend):
    try:
        import tensornetwork as tn
        import numpy as np
    except ImportError:
        pytest.skip("TensorNetwork is not installed.")

    nwires = 4
    chi = 5
    phys_dim = 2
    dtype = np.complex64

    tensors_mpo_tc = [np.random.rand(1, chi, phys_dim, phys_dim).astype(dtype)]
    for _ in range(nwires - 2):
        tensors_mpo_tc.append(
            np.random.rand(chi, chi, phys_dim, phys_dim).astype(dtype)
        )
    tensors_mpo_tc.append(np.random.rand(chi, 1, phys_dim, phys_dim).astype(dtype))

    nodes_mpo_tc = [tn.Node(t) for t in tensors_mpo_tc]
    for i in range(nwires - 1):
        nodes_mpo_tc[i][1] ^ nodes_mpo_tc[i + 1][0]

    qop_mpo = tc.QuOperator(
        out_edges=[n[3] for n in nodes_mpo_tc],
        in_edges=[n[2] for n in nodes_mpo_tc],
        ref_nodes=nodes_mpo_tc,
        ignore_edges=[nodes_mpo_tc[0][0], nodes_mpo_tc[-1][1]],
    )
    tn_mpo_rt = tc.quantum.qop2tn(qop_mpo)
    tensors_for_tn_mpo = [tc.backend.numpy(n.tensor) for n in nodes_mpo_tc]
    tn_mpo_orig = tn.matrixproductstates.mpo.FiniteMPO(tensors_for_tn_mpo)

    try:
        mat_original = tn_mpo_orig.todense()
        mat_roundtrip = tn_mpo_rt.todense()
    except AttributeError:
        nodes_orig = [tn.Node(t) for t in tn_mpo_orig.tensors]
        for i in range(len(nodes_orig) - 1):
            nodes_orig[i][1] ^ nodes_orig[i + 1][0]
        left_bc_vec_orig = np.zeros(nodes_orig[0].shape[0], dtype=dtype)
        left_bc_vec_orig[0] = 1.0
        left_node_orig = tn.Node(left_bc_vec_orig)
        nodes_orig[0][0] ^ left_node_orig[0]
        right_bc_vec_orig = np.zeros(nodes_orig[-1].shape[1], dtype=dtype)
        right_bc_vec_orig[0] = 1.0
        right_node_orig = tn.Node(right_bc_vec_orig)
        nodes_orig[-1][1] ^ right_node_orig[0]
        all_nodes_orig = nodes_orig + [left_node_orig, right_node_orig]
        output_edges_orig = [n[3] for n in nodes_orig]
        input_edges_orig = [n[2] for n in nodes_orig]
        result_node_orig = tn.contractors.auto(
            all_nodes_orig, output_edge_order=output_edges_orig + input_edges_orig
        )
        nwires_orig = len(tn_mpo_orig.tensors)
        phys_dim_orig = tn_mpo_orig.tensors[0].shape[2]
        D_orig = phys_dim_orig**nwires_orig
        mat_original = tc.backend.reshape(result_node_orig.tensor, (D_orig, D_orig))

        nodes_rt = [tn.Node(t) for t in tn_mpo_rt.tensors]
        for i in range(len(nodes_rt) - 1):
            nodes_rt[i][1] ^ nodes_rt[i + 1][0]
        left_bc_vec_rt = np.zeros(nodes_rt[0].shape[0], dtype=dtype)
        left_bc_vec_rt[0] = 1.0
        left_node_rt = tn.Node(left_bc_vec_rt)
        nodes_rt[0][0] ^ left_node_rt[0]
        right_bc_vec_rt = np.zeros(nodes_rt[-1].shape[1], dtype=dtype)
        right_bc_vec_rt[0] = 1.0
        right_node_rt = tn.Node(right_bc_vec_rt)
        nodes_rt[-1][1] ^ right_node_rt[0]
        all_nodes_rt = nodes_rt + [left_node_rt, right_node_rt]
        output_edges_rt = [n[3] for n in nodes_rt]
        input_edges_rt = [n[2] for n in nodes_rt]
        result_node_rt = tn.contractors.auto(
            all_nodes_rt, output_edge_order=output_edges_rt + input_edges_rt
        )
        nwires_rt = len(tn_mpo_rt.tensors)
        phys_dim_rt = tn_mpo_rt.tensors[0].shape[2]
        D_rt = phys_dim_rt**nwires_rt
        mat_roundtrip = tc.backend.reshape(result_node_rt.tensor, (D_rt, D_rt))

    np.testing.assert_allclose(mat_original, mat_roundtrip, atol=1e-5)

    tensors_mps_tc = [np.random.rand(1, phys_dim, chi).astype(dtype)]
    for _ in range(nwires - 2):
        tensors_mps_tc.append(np.random.rand(chi, phys_dim, chi).astype(dtype))
    tensors_mps_tc.append(np.random.rand(chi, phys_dim, 1).astype(dtype))

    nodes_mps_tc = [tn.Node(t) for t in tensors_mps_tc]
    for i in range(nwires - 1):
        nodes_mps_tc[i][2] ^ nodes_mps_tc[i + 1][0]

    qop_mps = tc.QuOperator(
        out_edges=[n[1] for n in nodes_mps_tc],
        in_edges=[],
        ref_nodes=nodes_mps_tc,
        ignore_edges=[nodes_mps_tc[0][0], nodes_mps_tc[-1][2]],
    )

    tn_mps_rt = tc.quantum.qop2tn(qop_mps)

    tensors_for_tn_mps = [tc.backend.numpy(n.tensor) for n in nodes_mps_tc]
    tn_mps_orig = tn.FiniteMPS(tensors_for_tn_mps, canonicalize=False)

    try:
        vec_original = tn_mps_orig.todense().flatten()
        vec_roundtrip = tn_mps_rt.todense().flatten()
    except AttributeError:
        nodes_mps_orig = [tn.Node(t) for t in tn_mps_orig.tensors]
        n_orig = len(nodes_mps_orig)
        if n_orig > 1:
            for i in range(n_orig - 1):
                nodes_mps_orig[i][2] ^ nodes_mps_orig[i + 1][0]
        physical_edges_orig = [n[1] for n in nodes_mps_orig]
        dangling_edges_orig = (
            [nodes_mps_orig[0][0]] + physical_edges_orig + [nodes_mps_orig[-1][2]]
        )
        result_node_orig = tn.contractors.auto(
            nodes_mps_orig, output_edge_order=dangling_edges_orig
        )
        vec_original = tc.backend.reshape(result_node_orig.tensor, [-1])

        nodes_mps_rt = [tn.Node(t) for t in tn_mps_rt.tensors]
        n_rt = len(nodes_mps_rt)
        if n_rt > 1:
            for i in range(n_rt - 1):
                nodes_mps_rt[i][2] ^ nodes_mps_rt[i + 1][0]
        physical_edges_rt = [n[1] for n in nodes_mps_rt]
        dangling_edges_rt = (
            [nodes_mps_rt[0][0]] + physical_edges_rt + [nodes_mps_rt[-1][2]]
        )
        result_node_rt = tn.contractors.auto(
            nodes_mps_rt, output_edge_order=dangling_edges_rt
        )
        vec_roundtrip = tc.backend.reshape(result_node_rt.tensor, [-1])

    vec_original /= np.linalg.norm(vec_original)
    vec_roundtrip /= np.linalg.norm(vec_roundtrip)

    assert abs(abs(np.vdot(vec_original, vec_roundtrip)) - 1.0) < 1e-5


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_counts_2(backend):
    z0 = tc.backend.convert_to_tensor(np.array([0.1, 0, -0.3, 0]))
    x, y = tc.quantum.count_d2s(z0)
    print(x, y)
    np.testing.assert_allclose(x, np.array([0, 2]))
    np.testing.assert_allclose(y, np.array([0.1, -0.3]))
    z = tc.quantum.count_s2d((x, y), 2)
    np.testing.assert_allclose(z, z0)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_measurement_results(backend):
    n = 4
    w = tc.backend.ones([2**n])
    r = tc.quantum.measurement_results(w, counts=9, format="sample_bin", jittable=True)
    assert tc.backend.shape_tuple(r) == (9, n)
    print(r)
    r = tc.quantum.measurement_results(w, counts=9, format="sample_int", jittable=True)
    assert tc.backend.shape_tuple(r) == (9,)
    print(r)
    for c in (9, -9):
        r = tc.quantum.measurement_results(
            w, counts=c, format="count_vector", jittable=True
        )
        assert tc.backend.shape_tuple(r) == (2**n,)
        print(r)
        r = tc.quantum.measurement_results(w, counts=c, format="count_tuple")
        print(r)
        r = tc.quantum.measurement_results(
            w, counts=c, format="count_dict_bin", jittable=True
        )
        print(r)
        r = tc.quantum.measurement_results(
            w, counts=c, format="count_dict_int", jittable=True
        )
        print(r)


def test_ps2xyz():
    xyz = {"x": [1], "z": [2]}
    assert tc.quantum.xyz2ps(xyz) == [0, 1, 3]
    assert tc.quantum.xyz2ps(xyz, 6) == [0, 1, 3, 0, 0, 0]
    xyz.update({"y": []})
    assert tc.quantum.ps2xyz([0, 1, 3]) == xyz
    assert tc.quantum.ps2xyz([0, 1, 3, 0]) == xyz


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_reduced_wavefunction(backend):
    c = tc.Circuit(3)
    c.h(0)
    c.cnot(0, 1)
    r = c.cond_measure(0)
    s = c.state()
    s1 = tc.quantum.reduced_wavefunction(s, [0, 2], [r, 0])
    if tc.backend.cast(r, tc.rdtypestr) < 0.5:
        np.testing.assert_allclose(s1, np.array([1, 0]), atol=1e-5)
    else:
        np.testing.assert_allclose(s1, np.array([0, 1]), atol=1e-5)

    c = tc.Circuit(3)
    c.h(0)
    c.cnot(0, 1)
    s = c.state()
    s1 = tc.quantum.reduced_wavefunction(s, [2], [0])

    c1 = tc.Circuit(2)
    c1.h(0)
    c1.cnot(0, 1)
    np.testing.assert_allclose(s1, c1.state(), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1_mask(backend):
    g = tc.templates.graphs.Line1D(8)
    sumz = tc.quantum.heisenberg_hamiltonian(g, hzz=0, hxx=0, hyy=0, hz=1)
    for i in range(9):
        s = tc.quantum.u1_mask(8, i)
        s /= tc.backend.norm(s)
        c = tc.Circuit(8, inputs=s)
        zexp = tc.templates.measurements.operator_expectation(c, sumz)
        np.testing.assert_allclose(zexp, 8 - 2 * i, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_u1_project(backend):
    c = tc.Circuit(8)
    c.x([0, 2, 4])
    c.exp1(0, 1, unitary=tc.gates._swap_matrix, theta=0.6)
    s = c.state()
    s1 = tc.quantum.u1_project(s, 8, 3)
    assert s1.shape[-1] == 56
    np.testing.assert_allclose(tc.quantum.u1_enlarge(s1, 8, 3), s)
