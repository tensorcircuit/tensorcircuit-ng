# pylint: disable=invalid-name

import os
import sys

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_any_measurement():
    c = tc.Circuit(2)
    c.H(0)
    c.H(1)
    mea = np.array([1, 1])
    r = tc.templates.measurements.any_measurements(c, mea, onehot=True)
    np.testing.assert_allclose(r, 1.0, atol=1e-5)
    mea2 = np.array([3, 0])
    r2 = tc.templates.measurements.any_measurements(c, mea2, onehot=True)
    np.testing.assert_allclose(r2, 0.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_parameterized_local_measurement(backend):
    c = tc.Circuit(3)
    c.X(0)
    c.cnot(0, 1)
    c.H(-1)
    basis = tc.backend.convert_to_tensor(np.array([3, 3, 1]))
    r = tc.templates.measurements.parameterized_local_measurements(
        c, structures=basis, onehot=True
    )
    np.testing.assert_allclose(r, np.array([-1, -1, 1]), atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_sparse_expectation(backend):
    ham = tc.backend.coo_sparse_matrix(
        indices=[[0, 1], [1, 0]], values=tc.backend.ones([2]), shape=(2, 2)
    )

    def f(param):
        c = tc.Circuit(1)
        c.rx(0, theta=param[0])
        c.H(0)
        return tc.templates.measurements.sparse_expectation(c, ham)

    fvag = tc.backend.jit(tc.backend.value_and_grad(f))
    param = tc.backend.zeros([1])
    v, g = fvag(param)
    # state is |+> after H on |0>; <X> = 1 at theta=0, d/dtheta <X> = 0
    np.testing.assert_allclose(tc.backend.numpy(v), 1.0, atol=1e-4)
    np.testing.assert_allclose(tc.backend.numpy(g), 0.0, atol=1e-4)


def test_bell_block():
    c = tc.Circuit(4)
    c = tc.templates.blocks.Bell_pair_block(c)
    for _ in range(10):
        s = c.perfect_sampling()[0]
        assert s[0] != s[1]
        assert s[2] != s[3]


def test_qft_block() -> None:
    n_qubits = 4
    c = tc.Circuit(n_qubits)
    c = tc.templates.blocks.qft(c, *range(n_qubits))
    mat = c.quoperator().eval().reshape(2 ** (n_qubits), -1)
    N = 2**n_qubits
    ref = np.exp(
        1j * 2 * np.pi * np.arange(N).reshape(-1, 1) * np.arange(N).reshape(1, -1) / N
    ) / np.sqrt(N)
    np.testing.assert_allclose(mat, ref, atol=1e-7)

    c = tc.Circuit(n_qubits)
    c = tc.templates.blocks.qft(c, *range(n_qubits), inverse=True)
    mat = c.quoperator().eval().reshape(2 ** (n_qubits), -1)
    np.testing.assert_allclose(mat, ref.T.conj(), atol=1e-7)


@pytest.mark.parametrize("flip_first, expected", [(False, 10.0), (True, 0.0)])
def test_line1d_periodic_ising_energy(npb, flip_first, expected):
    graph = tc.templates.graphs.Line1D(4, edge_weight=[1.0, 2.0, 3.0, 4.0])
    circuit = tc.Circuit(4)
    if flip_first:
        circuit.x(0)
    energy = tc.templates.measurements.spin_glass_measurements(circuit, graph)
    np.testing.assert_allclose(energy, expected, atol=1e-6)


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("flip_first", [False, True])
@pytest.mark.parametrize(
    "edge_weight, open_weight, periodic_weight",
    [
        (None, 1.0, 2.0),
        (2.5, 2.5, 5.0),
        (np.array(2.5), 2.5, 5.0),
        ([1.0, 2.0], 1.0, 3.0),
    ],
)
def test_line1d_two_sites(
    npb, pbc, flip_first, edge_weight, open_weight, periodic_weight
):
    original = None if edge_weight is None else np.copy(edge_weight)
    graph = tc.templates.graphs.Line1D(2, edge_weight=edge_weight, pbc=pbc)
    if original is not None:
        np.testing.assert_allclose(edge_weight, original)
    expected = periodic_weight if pbc else open_weight
    assert set(graph.edges) == {(0, 1)}
    np.testing.assert_allclose(graph[0][1]["weight"], expected)
    circuit = tc.Circuit(2)
    if flip_first:
        circuit.x(0)
    energy = tc.templates.measurements.spin_glass_measurements(circuit, graph)
    np.testing.assert_allclose(energy, -expected if flip_first else expected, atol=1e-6)


@pytest.mark.parametrize("n", [-1, 0, 1])
@pytest.mark.parametrize("pbc", [True, np.bool_(True)])
def test_line1d_rejects_too_few_sites(n, pbc):
    with pytest.raises(ValueError, match="at least two sites"):
        tc.templates.graphs.Line1D(n, pbc=pbc)


@pytest.mark.parametrize("pbc,node_weight", [(False, None), (np.bool_(False), [0.5])])
def test_line1d_open_single_site(npb, pbc, node_weight):
    graph = tc.templates.graphs.Line1D(1, node_weight=node_weight, pbc=pbc)
    assert list(graph.nodes) == [0]
    assert graph.number_of_edges() == 0
    expected = 0.0 if node_weight is None else 0.5
    np.testing.assert_allclose(graph.nodes[0]["weight"], expected)
    circuit = tc.Circuit(1)
    circuit.x(0)
    energy = tc.templates.measurements.spin_glass_measurements(circuit, graph)
    np.testing.assert_allclose(energy, -expected, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("container", [list, tuple, None])
def test_line1d_tensor_weight_gradients(backend, pbc, container):
    def energy(weights):
        nodes = weights[:3]
        edges = weights[3:]
        if container is not None:
            nodes = container(nodes[i] for i in range(3))
            edges = container(edges[i] for i in range(3))
        graph = tc.templates.graphs.Line1D(
            3, node_weight=nodes, edge_weight=edges, pbc=pbc
        )
        circuit = tc.Circuit(3)
        circuit.x(0)
        return tc.backend.real(
            tc.templates.measurements.spin_glass_measurements(circuit, graph)
        )

    weights = tc.backend.convert_to_tensor(np.arange(1, 7, dtype=np.float32))
    expected_grad = np.array([-1, 1, 1, -1, 1, -1 if pbc else 0])
    value, gradient = tc.backend.jit(tc.backend.value_and_grad(energy))(weights)
    np.testing.assert_allclose(
        tc.backend.numpy(value), np.dot(np.arange(1, 7), expected_grad), atol=1e-6
    )
    np.testing.assert_allclose(tc.backend.numpy(gradient), expected_grad, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
@pytest.mark.parametrize("pbc", [False, True])
def test_line1d_scalar_tensor_weight_gradients(backend, pbc):
    def energy(weights):
        graph = tc.templates.graphs.Line1D(
            3, node_weight=weights[0], edge_weight=weights[1], pbc=pbc
        )
        return tc.backend.real(
            tc.templates.measurements.spin_glass_measurements(tc.Circuit(3), graph)
        )

    weights = tc.backend.convert_to_tensor(np.array([1.5, 2.0], dtype=np.float32))
    value, gradient = tc.backend.jit(tc.backend.value_and_grad(energy))(weights)
    np.testing.assert_allclose(tc.backend.numpy(value), 10.5 if pbc else 8.5, atol=1e-6)
    np.testing.assert_allclose(
        tc.backend.numpy(gradient), [3, 3 if pbc else 2], atol=1e-6
    )


@pytest.mark.parametrize(
    "container,pbc",
    [
        (list, False),
        (tuple, True),
        (np.asarray, np.bool_(True)),
        (np.asarray, np.bool_(False)),
    ],
)
def test_line1d_weight_sequences(npb, container, pbc):
    graph = tc.templates.graphs.Line1D(
        3,
        node_weight=container([-0.25, 0.5, 1.25, 99.0]),
        edge_weight=container([1.0, 2.0, 3.0]),
        pbc=pbc,
    )
    edges = [(0, 1), (1, 2), (2, 0)] if pbc else [(0, 1), (1, 2)]
    assert graph.number_of_edges() == len(edges)
    np.testing.assert_allclose(
        [graph.nodes[q]["weight"] for q in range(3)], [-0.25, 0.5, 1.25]
    )
    np.testing.assert_allclose(
        [graph[u][v]["weight"] for u, v in edges], [1, 2, 3] if pbc else [1, 2]
    )
    energy = tc.templates.measurements.spin_glass_measurements(tc.Circuit(3), graph)
    assert np.ndim(energy) == 0
    np.testing.assert_allclose(energy, 7.5 if pbc else 4.5, atol=1e-6)


@pytest.mark.parametrize(
    "container,pbc,weights,closing_weight",
    [
        (list, True, [1.0, 2.0, 3.0], 3.0),
        (tuple, True, [1.0, 2.0, 3.0, 4.0, 99.0], 4.0),
        (np.asarray, False, [1.0, 2.0, 3.0], 3.0),
    ],
)
def test_line1d_weight_lengths(npb, container, pbc, weights, closing_weight):
    graph = tc.templates.graphs.Line1D(4, edge_weight=container(weights), pbc=pbc)
    np.testing.assert_allclose(
        [graph[q][q + 1]["weight"] for q in range(3)], [1.0, 2.0, 3.0]
    )
    if pbc:
        np.testing.assert_allclose(graph[3][0]["weight"], closing_weight)
    else:
        assert not graph.has_edge(3, 0)
    energy = tc.templates.measurements.spin_glass_measurements(tc.Circuit(4), graph)
    np.testing.assert_allclose(energy, 6.0 + closing_weight if pbc else 6.0, atol=1e-6)


@pytest.mark.parametrize("scalar", [float, np.float64, np.asarray])
@pytest.mark.parametrize("pbc", [False, True])
def test_line1d_scalar_weights(npb, scalar, pbc):
    graph = tc.templates.graphs.Line1D(
        3, node_weight=scalar(1.5), edge_weight=scalar(2.0), pbc=pbc
    )
    np.testing.assert_allclose([graph.nodes[q]["weight"] for q in range(3)], [1.5] * 3)
    np.testing.assert_allclose(
        [data["weight"] for _, _, data in graph.edges(data=True)],
        [2.0] * (3 if pbc else 2),
    )
    energy = tc.templates.measurements.spin_glass_measurements(tc.Circuit(3), graph)
    assert np.ndim(energy) == 0
    np.testing.assert_allclose(energy, 10.5 if pbc else 8.5, atol=1e-6)


def test_grid_coord():
    cd = tc.templates.graphs.Grid2DCoord(3, 2)
    assert cd.all_cols() == [(0, 3), (1, 4), (2, 5)]
    assert cd.all_rows() == [(0, 1), (1, 2), (3, 4), (4, 5)]


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_qaoa_template(backend):
    cd = tc.templates.graphs.Grid2DCoord(3, 2)
    g = cd.lattice_graph(pbc=False)
    for e1, e2 in g.edges:
        g[e1][e2]["weight"] = np.random.uniform()

    def forward(paramzz, paramx):
        c = tc.Circuit(6)
        for i in range(6):
            c.H(i)
        c = tc.templates.blocks.QAOA_block(c, g, paramzz, paramx)
        return tc.templates.measurements.spin_glass_measurements(c, g)

    fvag = tc.backend.jit(tc.backend.value_and_grad(forward, argnums=(0, 1)))
    paramzz = tc.backend.real(tc.backend.ones([1]))
    paramx = tc.backend.real(tc.backend.ones([1]))
    _, gr = fvag(paramzz, paramx)
    np.testing.assert_allclose(gr[1].shape, [1])
    paramzz = tc.backend.real(tc.backend.ones([7]))
    paramx = tc.backend.real(tc.backend.ones([1]))
    _, gr = fvag(paramzz, paramx)
    np.testing.assert_allclose(gr[0].shape, [7])
    paramzz = tc.backend.real(tc.backend.ones([1]))
    paramx = tc.backend.real(tc.backend.ones([6]))
    _, gr = fvag(paramzz, paramx)
    np.testing.assert_allclose(gr[0].shape, [1])
    np.testing.assert_allclose(gr[1].shape, [6])


def test_heisenberg_measurements_noncontiguous_graph():
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from([0, 2, 4])
    for n in g.nodes:
        g.nodes[n]["weight"] = 1.0
    g.add_edge(0, 2, weight=1.0)
    g.add_edge(2, 4, weight=1.0)

    c = tc.Circuit(5)
    c.X(0)
    c.X(2)
    c.X(4)

    energy = tc.templates.measurements.heisenberg_measurements(
        c, g, hzz=0.0, hxx=0.0, hyy=0.0, hz=1.0
    )
    np.testing.assert_allclose(tc.backend.numpy(energy), -3.0, atol=1e-5)


def test_state_wrapper():
    Bell_pair_block_state = tc.templates.blocks.state_centric(
        tc.templates.blocks.Bell_pair_block
    )
    s = Bell_pair_block_state(np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(
        s, np.array([0.0, 0.70710677 + 0.0j, -0.70710677 + 0.0j, 0]), atol=1e-5
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_amplitude_encoding(backend):
    batched_amplitude_encoding = tc.backend.vmap(
        tc.templates.dataset.amplitude_encoding, vectorized_argnums=0
    )
    figs = np.stack([np.eye(2), np.ones([2, 2])])
    figs = tc.array_to_tensor(figs)
    states = batched_amplitude_encoding(figs, 3)
    # note that you cannot use nqubits=3 here for jax backend
    # see this issue: https://github.com/google/jax/issues/7465
    np.testing.assert_allclose(states[1], np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]))
    states = batched_amplitude_encoding(
        figs, 2, tc.array_to_tensor(np.array([0, 3, 1, 2]), dtype="int32")
    )
    np.testing.assert_allclose(states[0], 1 / np.sqrt(2) * np.array([1, 1, 0, 0]))


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_mpo_measurement(backend):
    def f(theta):
        mpo = tc.quantum.QuOperator.from_local_tensor(
            tc.array_to_tensor(tc.gates._x_matrix), [2, 2, 2], [0]
        )
        c = tc.Circuit(3)
        c.ry(0, theta=theta)
        c.H(1)
        c.H(2)
        e = tc.templates.measurements.mpo_expectation(c, mpo)
        return e

    v, g = tc.backend.jit(tc.backend.value_and_grad(f))(tc.backend.ones([]))

    np.testing.assert_allclose(v, 0.84147, atol=1e-4)
    np.testing.assert_allclose(g, 0.54032, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_operator_measurement(backend):
    mpo = tc.quantum.QuOperator.from_local_tensor(
        tc.array_to_tensor(tc.gates._x_matrix), [2, 2], [0]
    )
    dense = tc.array_to_tensor(np.kron(tc.gates._x_matrix, np.eye(2)))
    sparse = tc.quantum.PauliString2COO([1, 0])

    for h in [dense, sparse, mpo]:

        def f(theta):
            c = tc.Circuit(2)
            c.ry(0, theta=theta)
            c.H(1)
            e = tc.templates.measurements.operator_expectation(c, h)
            return e

        v, g = tc.backend.jit(tc.backend.value_and_grad(f))(tc.backend.ones([]))

        np.testing.assert_allclose(v, 0.84147, atol=1e-4)
        np.testing.assert_allclose(g, 0.54032, atol=1e-4)


@pytest.fixture
def symmetric_matrix():
    matrix = np.array([[-5.0, -2.0], [-2.0, 6.0]])
    nsym_matrix = np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0], [8.0, 7.0, 6.0]]
    )
    return matrix, nsym_matrix


def test_QUBO_to_Ising(symmetric_matrix):
    matrix1, matrix2 = symmetric_matrix
    pauli_terms, weights, offset = tc.templates.conversions.QUBO_to_Ising(matrix1)
    n = matrix1.shape[0]
    expected_num_terms = n + n * (n - 1) // 2
    assert len(pauli_terms) == expected_num_terms
    assert len(weights) == expected_num_terms
    assert isinstance(pauli_terms, list)
    assert isinstance(weights, np.ndarray)
    assert isinstance(offset, float)
    assert pauli_terms == [
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    assert all(weights == np.array([3.5, -2.0, -1.0]))
    assert offset == -0.5

    with pytest.raises(ValueError):
        tc.templates.conversions.QUBO_to_Ising(matrix2)
