import numpy as np
import tensornetwork as tn
import tensorcircuit as tc
import tensorcircuit.cons as cons
from applications.tropical_algebra import _tropical_tensordot


def _be():
    return tc.backend


def _ref_tropical_tensordot(anp, bnp, axes):
    if isinstance(axes, int):
        a_axes = list(range(anp.ndim - axes, anp.ndim))
        b_axes = list(range(0, axes))
    else:
        a_axes, b_axes = list(axes[0]), list(axes[1])
    a_free = [i for i in range(anp.ndim) if i not in a_axes]
    b_free = [i for i in range(bnp.ndim) if i not in b_axes]
    a_t = np.transpose(anp, a_free + a_axes)
    b_t = np.transpose(bnp, b_axes + b_free)
    a_fs = [anp.shape[i] for i in a_free]
    b_fs = [bnp.shape[i] for i in b_free]
    a2 = a_t.reshape(-1, np.prod([anp.shape[i] for i in a_axes], dtype=int) or 1)
    b2 = b_t.reshape(np.prod([bnp.shape[i] for i in b_axes], dtype=int) or 1, -1)
    A, b_n = a2.shape[0], b2.shape[1]
    res = np.full((A, b_n), -np.inf)
    for i in range(A):
        for j in range(b_n):
            res[i, j] = np.max(a2[i, :] + b2[:, j])
    return res.reshape(tuple(a_fs) + tuple(b_fs))


def test_tropical_tensordot_matrix():
    be = _be()
    rng = np.random.default_rng(0)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_tensordot(be, a, b, axes=1))
    np.testing.assert_allclose(got, _ref_tropical_tensordot(anp, bnp, 1))


def test_tropical_tensordot_multi_axis():
    be = _be()
    rng = np.random.default_rng(1)
    anp, bnp = rng.normal(size=(2, 3, 4)), rng.normal(size=(3, 4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_tensordot(be, a, b, axes=([1, 2], [0, 1])))
    np.testing.assert_allclose(got, _ref_tropical_tensordot(anp, bnp, ([1, 2], [0, 1])))
    assert got.shape == (2, 5)


from applications.tropical_algebra import _tropical_einsum, MaxPlusAlgebra, tropical


def _ref_tropical_einsum(eq, a, b):
    import itertools

    lhs, rhs = eq.split("->")
    ia, ib = lhs.split(",")
    sizes = {}
    for s, t in zip([ia, ib], [a, b]):
        for c, dim in zip(s, t.shape):
            sizes[c] = dim
    out = np.full([sizes[c] for c in rhs], -np.inf)
    allc = list(dict.fromkeys(list(ia) + list(ib)))
    for combo in itertools.product(*[range(sizes[c]) for c in allc]):
        env = dict(zip(allc, combo))
        ia_idx = tuple(env[c] for c in ia)
        ib_idx = tuple(env[c] for c in ib)
        val = a[ia_idx] + b[ib_idx]
        oidx = tuple(env[c] for c in rhs)
        if val > out[oidx]:
            out[oidx] = val
    return out


def test_tropical_einsum_pair():
    be = _be()
    rng = np.random.default_rng(2)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "ab,bc->ac", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("ab,bc->ac", anp, bnp))


def test_tropical_einsum_hyperedge_shared_index():
    # 'a' shared (batch, kept in output) — this is the copy-node pair shape
    be = _be()
    rng = np.random.default_rng(3)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(3, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "ab,ac->abc", a, b))  # note: output order a,b,c
    ref = _ref_tropical_einsum("ab,ac->abc", anp, bnp)
    np.testing.assert_allclose(got, ref)


def test_maxplus_algebra_uses_tropical_ops():
    be = _be()
    rng = np.random.default_rng(4)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    alg = MaxPlusAlgebra()
    assert alg.name == "maxplus"
    np.testing.assert_allclose(
        np.array(alg.tensordot(be, a, b, 1)),
        _ref_tropical_tensordot(anp, bnp, 1),
    )
    np.testing.assert_allclose(
        np.array(alg.einsum(be, "ab,bc->ac", a, b)),
        _ref_tropical_einsum("ab,bc->ac", anp, bnp),
    )


def test_tropical_einsum_single_tensor_transpose_ok():
    # single tensor with NO repeated indices -> standard transpose is correct
    be = _be()
    rng = np.random.default_rng(5)
    anp = rng.normal(size=(3, 4))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    got = np.array(_tropical_einsum(be, "ab->ba", a))
    np.testing.assert_allclose(got, np.array(be.einsum("ab->ba", a)))


def test_tropical_einsum_single_tensor_trace():
    # "ii->" : tropical trace = max of the diagonal
    be = _be()
    rng = np.random.default_rng(6)
    square_a = rng.normal(size=(3, 3))
    a = be.cast(be.convert_to_tensor(square_a), "float64")
    got = float(np.array(_tropical_einsum(be, "ii->", a)))
    assert np.isclose(got, np.max(np.diag(square_a)))


def test_tropical_einsum_single_tensor_diagonal():
    # "ii->i" : tropical diagonal gather (repeated index kept, no reduction)
    be = _be()
    rng = np.random.default_rng(7)
    square_a = rng.normal(size=(4, 4))
    a = be.cast(be.convert_to_tensor(square_a), "float64")
    got = np.array(_tropical_einsum(be, "ii->i", a))
    np.testing.assert_allclose(got, np.diag(square_a))


def test_tropical_einsum_single_tensor_reduce():
    # "ab->a" : single-tensor axis reduction (no repeat) must be tropical max, not sum
    be = _be()
    rng = np.random.default_rng(8)
    anp = rng.normal(size=(3, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    got = np.array(_tropical_einsum(be, "ab->a", a))
    np.testing.assert_allclose(got, np.max(anp, axis=1))


def test_tropical_einsum_single_tensor_partial_trace():
    # "iij->j" : repeated index reduced (diagonal over i, then max)
    be = _be()
    rng = np.random.default_rng(9)
    anp = rng.normal(size=(4, 4, 3))  # axes: i, i, j
    a = be.cast(be.convert_to_tensor(anp), "float64")
    got = np.array(_tropical_einsum(be, "iij->j", a))
    ref = np.max(np.array([anp[i, i, :] for i in range(4)]), axis=0)
    np.testing.assert_allclose(got, ref)


def test_tropical_einsum_intra_operand_repeat_first():
    # "iij,jk->ik": first operand has a repeated index i (diagonal gather), then
    # ordinary pairwise tropical contraction over j. Spec §4.5 step 1.
    be = _be()
    rng = np.random.default_rng(20)
    anp, bnp = rng.normal(size=(3, 3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "iij,jk->ik", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("iij,jk->ik", anp, bnp))


def test_tropical_einsum_intra_operand_repeat_contracted():
    # "iij,jk->k": the repeated index i is neither in the output nor shared with
    # b -> diagonal gather of i, then tropical max over both i and j.
    be = _be()
    rng = np.random.default_rng(21)
    anp, bnp = rng.normal(size=(3, 3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "iij,jk->k", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("iij,jk->k", anp, bnp))


def test_tropical_einsum_both_operands_repeat():
    # "iij,jj->ij": both operands have intra-operand repeats; no contraction axis.
    be = _be()
    rng = np.random.default_rng(22)
    anp, bnp = rng.normal(size=(3, 3, 4)), rng.normal(size=(4, 4))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "iij,jj->ij", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("iij,jj->ij", anp, bnp))


def test_tropical_context_changes_result():
    # max-plus "ab,b->a" = max_b (A[a,b] + B[b]) differs from standard sum-product
    be = _be()
    anp = np.array([[1.0, 5.0], [3.0, 2.0]])
    bnp = np.array([10.0, 0.0])
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    na, nb = tn.Node(a), tn.Node(b)
    tn.connect(na[1], nb[0])
    expected = np.array([max(1 + 10, 5 + 0), max(3 + 10, 2 + 0)])  # [11, 13]
    with tropical():
        got = np.array(cons.contractor([na, nb], output_edge_order=[na[0]]).tensor)
    np.testing.assert_allclose(got, expected)


def test_preprocessing_does_not_corrupt_tropical():
    # Regression: default cons.contractor (greedy + preprocessing=True) calls
    # _merge_single_gates for hyperedge-free >=5-node networks, merging single-gate
    # nodes via STANDARD tn.contract_parallel BEFORE the algebraic path runs. For a
    # tropical TN of regular tn.Nodes (no CopyNode -> no hyperedge -> preprocessing
    # not skipped), that merge would corrupt the max-plus result with sum-product.
    # The in-source _merge_single_gates guard skips the merge under a non-standard algebra.
    import itertools

    be = _be()
    rng = np.random.default_rng(0)
    n0 = rng.normal(size=(2,))
    n1 = rng.normal(size=(2, 2))
    n2 = rng.normal(size=(2, 2))
    n3 = rng.normal(size=(2, 2))
    n4 = rng.normal(size=(2,))
    na = tn.Node(be.cast(be.convert_to_tensor(n0), "float64"))
    nb = tn.Node(be.cast(be.convert_to_tensor(n1), "float64"))
    nc = tn.Node(be.cast(be.convert_to_tensor(n2), "float64"))
    nd = tn.Node(be.cast(be.convert_to_tensor(n3), "float64"))
    ne = tn.Node(be.cast(be.convert_to_tensor(n4), "float64"))
    tn.connect(na[0], nb[0])
    tn.connect(nb[1], nc[0])
    tn.connect(nc[1], nd[0])
    tn.connect(nd[1], ne[0])
    nodes = [na, nb, nc, nd, ne]
    assert len(nodes) >= 5

    ref = -np.inf
    for s0, s1, s2, s3 in itertools.product(range(2), repeat=4):
        val = n0[s0] + n1[s0, s1] + n2[s1, s2] + n3[s2, s3] + n4[s3]
        ref = max(ref, val)

    with tropical():
        got = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_tropical_public_api_surface():
    import applications.tropical_algebra as ex

    for name in [
        "MaxPlusAlgebra",
        "MaxPlusTrackingAlgebra",
        "CountingTropicalAlgebra",
        "tropical",
        "counting_tropical",
        "recover_configuration",
        "split_energy_count",
    ]:
        assert hasattr(ex, name), name
