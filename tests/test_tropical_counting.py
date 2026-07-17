import numpy as np
import tensorcircuit as tc
from applications.tropical_algebra import (
    _counting_tensordot,
    split_energy_count,
)


def _be():
    return tc.backend


def _ref_counting_tensordot(anp, bnp, axes, eps=1e-9):
    # brute-force (max, degeneracy) tensordot on energy-only inputs (count starts at 1)
    if isinstance(axes, int):
        a_axes = list(range(anp.ndim - axes, anp.ndim))
        b_axes = list(range(0, axes))
    else:
        a_axes, b_axes = list(axes[0]), list(axes[1])
    a_free = [i for i in range(anp.ndim) if i not in a_axes]
    b_free = [i for i in range(bnp.ndim) if i not in b_axes]
    at = np.transpose(anp, a_free + a_axes)
    bt = np.transpose(bnp, b_axes + b_free)
    a_fs = [anp.shape[i] for i in a_free]
    b_fs = [bnp.shape[i] for i in b_free]
    k = int(np.prod([anp.shape[i] for i in a_axes]) or 1)
    A = int(np.prod(a_fs) or 1)
    b_n = int(np.prod(b_fs) or 1)
    a2 = at.reshape(A, k)
    b2 = bt.reshape(k, b_n)
    out_e = np.full((A, b_n), -np.inf)
    out_n = np.zeros((A, b_n))
    for i in range(A):
        for j in range(b_n):
            s = a2[i, :] + b2[:, j]
            mx = np.max(s)
            out_e[i, j] = mx
            out_n[i, j] = int(np.sum(np.abs(s - mx) < eps))  # degeneracy = #max-tied
    return out_e.reshape(a_fs + b_fs), out_n.reshape(a_fs + b_fs)


def test_counting_tensordot_matrix():
    be = _be()
    rng = np.random.default_rng(10)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    # stacked inputs: [x, n], counts init to 1
    a = be.cast(
        be.convert_to_tensor(np.stack([anp, np.ones_like(anp)], axis=-1)), "float64"
    )
    b = be.cast(
        be.convert_to_tensor(np.stack([bnp, np.ones_like(bnp)], axis=-1)), "float64"
    )
    got = np.array(_counting_tensordot(be, a, b, axes=1))
    ref_e, ref_n = _ref_counting_tensordot(anp, bnp, 1)
    np.testing.assert_allclose(got[..., 0], ref_e)
    np.testing.assert_allclose(got[..., 1], ref_n)


def test_split_energy_count():
    be = _be()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    stacked = be.cast(
        be.convert_to_tensor(np.stack([arr, arr * 2], axis=-1)), "float64"
    )
    e, n = split_energy_count(stacked)
    np.testing.assert_allclose(np.array(e), arr)
    np.testing.assert_allclose(np.array(n), arr * 2)


# --- Task B2: counting einsum (hyperedge / copy-node aware) ---

from applications.tropical_algebra import (  # noqa: E402
    _counting_einsum,
)


def _ref_counting_einsum(eq, a, b, eps=1e-9):
    """Brute-force (energy, degeneracy) 2-operand einsum on energy-only inputs."""
    import itertools

    lhs, rhs = eq.split("->")
    ia, ib = lhs.split(",")
    sizes = {}
    for s, t in zip([ia, ib], [a, b]):
        for c, dim in zip(s, t.shape):
            sizes[c] = dim
    out_e = np.full([sizes[c] for c in rhs], -np.inf)
    out_n = np.zeros([sizes[c] for c in rhs])
    allc = list(dict.fromkeys(list(ia) + list(ib)))
    for combo in itertools.product(*[range(sizes[c]) for c in allc]):
        env = dict(zip(allc, combo))
        e = a[tuple(env[c] for c in ia)] + b[tuple(env[c] for c in ib)]
        oidx = tuple(env[c] for c in rhs)
        if e > out_e[oidx] + eps:
            out_e[oidx] = e
            out_n[oidx] = 1
        elif abs(e - out_e[oidx]) < eps:
            out_n[oidx] += 1
    return out_e, out_n


def test_counting_einsum_pair():
    be = _be()
    rng = np.random.default_rng(11)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(np.stack([anp, np.ones_like(anp)], -1)), "float64")
    b = be.cast(be.convert_to_tensor(np.stack([bnp, np.ones_like(bnp)], -1)), "float64")
    got = np.array(_counting_einsum(be, "ab,bc->ac", a, b))
    ref_e, ref_n = _ref_counting_einsum("ab,bc->ac", anp, bnp)
    np.testing.assert_allclose(got[..., 0], ref_e)
    np.testing.assert_allclose(got[..., 1], ref_n)


def test_counting_einsum_hyperedge():
    # shared index 'a' (batch) — copy-node pair shape
    be = _be()
    rng = np.random.default_rng(12)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(3, 5))
    a = be.cast(be.convert_to_tensor(np.stack([anp, np.ones_like(anp)], -1)), "float64")
    b = be.cast(be.convert_to_tensor(np.stack([bnp, np.ones_like(bnp)], -1)), "float64")
    got = np.array(_counting_einsum(be, "ab,ac->abc", a, b))
    ref_e, ref_n = _ref_counting_einsum("ab,ac->abc", anp, bnp)
    np.testing.assert_allclose(got[..., 0], ref_e)
    np.testing.assert_allclose(got[..., 1], ref_n)


def test_counting_einsum_tie_degeneracy():
    # Symmetric inputs over the contracted axis -> a clean tie (degeneracy > 1).
    be = _be()
    # a: shape (1, 2) with equal values on the contracted axis; b: shape (2, 1).
    anp = np.array([[1.0, 1.0]])  # both k=0 and k=1 contribute the same energy
    bnp = np.array([[2.0], [2.0]])
    a = be.cast(be.convert_to_tensor(np.stack([anp, np.ones_like(anp)], -1)), "float64")
    b = be.cast(be.convert_to_tensor(np.stack([bnp, np.ones_like(bnp)], -1)), "float64")
    got = np.array(_counting_einsum(be, "ab,bc->ac", a, b))
    ref_e, ref_n = _ref_counting_einsum("ab,bc->ac", anp, bnp)
    np.testing.assert_allclose(got[..., 0], ref_e)
    np.testing.assert_allclose(got[..., 1], ref_n)
    # explicit assertion: energy 3.0 achieved by both k values -> degeneracy 2
    assert ref_n[0, 0] == 2
    np.testing.assert_allclose(got[0, 0, 0], 3.0)
    np.testing.assert_allclose(got[0, 0, 1], 2.0)


def test_counting_einsum_single_operand_diagonal():
    # eq "aa->a": diagonal, then no contraction -> energy=diag(e), count=diag(n)
    import tensorcircuit.backends.numpy_backend as nb

    be = nb.NumpyBackend()
    e = np.array([[1.0, 5.0], [3.0, 2.0]])
    n = np.array([[1.0, 2.0], [1.0, 1.0]])
    a = np.stack([e, n], axis=-1)
    out = _counting_einsum(be, "aa->a", a)
    np.testing.assert_allclose(out[..., 0], np.diag(e))
    np.testing.assert_allclose(out[..., 1], np.diag(n))


def test_counting_einsum_single_operand_reduce():
    # eq "ab->a": max over b of energy, tie-sum count
    import tensorcircuit.backends.numpy_backend as nb

    be = nb.NumpyBackend()
    e = np.array([[1.0, 5.0], [3.0, 3.0]])
    n = np.array([[1.0, 2.0], [1.0, 4.0]])
    a = np.stack([e, n], axis=-1)
    out = _counting_einsum(be, "ab->a", a)
    np.testing.assert_allclose(out[..., 0], [5.0, 3.0])
    np.testing.assert_allclose(out[..., 1], [2.0, 5.0])  # ties: b=0,1 both 3 -> 1+4


def test_counting_einsum_two_operand_intra_repeat():
    # eq "aa,b->ab": operand a has repeated index
    import tensorcircuit.backends.numpy_backend as nb

    be = nb.NumpyBackend()
    e_a = np.array([[2.0, 1.0], [1.0, 2.0]])
    n_a = np.ones((2, 2))
    a = np.stack([e_a, n_a], axis=-1)
    e_b = np.array([0.0, 0.0])
    n_b = np.array([1.0, 1.0])
    b = np.stack([e_b, n_b], axis=-1)
    out = _counting_einsum(be, "aa,b->ab", a, b)
    # diagonal of a is [2,2]; outer with b=[0,0] -> [[2,2],[2,2]], counts follow
    np.testing.assert_allclose(out[..., 0], [[2, 2], [2, 2]])
    # count stream: diag(n_a)=[1,1] outer n_b=[1,1] -> [[1,1],[1,1]]
    np.testing.assert_allclose(out[..., 1], [[1, 1], [1, 1]])


def test_counting_einsum_multi_axis():
    # Multi-axis contraction "ab,ac->c": shared index a (contracted) plus a free
    # index b of operand a (also summed out) -> two axes reduced. Compared to a
    # brute-force (energy, degeneracy) reference.
    be = _be()
    rng = np.random.default_rng(14)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(3, 5))
    a = be.cast(be.convert_to_tensor(np.stack([anp, np.ones_like(anp)], -1)), "float64")
    b = be.cast(be.convert_to_tensor(np.stack([bnp, np.ones_like(bnp)], -1)), "float64")
    got = np.array(_counting_einsum(be, "ab,ac->c", a, b))
    ref_e, ref_n = _ref_counting_einsum("ab,ac->c", anp, bnp)
    np.testing.assert_allclose(got[..., 0], ref_e)
    np.testing.assert_allclose(got[..., 1], ref_n)


# --- Task 8: degeneracy side-channel + standard-contraction clear discipline ---


def test_degeneracy_none_after_standard_contraction():
    """A counting contraction stashes count=deg into the aux side-channel; a
    subsequent STANDARD contraction must clear it (unconditional clear in
    ``cons._algebraic_base_contraction``), so ``degeneracy()`` returns None
    rather than a stale count from the earlier counting contraction.
    """
    import opt_einsum
    import tensornetwork as tn
    import tensorcircuit.cons as cons
    import applications.tropical_algebra as tr
    from tests._tropical_test_utils import (
        build_ising_tn,
        brute_force_energy_and_degeneracy,
    )

    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # 5-ring
    J = [1.0, 1.0, 1.0, 1.0, 1.0]  # ferromagnetic
    h = [0.0, 0.0, 0.0, 0.0, 0.0]  # zero field -> Z2 spin-flip symmetry, g=2
    expected_e, expected_n = brute_force_energy_and_degeneracy(len(spins), edges, J, h)
    assert (
        expected_n == 2
    )  # guard: this instance MUST be degenerate or the test is moot
    nodes = build_ising_tn(spins, edges, J, h)

    with tr.counting_tropical():
        node = cons._algebraic_base_contraction(
            nodes,
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[],
            ignore_edge_order=True,
        )
        # counting decode stashed the count; degeneracy() reads it (valid here)
        np.testing.assert_allclose(np.array(tr.degeneracy()), expected_n)
        np.testing.assert_allclose(np.array(node.tensor), -expected_e, atol=1e-6)

    # exit -> StandardAlgebra restored. A standard contraction must clear aux
    # (the unconditional _stash_aux_outputs({}) at the top of every contraction),
    # so a stale count cannot leak out of the counting block.
    rng = np.random.default_rng(0)
    a = tn.Node(rng.standard_normal((2, 3)).astype(np.complex64))
    b = tn.Node(rng.standard_normal((3, 4)).astype(np.complex64))
    tn.connect(a[1], b[0])
    cons._algebraic_base_contraction(
        [a, b],
        algorithm=opt_einsum.paths.dynamic_programming,
        output_edge_order=[a[0], b[1]],
    )
    assert tr.degeneracy() is None  # standard contraction cleared aux


# --- Task 10: non-scalar counting canary (aux co-indexed with output_edge_order) ---


def test_nonscalar_counting_energy_and_degeneracy_per_output():
    """A counting contraction with ONE free (dangling) spin returns energy AND
    degeneracy PER output configuration, with the count aux co-indexed with the
    energy via ``output_edge_order``.

    Validates two things at once:
    (1) non-scalar counting works at all -- ``decode`` strips the trailing count
        axis so ``tn.Node`` wraps a rank-correct energy tensor (rank == #free);
    (2) the aux-reorder logic (Task 5's ``output_edge_order`` permutation applied
        to aux) aligns count with energy. If the perm were wrong, count would be
        permuted differently from energy and the per-output assertion would fail.

    Brute force: for each value v in {0,1} of the free spin (cfg=0 -> s=+1,
    cfg=1 -> s=-1), enumerate the other spins, find min E and its degeneracy.
    """
    import opt_einsum
    import applications.tropical_algebra as tr
    import tensorcircuit.cons as cons
    from tests._tropical_test_utils import (
        build_ring_with_free_spin,
        brute_nonscalar_counting,
    )

    nodes, free_edge, J, h = build_ring_with_free_spin()
    expected_e, expected_n = brute_nonscalar_counting(nodes, free_edge, J, h)

    with tr.counting_tropical():
        node = cons._algebraic_base_contraction(
            nodes,
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[free_edge],
        )
        E = np.asarray(node.tensor)
        N = np.asarray(tr.degeneracy())

    np.testing.assert_allclose(E, expected_e)
    np.testing.assert_allclose(N, expected_n)
    # guard: the two output configs must differ in BOTH energy and count,
    # otherwise a permuted aux could still pass the assertion by accident.
    assert E[0] != E[1] or N[0] != N[1]


# --- Task 10 follow-up: 2-free-spin perm-direction canary ---


def test_nonscalar_counting_two_free_spins_reversed_order():
    """A counting contraction with TWO free (dangling) spins where
    ``output_edge_order`` is INTENTIONALLY REVERSED relative to the dangling-edge
    order, so the aux-reorder permutation in ``cons._algebraic_base_contraction``
    (``perm = [dangling_edges.index(e) for e in order]``) is non-trivial
    (``(1, 0)`` swap, not the identity ``(0,)`` exercised by the one-free-spin
    canary).

    Validates that the SAME permutation applied to the energy tensor (via
    ``final_node.reorder_edges``) is also applied to the aux degeneracy tensor
    (via ``kbe.transpose(v, tuple(perm))``). If the aux perm were ELIDED, the
    energy ``E`` would come out in the requested ``[b, a]`` order but the count
    ``N`` would be left in the dangling ``[a, b]`` order, so
    ``assert_allclose(N, expected_n_ab.T)`` would FAIL (the brute-force result is
    asymmetric under transpose by parameter choice -- see the guard below).

    The energy ``E`` itself is also checked against the transposed brute force,
    pinning down which order the contraction actually produced.
    """
    import opt_einsum
    import applications.tropical_algebra as tr
    import tensorcircuit.cons as cons
    from tests._tropical_test_utils import (
        build_ring_with_two_free_spins,
        brute_nonscalar_counting_2d,
    )

    nodes, edge_a, edge_b, J, h = build_ring_with_two_free_spins()
    # brute force in the natural (a, b) order:
    expected_e_ab, expected_n_ab = brute_nonscalar_counting_2d(
        nodes, edge_a, edge_b, J, h
    )
    # guard: the brute-force result MUST be asymmetric under transpose, otherwise
    # a wrong-order aux could still pass ``assert_allclose(N, expected_n_ab.T)``
    # by accident. The default parameters give E=[[5,7],[5,-1]], N=[[1,2],[1,1]],
    # both asymmetric.
    assert not np.allclose(expected_e_ab, expected_e_ab.T), (
        "weak canary: brute-force energy is symmetric under transpose, so the "
        "perm direction is not load-bearing for E"
    )
    assert not np.allclose(expected_n_ab, expected_n_ab.T), (
        "weak canary: brute-force degeneracy is symmetric under transpose, so "
        "the perm direction is not load-bearing for N"
    )

    # contract with output_edge_order REVERSED -> [edge_b, edge_a]; under
    # sorted_edges the dangling order is [edge_a, edge_b], so the perm is (1, 0).
    with tr.counting_tropical():
        node = cons._algebraic_base_contraction(
            nodes,
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[edge_b, edge_a],
        )
        E = np.asarray(node.tensor)  # in [b, a] order (reorder_edges applied)
        N = np.asarray(tr.degeneracy())  # MUST also be in [b, a] order if the
        #                                # aux perm is applied correctly.
    # expected in [b, a] order = transpose of the (a, b) brute force:
    np.testing.assert_allclose(E, expected_e_ab.T)
    np.testing.assert_allclose(N, expected_n_ab.T)
