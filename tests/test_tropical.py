"""Tropical (max-plus) and counting tropical algebra tests.

Covers: max-plus primitives, einsum/tensordot, configuration recovery,
counting (energy + degeneracy), Ising end-to-end, and the non-scalar
aux-reorder canaries.
"""

import itertools

import numpy as np
import pytest
import tensornetwork as tn
import opt_einsum

import tensorcircuit as tc
import tensorcircuit.cons as cons
import tensorcircuit.backends.numpy_backend as nb
import applications.tropical_algebra as tr
from applications.tropical_algebra import (
    MaxPlusAlgebra,
    MaxPlusTrackingAlgebra,
    CountingTropicalAlgebra,
    tropical,
    counting_tropical,
    recover_configuration,
    get_recorded_topology,
    degeneracy,
    split_energy_count,
    _tropical_tensordot,
    _tropical_einsum,
    _counting_tensordot,
    _counting_einsum,
    _td_backtrack_step,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities (was tests/_tropical_test_utils.py)
# ═══════════════════════════════════════════════════════════════════════════════


def build_ising_tn(spins, edges, j_vals, h):
    """Build the tropical (max-plus) Ising tensor network as tensornetwork nodes.

    Per spin: a ``tn.CopyNode`` of rank (degree+1), dimension 2. A CopyNode is a true
    hyperedge hub -- tensornetwork/cotengra identify all its legs as one shared index
    (the delta/copy constraint is structural, not stored as data). This is what makes
    the contraction route hyperedge-containing pairs through the tropical **einsum**
    branch and ordinary pairs through the tropical **tensordot** branch, covering both.
    Per edge (i,j): Te[si,sj] = J*si*sj -> [[J,-J],[-J,J]].
    Per spin: field Tv[s] = h*s -> [h,-h].
    Contracting over max-plus yields max_cfg(-E) = -E_ground.
    """
    be = tc.backend
    degree = dict.fromkeys(spins, 0)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    nodes = []
    copy_nodes = {}
    for i in spins:
        cn = tn.CopyNode(degree[i] + 1, 2)
        copy_nodes[i] = cn
        nodes.append(cn)
        tv = np.array([h[i], -h[i]], dtype=np.float64)
        tvn = tn.Node(be.cast(be.convert_to_tensor(tv), "float64"))
        tn.connect(cn[0], tvn[0])
        nodes.append(tvn)
    leg_idx = dict.fromkeys(spins, 1)
    for (i, j), jij in zip(edges, j_vals):
        te = np.array([[jij, -jij], [-jij, jij]], dtype=np.float64)
        ten = tn.Node(be.cast(be.convert_to_tensor(te), "float64"))
        tn.connect(ten[0], copy_nodes[i][leg_idx[i]])
        tn.connect(ten[1], copy_nodes[j][leg_idx[j]])
        leg_idx[i] += 1
        leg_idx[j] += 1
        nodes.append(ten)
    return nodes


def brute_force_energy(spins, edges, j_vals, h):
    """Brute-force min energy: E = -sum J s_i s_j - sum h s_i."""
    n = len(spins)
    best = np.inf
    for cfg in itertools.product([-1, 1], repeat=n):
        e = 0.0
        for (i, j), jij in zip(edges, j_vals):
            e -= jij * cfg[i] * cfg[j]
        for i, hi in zip(spins, h):
            e -= hi * cfg[i]
        best = min(best, e)
    return best


def brute_force_energy_and_degeneracy(n, edges, j_vals, h):
    """Brute-force ground energy + degeneracy over spin configs.

    cfg in {0,1}^n; spin s = 1 - 2*cfg (cfg=0 -> s=+1).
    E = -sum J s_i s_j - sum h s_i. Returns (best_e, degeneracy).
    """
    best_e = None
    deg = 0
    for cfg in itertools.product([0, 1], repeat=n):
        e = 0.0
        for (i, j), jij in zip(edges, j_vals):
            si, sj = 1 - 2 * cfg[i], 1 - 2 * cfg[j]
            e -= jij * si * sj
        for i, hi in zip(range(n), h):
            e -= hi * (1 - 2 * cfg[i])
        if best_e is None or e < best_e - 1e-9:
            best_e, deg = e, 1
        elif abs(e - best_e) < 1e-9:
            deg += 1
    return best_e, deg


# --- Non-scalar (free-spin) helpers ---

_RING_SPINS = (0, 1, 2, 3)
_RING_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_RING_J = (-2, 1, 1, -2)
_RING_H = (2, 0, 2, -1)
_FREE_SPIN = 0
_FREE_SPIN_A = 0
_FREE_SPIN_B = 1


def build_ring_with_free_spin(
    spins=_RING_SPINS,
    edges=_RING_EDGES,
    j_vals=_RING_J,
    h=_RING_H,
    free_spin=_FREE_SPIN,
):
    """Build a small ring Ising with ONE spin's CopyNode given an extra dangling
    leg (the free output index)."""
    be = tc.backend
    degree = dict.fromkeys(spins, 0)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    nodes = []
    copy_nodes = {}
    for i in spins:
        rank = degree[i] + 1
        if i == free_spin:
            rank += 1
        cn = tn.CopyNode(rank, 2)
        copy_nodes[i] = cn
        nodes.append(cn)
        tv = np.array([h[i], -h[i]], dtype=np.float64)
        tvn = tn.Node(be.cast(be.convert_to_tensor(tv), "float64"))
        tn.connect(cn[0], tvn[0])
        nodes.append(tvn)

    leg_idx = dict.fromkeys(spins, 1)
    for (i, j), jij in zip(edges, j_vals):
        te = np.array([[jij, -jij], [-jij, jij]], dtype=np.float64)
        ten = tn.Node(be.cast(be.convert_to_tensor(te), "float64"))
        tn.connect(ten[0], copy_nodes[i][leg_idx[i]])
        tn.connect(ten[1], copy_nodes[j][leg_idx[j]])
        leg_idx[i] += 1
        leg_idx[j] += 1
        nodes.append(ten)

    free_edge = copy_nodes[free_spin][degree[free_spin] + 1]
    return nodes, free_edge, j_vals, h


def build_ring_with_two_free_spins(
    spins=_RING_SPINS,
    edges=_RING_EDGES,
    j_vals=_RING_J,
    h=_RING_H,
    free_spin_a=_FREE_SPIN_A,
    free_spin_b=_FREE_SPIN_B,
):
    """Build a small ring Ising with TWO spins' CopyNodes given an extra dangling
    leg each (two free output indices)."""
    be = tc.backend
    free_spins = {free_spin_a, free_spin_b}
    degree = dict.fromkeys(spins, 0)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    nodes = []
    copy_nodes = {}
    for i in spins:
        rank = degree[i] + 1
        if i in free_spins:
            rank += 1
        cn = tn.CopyNode(rank, 2)
        copy_nodes[i] = cn
        nodes.append(cn)
        tv = np.array([h[i], -h[i]], dtype=np.float64)
        tvn = tn.Node(be.cast(be.convert_to_tensor(tv), "float64"))
        tn.connect(cn[0], tvn[0])
        nodes.append(tvn)

    leg_idx = dict.fromkeys(spins, 1)
    for (i, j), jij in zip(edges, j_vals):
        te = np.array([[jij, -jij], [-jij, jij]], dtype=np.float64)
        ten = tn.Node(be.cast(be.convert_to_tensor(te), "float64"))
        tn.connect(ten[0], copy_nodes[i][leg_idx[i]])
        tn.connect(ten[1], copy_nodes[j][leg_idx[j]])
        leg_idx[i] += 1
        leg_idx[j] += 1
        nodes.append(ten)

    edge_a = copy_nodes[free_spin_a][degree[free_spin_a] + 1]
    edge_b = copy_nodes[free_spin_b][degree[free_spin_b] + 1]
    return nodes, edge_a, edge_b, j_vals, h


def _ising_config_energy(cfg, edges, j_vals, spins, h):
    """Ising energy E = -sum J s_i s_j - sum h s_i for a config dict (spin idx -> 0/1)."""
    energy = 0.0
    for (i, j), jij in zip(edges, j_vals):
        energy -= jij * (1 - 2 * cfg[i]) * (1 - 2 * cfg[j])
    for i, hi in zip(spins, h):
        energy -= hi * (1 - 2 * cfg[i])
    return energy


def _min_energy_and_degeneracy(energies, eps=1e-9):
    """Return ``(min energy, degeneracy)`` -- degeneracy counts configs within eps of min."""
    best = None
    deg = 0
    for e in energies:
        if best is None or e < best - eps:
            best, deg = e, 1
        elif abs(e - best) < eps:
            deg += 1
    return best, deg


def brute_nonscalar_counting(
    nodes,
    free_edge,
    j_vals,
    h,
    spins=_RING_SPINS,
    edges=_RING_EDGES,
    free_spin=_FREE_SPIN,
):
    """Brute-force per-output (energy, degeneracy) for the non-scalar ring with
    one free spin."""
    del nodes, free_edge
    others = [s for s in spins if s != free_spin]

    expected_e = np.zeros(2, dtype=np.float64)
    expected_n = np.zeros(2, dtype=np.float64)
    for v_free in (0, 1):
        fixed = {free_spin: v_free}
        energies = [
            _ising_config_energy(
                {**fixed, **dict(zip(others, cfg_rest))}, edges, j_vals, spins, h
            )
            for cfg_rest in itertools.product([0, 1], repeat=len(others))
        ]
        best_e, deg = _min_energy_and_degeneracy(energies)
        expected_e[v_free] = -best_e
        expected_n[v_free] = deg
    return expected_e, expected_n


def brute_nonscalar_counting_2d(
    nodes,
    edge_a,
    edge_b,
    j_vals,
    h,
    spins=_RING_SPINS,
    edges=_RING_EDGES,
    free_spin_a=_FREE_SPIN_A,
    free_spin_b=_FREE_SPIN_B,
):
    """Brute-force per-output (energy, degeneracy) for the non-scalar ring with
    TWO free spins."""
    del nodes, edge_a, edge_b
    others = [s for s in spins if s not in (free_spin_a, free_spin_b)]

    expected_e = np.zeros((2, 2), dtype=np.float64)
    expected_n = np.zeros((2, 2), dtype=np.float64)
    for v_a in (0, 1):
        for v_b in (0, 1):
            fixed = {free_spin_a: v_a, free_spin_b: v_b}
            energies = [
                _ising_config_energy(
                    {**fixed, **dict(zip(others, cfg_rest))}, edges, j_vals, spins, h
                )
                for cfg_rest in itertools.product([0, 1], repeat=len(others))
            ]
            best_e, deg = _min_energy_and_degeneracy(energies)
            expected_e[v_a, v_b] = -best_e
            expected_n[v_a, v_b] = deg
    return expected_e, expected_n


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Core algebra: tensordot + einsum (was test_tropical_algebra.py)
# ═══════════════════════════════════════════════════════════════════════════════


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


def _ref_tropical_einsum(eq, a, b):
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


# --- tensordot tests ---


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


# --- einsum tests ---


def test_tropical_einsum_pair():
    be = _be()
    rng = np.random.default_rng(2)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "ab,bc->ac", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("ab,bc->ac", anp, bnp))


def test_tropical_einsum_hyperedge_shared_index():
    be = _be()
    rng = np.random.default_rng(3)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(3, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "ab,ac->abc", a, b))
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
    be = _be()
    rng = np.random.default_rng(5)
    anp = rng.normal(size=(3, 4))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    got = np.array(_tropical_einsum(be, "ab->ba", a))
    np.testing.assert_allclose(got, np.array(be.einsum("ab->ba", a)))


def test_tropical_einsum_single_tensor_trace():
    be = _be()
    rng = np.random.default_rng(6)
    square_a = rng.normal(size=(3, 3))
    a = be.cast(be.convert_to_tensor(square_a), "float64")
    got = float(np.array(_tropical_einsum(be, "ii->", a)))
    assert np.isclose(got, np.max(np.diag(square_a)))


def test_tropical_einsum_single_tensor_diagonal():
    be = _be()
    rng = np.random.default_rng(7)
    square_a = rng.normal(size=(4, 4))
    a = be.cast(be.convert_to_tensor(square_a), "float64")
    got = np.array(_tropical_einsum(be, "ii->i", a))
    np.testing.assert_allclose(got, np.diag(square_a))


def test_tropical_einsum_single_tensor_reduce():
    be = _be()
    rng = np.random.default_rng(8)
    anp = rng.normal(size=(3, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    got = np.array(_tropical_einsum(be, "ab->a", a))
    np.testing.assert_allclose(got, np.max(anp, axis=1))


def test_tropical_einsum_single_tensor_partial_trace():
    be = _be()
    rng = np.random.default_rng(9)
    anp = rng.normal(size=(4, 4, 3))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    got = np.array(_tropical_einsum(be, "iij->j", a))
    ref = np.max(np.array([anp[i, i, :] for i in range(4)]), axis=0)
    np.testing.assert_allclose(got, ref)


def test_tropical_einsum_intra_operand_repeat_first():
    be = _be()
    rng = np.random.default_rng(20)
    anp, bnp = rng.normal(size=(3, 3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "iij,jk->ik", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("iij,jk->ik", anp, bnp))


def test_tropical_einsum_intra_operand_repeat_contracted():
    be = _be()
    rng = np.random.default_rng(21)
    anp, bnp = rng.normal(size=(3, 3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "iij,jk->k", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("iij,jk->k", anp, bnp))


def test_tropical_einsum_both_operands_repeat():
    be = _be()
    rng = np.random.default_rng(22)
    anp, bnp = rng.normal(size=(3, 3, 4)), rng.normal(size=(4, 4))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(_tropical_einsum(be, "iij,jj->ij", a, b))
    np.testing.assert_allclose(got, _ref_tropical_einsum("iij,jj->ij", anp, bnp))


def test_tropical_context_changes_result():
    be = _be()
    anp = np.array([[1.0, 5.0], [3.0, 2.0]])
    bnp = np.array([10.0, 0.0])
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    na, nb = tn.Node(a), tn.Node(b)
    tn.connect(na[1], nb[0])
    expected = np.array([max(1 + 10, 5 + 0), max(3 + 10, 2 + 0)])
    with tropical():
        got = np.array(cons.contractor([na, nb], output_edge_order=[na[0]]).tensor)
    np.testing.assert_allclose(got, expected)


def test_preprocessing_does_not_corrupt_tropical():
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
    assert MaxPlusAlgebra is not None
    assert MaxPlusTrackingAlgebra is not None
    assert CountingTropicalAlgebra is not None
    for name in [
        "tropical",
        "counting_tropical",
        "recover_configuration",
        "split_energy_count",
    ]:
        assert hasattr(tr, name), name


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Configuration recovery (was test_tropical_config.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _brute_cfg(n, edges, j_vals, h):
    """Brute-force min energy over spin configs; returns (best_e, set_of_optimal_cfgs)."""
    best_e = None
    best_cfgs = set()
    for cfg in itertools.product([0, 1], repeat=n):
        e = 0.0
        for (i, j), jij in zip(edges, j_vals):
            e -= jij * (1 - 2 * cfg[i]) * (1 - 2 * cfg[j])
        for i, hi in zip(range(n), h):
            e -= hi * (1 - 2 * cfg[i])
        if best_e is None or e < best_e - 1e-9:
            best_e = e
            best_cfgs = {cfg}
        elif abs(e - best_e) < 1e-9:
            best_cfgs.add(cfg)
    return best_e, best_cfgs


def _energy_of(assignment, input_sets, raw_tensors):
    """Max-plus total for a symbol->value assignment = -E for that config."""
    total = 0.0
    for term, t in zip(input_sets, raw_tensors):
        idx = tuple(int(assignment[c]) for c in term)
        total += float(np.asarray(t)[idx])
    return total


def _contract_tropical_track(nodes):
    """Contract under tracking tropical; return (value, config)."""
    with tropical(track=True):
        val = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()
        cfg = recover_configuration()
    return val, cfg


def test_config_recovery_tiny_ising_unique():
    spins = [0, 1]
    edges = [(0, 1)]
    J = [0.7]
    h = [0.3, -0.2]
    best_e, best_cfgs = _brute_cfg(len(spins), edges, J, h)

    nodes = build_ising_tn(spins, edges, J, h)
    val, cfg = _contract_tropical_track(nodes)

    np.testing.assert_allclose(val, -best_e, atol=1e-6)

    _tree, input_sets, raw_tensors = get_recorded_topology()
    assert input_sets is not None and raw_tensors is not None
    all_labels = set().union(*[set(t) for t in input_sets])
    assert set(cfg.keys()) == all_labels

    recovered_total = _energy_of(cfg, input_sets, raw_tensors)
    np.testing.assert_allclose(recovered_total, val, atol=1e-6)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)

    spin_cfg = {}
    for term, _t in zip(input_sets, raw_tensors):
        if len(term) == 1:
            spin_cfg[len(spin_cfg)] = int(cfg[term[0]])
    assert tuple(spin_cfg[i] for i in range(len(spins))) in best_cfgs


def test_config_recovery_tiny_ising_degenerate():
    spins = [0, 1]
    edges = [(0, 1)]
    J = [1.0]
    h = [0.0, 0.0]
    best_e, best_cfgs = _brute_cfg(len(spins), edges, J, h)
    assert len(best_cfgs) == 2

    nodes = build_ising_tn(spins, edges, J, h)
    val, cfg = _contract_tropical_track(nodes)

    np.testing.assert_allclose(val, -best_e, atol=1e-6)
    _tree, input_sets, raw_tensors = get_recorded_topology()
    recovered_total = _energy_of(cfg, input_sets, raw_tensors)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)

    spin_cfg = {}
    for term, _t in zip(input_sets, raw_tensors):
        if len(term) == 1:
            spin_cfg[len(spin_cfg)] = int(cfg[term[0]])
    assert tuple(spin_cfg[i] for i in range(len(spins))) in best_cfgs


def test_config_recovery_three_ring():
    spins = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    J = [0.5, -0.8]
    h = [0.1, 0.2, -0.3]
    best_e, _best_cfgs = _brute_cfg(len(spins), edges, J, h)

    nodes = build_ising_tn(spins, edges, J, h)
    val, cfg = _contract_tropical_track(nodes)
    np.testing.assert_allclose(val, -best_e, atol=1e-6)

    _tree, input_sets, raw_tensors = get_recorded_topology()
    recovered_total = _energy_of(cfg, input_sets, raw_tensors)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)


def test_config_recovery_non_scalar_raises():
    be = tc.backend
    a_np = np.array([[1.0, 4.0, 2.0], [3.0, 1.0, 5.0], [0.0, 2.0, 1.0]])
    b_np = np.array([0.5, 1.0, -0.5])

    a_node = tn.Node(be.cast(be.convert_to_tensor(a_np), "float64"))
    b_node = tn.Node(be.cast(be.convert_to_tensor(b_np), "float64"))
    tn.connect(a_node[1], b_node[0])
    nodes = [a_node, b_node]

    with tropical(track=True):
        res = cons.contractor(nodes, output_edge_order=[a_node[0]])
        assert np.array(res.tensor).shape == (3,)
        with pytest.raises(NotImplementedError):
            recover_configuration()


class _StubTree:
    """Minimal stand-in for a cotengra ``ContractionTree``."""

    def __init__(self, inds, axes, perm):
        self._inds = inds
        self._axes = axes
        self._perm = perm

    def get_inds(self, node):
        return self._inds[node]

    def get_tensordot_axes(self, node):
        return self._axes[node]

    def get_tensordot_perm(self, node):
        return self._perm[node]


def test_td_backtrack_step_perm_inversion():
    l_node, r_node, p_node = "l", "r", "p"
    tree = _StubTree(
        inds={l_node: "ab", r_node: "bcd", p_node: "dac"},
        axes={p_node: ([1], [0])},
        perm={p_node: (2, 0, 1)},
    )
    argmax = np.zeros((2, 3, 2), dtype=np.int64)
    argmax[1, 2, 0] = 3
    rec = {"kind": "td", "argmax": argmax, "contract_dims": (4,)}
    p_pos = (0, 1, 2)
    assignment = {}
    l_pos, r_pos = _td_backtrack_step(
        tree, p_node, l_node, r_node, rec, p_pos, assignment
    )

    assert assignment["b"] == 3
    assert l_pos == (1, 3)
    assert r_pos == (3, 2, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Counting (was test_tropical_counting.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _ref_counting_tensordot(anp, bnp, axes, eps=1e-9):
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
            out_n[i, j] = int(np.sum(np.abs(s - mx) < eps))
    return out_e.reshape(a_fs + b_fs), out_n.reshape(a_fs + b_fs)


def _ref_counting_einsum(eq, a, b, eps=1e-9):
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


def test_counting_tensordot_matrix():
    be = _be()
    rng = np.random.default_rng(10)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
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
    be = _be()
    anp = np.array([[1.0, 1.0]])
    bnp = np.array([[2.0], [2.0]])
    a = be.cast(be.convert_to_tensor(np.stack([anp, np.ones_like(anp)], -1)), "float64")
    b = be.cast(be.convert_to_tensor(np.stack([bnp, np.ones_like(bnp)], -1)), "float64")
    got = np.array(_counting_einsum(be, "ab,bc->ac", a, b))
    ref_e, ref_n = _ref_counting_einsum("ab,bc->ac", anp, bnp)
    np.testing.assert_allclose(got[..., 0], ref_e)
    np.testing.assert_allclose(got[..., 1], ref_n)
    assert ref_n[0, 0] == 2
    np.testing.assert_allclose(got[0, 0, 0], 3.0)
    np.testing.assert_allclose(got[0, 0, 1], 2.0)


def test_counting_einsum_single_operand_diagonal():
    be = nb.NumpyBackend()
    e = np.array([[1.0, 5.0], [3.0, 2.0]])
    n = np.array([[1.0, 2.0], [1.0, 1.0]])
    a = np.stack([e, n], axis=-1)
    out = _counting_einsum(be, "aa->a", a)
    np.testing.assert_allclose(out[..., 0], np.diag(e))
    np.testing.assert_allclose(out[..., 1], np.diag(n))


def test_counting_einsum_single_operand_reduce():
    be = nb.NumpyBackend()
    e = np.array([[1.0, 5.0], [3.0, 3.0]])
    n = np.array([[1.0, 2.0], [1.0, 4.0]])
    a = np.stack([e, n], axis=-1)
    out = _counting_einsum(be, "ab->a", a)
    np.testing.assert_allclose(out[..., 0], [5.0, 3.0])
    np.testing.assert_allclose(out[..., 1], [2.0, 5.0])


def test_counting_einsum_two_operand_intra_repeat():
    be = nb.NumpyBackend()
    e_a = np.array([[2.0, 1.0], [1.0, 2.0]])
    n_a = np.ones((2, 2))
    a = np.stack([e_a, n_a], axis=-1)
    e_b = np.array([0.0, 0.0])
    n_b = np.array([1.0, 1.0])
    b = np.stack([e_b, n_b], axis=-1)
    out = _counting_einsum(be, "aa,b->ab", a, b)
    np.testing.assert_allclose(out[..., 0], [[2, 2], [2, 2]])
    np.testing.assert_allclose(out[..., 1], [[1, 1], [1, 1]])


def test_counting_einsum_multi_axis():
    be = _be()
    rng = np.random.default_rng(14)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(3, 5))
    a = be.cast(be.convert_to_tensor(np.stack([anp, np.ones_like(anp)], -1)), "float64")
    b = be.cast(be.convert_to_tensor(np.stack([bnp, np.ones_like(bnp)], -1)), "float64")
    got = np.array(_counting_einsum(be, "ab,ac->c", a, b))
    ref_e, ref_n = _ref_counting_einsum("ab,ac->c", anp, bnp)
    np.testing.assert_allclose(got[..., 0], ref_e)
    np.testing.assert_allclose(got[..., 1], ref_n)


def test_degeneracy_none_after_standard_contraction():
    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, 1.0, 1.0, 1.0, 1.0]
    h = [0.0, 0.0, 0.0, 0.0, 0.0]
    expected_e, expected_n = brute_force_energy_and_degeneracy(len(spins), edges, J, h)
    assert expected_n == 2
    nodes = build_ising_tn(spins, edges, J, h)

    with counting_tropical():
        node = cons._algebraic_base_contraction(
            nodes,
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[],
            ignore_edge_order=True,
        )
        np.testing.assert_allclose(np.array(degeneracy()), expected_n)
        np.testing.assert_allclose(np.array(node.tensor), -expected_e, atol=1e-6)

    rng = np.random.default_rng(0)
    a = tn.Node(rng.standard_normal((2, 3)).astype(np.complex64))
    b = tn.Node(rng.standard_normal((3, 4)).astype(np.complex64))
    tn.connect(a[1], b[0])
    cons._algebraic_base_contraction(
        [a, b],
        algorithm=opt_einsum.paths.dynamic_programming,
        output_edge_order=[a[0], b[1]],
    )
    assert degeneracy() is None


def test_nonscalar_counting_energy_and_degeneracy_per_output():
    nodes, free_edge, J, h = build_ring_with_free_spin()
    expected_e, expected_n = brute_nonscalar_counting(nodes, free_edge, J, h)

    with counting_tropical():
        node = cons._algebraic_base_contraction(
            nodes,
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[free_edge],
        )
        E = np.asarray(node.tensor)
        N = np.asarray(degeneracy())

    np.testing.assert_allclose(E, expected_e)
    np.testing.assert_allclose(N, expected_n)
    assert E[0] != E[1] or N[0] != N[1]


def test_nonscalar_counting_two_free_spins_reversed_order():
    nodes, edge_a, edge_b, J, h = build_ring_with_two_free_spins()
    expected_e_ab, expected_n_ab = brute_nonscalar_counting_2d(
        nodes, edge_a, edge_b, J, h
    )
    assert not np.allclose(
        expected_e_ab, expected_e_ab.T
    ), "weak canary: brute-force energy is symmetric under transpose"
    assert not np.allclose(
        expected_n_ab, expected_n_ab.T
    ), "weak canary: brute-force degeneracy is symmetric under transpose"

    with counting_tropical():
        node = cons._algebraic_base_contraction(
            nodes,
            algorithm=opt_einsum.paths.dynamic_programming,
            output_edge_order=[edge_b, edge_a],
        )
        E = np.asarray(node.tensor)
        N = np.asarray(degeneracy())

    np.testing.assert_allclose(E, expected_e_ab.T)
    np.testing.assert_allclose(N, expected_n_ab.T)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Example-level tests (was test_tropical_example.py)
# ═══════════════════════════════════════════════════════════════════════════════


def test_example_maxplus_tensordot_matches_brute():
    be = tc.backend
    rng = np.random.default_rng(0)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(MaxPlusAlgebra().tensordot(be, a, b, 1))
    ref = np.max(anp[:, :, None] + bnp[None, :, :], axis=1)
    np.testing.assert_allclose(got, ref)


def test_example_recover_configuration_on_tiny_ising():
    be = tc.backend
    n = tn.Node(be.cast(be.convert_to_tensor(np.array([0.0, 0.0])), "float64"))
    e = tn.Node(
        be.cast(be.convert_to_tensor(np.array([[1.0, -1.0], [-1.0, 1.0]])), "float64")
    )
    m = tn.Node(be.cast(be.convert_to_tensor(np.array([0.5, -0.5])), "float64"))
    tn.connect(n[0], e[0])
    tn.connect(e[1], m[0])
    nodes = [n, e, m]
    with tropical(track=True):
        float(cons.contractor(nodes, output_edge_order=[]).tensor)
        cfg = recover_configuration()
    assert isinstance(cfg, dict)
    assert len(cfg) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Ising end-to-end (was test_tropical_ising.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _ising_energy_of_cfg(cfg, edges, j_vals, h):
    """Ising energy of a {0,1}-config: E = -sum J s_i s_j - sum h s_i (s=1-2cfg)."""
    e = 0.0
    for (i, j), jij in zip(edges, j_vals):
        si, sj = 1 - 2 * cfg[i], 1 - 2 * cfg[j]
        e -= jij * si * sj
    for i, hi in zip(range(len(h)), h):
        e -= hi * (1 - 2 * cfg[i])
    return e


def _maxplus_total_of_assignment(cfg_sym, input_sets, raw_tensors):
    """Max-plus total of a symbol->value assignment = sum of tensor entries =
    -E for that configuration."""
    total = 0.0
    for term, t in zip(input_sets, raw_tensors):
        idx = tuple(int(cfg_sym[c]) for c in term)
        total += float(np.asarray(t)[idx])
    return total


def test_ising_ring_ground_state_matches_bruteforce():
    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, -1.0, 1.0, 1.0, -1.0]
    h = [0.5, -0.3, 0.2, 0.0, 0.4]
    nodes = build_ising_tn(spins, edges, J, h)

    e_ground = brute_force_energy(spins, edges, J, h)

    with tropical():
        val = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()

    np.testing.assert_allclose(val, -e_ground, atol=1e-6)


def test_ising_config_recovery_five_ring():
    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, -1.0, 1.0, 1.0, -1.0]
    h = [0.5, -0.3, 0.2, 0.0, 0.4]

    best_e = None
    for cfg in itertools.product([0, 1], repeat=len(spins)):
        e = _ising_energy_of_cfg(cfg, edges, J, h)
        if best_e is None or e < best_e - 1e-9:
            best_e = e

    nodes = build_ising_tn(spins, edges, J, h)
    with tropical(track=True):
        val = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()
        cfg_sym = recover_configuration()

    np.testing.assert_allclose(val, -best_e, atol=1e-6)

    _tree, input_sets, raw_tensors = get_recorded_topology()
    assert input_sets is not None and raw_tensors is not None
    all_labels = set().union(*[set(t) for t in input_sets])
    assert set(cfg_sym.keys()) == all_labels

    recovered_total = _maxplus_total_of_assignment(cfg_sym, input_sets, raw_tensors)
    np.testing.assert_allclose(recovered_total, val, atol=1e-6)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)

    spin_cfg = {}
    for term, _t in zip(input_sets, raw_tensors):
        if len(term) == 1:
            spin_cfg[len(spin_cfg)] = int(cfg_sym[term[0]])
    assert len(spin_cfg) == len(spins)
    recovered_e = _ising_energy_of_cfg(
        [spin_cfg[i] for i in range(len(spins))], edges, J, h
    )
    np.testing.assert_allclose(recovered_e, best_e, atol=1e-6)


def test_ising_config_recovery_degenerate_ring():
    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, 1.0, 1.0, 1.0, 1.0]
    h = [0.0, 0.0, 0.0, 0.0, 0.0]

    best_e = None
    for cfg in itertools.product([0, 1], repeat=len(spins)):
        e = _ising_energy_of_cfg(cfg, edges, J, h)
        if best_e is None or e < best_e - 1e-9:
            best_e = e
    assert best_e == -5.0

    nodes = build_ising_tn(spins, edges, J, h)
    with tropical(track=True):
        val = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()
        cfg_sym = recover_configuration()

    np.testing.assert_allclose(val, -best_e, atol=1e-6)
    _tree, input_sets, raw_tensors = get_recorded_topology()
    recovered_total = _maxplus_total_of_assignment(cfg_sym, input_sets, raw_tensors)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)

    spin_cfg = {}
    for term, _t in zip(input_sets, raw_tensors):
        if len(term) == 1:
            spin_cfg[len(spin_cfg)] = int(cfg_sym[term[0]])
    recovered_e = _ising_energy_of_cfg(
        [spin_cfg[i] for i in range(len(spins))], edges, J, h
    )
    np.testing.assert_allclose(recovered_e, best_e, atol=1e-6)


def test_ising_counting_ground_state_and_degeneracy():
    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, -1.0, 1.0, 1.0, -1.0]
    h = [0.5, -0.3, 0.2, 0.0, 0.4]
    best_e, deg = brute_force_energy_and_degeneracy(len(spins), edges, J, h)

    nodes = build_ising_tn(spins, edges, J, h)

    with counting_tropical():
        res = cons.contractor(nodes, output_edge_order=[], ignore_edge_order=True)
        count = degeneracy()

    energy = np.array(res.tensor)
    np.testing.assert_allclose(energy, -best_e, atol=1e-6)
    np.testing.assert_allclose(np.array(count), deg, atol=1e-6)


def test_ising_counting_degenerate_ring():
    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, 1.0, 1.0, 1.0, 1.0]
    h = [0.0, 0.0, 0.0, 0.0, 0.0]
    best_e, deg = brute_force_energy_and_degeneracy(len(spins), edges, J, h)

    assert best_e == -5.0
    assert deg == 2

    nodes = build_ising_tn(spins, edges, J, h)

    with counting_tropical():
        res = cons.contractor(nodes, output_edge_order=[], ignore_edge_order=True)
        count = degeneracy()

    energy = np.array(res.tensor)
    np.testing.assert_allclose(energy, -best_e, atol=1e-6)
    np.testing.assert_allclose(np.array(count), deg, atol=1e-6)
