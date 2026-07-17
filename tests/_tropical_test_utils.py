"""Shared helpers for the tropical Ising tensor-network tests.

Extracted from ``tests/test_tropical_ising.py`` and
``tests/test_tropical_config.py`` to remove duplication between their
near-identical CopyNode Ising TN builders and brute-force energy
enumeration loops.
"""

import itertools

import numpy as np
import tensornetwork as tn

import tensorcircuit as tc


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

    Note: an earlier draft built the delta as a dense ``tn.Node`` (0 on diagonal, a NEG
    proxy off-diagonal). That is a *regular* node, not a hyperedge, so it does not
    trigger the einsum branch and -- worse -- ``cons.contractor``'s ``_merge_single_gates``
    preprocessing collapses the whole network to one node before the algebraic path
    runs, silently bypassing the tropical primitives (result 0.0). Using ``tn.CopyNode``
    is both the faithful "copy node -> hyperedge" construction and the one that works
    end-to-end through the public ``cons.contractor`` API.
    """
    be = tc.backend
    degree = dict.fromkeys(spins, 0)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    nodes = []
    copy_nodes = {}
    # one field leg + one leg per incident edge
    for i in spins:
        cn = tn.CopyNode(degree[i] + 1, 2)  # rank (degree+1) delta, dim 2 -> hyperedge
        copy_nodes[i] = cn
        nodes.append(cn)
        # field tensor on the first leg
        tv = np.array([h[i], -h[i]], dtype=np.float64)
        tvn = tn.Node(be.cast(be.convert_to_tensor(tv), "float64"))
        tn.connect(cn[0], tvn[0])
        nodes.append(tvn)
    # bond tensors, each consuming one copy leg per endpoint
    leg_idx = dict.fromkeys(spins, 1)  # leg 0 reserved for field
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


# --- Task 10: non-scalar (one free spin) builder + brute force ---


# Topology shared by build_ring_with_free_spin / brute_nonscalar_counting.
# A 4-spin ring with spin 0 free: small enough to brute-force exactly, large
# enough to need both tensordot and einsum steps under cotengra. Integer
# couplings/fields keep the brute force exact (no rtol needed).
_RING_SPINS = (0, 1, 2, 3)
_RING_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_RING_J = (-2, 1, 1, -2)  # mixed ferro/antiferro of both magnitudes -> exercises
#                       # both tensordot and einsum branches under cotengra.
_RING_H = (2, 0, 2, -1)  # tuned (via brute-force search) so the two free-spin
#                      # values give DIFFERENT energy AND DIFFERENT degeneracy:
#                      # v_free=0 -> E=7, N=2; v_free=1 -> E=5, N=1. A swapped
#                      # or permuted aux cannot pass the per-output assertion.
_FREE_SPIN = 0


def build_ring_with_free_spin(
    spins=_RING_SPINS,
    edges=_RING_EDGES,
    j_vals=_RING_J,
    h=_RING_H,
    free_spin=_FREE_SPIN,
):
    """Build a small ring Ising with ONE spin's CopyNode given an extra dangling
    leg (the free output index). Same ``Tv``/``Te``/``tn.CopyNode`` construction
    as ``build_ising_tn`` -- the only difference is the free spin's CopyNode has
    one ADDITIONAL leg (beyond degree+1) left unconnected, which becomes the
    contraction's output index.

    All legs of a CopyNode share one hyperedge symbol, so leaving the extra leg
    dangling means the spin variable s_free appears in the einsum output -- for
    each of its two values, cotengra maximizes over the other spins. cfg=0 ->
    s=+1, cfg=1 -> s=-1 (matching ``build_ising_tn``'s ``Tv=[h,-h]`` convention).

    Returns ``(nodes, free_edge, j_vals, h)`` where ``free_edge`` is the dangling
    CopyNode leg to pass as ``output_edge_order=[free_edge]``.
    """
    be = tc.backend
    degree = dict.fromkeys(spins, 0)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    nodes = []
    copy_nodes = {}
    for i in spins:
        rank = degree[i] + 1  # one field leg + one per incident edge
        if i == free_spin:
            rank += 1  # extra leg -> dangling output index for the free spin
        cn = tn.CopyNode(rank, 2)
        copy_nodes[i] = cn
        nodes.append(cn)
        # field tensor on leg 0 (same as build_ising_tn)
        tv = np.array([h[i], -h[i]], dtype=np.float64)
        tvn = tn.Node(be.cast(be.convert_to_tensor(tv), "float64"))
        tn.connect(cn[0], tvn[0])
        nodes.append(tvn)

    # bond tensors, each consuming one copy leg per endpoint
    leg_idx = dict.fromkeys(spins, 1)  # leg 0 reserved for field
    for (i, j), jij in zip(edges, j_vals):
        te = np.array([[jij, -jij], [-jij, jij]], dtype=np.float64)
        ten = tn.Node(be.cast(be.convert_to_tensor(te), "float64"))
        tn.connect(ten[0], copy_nodes[i][leg_idx[i]])
        tn.connect(ten[1], copy_nodes[j][leg_idx[j]])
        leg_idx[i] += 1
        leg_idx[j] += 1
        nodes.append(ten)

    # the free leg is the LAST leg of the free spin's CopyNode
    # (legs 0..degree are field+bonds; leg degree+1 is the dangling output).
    free_edge = copy_nodes[free_spin][degree[free_spin] + 1]
    return nodes, free_edge, j_vals, h


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
    one free spin. Returns ``(expected_e, expected_n)`` -- two arrays of shape
    ``(2,)`` indexed by the free spin's value (0 -> s=+1, 1 -> s=-1).

    For each value v of the free spin, enumerate all configs of the OTHER spins,
    compute the Ising energy E = -sum J s_i s_j - sum h s_i with s_free fixed,
    and record min E + its degeneracy. The contraction returns max-plus(-E) per
    output (== -min E), so we return ``-min E`` to match.

    ``nodes`` and ``free_edge`` are accepted only for signature compatibility
    with the call site; the brute force is parameterised by the (hardcoded)
    ring topology that ``build_ring_with_free_spin`` emits.
    """
    del nodes, free_edge  # signature-only; topology is fixed by the defaults above
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
        # max-plus convention: contraction returns max(-E) = -min(E)
        expected_e[v_free] = -best_e
        expected_n[v_free] = deg
    return expected_e, expected_n


# --- Task 10 follow-up: non-scalar (two free spins) builder + brute force ---
#
# Extends the one-free-spin canary to a 2D output so the aux-reorder permutation
# in ``cons._algebraic_base_contraction`` (``perm = [dangling_edges.index(e) for e
# in order]``) is exercised with a NON-IDENTITY permutation. With one free edge
# the perm is always ``(0,)`` (identity); with two free edges + a reversed
# ``output_edge_order`` it becomes ``(1, 0)`` (swap), so an elided or
# wrong-direction transpose on the aux tensor is no longer silently a no-op.
#
# Same ring topology / parameters as the one-free-spin case (``_RING_*`` constants
# above), but TWO spins (0 and 1) get an extra dangling CopyNode leg. With
# ``h = [2, 0, 2, -1]`` the per-(a,b) brute-force result is ASYMMETRIC under
# transpose (E = [[5,7],[5,-1]], N = [[1,2],[1,1]]), which is load-bearing: it
# makes ``assert_allclose(N, expected_n_ab.T)`` fail if aux is left in the
# dangling order rather than the ``output_edge_order`` order.
_FREE_SPIN_A = 0
_FREE_SPIN_B = 1


def build_ring_with_two_free_spins(
    spins=_RING_SPINS,
    edges=_RING_EDGES,
    j_vals=_RING_J,
    h=_RING_H,
    free_spin_a=_FREE_SPIN_A,
    free_spin_b=_FREE_SPIN_B,
):
    """Build a small ring Ising with TWO spins' CopyNodes given an extra dangling
    leg each (two free output indices). Same ``Tv``/``Te``/``tn.CopyNode``
    construction as ``build_ising_tn`` / ``build_ring_with_free_spin`` -- the only
    difference is that BOTH ``free_spin_a`` and ``free_spin_b`` have a CopyNode of
    rank ``degree + 2`` (one field leg + one per incident edge + one dangling
    output leg).

    The two dangling legs become the contraction's two-axis output. Under
    ``cons.sorted_edges`` the dangling order comes out as ``[edge_a, edge_b]``
    (spin 0 before spin 1), so contracting with
    ``output_edge_order=[edge_b, edge_a]`` exercises a NON-trivial aux
    permutation ``(1, 0)``.

    Returns ``(nodes, edge_a, edge_b, j_vals, h)`` where ``edge_a`` is the
    dangling leg of ``free_spin_a`` (spin 0) and ``edge_b`` that of
    ``free_spin_b`` (spin 1). Pass them to ``output_edge_order`` in whichever
    order the test wants to validate.
    """
    be = tc.backend
    free_spins = {free_spin_a, free_spin_b}
    degree = dict.fromkeys(spins, 0)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    nodes = []
    copy_nodes = {}
    for i in spins:
        rank = degree[i] + 1  # one field leg + one per incident edge
        if i in free_spins:
            rank += 1  # extra leg -> dangling output index for this free spin
        cn = tn.CopyNode(rank, 2)
        copy_nodes[i] = cn
        nodes.append(cn)
        # field tensor on leg 0 (same as build_ising_tn)
        tv = np.array([h[i], -h[i]], dtype=np.float64)
        tvn = tn.Node(be.cast(be.convert_to_tensor(tv), "float64"))
        tn.connect(cn[0], tvn[0])
        nodes.append(tvn)

    # bond tensors, each consuming one copy leg per endpoint
    leg_idx = dict.fromkeys(spins, 1)  # leg 0 reserved for field
    for (i, j), jij in zip(edges, j_vals):
        te = np.array([[jij, -jij], [-jij, jij]], dtype=np.float64)
        ten = tn.Node(be.cast(be.convert_to_tensor(te), "float64"))
        tn.connect(ten[0], copy_nodes[i][leg_idx[i]])
        tn.connect(ten[1], copy_nodes[j][leg_idx[j]])
        leg_idx[i] += 1
        leg_idx[j] += 1
        nodes.append(ten)

    # each free leg is the LAST leg of its CopyNode
    # (legs 0..degree are field+bonds; leg degree+1 is the dangling output).
    edge_a = copy_nodes[free_spin_a][degree[free_spin_a] + 1]
    edge_b = copy_nodes[free_spin_b][degree[free_spin_b] + 1]
    return nodes, edge_a, edge_b, j_vals, h


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
    """Brute-force per-output ``(energy, degeneracy)`` for the non-scalar ring
    with TWO free spins. Returns ``(expected_e, expected_n)`` -- two ``(2, 2)``
    arrays indexed by ``[v_a, v_b]`` where ``v_a`` is the value of
    ``free_spin_a`` (cfg=0 -> s=+1, cfg=1 -> s=-1) and ``v_b`` that of
    ``free_spin_b``.

    For each ``(v_a, v_b)`` pair, enumerate the other spins, compute
    ``E = -sum J s_i s_j - sum h s_i`` with both free spins fixed, and record
    ``min E`` + its degeneracy. Returns ``-min E`` to match the max-plus sign
    convention of ``brute_nonscalar_counting``.

    ``nodes`` / ``edge_a`` / ``edge_b`` are signature-only (the topology is fixed
    by the defaults above); they are accepted so the call site reads symmetrically
    with ``build_ring_with_two_free_spins``'s return signature.

    For the default parameters the brute-force result is
    ``E = [[5, 7], [5, -1]]`` and ``N = [[1, 2], [1, 1]]``, both asymmetric under
    transpose -- a property the test relies on so a missed aux transpose cannot
    pass ``assert_allclose(N, expected_n_ab.T)``.
    """
    del nodes, edge_a, edge_b  # signature-only; topology is fixed by the defaults above
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
            # max-plus convention: contraction returns max(-E) = -min(E)
            expected_e[v_a, v_b] = -best_e
            expected_n[v_a, v_b] = deg
    return expected_e, expected_n
