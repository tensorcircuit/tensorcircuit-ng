"""Task A1 (SPIKE) validation: argmax-tracking + tree-backtracking recovers an
optimal configuration of a tiny tropical (max, +) Ising contraction.

The assertion is energy-based: the configuration returned by
``recover_configuration()`` must achieve the brute-force optimum energy. It need
not equal a *specific* brute-force argmin when the ground state is degenerate,
but its energy must be optimal.
"""

import itertools

import numpy as np
import pytest
import tensornetwork as tn

import tensorcircuit as tc
import tensorcircuit.cons as cons
from applications.tropical_algebra import (
    tropical,
    recover_configuration,
    get_recorded_topology,
    _td_backtrack_step,
)
from tests._tropical_test_utils import build_ising_tn


def _brute_cfg(n, edges, j_vals, h):
    """Brute-force min energy over spin configs; returns (best_e, set_of_optimal_cfgs).

    cfg in {0,1}^n; spin s = 1 - 2*cfg (cfg=0 -> s=+1). E = -sum J s_i s_j - sum h s_i.
    """
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
    # 2-spin Ising: spin0 --J-- spin1, fields h0, h1 -> unique ground state.
    spins = [0, 1]
    edges = [(0, 1)]
    J = [0.7]
    h = [0.3, -0.2]
    best_e, best_cfgs = _brute_cfg(len(spins), edges, J, h)

    nodes = build_ising_tn(spins, edges, J, h)
    val, cfg = _contract_tropical_track(nodes)

    # contraction returns max_cfg(-E) = -E_ground
    np.testing.assert_allclose(val, -best_e, atol=1e-6)

    _tree, input_sets, raw_tensors = get_recorded_topology()
    assert input_sets is not None and raw_tensors is not None
    # every index label got a value
    all_labels = set().union(*[set(t) for t in input_sets])
    assert set(cfg.keys()) == all_labels

    recovered_total = _energy_of(cfg, input_sets, raw_tensors)
    # the recovered config's max-plus total must equal the contraction value
    # and hence -best_e (optimal energy).
    np.testing.assert_allclose(recovered_total, val, atol=1e-6)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)

    # unique ground state: the recovered (symbol) config must map to one of the
    # brute-force optimal cfgs. Map symbols -> per-spin cfg via the field tensors:
    # the field tensor for spin i has a single symbol whose value is cfg[i].
    # Recover that mapping from the topology (field tensors are the rank-1 terms).
    spin_cfg = {}
    for term, _t in zip(input_sets, raw_tensors):
        if len(term) == 1:  # field tensor -> one spin
            spin_cfg[len(spin_cfg)] = int(cfg[term[0]])
    assert tuple(spin_cfg[i] for i in range(len(spins))) in best_cfgs


def test_config_recovery_tiny_ising_degenerate():
    # Degenerate ground state (Z2 spin-flip symmetry): ferromagnetic bond, no field.
    # Any recovered config must still be optimal (energy-based assertion).
    spins = [0, 1]
    edges = [(0, 1)]
    J = [1.0]
    h = [0.0, 0.0]
    best_e, best_cfgs = _brute_cfg(len(spins), edges, J, h)
    assert len(best_cfgs) == 2  # all-up and all-down

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
    # Slightly bigger: 3-spin open chain -> exercises a 3rd pairwise step.
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
    """``recover_configuration()`` is scalar-only: a contraction whose
    root has dangling/free output indices (non-scalar result) must raise
    ``NotImplementedError`` rather than return a (wrong) configuration.

    Builds a max-plus matrix-vector product
    ``result[i] = max_k(A[i,k] + B[k])`` (one free index ``i``, one contracted
    index ``k``) -- a non-scalar root -- and asserts recovery raises. The
    non-scalar backtracking wiring has known ordering bugs (tree.output order
    vs result-shape order; output-label values lost in the einsum branch), so it
    is gated instead of shipping wrong answers.
    """
    be = tc.backend
    a_np = np.array(
        [[1.0, 4.0, 2.0], [3.0, 1.0, 5.0], [0.0, 2.0, 1.0]]
    )  # A[i,k], shape (3,3)
    b_np = np.array([0.5, 1.0, -0.5])  # B[k], shape (3,)

    a_node = tn.Node(be.cast(be.convert_to_tensor(a_np), "float64"))
    b_node = tn.Node(be.cast(be.convert_to_tensor(b_np), "float64"))
    tn.connect(a_node[1], b_node[0])  # contract k; leave i (a_node[0]) dangling
    nodes = [a_node, b_node]

    with tropical(track=True):
        res = cons.contractor(nodes, output_edge_order=[a_node[0]])
        # sanity: the contraction itself is fine and non-scalar (shape (3,))
        assert np.array(res.tensor).shape == (3,)
        with pytest.raises(NotImplementedError):
            recover_configuration()


class _StubTree:
    """Minimal stand-in for a cotengra ``ContractionTree`` exposing only the
    three methods that ``_td_backtrack_step`` reads (``get_inds``,
    ``get_tensordot_axes``, ``get_tensordot_perm``). Used to feed the backtrack
    step a synthetic non-``None`` perm without spinning up a real contraction.
    """

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
    """Synthetic-perm unit test for ``_td_backtrack_step`` (the I1 guard).

    The scalar 5-ring canary contracts with all-``None`` ``get_tensordot_perm``
    values, so the perm!=None inversion branch -- mapping a canonical
    (``get_inds(p)``) position to the algebra-natural (``l_free + r_free``)
    order via ``td_pos[perm[i]] = p_pos[i]`` -- is NEVER exercised end-to-end.
    This feeds the backtrack step a synthetic 3-cycle perm (non-self-inverse, so
    a wrong inversion *direction* is caught, unlike a transposition) and asserts
    the recovered per-axis positions are exactly the inverse of the perm.

    Guards the inversion logic for future non-scalar support and against cotengra
    convention changes.
    """
    # Geometry: l = 'ab' (a free, b contracted), r = 'bcd' (b contracted, c,d
    # free). l_axes=[1] (b is axis 1 of l), r_axes=[0] (b is axis 0 of r).
    # Algebra-natural output order = l_free + r_free = 'acd'.
    # Canonical (get_inds(p)) order chosen as 'dac' (a 3-cycle of 'acd'), so the
    # perm is non-trivial and non-self-inverse:
    #   perm[i] = 'acd'.find('dac'[i]) = (2, 0, 1).
    l_node, r_node, p_node = "l", "r", "p"
    tree = _StubTree(
        inds={l_node: "ab", r_node: "bcd", p_node: "dac"},
        axes={p_node: ([1], [0])},
        perm={p_node: (2, 0, 1)},
    )
    # argmax lives in algebra-natural 'acd' order, shape (dim_a, dim_c, dim_d).
    # Place flattened contracted index 3 at td_pos (a=1, c=2, d=0); with
    # contract_dims=(4,) -> contracted label 'b' value 3.
    argmax = np.zeros((2, 3, 2), dtype=np.int64)
    argmax[1, 2, 0] = 3
    rec = {"kind": "td", "argmax": argmax, "contract_dims": (4,)}
    # Canonical p_pos in 'dac' order: d=0, a=1, c=2.
    p_pos = (0, 1, 2)
    assignment = {}
    l_pos, r_pos = _td_backtrack_step(
        tree, p_node, l_node, r_node, rec, p_pos, assignment
    )

    # Inversion applied: canonical 'dac'(0,1,2) -> algebra-natural 'acd'(1,2,0),
    # i.e. td_pos = (a=1, c=2, d=0); argmax[1,2,0]=3 -> 'b'=3.
    assert assignment["b"] == 3
    # l_pos in 'ab' order: a=1 (free via inversion), b=3 (contracted).
    assert l_pos == (1, 3)
    # r_pos in 'bcd' order: b=3 (contracted), c=2 (free), d=0 (free).
    assert r_pos == (3, 2, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
