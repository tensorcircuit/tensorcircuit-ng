import itertools
import numpy as np
import tensorcircuit.cons as cons
from applications.tropical_algebra import tropical, counting_tropical
from tests._tropical_test_utils import (
    build_ising_tn,
    brute_force_energy,
    brute_force_energy_and_degeneracy,
)


def test_ising_ring_ground_state_matches_bruteforce():
    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # 5-ring
    J = [1.0, -1.0, 1.0, 1.0, -1.0]
    h = [0.5, -0.3, 0.2, 0.0, 0.4]
    nodes = build_ising_tn(spins, edges, J, h)

    e_ground = brute_force_energy(spins, edges, J, h)

    with tropical():
        val = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()

    # contraction returns max_cfg(-E) = -E_ground
    np.testing.assert_allclose(val, -e_ground, atol=1e-6)


# --- Task A2: configuration recovery (argmax backtracking) canary ---


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
    -E for that configuration. Mapping-independent optimality proof."""
    total = 0.0
    for term, t in zip(input_sets, raw_tensors):
        idx = tuple(int(cfg_sym[c]) for c in term)
        total += float(np.asarray(t)[idx])
    return total


def test_ising_config_recovery_five_ring():
    """Task A2 canary: recover an optimal spin configuration of the 5-spin Ising
    ring under ``tropical(track=True)`` and verify its energy equals the
    brute-force ground energy.

    The 5-ring mixes ``tensordot`` (ordinary pair) and ``einsum`` (hyperedge --
    a CopyNode hub of degree >= 2 puts its symbol on >= 3 regular nodes)
    contraction steps, so this exercises both backtracking branches. Optimality
    is asserted two ways: (1) the max-plus total of the recovered symbol
    assignment equals the contraction value (== -E_ground) -- mapping-
    independent; (2) translating the assignment to per-spin values via the
    rank-1 field tensors and computing the Ising energy gives ``best_e``.
    """
    from applications.tropical_algebra import (
        get_recorded_topology,
        recover_configuration,
    )

    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # 5-ring
    J = [1.0, -1.0, 1.0, 1.0, -1.0]
    h = [0.5, -0.3, 0.2, 0.0, 0.4]

    # brute-force ground energy over the 2^5 configs
    best_e = None
    for cfg in itertools.product([0, 1], repeat=len(spins)):
        e = _ising_energy_of_cfg(cfg, edges, J, h)
        if best_e is None or e < best_e - 1e-9:
            best_e = e

    nodes = build_ising_tn(spins, edges, J, h)
    with tropical(track=True):
        val = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()
        cfg_sym = recover_configuration()

    # contraction value == -E_ground
    np.testing.assert_allclose(val, -best_e, atol=1e-6)

    _tree, input_sets, raw_tensors = get_recorded_topology()
    assert input_sets is not None and raw_tensors is not None
    # every index label received a value
    all_labels = set().union(*[set(t) for t in input_sets])
    assert set(cfg_sym.keys()) == all_labels

    # (1) mapping-independent optimality: max-plus total == contraction value
    recovered_total = _maxplus_total_of_assignment(cfg_sym, input_sets, raw_tensors)
    np.testing.assert_allclose(recovered_total, val, atol=1e-6)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)

    # (2) translate symbol config -> per-spin cfg via rank-1 field tensors
    # (Tv_i has a single index label; its value is cfg[i]). The field tensors
    # appear in input_sets in spin order (regular_nodes sorted by _stable_id_,
    # which follows creation order in build_ising_tn).
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
    """Config recovery under degeneracy (Z2-symmetric ferromagnetic ring, zero
    field): the recovered config must still be energy-optimal (it is one of the
    degenerate ground states, selected by first-argument-wins tie-breaking)."""
    from applications.tropical_algebra import (
        get_recorded_topology,
        recover_configuration,
    )

    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, 1.0, 1.0, 1.0, 1.0]  # ferromagnetic
    h = [0.0, 0.0, 0.0, 0.0, 0.0]  # zero field -> Z2 spin-flip symmetry, g=2

    best_e = None
    for cfg in itertools.product([0, 1], repeat=len(spins)):
        e = _ising_energy_of_cfg(cfg, edges, J, h)
        if best_e is None or e < best_e - 1e-9:
            best_e = e
    assert best_e == -5.0  # ferromagnetic ground state

    nodes = build_ising_tn(spins, edges, J, h)
    with tropical(track=True):
        val = np.array(cons.contractor(nodes, output_edge_order=[]).tensor).item()
        cfg_sym = recover_configuration()

    np.testing.assert_allclose(val, -best_e, atol=1e-6)
    _tree, input_sets, raw_tensors = get_recorded_topology()
    recovered_total = _maxplus_total_of_assignment(cfg_sym, input_sets, raw_tensors)
    np.testing.assert_allclose(recovered_total, -best_e, atol=1e-6)

    # the recovered spin config must be one of the two aligned ground states
    spin_cfg = {}
    for term, _t in zip(input_sets, raw_tensors):
        if len(term) == 1:
            spin_cfg[len(spin_cfg)] = int(cfg_sym[term[0]])
    recovered_e = _ising_energy_of_cfg(
        [spin_cfg[i] for i in range(len(spins))], edges, J, h
    )
    np.testing.assert_allclose(recovered_e, best_e, atol=1e-6)


# --- Task B3: counting tropical (energy*, degeneracy) canary ---


def test_ising_counting_ground_state_and_degeneracy():
    """Energy via ``node.tensor``; degeneracy via ``tr.degeneracy()`` side channel.

    Under the encode/decode design, ``CountingRepresentation.encode`` attaches
    count=1 to each leaf, so the test builder emits PLAIN energy tensors (the
    same ``build_ising_tn`` used by the max-plus tests). After contraction under
    ``counting_tropical()``, the primary tensor is the energy (-E_ground) and
    the count is stashed in the aux side-channel for ``degeneracy()`` to read.
    """
    import applications.tropical_algebra as tr

    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # 5-ring
    J = [1.0, -1.0, 1.0, 1.0, -1.0]
    h = [0.5, -0.3, 0.2, 0.0, 0.4]
    # brute-force ground energy + degeneracy (s = 1 - 2*cfg; cfg=0 -> s=+1, cfg=1 -> s=-1)
    best_e, deg = brute_force_energy_and_degeneracy(len(spins), edges, J, h)

    nodes = build_ising_tn(spins, edges, J, h)

    with counting_tropical():
        res = cons.contractor(nodes, output_edge_order=[], ignore_edge_order=True)
        count = tr.degeneracy()

    energy = np.array(res.tensor)
    # contraction returns max_cfg(-E) = -E_ground and the #cfgs achieving it.
    np.testing.assert_allclose(energy, -best_e, atol=1e-6)
    np.testing.assert_allclose(np.array(count), deg, atol=1e-6)


def test_ising_counting_degenerate_ring():
    """Degenerate ground state (g=2) canary for the counting tropical stream.

    The original ``test_ising_counting_ground_state_and_degeneracy`` instance is
    tuned to a *unique* ground state (g=1), so a regression that always returns
    ``count=1`` would pass undetected. A ferromagnetic ring with zero field has a
    Z2 (global spin-flip) symmetry -> exactly 2 ground states (all-up and
    all-down), so this case guards the degeneracy stream against such a regression.
    """
    import applications.tropical_algebra as tr

    spins = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # 5-ring
    J = [1.0, 1.0, 1.0, 1.0, 1.0]  # ferromagnetic
    h = [0.0, 0.0, 0.0, 0.0, 0.0]  # zero field -> Z2 spin-flip symmetry
    # brute-force ground energy + degeneracy (s = 1 - 2*cfg; cfg=0 -> s=+1, cfg=1 -> s=-1)
    best_e, deg = brute_force_energy_and_degeneracy(len(spins), edges, J, h)

    # sanity-check the derivation: 5 ferromagnetic bonds, all J=+1, h=0 ->
    # ground state all-spins-aligned, E_ground = -sum(J) = -5 over exactly 2
    # configs (Z2-related). Contraction returns max_cfg(-E) = -E_ground = 5.
    assert best_e == -5.0
    assert deg == 2

    nodes = build_ising_tn(spins, edges, J, h)

    with counting_tropical():
        res = cons.contractor(nodes, output_edge_order=[], ignore_edge_order=True)
        count = tr.degeneracy()

    energy = np.array(res.tensor)
    np.testing.assert_allclose(energy, -best_e, atol=1e-6)  # == 5
    np.testing.assert_allclose(np.array(count), deg, atol=1e-6)  # == 2
