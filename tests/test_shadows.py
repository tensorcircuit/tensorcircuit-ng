import pytest
from pytest_lazyfixture import lazy_fixture as lf
import numpy as np
import tensorcircuit as tc
from tensorcircuit.shadows import (
    shadow_bound,
    shadow_snapshots,
    global_shadow_state,
    entropy_shadow,
    renyi_entropy_2,
    expectation_ps_shadow,
    global_shadow_state1,
    global_shadow_state2,
)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_jit(backend):
    nq, repeat = 8, 5
    ps = [1, 0, 0, 0, 2, 0, 0, 0]
    sub = (1, 3, 6, 7)
    error = 0.1
    ns, k = shadow_bound(ps, error)
    ns //= repeat

    thetas = 2 * np.random.rand(2, nq) - 1

    c = tc.Circuit(nq)
    for i in range(nq):
        c.H(i)
    for i in range(2):
        for j in range(nq):
            c.cnot(j, (j + 1) % nq)
        for j in range(nq):
            c.rz(j, theta=thetas[i, j] * np.pi)

    psi = c.state()
    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, repeat))

    def classical_shadow(psi, pauli_strings, status):
        lss_states = shadow_snapshots(psi, pauli_strings, status)
        expc = expectation_ps_shadow(lss_states, ps=ps, k=k)
        ent = entropy_shadow(lss_states, sub=sub, alpha=2)
        return expc, ent

    csjit = tc.backend.jit(classical_shadow)

    exact_expc = c.expectation_ps(ps=ps)
    exact_rdm = tc.quantum.reduced_density_matrix(psi, cut=[0, 2, 4, 5])
    exact_ent = tc.quantum.renyi_entropy(exact_rdm, k=2)
    expc, ent = csjit(psi, pauli_strings, status)
    expc = np.median(expc)

    np.testing.assert_allclose(expc, exact_expc, atol=error)
    np.testing.assert_allclose(ent, exact_ent, atol=5 * error)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_state(backend):
    nq, ns = 2, 10000

    c = tc.Circuit(nq)
    c.H(0)
    c.cnot(0, 1)

    psi = c.state()
    bell_state = psi[:, None] @ psi[None, :]

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, 5))
    lss_states = shadow_snapshots(c.state(), pauli_strings, status)
    sdw_state = global_shadow_state(lss_states)
    sdw_state1 = global_shadow_state1(lss_states)

    np.testing.assert_allclose(sdw_state, bell_state, atol=0.1)
    np.testing.assert_allclose(sdw_state1, bell_state, atol=0.1)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_ent(backend):
    nq, ns, repeat = 6, 1000, 500

    thetas = 2 * np.random.rand(2, nq) - 1

    c = tc.Circuit(nq)
    for i in range(nq):
        c.H(i)
    for i in range(2):
        for j in range(nq):
            c.cnot(j, (j + 1) % nq)
        for j in range(nq):
            c.rz(j, theta=thetas[i, j] * np.pi)

    sub = [1, 4]
    psi = c.state()

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, repeat))
    snapshots = shadow_snapshots(psi, pauli_strings, status, measurement_only=True)

    exact_rdm = tc.quantum.reduced_density_matrix(
        psi, cut=[i for i in range(nq) if i not in sub]
    )
    exact_ent = tc.quantum.renyi_entropy(exact_rdm, k=2)
    ent = entropy_shadow(snapshots, pauli_strings, sub, alpha=2)
    ent2 = renyi_entropy_2(snapshots, sub)

    np.testing.assert_allclose(ent, exact_ent, atol=0.1)
    np.testing.assert_allclose(ent2, exact_ent, atol=0.1)


# @pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
# def test_expc(backend):
#     import pennylane as qml
#
#     nq, ns, repeat = 6, 2000, 1000
#
#     thetas = 2 * np.random.rand(2, nq) - 1
#
#     c = tc.Circuit(nq)
#     for i in range(nq):
#         c.H(i)
#     for i in range(2):
#         for j in range(nq):
#             c.cnot(j, (j + 1) % nq)
#         for j in range(nq):
#             c.rz(j, theta=thetas[i, j] * np.pi)
#
#     ps = [1, 0, 0, 0, 0, 3]
#     sub = [1, 4]
#     psi = c.state()
#
#     pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
#     status = tc.backend.convert_to_tensor(np.random.rand(ns, repeat))
#     snapshots = shadow_snapshots(psi, pauli_strings, status, measurement_only=True)
#
#     exact_expc = c.expectation_ps(ps=ps)
#     exact_rdm = tc.quantum.reduced_density_matrix(
#         psi, cut=[i for i in range(nq) if i not in sub]
#     )
#     exact_ent = tc.quantum.renyi_entropy(exact_rdm, k=2)
#     print(exact_expc, exact_ent)
#
#     expc = np.median(expection_ps_shadow(snapshots, pauli_strings, ps=ps, k=9))
#     ent = entropy_shadow(snapshots, pauli_strings, sub, alpha=2)
#     ent2 = renyi_entropy_2(snapshots, sub)
#     print(expc, ent, ent2)
#
#     pl_snapshots = np.asarray(snapshots).reshape(ns * repeat, nq)
#     pl_ps = np.tile(np.asarray(pauli_strings - 1)[:, None, :], (1, repeat, 1)).reshape(
#         ns * repeat, nq
#     )
#     shadow = qml.ClassicalShadow(pl_snapshots, pl_ps)
#     H = qml.PauliX(0) @ qml.PauliZ(5)
#     pl_expc = shadow.expval(H, k=9)
#     pl_ent = shadow.entropy(sub, alpha=2)
#     print(pl_expc, pl_ent)
#
#     assert np.isclose(expc, pl_expc)
#     assert np.isclose(ent, pl_ent)


def test_shadow_extra(jaxb):

    ps = [1, 2, 3]  # X, Y, Z
    N, k = shadow_bound(ps, 0.1)
    assert N > 0
    assert k > 0

    # test shadow_snapshots with measurement_only and sub
    c = tc.Circuit(3)
    c.h(range(3))
    psi = c.state()
    ns = 2
    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, 3)))

    snapshots = shadow_snapshots(psi, pauli_strings, measurement_only=True, sub=[0, 1])
    assert snapshots.shape == (ns, 1, 2)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_entropy_shadow_dual_and_validation(backend):
    nq, ns, repeat = 4, 2000, 1
    c = tc.Circuit(nq)
    c.h(0)
    c.rx(1, theta=0.7)
    c.cx(0, 1)
    c.ry(2, theta=0.4)
    c.cx(1, 2)
    c.rz(3, theta=1.1)
    c.cx(2, 3)
    psi = c.state()

    # Asymmetry guard: complementary halves of a pure state have equal
    # entropy, so compare non-complementary subsystems — otherwise a routing
    # bug picking the wrong subsystem would still pass.
    e_01 = float(np.real(tc.quantum.entanglement_entropy(psi, [0, 1])))
    e_02 = float(np.real(tc.quantum.entanglement_entropy(psi, [0, 2])))
    assert not np.isclose(
        e_01, e_02, atol=1e-3
    ), "fixture is symmetric; test is vacuous"

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(ns, nq)))
    status = tc.backend.convert_to_tensor(np.random.rand(ns, repeat))
    snapshots = shadow_snapshots(psi, pauli_strings, status, measurement_only=True)

    # The three forms below resolve to the same keep=[0,1], so they must
    # agree within shadow statistical noise (loose atol, not exact).
    e_sub = float(
        np.real(entropy_shadow(snapshots, pauli_strings, sub=[0, 1], alpha=2))
    )
    e_keep = float(
        np.real(
            entropy_shadow(snapshots, pauli_strings, subsystem_to_keep=[0, 1], alpha=2)
        )
    )
    e_to = float(
        np.real(
            entropy_shadow(
                snapshots, pauli_strings, subsystems_to_trace_out=[2, 3], alpha=2
            )
        )
    )
    assert np.isclose(e_sub, e_keep, atol=0.3)
    assert np.isclose(e_keep, e_to, atol=0.3)

    # renyi_entropy_2 dual
    r_sub = float(renyi_entropy_2(snapshots, sub=[0, 1]))
    r_keep = float(renyi_entropy_2(snapshots, subsystem_to_keep=[0, 1]))
    r_to = float(renyi_entropy_2(snapshots, subsystems_to_trace_out=[2, 3]))
    assert np.isclose(r_sub, r_keep, atol=0.3)
    assert np.isclose(r_keep, r_to, atol=0.3)

    # validation
    with pytest.raises(ValueError, match="only one of"):
        entropy_shadow(
            snapshots, pauli_strings, subsystem_to_keep=[0], subsystems_to_trace_out=[1]
        )
    with pytest.raises(ValueError, match="out of range"):
        entropy_shadow(snapshots, pauli_strings, subsystem_to_keep=[nq])


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_global_shadow_state_pauli_strings(backend):
    # Cover the ``pauli_strings is not None`` branch of ``global_shadow_state``
    # and ``global_shadow_state1`` (unreached by existing tests, which only call
    # the no-pauli_strings branch) plus the ``sub`` subsystem path and the
    # 3-d rank validation. Physics invariants:
    #   * the two implementations reconstruct the same density matrix;
    #   * on a known pure state they converge to the true density matrix;
    #   * the subsystem shadow of |+0> on qubit 0 reconstructs |+><+|.
    # ``status`` and ``pauli_strings`` are deterministic (no global RNG seed):
    # a balanced tiling of all 3^nq Pauli bases plus a stratified [0, 1) grid
    # samples the measurement distribution reproducibly across backends and
    # converges faster than i.i.d. draws.
    nq, ns, repeat = 2, 8000, 5

    c = tc.Circuit(nq)
    c.h(0)
    c.cnot(0, 1)
    psi = c.state()
    rho_true = tc.backend.numpy(psi[:, None] @ psi[None, :])

    base = np.array([[i, j] for i in (1, 2, 3) for j in (1, 2, 3)])
    pauli_strings = tc.backend.convert_to_tensor(
        np.tile(base, (ns // len(base) + 1, 1))[:ns]
    )
    status = tc.backend.convert_to_tensor(
        np.linspace(0, 1, ns * repeat, endpoint=False).reshape(ns, repeat)
    )
    snapshots = shadow_snapshots(psi, pauli_strings, status, measurement_only=True)

    rho0 = tc.backend.numpy(global_shadow_state(snapshots, pauli_strings))
    rho1 = tc.backend.numpy(global_shadow_state1(snapshots, pauli_strings))
    np.testing.assert_allclose(rho1, rho0, atol=1e-6)
    np.testing.assert_allclose(rho0, rho_true, atol=0.1)

    # subsystem: |+0> -> reduced state of qubit 0 is |+><+|.
    c = tc.Circuit(nq)
    c.h(0)
    psi_plus = c.state()
    snaps = shadow_snapshots(psi_plus, pauli_strings, status, measurement_only=True)
    rho_sub = tc.backend.numpy(global_shadow_state(snaps, pauli_strings, sub=[0]))
    assert rho_sub.shape == (2, 2)
    np.testing.assert_allclose(rho_sub, [[0.5, 0.5], [0.5, 0.5]], atol=0.1)

    # 3-d rank validation: already-local (5-d) snapshots are rejected.
    snapshots_5d = np.zeros((4, 3, 2, 2, 2), dtype=np.float32)
    ps = np.ones((4, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="should be 3-d"):
        global_shadow_state1(snapshots_5d, ps)


def test_global_shadow_state2_matches(jaxb):
    # ``global_shadow_state2`` (vmap-over-einsum variant) is unreached by existing
    # tests. It is not TF-graph-compatible (it iterates over a symbolic tensor
    # inside ``tf.vectorized_map``), so the equivalence check runs on jax only.
    # Invariant: it reconstructs the same density matrix as ``global_shadow_state``.
    nq, ns, repeat = 2, 8000, 5

    c = tc.Circuit(nq)
    c.h(0)
    c.cnot(0, 1)
    psi = c.state()

    base = np.array([[i, j] for i in (1, 2, 3) for j in (1, 2, 3)])
    pauli_strings = tc.backend.convert_to_tensor(
        np.tile(base, (ns // len(base) + 1, 1))[:ns]
    )
    status = tc.backend.convert_to_tensor(
        np.linspace(0, 1, ns * repeat, endpoint=False).reshape(ns, repeat)
    )
    snapshots = shadow_snapshots(psi, pauli_strings, status, measurement_only=True)

    rho0 = tc.backend.numpy(global_shadow_state(snapshots, pauli_strings))
    rho2 = tc.backend.numpy(global_shadow_state2(snapshots, pauli_strings))
    np.testing.assert_allclose(rho2, rho0, atol=1e-6)
