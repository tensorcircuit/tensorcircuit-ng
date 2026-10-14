"""
Compare fidelity, XEB, and shadow overlap for a noisy random circuit.

The shadow overlap follows Huang et al., arXiv:2404.07281. The final
single-qubit Pauli measurement is averaged analytically in this simulation.
The normalized value is for numerical comparison; the paper's certification
bound is stated for the unnormalized overlap.
"""

from itertools import product
from functools import partial
import time
import numpy as np
from scipy import stats

import tensorcircuit as tc

K = tc.set_backend("jax")


@partial(K.jit, static_argnums=(0, 2))
def get_projected_rm_from_oracle(oracle, traceout, left):
    deltas = []
    for i in range(len(traceout)):
        if i not in left:
            deltas.append([traceout[i]])
        else:
            deltas.append([0, 1])
    bitstrings = list(product(*deltas))
    r = []
    for b in bitstrings:
        r.append(oracle(K.convert_to_tensor(b)))
    r = K.stack(r)
    return r / (K.norm(r) + 1e-10)


def oracle_by_circuit(c):
    s = c.state()
    L = c._nqubits
    indexing = K.convert_to_tensor([2 ** (L - i - 1) for i in range(L)])

    @K.jit
    def f(b):
        ind = K.tensordot(indexing, b, 1)
        return K.gather1d(s, ind)

    return f


@partial(K.jit, static_argnums=(2, 3, 4))
def omega(s, status, L, oracle, left):
    c = tc.DMCircuit(L, dminputs=s)
    right = [i for i in range(L) if i not in left]
    traceout_wo_left = c.measure(*right, status=status)[0]
    traceout = []
    j = 0
    for i in range(L):
        if i not in left:
            traceout.append(traceout_wo_left[j])
            j += 1
        else:
            traceout.append(0.0)
    traceout = K.stack(traceout)
    traceout = K.cast(traceout, "int32")
    rho = c.projected_subsystem(traceout, left)
    psi = get_projected_rm_from_oracle(oracle, traceout, left)
    # print(rho.shape, psi.shape)
    return fidelity(rho, psi)


@K.jit
def fidelity(rho, psi):
    return K.real(
        (K.conj(K.reshape(psi, [1, -1])) @ rho @ K.reshape(psi, [-1, 1]))[0, 0]
    )


@K.jit
def fidelity_mc(ss, psi, with_std=True):
    rs = K.abs(K.tensordot(K.conj(ss), psi, 1)) ** 2
    m = K.mean(rs)
    if not with_std:
        return m
    return m, K.std(rs)


@partial(K.jit, static_argnums=(3))
def xeb(s, sideal, status, L):
    indexing = K.convert_to_tensor([2 ** (L - i - 1) for i in range(L)])
    c = tc.Circuit(L, inputs=s)
    rs = c.sample(batch=len(status), allow_state=True, status=status)
    corrs = []
    for r in rs:
        ind = K.tensordot(indexing, r[0], 1)
        ind = K.cast(ind, "int32")
        pideal = K.abs(K.gather1d(sideal, ind)) ** 2
        corrs.append(pideal)
    return K.mean(K.stack(corrs))


xeb_batch = K.jit(K.vmap(xeb, vectorized_argnums=(0, 2)), static_argnums=(3,))


def xeb_average(ss, sideal, L, batch=64, with_std=True, with_normalized=True):
    status = np.random.uniform(size=[len(ss), batch])
    rs = xeb_batch(ss, sideal, status, L)
    m = K.mean(rs)
    returns = [m]
    if with_normalized:
        sum_psquare = K.sum(K.abs(sideal) ** 4)
        # print(sum_psquare)
        returns.append((m - 1 / 2**L) / (sum_psquare - 1 / 2**L))
    if with_std:
        stds = K.std(rs)
        returns.append(stds)
    if with_std and with_normalized:
        returns.append(stds / (sum_psquare - 1 / 2**L))

    return tuple(returns)


@partial(K.jit, static_argnums=(2, 3, 4))
def omega_mc(s, status, L, oracle, left):
    c = tc.Circuit(L, inputs=s)
    right = [i for i in range(L) if i not in left]
    traceout_wo_left = c.measure(*right, status=status)[0]
    traceout = []
    j = 0
    for i in range(L):
        if i not in left:
            traceout.append(traceout_wo_left[j])
            j += 1
        else:
            traceout.append(0.0)
    traceout = K.stack(traceout)
    traceout = K.cast(traceout, "int32")
    rho = c.projected_subsystem(traceout, left)
    psi = get_projected_rm_from_oracle(oracle, traceout, left)
    return K.abs(K.tensordot(K.conj(rho), psi, 1)) ** 2


omega_mc_batch = K.jit(
    K.vmap(K.vmap(omega_mc, vectorized_argnums=1), vectorized_argnums=(0, 1)),
    static_argnums=(2, 3, 4),
)


def shadow_overlap_mc(ss, L, oracle, batch=16, with_normalized=True, with_std=True):
    """Estimate the shadow overlap with an equal number of shots per qubit."""
    if batch % L:
        raise ValueError("batch must be divisible by the number of qubits")
    numss = len(ss)
    status = np.random.uniform(size=[L, numss, batch // L, L])
    rs = []
    for left in range(L):
        rs.append(omega_mc_batch(ss, status[left], L, oracle, (left,)))
    rs = K.stack(rs)
    so = K.mean(rs)
    returns = [so]
    if with_normalized:
        returns.append(2 * (2**L - 1) / 2**L * (so - 1 / 2) + 1 / 2**L)
    if with_std:
        so_std = K.std(rs)
        returns.append(so_std)
    if with_std and with_normalized:
        returns.append(2 * (2**L - 1) / 2**L * (so_std))
    return returns


def shadow_overlap(s, L, oracle, batch=128, with_normalized=True, with_std=True):
    status = np.random.uniform(size=[batch, L])
    lefts = np.random.randint(L, size=[batch])
    rs = []
    for i in range(batch):
        rs.append(omega(s, status[i], L, oracle, (lefts[i],)))
    rs = K.stack(rs)
    so = K.mean(rs)
    returns = [so]
    if with_normalized:
        returns.append(2 * (2**L - 1) / 2**L * (so - 1 / 2) + 1 / 2**L)
    if with_std:
        so_std = K.std(rs)
        returns.append(so_std)
    if with_std and with_normalized:
        returns.append(2 * (2**L - 1) / 2**L * (so_std))
    return returns


@partial(K.jit, static_argnums=(0,))
def ideal_state(L, matrices):
    def loop_f(s, matrices):
        c = tc.Circuit(L, inputs=s)
        for i in range(0, L - 1, 2):
            c.unitary(i, i + 1, unitary=matrices[i])
        for i in range(1, L - 1, 2):
            c.unitary(i, i + 1, unitary=matrices[i])
        s = c.state()
        return s

    c = tc.Circuit(L)
    s = c.state()
    s1 = K.scan(loop_f, matrices, s)
    return s1


def get_matrices(L, d):
    rm = [stats.unitary_group.rvs(4) for _ in range(d * L)]
    rm = [r / np.linalg.det(r) for r in rm]
    rm = np.stack(rm)
    rm = K.reshape(rm, [d, L, 4, 4])
    return rm


@partial(K.jit, static_argnums=(0))
def noisy_state_mc(L, pn, matrices, status):
    def loop_f(s, matrices):
        c = tc.Circuit(L, inputs=s)
        for i in range(0, L - 1, 2):
            c.unitary(i, i + 1, unitary=matrices[0][i])
        for i in range(1, L - 1, 2):
            c.unitary(i, i + 1, unitary=matrices[0][i])
        for i in range(L):
            c.depolarizing(i, px=pn, py=pn, pz=pn, status=matrices[1][i])
        s = c.state()
        return s

    c = tc.Circuit(L)
    s = c.state()
    s1 = K.scan(loop_f, [matrices, status], s)
    return s1


noisy_states_mc = K.jit(
    K.vmap(noisy_state_mc, vectorized_argnums=3), static_argnums=(0,)
)


@partial(K.jit, static_argnums=(0))
def noisy_state_dm(L, pn, matrices):
    def loop_f(s, matrices):
        c = tc.DMCircuit(L, dminputs=s)
        for i in range(0, L - 1, 2):
            c.unitary(i, i + 1, unitary=matrices[i])
        for i in range(1, L - 1, 2):
            c.unitary(i, i + 1, unitary=matrices[i])
        for i in range(L):
            c.depolarizing(i, px=pn, py=pn, pz=pn)
        s = c.state()
        return s

    c = tc.DMCircuit(L)
    s = c.state()
    s1 = K.scan(loop_f, matrices, s)
    return s1


if __name__ == "__main__":
    L, d = 8, 8
    pn = 0.005
    allow_rho = True
    rm = get_matrices(L, d)
    s = ideal_state(L, rm)
    c = tc.Circuit(L, inputs=s)
    oracle = oracle_by_circuit(c)
    for _ in range(2):
        print("---------")
        # jit diff test for the 2nd time
        # also whether the number of trajectories is converged

        if allow_rho:
            time0 = time.time()
            rho = noisy_state_dm(L, pn, rm)
            rho.block_until_ready()
            print("rho time", time.time() - time0)
            time0 = time.time()
            f = fidelity(rho, s)
            f.block_until_ready()
            print("fidelity", f)
            print("fidelity time", time.time() - time0)
            time0 = time.time()
            so = shadow_overlap(rho, L, oracle, batch=512)
            so[0].block_until_ready()
            print("shadow time", time.time() - time0)
            print("shadow overlap:", so)
        time0 = time.time()
        status = np.random.uniform(size=[2000, d, L])
        ss = noisy_states_mc(L, pn, rm, status)
        ss.block_until_ready()
        print("mc states time", time.time() - time0)
        time0 = time.time()
        f_mc = fidelity_mc(ss, s)
        f_mc[0].block_until_ready()
        print("fidelity (mc)", f_mc)
        print("fidelity mc time", time.time() - time0)
        time0 = time.time()
        xeb_mc = xeb_average(ss, s, L)
        xeb_mc[0].block_until_ready()
        print("xeb mc", xeb_mc)
        print("xeb mc time", time.time() - time0)
        time0 = time.time()
        so_mc = shadow_overlap_mc(ss, L, oracle)
        so_mc[0].block_until_ready()
        print("shadow mc time", time.time() - time0)
        print("shadow overlap (mc):", so_mc)
        print(
            "xeb/fidelity",
            xeb_mc[1] / f_mc[0],
            "shadow_overlap/fidelity",
            so_mc[1] / f_mc[0],
        )
