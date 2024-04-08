"""
Entanglement asymmetry of tilted free fermion states
Refs: https://arxiv.org/pdf/2207.14693.pdf
Refs2: arXiv:2302.03330
"""

import numpy as np
import tensorcircuit as tc


def generate_hopping_h(J, L):
    h = np.zeros([2 * L, 2 * L], dtype=np.complex128)
    for i in range(L - 1):
        h[i, i + 1] = J
        h[i + 1 + L, i + L] = -J
        h[i + 1, i] = J
        h[i + L, i + 1 + L] = -J
    return h / 2


def generate_chemical_h(mu):
    L = len(mu)
    h = np.zeros([2 * L, 2 * L], dtype=np.complex128)
    for i in range(L):
        h[i, i] = mu[i] / 2
        h[i + L, i + L] = -mu[i] / 2
    return h


def generate_pairing_h(gamma, L):
    h = np.zeros([2 * L, 2 * L], dtype=np.complex128)
    for i in range(L - 1):
        h[i, i + 1 + L] = gamma / 2
        h[i + L, i + 1] = -gamma / 2
        h[i + 1 + L, i] = gamma / 2
        h[i + 1, i + L] = -gamma / 2
    return h


def xy_hamiltonian(theta, L):
    # $H = (1+\gamma) XX + (1-\gamma) YY + 2h Z$
    # whose GS is the cat version of titled ferromagnet state
    gamma = 2 / (np.cos(theta) ** 2 + 1) - 1
    mu1 = 4 * np.sqrt(1 - gamma**2) * np.ones([L])
    hi = (
        generate_hopping_h(2.0, L)
        + generate_pairing_h(gamma * 2, L)
        + generate_chemical_h(mu1)
    )
    return hi


def get_saq_sa(theta, l, L, k, batch=4096):
    traceout = [i for i in range(0, L // 2 - l // 2)] + [
        i for i in range(L // 2 + l // 2, L)
    ]
    hi = xy_hamiltonian(theta, L)
    c = tc.FGSSimulator(L, hc=hi)
    return np.real(
        c.renyi_entanglement_asymmetry(k, traceout, batch=batch)
    ), c.renyi_entropy(k, traceout)


def asymptotic_saq(theta, l, k):
    # eq 9 in 2207.14693, this term should be for Saq instead of Saq-Sa
    # as indicated in the paper
    return 1 / 2 * np.log(l) + 1 / 2 * np.log(
        1 / 2 * np.pi * k ** (1 / (k - 1)) * np.sin(theta) ** 2
    )


if __name__ == "__main__":
    # double check on Fig2 in 2207.14693
    print(get_saq_sa(np.pi / 4, 10, 200, 2), asymptotic_saq(np.pi / 4, 10, 2))
    print(get_saq_sa(np.pi / 2, 10, 200, 2), asymptotic_saq(np.pi / 2, 10, 2))
    # double check on the t=0 point for Fig 3
    saq, sa = get_saq_sa(np.pi / 3, 60, 600, 2)
    print(saq, asymptotic_saq(np.pi / 3, 60, 2), saq - sa)
    saq, sa = get_saq_sa(3 / 2, 100, 600, 3)
    print(saq, asymptotic_saq(3 / 2, 100, 3), saq - sa)
