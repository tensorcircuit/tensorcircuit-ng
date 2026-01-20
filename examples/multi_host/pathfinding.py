"""
The first stage of the multi host run to get the contraction path
before initializing the cluster.
"""

import logging

import jax
import numpy as np
import tensornetwork as tn

import tensorcircuit as tc
from tensorcircuit.experimental import DistributedContractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

K = tc.set_backend("jax")
tc.set_dtype("complex64")

N_QUBITS = 15
DEPTH = 5


def circuit_ansatz(n, d, params):
    c = tc.Circuit(n)
    c.h(range(n))
    for i in range(d):
        for j in range(0, n - 1):
            c.rzz(j, j + 1, theta=params[j, i, 0])
        for j in range(n):
            c.rx(j, theta=params[j, i, 1])
        for j in range(n):
            c.ry(j, theta=params[j, i, 2])
    return c


def get_tfi_mpo(n):
    Jx = np.ones(n - 1)
    Bz = -1.0 * np.ones(n)
    tn_mpo = tn.matrixproductstates.mpo.FiniteTFI(Jx, Bz, dtype=np.complex64)
    return tc.quantum.tn2qop(tn_mpo)


def get_nodes_fn(n, d, mpo):
    def nodes_fn(params):
        psi = circuit_ansatz(n, d, params).get_quvector()
        expression = psi.adjoint() @ mpo @ psi
        return expression.nodes

    return nodes_fn


def run_path_main():

    tfi_mpo = get_tfi_mpo(N_QUBITS)
    nodes_fn = get_nodes_fn(N_QUBITS, DEPTH, tfi_mpo)
    params_shape = [N_QUBITS, DEPTH, 3]

    key = jax.random.PRNGKey(42)
    params_cpu = jax.random.normal(key, shape=params_shape, dtype=tc.rdtypestr) * 0.1

    DistributedContractor.find_path(
        nodes_fn=nodes_fn,
        params=params_cpu,
        cotengra_options={
            "slicing_reconf_opts": {"target_size": 2**12},
            "max_repeats": 32,
            "progbar": True,
            "minimize": "write",
            "parallel": 4,
        },
        filepath="tree.pkl",
    )


if __name__ == "__main__":
    run_path_main()
