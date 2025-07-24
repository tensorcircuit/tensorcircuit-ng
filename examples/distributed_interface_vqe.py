"""
VQE on multiple GPU cards with neat interface `DistributedContractor`
"""

import os

NUM_DEVICES = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_DEVICES}"

import time
import jax
import numpy as np
import optax
import tensornetwork as tn
import tensorcircuit as tc
from tensorcircuit.experimental import DistributedContractor

K = tc.set_backend("jax")
tc.set_dtype("complex64")


N_QUBITS = 10
DEPTH = 4


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


if __name__ == "__main__":
    # g = tc.templates.graphs.Line1D(N_QUBITS, pbc=False)
    # h = tc.quantum.heisenberg_hamiltonian(g, hxx=1, hyy=0, hzz=0, hz=-1,numpy=True, sparse=False)
    # print(np.linalg.eigvalsh(h)[0])

    print(f"JAX is using {jax.local_device_count()} devices.")

    tfi_mpo = get_tfi_mpo(N_QUBITS)
    nodes_fn = get_nodes_fn(N_QUBITS, DEPTH, tfi_mpo)

    @K.jit
    def baseline(params):
        nodes = nodes_fn(params)
        return K.real(
            K.sum(
                tc.cons.contractor(
                    nodes, output_edge_order=tn.get_all_dangling(nodes)
                ).tensor
            )
        )

    key = jax.random.PRNGKey(42)
    params = (
        jax.random.normal(key, shape=[N_QUBITS, DEPTH, 3], dtype=tc.rdtypestr) * 0.1
    )

    DC = DistributedContractor(
        nodes_fn=nodes_fn,
        params=params,
        cotengra_options={
            "slicing_reconf_opts": {"target_size": 2**8},
            "max_repeats": 16,
            "progbar": True,
            "minimize": "write",
            "parallel": 4,
        },
    )

    optimizer = optax.adam(2e-2)
    opt_state = optimizer.init(params)

    @K.jit
    def opt_update(params, opt_state, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    n_steps = 100
    print("\nStarting VQE optimization loop...")
    for i in range(n_steps):
        t0 = time.time()
        loss, grads = DC.value_and_grad(params)

        print(f"loss from baseline: {baseline(params):.8f}")
        params, opt_state = opt_update(params, opt_state, grads)

        t1 = time.time()
        print(f"Step {i+1:03d} | " f"Loss: {loss:.8f} | " f"Time: {t1 - t0:.4f} s")

    print("\nOptimization finished.")
    final_energy = DC.value(
        params, op=lambda x: K.real(K.sum(x)), output_dtype=tc.rdtypestr
    )
    print(f"Final energy: {final_energy:.8f}")
