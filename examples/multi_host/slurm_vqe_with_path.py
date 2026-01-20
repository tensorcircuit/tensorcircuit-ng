"""
Slurm GPU cluster version, so there is no fake multi devices flags.
The master ip and process id can be automatically detected under `srun`.
"""

import time
import logging

# --- JAX and related library imports ---
import jax
import jax.distributed
import numpy as np
import optax
import tensornetwork as tn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import tensorcircuit as tc
from tensorcircuit.experimental import DistributedContractor, broadcast_py_object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

K = tc.set_backend("jax")
tc.set_dtype("complex64")

N_QUBITS = 26
DEPTH = 12


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


def run_vqe_main():
    """
    Main logic run by ALL processes.
    """
    jax.distributed.initialize()
    print(f"jax.process_index() reports: {jax.process_index()}")

    global_mesh = Mesh(jax.devices(), axis_names=("devices",))
    if jax.process_index() == 0:
        print(f"--- Global mesh created with devices: {global_mesh.devices}")

    tfi_mpo = get_tfi_mpo(N_QUBITS)
    nodes_fn = get_nodes_fn(N_QUBITS, DEPTH, tfi_mpo)
    params_shape = [N_QUBITS, DEPTH, 3]

    # --- KEY CHANGE: Create params on host 0 and broadcast to all others ---
    params_cpu = None
    if jax.process_index() == 0:
        key = jax.random.PRNGKey(42)
        params_cpu = (
            jax.random.normal(key, shape=params_shape, dtype=tc.rdtypestr) * 0.1
        )

    # Broadcast the CPU array. Now all processes have a concrete `params_cpu`.
    # This is CRITICAL to prevent the NoneType error upon contractor initialization.
    params_cpu = broadcast_py_object(params_cpu)
    # Shard the parameters onto devices for the actual GPU/TPU computation.
    params_sharding = NamedSharding(global_mesh, P(*([None] * len(params_shape))))
    params = jax.device_put(params_cpu, params_sharding)

    DC = DistributedContractor.from_path(
        filepath="tree.pkl",
        nodes_fn=nodes_fn,
        mesh=global_mesh,
        params=params,
    )

    # Initialize the optimizer and its state.
    optimizer = optax.adam(2e-2)
    opt_state = optimizer.init(params)  # Can init directly with sharded params

    @jax.jit
    def opt_update(params, opt_state, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Run the optimization loop.
    n_steps = 200
    if jax.process_index() == 0:
        print("\nStarting VQE optimization loop...")

    for i in range(n_steps):
        t0 = time.time()
        loss, grads = DC.value_and_grad(params)
        params, opt_state = opt_update(params, opt_state, grads)
        t1 = time.time()

        if jax.process_index() == 0:
            print(
                f"Step {i+1:03d} | " f"Loss: {loss:.8f} | " f"Time: {t1 - t0:.4f} s",
                flush=True,
            )

    jax.distributed.shutdown()


if __name__ == "__main__":
    run_vqe_main()
