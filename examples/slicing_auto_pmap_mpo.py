"""
This script illustrates how to parallelize both the contraction path
finding and sliced contraction computation for MPO expectation
"""

from functools import partial
import os

num_device = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_device}"
import cotengra as ctg
import tensornetwork as tn

import numpy as np
import scipy
import jax
import optax
import tensorcircuit as tc

backend = "jax"
K = tc.set_backend(backend)
tc.set_dtype("complex128")


def circuit2nodes(n, d, params, tc_mpo):
    c = tc.Circuit(n)
    c.h(range(n))
    for i in range(d):
        for j in range(0, n - 1):
            c.rzz(j, j + 1, theta=params[j, i, 0])
        for j in range(n):
            c.rx(j, theta=params[j, i, 1])
        for j in range(n):
            c.ry(j, theta=params[j, i, 2])

    mps = c.get_quvector()
    e = mps.adjoint() @ tc_mpo @ mps
    return e.nodes


def core(params, i, tree, n, d, tc_mpo):
    nodes = circuit2nodes(n, d, params, tc_mpo)
    _, nodes = tc.cons.get_tn_info(nodes)
    input_arrays = [node.tensor for node in nodes]
    sliced_arrays = tree.slice_arrays(input_arrays, i)
    return K.real(tree.contract_core(sliced_arrays, backend=backend))[0, 0]


core_vag = K.value_and_grad(core)


if __name__ == "__main__":
    nqubit = 12
    d = 6

    # baseline results
    lattice = tc.templates.graphs.Line1D(nqubit, pbc=False)
    h = tc.quantum.heisenberg_hamiltonian(lattice, hzz=0, hyy=0, hxx=1.0, hz=-1.0)
    es0 = scipy.sparse.linalg.eigsh(K.numpy(h), k=1, which="SA")[0]
    print("exact ground state energy: ", es0)

    params = K.implicit_randn(stddev=0.1, shape=[nqubit, d, 3], dtype=tc.rdtypestr)
    replicated_params = K.reshape(params, [1] + list(params.shape))
    replicated_params = K.tile(replicated_params, [num_device, 1, 1, 1])

    optimizer = optax.adam(5e-2)
    base_opt_state = optimizer.init(params)
    replicated_opt_state = jax.tree.map(
        lambda x: (
            jax.numpy.broadcast_to(x, (num_device,) + x.shape)
            if isinstance(x, jax.numpy.ndarray)
            else x
        ),
        base_opt_state,
    )

    @partial(
        jax.pmap,
        axis_name="pmap",
        in_axes=(0, 0, None, None, None, None, 0),
        static_broadcasted_argnums=(2, 3, 4, 5),
    )
    def para_vag(params, i, tree, n, d, tc_mpo, opt_state):
        loss, grads = core_vag(params, i, tree, n, d, tc_mpo)
        grads = jax.lax.psum(grads, axis_name="pmap")
        loss = jax.lax.psum(loss, axis_name="pmap")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    Jx = jax.numpy.array([1.0] * (nqubit - 1))  # XX coupling strength
    Bz = jax.numpy.array([-1.0] * nqubit)  # Transverse field strength
    # Create TensorNetwork MPO
    tn_mpo = tn.matrixproductstates.mpo.FiniteTFI(Jx, Bz, dtype=np.complex64)
    tc_mpo = tc.quantum.tn2qop(tn_mpo)

    nodes = circuit2nodes(nqubit, d, params, tc_mpo)
    tn_info, _ = tc.cons.get_tn_info(nodes)

    # Create ReusableHyperOptimizer for finding optimal contraction paths
    opt = ctg.ReusableHyperOptimizer(
        parallel=True,  # Enable parallel path finding
        slicing_opts={
            "target_slices": num_device,  # Split computation across available devices
            # "target_size": 2**20,  # Optional: Set memory limit per slice
        },
        max_repeats=256,  # Maximum number of path finding attempts
        progbar=True,  # Show progress bar during optimization
        minimize="combo",  # Optimize for both time and memory
    )
    tree = opt.search(*tn_info)

    inds = K.arange(num_device)
    for j in range(100):
        print(f"training loop: {j}-step")
        replicated_params, replicated_opt_state, loss = para_vag(
            replicated_params, inds, tree, nqubit, d, tc_mpo, replicated_opt_state
        )
        print(loss[0])
