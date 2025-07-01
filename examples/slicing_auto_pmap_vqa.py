"""
This script illustrates how to parallelize both the contraction path
finding and sliced contraction computation
"""

from functools import partial
import os

num_device = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_device}"
import cotengra as ctg
import jax
import optax
import tensorcircuit as tc

backend = "jax"
K = tc.set_backend(backend)


def get_circuit(n, d, params):
    c = tc.Circuit(n)
    for i in range(d):
        for j in range(0, n - 1):
            c.rzz(j, j + 1, theta=params[j, i, 0])
        for j in range(0, n):
            c.rx(j, theta=params[j, i, 1])
    return c


def core(params, i, tree, n, d):
    c = get_circuit(n, d, params)
    nodes = c.expectation_before([tc.gates.z(), [0]], reuse=False)
    _, nodes = tc.cons.get_tn_info(nodes)
    input_arrays = [node.tensor for node in nodes]
    sliced_arrays = tree.slice_arrays(input_arrays, i)
    return K.real(tree.contract_core(sliced_arrays, backend=backend))


core_vag = K.value_and_grad(core)


if __name__ == "__main__":
    nqubit = 14
    d = 7

    params = K.ones([1, nqubit, d, 2], dtype=tc.rdtypestr)
    params = K.tile(params, [num_device, 1, 1, 1])

    optimizer = optax.adam(5e-2)
    base_opt_state = optimizer.init(params[0])
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
        in_axes=(0, 0, None, None, None, 0),
        static_broadcasted_argnums=(2, 3, 4),
    )
    def para_vag(params, i, tree, n, d, opt_state):
        loss, grads = core_vag(params, i, tree, n, d)
        grads = jax.lax.psum(grads, axis_name="pmap")
        loss = jax.lax.psum(loss, axis_name="pmap")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    c = get_circuit(nqubit, d, params[0])
    nodes = c.expectation_before([tc.gates.z(), [0]], reuse=False)
    tn_info, _ = tc.cons.get_tn_info(nodes)

    opt = ctg.ReusableHyperOptimizer(
        parallel=True,
        slicing_opts={
            "target_slices": num_device,
            # "target_size": 2**20,  # Add memory target
        },
        max_repeats=256,
        progbar=True,
        minimize="combo",
    )

    tree = opt.search(*tn_info)

    inds = K.arange(num_device)
    for j in range(20):
        print(f"training loop: {j}-step")
        params, replicated_opt_state, loss = para_vag(
            params, inds, tree, nqubit, d, replicated_opt_state
        )
        print(loss[0])
