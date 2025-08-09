"""
amplitude constraction on multiple GPU cards with neat interface `DistributedContractor`
"""

import os

NUM_DEVICES = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_DEVICES}"

import time
import jax
from jax import numpy as jnp
import tensorcircuit as tc
from tensorcircuit.experimental import DistributedContractor

K = tc.set_backend("jax")
tc.set_dtype("complex64")


N_QUBITS = 16
DEPTH = 8


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


def get_nodes_fn(n, d):
    def nodes_fn(params):
        psi = circuit_ansatz(n, d, params["circuit"])
        return psi.amplitude_before(params["amplitude"])

    return nodes_fn


def get_binary_representation(i: int, N: int) -> jax.Array:
    """
    Generates the binary representation of an integer as a JAX array.
    """
    # Create an array of shift amounts, from N-1 down to 0
    # For N=8, this is [7, 6, 5, 4, 3, 2, 1, 0]
    shifts = jnp.arange(N - 1, -1, -1)
    # Right-shift the integer 'i' by each amount in 'shifts'.
    # This effectively isolates each bit at the rightmost position.
    # For i=5 (..0101) and shifts=[..., 3, 2, 1, 0]
    # shifted_i will be [..0, ..0, ..1, ..10, ..101] -> [0, 0, 1, 2, 5]
    shifted_i = i >> shifts
    # Use a bitwise AND with 1 to extract just the last bit from each shifted value.
    # [0&1, 0&1, 1&1, 2&1, 5&1] -> [0, 0, 1, 0, 1]
    # We explicitly cast to int32 as requested.
    bits = (shifted_i & 1).astype(jnp.int32)
    return bits


if __name__ == "__main__":
    print(f"JAX is using {jax.local_device_count()} devices.")

    nodes_fn = get_nodes_fn(N_QUBITS, DEPTH)

    @K.jit
    def baseline(params):
        psi = circuit_ansatz(N_QUBITS, DEPTH, params["circuit"])
        return psi.amplitude(params["amplitude"])

    key = jax.random.PRNGKey(42)
    params_circuit = (
        jax.random.normal(key, shape=[N_QUBITS, DEPTH, 3], dtype=tc.rdtypestr) * 0.1
    )
    params = {
        "circuit": params_circuit,
        "amplitude": get_binary_representation(0, N_QUBITS),
    }
    DC = DistributedContractor(
        nodes_fn=nodes_fn,
        params=params,
        cotengra_options={
            "slicing_reconf_opts": {"target_size": 2**16},
            "max_repeats": 64,
            "progbar": True,
            "minimize": "write",
            "parallel": 4,
        },
    )

    n_steps = 100

    print("\nStarting amplitude loop...")
    for i in range(n_steps):
        bs_vector = get_binary_representation(i, N_QUBITS)
        t0 = time.time()
        params = {"circuit": params_circuit, "amplitude": bs_vector}
        amp = DC.value(params)
        t1 = time.time()
        print(
            f"Bitstring: {K.numpy(bs_vector).tolist()} | "
            f"amp: {amp:.8f} | "
            f"baseline_amp: {baseline(params):.8f} | "
            f"Time: {t1 - t0:.4f} s"
        )
