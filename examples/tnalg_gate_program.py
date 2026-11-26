"""
Route different circuits, simulate them as one vmapped MPS batch, and sample.
"""

import jax
import jax.numpy as jnp
import numpy as np

import tensorcircuit as tc


def main() -> None:
    """Simulate two interaction graphs in one batch and sample in shot chunks."""
    tc.set_backend("jax")
    n, chi, chunk, chunks = 8, 16, 500, 2

    programs, layouts = [], []
    for edges in (((0, 7), (1, 5), (2, 6)), ((0, 4), (1, 7), (3, 6))):
        circuit = tc.Circuit(n)
        for site in range(n):
            circuit.h(site)
        for i, j in edges:
            circuit.rzz(i, j, theta=0.37 + 0.03 * i)
        for site in range(n):
            circuit.rx(site, theta=0.21)
        routed, info = tc.compiler.linear_route(circuit)
        programs.append(tc.tnalg.prepare_gate_program(routed))
        layouts.append(info["final_layout"])
    batch = tc.tnalg.stack_gate_programs(programs)

    @jax.jit
    def simulate(batch):
        return jax.vmap(lambda p: tc.tnalg.run_gate_program(p, chi=chi))(batch)

    @jax.jit
    def sample(results, status):
        def one(result, st):
            mps = tc.tnalg.gate_program_to_mpscircuit(result)
            return jax.vmap(lambda s: mps.sample(status=s))(st)

        return jax.vmap(one)(results, status)

    results = simulate(batch)
    keys = jax.random.split(jax.random.PRNGKey(7), chunks)
    samples = jnp.concatenate(
        [
            sample(results, jax.random.uniform(k, (len(programs), chunk, n)))
            for k in keys
        ],
        axis=1,
    )
    for i, layout in enumerate(layouts):
        logical = samples[i][:, jnp.asarray(layout)]
        print("retained fraction:", float(results.fidelity[i]))
        print("first five logical samples:\n", np.asarray(logical[:5]))


if __name__ == "__main__":
    main()
