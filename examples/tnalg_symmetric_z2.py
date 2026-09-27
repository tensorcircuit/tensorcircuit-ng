"""
Run Z2-symmetric transverse-field Ising TEBD, TDVP, and DMRG with TNALG.
"""

import argparse

import jax
import jax.numpy as jnp

import tensorcircuit as tc


def tfi_terms(length):
    """
    Return ``-sum(XX) - 0.8 sum(Z)`` in Pauli-string form.

    :param length: Number of physical sites.
    :type length: int
    :return: Operator-code rows and one weight per row.
    :rtype: Tuple[Array, Array]
    """
    rows, weights = [], []
    for site in range(length - 1):
        row = [0] * length
        row[site] = row[site + 1] = 1
        rows.append(row)
        weights.append(-1.0)
    for site in range(length):
        row = [0] * length
        row[site] = 3
        rows.append(row)
        weights.append(-0.8)
    return jnp.asarray(rows, jnp.int32), jnp.asarray(weights, jnp.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--krylov-dim", type=int, default=8)
    args = parser.parse_args()
    if args.length < 2 or args.length % 2:
        raise ValueError("length must be an even number >= 2")

    jax.config.update("jax_enable_x64", True)
    tnalg = tc.tnalg
    spec = tnalg.MPSSpec.z2(args.length, chi=args.chi)
    codes, weights = tfi_terms(args.length)
    mpo_spec, build_mpo = tnalg.compile_mpo(codes, spec.physical_indices)
    gate_spec, build_gates = tnalg.compile_tebd_gates(codes, spec.physical_indices)
    mpo = build_mpo(weights)
    generators = build_gates(weights)
    initial, _ = tnalg.product_state(
        (0,) * args.length, spec=spec, dtype=jnp.complex128
    )
    state = initial
    dt = jnp.asarray(args.dt)
    tebd_step = tnalg.make_tebd_step(spec, gate_spec, tnalg.TEBDOptions(order=2))
    tdvp_step = tnalg.make_tdvp_step(
        spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=args.krylov_dim)
    )
    dmrg_sweep = tnalg.make_dmrg_sweep(
        spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=args.krylov_dim)
    )

    def run_tebd(carry):
        def body(value, _):
            return tebd_step(value, generators, dt), None

        return jax.lax.scan(body, carry, None, length=args.steps)[0]

    def run_tdvp(carry):
        def body(value, _):
            return tdvp_step(value, mpo, dt), None

        return jax.lax.scan(body, carry, None, length=args.steps)[0]

    def run_dmrg(mps):
        def body(value, _):
            return dmrg_sweep(value, mpo)[0], None

        return jax.lax.scan(body, mps, None, length=args.steps)[0]

    tebd_state = jax.jit(run_tebd)(state)
    tdvp_state = jax.jit(run_tdvp)(state)
    dmrg_state = jax.jit(run_dmrg)(initial)
    print(
        {
            "max_bond_dim": max(spec.bond_dims),
            "tebd_energy": float(jnp.real(tnalg.expectation(tebd_state, mpo))),
            "tdvp_energy": float(jnp.real(tnalg.expectation(tdvp_state, mpo))),
            "dmrg_energy": float(jnp.real(tnalg.expectation(dmrg_state, mpo))),
        }
    )


if __name__ == "__main__":
    main()
