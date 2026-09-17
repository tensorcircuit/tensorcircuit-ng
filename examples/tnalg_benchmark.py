"""Compare whole-step JIT and TeNPy using the same full-rank initial MPS.

Run with JAX and TeNPy installed. The TNALG side can run on CPU or GPU.
Lowering, XLA compilation, first execution, and repeated execution are
reported separately. TeNPy receives an explicit export of the same random
state; its initial bond dimensions are checked.
TEBD may subsequently redistribute its total chi across charge sectors,
whereas TNALG retains its declared quotas. TDVP and DMRG keep those quotas;
the DMRG mixer is disabled and both local solvers use the same maximum Krylov
budget. TeNPy's DMRG solver may stop at convergence instead of extending an
already exhausted Krylov space.
TEBD's separate onsite layer also differs from TeNPy's bond-Hamiltonian split.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import tenpy
from tenpy.algorithms import dmrg, tebd
from tenpy.algorithms.tdvp import SingleSiteTDVPEngine
from tenpy.models.spins import SpinChain
from tenpy.models.tf_ising import TFIChain
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS

import tensorcircuit as tc


def xxz_terms(length):
    """
    Return Pauli strings for the ``Jxx=1``, ``Jz=0.7`` XXZ Hamiltonian.

    :param length: Number of physical sites.
    :type length: int
    :return: Operator-code rows, coefficient-group indices, and group weights.
    :rtype: Tuple[Array, Array, Array]
    """
    rows, groups, weights = [], [], []
    for site in range(length - 1):
        for code in (1, 2):
            row = [0] * length
            row[site] = row[site + 1] = code
            rows.append(row)
            groups.append(2 * site)
        weights.append(0.25)
        row = [0] * length
        row[site] = row[site + 1] = 3
        rows.append(row)
        groups.append(2 * site + 1)
        weights.append(0.175)
    return (
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(groups, dtype=jnp.int32),
        jnp.asarray(weights, dtype=jnp.float64),
    )


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
    return jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(weights, dtype=jnp.float64)


def as_tenpy(state, model):
    """
    Perform an explicit comparison-only dense export.

    :param state: TNALG MPS state to export.
    :type state: tc.tnalg.MPSState
    :param model: TeNPy model used to define the physical sites.
    :type model: Any
    :return: TeNPy finite MPS with the same dense tensors.
    :rtype: MPS
    """
    elements = sum(
        a * d * b
        for a, d, b in zip(
            state.spec.bond_dims, state.spec.physical_dims, state.spec.bond_dims[1:]
        )
    )
    tensors = tc.tnalg.to_tn_mps(state, allow_dense=True, max_elements=elements).tensors
    return MPS.from_Bflat(
        model.lat.mps_sites(),
        [np.asarray(tensor).transpose(1, 0, 2) for tensor in tensors],
        bc="finite",
        form=None,
    )


def parse_arguments():
    """
    Parse dense or symmetric benchmark workloads.

    :return: Parsed command-line options.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetry", choices=("dense", "u1", "z2"), default="u1")
    parser.add_argument(
        "--assert-numerics",
        action="store_true",
        help="assert the dense TNALG/TeNPy energy and infidelity tolerances",
    )
    parser.add_argument(
        "--skip-initial-checks",
        action="store_true",
        help="skip the initial energy agreement check after preparing TeNPy",
    )
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--charge-sectors", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--krylov-dim", type=int, default=8)
    parser.add_argument("--dmrg-krylov-dim", type=int, default=64)
    parser.add_argument(
        "--algorithm", choices=("all", "tebd", "tdvp", "dmrg"), default="all"
    )
    arguments = parser.parse_args()
    if arguments.length < 2 or arguments.length % 2 or arguments.repeats < 1:
        raise ValueError("require even length >= 2 and repeats >= 1")
    if arguments.chi < 2 or min(arguments.krylov_dim, arguments.dmrg_krylov_dim) < 2:
        raise ValueError("the TeNPy comparison requires chi and Krylov budgets >= 2")
    if arguments.assert_numerics and arguments.symmetry != "dense":
        raise ValueError("--assert-numerics is only defined for --symmetry dense")
    jax.config.update("jax_enable_x64", True)
    arguments.energy_atol = 1e-7 if arguments.assert_numerics else None
    arguments.infidelity_atol = 1e-8 if arguments.assert_numerics else None
    return arguments


def main():
    arguments = parse_arguments()
    tnalg = tc.tnalg
    if arguments.symmetry == "dense":
        spec = tnalg.MPSSpec.dense((2,) * arguments.length, chi=arguments.chi)
        codes, groups, weights = xxz_terms(arguments.length)
        compiler_options = {"coefficient_indices": groups}
        model = SpinChain(
            {
                "L": arguments.length,
                "S": 0.5,
                "bc_MPS": "finite",
                "Jx": 1.0,
                "Jy": 1.0,
                "Jz": 0.7,
                "conserve": None,
            }
        )
    elif arguments.symmetry == "u1":
        spec = tnalg.MPSSpec.u1(
            arguments.length,
            total_charge=arguments.length // 2,
            chi=arguments.chi,
            charge_sectors=arguments.charge_sectors,
        )
        codes, groups, weights = xxz_terms(arguments.length)
        compiler_options = {"coefficient_indices": groups}
        model = XXZChain(
            {
                "L": arguments.length,
                "bc_MPS": "finite",
                "Jxx": 1.0,
                "Jz": 0.7,
                "conserve": "Sz",
            }
        )
    else:
        spec = tnalg.MPSSpec.z2(arguments.length, chi=arguments.chi)
        codes, weights = tfi_terms(arguments.length)
        compiler_options = {}
        model = TFIChain(
            {
                "L": arguments.length,
                "bc_MPS": "finite",
                "J": 1.0,
                "g": 0.8,
                "conserve": "parity",
            }
        )
    benchmark(
        arguments,
        spec,
        codes,
        weights,
        model,
        compiler_options,
    )


def benchmark(
    arguments,
    spec,
    codes,
    weights,
    model,
    compiler_options,
):
    """
    Compare identical initial states with whole-step JIT and synchronized timing.

    :param arguments: Parsed benchmark options.
    :type arguments: argparse.Namespace
    :param spec: Static TNALG MPS specification.
    :type spec: tc.tnalg.MPSSpec
    :param codes: Operator-code rows for the model Hamiltonian.
    :type codes: Array
    :param weights: Dynamic coefficient values for the operator rows or groups.
    :type weights: Array
    :param model: TeNPy reference model.
    :type model: Any
    :param compiler_options: Optional keyword arguments for TNALG compilers.
    :type compiler_options: Optional[dict[str, Any]]
    """
    tnalg = tc.tnalg
    compiler_options = {} if compiler_options is None else compiler_options
    setup_start = time.perf_counter()
    mpo_spec, build_mpo = tnalg.compile_mpo(
        codes, spec.physical_indices, **compiler_options
    )
    mpo = build_mpo(weights)
    jax.block_until_ready(mpo)
    setup_s = time.perf_counter() - setup_start
    state_start = time.perf_counter()
    initial, _ = tnalg.random_mps(jax.random.key(7), spec=spec, dtype=jnp.complex128)
    jax.block_until_ready(initial)
    random_mps_s = time.perf_counter() - state_start
    print(
        json.dumps(
            {
                "phase": "random_mps",
                "symmetry": arguments.symmetry,
                "N": arguments.length,
                "chi": arguments.chi,
                "random_mps_s": random_mps_s,
            }
        ),
        flush=True,
    )
    export_start = time.perf_counter()
    reference = as_tenpy(initial, model)
    reference_export_s = time.perf_counter() - export_start
    np.testing.assert_array_equal(reference.chi, spec.bond_dims[1:-1])
    initial_energy = None
    initial_checks_s = 0.0
    if not arguments.skip_initial_checks:
        check_start = time.perf_counter()
        initial_energy = float(jnp.real(tnalg.expectation(initial, mpo)))
        np.testing.assert_allclose(
            model.H_MPO.expectation_value(reference), initial_energy, atol=1e-9
        )
        initial_checks_s = time.perf_counter() - check_start
    print(
        json.dumps(
            {
                "symmetry": arguments.symmetry,
                "N": arguments.length,
                "chi": max(reference.chi),
                "same_initial_bonds": True,
                "initial_energy": initial_energy,
                "dtype": str(initial.buffers[0][0].dtype),
                "dt": arguments.dt,
                "repeats": arguments.repeats,
                "jax_version": jax.__version__,
                "tenpy_version": tenpy.__version__,
                "device": str(jax.devices()[0]),
                "max_charge_sectors": (
                    max(len(index.charges) for index in spec.bond_indices)
                    if spec.bond_indices is not None
                    else 1
                ),
                "mpo_bond_dimension": max(mpo_spec.bond_dims),
                "mpo_elements": sum(a.size for a in jax.tree_util.tree_leaves(mpo)),
                "mpo_setup_s": setup_s,
                "reference_export_s": reference_export_s,
                "initial_checks_s": initial_checks_s,
                "initial_checks_skipped": arguments.skip_initial_checks,
            }
        ),
        flush=True,
    )
    state = initial
    dt = jnp.asarray(arguments.dt)
    algorithms = (
        ("tebd", "tdvp", "dmrg")
        if arguments.algorithm == "all"
        else (arguments.algorithm,)
    )
    for algorithm in algorithms:
        krylov = (
            arguments.dmrg_krylov_dim if algorithm == "dmrg" else arguments.krylov_dim
        )
        lanczos = {
            "N_min": krylov,
            "N_max": krylov,
            "reortho": True,
            "P_tol": 0.0,
            "E_tol": 0.0,
        }
        if algorithm == "dmrg":
            lanczos.update(N_min=2, P_tol=1e-20, E_tol=1e-13)
        if algorithm == "tebd":
            gate_spec, build_gates = tnalg.compile_tebd_gates(
                codes, spec.physical_indices, **compiler_options
            )
            function = tnalg.make_tebd_step(spec, gate_spec, tnalg.TEBDOptions())
            inputs = (state, build_gates(weights), dt)

            def make_engine():
                return tebd.TEBDEngine(
                    reference.copy(),
                    model,
                    {
                        "dt": arguments.dt,
                        "N_steps": 1,
                        "order": 2,
                        "trunc_params": {
                            "chi_max": arguments.chi,
                            "svd_min": np.finfo(float).tiny,
                            "trunc_cut": None,
                        },
                    },
                )

        elif algorithm == "tdvp":
            function = tnalg.make_tdvp_step(
                spec, mpo_spec, tnalg.TDVPOptions(krylov_dim=krylov)
            )
            inputs = (state, mpo, dt)

            def make_engine():
                return SingleSiteTDVPEngine(
                    reference.copy(),
                    model,
                    {"dt": arguments.dt, "N_steps": 1, "lanczos_params": dict(lanczos)},
                )

        else:
            function = tnalg.make_dmrg_sweep(
                spec, mpo_spec, tnalg.DMRGOptions(krylov_dim=krylov)
            )
            inputs = (initial, mpo)

            def make_engine():
                engine = dmrg.SingleSiteDMRGEngine(
                    reference.copy(),
                    model,
                    {
                        "diag_method": "lanczos",
                        "lanczos_params": dict(lanczos),
                        "trunc_params": {
                            "chi_max": arguments.chi,
                            "svd_min": np.finfo(float).tiny,
                            "trunc_cut": None,
                        },
                    },
                )
                assert engine.mixer is None
                return engine

        record = {"algorithm": algorithm}
        if algorithm != "tebd":
            record["krylov_dim"] = krylov
        if algorithm == "dmrg":
            record["tenpy_lanczos"] = lanczos
        start = time.perf_counter()
        lowered = jax.jit(function).lower(*inputs)
        record["lower_s"] = time.perf_counter() - start
        print(json.dumps(record), flush=True)
        start = time.perf_counter()
        compiled = lowered.compile()
        record["compile_s"] = time.perf_counter() - start
        print(json.dumps(record), flush=True)
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(*inputs))
        record["first_s"] = time.perf_counter() - start
        print(json.dumps(record), flush=True)
        start = time.perf_counter()
        for _ in range(arguments.repeats):
            result = jax.block_until_ready(compiled(*inputs))
        record["tnalg_s"] = (time.perf_counter() - start) / arguments.repeats
        print(json.dumps(record), flush=True)
        engines = [make_engine() for _ in range(arguments.repeats)]
        start = time.perf_counter()
        for engine in engines:
            if algorithm == "dmrg":
                engine.sweep()
            else:
                engine.run()
        record["tenpy_s"] = (time.perf_counter() - start) / arguments.repeats
        record["speedup"] = record["tenpy_s"] / record["tnalg_s"]
        final = result[0] if algorithm == "dmrg" else result
        exported = as_tenpy(final, model)
        reference_final = engines[-1].psi
        record["tnalg_energy"] = float(np.real(model.H_MPO.expectation_value(exported)))
        record["tenpy_energy"] = float(
            np.real(model.H_MPO.expectation_value(reference_final))
        )
        record["energy_error"] = abs(record["tnalg_energy"] - record["tenpy_energy"])
        norm_product = abs(
            exported.overlap(exported) * reference_final.overlap(reference_final)
        )
        record["infidelity"] = float(
            max(0.0, 1 - abs(exported.overlap(reference_final)) ** 2 / norm_product)
        )
        record["tenpy_final_chi"] = max(engines[-1].psi.chi)
        record["same_final_bonds"] = tuple(reference_final.chi) == spec.bond_dims[1:-1]
        print(json.dumps(record), flush=True)
        if arguments.energy_atol is not None:
            np.testing.assert_allclose(
                record["tnalg_energy"],
                record["tenpy_energy"],
                atol=arguments.energy_atol,
                rtol=0,
            )
            np.testing.assert_allclose(
                record["infidelity"], 0.0, atol=arguments.infidelity_atol, rtol=0
            )


if __name__ == "__main__":
    main()
