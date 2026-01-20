"""
1D TEBD on imaginary time toward ground state with OBC
"""

import time
from functools import partial
import numpy as np
import scipy
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex128")


@partial(K.jit, static_argnums=(8, 9))
def apply_trotter_step(
    mps_tensors, hxx, hyy, hzz, hx, hy, hz, dt_step, nqubits, bond_dim
):
    split_rules = {"max_singular_values": bond_dim}
    mps_circuit = tc.MPSCircuit(nqubits, tensors=mps_tensors, split=split_rules)
    # Apply odd bonds (1-2, 3-4, ...)

    for i in range(0, nqubits - 1, 2):
        mps_circuit.rxx(i, (i + 1), theta=-2.0j * hxx * dt_step)
        mps_circuit.ryy(i, (i + 1), theta=-2.0j * hyy * dt_step)
        mps_circuit.rzz(i, (i + 1), theta=-2.0j * hzz * dt_step)
    mps_circuit._mps.position(nqubits - 1, normalize=True)

    # Apply even bonds (2-3, 4-5, ...)
    for i in reversed(range(1, nqubits - 1, 2)):
        mps_circuit.rxx(i, (i + 1), theta=-2.0j * hxx * dt_step)
        mps_circuit.ryy(i, (i + 1), theta=-2.0j * hyy * dt_step)
        mps_circuit.rzz(i, (i + 1), theta=-2.0j * hzz * dt_step)
    mps_circuit._mps.position(0, normalize=True)

    for i in range(nqubits):
        mps_circuit.rx(i, theta=-2.0j * hx * dt_step)
        mps_circuit.ry(i, theta=-2.0j * hy * dt_step)
        mps_circuit.rz(i, theta=-2.0j * hz * dt_step)
    # mps_circuit._mps.position(0, normalize=True)

    return mps_circuit._mps.tensors


def heisenberg_imag_time_evolution_mps(
    nqubits: int,
    total_time: float,
    dt: float,
    hxx: float = 1.0,
    hyy: float = 1.0,
    hzz: float = 1.0,
    hz: float = 0.0,
    hx: float = 0.0,
    hy: float = 0.0,
    initial_state=None,
    split_rules=None,
):

    # Initialize MPS circuit
    if initial_state is not None:
        mps = tc.MPSCircuit(nqubits, wavefunction=initial_state, split=split_rules)
    else:
        mps = tc.MPSCircuit(nqubits, split=split_rules)
    tensors = mps._mps.tensors

    # Number of Trotter steps
    nsteps = int(total_time / dt)
    dt_step = dt

    # Perform time evolution
    for _ in range(nsteps):
        tensors = apply_trotter_step(
            tensors,
            hxx,
            hyy,
            hzz,
            hx,
            hy,
            hz,
            dt_step,
            nqubits,
            split_rules["max_singular_values"],
        )

    return tc.MPSCircuit(nqubits, tensors=tensors, split=split_rules)


def compare_baseline(nqubits=12):
    # Parameters
    total_time = 6
    dt = 0.02

    # Heisenberg parameters
    hxx = 0.8
    hyy = 1.0
    hzz = 2.0
    hz = 0.01
    hy = 0.0
    hx = 0.0

    split_rules = {"max_singular_values": 24}

    c = tc.Circuit(nqubits)
    c.x([2 * i for i in range(nqubits // 2)])
    initial_state = c.state()

    # TEBD evolution
    final_mps = heisenberg_imag_time_evolution_mps(
        nqubits=nqubits,
        total_time=total_time,
        dt=dt,
        hxx=hxx,
        hyy=hyy,
        hzz=hzz,
        hz=hz,
        hx=hx,
        hy=hy,
        initial_state=initial_state,
        split_rules=split_rules,
    )
    # Exact evolution
    g = tc.templates.graphs.Line1D(nqubits, pbc=False)
    H = tc.quantum.heisenberg_hamiltonian(
        g, hxx=hxx, hyy=hyy, hzz=hzz, hz=hz, hy=hy, hx=hx, sparse=False
    )
    U = scipy.linalg.expm(-total_time * H)
    exact_final = K.reshape(U @ K.reshape(initial_state, [-1, 1]), [-1])
    exact_final /= K.norm(exact_final)
    # Compare results
    mps_state = final_mps.wavefunction()
    fidelity = np.abs(np.vdot(exact_final, mps_state)) ** 2
    print(f"Fidelity between TEBD and exact evolution: {fidelity}")
    c_exact = tc.Circuit(nqubits, inputs=exact_final)
    # Measure observables
    z_magnetization_mps = []
    z_magnetization_exact = []
    for i in range(nqubits):
        mag_mps = final_mps.expectation((tc.gates.z(), [i]))
        z_magnetization_mps.append(mag_mps)

        mag_exact = c_exact.expectation((tc.gates.z(), [i]))
        z_magnetization_exact.append(mag_exact)

    print("MPS Z magnetization:", K.stack(z_magnetization_mps))
    print("Exact Z magnetization:", K.stack(z_magnetization_exact))
    print("Final bond dimensions:", final_mps.get_bond_dimensions())

    return final_mps, exact_final


def benchmark_efficiency(nqubits, bond_d):
    total_time = 0.2
    dt = 0.01
    hxx = 0.9
    hyy = 1.0
    hzz = 0.3
    split_rules = {"max_singular_values": bond_d}

    # TEBD evolution
    time0 = time.time()
    final_mps = heisenberg_imag_time_evolution_mps(
        nqubits=nqubits,
        total_time=total_time,
        dt=dt,
        hxx=hxx,
        hyy=hyy,
        hzz=hzz,
        split_rules=split_rules,
    )
    print(final_mps._mps.tensors[0])
    print("1 step cold start run:", (time.time() - time0) / 20)
    time0 = time.time()
    final_mps = heisenberg_imag_time_evolution_mps(
        nqubits=nqubits,
        total_time=total_time,
        dt=dt,
        hxx=hxx,
        hyy=hyy,
        hzz=hzz,
        split_rules=split_rules,
    )
    print(final_mps._mps.tensors[0])
    print("1 step jitted run:", (time.time() - time0) / 20)


if __name__ == "__main__":
    compare_baseline()
    benchmark_efficiency(32, 32)
