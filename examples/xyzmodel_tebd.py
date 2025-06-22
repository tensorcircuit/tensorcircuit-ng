"""
1D TEBD using MPSCircuit
"""

import time
import numpy as np
import scipy
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex128")


def heisenberg_time_evolution_mps(
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

    @K.jit
    def apply_trotter_step(mps_tensors):
        mps_circuit = tc.MPSCircuit(nqubits, tensors=mps_tensors, split=split_rules)
        # Apply odd bonds (1-2, 3-4, ...)

        for i in range(0, nqubits, 2):
            mps_circuit.rxx(i, (i + 1) % nqubits, theta=hxx * dt_step)
            mps_circuit.ryy(i, (i + 1) % nqubits, theta=hyy * dt_step)
            mps_circuit.rzz(i, (i + 1) % nqubits, theta=hzz * dt_step)

        # Apply even bonds (2-3, 4-5, ...)
        for i in range(1, nqubits, 2):
            mps_circuit.rxx(i, (i + 1) % nqubits, theta=2 * hxx * dt_step)
            mps_circuit.ryy(i, (i + 1) % nqubits, theta=2 * hyy * dt_step)
            mps_circuit.rzz(i, (i + 1) % nqubits, theta=2 * hzz * dt_step)

            # mps_circuit.unitary(i, unitary=unitary)

        for i in range(0, nqubits, 2):
            mps_circuit.rxx(i, (i + 1) % nqubits, theta=hxx * dt_step)
            mps_circuit.ryy(i, (i + 1) % nqubits, theta=hyy * dt_step)
            mps_circuit.rzz(i, (i + 1) % nqubits, theta=hzz * dt_step)

        for i in range(nqubits):
            mps_circuit.rx(i, theta=2 * hx * dt_step)
            mps_circuit.ry(i, theta=2 * hy * dt_step)
            mps_circuit.rz(i, theta=2 * hz * dt_step)

        return mps_circuit._mps.tensors

    # Perform time evolution
    for step in range(nsteps):
        tensors = apply_trotter_step(tensors)

    return tc.MPSCircuit(nqubits, tensors=tensors, split=split_rules)


def compare_baseline():
    # Parameters
    nqubits = 10
    total_time = 2
    dt = 0.01

    # Heisenberg parameters
    hxx = 0.9
    hyy = 1.0
    hzz = 0.3
    hz = -0.1
    hy = 0.16
    hx = 0.43

    split_rules = {"max_singular_values": 32}

    c = tc.Circuit(nqubits)
    c.x(nqubits // 2)
    initial_state = c.state()

    # TEBD evolution
    final_mps = heisenberg_time_evolution_mps(
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
    g = tc.templates.graphs.Line1D(nqubits, pbc=True)
    H = tc.quantum.heisenberg_hamiltonian(
        g, hxx=hxx, hyy=hyy, hzz=hzz, hz=hz, hy=hy, hx=hx, sparse=False
    )
    U = scipy.linalg.expm(-1j * total_time * H)
    exact_final = K.reshape(U @ K.reshape(initial_state, [-1, 1]), [-1])

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
    total_time = 0.1
    dt = 0.01
    hxx = 0.9
    hyy = 1.0
    hzz = 0.3
    split_rules = {"max_singular_values": bond_d}

    # TEBD evolution
    time0 = time.time()
    final_mps = heisenberg_time_evolution_mps(
        nqubits=nqubits,
        total_time=total_time,
        dt=dt,
        hxx=hxx,
        hyy=hyy,
        hzz=hzz,
        split_rules=split_rules,
    )
    print(final_mps._mps.tensors[0])
    print("cold start run:", time.time() - time0)
    time0 = time.time()
    final_mps = heisenberg_time_evolution_mps(
        nqubits=nqubits,
        total_time=total_time,
        dt=dt,
        hxx=hxx,
        hyy=hyy,
        hzz=hzz,
        split_rules=split_rules,
    )
    print(final_mps._mps.tensors[0])
    print("jitted run:", time.time() - time0)


if __name__ == "__main__":
    compare_baseline()
    benchmark_efficiency(24, 48)
