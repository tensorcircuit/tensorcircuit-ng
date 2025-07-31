"""
Demonstration of TeNPy-DMRG and TensorCircuit integration
1. Compute ground state (MPS) of 1D Heisenberg model using TeNPy
2. Convert MPS to TensorCircuit's QuOperator
3. Initialize MPSCircuit with converted state and verify results
"""

import numpy as np
import tenpy
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import dmrg

import tensorcircuit as tc


def tenpy2qop(tenpy_mps):
    """
    Convert TeNPy MPS to TensorCircuit QuOperator

    Args:
        tenpy_mps: tenpy.networks.mps.MPS object

    Returns:
        tc.quantum.QuOperator object
    """
    n = len(tenpy_mps.sites)
    tensors = [
        np.transpose(tenpy_mps.get_B(i).to_ndarray(), (1, 0, 2)) for i in range(n)
    ]
    return tc.quantum.QuOperator.from_mps_tensors(tensors)


def main():
    model_params = {
        "L": 10,
        "S": 0.5,
        "Jz": 1.0,
        "Jy": 1.0,
        "Jx": 1.0,
        "bc_MPS": "finite",
    }
    model = XXZChain(model_params)

    dmrg_params = {
        "trunc_params": {
            "chi_max": 100,
            "svd_min": 1e-10,
        },
        "mixer": True,
        "max_sweeps": 20,
    }
    psi = tenpy.networks.mps.MPS.from_lat_product_state(model.lat)
    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, _ = eng.run()

    print(f"TeNPy ground state energy: {E:.10f}")

    tc_mps_state = tenpy2qop(psi)
    n = len(psi.sites)
    circuit = tc.MPSCircuit(n, wavefunction=tc_mps_state)

    tenpy_sz = psi.expectation_value("Sz")

    tc_sz = []
    sz_matrix = tc.backend.convert_to_tensor(np.diag([0.5, -0.5]))
    for i in range(n):
        tc_sz.append(float(circuit.local_expectation(i, sz_matrix)))

    print(np.round(tenpy_sz, 6))
    print(np.round(tc_sz, 6))

    corr_tenpy = psi.correlation_function("Sz", "Sz", sites1=0)
    corr_tc = [
        circuit.two_point_expectation(0, i, sz_matrix, sz_matrix) for i in range(n)
    ]

    print(np.round(corr_tenpy, 6))
    print(np.round(corr_tc, 6))
