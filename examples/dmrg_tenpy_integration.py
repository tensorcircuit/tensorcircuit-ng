"""
Demonstration of TeNPy-DMRG and TensorCircuit integration
1. Compute ground state (MPS) of 1D Heisenberg model using TeNPy
2. Convert MPS to TensorCircuit's QuOperator
3. Initialize MPSCircuit with converted state and verify results
"""

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import dmrg
import tensornetwork as tn

import tensorcircuit as tc

Node = tn.Node
Edge = tn.Edge
connect = tn.connect

QuOperator = tc.quantum.QuOperator
quantum_constructor = tc.quantum.quantum_constructor

print("1. Setting up the Heisenberg model in TeNPy using SpinChain...")
L = 10
model_params = {
    "L": L,
    "bc_MPS": "finite",
    "Jxx": 1.0,
    "Jz": 0.5,
    "hz": 0.1,
}
M = XXZChain(model_params)

print("\n2. Running DMRG in TeNPy to find the ground state...")
product_state = ["up", "down"] * (L // 2)
print(f"   - Initializing DMRG from the Neel state: {product_state}")

print(f"   - Initializing DMRG from a random product state: {product_state}")

psi0 = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
dmrg_params = {
    "mixer": True,
    "max_E_err": 1.0e-10,
    "trunc_params": {
        "chi_max": 100,
        "svd_min": 1.0e-10,
    },
    "max_sweeps": 10,
}
eng = dmrg.TwoSiteDMRGEngine(psi0, M, dmrg_params)
E, psi_tenpy = eng.run()
print(f"   - DMRG finished. Ground state energy: {E:.10f}")

print("\n3. Converting TeNPy MPS to TensorCircuit QuOperator...")
tc_mps_state = tc.quantum.tenpy2qop(psi_tenpy)
print("   - Conversion successful.")

print("\n4. Initializing TensorCircuit MPSCircuit with the converted state...")
circuit = tc.MPSCircuit(L, wavefunction=tc_mps_state)
print("   - MPSCircuit initialized.")

print("\n5. Verification: Comparing expectation values...")

mag_z_tenpy = psi_tenpy.expectation_value("Sz")
avg_mag_z_tenpy = np.mean(mag_z_tenpy)
print(f"   - TeNPy: Average magnetization <Sz> = {avg_mag_z_tenpy:.10f}")

mag_z_tc_raw = [circuit.expectation((tc.gates.z(), [i])) for i in range(L)]
mag_z_tc_raw = tc.backend.real(tc.backend.stack(mag_z_tc_raw))
mag_z_tc_avg = tc.backend.numpy(mag_z_tc_raw) / 2.0
avg_mag_z_tc = np.mean(mag_z_tc_avg)
print(f"   - TensorCircuit: Average magnetization <Sz> = {avg_mag_z_tc:.10f}")

np.testing.assert_allclose(avg_mag_z_tenpy, avg_mag_z_tc, atol=1e-5)
print("\n[SUCCESS] Average magnetization matches between TeNPy and TensorCircuit.")

mag_z_tc = mag_z_tc_avg
print("\nComparing site-by-site magnetization:")
print("TeNPy:", mag_z_tenpy)
print("TC:   ", mag_z_tc)
np.testing.assert_allclose(mag_z_tenpy, mag_z_tc, atol=1e-5)
print("\n[SUCCESS] Site-by-site magnetization matches perfectly.")
print("\nWorkflow demonstration complete!")
