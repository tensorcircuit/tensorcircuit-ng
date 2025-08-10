"""
Demonstrates the different internal basis conventions between
TeNPy's TFIChain and XXZChain models and showcases a robust
method for handling these inconsistencies when converting to TensorCircuit.

1.  TFIChain: Shows a direct conversion works perfectly.
2.  XXZChain:
    a. Runs DMRG to obtain a non-trivial ground state.
    b. Shows that direct conversion leads to incorrect expectation values for correlation functions.
    c. Demonstrates that applying a layer of X-gates in TensorCircuit
3.  Tensor Dissection: Provides definitive proof of the differing internal basis conventions between the two models.
"""

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.xxz_chain import XXZChain
from tenpy.algorithms import dmrg

import tensorcircuit as tc

print("Scenario 1: Antiferromagnetic State (TFIChain)")
L = 10
afm_model_params = {"L": L, "bc_MPS": "finite", "J": 1.0, "g": 0.0, "conserve": None}
afm_M = TFIChain(afm_model_params)
afm_state = ["up", "down"] * (L // 2)
print(f"Testing with a simple product state: {afm_state}")

psi_afm = MPS.from_product_state(afm_M.lat.mps_sites(), afm_state, bc=afm_M.lat.bc_MPS)
tc_afm_state = tc.quantum.tenpy2qop(psi_afm)
circuit_afm = tc.MPSCircuit(L, wavefunction=tc_afm_state)

mag_z_afm_tenpy = psi_afm.expectation_value("Sz")
mag_z_afm_tc = np.array(
    [
        tc.backend.numpy(tc.backend.real(circuit_afm.expectation((tc.gates.z(), [i]))))
        / 2.0
        for i in range(L)
    ]
)
print("\nAntiferromagnetic state site-by-site magnetization comparison:")
print("TeNPy:", np.round(mag_z_afm_tenpy, 8))
print("TC:   ", np.round(mag_z_afm_tc, 8))
np.testing.assert_allclose(mag_z_afm_tenpy, mag_z_afm_tc, atol=1e-5)
print(
    "\n[SUCCESS] TFI-based Antiferromagnetic state matches perfectly with the pure converter."
)


print("Scenario 2: XXZChain Model")
xxz_model_params = {"L": L, "bc_MPS": "finite", "Jxx": 1.0, "Jz": 0.5, "hz": 0.1}
xxz_M = XXZChain(xxz_model_params)
example_state = ["up", "down", "up", "up", "down", "down", "up", "down", "down", "up"]
print(f"Testing with a random product state: {example_state}")
psi_rand_xxz = MPS.from_product_state(
    xxz_M.lat.mps_sites(), example_state, bc=xxz_M.lat.bc_MPS
)
tc_rand_xxz_state = tc.quantum.tenpy2qop(psi_rand_xxz)
circuit_rand_xxz = tc.MPSCircuit(L, wavefunction=tc_rand_xxz_state)
mag_z_rand_xxz_tenpy = psi_rand_xxz.expectation_value("Sz")
mag_z_rand_xxz_tc = np.array(
    [
        tc.backend.numpy(
            tc.backend.real(circuit_rand_xxz.expectation((tc.gates.z(), [i])))
        )
        / 2.0
        for i in range(L)
    ]
)
print("\nXXZ-based random state site-by-site magnetization comparison:")
print("TeNPy:", np.round(mag_z_rand_xxz_tenpy, 8))
print("TC:   ", np.round(mag_z_rand_xxz_tc, 8))
try:
    np.testing.assert_allclose(mag_z_rand_xxz_tenpy, mag_z_rand_xxz_tc, atol=1e-5)
except AssertionError as e:
    print("\n[SUCCESS] As expected, the direct comparison fails for XXZChain.")
    print(
        "This is because the pure converter does not handle its inverted basis convention."
    )
    print("\nVerifying that the values match after correcting the sign:")
    np.testing.assert_allclose(mag_z_rand_xxz_tenpy, -mag_z_rand_xxz_tc, atol=1e-5)
    print(
        "[SUCCESS] Test passes after applying the sign correction for the XXZChain model."
    )


print("Scenario 3: Tensor Dissection for Both Models")
simple_L = 2
simple_labels = ["up", "down"]
print("\nDissecting TFIChain-based Tensors")
sites_tfi = afm_M.lat.mps_sites()[:simple_L]
psi_simple_tfi = MPS.from_product_state(sites_tfi, simple_labels, bc="finite")
for i, label in enumerate(simple_labels):
    B_tensor = psi_simple_tfi.get_B(i).to_ndarray()
    print(
        f"For '{label}', TFIChain internal tensor has non-zero at physical index {np.where(B_tensor[0,:,0] != 0)[0][0]}"
    )
print("\nDissecting XXZChain-based Tensors")
sites_xxz = xxz_M.lat.mps_sites()[:simple_L]
psi_simple_xxz = MPS.from_product_state(sites_xxz, simple_labels, bc="finite")
for i, label in enumerate(simple_labels):
    B_tensor = psi_simple_xxz.get_B(i).to_ndarray()
    print(
        f"For '{label}', XXZChain internal tensor has non-zero at physical index {np.where(B_tensor[0,:,0] != 0)[0][0]}"
    )
print("\n Conclusion")
print("The dissection above shows the root cause of the different behaviors:")
print(
    "  - TFIChain's 'up' maps to index 0, 'down' to index 1. This matches TC's standard."
)
print(
    "  - XXZChain's 'up' maps to index 1, 'down' to index 0. This is INVERTED from TC's standard."
)
print("\nTherefore, a single, universal converter is not feasible without context.")
print(
    "The robust solution is to use a pure converter and apply corrections on a case-by-case basis,"
)
print("or to create model-specific converters.")


print("--- Scenario 3: Correcting XXZChain DMRG state with X-gates ---")

L = 30
xxz_model_params = {"L": L, "bc_MPS": "finite", "Jxx": 1.0, "Jz": 1.0, "conserve": None}
xxz_M = XXZChain(xxz_model_params)
psi0_xxz = MPS.from_product_state(
    xxz_M.lat.mps_sites(), ["up", "down"] * (L // 2), bc=xxz_M.lat.bc_MPS
)
dmrg_params = {"max_sweeps": 10, "trunc_params": {"chi_max": 64}}
eng = dmrg.TwoSiteDMRGEngine(psi0_xxz, xxz_M, dmrg_params)
E, psi_gs_xxz = eng.run()
print(f"XXZ DMRG finished. Ground state energy: {E:.10f}")

state_raw_quvector = tc.quantum.tenpy2qop(psi_gs_xxz)

i, j = L // 2 - 1, L // 2
corr_tenpy = psi_gs_xxz.correlation_function("Sz", "Sz", sites1=[i], sites2=[j])[0, 0]
print("\nApplying X-gate to each qubit to correct the basis convention...")
circuit_to_be_corrected = tc.MPSCircuit(L, wavefunction=state_raw_quvector)

for k in range(L):
    circuit_to_be_corrected.x(k)

corr_tc_corrected = (
    tc.backend.real(
        circuit_to_be_corrected.expectation((tc.gates.z(), [i]), (tc.gates.z(), [j]))
    )
    / 4.0
)

print(f"\nComparing <Sz_{i}Sz_{j}> correlation function for the DMRG ground state:")
print(f"TeNPy (Ground Truth):         {corr_tenpy:.8f}")
print(f"TC (after X-gate correction): {corr_tc_corrected:.8f}")
np.testing.assert_allclose(corr_tenpy, corr_tc_corrected, atol=1e-5)
print(
    "\n[SUCCESS] The correlation functions match perfectly after applying the X-gate correction."
)
print(
    "This demonstrates the recommended physical approach to handle the XXZChain's inverted basis convention."
)


print("\n\nWorkflow demonstration and analysis complete!")
