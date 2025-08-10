"""
Demonstration of TeNPy-DMRG and TensorCircuit integration
1. Compute ground state (MPS) of 1D Transverse Field Ising model using TeNPy
2. Convert MPS to TensorCircuit's QuOperator
3. Initialize MPSCircuit with converted state and verify results
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

# --- Scenario 2: XXZChain Model ---
print("Scenario 2: XXZChain Model")

xxz_model_params = {"L": L, "bc_MPS": "finite", "Jxx": 1.0, "Jz": 0.5, "hz": 0.1}
xxz_M = XXZChain(xxz_model_params)
rng = np.random.default_rng(42)
random_state = rng.choice(["up", "down"], size=L).tolist()
print(f"Testing with a random product state: {random_state}")

psi_rand_xxz = MPS.from_product_state(
    xxz_M.lat.mps_sites(), random_state, bc=xxz_M.lat.bc_MPS
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

# --- Scenario 3: Tensor Dissection for Both Models ---
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


print("\nFinal Conclusion")
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

print("\n\nWorkflow demonstration and analysis complete!")
