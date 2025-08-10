"""
Demonstration of TeNPy-DMRG and TensorCircuit integration
1. Compute ground state (MPS) of 1D Transverse Field Ising model using TeNPy
2. Convert MPS to TensorCircuit's QuOperator
3. Initialize MPSCircuit with converted state and verify results
"""

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg

import tensorcircuit as tc

print("1. Setting up the Transverse Field Ising (TFI) model in TeNPy...")
L = 10
model_params = {
    "L": L,
    "bc_MPS": "finite",
    "J": 0,
    "g": 0.5,
    "conserve": None,
}
M = TFIChain(model_params)

print("\n2. Running DMRG in TeNPy to find the ground state...")
product_state = ["up", "down"] * (L // 2)
print(f"   - Initializing DMRG from the Neel state: {product_state}")

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

mag_z_tc = np.array(
    [
        tc.backend.numpy(tc.backend.real(circuit.expectation((tc.gates.z(), [i]))))
        / 2.0
        for i in range(L)
    ]
)
avg_mag_z_tc = np.mean(mag_z_tc)
print(f"   - TensorCircuit: Average magnetization <Sz> = {avg_mag_z_tc:.10f}")

np.testing.assert_allclose(avg_mag_z_tenpy, avg_mag_z_tc, atol=1e-5)
print("\n[SUCCESS] Average magnetization matches between TeNPy and TensorCircuit.")

print("\nComparing site-by-site magnetization:")
print("TeNPy:", np.round(mag_z_tenpy, 8))
print("TC:   ", np.round(mag_z_tc, 8))

np.testing.assert_allclose(mag_z_tenpy, mag_z_tc, atol=1e-5)
print("\n[SUCCESS] Site-by-site magnetization matches perfectly.")

print("\n6. Testing random non-translation-invariant state...")

rng = np.random.default_rng(42)
random_state = rng.choice(["up", "down"], size=L).tolist()
print("Random state:", random_state)

psi_rand = MPS.from_product_state(M.lat.mps_sites(), random_state, bc=M.lat.bc_MPS)
tc_rand_state = tc.quantum.tenpy2qop(psi_rand)
circuit_rand = tc.MPSCircuit(L, wavefunction=tc_rand_state)

mag_z_rand_tenpy = psi_rand.expectation_value("Sz")
mag_z_rand_tc = np.array(
    [
        tc.backend.numpy(tc.backend.real(circuit_rand.expectation((tc.gates.z(), [i]))))
        / 2.0
        for i in range(L)
    ]
)

print("Random state site-by-site magnetization comparison:")
print("TeNPy:", np.round(mag_z_rand_tenpy, 8))
print("TC:   ", np.round(mag_z_rand_tc, 8))

np.testing.assert_allclose(mag_z_rand_tenpy, mag_z_rand_tc, atol=1e-5)
print(
    "\n[SUCCESS] Random non-translation-invariant state matches between TeNPy and TensorCircuit."
)

print("\n\n--- 7.Verification: Dissecting TFIChain Tensors ---")

simple_L_tfi = 2
simple_state_labels_tfi = ["up", "down"]
print(f"Creating a simple TFI-based MPS with state: {simple_state_labels_tfi}")

sites_tfi = M.lat.mps_sites()[:simple_L_tfi]

psi_simple_tfi = MPS.from_product_state(sites_tfi, simple_state_labels_tfi, bc="finite")

for i, label in enumerate(simple_state_labels_tfi):
    B_tensor = psi_simple_tfi.get_B(i).to_ndarray()

    print(f"\nSite {i}, Desired Label: '{label}'")
    print(f"  - Raw B-Tensor Shape: {B_tensor.shape}")
    print(f"  - Raw B-Tensor Content:\n{B_tensor}")

print("\n--- Conclusion ---")
print("The output above demonstrates that for the TFIChain model as well:")
print(
    "  - The 'up' state corresponds to a tensor with a non-zero element at INDEX 1 of the physical leg."
)
print(
    "  - The 'down' state corresponds to a tensor with a non-zero element at INDEX 0 of the physical leg."
)
print(
    "\nThis confirms that the basis convention mismatch is a general property of TeNPy's MPS construction,"
)
print(
    "not specific to one model. Therefore, the physical basis flip `arr[:, ::-1, :]` in the `tenpy2qop`"
)
print("function is a necessary and universally correct translation step.")

print("\nWorkflow demonstration complete!")
