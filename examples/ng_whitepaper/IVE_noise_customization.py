"""
Noise Profile Customization and Readout Error Mitigation
========================================================

This script demonstrates the features of quantum circuit noise profile customization
in TensorCircuit in a unified and accessible way. It covers:
1. Global noise configuration (e.g. depolarizing noise on all CNOTs).
2. Site-dependent noise configuration (different noise on different qubits).
3. Custom noise conditions (logic-based noise application).
4. Readout error simulation.
5. Readout error mitigation using standard techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
from tensorcircuit.noisemodel import NoiseConf
from tensorcircuit.results.readout_mitigation import ReadoutMit

# Use JAX backend for performance
tc.set_backend("jax")


# ------------------------------------------------------------------------------
# 1. Setup Wrapper and Circuit
# ------------------------------------------------------------------------------

n = 6
nlayers = 4


def get_circuit(params, n, nlayers):
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    k = 0
    for _ in range(nlayers):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.ry(i, theta=params[k])
            k += 1
    return c


# Fixed random parameters for reproducibility
np.random.seed(42)
params = np.random.uniform(0, 2 * np.pi, size=n * nlayers)

# Calculate ideal expectation value (Z on the middle qubit)
c_ideal = get_circuit(params, n, nlayers)
ideal_val = c_ideal.expectation((tc.gates.z(), [n // 2]))
print(f"Ideal Expectation Value <Z_{n//2}>: {ideal_val:.4f}")

# ------------------------------------------------------------------------------
# 2. Noise Configuration Scenarios
# ------------------------------------------------------------------------------

print("\n--- noise configuration scenarios ---")

# Scenario A: Global Noise
# Apply Depolarizing noise to all CNOT gates
noise_conf_global = NoiseConf()
noise_conf_global.add_noise("cnot", tc.channels.generaldepolarizingchannel(1e-3, 2))

# Scenario B: Site-dependent Noise
# Apply Phase Damping with different rates on different qubits for 'ry' gates
noise_conf_site = NoiseConf()
for i in range(n):
    # Higher index qubits have more noise
    damping_rate = 0.005 * (i + 1)
    noise_conf_site.add_noise(
        "ry", [tc.channels.phasedampingchannel(damping_rate)], qubit=[(i,)]
    )

# Scenario C: Custom Noise Condition
# Apply noise only if the gate is 'ry' and acts on even qubits
noise_conf_custom = NoiseConf()


def even_qubit_rx_condition(d):
    return d["gatef"].n == "ry" and d["index"][0] % 2 == 0


noise_conf_custom.add_noise_by_condition(
    even_qubit_rx_condition, tc.channels.amplitudedampingchannel(0.05, 1.0)
)

# Scenario D: Thermal Relaxation (T1/T2)
noise_conf_thermal = NoiseConf()
t1 = 100
t2 = 80
time_gate = 2
# Apply thermal relaxation to 'ry' gates
noise_conf_thermal.add_noise(
    "ry", tc.channels.thermalrelaxationchannel(t1, t2, time_gate, "ByChoi", 0)
)

# Scenario E: Readout Error
# Here we use simple global readout error.
# Readout error params: [p(0|0), p(1|1)]
readout_error_param = [0.95, 0.92]
noise_conf_readout = NoiseConf()
# Must provide error model for each qubit
noise_conf_readout.add_noise("readout", [readout_error_param] * n)

# ------------------------------------------------------------------------------
# 3. Running Simulations
# ------------------------------------------------------------------------------


def evaluate_with_noise(conf, label):
    # Use c.expectation with noise_conf for cleaner API
    c = get_circuit(params, n, nlayers)
    val = c.expectation(
        (tc.gates.z(), [n // 2]),
        noise_conf=conf,
        nmc=2000,  # number of monte carlo samples
    )
    print(f"{label}: {val:.4f}")
    return val


val_global = evaluate_with_noise(noise_conf_global, "Global CNOT Noise")
val_site = evaluate_with_noise(noise_conf_site, "Site-dependent RY Noise")
val_custom = evaluate_with_noise(noise_conf_custom, "Custom Condition Noise")
val_thermal = evaluate_with_noise(noise_conf_thermal, "Thermal Relaxation")

# For readout error, we typically look at sampling results
c_readout = tc.noisemodel.circuit_with_noise(
    get_circuit(params, n, nlayers), noise_conf_readout
)
# Sample from the noisy circuit
shots = 8192
counts_readout = c_readout.sample(
    batch=shots,
    allow_state=True,
    format="count_dict_bin",
    readout_error=noise_conf_readout.readout_error,
)
# Calculate expectation from counts
exp_val_readout = tc.results.counts.expectation(counts_readout, z=[n // 2])
print(f"Readout Error Only: {exp_val_readout:.4f}")


# ------------------------------------------------------------------------------
# 4. Readout Error Mitigation
# ------------------------------------------------------------------------------

print("\n--- Readout Error Mitigation ---")

# To use mitigation, we need an 'execute' function that returns counts for a list of circuits
# Ideally this runs on real hardware or a noisy simulator.
# Here we simulate the 'device' which has the specific readout error defined above.


def execute_on_noisy_device(circuits, shots=8192):
    results = []
    for c in circuits:
        c_noisy = tc.noisemodel.circuit_with_noise(c, noise_conf_readout)
        counts = c_noisy.sample(
            batch=shots,
            allow_state=True,
            format="count_dict_bin",
            readout_error=noise_conf_readout.readout_error,
        )
        results.append(counts)
    return results


# Initialize Mitigator
mitigator = ReadoutMit(execute_on_noisy_device)

# 1. Calibration
# Calibrate on all qubits. cals_from_system runs calibration circuits on the 'device'
print("Calibrating readout error...")
mitigator.cals_from_system(list(range(n)), shots=8192, method="local")

# 2. Mitigation
print("Applying mitigation...")
# mitigate_counts takes raw counts and applies the inverse operations
mitigated_counts = mitigator.apply_correction(
    counts_readout, list(range(n)), method="inverse"
)

# Calculate expectation from mitigated counts
exp_val_mitigated = tc.results.counts.expectation(mitigated_counts, z=[n // 2])
print(f"Mitigated Expectation: {exp_val_mitigated:.4f}")

# ------------------------------------------------------------------------------
# 5. Visualization
# ------------------------------------------------------------------------------

labels = ["Ideal", "Global", "Site", "Custom", "Thermal", "Readout", "Mitigated"]
values = [
    float(np.real(v))
    for v in [
        ideal_val,
        val_global,
        val_site,
        val_custom,
        val_thermal,
        exp_val_readout,
        exp_val_mitigated,
    ]
]
colors = ["gray", "red", "orange", "yellow", "blue", "purple", "green"]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color=colors, alpha=0.7)
plt.axhline(y=float(np.real(ideal_val)), color="gray", linestyle="--", label="Ideal")
plt.ylabel("Expectation Value <Z>")
plt.title(f"Noise Simulation and Mitigation for {n}-qubit Circuit")
plt.legend()

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("noise_customization_demo.pdf")
print("\nPlot saved to 'noise_customization_demo.pdf'")
