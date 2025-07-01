"""
Stabilizer Circuit Benchmark
"""

from time import time
import numpy as np
import tensorcircuit as tc

tc.set_dtype("complex128")

clifford_one_qubit_gates = ["H", "X", "Y", "Z", "S", "SD"]
clifford_two_qubit_gates = ["CNOT", "CZ", "SWAP"]
clifford_gates = clifford_one_qubit_gates + clifford_two_qubit_gates


def genpair(num_qubits, count):
    choice = list(range(num_qubits))
    for _ in range(count):
        np.random.shuffle(choice)
        x, y = choice[:2]
        yield (x, y)


def random_clifford_circuit(num_qubits, depth):
    c = tc.Circuit(num_qubits)
    for _ in range(depth):
        for j, k in genpair(num_qubits, num_qubits // 2):
            gate_name = np.random.choice(clifford_two_qubit_gates)
            getattr(c, gate_name)(j, k)
        for j in range(num_qubits):
            gate_name = np.random.choice(clifford_one_qubit_gates)
            getattr(c, gate_name)(j)
    return c


if __name__ == "__main__":
    time_cir_gen = 0
    time_cir_state = 0
    time_cir_ee = 0
    time_scir_gen = 0
    time_scir_ee = 0
    for _ in range(30):
        t0 = time()
        c = random_clifford_circuit(10, 15)
        time_cir_gen += time() - t0
        t0 = time()
        s = c.state()
        time_cir_state += time() - t0
        t0 = time()
        ee0 = tc.quantum.entanglement_entropy(s, list(range(5)))
        time_cir_ee += time() - t0
        t0 = time()
        c1 = tc.StabilizerCircuit.from_qir(c.to_qir())
        time_scir_gen += time() - t0
        t0 = time()
        ee1 = c1.entanglement_entropy(list(range(5)))
        time_scir_ee += time() - t0
        np.testing.assert_allclose(
            ee0,
            ee1,
            atol=1e-6,
        )

    print("time_cir_gen", time_cir_gen / 30)
    print("time_cir_state", time_cir_state / 30)
    print("time_cir_ee", time_cir_ee / 30)
    print("time_scir_gen", time_scir_gen / 30)
    print("time_scir_ee", time_scir_ee / 30)
