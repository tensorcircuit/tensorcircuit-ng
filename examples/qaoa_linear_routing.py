"""Route a QAOA circuit to a chain and replay it as an exact small MPS."""

import networkx as nx
import numpy as np

import tensorcircuit as tc

n = 10
graph = nx.random_regular_graph(3, n, seed=5)
circuit = tc.Circuit(n)
for q in range(n):
    circuit.h(q)
for gamma, beta in [(0.3, 0.2), (0.5, 0.1)]:
    for i, j in graph.edges():
        circuit.rzz(i, j, theta=2 * gamma)
    for q in range(n):
        circuit.rx(q, theta=2 * beta)

routed, info = tc.compiler.linear_route(circuit)
mps = tc.MPSCircuit.from_qir(
    routed.to_qir(), {"nqubits": n, "split": {"max_singular_values": 32}}
)
logical_state = tc.backend.numpy(circuit.state()).reshape([2] * n)
physical_state = logical_state.transpose(np.argsort(info["final_layout"])).reshape(-1)
routed_state = tc.backend.numpy(routed.state())
np.testing.assert_allclose(routed_state, physical_state, atol=4e-6)
np.testing.assert_allclose(
    tc.backend.numpy(mps.wavefunction()), physical_state, atol=4e-6
)

print("Strategy:", info["strategy"])
print("Inserted SWAPs before fusion:", info["num_swaps"])
print("Two-qubit gates after fusion:", info["num_two_qubit_gates"])
print("Maximum state error:", np.max(np.abs(routed_state - physical_state)))
physical_samples = tc.backend.numpy(mps.sample(batch=5)).astype(int)
print("Logical samples:\n", physical_samples[:, info["final_layout"]])
