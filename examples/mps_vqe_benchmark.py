"""
benchmark MPSCircuit on tf and jax backend
"""

import time
import numpy as np
import tensorcircuit as tc


def Hamiltonian(c: tc.MPSCircuit, n: int):
    e = 0.0
    for i in range(n):
        e += -c.expectation_ps(z=[i])
    return -tc.backend.real(e)


def vqe(params, n):
    circuit = tc.MPSCircuit(n)
    circuit.set_split_rules({"max_singular_values": 32})

    for i in range(n):
        circuit.rx(i, theta=params[i][0])
        circuit.ry(i, theta=params[i][1])
        circuit.rz(i, theta=params[i][2])
    for i in range(n - 1):
        circuit.cx(i, i + 1)

    energy = Hamiltonian(circuit, n)
    return energy


if __name__ == "__main__":
    batch = 16
    n = 12
    maxiter = 100
    params0 = np.random.uniform(size=[batch, n, 3])

    for b in ["tensorflow", "jax"]:
        with tc.runtime_backend(b):
            vqe_vvag = tc.backend.jit(
                tc.backend.vectorized_value_and_grad(vqe, vectorized_argnums=(0,)),
                static_argnums=(1,),
            )
            print("benchmarking backend: %s" % b)
            time0 = time.time()
            params = tc.backend.convert_to_tensor(params0)
            energy, grad = vqe_vvag(params, n)
            print(energy[0], grad[0, 0])
            print("jit time", time.time() - time0)
            time0 = time.time()
            for _ in range(5):
                energy, grad = vqe_vvag(params, n)
            print("running time", (time.time() - time0) / 5)
