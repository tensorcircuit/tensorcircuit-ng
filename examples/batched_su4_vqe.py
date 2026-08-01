"""
Compare conventional and batched SU(4) gate construction in a JAX VQE.

Both VQE objectives use the same digital staircase SU(4) ansatz and evaluate
the TFIM Hamiltonian from local Pauli expectations. The conventional path
constructs every ``su4`` gate separately; the batched path constructs all gate
matrices with ``batched_unitary`` before attaching them to the circuit. Run,
for example,

``python examples/batched_su4_vqe.py --n-qubits 12 --layers 8``.
"""

import argparse
import time

import jax
import numpy as np

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")


def tfim_energy(circuit, n_qubits, coupling=1.0, field=-1.0):
    """Evaluate an open-boundary TFIM Hamiltonian from Pauli expectations."""
    energy = 0.0
    for i in range(n_qubits - 1):
        energy += coupling * K.real(circuit.expectation_ps(z=[i, i + 1]))
    for i in range(n_qubits):
        energy += field * K.real(circuit.expectation_ps(x=[i]))
    return energy


def conventional_ansatz(params, n_qubits, layers):
    """Build the SU(4) ansatz by constructing every gate separately."""
    circuit = tc.Circuit(n_qubits)
    circuit.h(range(n_qubits))
    for layer in range(layers):
        for i in range(n_qubits - 1):
            circuit.su4(i, i + 1, theta=params[layer, i])
    return circuit


def batched_ansatz(params, n_qubits, layers):
    """Build the same ansatz from one batch of dense SU(4) matrices."""
    circuit = tc.Circuit(n_qubits)
    circuit.h(range(n_qubits))
    flat_params = K.reshape(params, (-1, 15))
    matrices = tc.gates.batched_unitary(
        tc.gates.su4,
        vectorized_argnames="theta",
        theta=flat_params,
    )
    offset = 0
    for _ in range(layers):
        for i in range(n_qubits - 1):
            circuit.any(i, i + 1, unitary=matrices[offset], name="su4")
            offset += 1
    return circuit


def make_energy(ansatz, n_qubits, layers):
    """Create the VQE energy function for one ansatz implementation."""

    def energy(params):
        circuit = ansatz(params, n_qubits, layers)
        return tfim_energy(circuit, n_qubits)

    return energy


def block_until_ready(tree):
    """Synchronize every JAX array in a result PyTree."""
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), tree)


def benchmark(name, energy, params, warm_runs):
    """Measure JAX lowering, compilation, first execution, and warm execution."""
    jax.clear_caches()
    value_and_grad = jax.jit(jax.value_and_grad(energy))

    start = time.perf_counter()
    lowered = value_and_grad.lower(params)
    lower_time = time.perf_counter() - start

    start = time.perf_counter()
    executable = lowered.compile()
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    result = block_until_ready(executable(params))
    first_execution = time.perf_counter() - start

    warm_times = []
    for _ in range(warm_runs):
        start = time.perf_counter()
        block_until_ready(executable(params))
        warm_times.append(time.perf_counter() - start)

    return {
        "name": name,
        "result": result,
        "lower_s": lower_time,
        "compile_s": compile_time,
        "first_execution_s": first_execution,
        "cold_total_s": lower_time + compile_time + first_execution,
        "warm_median_ms": 1e3 * float(np.median(warm_times)),
    }


def print_results(results):
    """Print a compact timing comparison."""
    header = (
        f"{'method':<16} {'lower (s)':>10} {'compile (s)':>12} "
        f"{'first run (s)':>14} {'cold total (s)':>15} {'warm (ms)':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['name']:<16} {result['lower_s']:>10.3f} "
            f"{result['compile_s']:>12.3f} "
            f"{result['first_execution_s']:>14.3f} "
            f"{result['cold_total_s']:>15.3f} "
            f"{result['warm_median_ms']:>11.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=12)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--warm-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_qubits < 2:
        raise ValueError("n-qubits must be at least 2")
    if args.layers < 1:
        raise ValueError("layers must be positive")
    if args.warm_runs < 1:
        raise ValueError("warm-runs must be positive")

    device = jax.devices()[0]

    rng = np.random.default_rng(args.seed)
    params = rng.normal(
        scale=0.05,
        size=(args.layers, args.n_qubits - 1, 15),
    ).astype(np.float32)
    params = K.convert_to_tensor(params)

    conventional_energy = make_energy(conventional_ansatz, args.n_qubits, args.layers)
    batched_energy = make_energy(batched_ansatz, args.n_qubits, args.layers)

    conventional = benchmark(
        "conventional", conventional_energy, params, args.warm_runs
    )
    batched = benchmark("batched", batched_energy, params, args.warm_runs)

    conventional_value, conventional_grad = conventional["result"]
    batched_value, batched_grad = batched["result"]
    np.testing.assert_allclose(
        K.numpy(batched_value), K.numpy(conventional_value), atol=1e-5
    )
    np.testing.assert_allclose(
        K.numpy(batched_grad), K.numpy(conventional_grad), atol=1e-5
    )

    print(
        f"JAX device: {device}, contractor: omeco, "
        f"qubits: {args.n_qubits}, "
        f"layers: {args.layers}, SU(4) gates: {args.layers * (args.n_qubits - 1)}"
    )
    print(f"VQE energy: {float(batched_value):.8f}")
    print_results([conventional, batched])
    print(
        "Speedup: "
        f"{conventional['cold_total_s'] / batched['cold_total_s']:.2f}x cold, "
        f"{conventional['warm_median_ms'] / batched['warm_median_ms']:.2f}x warm"
    )


if __name__ == "__main__":
    main()
