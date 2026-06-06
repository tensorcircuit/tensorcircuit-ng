"""
Benchmark TFIM VQE value and gradient calculation between TensorCircuit (JAX) and MindQuantum.
"""

import argparse
import time

import numpy as np

try:
    import jax
except ImportError:
    jax = None

try:
    import tensorcircuit as tc
except ImportError:
    tc = None

try:
    import mindquantum as mq
    from mindquantum.core.operators import QubitOperator
    from mindquantum.core.circuit import Circuit
    from mindquantum.simulator import Simulator
except ImportError:
    mq = None

PARAM_INIT = 0.1
OMECO_TRIALS = 16
OMECO_ITERS = 24
OMECO_BETAS = np.geomspace(0.1, 10.0, OMECO_ITERS).tolist()


def parse_args():
    parser = argparse.ArgumentParser(
        description="TFIM VQE Benchmark: TensorCircuit JAX vs MindQuantum"
    )
    parser.add_argument("--nqubits", type=int, default=14, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=10, help="Depth of ansatz layers")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["tc-jax", "mindquantum"],
        required=True,
        help="Framework backend to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        required=True,
        help="Device to run on",
    )
    parser.add_argument(
        "--tc-dtype",
        type=str,
        choices=["complex64", "complex128"],
        default="complex64",
        help="Data type precision",
    )
    return parser.parse_args()


def configure_tc_contractor() -> None:
    import omeco

    score = omeco.ScoreFunction(
        tc_weight=1.0,
        sc_weight=0.0,
        rw_weight=64.0,
        sc_target=20.0,
    )
    optimizer = omeco.TreeSA(
        ntrials=OMECO_TRIALS,
        niters=OMECO_ITERS,
        betas=OMECO_BETAS,
        score=score,
    )
    tc.set_contractor("custom", optimizer=optimizer, preprocessing=True)
    print(
        f"[TC-JAX] Contractor: OMECo TreeSA "
        f"(ntrials={OMECO_TRIALS}, niters={OMECO_ITERS})",
        flush=True,
    )


def build_tfim_mvp(nqubits: int):
    structures = []
    weights = []
    for i in range(nqubits - 1):
        term = [0] * nqubits
        term[i] = 3
        term[i + 1] = 3
        structures.append(term)
        weights.append(-1.0)
    for i in range(nqubits):
        term = [0] * nqubits
        term[i] = 1
        structures.append(term)
        weights.append(-1.0)
    return tc.quantum.PauliStringSum2MVP(structures, weights)


def tc_jax_vqe(nqubits: int, depth: int, dtype_str: str, device: str):
    if tc is None:
        raise ImportError("TensorCircuit is not installed in this environment.")
    if jax is None:
        raise ImportError("JAX is not installed in this environment.")

    jax.config.update("jax_platform_name", "gpu" if device == "gpu" else "cpu")
    K = tc.set_backend("jax")
    tc.set_dtype(dtype_str)
    configure_tc_contractor()
    print(f"[TC-JAX] JAX devices: {jax.devices()}", flush=True)

    mvp_fn = build_tfim_mvp(nqubits)

    def value_fn(params):
        def one_layer(state, layer_params):
            circuit = tc.Circuit(nqubits, inputs=state)
            for i in range(nqubits - 1):
                circuit.rzz(i, i + 1, theta=layer_params[0, i])
            for i in range(nqubits):
                circuit.rx(i, theta=layer_params[1, i])
            return circuit.state()

        circuit = tc.Circuit(nqubits)
        for i in range(nqubits):
            circuit.h(i)

        state = K.scan(one_layer, params, circuit.state())
        state_1d = K.reshape(state, (-1,))
        h_state = mvp_fn(state_1d)
        energy = K.adjoint(state_1d) @ h_state
        return K.real(energy)

    value_and_grad_fn = K.jit(K.value_and_grad(value_fn))
    params_init = PARAM_INIT * K.ones([depth, 2, nqubits], dtype="float32")

    def wrapper(params):
        val, grad = value_and_grad_fn(params)
        val.block_until_ready()
        grad.block_until_ready()
        return np.array(val), np.array(grad)

    return wrapper, params_init


def mq_vqe(nqubits: int, depth: int, dtype_str: str, device: str):
    if mq is None:
        raise ImportError("MindQuantum is not installed in this environment.")

    dtype = mq.complex64 if dtype_str == "complex64" else mq.complex128
    simulator_name = "mqvector_gpu" if device == "gpu" else "mqvector"

    ham = QubitOperator()
    for i in range(nqubits - 1):
        ham += QubitOperator(f"Z{i} Z{i+1}", -1.0)
    for i in range(nqubits):
        ham += QubitOperator(f"X{i}", -1.0)
    hamiltonian = mq.Hamiltonian(ham).astype(dtype)

    circ = Circuit()
    for i in range(nqubits):
        circ.h(i)
    for layer in range(depth):
        for i in range(nqubits - 1):
            circ.rzz(f"theta_{layer}_0_{i}", [i, i + 1])
        for i in range(nqubits):
            circ.rx(f"theta_{layer}_1_{i}", i)

    sim = Simulator(simulator_name, nqubits, dtype=dtype)
    print(f"[MINDQUANTUM] Simulator: {simulator_name}", flush=True)
    grad_ops = sim.get_expectation_with_grad(hamiltonian, circ)

    # 3D parameter structure to match JAX
    params_init = np.ones([depth, 2, nqubits], dtype=np.float32) * PARAM_INIT

    def wrapper(params):
        p_list = []
        for layer in range(depth):
            for i in range(nqubits - 1):
                p_list.append(params[layer, 0, i])
            for i in range(nqubits):
                p_list.append(params[layer, 1, i])
        p_1d = np.array(p_list, dtype=np.float32)

        val, grad = grad_ops(p_1d)
        v = val[0][0].real
        g_1d = grad[0][0].real

        g_3d = np.zeros_like(params)
        idx = 0
        for layer in range(depth):
            for i in range(nqubits - 1):
                g_3d[layer, 0, i] = g_1d[idx]
                idx += 1
            for i in range(nqubits):
                g_3d[layer, 1, i] = g_1d[idx]
                idx += 1
        return v, g_3d

    return wrapper, params_init


def main():
    args = parse_args()

    if args.backend == "tc-jax":
        value_and_grad, params = tc_jax_vqe(
            args.nqubits, args.depth, args.tc_dtype, args.device
        )
    else:
        value_and_grad, params = mq_vqe(
            args.nqubits, args.depth, args.tc_dtype, args.device
        )

    print(f"[{args.backend.upper()}] Warmup/Compile...", flush=True)
    t_start = time.perf_counter()
    v, g = value_and_grad(params)
    warmup_time = time.perf_counter() - t_start
    print(
        f"[{args.backend.upper()}] Warmup done in {warmup_time:.5f}s "
        f"(val={v:.6f}, grad_norm={np.linalg.norm(g):.6f})",
        flush=True,
    )

    runs = 5
    times = []
    for i in range(runs):
        t_start = time.perf_counter()
        v, g = value_and_grad(params)
        run_time = time.perf_counter() - t_start
        times.append(run_time)
        print(f"[{args.backend.upper()}] Run {i+1}: {run_time:.5f}s", flush=True)

    mean_time = np.mean(times)
    print(f"[{args.backend.upper()}] Mean Run Time: {mean_time:.5f}s", flush=True)


if __name__ == "__main__":
    main()
