"""
Benchmark TFIM VQE value and gradient evaluation.

The TensorCircuit-NG paths mirror ``benchmark_mq_vs_tc_vqe.py``: JAX backend,
TFIM Hamiltonian, ``depth=10`` by default, ``complex64`` by default, initial
``|+>^n``, nearest-neighbor ``RZZ`` gates, and single-qubit ``RX`` gates.

Backends:

- ``tc-jax-scan``: scan VQE layers to reduce JAX staging time.
- ``tc-jax-unrolled``: build all layers directly for faster post-JIT runtime.
- ``cuquantum-statevector``: cuStateVec gate application plus full adjoint
  differentiation.
- ``cuquantum-tensornet``: cuTensorNet full-state contraction plus PyTorch
  autograd for the TFIM expectation.
"""

import argparse
import time

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import cuquantum
    from cuquantum import ComputeType, cudaDataType
    from cuquantum.bindings import custatevec
    from cuquantum.tensornet import contract
except ImportError:
    cuquantum = None
    ComputeType = None
    cudaDataType = None
    custatevec = None
    contract = None

try:
    import jax
except ImportError:
    jax = None

try:
    import tensorcircuit as tc
except ImportError:
    tc = None

try:
    import torch
except ImportError:
    torch = None

PARAM_INIT = 0.1
OMECO_TRIALS = 16
OMECO_ITERS = 24
OMECO_BETAS = np.geomspace(0.1, 10.0, OMECO_ITERS).tolist()


def parse_args():
    parser = argparse.ArgumentParser(
        description="TFIM VQE benchmark: TensorCircuit-NG JAX vs cuQuantum"
    )
    parser.add_argument("--nqubits", type=int, default=14)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument(
        "--backend",
        choices=[
            "tc-jax-scan",
            "tc-jax-unrolled",
            "cuquantum-statevector",
            "cuquantum-tensornet",
        ],
        required=True,
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument(
        "--tc-dtype",
        choices=["complex64", "complex128"],
        default="complex64",
    )
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def require(module, name):
    if module is None:
        raise ImportError(f"{name} is required for the selected backend.")


def configure_tc_contractor():
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
    log(
        "[TC-JAX] Contractor: OMECo TreeSA "
        f"(ntrials={OMECO_TRIALS}, niters={OMECO_ITERS})"
    )


def build_tfim_mvp(nqubits):
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


def tc_initial_state(nqubits):
    circuit = tc.Circuit(nqubits)
    for i in range(nqubits):
        circuit.h(i)
    return circuit


def apply_tc_layer(circuit, layer_params, nqubits):
    for i in range(nqubits - 1):
        circuit.rzz(i, i + 1, theta=layer_params[0, i])
    for i in range(nqubits):
        circuit.rx(i, theta=layer_params[1, i])


def tc_jax_vqe(nqubits, depth, dtype_str, device, mode):
    require(tc, "TensorCircuit-NG")
    require(jax, "JAX")

    jax.config.update("jax_platform_name", "gpu" if device == "gpu" else "cpu")
    backend = tc.set_backend("jax")
    tc.set_dtype(dtype_str)
    configure_tc_contractor()
    log(f"[TC-JAX] JAX devices: {jax.devices()}")
    log(f"[TC-JAX] Mode: {mode}")

    mvp_fn = build_tfim_mvp(nqubits)

    def energy_from_state(state):
        state_1d = backend.reshape(state, (-1,))
        h_state = mvp_fn(state_1d)
        return backend.real(backend.adjoint(state_1d) @ h_state)

    def value_scan(params):
        def one_layer(state, layer_params):
            circuit = tc.Circuit(nqubits, inputs=state)
            apply_tc_layer(circuit, layer_params, nqubits)
            return circuit.state()

        circuit = tc_initial_state(nqubits)
        state = backend.scan(one_layer, params, circuit.state())
        return energy_from_state(state)

    def value_unrolled(params):
        circuit = tc_initial_state(nqubits)
        for layer in range(depth):
            apply_tc_layer(circuit, params[layer], nqubits)
        return energy_from_state(circuit.state())

    value_fn = value_unrolled if mode == "unrolled" else value_scan
    value_and_grad = backend.jit(backend.value_and_grad(value_fn))
    params = PARAM_INIT * backend.ones([depth, 2, nqubits], dtype="float32")

    def wrapper(params):
        value, grad = value_and_grad(params)
        value.block_until_ready()
        grad.block_until_ready()
        return np.array(value), np.array(grad)

    return wrapper, params


def numpy_complex_dtype(dtype_str):
    return np.complex64 if dtype_str == "complex64" else np.complex128


def torch_dtypes(dtype_str):
    if dtype_str == "complex64":
        return torch.complex64, torch.float32
    return torch.complex128, torch.float64


def cu_data_type(dtype_str):
    if dtype_str == "complex64":
        return cudaDataType.CUDA_C_32F
    return cudaDataType.CUDA_C_64F


def cu_compute_type(dtype_str):
    if dtype_str == "complex64":
        return ComputeType.COMPUTE_32F
    return ComputeType.COMPUTE_64F


def cp_rx_gate(theta, complex_dtype):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=complex_dtype)


def cp_rzz_gate(theta, complex_dtype):
    phase_minus = np.exp(-0.5j * theta)
    phase_plus = np.exp(0.5j * theta)
    return np.diag([phase_minus, phase_plus, phase_plus, phase_minus]).astype(
        complex_dtype
    )


def apply_custatevec_matrix(handle, state, matrix, targets, dtype_str):
    # cuQuantum Python 26.3.x uses the compact workspace-size signature below.
    # Older examples may include the state pointer and target/control arrays.
    matrix_device = cp.ascontiguousarray(cp.asarray(matrix, dtype=state.dtype))
    data_type = cu_data_type(dtype_str)
    compute_type = cu_compute_type(dtype_str)
    n_index_bits = int(np.log2(state.size))

    workspace_size = custatevec.apply_matrix_get_workspace_size(
        handle,
        data_type,
        n_index_bits,
        matrix_device.data.ptr,
        data_type,
        custatevec.MatrixLayout.ROW,
        0,
        len(targets),
        0,
        compute_type,
    )
    workspace = cp.empty(workspace_size, dtype=cp.uint8) if workspace_size else None
    workspace_ptr = workspace.data.ptr if workspace is not None else 0

    custatevec.apply_matrix(
        handle,
        state.data.ptr,
        data_type,
        n_index_bits,
        matrix_device.data.ptr,
        data_type,
        custatevec.MatrixLayout.ROW,
        0,
        targets,
        len(targets),
        [],
        [],
        0,
        compute_type,
        workspace_ptr,
        workspace_size,
    )


def apply_custatevec_circuit(state, params, nqubits, depth, dtype_str, adjoint):
    complex_dtype = numpy_complex_dtype(dtype_str)
    handle = custatevec.create()
    try:
        layers = range(depth - 1, -1, -1) if adjoint else range(depth)
        h_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex_dtype)
        h_gate = h_gate / np.sqrt(2.0)

        if not adjoint:
            for qubit in range(nqubits):
                apply_custatevec_matrix(handle, state, h_gate, [qubit], dtype_str)

        for layer in layers:
            if adjoint:
                rx_qubits = range(nqubits - 1, -1, -1)
                rzz_qubits = range(nqubits - 2, -1, -1)
            else:
                rx_qubits = range(nqubits)
                rzz_qubits = range(nqubits - 1)

            if adjoint:
                for qubit in rx_qubits:
                    theta = -float(params[layer, 1, qubit])
                    gate = cp_rx_gate(theta, complex_dtype)
                    apply_custatevec_matrix(handle, state, gate, [qubit], dtype_str)

            for qubit in rzz_qubits:
                theta = float(params[layer, 0, qubit])
                gate = cp_rzz_gate(-theta if adjoint else theta, complex_dtype)
                apply_custatevec_matrix(
                    handle, state, gate, [qubit, qubit + 1], dtype_str
                )

            if not adjoint:
                for qubit in rx_qubits:
                    theta = float(params[layer, 1, qubit])
                    gate = cp_rx_gate(theta, complex_dtype)
                    apply_custatevec_matrix(handle, state, gate, [qubit], dtype_str)
    finally:
        custatevec.destroy(handle)


def cu_state_after_circuit(params, nqubits, depth, dtype_str):
    state = cp.zeros(1 << nqubits, dtype=numpy_complex_dtype(dtype_str))
    state[0] = state.dtype.type(1.0)
    apply_custatevec_circuit(state, params, nqubits, depth, dtype_str, False)
    return state


def apply_tfim_hamiltonian(state, nqubits):
    tensor = state.reshape([2] * nqubits)
    result = cp.zeros_like(tensor)
    signs = cp.asarray([1.0, -1.0], dtype=cp.float32)

    for i in range(nqubits - 1):
        shape_i = [1] * nqubits
        shape_j = [1] * nqubits
        shape_i[i] = 2
        shape_j[i + 1] = 2
        result -= tensor * signs.reshape(shape_i) * signs.reshape(shape_j)

    for i in range(nqubits):
        result -= cp.flip(tensor, axis=i)

    return result.reshape(-1)


def z_string_inner(left, right, nqubits, q0, q1):
    left_tensor = left.reshape([2] * nqubits)
    right_tensor = right.reshape([2] * nqubits)
    signs = cp.asarray([1.0, -1.0], dtype=cp.float32)
    shape0 = [1] * nqubits
    shape1 = [1] * nqubits
    shape0[q0] = 2
    shape1[q1] = 2
    return cp.sum(
        cp.conj(left_tensor)
        * right_tensor
        * signs.reshape(shape0)
        * signs.reshape(shape1)
    )


def x_string_inner(left, right, nqubits, qubit):
    left_tensor = left.reshape([2] * nqubits)
    right_tensor = right.reshape([2] * nqubits)
    return cp.sum(cp.conj(left_tensor) * cp.flip(right_tensor, axis=qubit))


def cu_statevector_value_and_grad(params, nqubits, depth, dtype_str):
    state = cu_state_after_circuit(params, nqubits, depth, dtype_str)
    adjoint_state = apply_tfim_hamiltonian(state, nqubits)
    energy = cp.vdot(state, adjoint_state).real
    grad = np.zeros_like(params)

    handle = custatevec.create()
    try:
        for layer in range(depth - 1, -1, -1):
            for qubit in range(nqubits - 1, -1, -1):
                inner = x_string_inner(adjoint_state, state, nqubits, qubit)
                grad[layer, 1, qubit] = float(cp.imag(inner).get())
                gate = cp_rx_gate(
                    -float(params[layer, 1, qubit]),
                    numpy_complex_dtype(dtype_str),
                )
                for wavefunction in [state, adjoint_state]:
                    apply_custatevec_matrix(
                        handle, wavefunction, gate, [qubit], dtype_str
                    )

            for qubit in range(nqubits - 2, -1, -1):
                inner = z_string_inner(adjoint_state, state, nqubits, qubit, qubit + 1)
                grad[layer, 0, qubit] = float(cp.imag(inner).get())
                gate = cp_rzz_gate(
                    -float(params[layer, 0, qubit]),
                    numpy_complex_dtype(dtype_str),
                )
                for wavefunction in [state, adjoint_state]:
                    apply_custatevec_matrix(
                        handle,
                        wavefunction,
                        gate,
                        [qubit, qubit + 1],
                        dtype_str,
                    )
    finally:
        custatevec.destroy(handle)

    cp.cuda.Stream.null.synchronize()
    return float(energy.get()), grad


def cu_statevector_vqe(nqubits, depth, dtype_str, device):
    require(cp, "CuPy")
    require(custatevec, "cuQuantum cuStateVec")
    if device != "gpu":
        raise ValueError("cuStateVec benchmark requires --device gpu.")

    params = np.ones([depth, 2, nqubits], dtype=np.float32) * PARAM_INIT
    log(f"[CUQUANTUM-SV] cuQuantum version: {cuquantum.__version__}")
    log(f"[CUQUANTUM-SV] cuStateVec device: {cp.cuda.Device()}")
    log("[CUQUANTUM-SV] Full-gradient method: adjoint differentiation")

    def wrapper(params):
        return cu_statevector_value_and_grad(params, nqubits, depth, dtype_str)

    return wrapper, params


def torch_rx_gate(theta, complex_dtype):
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    minus_i_sin = -1j * sin.to(complex_dtype)
    return torch.stack(
        [
            torch.stack([cos.to(complex_dtype), minus_i_sin]),
            torch.stack([minus_i_sin, cos.to(complex_dtype)]),
        ]
    )


def torch_rzz_gate(theta, complex_dtype):
    phase_minus = torch.exp((-0.5j * theta).to(complex_dtype))
    phase_plus = torch.exp((0.5j * theta).to(complex_dtype))
    return torch.diag(
        torch.stack([phase_minus, phase_plus, phase_plus, phase_minus])
    ).reshape(2, 2, 2, 2)


def append_one_qubit_gate(network, current_modes, gate, qubit, next_mode):
    output_mode = next_mode
    network.extend([gate, [output_mode, current_modes[qubit]]])
    current_modes[qubit] = output_mode
    return next_mode + 1


def append_two_qubit_gate(network, current_modes, gate, q0, q1, next_mode):
    output0 = next_mode
    output1 = next_mode + 1
    network.extend([gate, [output0, output1, current_modes[q0], current_modes[q1]]])
    current_modes[q0] = output0
    current_modes[q1] = output1
    return next_mode + 2


def append_tensornet_circuit(network, params, current_modes, nqubits, depth):
    device = params.device
    complex_dtype, _ = torch_dtypes(
        "complex64" if params.dtype == torch.float32 else "complex128"
    )
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    h_gate = torch.tensor(
        [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]],
        dtype=complex_dtype,
        device=device,
    )
    next_mode = nqubits

    for qubit in range(nqubits):
        next_mode = append_one_qubit_gate(
            network, current_modes, h_gate, qubit, next_mode
        )

    for layer in range(depth):
        for qubit in range(nqubits - 1):
            gate = torch_rzz_gate(params[layer, 0, qubit], complex_dtype)
            next_mode = append_two_qubit_gate(
                network, current_modes, gate, qubit, qubit + 1, next_mode
            )
        for qubit in range(nqubits):
            gate = torch_rx_gate(params[layer, 1, qubit], complex_dtype)
            next_mode = append_one_qubit_gate(
                network, current_modes, gate, qubit, next_mode
            )


def cu_tensornet_state(params, nqubits, depth, complex_dtype):
    ket0 = torch.tensor([1.0, 0.0], dtype=complex_dtype, device=params.device)
    current_modes = list(range(nqubits))
    network = []
    for mode in current_modes:
        network.extend([ket0, [mode]])

    append_tensornet_circuit(network, params, current_modes, nqubits, depth)
    network.append(current_modes)
    return contract(*network)


def cu_tensornet_energy(params, nqubits, depth, complex_dtype, real_dtype):
    state = cu_tensornet_state(params, nqubits, depth, complex_dtype)
    state = state.reshape([2] * nqubits)
    probs = state.abs().square()
    signs = torch.tensor([1.0, -1.0], dtype=real_dtype, device=params.device)
    energy = torch.zeros((), dtype=real_dtype, device=params.device)

    for i in range(nqubits - 1):
        shape_i = [1] * nqubits
        shape_j = [1] * nqubits
        shape_i[i] = 2
        shape_j[i + 1] = 2
        energy -= torch.sum(probs * signs.reshape(shape_i) * signs.reshape(shape_j))

    for i in range(nqubits):
        energy -= torch.sum(torch.conj(state) * torch.flip(state, dims=[i])).real

    return energy


def cu_tensornet_vqe(nqubits, depth, dtype_str, device):
    require(torch, "PyTorch")
    require(contract, "cuQuantum cuTensorNet")
    if device != "gpu":
        raise ValueError("cuTensorNet benchmark requires --device gpu.")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available.")

    complex_dtype, real_dtype = torch_dtypes(dtype_str)
    params = torch.full(
        [depth, 2, nqubits],
        PARAM_INIT,
        dtype=real_dtype,
        device=torch.device("cuda"),
        requires_grad=True,
    )
    log(f"[CUQUANTUM-TN] cuQuantum version: {cuquantum.__version__}")
    log("[CUQUANTUM-TN] cuTensorNet full-state contraction with PyTorch autograd")

    def wrapper(params):
        params.grad = None
        value = cu_tensornet_energy(params, nqubits, depth, complex_dtype, real_dtype)
        value.backward()
        torch.cuda.synchronize()
        grad = params.grad.detach().cpu().numpy()
        return float(value.detach().cpu()), grad

    return wrapper, params


def build_backend(args):
    if args.backend in {"tc-jax-scan", "tc-jax-unrolled"}:
        mode = "unrolled" if args.backend == "tc-jax-unrolled" else "scan"
        return tc_jax_vqe(args.nqubits, args.depth, args.tc_dtype, args.device, mode)
    if args.backend == "cuquantum-statevector":
        return cu_statevector_vqe(args.nqubits, args.depth, args.tc_dtype, args.device)
    return cu_tensornet_vqe(args.nqubits, args.depth, args.tc_dtype, args.device)


def run_benchmark(name, value_and_grad, params, runs):
    log(f"[{name}] Warmup/Compile...")
    start = time.perf_counter()
    value, grad = value_and_grad(params)
    warmup_time = time.perf_counter() - start
    log(
        f"[{name}] Warmup done in {warmup_time:.5f}s "
        f"(val={value:.6f}, grad_norm={np.linalg.norm(grad):.6f})"
    )

    times = []
    for i in range(runs):
        start = time.perf_counter()
        value, grad = value_and_grad(params)
        run_time = time.perf_counter() - start
        times.append(run_time)
        log(f"[{name}] Run {i + 1}: {run_time:.5f}s")

    log(f"[{name}] Mean Run Time: {np.mean(times):.5f}s")


def main():
    args = parse_args()
    value_and_grad, params = build_backend(args)
    run_benchmark(args.backend.upper(), value_and_grad, params, args.runs)


if __name__ == "__main__":
    main()
