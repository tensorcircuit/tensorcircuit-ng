"""
Benchmark one post-JIT value-and-gradient step for TFIM VQE.

Fixed problem:

    H = -sum_i Z_i Z_{i+1} - sum_i X_i

Fixed ansatz:

    |+>^n, then each layer applies nearest-neighbor RZZ gates and RX gates.

TensorCircuit uses a prebuilt PauliStringSum2COO sparse Hamiltonian and
operator_expectation. TorchQuantum uses the same statevector and a TFIM-specific
expectation routine.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from typing import Any

import numpy as np

import tensorcircuit as tc

PARAM_INIT = 0.1
OMECO_TRIALS = 16
OMECO_ITERS = 16
OMECO_BETAS = np.geomspace(0.1, 10.0, OMECO_ITERS).tolist()

torch: Any = None
tq: Any = None
tqf: Any = None


def log(message: str) -> None:
    print(message, flush=True)


def load_torch() -> Any:
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch


def load_torchquantum() -> tuple[Any, Any]:
    global tq, tqf
    load_torch()
    if tq is None:
        import torchquantum as tq_module
        import torchquantum.functional as tqf_module

        tq = tq_module
        tqf = tqf_module
    return tq, tqf


@contextlib.contextmanager
def torch_default_device(device: str) -> Any:
    torch_module = load_torch()
    previous = torch_module.get_default_device()
    torch_module.set_default_device(device)
    try:
        yield
    finally:
        torch_module.set_default_device(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=20)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument(
        "--backend",
        nargs="+",
        default=["all"],
        choices=["all", "tc-jax", "tc-torch", "torchquantum"],
        help="Run all, or select a subset for separate JAX/Torch conda envs.",
    )
    parser.add_argument("--torch-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--tc-dtype", choices=["complex64", "complex128"], default="complex64"
    )
    return parser.parse_args()


def configure_omeco() -> None:
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


def sync_result(result: Any) -> None:
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, tuple):
        for item in result:
            sync_result(item)


def sync_cuda(device: str) -> None:
    if device == "cuda":
        load_torch().cuda.synchronize()


def benchmark(
    name: str, value_and_grad: Any, params: Any, device: str | None = None
) -> dict[str, float]:
    log(f"[{name}] warmup/compile: start")
    start = time.perf_counter()
    value, grad = value_and_grad(params)
    sync_result((value, grad))
    if device is not None:
        sync_cuda(device)
    warmup = time.perf_counter() - start
    log(f"[{name}] warmup/compile: {warmup:.6f} s")

    log(f"[{name}] measured value-and-grad: start")
    start = time.perf_counter()
    value, grad = value_and_grad(params)
    sync_result((value, grad))
    if device is not None:
        sync_cuda(device)
    step = time.perf_counter() - start

    stats = {
        "value": scalar(value),
        "grad_norm": norm(grad),
        "warmup": warmup,
        "step": step,
    }
    log(
        f"[{name}] value={stats['value']:.8f}, "
        f"||grad||={stats['grad_norm']:.8f}, step={step:.6f} s"
    )
    return stats


def parameter_shape(nqubits: int, depth: int) -> list[int]:
    return [depth, 2, nqubits]


def apply_tc_ansatz(circuit: tc.Circuit, params: Any, nqubits: int, depth: int) -> None:
    for layer in range(depth):
        for i in range(nqubits - 1):
            circuit.rzz(i, i + 1, theta=params[layer, 0, i])
        for i in range(nqubits):
            circuit.rx(i, theta=params[layer, 1, i])


def tfim_sparse_hamiltonian(nqubits: int) -> Any:
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

    with tc.runtime_backend("numpy"):
        coo = tc.quantum.PauliStringSum2COO(structures, weights, numpy=True)
    hamiltonian = tc.backend.coo_sparse_matrix_from_numpy(coo)
    return hamiltonian


def tc_value_function(nqubits: int, depth: int) -> Any:
    hamiltonian = tfim_sparse_hamiltonian(nqubits)
    backend = tc.backend

    def value(params: Any) -> Any:
        circuit = tc.Circuit(nqubits)
        for i in range(nqubits):
            circuit.h(i)
        apply_tc_ansatz(circuit, params, nqubits, depth)
        energy = tc.templates.measurements.operator_expectation(circuit, hamiltonian)
        return backend.real(energy)

    return value


def tc_jax_benchmark(nqubits: int, depth: int) -> tuple[Any, Any]:
    value = tc_value_function(nqubits, depth)
    value_and_grad = tc.backend.jit(tc.backend.value_and_grad(value))
    params = PARAM_INIT * tc.backend.ones(
        parameter_shape(nqubits, depth), dtype="float32"
    )
    return value_and_grad, params


def tc_torch_benchmark(nqubits: int, depth: int, device: str) -> tuple[Any, Any]:
    torch_module = load_torch()
    value = tc_value_function(nqubits, depth)

    def value_and_grad(params: Any) -> tuple[Any, Any]:
        params = params.detach().clone().requires_grad_(True)
        val = value(params)
        (grad,) = torch_module.autograd.grad(val, params)
        return val, grad

    params = PARAM_INIT * tc.backend.ones(
        parameter_shape(nqubits, depth), dtype="float32"
    )
    if device == "cuda":
        params = params.to("cuda")
    return value_and_grad, params


def z_signs(nqubits: int, device: str) -> tuple[Any, ...]:
    torch_module = load_torch()
    z = torch_module.tensor([1.0, -1.0], device=device)
    return tuple(
        z.reshape([1] + [2 if j == i else 1 for j in range(nqubits)])
        for i in range(nqubits)
    )


def apply_tq_ansatz(qdev: Any, params: Any, nqubits: int, depth: int) -> None:
    _, tqf_module = load_torchquantum()
    for layer in range(depth):
        for i in range(nqubits - 1):
            tqf_module.rzz(qdev, wires=[i, i + 1], params=params[layer, 0, i])
        for i in range(nqubits):
            tqf_module.rx(qdev, wires=i, params=params[layer, 1, i])


def tq_value(params: Any, nqubits: int, depth: int, signs: tuple[Any, ...]) -> Any:
    tq_module, tqf_module = load_torchquantum()
    torch_module = load_torch()
    qdev = tq_module.QuantumDevice(n_wires=nqubits, bsz=1, device=params.device)
    for i in range(nqubits):
        tqf_module.hadamard(qdev, wires=i)
    apply_tq_ansatz(qdev, params, nqubits, depth)

    states = qdev.states
    probs = states.abs().square()
    energy = torch_module.zeros((), dtype=probs.dtype, device=params.device)
    for i in range(nqubits - 1):
        energy = energy - (probs * signs[i] * signs[i + 1]).sum()
    for i in range(nqubits):
        flipped = torch_module.flip(states, dims=[i + 1])
        energy = energy - (states.conj() * flipped).sum().real
    return energy


def torchquantum_benchmark(nqubits: int, depth: int, device: str) -> tuple[Any, Any]:
    torch_module = load_torch()
    load_torchquantum()
    signs = z_signs(nqubits, device)

    def value(params: Any) -> Any:
        return tq_value(params, nqubits, depth, signs)

    grad_and_value = torch_module.func.grad_and_value(value)

    def value_and_grad(params: Any) -> tuple[Any, Any]:
        grad, val = grad_and_value(params)
        return val, grad

    params = PARAM_INIT * torch_module.ones(
        parameter_shape(nqubits, depth),
        dtype=torch_module.float32,
        device=device,
    )
    return torch_module.compile(value_and_grad), params


def scalar(value: Any) -> float:
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()))
    return float(np.asarray(value))


def norm(grad: Any) -> float:
    if torch is not None and isinstance(grad, torch.Tensor):
        return float(torch.linalg.vector_norm(grad.detach()).cpu())
    return float(np.linalg.norm(np.asarray(grad)))


def print_results(results: dict[str, dict[str, float]]) -> None:
    log("\nOne post-warmup value-and-grad call")
    log(
        f"{'backend':<28} {'value':>14} {'||grad||':>14} {'warmup (s)':>12} {'step (s)':>12}"
    )
    log("-" * 84)
    for name, stats in results.items():
        log(
            f"{name:<28} {stats['value']:>14.8f} {stats['grad_norm']:>14.8f} "
            f"{stats['warmup']:>12.6f} {stats['step']:>12.6f}"
        )


def main() -> None:
    args = parse_args()
    selected = set(args.backend)
    run_all = "all" in selected
    torch_needed = run_all or "tc-torch" in selected or "torchquantum" in selected

    tc_dtype, tc_rdtype = tc.set_dtype(args.tc_dtype)
    if torch_needed:
        load_torch().set_default_dtype(torch.float32)

    log(
        "TFIM VQE sparse-operator benchmark: "
        f"nqubits={args.nqubits}, depth={args.depth}, "
        f"tc_dtype={tc_dtype}, tc_rdtype={tc_rdtype}, "
        f"torch_device={args.torch_device}, warmup=1"
    )

    results: dict[str, dict[str, float]] = {}

    if run_all or "tc-jax" in selected or "tc-torch" in selected:
        configure_omeco()
        try:
            if run_all or "tc-jax" in selected:
                with tc.runtime_backend("jax"):
                    log("[TensorCircuit/JAX] build callable")
                    vag, params = tc_jax_benchmark(args.nqubits, args.depth)
                    results["TensorCircuit/JAX"] = benchmark(
                        "TensorCircuit/JAX", vag, params
                    )

            if run_all or "tc-torch" in selected:
                with (
                    tc.runtime_backend("pytorch"),
                    torch_default_device(args.torch_device),
                ):
                    log("[TensorCircuit/PyTorch] build callable")
                    vag, params = tc_torch_benchmark(
                        args.nqubits, args.depth, args.torch_device
                    )
                    results["TensorCircuit/PyTorch"] = benchmark(
                        "TensorCircuit/PyTorch",
                        vag,
                        params,
                        device=args.torch_device,
                    )
        finally:
            tc.set_contractor("greedy", preprocessing=True)

    if run_all or "torchquantum" in selected:
        log("[TorchQuantum] build callable")
        vag, params = torchquantum_benchmark(
            args.nqubits, args.depth, args.torch_device
        )
        results["TorchQuantum"] = benchmark(
            "TorchQuantum", vag, params, args.torch_device
        )

    print_results(results)


if __name__ == "__main__":
    main()
