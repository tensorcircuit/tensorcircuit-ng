import uuid
import os
import json
import argparse
import time
import sys
import datetime
import numpy as np
import cpuinfo
import tensorcircuit as tc

from benchmark_core import benchmark_mega_function


def arg():
    parser = argparse.ArgumentParser(description="TensorCircuit Benchmark Parameters.")
    parser.add_argument(
        "-n", dest="n", type=int, nargs=1, help="# of Qubits", default=[12]
    )
    parser.add_argument(
        "-nlayer", dest="nlayer", type=int, nargs=1, help="# of layers", default=[3]
    )
    parser.add_argument(
        "-nitrs", dest="nitrs", type=int, nargs=1, help="# of iterations", default=[10]
    )
    parser.add_argument(
        "-t", dest="timeLimit", type=int, nargs=1, help="Time limit(s)", default=[600]
    )
    parser.add_argument(
        "-gpu", dest="isgpu", type=int, nargs=1, help="GPU available", default=[0]
    )
    parser.add_argument(
        "-lx", dest="lx", type=int, nargs=1, help="Lattice size x (for 2D)", default=[3]
    )
    parser.add_argument(
        "-ly", dest="ly", type=int, nargs=1, help="Lattice size y (for 2D)", default=[4]
    )
    parser.add_argument(
        "-path",
        dest="path",
        type=str,
        nargs=1,
        help="output json dir path ended with /",
        default=[None],
    )
    parser.add_argument(
        "-circuit_type",
        dest="circuit_type",
        type=str,
        nargs=1,
        help="Type of circuit (circuit, dmcircuit, mpscircuit)",
        default=["circuit"],
    )
    parser.add_argument(
        "-layout_type",
        dest="layout_type",
        type=str,
        nargs=1,
        help="Circuit layout (1d, 2d)",
        default=["1d"],
    )
    parser.add_argument(
        "-operation",
        dest="operation",
        type=str,
        nargs=1,
        help="Operation to perform (state, sample, exps)",
        default=["state"],
    )
    parser.add_argument(
        "-noisy",
        dest="noisy",
        type=int,
        nargs=1,
        help="Whether to add noise (0 or 1)",
        default=[0],
    )
    parser.add_argument(
        "-noisy_type",
        dest="noisy_type",
        type=str,
        nargs=1,
        help="Type of noise channel (depolarizing, amplitudedamping)",
        default=["depolarizing"],
    )
    parser.add_argument(
        "-use_grad",
        dest="use_grad",
        type=int,
        nargs=1,
        help="Whether to compute gradient (0 or 1)",
        default=[0],
    )
    parser.add_argument(
        "-use_vmap",
        dest="use_vmap",
        type=int,
        nargs=1,
        help="Whether to use vectorized operations (0 or 1)",
        default=[0],
    )
    parser.add_argument(
        "-batch_size",
        dest="batch_size",
        type=int,
        nargs=1,
        help="Batch size for vmap operations",
        default=[5],
    )
    parser.add_argument(
        "-backend",
        dest="backend",
        type=str,
        nargs=1,
        help="Backend to use (tensorflow, jax, pytorch)",
        default=["tensorflow"],
    )
    parser.add_argument(
        "-dtype",
        dest="dtype",
        type=str,
        nargs=1,
        help="Data type (complex64, complex128)",
        default=["complex64"],
    )
    parser.add_argument(
        "-contractor",
        dest="contractor",
        type=str,
        nargs=1,
        help="Contractor setting (e.g., cotengra-16-128)",
        default=[None],
    )
    parser.add_argument(
        "-bond_dim",
        dest="bond_dim",
        type=int,
        nargs=1,
        help="Bond dimension for MPS circuits",
        default=[16],
    )
    parser.add_argument(
        "-jit_compile",
        dest="jit_compile",
        type=int,
        nargs=1,
        help="Whether to use JIT compilation (0 or 1)",
        default=[1],
    )
    args = parser.parse_args()
    return [
        args.n[0],
        args.nlayer[0],
        args.nitrs[0],
        args.timeLimit[0],
        args.isgpu[0],
        args.lx[0],
        args.ly[0],
        args.path[0],
        args.circuit_type[0],
        args.layout_type[0],
        args.operation[0],
        args.noisy[0],
        args.noisy_type[0],
        args.use_grad[0],
        args.use_vmap[0],
        args.batch_size[0],
        args.backend[0],
        args.dtype[0],
        args.contractor[0],
        args.bond_dim[0],
        args.jit_compile[0],
    ]


def timing(f, nitrs, timeLimit, params):
    t0 = time.time()
    a = f(params)
    if hasattr(a, "block_until_ready"):
        a.block_until_ready()
    t1 = time.time()
    Nitrs = 1e-8
    for _ in range(nitrs):
        a = f(params)
        if hasattr(a, "block_until_ready"):
            a.block_until_ready()
        Nitrs += 1
        if time.time() - t1 > timeLimit:
            break
    t2 = time.time()
    return t1 - t0, (t2 - t1) / Nitrs, int(Nitrs)


def save(data, _uuid, path):
    if path is None:
        return
    with open(path + _uuid + ".json", "w") as f:
        json.dump(
            data,
            f,
            indent=4,
        )


def benchmark_cli(
    uuid,
    n,
    nlayer,
    nitrs,
    timeLimit,
    isgpu,
    lx,
    ly,
    circuit_type,
    layout_type,
    operation,
    noisy,
    noisy_type,
    use_grad,
    use_vmap,
    batch_size,
    backend,
    dtype,
    contractor,
    bond_dim,
    jit_compile,
    path,
):
    meta = {}

    # Setup GPU
    if isgpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        meta["isgpu"] = "off"
    else:
        meta["isgpu"] = "on"

    # Setup backend and dtype
    tc.set_backend(backend)
    tc.set_dtype(dtype)

    meta["Software"] = "tensorcircuit-ng"
    meta["Cpuinfo"] = cpuinfo.get_cpu_info()["brand_raw"]
    meta["Version"] = {
        "sys": sys.version,
        "tensorcircuit": tc.__version__,
        "numpy": np.__version__,
    }
    meta["Benchmark test parameters"] = {
        "nQubits": n,
        "nlayer": nlayer,
        "nitrs": nitrs,
        "timeLimit": timeLimit,
        "lx": lx,
        "ly": ly,
        "circuit_type": circuit_type,
        "layout_type": layout_type,
        "operation": operation,
        "noisy": noisy,
        "noisy_type": noisy_type,
        "use_grad": use_grad,
        "use_vmap": use_vmap,
        "batch_size": batch_size,
        "backend": backend,
        "dtype": dtype,
        "contractor": contractor,
        "bond_dim": bond_dim,
        "jit_compile": jit_compile,
    }
    meta["UUID"] = uuid
    meta["Benchmark Time"] = (
        datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    )
    meta["Results"] = {}

    # Create benchmark function using the mega function
    benchmark_func = benchmark_mega_function(
        nqubits=n,
        nlayers=nlayer,
        lx=lx,
        ly=ly,
        circuit_type=circuit_type,
        bond_dim=bond_dim,
        layout_type=layout_type,
        operation=operation,
        noisy=bool(noisy),
        noisy_type=noisy_type,
        use_grad=bool(use_grad),
        use_vmap=bool(use_vmap),
        contractor=contractor,
        jit_compile=bool(jit_compile),
    )

    # Create parameters for testing
    params_shape = (nlayer, n)  # Match the format in generate_1d_circuit
    if use_vmap:
        params_shape = (batch_size, nlayer, n)

    params = tc.backend.convert_to_tensor(
        np.random.uniform(0, 2 * np.pi, size=params_shape).astype(
            dtype.replace("complex", "float")
        )
    )

    # Run benchmark
    ct, it, Nitrs = timing(benchmark_func, nitrs, timeLimit, params)

    meta["Results"] = {
        "Construction time": ct,
        "Iteration time": it,
        "# of actual iterations": Nitrs,
    }

    print(meta)
    return meta


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    (
        n,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        lx,
        ly,
        path,
        circuit_type,
        layout_type,
        operation,
        noisy,
        noisy_type,
        use_grad,
        use_vmap,
        batch_size,
        backend,
        dtype,
        contractor,
        bond_dim,
        jit_compile,
    ) = arg()

    results = benchmark_cli(
        _uuid,
        n,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        lx,
        ly,
        circuit_type,
        layout_type,
        operation,
        noisy,
        noisy_type,
        use_grad,
        use_vmap,
        batch_size,
        backend,
        dtype,
        contractor,
        bond_dim,
        jit_compile,
        path,
    )
    save(results, _uuid, path)
