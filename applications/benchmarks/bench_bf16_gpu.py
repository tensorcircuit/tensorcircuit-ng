"""GPU bf16 benchmark harness: peak GPU mem + wall-time + accuracy, bf16 vs complex64.

End-to-end mode: build tc.Circuit -> cotengra contraction. Micro mode: one big
complex-bf16 matmul (4 bf16 GEMMs) -> K3 native-GEMM evidence. One subprocess per
trial for clean peak-memory attribution (nvidia-smi polling as the cross-backend
common truth; backend API as fine-grained cross-check).
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, List, Optional

MICRO_M_DEFAULT = 4096


def build_circuit(circuit: str, n: int) -> Any:
    import tensorcircuit as tc

    c = tc.Circuit(n)
    if circuit == "ghz":
        c.H(0)
        for i in range(n - 1):
            c.cnot(i, i + 1)
    elif circuit == "brickwork":
        for i in range(n):
            c.H(i)
        for _ in range(3):
            for i in range(0, n - 1, 2):
                c.cnot(i, i + 1)
            for i in range(1, n - 1, 2):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rz(i, theta=0.7)
    elif circuit == "qaoa-ising":
        gamma, beta = 0.5, 0.3
        for i in range(n):
            c.H(i)
        for i in range(n - 1):  # 1D chain ZZ cost, p=1
            c.cnot(i, i + 1)
            c.rz(i + 1, theta=gamma)
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=beta)
    else:
        raise ValueError(f"unknown circuit {circuit!r}")
    return c


def contract(circuit: str, n: int, bf16: bool) -> Any:
    import numpy as np
    from applications.bcomplex32_algebra import bcomplex32

    c = build_circuit(circuit, n)
    if bf16:
        with bcomplex32():
            return np.asarray(c.state())
    return np.asarray(c.state())


class GpuSmiPoller:
    """Background thread polling nvidia-smi memory.used; reports peak bytes."""

    def __init__(self, gpu: int = 0, interval_s: float = 0.05) -> None:
        self.gpu = gpu
        self.interval_s = interval_s
        self._peak_mib = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "GpuSmiPoller":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                used = int(out.stdout.strip())
                if used > self._peak_mib:
                    self._peak_mib = used
            except Exception:
                pass
            time.sleep(self.interval_s)

    def peak_bytes(self) -> int:
        return self._peak_mib * 1024 * 1024  # nvidia-smi reports MiB


def reset_backend_mem(backend: str) -> None:
    try:
        if backend == "pytorch":
            import torch

            torch.cuda.reset_peak_memory_stats()
        elif backend == "tensorflow":
            import tensorflow as tf

            tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def backend_alloc_peak(backend: str) -> Optional[int]:
    """Precise allocated-bytes peak via backend API (torch clean); else None."""
    try:
        if backend == "pytorch":
            import torch

            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None


def worker(args: argparse.Namespace) -> None:
    """Run one (backend, dtype, circuit, n) end-to-end trial; print JSON to stdout."""
    import numpy as np
    import tensorcircuit as tc

    tc.set_backend(args.backend)
    ref = None
    if args.dtype == "bf16":
        ref = np.asarray(contract(args.circuit, args.n, bf16=False))
    reset_backend_mem(args.backend)
    with GpuSmiPoller(gpu=args.gpu) as poller:
        res = contract(args.circuit, args.n, bf16=(args.dtype == "bf16"))
        _ = np.asarray(res)
    peak_smi = poller.peak_bytes()
    alloc = backend_alloc_peak(args.backend)

    walls: List[float] = []
    for _ in range(args.trials):
        t0 = time.perf_counter()
        res = contract(args.circuit, args.n, bf16=(args.dtype == "bf16"))
        _ = np.asarray(res)
        walls.append(time.perf_counter() - t0)
    wall = statistics.median(walls)

    max_abs = rel = None
    if args.dtype == "bf16" and ref is not None:
        got = np.asarray(contract(args.circuit, args.n, bf16=True))
        diff = np.abs(got - ref)
        max_abs = float(diff.max())
        rel = float(diff.max() / (np.abs(ref).max() + 1e-12))

    print(
        json.dumps(
            {
                "backend": args.backend,
                "dtype": args.dtype,
                "circuit": args.circuit,
                "n": args.n,
                "mode": "end-to-end",
                "peak_smi_bytes": peak_smi,
                "peak_alloc_bytes": alloc,
                "wall_s": wall,
                "max_abs_err": max_abs,
                "rel_err": rel,
                "trials": args.trials,
            }
        )
    )


def micro_worker(args: argparse.Namespace) -> None:
    """Single big complex-bf16 matmul (4 bf16 GEMMs), K trials median. K3 evidence."""
    import numpy as np
    import tensorcircuit as tc
    from applications.bcomplex32_algebra import (
        _complex_to_pair,
        _pair_tensordot,
        _pair_to_complex,
    )

    tc.set_backend(args.backend)
    be = tc.backend
    m = args.micro_m
    a = be.cast(
        be.convert_to_tensor(np.random.standard_normal((m, m)).astype(np.complex64)),
        "complex64",
    )
    b = be.cast(
        be.convert_to_tensor(np.random.standard_normal((m, m)).astype(np.complex64)),
        "complex64",
    )
    pa, pb = _complex_to_pair(be, a), _complex_to_pair(be, b)
    axes = ([1], [0])
    out = _pair_to_complex(be, _pair_tensordot(be, pa, pb, axes=axes))
    _ = np.asarray(out)  # warmup
    walls: List[float] = []
    for _ in range(args.trials):
        t0 = time.perf_counter()
        out = _pair_to_complex(be, _pair_tensordot(be, pa, pb, axes=axes))
        _ = np.asarray(out)
        walls.append(time.perf_counter() - t0)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "dtype": "bf16",
                "circuit": "micro-matmul",
                "n": m,
                "mode": "micro",
                "wall_s": statistics.median(walls),
                "trials": args.trials,
            }
        )
    )


import csv
import os


def _env_with_repo(repo: str) -> dict:
    env = dict(os.environ)
    pp = repo
    if "PYTHONPATH" in env and env["PYTHONPATH"]:
        pp = repo + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pp
    return env


def _build_matrix(args: argparse.Namespace) -> List[tuple]:
    rows: List[tuple] = []
    backends = args.backends.split(",")
    circuits = args.circuits.split(",")
    if args.mode == "micro":
        for backend in backends:
            rows.append((backend, "bf16", "micro-matmul", args.micro_m))
        return rows
    for backend in backends:
        for circuit in circuits:
            ns = args.mem_ns.split(",") if circuit == "ghz" else args.speed_ns.split(",")
            for dtype in ("complex64", "bf16"):
                for n in ns:
                    rows.append((backend, dtype, circuit, int(n)))
    return rows


def _run_matrix(args: argparse.Namespace) -> List[dict]:
    results: List[dict] = []
    for backend, dtype, circuit, n in _build_matrix(args):
        sys.stderr.write(
            f"  trial backend={backend} dtype={dtype} circuit={circuit} n={n}\n"
        )
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "worker" if args.mode == "end-to-end" else "micro",
            "--backend",
            backend,
            "--dtype",
            dtype,
            "--circuit",
            str(circuit),
            "--n",
            str(n),
            "--trials",
            str(args.trials),
            "--gpu",
            str(args.gpu),
        ]
        if args.mode == "micro":
            cmd += ["--micro-m", str(args.micro_m)]
        try:
            out = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=_env_with_repo(args.repo),
                timeout=args.timeout,
                cwd=args.repo,
            )
        except subprocess.TimeoutExpired:
            sys.stderr.write("    TIMEOUT\n")
            continue
        if out.returncode != 0:
            sys.stderr.write(f"    FAILED: {out.stderr.strip()[:300]}\n")
            continue
        lines = out.stdout.strip().splitlines()
        if not lines:
            sys.stderr.write("    EMPTY OUTPUT (no JSON)\n")
            continue
        line = lines[-1]
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            sys.stderr.write(f"    BAD OUTPUT: {line[:200]}\n")
    return results


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    _worker_args(w)
    mw = sub.add_parser("micro")
    _worker_args(mw)
    o = sub.add_parser("run")
    o.add_argument("--backends", default="numpy")
    o.add_argument("--mode", choices=["end-to-end", "micro"], default="end-to-end")
    o.add_argument("--circuits", default="ghz,brickwork")
    o.add_argument("--mem-ns", default="16,18,20,22")
    o.add_argument("--speed-ns", default="20,22,24")
    o.add_argument("--trials", type=int, default=5)
    o.add_argument("--timeout", type=int, default=1800)
    o.add_argument("--gpu", type=int, default=0)
    o.add_argument("--repo", default=os.getcwd())
    o.add_argument("--out", default="bench_bf16_results.csv")
    o.add_argument("--micro-m", type=int, default=MICRO_M_DEFAULT)
    args = p.parse_args()
    if args.cmd in ("worker", "micro"):
        if args.cmd == "micro":
            micro_worker(args)
        else:
            worker(args)
        return
    rows = _run_matrix(args)
    _write_csv(args.out, rows)
    sys.stderr.write(f"wrote {len(rows)} rows to {args.out}\n")


def _worker_args(w: argparse.ArgumentParser) -> None:
    w.add_argument("--backend", required=True)
    w.add_argument("--dtype", default="bf16")
    w.add_argument("--circuit", default="ghz")
    w.add_argument("--n", type=int, default=8)
    w.add_argument("--mode", default="end-to-end")
    w.add_argument("--trials", type=int, default=5)
    w.add_argument("--gpu", type=int, default=0)
    w.add_argument("--micro-m", type=int, default=MICRO_M_DEFAULT)


if __name__ == "__main__":
    main()
