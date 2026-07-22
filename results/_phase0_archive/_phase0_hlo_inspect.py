"""Throwaway HLO inspect: ONE n=24 expectation case per process (XLA_FLAGS must be set before jax init).
Disambiguates the bit-identical fusion A/B peak + informs C2 coverability.
Run twice (fusion ON then OFF), compare the two dumped files:
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_hlo_inspect.py
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe bash .wsl_run.sh python results/_phase0_hlo_inspect.py --disable-fusion
"""
from __future__ import annotations
import hashlib
import os
import re
import sys

# Set XLA_FLAGS BEFORE importing jax (read at init).
DISABLE_FUSION = "--disable-fusion" in sys.argv
if DISABLE_FUSION:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_disable_hlo_passes=fusion").strip()

import jax  # noqa: E402
import tensorcircuit as tc  # noqa: E402

tc.set_backend("jax")


def _build_deep(n, depth):
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for _ in range(depth):
        for i in range(0, n - 1, 2):
            c.cnot(i, i + 1)
        for i in range(1, n - 1, 2):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rz(i, theta=0.7)
    return c


def _summarize(text):
    dots = len(re.findall(r"dot_general", text))
    fusions = len(re.findall(r"%fusion\.", text))
    dims = sorted((int(x) for x in re.findall(r"(?:f32|c64|bf16)\[(\d+)", text)), reverse=True)[:6]
    return {"len": len(text), "dot_general": dots, "fusion": fusions,
            "top6_dims": dims, "sha8": hashlib.sha1(text.encode()).hexdigest()[:8]}


def main():
    n, depth = 24, 10
    c = _build_deep(n, depth)
    f = jax.jit(lambda: c.expectation((tc.gates.z(), [0])))
    stablehlo = str(f.lower().compiler_ir(dialect="stablehlo"))
    compiled = f.lower().compile()
    jax.block_until_ready(compiled())
    peak = int(jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0))

    out = (f"# n={n} d={depth} expectation (stablehlo, pre-opt)\n"
           f"# peak_bytes_in_use = {peak} ({peak/1e9:.3f} GB)\n"
           f"# stablehlo: {_summarize(stablehlo)}\n")
    path = "results/_phase0_hlo_n24_stablehlo.txt"
    with open(path, "w") as fh:
        fh.write(out)
        fh.write("\n### stablehlo (head 4000 chars):\n")
        fh.write(stablehlo[:4000])
    print(out)
    print(f"# written to {path}")
    print("=== phase0_hlo_inspect done ===")


if __name__ == "__main__":
    main()
