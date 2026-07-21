"""Task 9 K3: capture XLA HLO for the 4M bf16 complex matmul on jax/GPU.

_pair_tensordot computes cr=ar.br-ai.bi; ci=ar.bi+ai.br (4 real bf16 matmuls).
PairTensor isn't jax-traceable, so we lower the identical raw-array computation.
Confirms XLA lowers the bf16 GEMMs to ``dot`` ops (cuBLAS/Tensor Core) and shows
how it schedules/fuses the 4 dots."""
import numpy as np
import jax
import jax.numpy as jnp

m = 1024
keys = jax.random.split(jax.random.key(0), 4)
ar = jax.random.normal(keys[0], (m, m), dtype=jnp.bfloat16)
ai = jax.random.normal(keys[1], (m, m), dtype=jnp.bfloat16)
br = jax.random.normal(keys[2], (m, m), dtype=jnp.bfloat16)
bi = jax.random.normal(keys[3], (m, m), dtype=jnp.bfloat16)


@jax.jit
def complex_matmul(ar, ai, br, bi):
    cr = ar @ br - ai @ bi
    ci = ar @ bi + ai @ br
    return cr, ci


hlo = str(complex_matmul.lower(ar, ai, br, bi).compiler_ir(dialect="stablehlo"))
n_bf16 = hlo.count("bf16")
n_dot = hlo.count("%dot")
print(f"HLO length: {len(hlo)} chars")
print(f"'bf16' occurrences: {n_bf16}")
print(f"'%dot' (dot op) occurrences: {n_dot}  (4 = one per real GEMM in the 4M complex mult)")
for line in hlo.splitlines():
    s = line.strip()
    if "dot" in s:
        print("  DOT:", s[:150])
print("--- raw HLO (first 400 chars) ---")
print(hlo[:400])
print("=== HLO probe done ===")
