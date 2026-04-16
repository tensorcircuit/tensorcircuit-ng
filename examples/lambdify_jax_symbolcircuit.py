"""
Use sympy.lambdify to lift a SymbolCircuit expectation expression
into a JAX numerical function that supports jit, grad, and vmap.

Gate parameters must be declared with real=True so that sympy can resolve
conjugate(sin(theta)) → sin(theta) when building the bra side of <psi|O|psi>.
Without real=True, conjugate() wrappers remain unevaluated and the symbolic
expression looks complex even though the value is always real for real inputs.
"""

import sympy
import jax
import jax.numpy as jnp

import tensorcircuit as tc

jax.config.update("jax_enable_x64", True)


def make_jax_fn(expr, symbols):
    """
    Lift a sympy expression into a JAX function.
    `symbols` is an ordered list of sympy.Symbol — callers must match that order.
    Passing [jnp, "math"] lets sympy map sin/cos/exp/... to jnp equivalents,
    with "math" as fallback for any scalar constant not in jnp.
    """
    return sympy.lambdify(symbols, expr, modules=[jnp, "math"])


# ── 1. Build all symbolic circuits BEFORE any tc.set_backend call ─────────────
# SymbolCircuit uses numpy object-dtype arrays internally; creating one after
# tc.set_backend("jax") would fail because JAX rejects object-dtype tensors.

theta = sympy.Symbol("theta", real=True)
phi = sympy.Symbol("phi", real=True)

sc = tc.SymbolCircuit(2)
sc.h(0)
sc.rx(1, theta=theta)
sc.rz(0, theta=phi)
sc.cnot(0, 1)

# ── 2. Symbolic expressions ───────────────────────────────────────────────────

expr_zz = sc.expectation_ps(z=[0, 1])
expr_y0 = sc.expectation_ps(y=[0])

print("Symbolic <ZZ>:", sympy.simplify(expr_zz))
print("Symbolic <Y0>:", sympy.simplify(expr_y0))

syms_zz = sorted(expr_zz.free_symbols, key=lambda s: s.name)
syms_y0 = sorted(expr_y0.free_symbols, key=lambda s: s.name)
print("\nfree(<ZZ>):", syms_zz)
print("free(<Y0>):", syms_y0)


# ── 4. lambdify → JAX functions ───────────────────────────────────────────────

f_zz = make_jax_fn(expr_zz, syms_zz)  # f_zz(theta)
f_y0 = make_jax_fn(expr_y0, syms_y0)  # f_y0(...) over whatever syms appear

# ── 5. Switch backend for cross-checks ───────────────────────────────────────

TH, PH = 0.7, 0.3
tc.set_backend("jax")
c_ref = sc.to_circuit({theta: TH, phi: PH})

print("\n--- scalar evaluation ---")
val_zz = f_zz(jnp.array(TH))
ref_zz = c_ref.expectation_ps(z=[0, 1])
print(f"f_zz(theta={TH}):", val_zz)
print(f"tc  <ZZ>:", ref_zz)
print("match:", jnp.allclose(jnp.array(val_zz, dtype=jnp.complex128), ref_zz))

# ── 6. JIT ────────────────────────────────────────────────────────────────────

print("\n--- jit ---")
f_zz_jit = jax.jit(f_zz)
print("jit f_zz(0.7):", f_zz_jit(jnp.array(TH)))

# ── 7. Gradient ───────────────────────────────────────────────────────────────

print("\n--- grad ---")
# With real=True the expression is genuinely real; jnp.real() is a no-op here
# but kept for safety since lambdify returns a JAX scalar of unspecified dtype.
df_dtheta = jax.grad(lambda th: jnp.real(f_zz(th)))
th0 = jnp.array(TH)
print("df_zz/dtheta (AD):", df_dtheta(th0))

eps = 1e-5
fd = (jnp.real(f_zz(th0 + eps)) - jnp.real(f_zz(th0 - eps))) / (2 * eps)
print("df_zz/dtheta (FD):", fd)

# ── 8. vmap over a parameter batch ───────────────────────────────────────────

print("\n--- vmap ---")
batch = jnp.linspace(0.0, jnp.pi, 9)
f_zz_vmap = jax.vmap(f_zz)
print("batch <ZZ> over [0, π]:", f_zz_vmap(batch))
