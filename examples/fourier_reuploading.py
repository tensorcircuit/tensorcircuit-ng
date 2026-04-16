"""
Fourier series structure of data re-uploading PQC.

Reference: Schuld et al., "The effect of data encoding on the expressive
power of variational quantum-classical algorithms", PRA 103, 032430 (2021).

Circuit: L-layer single-qubit re-uploading.
  Per layer: Ry(θ_l) · Rz(φ_l) · Rz(x)
  Observable: <X>

Rz(x) is the data-encoding gate.  Rz(φ_l+x) = Rz(φ_l)·Rz(x) introduces a
phase between the |0⟩ and |1⟩ amplitudes that depends on x; the off-diagonal
observable <X> is sensitive to this phase, making the output a Fourier series
in x.  With L layers, the accessible frequencies are integers {-L,...,0,...,L},
giving a truncated Fourier series of degree L.

All SymbolCircuit construction must happen BEFORE tc.set_backend("jax").
"""

import sympy
import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc

# ── helpers ────────────────────────────────────────────────────────────────────


def make_jax_fn(expr, syms):
    return sympy.lambdify(syms, expr, modules=[jnp, "math"])


def build_reuploading(L, x_sym):
    """
    Single-qubit L-layer re-uploading: [Ry(θ_l) · Rz(φ_l) · Rz(x)]^L.
    Rz(φ_l)·Rz(x) = Rz(φ_l+x): trainable phase + data encoding combined.
    Returns (SymbolCircuit, param_syms) where param_syms = [θ_0,...,φ_0,...].
    """
    thetas = [sympy.Symbol(f"t{l}", real=True) for l in range(L)]
    phis = [sympy.Symbol(f"p{l}", real=True) for l in range(L)]
    sc = tc.SymbolCircuit(1)
    for l in range(L):
        sc.ry(0, theta=thetas[l])
        sc.rz(0, theta=phis[l])  # trainable phase
        sc.rz(0, theta=x_sym)  # data encoding
    return sc, thetas + phis


def fourier_coeffs_sym(expr, x_sym, L):
    """
    Symbolically extract Fourier coefficients via trig orthogonality:
      a_k = (1/π) ∫₀^{2π} f(x) cos(kx) dx    (a_0 uses 1/2π)
      b_k = (1/π) ∫₀^{2π} f(x) sin(kx) dx
    Returns (a_dict, b_dict).  Results are rewritten in trig form.
    """
    pi = sympy.pi

    def _trig(e):
        return sympy.trigsimp(sympy.expand(e.rewrite(sympy.cos)))

    a, b = {}, {}
    a[0] = _trig(sympy.integrate(expr, (x_sym, 0, 2 * pi)) / (2 * pi))
    for k in range(1, L + 1):
        a[k] = _trig(
            sympy.integrate(expr * sympy.cos(k * x_sym), (x_sym, 0, 2 * pi)) / pi
        )
        b[k] = _trig(
            sympy.integrate(expr * sympy.sin(k * x_sym), (x_sym, 0, 2 * pi)) / pi
        )
    return a, b


def dft_power(expr, x_sym, params, param_vals, N=512):
    """
    Evaluate expr numerically at N evenly-spaced x in [0, 2π],
    then compute the DFT power spectrum.
    """
    param_dict = dict(zip(params, param_vals))
    expr_x = expr.subs(param_dict)
    f_x = sympy.lambdify(x_sym, expr_x, modules="numpy")
    x_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)
    y = np.array([float(np.real(f_x(xi))) for xi in x_grid])
    ck = np.fft.rfft(y) / N
    return np.arange(len(ck)), np.abs(ck) ** 2


# ══ build all circuits BEFORE tc.set_backend ══════════════════════════════════

x = sympy.Symbol("x", real=True)
L_MAX = 3

circuits = {}  # L → (sc, params, expr_X)
for L in range(1, L_MAX + 1):
    sc, params = build_reuploading(L, x)
    # <X> = expectation of Pauli X on qubit 0
    expr_X = sc.expectation_ps(x=[0])
    circuits[L] = (sc, params, expr_X)


# ── 1. Symbolic expressions ────────────────────────────────────────────────────

print("=== Symbolic <X>(x; θ, φ) ===")
print()
for L in range(1, L_MAX + 1):
    _, params, expr = circuits[L]
    fully = sympy.trigsimp(sympy.expand(expr))
    n_add = len(fully.as_ordered_terms())
    print(f"L={L}: {n_add} additive terms")
    print(f"  {fully}")
print()


# ── 2. Symbolic Fourier coefficients for L=1 ──────────────────────────────────
# L=1: <X> = sin(t0)·cos(x + p0) = sin(t0)cos(p0)·cos(x) − sin(t0)sin(p0)·sin(x)
# Both a_1 and b_1 are controlled independently via t0 (amplitude) and p0 (phase).

print("=== Symbolic Fourier coefficients, L=1 ===")
_, params1, expr1 = circuits[1]
a, b = fourier_coeffs_sym(expr1, x, L=1)
print(f"  a_0 = {a[0]}")
print(f"  a_1 = {a[1]}    (cos x coefficient)")
print(f"  b_1 = {b[1]}    (sin x coefficient)")

# Verify: frequency-2 coefficient is zero for L=1
a2_check = sympy.simplify(
    sympy.integrate(expr1 * sympy.cos(2 * x), (x, 0, 2 * sympy.pi)) / sympy.pi
)
print(f"  a_2 = {a2_check}    (should be 0 — freq-2 inaccessible at L=1)")
print()


# ── 3. Numerical Fourier power spectrum (fixed random params) ─────────────────
# Peaks above frequency L must be zero regardless of parameter values.

print("=== Fourier power spectrum (DFT, fixed random params) ===")
rng = np.random.default_rng(0)
for L in range(1, L_MAX + 1):
    _, params, expr = circuits[L]
    param_vals = rng.uniform(0.3, 2.0, size=len(params))
    freqs, power = dft_power(expr, x, params, param_vals)
    shown = "  ".join(f"k={k}: {power[k]:.4f}" for k in range(L + 3))
    print(f"  L={L}: {shown}")
print()


# ── 4. JAX gradient fitting ────────────────────────────────────────────────────
# Target: sin(2x) — a pure second harmonic.
#   L=1: b_2 = 0 identically → cannot represent sin(2x) → MSE stays at ~0.5.
#   L=2: b_2 free → can represent sin(2x) → MSE → 0.
#   L=3: same, with more params.
#
# Analytically (from the derivation):
#   L=2 achieves sin(2x) exactly with:
#     t1=0, t0=π/2, p0+p1=−π/2   →   <X> = sin(2x)

tc.set_backend("jax")

target_fn = lambda xv: jnp.sin(2.0 * xv)


def fit(L, n_steps=2000, lr=0.05, seed=0):
    _, params, expr = circuits[L]
    all_syms = [x] + params
    f_jax = make_jax_fn(expr, all_syms)

    # batch evaluation: vmap over x, broadcast params
    def predict(pv, xb):
        return jax.vmap(lambda xi: jnp.real(f_jax(xi, *pv)))(xb)

    @jax.jit
    def loss(pv, xb, yb):
        return jnp.mean((predict(pv, xb) - yb) ** 2)

    grad_loss = jax.jit(jax.grad(loss))

    # try multiple seeds, keep best
    best_mse, best_pv = jnp.inf, None
    x_train = jnp.linspace(0.0, 2.0 * jnp.pi, 80)
    y_train = target_fn(x_train)
    for s in range(3):
        key = jax.random.PRNGKey(seed + s)
        pv = jax.random.uniform(key, (len(params),), minval=0.1, maxval=2.0)
        for _ in range(n_steps):
            pv = pv - lr * grad_loss(pv, x_train, y_train)
        mse = float(loss(pv, x_train, y_train))
        if mse < best_mse:
            best_mse, best_pv = mse, pv

    return best_mse, np.array(best_pv)


print("=== Gradient fitting of target sin(2x) ===")
print("    (MSE ≈ 0.5 = Var[sin(2x)] means the circuit cannot represent the target)")
for L in range(1, L_MAX + 1):
    mse, pv = fit(L, n_steps=2000)
    print(f"  L={L} ({len(pv)} params): final MSE = {mse:.6f}")
print()


# ── 5. Analytical verification for L=2 ────────────────────────────────────────
# Show the exact parameter values that make <X> = sin(2x), then cross-check
# with the numerical tc circuit.

print("=== Analytical solution: L=2, <X> = sin(2x) ===")
# t1=0, t0=π/2, p0+p1=−π/2  →  pick p0=0, p1=−π/2
t0_sol, t1_sol = np.pi / 2, 0.0
p0_sol, p1_sol = 0.0, -np.pi / 2

_, params2, expr2 = circuits[2]
all_syms2 = [x] + params2
f2 = make_jax_fn(expr2, all_syms2)

x_check = jnp.linspace(0.0, 2.0 * jnp.pi, 200)
pv_sol = jnp.array([t0_sol, t1_sol, p0_sol, p1_sol])
y_model = jax.vmap(lambda xi: jnp.real(f2(xi, *pv_sol)))(x_check)
y_target = target_fn(x_check)
mse_sol = float(jnp.mean((y_model - y_target) ** 2))
print(f"  params: t0={t0_sol:.4f}, t1={t1_sol:.4f}, p0={p0_sol:.4f}, p1={p1_sol:.4f}")
print(f"  MSE against sin(2x): {mse_sol:.2e}  (numerical zero)")

# Cross-check with tc numerical circuit (bind x too, since it's also a gate param)
sc2, params2, _ = circuits[2]
param_dict_sol = dict(zip(params2, [t0_sol, t1_sol, p0_sol, p1_sol]))
x_sample = 0.7
c_num = sc2.to_circuit({**param_dict_sol, x: x_sample})
ref = float(jnp.real(c_num.expectation_ps(x=[0])))
model_val = float(jnp.real(f2(jnp.array(x_sample), *pv_sol)))
print(f"  at x=0.7: lambdify = {model_val:.8f},  tc.Circuit = {ref:.8f}")
