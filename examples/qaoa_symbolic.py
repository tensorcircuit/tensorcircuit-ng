"""
One-layer QAOA for a QUBO Hamiltonian — closed-form symbolic expectation.

This example uses ``tc.SymbolCircuit`` to derive the QAOA cost-function
expectation

    F(gamma, beta) = <psi(gamma, beta)| H_C |psi(gamma, beta)>

as a closed-form sympy expression in the two variational parameters gamma
(phase separator) and beta (mixer), then cross-validates against a numeric
tc.Circuit evaluation and plots the energy landscape.

Problem: Max-Cut on the 3-node path graph P3  (0 -- 1 -- 2)
---------------------------------------------------------
QUBO (minimisation form, variables x_i in {0, 1}):

    min  f(x) = Q_00 x_0 + Q_11 x_1 + Q_22 x_2
                + Q_01 x_0 x_1 + Q_12 x_1 x_2

    Q_diag = {0: -1, 1: -2, 2: -1}
    Q_off  = {(0,1): +2, (1,2): +2}

    (f(x) = -MaxCut(x), so minimising f maximises the cut.)

Ising mapping  (x_i = (1 - Z_i) / 2, H_C = -f(x)):
    H_C  =  1  -  (Z_0 Z_1 + Z_1 Z_2) / 2        (to maximise)

QAOA ansatz (1 layer):
    |psi(gamma, beta)> = U_M(beta) U_C(gamma) |+>^3

    U_C(gamma) = exp(-i gamma H_C)
                 x RZZ(-gamma) on (0,1)  and  RZZ(-gamma) on (1,2)

    U_M(beta)  = prod_i RX(2 beta)
"""

import sympy
import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

# ── QUBO → Ising helpers ──────────────────────────────────────────────────────


def qubo_to_ising(Q_diag: dict, Q_off: dict):
    """Convert QUBO coefficients to Ising (const, h, J) form.

    QUBO:  minimise sum_i Q_diag[i] x_i + sum_{i<j} Q_off[(i,j)] x_i x_j
    Ising: maximise  const + sum_i h[i] Z_i + sum_{i<j} J[(i,j)] Z_i Z_j

    The sign flip converts min→max.
    """
    const = 0.0
    h: dict = {}
    J: dict = {}
    for i, q in Q_diag.items():
        # -q * x_i = -q * (1 - Z_i)/2 → const term -q/2, linear term +q/2 Z_i
        const -= q / 2
        h[i] = h.get(i, 0) + q / 2
    for (i, j), q in Q_off.items():
        # -q * x_i x_j = -q (1-Z_i)(1-Z_j)/4
        # constant: -q/4,  linear: +q/4 Z_i + q/4 Z_j,  quadratic: -q/4 Z_i Z_j
        const -= q / 4
        h[i] = h.get(i, 0) + q / 4
        h[j] = h.get(j, 0) + q / 4
        J[(i, j)] = J.get((i, j), 0) - q / 4
    return const, h, J


# ── QAOA circuit builder ──────────────────────────────────────────────────────


def build_qaoa(n, const, h, J, gamma, beta):
    """Return a SymbolCircuit for 1-layer QAOA with symbolic (gamma, beta)."""
    tc.set_backend("numpy")
    sc = tc.SymbolCircuit(n)

    # Initial state |+>^n = H^n |0>^n
    for i in range(n):
        sc.h(i)

    # Cost unitary: exp(-i gamma H_C)
    #   exp(-i gamma h_i Z_i)       = RZ(2 gamma h_i)
    #   exp(-i gamma J_ij Z_i Z_j)  = RZZ(2 gamma J_ij)
    for i, hi in sorted(h.items()):
        if hi:
            sc.rz(i, theta=2 * hi * gamma)
    for (i, j), Jij in sorted(J.items()):
        if Jij:
            sc.rzz(i, j, theta=2 * Jij * gamma)

    # Mixer unitary: exp(-i beta sum X_i) = prod RX(2 beta)
    for i in range(n):
        sc.rx(i, theta=2 * beta)

    return sc


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    # ── Problem ───────────────────────────────────────────────────────────────
    n = 3
    # MaxCut QUBO (minimisation): f(x) = -MaxCut(x), P3 path graph (0--1--2)
    #   Q_diag[i] = -degree(i),  Q_off[(i,j)] = +2 per edge
    Q_diag = {0: -1, 1: -2, 2: -1}
    Q_off = {(0, 1): 2, (1, 2): 2}

    print("=" * 60)
    print("QUBO: minimise  f(x) = x^T Q x  (= -MaxCut(x))")
    print("Q_diag =", Q_diag)
    print("Q_off  =", Q_off)
    print("(Minimising f = Maximising MaxCut; optimal cut = 2)")
    print()

    # ── Ising Hamiltonian ─────────────────────────────────────────────────────
    const, h, J = qubo_to_ising(Q_diag, Q_off)
    print("Ising Hamiltonian H_C (to maximise):")
    terms = [f"{const:+.4f}"]
    for i, hi in sorted(h.items()):
        if hi:
            terms.append(f"{hi:+.4f} Z_{i}")
    for (i, j), Jij in sorted(J.items()):
        if Jij:
            terms.append(f"{Jij:+.4f} Z_{i}Z_{j}")
    print("  H_C =", "  ".join(terms))
    print()

    # ── Symbolic parameters ───────────────────────────────────────────────────
    gamma = sympy.Symbol("gamma", real=True)
    beta = sympy.Symbol("beta", real=True)

    # ── Build circuit ─────────────────────────────────────────────────────────
    sc = build_qaoa(n, const, h, J, gamma, beta)
    print(
        f"QAOA circuit: {sc.gate_count()} gates, " f"free symbols: {sc.free_symbols()}"
    )
    print()

    # ── Symbolic expectation F(gamma, beta) = <H_C> ──────────────────────────
    F = sympy.Integer(0) + const
    for i, hi in h.items():
        if hi:
            F = F + hi * sc.expectation_ps(z=[i], enable_lightcone=True)
    for (i, j), Jij in J.items():
        if Jij:
            F = F + Jij * sc.expectation_ps(z=[i, j], enable_lightcone=True)

    F = sympy.re(F)
    F = sympy.trigsimp(F.rewrite(sympy.cos))
    print("Closed-form expectation  F(gamma, beta) = <H_C>:")
    print(" ", F)
    print()

    # ── Symbolic gradient ─────────────────────────────────────────────────────
    dF_dgamma = sympy.trigsimp(sympy.diff(F, gamma))
    dF_dbeta = sympy.trigsimp(sympy.diff(F, beta))
    print("dF/dgamma =", dF_dgamma)
    print("dF/dbeta  =", dF_dbeta)
    print()

    # ── Numerical cross-validation ────────────────────────────────────────────
    print("Numerical cross-validation (symbolic vs tc.Circuit):")
    test_points = [(0.5, 0.3), (np.pi / 3, np.pi / 8), (1.0, 0.7)]
    for gv, bv in test_points:
        sym_val = float(F.subs({gamma: gv, beta: bv}))
        c = sc.to_circuit({gamma: gv, beta: bv})
        ref_val = float(const)
        for i, hi in h.items():
            if hi:
                ref_val += float(hi) * float(c.expectation_ps(z=[i]).real)
        for (i, j), Jij in J.items():
            if Jij:
                ref_val += float(Jij) * float(c.expectation_ps(z=[i, j]).real)
        print(
            f"  gamma={gv:.4f}, beta={bv:.4f}: "
            f"sym={sym_val:.6f}  ref={ref_val:.6f}  "
            f"ok={abs(sym_val - ref_val) < 1e-6}"
        )
    print()

    # ── Landscape: lambdify for fast numpy evaluation ─────────────────────────
    F_func = sympy.lambdify([gamma, beta], F, "numpy", cse=True)

    gammas = np.linspace(1e-6, np.pi, 200)
    betas = np.linspace(0, np.pi / 2, 200)
    GG, BB = np.meshgrid(gammas, betas)
    FF = F_func(GG, BB)

    # QUBO cost landscape: <f(x)> = -<H_C>  (minimisation framing)
    FF_loss = -FF

    # Optimal parameters (grid-search on QUBO cost → argmin)
    idx = np.unravel_index(np.argmin(FF_loss), FF_loss.shape)
    g_opt, b_opt, F_opt = GG[idx], BB[idx], FF_loss[idx]
    print(
        f"Grid-search optimum:  gamma*={g_opt:.4f}  beta*={b_opt:.4f}  "
        f"QUBO cost*={F_opt:.4f}  (ideal = {-2:.4f})"
    )
    print()

    # ── Plot 2-D energy landscape ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(GG, BB, FF_loss, shading="auto", cmap="RdYlGn_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\langle f(x)\rangle(\gamma,\beta)$ (QUBO cost)", fontsize=12)

    ax.scatter(
        g_opt,
        b_opt,
        marker="*",
        s=200,
        color="white",
        zorder=5,
        label=f"minimum ({g_opt:.2f}, {b_opt:.2f})",
    )
    ax.set_xlabel(r"$\gamma$", fontsize=13)
    ax.set_ylabel(r"$\beta$", fontsize=13)
    ax.set_title(
        "1-layer QAOA energy landscape\n"
        r"$F(\gamma,\beta)$ from symbolic expression (Max-Cut P3)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()

    out = "examples/qaoa_symbolic_landscape.png"
    plt.savefig(out, dpi=150)
    print(f"Landscape plot saved to {out}")


if __name__ == "__main__":
    main()
