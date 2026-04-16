"""
One-layer QAOA for a weighted QUBO — tunable n, random 3-regular graph, closed-form symbolic expectation.

Generalises examples/qaoa_symbolic.py to an arbitrary even n with a random
3-regular graph and uniformly-sampled edge weights.

Outputs:
  * Graph edges with their random weights
  * Symbolic QUBO Hamiltonian H_QUBO in binary variables x_0, ..., x_{n-1}
  * Ising H_C after the substitution  x_i = (1 - Z_i) / 2
  * Closed-form QAOA expectation F(gamma, beta) = <H_C> as a sympy expression
  * 2-D energy landscape plot

Problem: weighted Max-Cut on a random 3-regular graph G(n)
----------------------------------------------------------
Each node has exactly 3 neighbours.  Edge (i,j) carries a weight w_{ij}
drawn uniformly from [0.5, 1.5] (rounded to 2 d.p.).

QUBO (minimisation, variables x_i in {0, 1}):

    min  f(x) = sum_i Q_diag[i] x_i + sum_{(i,j) in E} Q_off[(i,j)] x_i x_j

    Q_diag[i]    = -sum_{j: (i,j) in E} w_{ij}   (weighted degree)
    Q_off[(i,j)] = +2 w_{ij}

Ising mapping  (x_i = (1 - Z_i) / 2, H_C = -f(x) to maximise):

    H_C = const + sum_i h_i Z_i + sum_{(i,j) in E} J_{ij} Z_i Z_j

    const       = (1/2) sum_i deg_w(i) - (1/2) sum_{(i,j)} w_{ij}
    h_i         = -(1/2) deg_w(i) + (1/2) sum_{j: (i,j) in E} w_{ij}   (= 0 for uniform w)
    J_{ij}      = -(1/2) w_{ij}

    (For non-uniform weights h_i need not vanish.)

QAOA ansatz (1 layer):
    |psi> = U_M(beta) U_C(gamma) |+>^n
    U_C(gamma) = exp(-i gamma H_C)  -> RZ + RZZ gates
    U_M(beta)  = prod_i RX(2 beta)
"""

import argparse
import sys
import networkx as nx
import numpy as np
import sympy
import matplotlib.pyplot as plt
import tensorcircuit as tc

# ── graph / QUBO helpers ──────────────────────────────────────────────────────


def build_random_3regular_qubo(n: int, rng: np.random.Generator):
    """Return QUBO coefficients for weighted Max-Cut on a random 3-regular graph.

    Edge weights drawn uniformly from [0.5, 1.5], rounded to 2 decimal places.
    Requires n even (and n >= 4) so that a 3-regular graph exists.

    Returns
    -------
    edges   : list of (i, j, w) triples with i < j
    Q_diag  : dict  i -> diagonal QUBO coefficient
    Q_off   : dict (i,j) -> off-diagonal QUBO coefficient  (i < j)
    """
    if n % 2 != 0 or n < 4:
        raise ValueError("n must be even and >= 4 for a 3-regular graph to exist.")

    G = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**31)))
    edges = []
    for u, v in G.edges():
        i, j = min(u, v), max(u, v)
        w = round(float(rng.uniform(0.5, 1.5)), 2)
        edges.append((i, j, w))
    edges.sort()

    Q_diag: dict = {i: 0.0 for i in range(n)}
    Q_off: dict = {}
    for i, j, w in edges:
        Q_diag[i] -= w
        Q_diag[j] -= w
        Q_off[(i, j)] = 2.0 * w

    return edges, Q_diag, Q_off


# ── symbolic helpers ──────────────────────────────────────────────────────────


def qubo_sympy_expr(n: int, Q_diag: dict, Q_off: dict):
    """Return the QUBO Hamiltonian as a sympy expression in x_0,...,x_{n-1}."""
    x = [sympy.Symbol(f"x_{i}", nonneg=True) for i in range(n)]
    H = sympy.Integer(0)
    for i, q in sorted(Q_diag.items()):
        H += sympy.Float(round(q, 6)) * x[i]
    for (i, j), q in sorted(Q_off.items()):
        H += sympy.Float(round(q, 6)) * x[i] * x[j]
    return H, x


def qubo_to_ising(Q_diag: dict, Q_off: dict):
    """Convert QUBO coefficients to Ising (const, h, J) form.

    QUBO:  minimise  sum_i Q_diag[i] x_i + sum_{i<j} Q_off[(i,j)] x_i x_j
    Ising: maximise  const + sum_i h[i] Z_i + sum_{i<j} J[(i,j)] Z_i Z_j
    """
    const = 0.0
    h: dict = {}
    J: dict = {}
    for i, q in Q_diag.items():
        const -= q / 2
        h[i] = h.get(i, 0.0) + q / 2
    for (i, j), q in Q_off.items():
        const -= q / 4
        h[i] = h.get(i, 0.0) + q / 4
        h[j] = h.get(j, 0.0) + q / 4
        J[(i, j)] = J.get((i, j), 0.0) - q / 4
    return const, h, J


def ising_sympy_expr(n: int, const: float, h: dict, J: dict):
    """Return the Ising Hamiltonian as a sympy expression in Z_0,...,Z_{n-1}."""
    Z = [sympy.Symbol(f"Z_{i}") for i in range(n)]
    H = sympy.Float(round(const, 6))
    for i, hi in sorted(h.items()):
        if abs(hi) > 1e-12:
            H += sympy.Float(round(hi, 6)) * Z[i]
    for (i, j), Jij in sorted(J.items()):
        if abs(Jij) > 1e-12:
            H += sympy.Float(round(Jij, 6)) * Z[i] * Z[j]
    return H, Z


# ── QAOA circuit ──────────────────────────────────────────────────────────────


def build_qaoa(n: int, h: dict, J: dict, gamma, beta) -> tc.SymbolCircuit:
    """Build a 1-layer QAOA SymbolCircuit with symbolic (gamma, beta)."""
    tc.set_backend("numpy")
    sc = tc.SymbolCircuit(n)

    for i in range(n):
        sc.h(i)

    for i, hi in sorted(h.items()):
        if abs(hi) > 1e-12:
            sc.rz(i, theta=2 * hi * gamma)
    for (i, j), Jij in sorted(J.items()):
        if abs(Jij) > 1e-12:
            sc.rzz(i, j, theta=2 * Jij * gamma)

    for i in range(n):
        sc.rx(i, theta=2 * beta)

    return sc


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QAOA symbolic example on a random 3-regular graph"
    )
    parser.add_argument(
        "--n", type=int, default=6, help="Number of qubits (even, >= 4; default: 6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for graph and weights (default: 0)",
    )
    args = parser.parse_args()
    n = args.n
    rng = np.random.default_rng(args.seed)

    sep = "=" * 60
    print(sep)
    print(f"1-layer QAOA  |  random 3-regular graph  |  n={n}, seed={args.seed}")
    print(sep)
    print()

    # ── Graph + QUBO ──────────────────────────────────────────────────────────
    edges, Q_diag, Q_off = build_random_3regular_qubo(n, rng)

    print("Graph edges  (i, j, weight):")
    for i, j, w in edges:
        print(f"  ({i}, {j})  w = {w:.2f}")
    print()

    H_qubo, _ = qubo_sympy_expr(n, Q_diag, Q_off)
    print("QUBO Hamiltonian H_QUBO(x)  [binary variables x_i ∈ {0,1}, minimise]:")
    print(" ", sympy.expand(H_qubo))
    print()

    # ── Ising ─────────────────────────────────────────────────────────────────
    const, h, J = qubo_to_ising(Q_diag, Q_off)
    H_ising, _ = ising_sympy_expr(n, const, h, J)

    print("Ising Hamiltonian H_C(Z)  [spin variables Z_i ∈ {-1,+1}, maximise]:")
    print(" ", H_ising)
    print()

    # ── Symbolic QAOA expectation ─────────────────────────────────────────────
    gamma = sympy.Symbol("gamma", real=True)
    beta = sympy.Symbol("beta", real=True)

    sc = build_qaoa(n, h, J, gamma, beta)
    print(f"QAOA circuit: {sc.gate_count()} gates, free symbols: {sc.free_symbols()}")
    print()

    print("Computing symbolic expectation F(γ,β) = ⟨H_C⟩ …")
    F = sympy.Integer(0) + const
    for i, hi in h.items():
        if abs(hi) > 1e-12:
            F += hi * sc.expectation_ps(z=[i], enable_lightcone=True)
    for (i, j), Jij in J.items():
        if abs(Jij) > 1e-12:
            F += Jij * sc.expectation_ps(z=[i, j], enable_lightcone=True)

    F = sympy.re(F)
    # F = sympy.trigsimp(F.rewrite(sympy.cos))
    F_expanded = sympy.expand(F)
    print()
    print("Closed-form expectation  F(γ,β) = ⟨H_C⟩:")
    print(" ", F_expanded)
    print()

    # ── 2-D landscape ─────────────────────────────────────────────────────────

    sys.setrecursionlimit(5000)
    F_func = sympy.lambdify([gamma, beta], F, "numpy", cse=True)

    gammas = np.linspace(1e-6, np.pi, 300)
    betas = np.linspace(0, np.pi / 2, 300)
    GG, BB = np.meshgrid(gammas, betas)
    FF_loss = -F_func(GG, BB).real  # QUBO cost = -⟨H_C⟩

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(GG, BB, FF_loss, shading="auto", cmap="RdYlGn_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\langle f(x)\rangle(\gamma,\beta)$ (QUBO cost)", fontsize=12)
    ax.set_xlabel(r"$\gamma$", fontsize=13)
    ax.set_ylabel(r"$\beta$", fontsize=13)
    ax.set_title(
        f"1-layer QAOA energy landscape  (random 3-reg, $n={n}$, seed={args.seed})\n"
        r"$F(\gamma,\beta)$ from closed-form symbolic expression",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
