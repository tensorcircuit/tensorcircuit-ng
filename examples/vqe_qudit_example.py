r"""
You must set the backend explicitly via --backend {jax, tensorflow}.
AD-based optimization (gradient descent) is enabled for these backends.
A fallback random-search optimizer is also provided.

Example runs:
  python vqe_qudit_example.py --backend jax --optimizer gd --dim 3 --layers 2 --steps 200 --lr 0.1 --jit
  python vqe_qudit_example.py --backend tensorflow --optimizer gd --dim 3 --layers 2 --steps 200 --lr 0.1
  python vqe_qudit_example.py --backend jax --optimizer random --dim 3 --layers 2 --iters 300

What this script does:
  - Builds a 2-qudit (d>=3) ansatz with native RY/RZ single-qudit rotations on adjacent levels
    and an RXX entangler on (0,1) level pairs.
  - Minimizes the expectation of a simple 2-site Hermitian Hamiltonian:
        H = N(0) + N(1) + J * [ X_sym(0)\otimes X_sym(1) + Z_sym(0)\otimes Z_sym(1) ]
    where N = diag(0,1,...,d-1), X_sym = (X + X^\dagger)/2, Z_sym = (Z + Z^\dagger)/2.
"""

import argparse
import math
import sys
from typing import Sequence, Tuple

import numpy as np
import tensorcircuit as tc
from tensorcircuit.quditcircuit import QuditCircuit


# ---------- Hamiltonian helpers ----------
def symmetrize_hermitian(U: np.ndarray) -> np.ndarray:
    return 0.5 * (U + U.conj().T)


def build_2site_hamiltonian(
    d: int, J: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    N = np.diag(np.arange(d))
    Xsym = symmetrize_hermitian(tc.backend.numpy(tc.quditgates._x_matrix_func(d)))
    Zsym = symmetrize_hermitian(tc.backend.numpy(tc.quditgates._z_matrix_func(d)))
    H0 = N.copy()
    H1 = N.copy()
    return H0, H1, Xsym, Zsym, J


# ---------- Ansatz ----------


def apply_single_qudit_layer(c: QuditCircuit, qudit: int, thetas: Sequence) -> None:
    """
    Apply RY(j,j+1) then RZ(j) for each adjacent level pair.
    Number of params per site = 2*(d-1).
    """
    d = c._d
    idx = 0
    for j, k in [(p, p + 1) for p in range(d - 1)]:
        c.ry(qudit, theta=thetas[idx], j=j, k=k)
        idx += 1
        c.rz(qudit, theta=thetas[idx], j=j)
        idx += 1


def apply_entangler(c: QuditCircuit, theta) -> None:
    # generalized RXX on (0,1) level pair for both qudits
    c.rxx(0, 1, theta=theta, j1=0, k1=1, j2=0, k2=1)


def build_ansatz(nlayers: int, d: int, params: Sequence) -> QuditCircuit:
    c = QuditCircuit(2, dim=d)
    per_site = 2 * (d - 1)
    per_layer = 2 * per_site + 1  # two sites + entangler
    assert (
        len(params) == nlayers * per_layer
    ), f"params length {len(params)} != {nlayers * per_layer}"
    off = 0
    for _ in range(nlayers):
        th0 = params[off : off + per_site]
        off += per_site
        th1 = params[off : off + per_site]
        off += per_site
        thE = params[off]
        off += 1
        apply_single_qudit_layer(c, 0, th0)
        apply_single_qudit_layer(c, 1, th1)
        apply_entangler(c, thE)
    return c


# ---------- Energy ----------


def energy_expectation_backend(
    params_b,
    d: int,
    nlayers: int,
    ham: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
):
    """
    params_b: 1D backend tensor (jax/tf) of shape [nparams].
    Returns backend scalar.
    """
    bk = tc.backend
    # Keep differentiability by passing backend scalars into gates
    plist = [params_b[i] for i in range(params_b.shape[0])]
    c = build_ansatz(nlayers, d, plist)
    E = 0.0 + 0.0j
    H0, H1, Xsym, Zsym, J = ham
    # Local number operators
    E = E + c.expectation((tc.gates.Gate(H0), [0]))
    E = E + c.expectation((tc.gates.Gate(H1), [1]))
    # Coupling terms as products on separate sites (avoids 9x9 reshaping issues)
    E = E + J * c.expectation((tc.gates.Gate(Xsym), [0]), (tc.gates.Gate(Xsym), [1]))
    E = E + J * c.expectation((tc.gates.Gate(Zsym), [0]), (tc.gates.Gate(Zsym), [1]))
    return bk.real(E)


def energy_expectation_numpy(
    params_np: np.ndarray,
    d: int,
    nlayers: int,
    ham: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
) -> float:
    c = build_ansatz(nlayers, d, params_np.tolist())
    E = 0.0 + 0.0j
    H0, H1, Xsym, Zsym, J = ham
    E += c.expectation((tc.gates.Gate(H0), [0]))
    E += c.expectation((tc.gates.Gate(H1), [1]))
    # Coupling terms as products on separate sites (avoids 9x9 reshaping issues)
    E += J * c.expectation((tc.gates.Gate(Xsym), [0]), (tc.gates.Gate(Xsym), [1]))
    E += J * c.expectation((tc.gates.Gate(Zsym), [0]), (tc.gates.Gate(Zsym), [1]))
    return float(np.real(E))


# ---------- Optimizers ----------


def random_search(fun_numpy, x0_shape, iters=300, seed=42):
    rng = np.random.default_rng(seed)
    best_x, best_y = None, float("inf")
    for _ in range(iters):
        x = rng.uniform(-math.pi, math.pi, size=x0_shape)
        y = fun_numpy(x)
        if y < best_y:
            best_x, best_y = x, y
    return best_x, float(best_y)


def gradient_descent_ad(energy_bk, x0_np: np.ndarray, steps=200, lr=0.1, jit=False):
    """
    energy_bk: (backend_tensor[nparams]) -> backend_scalar
    Simple gradient descent in numpy space with backend-gradients.
    """
    bk = tc.backend
    if jit:
        energy_bk = bk.jit(energy_bk)
    grad_f = bk.grad(energy_bk)

    x_np = x0_np.copy()
    best_x, best_y = x_np.copy(), float("inf")

    def to_np(x):
        return x if isinstance(x, np.ndarray) else bk.numpy(x)

    for _ in range(steps):
        x_b = bk.convert_to_tensor(x_np)  # numpy -> backend tensor
        g_b = grad_f(x_b)  # backend gradient
        g = to_np(g_b)  # backend -> numpy
        x_np = x_np - lr * g  # SGD step in numpy
        y = float(to_np(energy_bk(bk.convert_to_tensor(x_np))))
        if y < best_y:
            best_x, best_y = x_np.copy(), y
    return best_x, float(best_y)


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser(description="Qudit VQE (explicit backend)")
    ap.add_argument(
        "--backend",
        required=True,
        choices=["jax", "tensorflow"],
        help="tensorcircuit backend",
    )
    ap.add_argument("--dim", type=int, default=3, help="local qudit dimension d (>=3)")
    ap.add_argument("--layers", type=int, default=2, help="# ansatz layers")
    ap.add_argument("--J", type=float, default=0.5, help="coupling strength")
    ap.add_argument(
        "--optimizer",
        type=str,
        default="gd",
        choices=["gd", "random"],
        help="gradient descent (AD) or random search",
    )
    ap.add_argument("--steps", type=int, default=200, help="GD steps")
    ap.add_argument("--lr", type=float, default=0.1, help="GD learning rate")
    ap.add_argument("--iters", type=int, default=300, help="random search steps")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument(
        "--jit",
        action="store_true",
        help="enable backend JIT (all backends implement .jit; numpy backend no-ops)",
    )
    args = ap.parse_args()

    tc.set_backend(args.backend)

    if args.dim < 3:
        print("Please use dim >= 3 for qudits.", file=sys.stderr)
        sys.exit(1)

    d, L = args.dim, args.layers
    per_layer = 4 * (d - 1) + 1
    nparams = L * per_layer

    ham = build_2site_hamiltonian(d, args.J)

    print(
        f"[info] backend={args.backend}, d={d}, layers={L}, params={nparams}, J={args.J}"
    )

    if args.optimizer == "random":

        def obj_np(theta_np):
            return energy_expectation_numpy(theta_np, d, L, ham)

        x, y = random_search(
            obj_np, x0_shape=(nparams,), iters=args.iters, seed=args.seed
        )
    else:

        def obj_bk(theta_b):
            return energy_expectation_backend(theta_b, d, L, ham)

        rng = np.random.default_rng(args.seed)
        x0 = rng.uniform(-math.pi, math.pi, size=(nparams,))
        x, y = gradient_descent_ad(
            obj_bk, x0_np=x0, steps=args.steps, lr=args.lr, jit=args.jit
        )

    print("\n=== Result ===")
    print(f"Energy      : {y:.6f}")
    print(f"Params shape: {x.shape}")
    np.set_printoptions(precision=4, suppress=True)
    print(x[: min(10, x.size)])


if __name__ == "__main__":
    main()
