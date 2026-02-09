r"""
VQE on QuditCircuits.

This example shows how to run a simple VQE on a qudit system using
`tensorcircuit.QuditCircuit`. We build a compact ansatz using single-qudit
rotations in selected two-level subspaces and RXX-type entanglers, then
optimize the energy of a Hermitian "clock-shift" Hamiltonian:

    H(d) = - J * (X_c \otimes X_c)  -  h * (Z_c \otimes I + I \otimes Z_c)

where, for local dimension `d`,
- Z_c = (Z + Z^\dagger)/2 is the Hermitian "clock" observable with Z = diag(1, \omega, \omega^2, ..., \omega^{d-1})
- X_c = (S + S^\dagger)/2 is the Hermitian "shift" observable with S the cyclic shift
- \omega = exp(2\pi i/d)

The code defaults to a 2-qutrit (d=3) problem but can be changed via CLI flags.
"""

# import os
# import sys
#
# base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if base_dir not in sys.path:
#     sys.path.insert(0, base_dir)

import time
import argparse
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")


def vqe_forward(param, *, nqudits: int, d: int, nlayers: int, J: float, h: float):
    r"""Build a QuditCircuit ansatz and compute :math:`\langle H\rangle`.

    Ansatz:
      [ for L in 1...nlayers ]
        - On each site q:
            :math:`RX(q; \theta_L q^{(01)}) RY(q; \theta_L q^{(12)}) RZ(q; \phi_L q^{(0)})`
          (subspace indices shown as superscripts)
        - Entangle neighboring pairs with RXX on subspaces (0,1)
    """
    if d < 3:
        raise ValueError("This example assumes d >= 3 (qutrit or higher).")

    S = tc.quditgates.x_matrix_func(d)
    Z = tc.quditgates.z_matrix_func(d)
    Sdag = tc.backend.adjoint(S)
    Zdag = tc.backend.adjoint(Z)

    c = tc.QuditCircuit(nqudits, dim=d)

    pairs = [(i, i + 1) for i in range(nqudits - 1)]

    it = iter(param)

    for _ in range(nlayers):
        for q in range(nqudits):
            c.rx(q, theta=next(it), j=0, k=1)
            c.ry(q, theta=next(it), j=1, k=2)
            c.rz(q, theta=next(it), j=0)

        for i, j in pairs:
            c.rxx(i, j, theta=next(it), j1=0, k1=1, j2=0, k2=1)

    # H = -J * 1/2 (S_i S_j^\dagger + S_i^\dagger S_j) - h * 1/2 (Z + Z^\dagger)
    energy = 0.0
    for i, j in pairs:
        e_ij = 0.5 * (
            c.expectation((S, [i]), (Sdag, [j])) + c.expectation((Sdag, [i]), (S, [j]))
        )
        energy += -J * tc.backend.real(e_ij)
    for q in range(nqudits):
        zq = 0.5 * (c.expectation((Z, [q])) + c.expectation((Zdag, [q])))
        energy += -h * tc.backend.real(zq)
    return tc.backend.real(energy)


def build_param_shape(nqudits: int, nlayers: int):
    # Per layer per qudit: RX^{(01)}, RY^{(12)} (or dummy), RZ^{(0)} = 3 params
    # Per layer entanglers: len(pairs) parameters
    pairs = nqudits - 1
    per_layer = 3 * nqudits + pairs
    return (nlayers * per_layer,)


def main():
    parser = argparse.ArgumentParser(
        description="VQE on QuditCircuit (clock-shift model)"
    )
    parser.add_argument(
        "--d", type=int, default=3, help="Local dimension per site (>=3)"
    )
    parser.add_argument("--nqudits", type=int, default=2, help="Number of sites")
    parser.add_argument("--nlayers", type=int, default=3, help="Ansatz depth (layers)")
    parser.add_argument(
        "--J", type=float, default=1.0, help="Coupling strength for XcXc term"
    )
    parser.add_argument(
        "--h", type=float, default=0.6, help="Field strength for Zc terms"
    )
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    assert args.d >= 3, "d must be >= 3"

    shape = build_param_shape(args.nqudits, args.nlayers)
    param = tc.backend.random_uniform(shape, boundaries=(-0.1, 0.1), seed=42)

    try:
        import optax

        optimizer = optax.adam(args.lr)
        vgf = tc.backend.jit(
            tc.backend.value_and_grad(
                lambda p: vqe_forward(
                    p,
                    nqudits=args.nqudits,
                    d=args.d,
                    nlayers=args.nlayers,
                    J=args.J,
                    h=args.h,
                )
            )
        )
        opt_state = optimizer.init(param)

        @tc.backend.jit
        def train_step(p, opt_state):
            loss, grads = vgf(p)
            updates, opt_state = optimizer.update(grads, opt_state, p)
            p = optax.apply_updates(p, updates)
            return p, opt_state, loss

        print("Starting VQE optimization (optax/adam)...")
        loss = None
        for i in range(args.steps):
            t0 = time.time()
            param, opt_state, loss = train_step(param, opt_state)
            # ensure sync for accurate timing
            _ = float(loss)
            if i % 20 == 0:
                dt = time.time() - t0
                print(f"Step {i:4d}  loss={loss:.6f}  dt/step={dt:.4f}s")
        print("Final loss:", float(loss) if loss is not None else "n/a")

    except ModuleNotFoundError:
        print("Optax not available; using naive gradient descent.")
        value_and_grad = tc.backend.value_and_grad(
            lambda p: vqe_forward(
                p,
                nqudits=args.nqudits,
                d=args.d,
                nlayers=args.nlayers,
                J=args.J,
                h=args.h,
            )
        )
        lr = args.lr
        loss = None
        for i in range(args.steps):
            loss, grads = value_and_grad(param)
            param = param - lr * grads
            if i % 20 == 0:
                print(f"Step {i:4d}  loss={float(loss):.6f}")
        print("Final loss:", float(loss) if loss is not None else "n/a")


if __name__ == "__main__":
    main()
