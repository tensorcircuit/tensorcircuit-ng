"""
Gradient descent VQE of a two-qubit transverse field Ising model,
H = -Z0Z1 - X0 - X1, on a quantum cloud device, with the gradient from the
parameter shift rule.

The ansatz is compiled once for the device as a ``SymbolCircuit``. Each step binds
the current parameters and their ``±pi/2`` shifts, and evaluates all of them in
one ``batch_expectation_ps`` call, i.e. one batch of 2 x 9 circuits for the two
qubit-wise commuting groups {Z0Z1} and {X0, X1}. Without ``--device``, the
energies are exact.

.. code-block:: bash

    python cloud_parameter_shift_vqe.py
    export TC_TOKEN_LQCLOUD=<token>
    python cloud_parameter_shift_vqe.py --device lqcloud::<device> --shots 1000
"""

import argparse

import numpy as np
import sympy

import tensorcircuit as tc

pss = [[3, 3], [1, 0], [0, 1]]
ws = [-1.0, -1.0, -1.0]
thetas = sympy.symbols("theta0:4", real=True)


def ansatz() -> tc.SymbolCircuit:
    c = tc.SymbolCircuit(2)
    c.ry(0, theta=thetas[0])
    c.ry(1, theta=thetas[1])
    c.cz(0, 1)
    c.ry(0, theta=thetas[2])
    c.ry(1, theta=thetas[3])
    return c


def energies(c, values, device, info, shots):
    params = [dict(zip(thetas, v)) for v in values]
    es = tc.cloud.wrapper.batch_expectation_ps(
        c, pss, device=device, ws=ws, shots=shots, params=params, info=info
    )
    return np.real(tc.backend.numpy(es))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="e.g. lqcloud::<device>")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.2)
    args = parser.parse_args()

    c, info = ansatz(), None
    if args.device is not None:
        tc.cloud.apis.set_store(True)
        properties = tc.cloud.apis.get_device(args.device).list_properties()
        c, info = tc.compiler.DefaultCompiler(properties=properties)(c)
        print("physical qubits:", info["logical_physical_mapping"])

    theta = np.full(4, -0.5)
    shifts = np.pi / 2 * np.eye(4)
    exact = np.linalg.eigvalsh(tc.quantum.PauliStringSum2Dense(pss, ws))[0]
    print("exact ground energy:", np.real(exact))
    for step in range(args.steps):
        values = [theta] + [theta + sign * s for s in shifts for sign in (1, -1)]
        es = energies(c, values, args.device, info, args.shots)
        grad = (es[1::2] - es[2::2]) / 2
        print(f"step {step}: energy {es[0]:.4f}, gradient {np.round(grad, 4)}")
        theta = theta - args.lr * grad
    e = energies(c, [theta], args.device, info, args.shots)[0]
    print(f"step {args.steps}: energy {e:.4f}")


if __name__ == "__main__":
    main()
