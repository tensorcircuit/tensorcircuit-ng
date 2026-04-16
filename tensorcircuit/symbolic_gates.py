"""
Symbolic gate matrix factories using sympy.

These return Gate objects containing numpy object arrays with sympy expressions,
compatible with tn.Node and the tc.contractor contraction pipeline.

Convention: for an n-qubit gate, the tensor is shaped [2]*2n with the first n axes
being output legs and the last n axes being input legs, matching the Gate convention
in tensorcircuit.
"""

from typing import Any

import numpy as np
import sympy

from .gates import Gate

# ── helpers ───────────────────────────────────────────────────────────────────


def _sym_gate(m: np.ndarray) -> Gate:  # type: ignore[type-arg]
    """Wrap a numpy object array as a Gate, reshaping an n x n matrix to rank-2n."""
    size = m.size
    n = int(round(np.log2(size)))
    return Gate(m.reshape([2] * n))


# ── fixed (parameter-free) gates ─────────────────────────────────────────────


def sym_i() -> Gate:
    return _sym_gate(np.array([[1, 0], [0, 1]], dtype=object))


def sym_x() -> Gate:
    return _sym_gate(np.array([[0, 1], [1, 0]], dtype=object))


def sym_y() -> Gate:
    return _sym_gate(np.array([[0, -sympy.I], [sympy.I, 0]], dtype=object))


def sym_z() -> Gate:
    return _sym_gate(np.array([[1, 0], [0, -1]], dtype=object))


def sym_h() -> Gate:
    v = sympy.Rational(1, 1) / sympy.sqrt(2)
    return _sym_gate(np.array([[v, v], [v, -v]], dtype=object))


def sym_s() -> Gate:
    return _sym_gate(np.array([[1, 0], [0, sympy.I]], dtype=object))


def sym_t() -> Gate:
    return _sym_gate(
        np.array([[1, 0], [0, sympy.exp(sympy.pi * sympy.I / 4)]], dtype=object)
    )


def sym_sd() -> Gate:
    return _sym_gate(np.array([[1, 0], [0, -sympy.I]], dtype=object))


def sym_td() -> Gate:
    return _sym_gate(
        np.array([[1, 0], [0, sympy.exp(-sympy.pi * sympy.I / 4)]], dtype=object)
    )


def sym_wroot() -> Gate:
    v = sympy.Rational(1, 1) / sympy.sqrt(2)
    return _sym_gate(
        np.array(
            [
                [v, -v * (1 + sympy.I) / sympy.sqrt(2)],
                [v * (1 - sympy.I) / sympy.sqrt(2), v],
            ],
            dtype=object,
        )
    )


def sym_cnot() -> Gate:
    m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=object)
    return _sym_gate(m)


def sym_cz() -> Gate:
    m = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=object
    )
    return _sym_gate(m)


def sym_cy() -> Gate:
    m = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -sympy.I], [0, 0, sympy.I, 0]],
        dtype=object,
    )
    return _sym_gate(m)


def sym_swap() -> Gate:
    m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=object)
    return _sym_gate(m)


def sym_ox() -> Gate:
    m = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=object)
    return _sym_gate(m)


def sym_oy() -> Gate:
    m = np.array(
        [[0, -sympy.I, 0, 0], [sympy.I, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=object,
    )
    return _sym_gate(m)


def sym_oz() -> Gate:
    m = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=object
    )
    return _sym_gate(m)


def sym_toffoli() -> Gate:
    m = np.eye(8, dtype=object)
    m[6, 6] = 0
    m[7, 7] = 0
    m[6, 7] = 1
    m[7, 6] = 1
    return _sym_gate(m)


def sym_fredkin() -> Gate:
    m = np.eye(8, dtype=object)
    m[5, 5] = 0
    m[6, 6] = 0
    m[5, 6] = 1
    m[6, 5] = 1
    return _sym_gate(m)


# Map: gate name → symbolic factory (no parameters)
SYM_SGATE_MAP = {
    "i": sym_i,
    "x": sym_x,
    "y": sym_y,
    "z": sym_z,
    "h": sym_h,
    "s": sym_s,
    "t": sym_t,
    "sd": sym_sd,
    "td": sym_td,
    "wroot": sym_wroot,
    "cnot": sym_cnot,
    "cx": sym_cnot,
    "cz": sym_cz,
    "cy": sym_cy,
    "swap": sym_swap,
    "ox": sym_ox,
    "oy": sym_oy,
    "oz": sym_oz,
    "toffoli": sym_toffoli,
    "ccnot": sym_toffoli,
    "ccx": sym_toffoli,
    "fredkin": sym_fredkin,
    "cswap": sym_fredkin,
}


# ── variable (parameterized) gates ───────────────────────────────────────────


def sym_rx(theta: Any = 0) -> Gate:
    c = sympy.cos(theta / 2)
    s = sympy.sin(theta / 2)
    return _sym_gate(np.array([[c, -sympy.I * s], [-sympy.I * s, c]], dtype=object))


def sym_ry(theta: Any = 0) -> Gate:
    c = sympy.cos(theta / 2)
    s = sympy.sin(theta / 2)
    return _sym_gate(np.array([[c, -s], [s, c]], dtype=object))


def sym_rz(theta: Any = 0) -> Gate:
    ep = sympy.exp(-sympy.I * theta / 2)
    em = sympy.exp(sympy.I * theta / 2)
    return _sym_gate(np.array([[ep, 0], [0, em]], dtype=object))


def sym_phase(theta: Any = 0) -> Gate:
    return _sym_gate(np.array([[1, 0], [0, sympy.exp(sympy.I * theta)]], dtype=object))


def sym_u(theta: Any = 0, phi: Any = 0, lbd: Any = 0) -> Gate:
    c = sympy.cos(theta / 2)
    s = sympy.sin(theta / 2)
    return _sym_gate(
        np.array(
            [
                [c, -sympy.exp(sympy.I * lbd) * s],
                [sympy.exp(sympy.I * phi) * s, sympy.exp(sympy.I * (phi + lbd)) * c],
            ],
            dtype=object,
        )
    )


def sym_r(theta: Any = 0, alpha: Any = 0, phi: Any = 0) -> Gate:
    c = sympy.cos(theta)
    s = sympy.sin(theta)
    sa = sympy.sin(alpha)
    ca = sympy.cos(alpha)
    sp = sympy.sin(phi)
    cp = sympy.cos(phi)
    return _sym_gate(
        np.array(
            [
                [c - sympy.I * ca * s, sa * s * (-sympy.I * cp - sp)],
                [sa * s * (-sympy.I * cp + sp), c + sympy.I * ca * s],
            ],
            dtype=object,
        )
    )


def sym_rxx(theta: Any = 0) -> Gate:
    c = sympy.cos(theta / 2)
    s = sympy.sin(theta / 2)
    m = np.array(
        [
            [c, 0, 0, -sympy.I * s],
            [0, c, -sympy.I * s, 0],
            [0, -sympy.I * s, c, 0],
            [-sympy.I * s, 0, 0, c],
        ],
        dtype=object,
    )
    return _sym_gate(m)


def sym_ryy(theta: Any = 0) -> Gate:
    c = sympy.cos(theta / 2)
    s = sympy.sin(theta / 2)
    m = np.array(
        [
            [c, 0, 0, sympy.I * s],
            [0, c, -sympy.I * s, 0],
            [0, -sympy.I * s, c, 0],
            [sympy.I * s, 0, 0, c],
        ],
        dtype=object,
    )
    return _sym_gate(m)


def sym_rzz(theta: Any = 0) -> Gate:
    ep = sympy.exp(-sympy.I * theta / 2)
    em = sympy.exp(sympy.I * theta / 2)
    m = np.array(
        [[ep, 0, 0, 0], [0, em, 0, 0], [0, 0, em, 0], [0, 0, 0, ep]],
        dtype=object,
    )
    return _sym_gate(m)


def sym_iswap(theta: Any = 1) -> Gate:
    c = sympy.cos(theta * sympy.pi / 2)
    s = sympy.sin(theta * sympy.pi / 2)
    m = np.array(
        [[1, 0, 0, 0], [0, c, sympy.I * s, 0], [0, sympy.I * s, c, 0], [0, 0, 0, 1]],
        dtype=object,
    )
    return _sym_gate(m)


def sym_cphase(theta: Any = 0) -> Gate:
    m = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, sympy.exp(sympy.I * theta)],
        ],
        dtype=object,
    )
    return _sym_gate(m)


def _sym_controlled_1q(sym_gate_fn: Any, *args: Any, **kwargs: Any) -> Gate:
    """Build a controlled single-qubit gate: [[I, 0], [0, U]] as a 4×4 matrix."""
    g = sym_gate_fn(*args, **kwargs)
    u = g.tensor.reshape(2, 2)
    m = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, u[0, 0], u[0, 1]],
            [0, 0, u[1, 0], u[1, 1]],
        ],
        dtype=object,
    )
    return _sym_gate(m)


def sym_crx(theta: Any = 0) -> Gate:
    return _sym_controlled_1q(sym_rx, theta)


def sym_cry(theta: Any = 0) -> Gate:
    return _sym_controlled_1q(sym_ry, theta)


def sym_crz(theta: Any = 0) -> Gate:
    return _sym_controlled_1q(sym_rz, theta)


def sym_cu(theta: Any = 0, phi: Any = 0, lbd: Any = 0) -> Gate:
    return _sym_controlled_1q(sym_u, theta, phi, lbd)


def sym_cr(theta: Any = 0, alpha: Any = 0, phi: Any = 0) -> Gate:
    return _sym_controlled_1q(sym_r, theta, alpha, phi)


def _sym_ocontrolled_1q(sym_gate_fn: Any, *args: Any, **kwargs: Any) -> Gate:
    """Build open-controlled single-qubit gate: [[U, 0], [0, I]] as a 4×4 matrix."""
    g = sym_gate_fn(*args, **kwargs)
    u = g.tensor.reshape(2, 2)
    m = np.array(
        [
            [u[0, 0], u[0, 1], 0, 0],
            [u[1, 0], u[1, 1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=object,
    )
    return _sym_gate(m)


def sym_orx(theta: Any = 0) -> Gate:
    return _sym_ocontrolled_1q(sym_rx, theta)


def sym_ory(theta: Any = 0) -> Gate:
    return _sym_ocontrolled_1q(sym_ry, theta)


def sym_orz(theta: Any = 0) -> Gate:
    return _sym_ocontrolled_1q(sym_rz, theta)


# Map: gate name → symbolic factory (accepts **parameters)
SYM_VGATE_MAP = {
    "rx": sym_rx,
    "ry": sym_ry,
    "rz": sym_rz,
    "phase": sym_phase,
    "u": sym_u,
    "r": sym_r,
    "rxx": sym_rxx,
    "ryy": sym_ryy,
    "rzz": sym_rzz,
    "iswap": sym_iswap,
    "cphase": sym_cphase,
    "crx": sym_crx,
    "cry": sym_cry,
    "crz": sym_crz,
    "cu": sym_cu,
    "cr": sym_cr,
    "orx": sym_orx,
    "ory": sym_ory,
    "orz": sym_orz,
}
