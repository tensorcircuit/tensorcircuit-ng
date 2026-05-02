from typing import Any

import numpy as np
import pytest
import sympy
import tensornetwork as tn

import tensorcircuit as tc
from tensorcircuit.gates import Gate
from tensorcircuit.simplify import _full_light_cone_cancel
from tensorcircuit.symbolgates import (
    sym_cphase,
    sym_cr,
    sym_crx,
    sym_cry,
    sym_crz,
    sym_cu,
    sym_cy,
    sym_cz,
    sym_fredkin,
    sym_h,
    sym_i,
    sym_iswap,
    sym_orx,
    sym_ory,
    sym_orz,
    sym_ox,
    sym_oy,
    sym_oz,
    sym_phase,
    sym_r,
    sym_rxx,
    sym_rx,
    sym_ryy,
    sym_rzz,
    sym_s,
    sym_sd,
    sym_swap,
    sym_t,
    sym_td,
    sym_u,
    sym_wroot,
    sym_y,
)


def _sc():
    return tc.SymbolCircuit


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sym():
    return {
        "theta": sympy.Symbol("theta", real=True),
        "phi": sympy.Symbol("phi", real=True),
    }


@pytest.fixture()
def SymbolCircuit():
    return _sc()


# ── construction ──────────────────────────────────────────────────────────────


def test_init(SymbolCircuit):
    sc = SymbolCircuit(3)
    assert sc._nqubits == 3
    assert sc._d == 2
    assert len(sc._nodes) == 3
    assert len(sc._front) == 3
    assert sc._qir == []


# ── fixed gates ────────────────────────────────────────────────────────────────


def test_fixed_gates_recorded_in_qir(SymbolCircuit):
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)
    sc.z(1)
    assert sc.gate_count() == 3
    assert [d["name"] for d in sc.to_qir()] == ["h", "cnot", "z"]


# ── wavefunction / state ──────────────────────────────────────────────────────


def test_zero_state(SymbolCircuit):
    sc = SymbolCircuit(2)
    psi = sc.wavefunction()
    assert psi.dtype == object
    assert psi.shape == (4,)
    assert psi[0] == 1
    for i in range(1, 4):
        assert psi[i] == 0


def test_bell_state(SymbolCircuit):
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)
    psi = sc.wavefunction()
    v = sympy.Rational(1, 1) / sympy.sqrt(2)
    assert sympy.simplify(psi[0] - v) == 0
    assert psi[1] == 0
    assert psi[2] == 0
    assert sympy.simplify(psi[3] - v) == 0


def test_rx_state(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    psi = sc.wavefunction()
    assert sympy.simplify(psi[0] - sympy.cos(theta / 2)) == 0
    assert sympy.simplify(psi[1] + sympy.I * sympy.sin(theta / 2)) == 0


# ── amplitude ──────────────────────────────────────────────────────────────────


def test_amplitude_zero_state(SymbolCircuit):
    sc = SymbolCircuit(2)
    assert sc.amplitude("00") == 1
    assert sc.amplitude("01") == 0
    assert sc.amplitude("10") == 0
    assert sc.amplitude("11") == 0


def test_amplitude_rx_symbolic(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    amp0 = sc.amplitude("0")
    assert sympy.simplify(amp0 - sympy.cos(theta / 2)) == 0
    # verify at theta=0: amplitude of |0> = 1
    assert sympy.simplify(amp0.subs(theta, 0) - 1) == 0


def test_amplitude_numerical_consistency(SymbolCircuit, sym):
    """Symbolic amplitude evaluated at a point matches tc.Circuit."""

    theta = sym["theta"]
    phi = sym["phi"]

    sc = SymbolCircuit(2)
    sc.h(0)
    sc.rx(1, theta=theta)
    sc.cnot(0, 1)
    sc.rz(0, theta=phi)

    amp = sc.amplitude("10")

    for t_val, p_val in [(0.0, 0.0), (0.5, 0.3), (1.0, 1.5)]:
        sym_val = complex(amp.subs({theta: t_val, phi: p_val}))

        c = tc.Circuit(2)
        c.h(0)
        c.rx(1, theta=t_val)
        c.cnot(0, 1)
        c.rz(0, theta=p_val)
        ref_val = complex(c.amplitude("10"))

        np.testing.assert_allclose(
            sym_val,
            ref_val,
            atol=1e-7,
            rtol=0.0,
            err_msg=(
                f"Mismatch at theta={t_val}, phi={p_val}: "
                f"sym={sym_val}, ref={ref_val}"
            ),
        )


# ── expectation ────────────────────────────────────────────────────────────────


def test_expectation_z_after_rx(SymbolCircuit, sym):
    """<0|RX† Z RX|0> = cos(theta) — the canonical demo from symbolic_demo.py."""
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    expr = sc.expectation((tc.gates.z(), [0]))
    assert sympy.simplify(expr - sympy.cos(theta)) == 0


def test_expectation_ps_z_after_rx(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    expr = sc.expectation_ps(z=[0])
    assert sympy.simplify(expr - sympy.cos(theta)) == 0
    assert sympy.simplify(expr.subs(theta, 0) - 1) == 0
    assert sympy.simplify(expr.subs(theta, sympy.pi) + 1) == 0


def test_expectation_numerical_consistency(SymbolCircuit, sym):
    """Symbolic expectation evaluated at a point matches tc.Circuit."""
    theta = sym["theta"]
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.rx(1, theta=theta)
    sc.cnot(0, 1)

    expr = sc.expectation_ps(z=[0])

    for t_val in [0.0, 0.5, 1.0, 1.5]:
        sym_val = complex(expr.subs(theta, t_val))

        c = tc.Circuit(2)
        c.h(0)
        c.rx(1, theta=t_val)
        c.cnot(0, 1)
        ref_val = complex(c.expectation_ps(z=[0]))

        np.testing.assert_allclose(
            sym_val,
            ref_val,
            atol=1e-7,
            rtol=0.0,
            err_msg=f"Mismatch at theta={t_val}: sym={sym_val}, ref={ref_val}",
        )


def test_expectation_bell_state_zz(SymbolCircuit):
    """Bell state: <Z_0 Z_1> = 1."""
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)
    expr = sc.expectation_ps(z=[0, 1])
    assert sympy.simplify(expr - 1) == 0


def test_lightcone_cancellation(SymbolCircuit, sym):
    sc = SymbolCircuit(4)
    # Layer 1
    for i in range(4):
        sc.rx(i, theta=sym["theta"])
    for i in range(3):
        sc.cnot(i, i + 1)
    # Layer 2
    for i in range(4):
        sc.rx(i, theta=sym["theta"])
    for i in range(3):
        sc.cnot(i, i + 1)

    # Check expectation with and without lightcone
    res1 = sc.expectation((tc.gates.z(), [0]), enable_lightcone=False)
    res2 = sc.expectation((tc.gates.z(), [0]), enable_lightcone=True)

    # Check correctness at theta=0.5
    val1 = res1.subs(sym["theta"], 0.5).evalf()
    val2 = res2.subs(sym["theta"], 0.5).evalf()
    np.testing.assert_allclose(complex(val1), complex(val2), atol=1e-10, rtol=0.0)

    # Check node reduction from light cone cancellation
    nodes_full = sc.expectation_before((tc.gates.z(), [0]), reuse=False)
    nodes_lc = _full_light_cone_cancel(
        sc.expectation_before((tc.gates.z(), [0]), reuse=False)
    )
    assert len(nodes_lc) < len(nodes_full)


# ── free_symbols ───────────────────────────────────────────────────────────────


def test_free_symbols(SymbolCircuit, sym):
    theta, phi = sym["theta"], sym["phi"]
    sc = SymbolCircuit(2)
    sc.rx(0, theta=theta)
    sc.rz(1, theta=phi)
    assert sc.free_symbols() == {theta, phi}


def test_free_symbols_no_var_gates(SymbolCircuit):
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)
    assert sc.free_symbols() == set()


# ── to_circuit ─────────────────────────────────────────────────────────────────


def test_to_circuit(SymbolCircuit, sym):
    theta, phi = sym["theta"], sym["phi"]

    sc = SymbolCircuit(2)
    sc.h(0)
    sc.rx(1, theta=theta)
    sc.cnot(0, 1)
    sc.rz(0, theta=phi)

    t_val, p_val = 0.7, 0.4
    c = sc.to_circuit({theta: t_val, phi: p_val})

    ref = tc.Circuit(2)
    ref.h(0)
    ref.rx(1, theta=t_val)
    ref.cnot(0, 1)
    ref.rz(0, theta=p_val)

    for bitstring in ["00", "01", "10", "11"]:
        np.testing.assert_allclose(
            complex(c.amplitude(bitstring)),
            complex(ref.amplitude(bitstring)),
            atol=1e-8,
            rtol=0.0,
        )


def test_gate_definitions_match_tc_circuit(SymbolCircuit):
    """Comprehensive gate-alignment test across all major parameterised gate types.

    Builds a SymbolCircuit with one symbolic parameter per gate (or per gate
    parameter for multi-parameter gates), binds them to concrete numerical values
    via ``to_circuit()``, and asserts that every amplitude matches a ``tc.Circuit``
    constructed directly with the same numerical values.

    Gate types covered: rx, ry, rz, phase, rxx, ryy, rzz, crx, cry, crz,
    cphase, iswap, u (3-param), orx, ory, orz.
    """
    # Concrete numerical values — generic, away from 0 and π/2 to avoid
    # accidental cancellations that could hide gate-definition mismatches.
    p = [
        0.31,
        0.57,
        0.83,
        0.42,
        0.19,
        0.74,
        0.66,
        0.28,
        0.91,
        0.37,
        0.53,
        0.14,
        0.62,
        0.45,
        0.78,
        0.22,
        0.69,
        0.35,
    ]

    s = [sympy.Symbol(f"p{i}", real=True) for i in range(len(p))]
    param_dict = {s[i]: p[i] for i in range(len(p))}

    nq = 4
    sc = SymbolCircuit(nq)
    c = tc.Circuit(nq)

    def add(gate_name, qubits, **sym_kw):
        """Apply gate symbolically to sc and numerically to c."""
        num_kw = {k: float(v.subs(param_dict)) for k, v in sym_kw.items()}
        getattr(sc, gate_name)(*qubits, **sym_kw)
        getattr(c, gate_name)(*qubits, **num_kw)

    # Single-qubit rotations
    add("rx", [0], theta=s[0])
    add("ry", [1], theta=s[1])
    add("rz", [2], theta=s[2])
    add("phase", [3], theta=s[3])
    # Two-qubit rotations
    add("rxx", [0, 1], theta=s[4])
    add("ryy", [1, 2], theta=s[5])
    add("rzz", [2, 3], theta=s[6])
    # Controlled single-qubit rotations
    add("crx", [0, 1], theta=s[7])
    add("cry", [1, 2], theta=s[8])
    add("crz", [2, 3], theta=s[9])
    # Controlled phase and parameterised two-qubit interaction
    add("cphase", [0, 1], theta=s[10])
    add("iswap", [0, 2], theta=s[11])
    # Three-parameter single-qubit gate
    add("u", [3], theta=s[12], phi=s[13], lbd=s[14])
    # Open-controlled rotations (control fires on |0>)
    add("orx", [0, 1], theta=s[15])
    add("ory", [1, 2], theta=s[16])
    add("orz", [2, 3], theta=s[17])

    bound = sc.to_circuit(param_dict)

    all_bitstrings = [format(i, f"0{nq}b") for i in range(2**nq)]
    for bitstring in all_bitstrings:
        sym_amp = complex(bound.amplitude(bitstring))
        ref_amp = complex(c.amplitude(bitstring))
        np.testing.assert_allclose(
            sym_amp,
            ref_amp,
            atol=1e-6,
            rtol=0.0,
            err_msg=(
                f"Amplitude mismatch at |{bitstring}>: "
                f"symbolic={sym_amp:.8f}, reference={ref_amp:.8f}"
            ),
        )


# ── bind ───────────────────────────────────────────────────────────────────────


def test_bind_partial(SymbolCircuit, sym):
    theta, phi = sym["theta"], sym["phi"]

    sc = SymbolCircuit(2)
    sc.rx(0, theta=theta)
    sc.rz(1, theta=phi)

    sc2 = sc.bind({theta: sympy.pi / 2})
    assert sc2.free_symbols() == {phi}

    # After binding theta=pi/2, Z[0] expectation should be ~0
    expr = sc2.expectation_ps(z=[0])
    assert sympy.simplify(expr) == 0


def test_bind_full(SymbolCircuit, sym):
    theta = sym["theta"]

    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)

    sc2 = sc.bind({theta: 0.5})
    assert sc2.free_symbols() == set()


# ── Qiskit translation ─────────────────────────────────────────────────────────


def test_to_qiskit_parameters(SymbolCircuit, sym):
    pytest.importorskip("qiskit")

    theta, phi = sym["theta"], sym["phi"]

    sc = SymbolCircuit(2)
    sc.h(0)
    sc.rx(1, theta=theta)
    sc.rz(0, theta=phi)
    sc.cnot(0, 1)

    qc = sc.to_qiskit()
    param_names = {str(p) for p in qc.parameters}
    assert "theta" in param_names
    assert "phi" in param_names


def test_to_qiskit_no_parameters(SymbolCircuit):
    pytest.importorskip("qiskit")

    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)

    qc = sc.to_qiskit()
    assert len(qc.parameters) == 0


def test_to_qiskit_state_matches(SymbolCircuit, sym):
    """Bound Qiskit circuit state vector matches tc.Circuit state."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit.quantum_info")
    from qiskit.quantum_info import Statevector

    theta, phi = sym["theta"], sym["phi"]
    t_val, p_val = 0.5, 0.3

    sc = SymbolCircuit(2)
    sc.h(0)
    sc.rx(1, theta=theta)
    sc.cnot(0, 1)
    sc.rz(0, theta=phi)

    qc = sc.to_qiskit()
    # Bind Qiskit parameters
    params_map = {p: t_val if str(p) == "theta" else p_val for p in qc.parameters}
    bound_qc = qc.assign_parameters(params_map)
    sv = Statevector.from_instruction(bound_qc)

    c = sc.to_circuit({theta: t_val, phi: p_val})
    tc_state = c.state()

    # Qiskit uses reversed qubit ordering — permute to compare
    # Qiskit index maps: qubit-0 is LSB, TC qubit-0 is MSB
    n = sc._nqubits
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)]
    qk_state = np.array([sv.data[perm[i]] for i in range(2**n)])

    assert np.allclose(tc_state, qk_state, atol=1e-6)


# ── multi-qubit / multi-symbol circuits ───────────────────────────────────────


def test_two_symbol_circuit(SymbolCircuit, sym):
    theta, phi = sym["theta"], sym["phi"]

    sc = SymbolCircuit(2)
    sc.rx(0, theta=theta)
    sc.rz(1, theta=phi)

    expr_z0 = sc.expectation_ps(z=[0])
    expr_z1 = sc.expectation_ps(z=[1])

    # Z[0] after rx(theta): cos(theta); Z[1] after rz(phi): 1 (rz doesn't flip Z)
    assert sympy.simplify(expr_z0 - sympy.cos(theta)) == 0
    assert sympy.simplify(expr_z1 - 1) == 0


def test_rz_does_not_flip_z(SymbolCircuit, sym):
    """RZ is a rotation in Z basis — it commutes with Z, so <Z> = 1 unchanged."""
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rz(0, theta=theta)
    expr = sc.expectation_ps(z=[0])
    assert sympy.simplify(expr - 1) == 0


def test_three_qubit_toffoli(SymbolCircuit):
    sc = SymbolCircuit(3)
    sc.x(0)
    sc.x(1)
    sc.toffoli(0, 1, 2)
    amp = sc.amplitude("111")
    assert amp == 1


# ── inherited AbstractCircuit utilities ───────────────────────────────────────


def test_gate_count(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.rx(1, theta=theta)
    sc.cnot(0, 1)
    assert sc.gate_count() == 3


def test_qir_structure(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(2)
    sc.rx(0, theta=theta)
    sc.cnot(0, 1)
    qir = sc.to_qir()
    assert len(qir) == 2
    assert qir[0]["name"] == "rx"
    assert qir[0]["parameters"]["theta"] is theta
    assert qir[1]["name"] == "cnot"


# ── ry gate ────────────────────────────────────────────────────────────────────


def test_ry_state(SymbolCircuit, sym):
    """RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>."""
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.ry(0, theta=theta)
    psi = sc.wavefunction()
    assert sympy.simplify(psi[0] - sympy.cos(theta / 2)) == 0
    assert sympy.simplify(psi[1] - sympy.sin(theta / 2)) == 0


def test_ry_expectation_z(SymbolCircuit, sym):
    """<0|RY† Z RY|0> = cos(theta)."""
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.ry(0, theta=theta)
    expr = sc.expectation_ps(z=[0])
    assert sympy.simplify(expr - sympy.cos(theta)) == 0


# ── to_circuit unbound symbol rejection ───────────────────────────────────────


def test_to_circuit_rejects_unbound_symbols(SymbolCircuit, sym):
    """to_circuit() must raise if free symbols remain after substitution."""
    theta, phi = sym["theta"], sym["phi"]
    sc = SymbolCircuit(2)
    sc.rx(0, theta=theta)
    sc.rz(1, theta=phi)

    with pytest.raises(ValueError, match="free symbols"):
        sc.to_circuit({theta: 0.5})  # phi is still unbound


# ── mixed symbolic and numeric parameters ─────────────────────────────────────


def test_mixed_sympy_and_numeric_params(SymbolCircuit, sym):
    """Circuit where some gates use sympy Symbols, others use plain floats/ints.

    Verifies that tensor-network contraction works when object-dtype tensors
    contain a mix of sympy expressions and bare Python/NumPy numbers.
    """
    theta = sym["theta"]
    phi_num = 0.4  # plain Python float — not a sympy Symbol

    sc = SymbolCircuit(2)
    sc.rx(0, theta=theta)  # sympy Symbol param — affects Z[0]
    sc.rz(1, theta=phi_num)  # plain float param (numeric, no free symbol)

    # Only theta is a free symbol; the float is fully numeric
    assert sc.free_symbols() == {theta}

    # wavefunction must be a sympy object array containing theta
    psi = sc.wavefunction()
    assert psi.dtype == object
    free_in_psi = {s for amp in psi for s in getattr(amp, "free_symbols", set())}
    assert theta in free_in_psi

    # expectation of Z[0] = cos(theta), independent of phi_num (RZ commutes with Z)
    expr = sc.expectation_ps(z=[0])
    assert hasattr(expr, "free_symbols") and theta in expr.free_symbols
    assert sympy.simplify(expr - sympy.cos(theta)) == 0

    # Cross-validate: bind theta numerically and compare to tc.Circuit
    theta_val = 0.7
    sym_val = float(expr.subs(theta, theta_val))

    c = sc.to_circuit({theta: theta_val})
    ref_val = float(c.expectation_ps(z=[0]).real)
    np.testing.assert_allclose(sym_val, ref_val, atol=1e-6, rtol=0.0)


def test_mixed_sympy_and_numpy_numeric_params(SymbolCircuit, sym):
    """Same as above but the numeric param is a numpy float64 (not a Python float)."""
    theta = sym["theta"]
    phi_np = np.float64(0.3)  # numpy scalar

    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    sc.rz(0, theta=phi_np)

    assert sc.free_symbols() == {theta}

    expr = sc.expectation_ps(z=[0])
    assert theta in expr.free_symbols

    # Numeric check
    theta_val = 0.5
    sym_val = float(expr.subs(theta, theta_val))

    c = sc.to_circuit({theta: theta_val})
    ref_val = float(c.expectation_ps(z=[0]).real)
    np.testing.assert_allclose(sym_val, ref_val, atol=1e-6, rtol=0.0)


# ── symbolgates factory tests ─────────────────────────────────────────────────


def _mat(gate: Any, n: int) -> np.ndarray:  # type: ignore[type-arg]
    return gate.tensor.reshape(n, n)


# ── fixed gates ───────────────────────────────────────────────────────────────


def test_sym_i_is_identity() -> None:
    m = _mat(sym_i(), 2)
    assert m[0, 0] == 1 and m[1, 1] == 1 and m[0, 1] == 0 and m[1, 0] == 0


def test_sym_y_entries() -> None:
    m = _mat(sym_y(), 2)
    assert m[0, 1] == -sympy.I and m[1, 0] == sympy.I


def test_sym_s_diagonal() -> None:
    m = _mat(sym_s(), 2)
    assert m[0, 0] == 1 and m[1, 1] == sympy.I


def test_sym_t_diagonal() -> None:
    m = _mat(sym_t(), 2)
    assert sympy.simplify(m[1, 1] - sympy.exp(sympy.pi * sympy.I / 4)) == 0


def test_sym_sd_diagonal() -> None:
    m = _mat(sym_sd(), 2)
    assert m[1, 1] == -sympy.I


def test_sym_td_diagonal() -> None:
    m = _mat(sym_td(), 2)
    assert sympy.simplify(m[1, 1] - sympy.exp(-sympy.pi * sympy.I / 4)) == 0


def test_sym_wroot_shape() -> None:
    m = _mat(sym_wroot(), 2)
    v = sympy.Rational(1, 1) / sympy.sqrt(2)
    assert sympy.simplify(m[0, 0] - v) == 0
    assert sympy.simplify(m[1, 1] - v) == 0


def test_sym_cz_corner() -> None:
    m = _mat(sym_cz(), 4)
    assert m[3, 3] == -1 and m[0, 0] == 1


def test_sym_cy_off_diagonal() -> None:
    m = _mat(sym_cy(), 4)
    assert m[2, 3] == -sympy.I and m[3, 2] == sympy.I


def test_sym_swap_off_diagonal() -> None:
    m = _mat(sym_swap(), 4)
    assert m[1, 2] == 1 and m[2, 1] == 1 and m[0, 0] == 1 and m[3, 3] == 1


def test_sym_ox_flips_control_zero() -> None:
    m = _mat(sym_ox(), 4)
    assert m[0, 1] == 1 and m[1, 0] == 1 and m[2, 2] == 1 and m[3, 3] == 1


def test_sym_oy_entries() -> None:
    m = _mat(sym_oy(), 4)
    assert m[0, 1] == -sympy.I and m[1, 0] == sympy.I


def test_sym_oz_entries() -> None:
    m = _mat(sym_oz(), 4)
    assert m[0, 0] == 1 and m[1, 1] == -1 and m[2, 2] == 1


def test_sym_fredkin_swaps_101_110() -> None:
    m = _mat(sym_fredkin(), 8)
    assert m[5, 6] == 1 and m[6, 5] == 1
    assert m[0, 0] == 1 and m[7, 7] == 1


# ── variable gates ────────────────────────────────────────────────────────────


def test_sym_phase_entries() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_phase(theta), 2)
    assert m[0, 0] == 1 and m[0, 1] == 0
    assert sympy.simplify(m[1, 1] - sympy.exp(sympy.I * theta)) == 0


def test_sym_u_diagonal() -> None:
    theta, phi, lbd = sympy.symbols("theta phi lbd", real=True)
    m = _mat(sym_u(theta, phi, lbd), 2)
    assert sympy.simplify(m[0, 0] - sympy.cos(theta / 2)) == 0


def test_sym_r_shape() -> None:
    theta, alpha, phi = sympy.symbols("theta alpha phi", real=True)
    g = sym_r(theta, alpha, phi)
    assert g.tensor.shape == (2, 2)


def test_sym_rxx_diagonal() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_rxx(theta), 4)
    assert sympy.simplify(m[0, 0] - sympy.cos(theta / 2)) == 0
    assert sympy.simplify(m[0, 3] + sympy.I * sympy.sin(theta / 2)) == 0


def test_sym_ryy_corner() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_ryy(theta), 4)
    assert sympy.simplify(m[0, 0] - sympy.cos(theta / 2)) == 0


def test_sym_rzz_diagonal() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_rzz(theta), 4)
    ep = sympy.exp(-sympy.I * theta / 2)
    assert sympy.simplify(m[0, 0] - ep) == 0


def test_sym_iswap_identity_corners() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_iswap(theta), 4)
    assert m[0, 0] == 1 and m[3, 3] == 1


def test_sym_cphase_corner() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_cphase(theta), 4)
    assert sympy.simplify(m[3, 3] - sympy.exp(sympy.I * theta)) == 0
    assert m[0, 0] == 1


# ── controlled factories ──────────────────────────────────────────────────────


def test_sym_crx_upper_identity() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_crx(theta), 4)
    assert m[0, 0] == 1 and m[1, 1] == 1
    assert sympy.simplify(m[2, 2] - sympy.cos(theta / 2)) == 0


def test_sym_cry_upper_identity() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_cry(theta), 4)
    assert m[0, 0] == 1 and m[1, 1] == 1


def test_sym_crz_upper_identity() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_crz(theta), 4)
    assert m[0, 0] == 1 and m[1, 1] == 1


def test_sym_cu_upper_identity() -> None:
    theta, phi, lbd = sympy.symbols("theta phi lbd", real=True)
    m = _mat(sym_cu(theta, phi, lbd), 4)
    assert m[0, 0] == 1 and m[1, 1] == 1


def test_sym_cr_upper_identity() -> None:
    theta, alpha, phi = sympy.symbols("theta alpha phi", real=True)
    m = _mat(sym_cr(theta, alpha, phi), 4)
    assert m[0, 0] == 1 and m[1, 1] == 1


# ── open-controlled factories ─────────────────────────────────────────────────


def test_sym_orx_lower_identity() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_orx(theta), 4)
    assert m[2, 2] == 1 and m[3, 3] == 1
    assert sympy.simplify(m[0, 0] - sympy.cos(theta / 2)) == 0


def test_sym_ory_lower_identity() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_ory(theta), 4)
    assert m[2, 2] == 1 and m[3, 3] == 1


def test_sym_orz_lower_identity() -> None:
    theta = sympy.Symbol("theta", real=True)
    m = _mat(sym_orz(theta), 4)
    assert m[2, 2] == 1 and m[3, 3] == 1


# ── SymbolCircuit additional coverage ─────────────────────────────────────────


# -- apply_general_gate_delayed: name=None fallback (line 119) -----------------


def test_apply_gate_delayed_name_from_attr(SymbolCircuit: Any) -> None:
    """apply_general_gate_delayed with name=None falls back to gatef.n."""

    class _FakeH:
        n = "h"

        def __call__(self) -> Any:
            return sym_h()

    apply_fn = SymbolCircuit.apply_general_gate_delayed(_FakeH())
    sc = SymbolCircuit(1)
    apply_fn(sc, 0)
    psi = sc.wavefunction()
    v = sympy.Rational(1, 1) / sympy.sqrt(2)
    assert sympy.simplify(psi[0] - v) == 0


# -- apply_list: fixed gate with sequence index (lines 148-152) ---------------


def test_fixed_gate_apply_list_sequence(SymbolCircuit: Any) -> None:
    """sc.h([0, 1, 2]) applies H to each qubit — exercises apply_list branch."""
    sc = SymbolCircuit(3)
    sc.h([0, 1, 2])
    assert sc.gate_count() == 3
    psi = sc.wavefunction()
    v = sympy.Rational(1, 1) / sympy.sqrt(8)
    for amp in psi:
        assert sympy.simplify(amp - v) == 0


def test_cnot_apply_list_sequence(SymbolCircuit: Any) -> None:
    """sc.cnot([0, 1], [1, 2]) applies CNOT to pairs (0,1) and (1,2)."""
    sc = SymbolCircuit(3)
    sc.cnot([0, 1], [1, 2])
    assert sc.gate_count() == 2


def test_fixed_gate_apply_list_range(SymbolCircuit: Any) -> None:
    """range index also goes through the sequence branch of apply_list."""
    sc = SymbolCircuit(3)
    sc.h(range(3))
    assert sc.gate_count() == 3


# -- apply_general_variable_gate_delayed name=None and sequence (lines 168, 206-216)


def test_variable_gate_delayed_name_from_attr(SymbolCircuit: Any) -> None:
    """apply_general_variable_gate_delayed with name=None falls back to gatef.n."""

    class _FakeRx:
        n = "rx"

        def __call__(self, **kws: Any) -> Any:
            return sym_rx(**kws)

    theta = sympy.Symbol("theta", real=True)
    apply_fn = SymbolCircuit.apply_general_variable_gate_delayed(_FakeRx())
    sc = SymbolCircuit(1)
    apply_fn(sc, 0, theta=theta)
    expr = sc.expectation_ps(z=[0])
    assert sympy.simplify(expr - sympy.cos(theta)) == 0


def test_variable_gate_apply_list_sequence(SymbolCircuit: Any, sym: Any) -> None:
    """sc.rx([0, 1, 2], theta=theta) applies rx to each qubit in sequence."""
    theta = sym["theta"]
    sc = SymbolCircuit(3)
    sc.rx([0, 1, 2], theta=theta)
    assert sc.gate_count() == 3
    expr0 = sc.expectation_ps(z=[0])
    assert sympy.simplify(expr0 - sympy.cos(theta)) == 0


def test_variable_gate_apply_list_per_qubit_params(SymbolCircuit: Any) -> None:
    """Per-qubit scalar params via list: each v[i] is fetched for each qubit."""
    theta0 = sympy.Symbol("theta0", real=True)
    theta1 = sympy.Symbol("theta1", real=True)
    sc = SymbolCircuit(2)
    # Pass a list of thetas, one per qubit
    sc.rx([0, 1], theta=[theta0, theta1])
    assert sc.gate_count() == 2
    expr0 = sc.expectation_ps(z=[0])
    assert sympy.simplify(expr0 - sympy.cos(theta0)) == 0


# -- amplitude with list input (line 236) -------------------------------------


def test_amplitude_list_input(SymbolCircuit: Any) -> None:
    sc = SymbolCircuit(2)
    assert sc.amplitude([0, 0]) == 1
    assert sc.amplitude([0, 1]) == 0


# -- wavefunction form="ket" / "bra" (lines 272, 274) ------------------------


def test_wavefunction_ket_shape(SymbolCircuit: Any) -> None:
    sc = SymbolCircuit(2)
    psi = sc.wavefunction(form="ket")
    assert psi.shape == (4, 1)
    assert psi[0, 0] == 1


def test_wavefunction_bra_shape(SymbolCircuit: Any) -> None:
    sc = SymbolCircuit(2)
    psi = sc.wavefunction(form="bra")
    assert psi.shape == (1, 4)
    assert psi[0, 0] == 1


# -- expectation_before edge cases (lines 307-309, 316, 322) ------------------


def test_expectation_before_duplicate_index_raises(SymbolCircuit: Any) -> None:
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)
    z = np.array([[1, 0], [0, -1]], dtype=object)
    with pytest.raises(ValueError, match="Cannot measure two operators"):
        sc.expectation(
            (z, [0]),
            (z, [0]),  # same index twice
        )


def test_expectation_before_tn_node_numeric_dtype(SymbolCircuit: Any) -> None:
    """Operator passed as a tn.Node with float dtype should be converted to object."""
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    # Build a Z node with float64 tensor (not object dtype)
    z_numeric = tn.Node(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float))
    z_numeric = z_numeric.reorder_edges([z_numeric[0], z_numeric[1]])

    z_gate = Gate(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float).reshape(2, 2))
    expr = sc.expectation((z_gate, [0]))
    assert sympy.simplify(expr - sympy.cos(theta)) == 0


def test_expectation_before_integer_index(SymbolCircuit: Any, sym: Any) -> None:
    """Passing an integer (not list) as qubit index should work."""
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    z = np.array([[1, 0], [0, -1]], dtype=object)
    expr = sc.expectation((z, 0))  # integer index, not list
    assert sympy.simplify(expr - sympy.cos(theta)) == 0


# -- to_circuit(None) (line 391) ----------------------------------------------


def test_to_circuit_none_param_dict(SymbolCircuit: Any) -> None:
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)
    c = sc.to_circuit()  # param_dict defaults to None

    ref = tc.Circuit(2)
    ref.h(0)
    ref.cnot(0, 1)
    for bs in ["00", "01", "10", "11"]:
        np.testing.assert_allclose(
            complex(c.amplitude(bs)),
            complex(ref.amplitude(bs)),
            atol=1e-8,
            rtol=0.0,
        )


# -- bind with fixed gates (line 439) -----------------------------------------


def test_bind_with_fixed_and_variable_gates(SymbolCircuit: Any, sym: Any) -> None:
    """bind() should replay both fixed and variable gates."""
    theta = sym["theta"]
    sc = SymbolCircuit(2)
    sc.h(0)  # fixed gate — hits the else branch in bind (line 439)
    sc.rx(1, theta=theta)  # variable gate
    sc2 = sc.bind({theta: sympy.pi / 2})
    assert sc2.free_symbols() == set()
    assert sc2.gate_count() == 2


# -- to_qiskit coverage (various gate branches) --------------------------------


def test_to_qiskit_sd_td(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    sc = SymbolCircuit(1)
    sc.s(0)
    sc.t(0)
    sc.sd(0)
    sc.td(0)
    qc = sc.to_qiskit()
    # sdg and tdg should appear in the circuit
    op_names = [instr.operation.name for instr in qc.data]
    assert "s" in op_names and "t" in op_names
    assert "sdg" in op_names and "tdg" in op_names


def test_to_qiskit_ox_oy_oz(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    sc = SymbolCircuit(2)
    sc.oz(0, 1)
    qc = sc.to_qiskit()
    assert len(qc.data) == 1


def test_to_qiskit_orx_ory_orz(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(2)
    sc.orx(0, 1, theta=theta)
    qc = sc.to_qiskit()
    assert len(qc.parameters) == 1


def test_to_qiskit_rotation_gates(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(2)
    sc.rx(0, theta=theta)
    sc.ry(0, theta=theta)
    sc.rz(0, theta=theta)
    sc.crx(0, 1, theta=theta)
    sc.cry(0, 1, theta=theta)
    sc.crz(0, 1, theta=theta)
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "rx" in op_names and "ry" in op_names and "rz" in op_names
    assert "crx" in op_names and "cry" in op_names and "crz" in op_names


def test_to_qiskit_rxx_ryy_rzz(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(2)
    sc.rxx(0, 1, theta=theta)
    sc.ryy(0, 1, theta=theta)
    sc.rzz(0, 1, theta=theta)
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "rxx" in op_names and "ryy" in op_names and "rzz" in op_names


def test_to_qiskit_phase_cphase(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(2)
    sc.phase(0, theta=theta)
    sc.cphase(0, 1, theta=theta)
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "p" in op_names and "cp" in op_names


def test_to_qiskit_u_gate(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    theta, phi, lbd = sympy.symbols("theta phi lbd", real=True)
    sc = SymbolCircuit(1)
    sc.u(0, theta=theta, phi=phi, lbd=lbd)
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "u" in op_names
    assert len(qc.parameters) == 3


def test_to_qiskit_cu_gate(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    theta, phi, lbd = sympy.symbols("theta phi lbd", real=True)
    sc = SymbolCircuit(2)
    sc.cu(0, 1, theta=theta, phi=phi, lbd=lbd)
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "cu" in op_names


def test_to_qiskit_r_numeric(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit.extensions")
    sc = SymbolCircuit(1)
    sc.r(0, theta=0.5, alpha=0.3, phi=0.2)
    qc = sc.to_qiskit()
    assert len(qc.data) == 1


def test_to_qiskit_wroot_raises(SymbolCircuit: Any) -> None:
    pytest.importorskip("qiskit")
    sc = SymbolCircuit(1)
    sc.wroot(0)
    with pytest.raises(NotImplementedError, match="wroot"):
        sc.to_qiskit()


# -- _to_qk numeric paths and _sym_expr_to_qk ---------------------------------


def test_to_qiskit_numeric_param(SymbolCircuit: Any) -> None:
    """_to_qk(float) → float path (line 480)."""
    pytest.importorskip("qiskit")
    sc = SymbolCircuit(1)
    sc.rx(0, theta=0.5)  # numeric, not sympy
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "rx" in op_names


def test_to_qiskit_sympy_number_param(SymbolCircuit: Any) -> None:
    """_to_qk(sympy.Number) → complex path (line 482)."""
    pytest.importorskip("qiskit")
    sc = SymbolCircuit(1)
    sc.rz(0, theta=sympy.Rational(1, 2))  # sympy.Number
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "rz" in op_names


def test_to_qiskit_sympy_pi_param(SymbolCircuit: Any) -> None:
    """_to_qk(sympy.pi) → no-free-symbols complex path (line 500)."""
    pytest.importorskip("qiskit")
    sc = SymbolCircuit(1)
    sc.rz(0, theta=sympy.pi)
    qc = sc.to_qiskit()
    op_names = [instr.operation.name for instr in qc.data]
    assert "rz" in op_names


def test_to_qiskit_arithmetic_expr(SymbolCircuit: Any) -> None:
    """_to_qk with Add/Mul expr hits _sym_expr_to_qk (lines 488-496, 616-645)."""
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(1)
    sc.rz(0, theta=2 * theta + sympy.Rational(1, 4))  # Add(Mul(2, theta), 1/4)
    qc = sc.to_qiskit()
    assert len(qc.parameters) == 1


def test_to_qiskit_pow_expr(SymbolCircuit: Any) -> None:
    """_sym_expr_to_qk handles Pow with integer exponent (line 632-640)."""
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(1)
    sc.rz(0, theta=theta**2)  # Pow(theta, 2)
    qc = sc.to_qiskit()
    assert len(qc.parameters) == 1


def test_sym_expr_to_qk_negative_pow(SymbolCircuit: Any) -> None:
    """_sym_expr_to_qk: Pow with negative integer exponent (line 638-639)."""
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(1)
    sc.rz(0, theta=theta ** sympy.Integer(-1))  # 1/theta
    qc = sc.to_qiskit()
    assert len(qc.parameters) == 1


def test_to_qiskit_iswap_gate(SymbolCircuit: Any) -> None:
    """iswap branch in to_qiskit (lines 557-560)."""
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(2)
    sc.iswap(0, 1, theta=theta)
    qc = sc.to_qiskit()
    assert len(qc.parameters) == 1


def test_to_qiskit_pi_in_arithmetic_expr(SymbolCircuit: Any) -> None:
    """theta + sympy.pi hits the non-translatable fallback in _sym_expr_to_qk (644-646)."""
    pytest.importorskip("qiskit")
    theta = sympy.Symbol("theta", real=True)
    sc = SymbolCircuit(1)
    sc.rz(
        0, theta=theta + sympy.pi
    )  # Add(theta, pi); pi is non-translatable in _sym_expr_to_qk
    qc = sc.to_qiskit()
    assert len(qc.parameters) == 1


def test_to_qiskit_r_symbolic_raises(SymbolCircuit: Any) -> None:
    """r gate with symbolic params raises NotImplementedError (lines 583-584)."""
    pytest.importorskip("qiskit")
    theta, alpha, phi = sympy.symbols("theta alpha phi", real=True)
    sc = SymbolCircuit(1)
    sc.r(0, theta=theta, alpha=alpha, phi=phi)
    with pytest.raises(NotImplementedError, match="r gate with symbolic"):
        sc.to_qiskit()


def test_apply_gate_with_custom_name_fallback(SymbolCircuit: Any) -> None:
    """Calling a gate with a name= override triggers fallback SYM_SGATE_MAP lookup (line 131)."""
    sc = SymbolCircuit(1)
    # Pass name="custom_alias" — localname not in SYM_SGATE_MAP, falls back to defaultname="h"
    sc.h(0, name="custom_alias")
    psi = sc.wavefunction()
    v = sympy.Rational(1, 1) / sympy.sqrt(2)
    assert sympy.simplify(psi[0] - v) == 0


def test_to_qiskit_else_fallback_logs_warning(SymbolCircuit: Any) -> None:
    """Gate not in any to_qiskit branch hits else fallback with AttributeError (lines 596-603)."""
    pytest.importorskip("qiskit")
    theta, alpha, phi = sympy.symbols("theta alpha phi", real=True)
    sc = SymbolCircuit(2)
    sc.cr(0, 1, theta=theta, alpha=alpha, phi=phi)  # cr has no Qiskit native method
    qc = sc.to_qiskit()  # should not raise; AttributeError is caught and warning logged
    assert len(qc.data) == 0  # gate was skipped


def test_backend_isolation(jaxb, SymbolCircuit):
    # Ensure SymbolCircuit works even when global backend is JAX
    sc = SymbolCircuit(2)
    sc.h(0)
    theta = sympy.Symbol("theta", real=True)
    sc.rx(1, theta=theta)

    # Test methods that involve contraction
    m = sc.matrix()
    assert m.shape == (4, 4)

    wv = sc.wavefunction()
    assert wv.shape == (4,)

    prob = sc.probability()
    assert prob.shape == (4,)

    exp = sc.expectation_ps(z=[0])
    # |00> -> H0 -> (|00>+|10>)/sqrt(2). <Z0> = (1 + (-1))/2 = 0.
    assert sympy.simplify(exp) == 0


def test_symbol_circuit_matrix(SymbolCircuit):
    sc = SymbolCircuit(1)
    sc.x(0)
    m = sc.matrix()
    assert np.all(m == np.array([[0, 1], [1, 0]], dtype=object))


def test_symbol_circuit_probability(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)
    prob = sc.probability()
    # |cos(theta/2)|**2 and |sin(theta/2)|**2
    assert sympy.simplify(prob[0] - sympy.cos(theta / 2) ** 2) == 0
    assert sympy.simplify(prob[1] - sympy.sin(theta / 2) ** 2) == 0


def test_symbol_circuit_quoperator(SymbolCircuit):
    sc = SymbolCircuit(1)
    sc.x(0)
    quo = sc.get_quoperator()
    assert isinstance(quo, tc.quantum.QuOperator)


def test_symbol_circuit_projected_subsystem(SymbolCircuit):
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.cnot(0, 1)
    # state (|00> + |11>)/sqrt(2)
    # project qubit 0 to |0>, should get |0> on qubit 1
    res = sc.projected_subsystem(np.array([0, 0]), left=(1,))
    assert res[0] == 1
    assert res[1] == 0


def test_symbol_circuit_any_gate(SymbolCircuit):
    sc = SymbolCircuit(1)
    unitary = np.array([[0, 1], [1, 0]])
    sc.any(0, unitary=unitary)
    assert sc.amplitude("1") == 1


def test_symbol_circuit_sampling_error(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)

    with pytest.raises(NotImplementedError, match="Bind symbols first"):
        sc.sample(batch=1)

    with pytest.raises(NotImplementedError, match="Bind symbols first"):
        sc.measure(0)

    with pytest.raises(NotImplementedError, match="Bind symbols first"):
        sc.measure_reference(0)

    with pytest.raises(NotImplementedError, match="does not support cond_measurement"):
        sc.cond_measurement(0)


def test_symbol_circuit_sample_expectation_ps_shots(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(1)
    sc.rx(0, theta=theta)

    # Analytical should work
    res = sc.sample_expectation_ps(z=[0], shots=None)
    assert sympy.simplify(res - sympy.cos(theta)) == 0

    # Numerical shots should raise
    with pytest.raises(
        NotImplementedError, match="does not support numerical sampling"
    ):
        sc.sample_expectation_ps(z=[0], shots=100)


def test_symbol_circuit_dim_raise(SymbolCircuit):
    with pytest.raises(ValueError, match="only supports qubit"):
        SymbolCircuit(1, dim=3)


def test_symbol_circuit_any_to_circuit(SymbolCircuit):
    sc = SymbolCircuit(1)
    unitary = np.array([[0, 1], [1, 0]])
    sc.any(0, unitary=unitary)
    # This used to crash because to_circuit tried to complex(ndarray)
    c = sc.to_circuit({})
    np.testing.assert_allclose(complex(c.amplitude("1")), 1.0, atol=1e-7, rtol=0.0)


def test_symbol_circuit_any_bind(SymbolCircuit):
    sc = SymbolCircuit(1)
    unitary = np.array([[0, 1], [1, 0]])
    sc.any(0, unitary=unitary)
    # Test partial/full bind with any gate present
    sc2 = sc.bind({})
    assert sc2.gate_count() == 1
    np.testing.assert_allclose(complex(sc2.amplitude("1")), 1.0, atol=1e-7, rtol=0.0)


def test_symbol_circuit_qir_roundtrip(SymbolCircuit, sym):
    theta = sym["theta"]
    sc = SymbolCircuit(2)
    sc.h(0)
    sc.rx(1, theta=theta)
    sc.cnot(0, 1)

    qir = sc.to_qir()
    # Reconstruction via from_qir
    sc2 = SymbolCircuit.from_qir(qir)

    assert sc2.gate_count() == 3
    assert sc2.free_symbols() == {theta}

    # Check that reconstructed circuit produces same symbolic wavefunction
    psi1 = sc.wavefunction()
    psi2 = sc2.wavefunction()
    for i in range(len(psi1)):
        assert sympy.simplify(psi1[i] - psi2[i]) == 0
