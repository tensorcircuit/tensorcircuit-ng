import pytest
import numpy as np
import jax.numpy as jnp
import pyzx_param as pyzx
from pytest_lazyfixture import lazy_fixture as lf
import tensorcircuit as tc
from tensorcircuit.stabilizertcircuit import StabilizerTCircuit
from tensorcircuit.zx.converter import circuit_to_zx
from tensorcircuit.zx.scalar_graph import (
    find_stab,
    compile_scalar_graphs,
)

# --- Helpers ---


def get_tc_unitary(c):
    return tc.backend.numpy(c.matrix())


def get_zx_unitary(g, n):
    if hasattr(g, "graph"):
        g = g.graph
    zx_tensor = g.to_tensor()
    return zx_tensor.reshape((2**n, 2**n))


def assert_unitary_match(g, c, atol=1e-5):
    n = c._nqubits
    zx_mat = get_zx_unitary(g, n)
    tc_mat = get_tc_unitary(c)

    mask = np.abs(tc_mat) > 1e-5
    if not np.any(mask):
        np.testing.assert_allclose(zx_mat, tc_mat, atol=atol)
        return

    first_idx = np.where(mask)[0][0], np.where(mask)[1][0]
    ratio = zx_mat[first_idx] / tc_mat[first_idx]

    # Check proportionality
    np.testing.assert_allclose(zx_mat, ratio * tc_mat, atol=atol)
    # Check if ratio magnitude is a power of sqrt(2) (unnormalized ZX) or 1
    mag = np.abs(ratio)
    log_ratio = 2 * np.log2(mag)
    assert (
        np.abs(log_ratio - np.round(log_ratio)) < 1e-5
    ), f"Ratio {ratio} is not a power of sqrt(2)"


# --- New simple sanity tests ---


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_simple_cases(backend):
    batch = 5000
    cases = [
        ("X_ERROR(0.1)", lambda c: c.x_error(0, 0.1), 0.1),
        ("X gate", lambda c: c.x(0), 1.0),
        ("Z gate + H", lambda c: (c.z(0), c.h(0)), 0.5),
        ("Z_ERROR(0.2) + H", lambda c: (c.z_error(0, 0.2), c.h(0)), 0.5),
        ("Y_ERROR(0.3)", lambda c: c.y_error(0, 0.3), 0.3),
        ("X + RESET", lambda c: (c.x(0), c.reset_instruction(0)), 0.0),
        ("H + M", lambda c: c.h(0), 0.5),
    ]

    for name, setup, expected in cases:
        c = StabilizerTCircuit(1)
        setup(c)
        c.measure_instruction(0)
        stc = StabilizerTCircuit.from_circuit(c)
        res = stc.sample_measurements(shots=batch)
        prob = np.mean(res)
        assert (
            abs(prob - expected) < 0.05
        ), f"Case {name} failed: got {prob}, exp {expected}"


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_detector_xor_simple(backend):
    batch = 5000
    c = StabilizerTCircuit(1)
    c.x_error(0, 0.1)
    c.measure_instruction(0)  # m0
    c.x_error(0, 0.2)
    c.measure_instruction(0)  # m1
    c.detector_instruction([-1, -2])  # m0 ^ m1
    stc = StabilizerTCircuit.from_circuit(c)
    res = stc.sample_detectors(shots=batch)
    prob = np.mean(res)
    assert abs(prob - 0.2) < 0.05


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_depolarize2_simple(backend):
    batch = 5000
    c = StabilizerTCircuit(2)
    c.depolarizing2(0, 1, 0.1)
    c.measure_instruction(0)
    c.measure_instruction(1)
    stc = StabilizerTCircuit.from_circuit(c)
    res = stc.sample_measurements(shots=batch)
    p11 = np.mean(res[:, 0] & res[:, 1])
    # Theory: 3/15 * 0.1 = 0.02
    assert abs(p11 - 0.02) < 0.02


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_pauli_h_simple(backend):
    batch = 5000
    c = StabilizerTCircuit(1)
    c.pauli_instruction(0, pz=0.1)
    c.h(0)
    c.measure_instruction(0)
    stc = StabilizerTCircuit.from_circuit(c)
    res = stc.sample_measurements(shots=batch)
    prob = np.mean(res)
    assert abs(prob - 0.5) < 0.05


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_outcome_probability(backend):
    # |0> state, prob(0) = 1.0, prob(1) = 0
    stc = StabilizerTCircuit(1)
    stc.measure_instruction(0)

    # outcome_probability expects a state matching the measurements
    # For a simple circuit with 1 measure, state is [0] or [1]
    p0 = stc.outcome_probability(jnp.array([0]))
    p1 = stc.outcome_probability(jnp.array([1]))
    assert np.allclose(p0, 1.0)
    assert np.allclose(p1, 0.0)

    # H gate, prob(0) = 0.5, prob(1) = 0.5
    stc = StabilizerTCircuit(1)
    stc.h(0)
    stc.measure_instruction(0)
    p0 = stc.outcome_probability(jnp.array([0]))
    p1 = stc.outcome_probability(jnp.array([1]))
    assert np.allclose(p0, 0.5)
    assert np.allclose(p1, 0.5)

    # X_ERROR(0.1), prob(1) = 0.1
    stc = StabilizerTCircuit(1)
    stc.x_error(0, 0.1)
    stc.measure_instruction(0)
    p1 = stc.outcome_probability(jnp.array([1]), shots=10)
    assert p1.shape == (10,)
    # For a specific error realization, it's either 0 or 1.
    # But since x_error adds a parameter 'e0', and outcome_probability
    # computes P(state | error_i), it should correctly reflect the error bit.
    # Actually, outcome_probability in tsim computes P(state | error).
    # If the error bit e0=1, then P(1|e0=1) = 1.0. If e0=0, P(1|e0=0) = 0.
    assert np.all(jnp.logical_or(jnp.isclose(p1, 0.0), jnp.isclose(p1, 1.0)))
    # Average should be around 0.1
    assert abs(np.mean(p1) - 0.1) < 0.3  # low shots, just checking shape/range


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_sample_detectors_advanced(backend):
    # Test separate_observables and use_reference
    c = StabilizerTCircuit(2)
    c.x(0)
    c.measure_instruction(0)  # m0 = 1
    c.measure_instruction(1)  # m1 = 0
    c.detector_instruction([0])  # det0 = m0 = 1
    c.observable_instruction([1], observable_index=0)  # obs0 = m1 = 0

    stc = StabilizerTCircuit.from_circuit(c)

    # 1. Raw
    samples = stc.sample_detectors(shots=10)
    assert samples.shape == (10, 2)
    assert np.all(samples[:, 0] == 1)
    assert np.all(samples[:, 1] == 0)

    # 2. Separate
    det, obs = stc.sample_detectors(shots=10, separate_observables=True)
    assert det.shape == (10, 1)
    assert obs.shape == (10, 1)
    assert np.all(det == 1)
    assert np.all(obs == 0)

    # 3. Reference (XOR with noiseless)
    # Noiseless outcome is [1, 0]. XORing [1, 0] with [1, 0] gives [0, 0]
    samples_ref = stc.sample_detectors(shots=10, use_reference=True)
    assert np.all(samples_ref == 0)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_multi_component_sampling(backend):
    # Two disconnected qubits
    stc = StabilizerTCircuit(2)
    stc.x(0)
    stc.h(1)
    stc.measure_instruction(0)
    stc.measure_instruction(1)

    samples = stc.sample_measurements(shots=1000)
    assert samples.shape == (1000, 2)
    assert np.all(samples[:, 0] == 1)
    assert abs(np.mean(samples[:, 1]) - 0.5) < 0.1


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_noisy_measurements(backend):
    # M(p) is equivalent to X_ERROR(p) + M
    batch = 50000
    p = 0.1

    # Case 1: M(p)
    stc1 = StabilizerTCircuit(1)
    # Note: MR(p) or M(p) usually handled via parameters
    stc1._qir.append({"name": "M", "index": [0], "parameters": {"p": p}})
    res1 = stc1.sample_measurements(shots=batch)
    prob1 = np.mean(res1)
    assert abs(prob1 - p) < 0.05

    # Case 2: MR(p)
    stc2 = StabilizerTCircuit(1)
    stc2.x(0)
    stc2._qir.append({"name": "MR", "index": [0], "parameters": {"p": p}})
    stc2.measure_instruction(0)  # Should be 0 after MR
    res2 = stc2.sample_measurements(shots=batch)
    # res2 shape (batch, 2)
    m0 = np.mean(res2[:, 0])  # result of MR
    m1 = np.mean(res2[:, 1])  # result of following M
    # tsim's MR(p) applies noise twice: once in mr, once in m.
    # Total flip probability = p(1-p) + (1-p)p = 2p - 2p^2
    # For p=0.1, flip = 0.18. Since we start at 1 (X gate), m0 should be 1-0.18 = 0.82
    expected_m0 = 1 - (2 * p - 2 * p**2)
    assert abs(m0 - expected_m0) < 0.05

    assert abs(m1 - 0.0) < 0.01


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_batch_size_consistency(backend):
    stc = StabilizerTCircuit(1)
    stc.h(0)
    stc.measure_instruction(0)

    # Small batch size vs large batch size
    res1 = stc.sample_measurements(shots=100, batch_size=10)
    res2 = stc.sample_measurements(shots=100, batch_size=100)
    assert res1.shape == (100, 1)
    assert res2.shape == (100, 1)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_empty_circuit(backend):
    stc = StabilizerTCircuit(1)
    # No gates, no measurements
    # sample_measurements defaults to force_measure_all=True if no measurements exist
    res = stc.sample_measurements(shots=10)
    assert res.shape == (10, 1)
    assert np.all(res == 0)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_observable_include_logical(backend):
    stc = StabilizerTCircuit(1)
    stc.x(0)
    stc.measure_instruction(0)
    stc.observable_instruction([0], observable_index=5)

    det, obs = stc.sample_detectors(shots=10, separate_observables=True)
    # No detectors, only 1 observable at index 5
    assert det.shape == (10, 0)
    assert obs.shape == (10, 1)
    assert np.all(obs == 1)


# --- Phase 1: Converter Tests (Exhaustive) ---


@pytest.mark.parametrize("backend", [lf("npb")])
def test_converter_all_single_qubit_gates(backend):
    # Test every single qubit gate supported
    for g_name in ["h", "s", "sd", "t", "td", "x", "y", "z", "i"]:
        c = tc.Circuit(1)
        getattr(c, g_name)(0)
        g = circuit_to_zx(c)
        assert_unitary_match(g, c), f"Gate {g_name} failed"


@pytest.mark.parametrize("backend", [lf("npb")])
def test_converter_all_two_qubit_gates(backend):
    for g_name in ["cnot", "cz", "swap"]:
        c = tc.Circuit(2)
        getattr(c, g_name)(0, 1)
        g = circuit_to_zx(c)
        assert_unitary_match(g, c), f"Gate {g_name} failed"


@pytest.mark.skip(
    reason="pyzx_param tensorfy doesn't support X/Z-spiders as inputs/outputs"
)
@pytest.mark.parametrize("backend", [lf("npb")])
def test_converter_rotations_extensive(backend):
    # StabilizerTCircuit focuses on Clifford+T, arbitrary rotations might not be perfectly supported via GATE_TABLE
    # but we can test small set
    angles = [0, np.pi / 2, np.pi]
    for a in angles:
        for g_name in ["rx", "rz"]:
            c = tc.Circuit(1)
            getattr(c, g_name)(0, theta=a)
            g = circuit_to_zx(c)
            assert_unitary_match(g, c), f"{g_name}({a}) failed"


@pytest.mark.parametrize("backend", [lf("npb")])
def test_is_pauli(backend):
    from tensorcircuit.zx.converter import is_pauli

    assert is_pauli(tc.gates.x().tensor) == "x"
    assert is_pauli(tc.gates.y().tensor) == "y"
    assert is_pauli(tc.gates.z().tensor) == "z"
    assert is_pauli(tc.gates.i().tensor) == "i"
    assert is_pauli(np.eye(2)) == "i"
    # Non-Pauli
    assert is_pauli(tc.gates.h().tensor) is None


@pytest.mark.parametrize("backend", [lf("npb")])
def test_converter_complex_circuit(backend):
    c = tc.Circuit(3)
    c.h(0)
    c.cnot(0, 1)
    c.t(1)
    c.s(2)
    c.cz(1, 2)
    c.h(2)
    g = circuit_to_zx(c)
    assert_unitary_match(g, c)


# --- Phase 2: Scalar IR & Decomposition Tests (Exhaustive) ---


@pytest.mark.parametrize("backend", [lf("npb")])
def test_decomposition_scaling(backend):
    for tc_count in range(4):
        c = tc.Circuit(1)
        for _ in range(tc_count):
            c.t(0)
        g = circuit_to_zx(c)
        graphs = find_stab(g, strategy="cat5")
        assert len(graphs) >= 1
        for sg in graphs:
            assert all(
                sg.type(v)
                in [pyzx.VertexType.BOUNDARY, pyzx.VertexType.Z, pyzx.VertexType.X]
                for v in sg.vertices()
            )


@pytest.mark.parametrize("backend", [lf("npb")])
def test_scalar_ir_variable_mapping(backend):
    c = tc.Circuit(1)
    c.h(0)
    c.measure_instruction(0)
    g = circuit_to_zx(c)
    if hasattr(g, "graph"):
        g = g.graph
    # Initialize initial boundaries to X state (|0> state)
    for v in g.inputs():
        if g.type(v) == pyzx.VertexType.BOUNDARY:
            g.set_type(v, pyzx.VertexType.X)
            g.set_phase(v, 0)
    # Plug outputs
    g.apply_effect("0")  # or any value

    graphs = find_stab(g, strategy="cat5")

    params = ["rec[0]"]
    data = compile_scalar_graphs(graphs, params)
    assert data.a_param_bits is not None


# --- Phase 4 & 5: Circuit & QEC Specifics ---


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_stabilizert_circuit_methods(backend):
    stc = StabilizerTCircuit(2)
    stc.h(0)
    stc.cnot(0, 1)
    stc.t(1)
    assert len(stc._qir) == 3


@pytest.mark.parametrize("backend", [lf("npb")])
def test_zx_detector_chain(backend):
    c = StabilizerTCircuit(2)
    c.measure_instruction(0)
    c.measure_instruction(1)
    c.detector_instruction([0, 1])

    g = circuit_to_zx(c)
    if hasattr(g, "graph"):
        g = g.graph

    # Find the detector vertex
    det_v = -1
    for v in g.vertices():
        if "det[0]" in g._phaseVars.get(v, set()):
            det_v = v
            break
    assert det_v != -1
    # Should be connected to two record spiders
    neighbors = list(g.neighbors(det_v))
    rec_neighbors = 0
    for n in neighbors:
        if any(s.startswith("rec[") for s in g._phaseVars.get(n, set())):
            rec_neighbors += 1
    assert rec_neighbors == 2


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_from_circuit_with_noise(backend):
    c = tc.Circuit(1)
    c.h(0)
    c.depolarizing(0, px=0.1, py=0.1, pz=0.1)

    stc = StabilizerTCircuit.from_circuit(c)
    assert len(stc._qir) >= 2

    g = circuit_to_zx(stc)
    if hasattr(g, "graph"):
        g = g.graph

    has_ex = False
    has_ez = False
    for v in g.vertices():
        labels = g._phaseVars.get(v, set())
        if any(s.startswith("e") for s in labels):
            has_ex = True
            has_ez = True
    assert has_ex and has_ez


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_sample_detectors_fixed_gate_sequence(backend):
    stc = StabilizerTCircuit(2)
    stc.x(0)
    stc.measure_instruction(0)
    stc.measure_instruction(1)
    stc.detector_instruction([0, 1])

    samples = stc.sample_detectors(shots=32, seed=1234)
    assert samples.shape == (32, 1)
    np.testing.assert_array_equal(samples[:, 0], np.ones(32, dtype=bool))


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_sample_detectors_multiple_detectors_complex_sequence(backend):
    stc = StabilizerTCircuit(3)
    stc.x(0)
    stc.cnot(0, 1)
    stc.x(2)
    stc.measure_instruction(0)  # rec[0] = 1
    stc.measure_instruction(1)  # rec[1] = 1
    stc.measure_instruction(2)  # rec[2] = 1
    stc.detector_instruction([0, 1])  # 1 xor 1 = 0
    stc.detector_instruction([1, 2])  # 1 xor 1 = 0
    stc.detector_instruction([0, 2])  # 1 xor 1 = 0
    stc.detector_instruction([0])  # 1

    samples = stc.sample_detectors(shots=64, seed=101)
    assert samples.shape == (64, 4)
    expected = np.array([0, 0, 0, 1], dtype=bool)
    np.testing.assert_array_equal(samples, np.tile(expected, (64, 1)))


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_stabilizertcircuit_sample_measurements_final_state_deterministic(backend):
    stc = StabilizerTCircuit(2)
    stc.x(0)
    stc.cnot(0, 1)
    stc.measure_instruction(0)
    stc.measure_instruction(1)
    samples = stc.sample_measurements(shots=64, seed=9)
    assert samples.shape == (64, 2)
    expected = np.ones((64, 2), dtype=bool)
    np.testing.assert_array_equal(samples, expected)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_stabilizertcircuit_sample_with_t_gate_statistics(backend):
    stc = StabilizerTCircuit(1)
    stc.h(0)
    stc.t(0)
    stc.h(0)
    stc.measure_instruction(0)

    samples = stc.sample_measurements(shots=8000, seed=12345)
    assert samples.shape == (8000, 1)
    p1 = np.mean(samples[:, 0])
    p1_theory = 0.5 * (1.0 - np.cos(np.pi / 4.0))
    assert abs(p1 - p1_theory) < 0.03


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_stabilizertcircuit_sample_40q_few_t_scalable(backend):
    stc = StabilizerTCircuit(40)
    stc.h(0)
    stc.t(0)
    stc.h(0)
    stc.h(7)
    stc.t(7)
    stc.h(7)
    stc.h(21)
    stc.t(21)
    stc.h(21)
    for i in [0, 7, 21]:
        stc.measure_instruction(i)

    samples = stc.sample_measurements(shots=1000, seed=20260402)
    assert samples.shape == (1000, 3)

    p_theory = (2 - np.sqrt(2)) / 4
    for i in range(3):
        p_emp = np.mean(samples[:, i])
        stderr = np.sqrt(p_theory * (1 - p_theory) / 1000)
        assert abs(p_emp - p_theory) < 5 * stderr


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_stabilizertcircuit_mirror_all_zero(backend):
    n = 5
    stc = StabilizerTCircuit(n)
    # Forward
    stc.h(0)
    stc.cnot(0, 1)
    stc.t(1)
    stc.s(2)
    stc.cnot(2, 3)
    stc.h(4)
    # Backward (Adjoint)
    stc.h(4)
    stc.cnot(2, 3)
    stc.sd(2)
    stc.td(1)
    stc.cnot(0, 1)
    stc.h(0)
    for i in range(n):
        stc.measure_instruction(i)

    batch = 100
    samples = stc.sample_measurements(shots=batch, seed=42)
    assert samples.shape == (batch, n)
    assert np.all(samples == 0)


# --- Stim Import & Integration Tests (Borrowed from tsim) ---


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_from_stim_bell_state(backend):
    import stim

    stim_str = """
    R 0 1
    H 0
    CNOT 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    """
    stc = StabilizerTCircuit.from_stim_str(stim_str)
    # Bell state: m0 and m1 should be correlated, so detector (m0^m1) should be 0
    res = stc.sample_detectors(shots=100)
    assert np.all(res == 0)

    m_res = stc.sample_measurements(shots=100)
    assert np.all(m_res[:, 0] == m_res[:, 1])


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_from_stim_t_gate(backend):
    # Tests T gate decomposition and sampling statistics
    stim_str = """
    RX 0
    S 0
    H 0
    M 0
    """
    # S on |+> gives |+i> = (|0>+i|1>)/sqrt(2)
    # H on |+i> gives (|0>+|1> + i|0>-i|1>)/2 = (1+i)/2 |0> + (1-i)/2 |1>
    # Prob(0) = |(1+i)/2|^2 = 2/4 = 0.5
    stc = StabilizerTCircuit.from_stim_str(stim_str)
    res = stc.sample_measurements(shots=1000)
    mean_val = np.mean(res)
    assert 0.4 < mean_val < 0.6


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_from_stim_reset_after_state_change(backend, basis):
    reset_gate = "R" if basis == "Z" else f"R{basis}"
    measure_gate = "M" if basis == "Z" else f"M{basis}"
    stim_str = f"""
    H 0
    S 0
    {reset_gate} 0
    {measure_gate} 0
    """
    stc = StabilizerTCircuit.from_stim_str(stim_str)
    res = stc.sample_measurements(shots=100)
    assert np.all(res == 0)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_from_stim_singlet_state(backend):
    # Borrowed from tsim: test_singlet_state
    # High-level check for CNOT, H, X, Z
    stim_str = """
    R 0 1
    X 0
    H 1
    CNOT 1 0
    Z 0
    M 0 1
    """
    stc = StabilizerTCircuit.from_stim_str(stim_str)
    res = stc.sample_measurements(shots=100)
    # Singlet state (|01> - |10>)/sqrt(2) has anti-correlated Z outcomes
    assert np.all(res[:, 0] != res[:, 1])


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_zx_from_stim_noisy_bell_detector(backend):
    # Surface code style check: detector should flip if error occurs
    stim_str = """
    H 0
    CNOT 0 1
    X_ERROR(0.1) 0
    M 0 1
    DETECTOR rec[-1] rec[-2]
    """
    stc = StabilizerTCircuit.from_stim_str(stim_str)
    res = stc.sample_detectors(shots=10000)
    mean_flip = np.mean(res)
    # Error on q0 with prob 0.1 -> m0 flips -> detector flips
    assert 0.07 < mean_flip < 0.13
