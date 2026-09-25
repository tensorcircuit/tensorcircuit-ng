import os
import sys

import pytest
import numpy as np
from scipy.linalg import expm

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


def test_analog_circuit_init(jaxb):
    # Test initialization of AnalogCircuit
    nqubits = 4
    ac = tc.AnalogCircuit(nqubits)

    assert ac.num_qubits == nqubits
    assert len(ac.digital_circuits) == 1
    assert len(ac.analog_blocks) == 0
    assert ac.current_digital_circuit is not None
    print(ac.effective_circuit)


def test_analog_circuit_qudit_rejected(jaxb):
    # AnalogCircuit does not yet support qudits: the analog-block solvers in
    # `timeevol` are qubit-only (log2 / reshape2). Construction with d != 2 must
    # fail fast instead of crashing deep in state() later.
    with pytest.raises(ValueError):
        tc.AnalogCircuit(2, dim=3)
    # qubit (d=2 and default) still constructs fine.
    assert tc.AnalogCircuit(2).num_qubits == 2
    assert tc.AnalogCircuit(2, dim=2).num_qubits == 2


def test_analog_circuit_digital_gates(jaxb):
    # Test that digital gates can be applied to the analog circuit
    ac = tc.AnalogCircuit(2)
    ac.h(1)
    ac.H([0, 1])
    ac.cx(0, 1)

    # Should have added gates to the current digital circuit
    state = ac.state()
    expected_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # Bell state
    np.testing.assert_allclose(tc.backend.numpy(state), expected_state, atol=1e-6)


def test_analog_circuit_add_analog_block(jaxb):
    # Test adding an analog block
    def hamiltonian_func(t):
        return tc.quantum.PauliString2COO([1, 1])

    ac = tc.AnalogCircuit(2)
    ac.h(0)  # Add a digital gate first

    # Add an analog block
    ac.add_analog_block(hamiltonian_func, 1.0, [0, 1])

    # Check that the analog block was added
    assert len(ac.analog_blocks) == 1
    assert len(ac.digital_circuits) == 2  # Should have created a new digital circuit
    np.testing.assert_allclose(ac.analog_blocks[0].time, np.array([0, 1.0]))
    assert ac.analog_blocks[0].index == [0, 1]
    assert ac.analog_blocks[0].hamiltonian_func == hamiltonian_func

    ac1 = tc.AnalogCircuit(2)
    ac1.h(0)  # Add a digital gate first

    # Add an analog block
    ac1.add_analog_block(hamiltonian_func, 1.0)
    np.testing.assert_allclose(ac1.state(), ac.state(), atol=1e-6)

    c = tc.Circuit(2)
    c.h(0)
    c.rxx(0, 1, theta=2.0)
    s_ref = c.state()
    np.testing.assert_allclose(s_ref, ac.state(), atol=1e-6)


def test_analog_circuit_effective_circuit(jaxb):
    # Test that the effective circuit is properly created
    ac = tc.AnalogCircuit(2)
    ac.x(0)

    def hamiltonian_func(t):
        return tc.quantum.PauliString2COO([3])

    ac.add_analog_block(hamiltonian_func, 1.0, [0])

    # Before calling state(), effective_circuit should be None
    assert ac._effective_circuit is None

    # After calling state(), effective_circuit should be created
    ac.state()
    assert ac._effective_circuit is not None
    ac.s(1)
    assert ac._effective_circuit is None
    print(ac)


def test_analog_circuit_expectation(jaxb):
    # Test expectation value calculation
    ac = tc.AnalogCircuit(2)
    ac.x(0)
    ac.cnot(0, 1)

    def hamiltonian_func(t):
        return tc.quantum.PauliString2COO([3, 3]) * tc.backend.cos(t)

    ac.add_analog_block(hamiltonian_func, 1.0, atol=1e-9, rtol=1e-8)

    # Measure Z expectation value
    z_op = tc.gates.Gate(tc.gates._z_matrix)
    expectation_val = ac.expectation((z_op, [0]))
    np.testing.assert_allclose(tc.backend.real(expectation_val), -1.0, atol=1e-6)
    expectation_val = ac.expectation_ps(z=[1])
    np.testing.assert_allclose(tc.backend.real(expectation_val), -1.0, atol=1e-6)


def test_analog_circuit_sample(jaxb):
    # Test sampling from the circuit
    ac = tc.AnalogCircuit(3)
    ac.h(0)  # Prepare |+> state
    hmatrix = tc.quantum.PauliStringSum2COO([[1, 1, 0], [0, 2, 2]], [0.3, 0.5])

    def hamiltonian_func(t):
        return hmatrix

    ac.add_analog_block(hamiltonian_func, 1.1)
    ac.rz(2, theta=-0.4)
    ac.cx(1, 2)
    ac.add_analog_block(hamiltonian_func, 0.3)
    ac.h([0, 1])
    # Sample from the circuit
    samples = ac.sample(batch=100, allow_state=True, format="count_dict_bin")
    print(samples)
    print(ac.amplitude("010"))
    print(ac.probability())


def test_analog_circuit_ad_jit(jaxb):
    try:
        import diffrax  # pylint: disable=unused-import
    except ImportError:
        pytest.skip("diffrax not installed, skipping test")

    def cost_fn(param):
        ac = tc.AnalogCircuit(2)
        ac.set_solver_options(ode_backend="diffrax")
        ac.h(0)
        h0 = tc.quantum.PauliString2COO([1, 1])

        def hamiltonian(t):
            return h0 * param[0] * tc.backend.cos(t * param[1] + param[2])

        ac.add_analog_block(hamiltonian, [1.0, 2.3])
        # jaxodeint somehow incompatible with h0 outside hamiltonian function
        # https://github.com/jax-ml/jax/issues/31792
        ac.cx(1, 0)
        return tc.backend.real(ac.expectation_ps(z=[1]))

    # Test JIT compilation
    gf = tc.backend.jit(tc.backend.grad(cost_fn))
    param = tc.backend.ones([3])
    param = tc.backend.cast(param, dtype="float32")

    eps = 1e-4 * np.array([1.0, 0, 0])
    num_grad = (cost_fn(param + eps) - cost_fn(param - eps)) / (2e-4)

    np.testing.assert_allclose(gf(param)[0], num_grad, atol=1e-3)


def test_analog_circuit_consistency(jaxb):
    def hfunc(t):
        return tc.backend.convert_to_tensor(
            np.array([[1, 0], [0, -1]], dtype=np.complex64)
        )

    ac = tc.AnalogCircuit(2)
    ac.h(0)
    ac.add_analog_block(hfunc, 1.0, [0])
    ac.x(1)

    # Test append
    ac2 = tc.AnalogCircuit(2)
    ac2.h(0)
    ac2.add_analog_block(hfunc, 0.5, [0])
    ac.append(ac2)

    # Sequence: D0 -> B0 -> D1 -> B1 -> D2
    assert len(ac.analog_blocks) == 2
    assert len(ac.digital_circuits) == 3

    # Test inverse
    inv_ac = ac.inverse()
    assert len(inv_ac.analog_blocks) == 2
    assert len(inv_ac.digital_circuits) == 3
    # Verify time is preserved (not negated) — inverse negates Hamiltonian instead
    np.testing.assert_allclose(
        tc.backend.numpy(inv_ac.analog_blocks[0].time), [0, 0.5], atol=1e-5
    )
    np.testing.assert_allclose(
        tc.backend.numpy(inv_ac.analog_blocks[1].time), [0, 1.0], atol=1e-5
    )
    # Verify Hamiltonian is negated
    t0 = tc.backend.convert_to_tensor(0.0)
    orig_h = tc.backend.numpy(hfunc(t0))
    inv_h = tc.backend.numpy(inv_ac.analog_blocks[0].hamiltonian_func(t0))
    np.testing.assert_allclose(inv_h, -orig_h, atol=1e-7)

    # Test identity (simple): 1 analog block
    ac3 = tc.AnalogCircuit(2)
    ac3.h(0)
    ac3.add_analog_block(hfunc, 0.5, [0])
    ac3.append(ac3.inverse())
    np.testing.assert_allclose(tc.backend.numpy(ac3.amplitude("00")), 1.0, atol=1e-5)

    # Test identity (deep): 3 qubits, 2 analog blocks, rich digital layers
    def hfunc_xx(t):
        # XX coupling Hamiltonian on 2 qubits
        return tc.backend.convert_to_tensor(
            np.array(
                [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                dtype=np.complex64,
            )
        )

    ac4 = tc.AnalogCircuit(3)
    ac4.h(0)
    ac4.cnot(0, 1)
    ac4.rx(2, theta=tc.array_to_tensor(0.7))
    ac4.add_analog_block(hfunc, 0.3, [1])
    ac4.h(2)
    ac4.cnot(1, 2)
    ac4.add_analog_block(hfunc_xx, 0.4, [0, 1])
    ac4.rx(0, theta=tc.array_to_tensor(1.2))
    ac4.h(1)

    ac4_combined = tc.AnalogCircuit(3)
    ac4_combined.h(0)
    ac4_combined.cnot(0, 1)
    ac4_combined.rx(2, theta=tc.array_to_tensor(0.7))
    ac4_combined.add_analog_block(hfunc, 0.3, [1])
    ac4_combined.h(2)
    ac4_combined.cnot(1, 2)
    ac4_combined.add_analog_block(hfunc_xx, 0.4, [0, 1])
    ac4_combined.rx(0, theta=tc.array_to_tensor(1.2))
    ac4_combined.h(1)
    ac4_combined.append(ac4.inverse())

    assert len(ac4_combined.analog_blocks) == 4
    assert len(ac4_combined.digital_circuits) == 5
    np.testing.assert_allclose(
        tc.backend.numpy(ac4_combined.amplitude("000")), 1.0, atol=1e-4
    )

    # Test identity (global Hamiltonian): 2 qubits, global evolution
    def hfunc_global(t):
        return tc.backend.convert_to_tensor(
            np.array(
                [[1, 0, 0, 0], [0, -1, 0.5, 0], [0, 0.5, -1, 0], [0, 0, 0, 1]],
                dtype=np.complex64,
            )
        )

    ac5 = tc.AnalogCircuit(2)
    ac5.h(0)
    ac5.h(1)
    ac5.add_analog_block(hfunc_global, 0.6)
    ac5.rx(0, theta=tc.array_to_tensor(0.5))
    ac5.append(ac5.inverse())
    np.testing.assert_allclose(tc.backend.numpy(ac5.amplitude("00")), 1.0, atol=1e-4)


@pytest.mark.parametrize("time", [1.0, [0.4, 1.3]])
@pytest.mark.parametrize("index", [None, [0]])
def test_analog_circuit_time_dependent_inverse(jaxb, highp, time, index):
    x, z = tc.gates.x().tensor, tc.gates.z().tensor

    def hamiltonian(t):
        return (1 - t) * x + t * z

    circuit = tc.AnalogCircuit(1)
    circuit.add_analog_block(hamiltonian, time, index, rtol=1e-10, atol=1e-10)
    circuit.append(circuit.inverse())
    np.testing.assert_allclose(circuit.state(), [1.0, 0.0], atol=1e-8, rtol=1e-8)


def test_analog_circuit_time_dependent_inverse_multiple_blocks(jaxb, highp):
    x, y, z = tc.gates.x().tensor, tc.gates.y().tensor, tc.gates.z().tensor

    def first_hamiltonian(t):
        return (1 - t) * x + t * z

    def second_hamiltonian(t):
        return tc.backend.cos(t) * y + tc.backend.sin(t) * z

    circuit = tc.AnalogCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.add_analog_block(first_hamiltonian, [0.2, 0.9], [1], rtol=1e-10, atol=1e-10)
    circuit.rx(0, theta=0.3)
    circuit.cnot(1, 0)
    circuit.add_analog_block(
        second_hamiltonian, [1.1, 1.6], [0], rtol=1e-10, atol=1e-10
    )
    circuit.s(1)
    circuit.append(circuit.inverse())
    np.testing.assert_allclose(
        circuit.state(), [1.0, 0.0, 0.0, 0.0], atol=1e-8, rtol=1e-8
    )


def test_analog_circuit_time_dependent_inverse_ad_jit(jaxb, highp):
    def cost_fn(strength):
        def hamiltonian(t):
            x, z = tc.gates.x().tensor, tc.gates.z().tensor
            return strength * (1 - t) * x + t * z

        circuit = tc.AnalogCircuit(1)
        circuit.add_analog_block(hamiltonian, [0.2, 1.1], rtol=1e-9, atol=1e-9)
        return tc.backend.real(circuit.inverse().expectation_ps(z=[0]))

    strength = tc.backend.convert_to_tensor(0.7)
    value, gradient = tc.backend.jit(tc.backend.value_and_grad(cost_fn))(strength)
    epsilon = 1e-4
    numerical_gradient = (cost_fn(strength + epsilon) - cost_fn(strength - epsilon)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(value, cost_fn(strength), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(gradient, numerical_gradient, atol=1e-6, rtol=1e-6)


def _analog_mps_input(theta, scale=1.0):
    coefficients = tc.backend.stack([tc.backend.cos(theta), 1j * tc.backend.sin(theta)])
    left = tc.gates.Gate(scale * coefficients * tc.backend.eye(2))
    right = tc.gates.x()
    left[1] ^ right[0]
    return tc.quantum.QuVector([left[0], right[1]])


@pytest.mark.parametrize("theta", [0.0, 0.31])
@pytest.mark.parametrize("scale", [1.0, 1.7])
@pytest.mark.parametrize("single_tensor", [False, True])
def test_analog_mps_initial_state(jaxb, theta, scale, single_tensor):
    expected = scale * np.array([0, np.cos(theta), 1j * np.sin(theta), 0])
    if single_tensor:
        mps = tc.quantum.QuVector.from_tensor(
            tc.backend.convert_to_tensor(expected.reshape(2, 2))
        )
    else:
        mps = _analog_mps_input(theta, scale)
    circuit = tc.AnalogCircuit(2, mps_inputs=mps)
    for form, shape in [("default", (4,)), ("ket", (4, 1)), ("bra", (1, 4))]:
        np.testing.assert_allclose(
            circuit.state(form=form), expected.reshape(shape), atol=1e-6
        )
    np.testing.assert_allclose(circuit.amplitude("10"), expected[2], atol=1e-6)
    np.testing.assert_allclose(
        circuit.expectation_ps(z=[1]), -(scale**2) * np.cos(2 * theta), atol=1e-6
    )
    circuit.x(1)
    np.testing.assert_allclose(circuit.state(), expected[[1, 0, 3, 2]], atol=1e-6)
    np.testing.assert_allclose(
        tc.backend.reshape(mps.copy().eval(), [-1]), expected, atol=1e-6
    )


@pytest.mark.parametrize("input_mode", ["default", "dense", "both"])
def test_analog_mps_input_precedence(jaxb, input_mode):
    expected = np.array([1, 0, 0, 0])
    kwargs = {}
    if input_mode != "default":
        expected = np.array([0, 0, 1, 0])
        kwargs["inputs"] = tc.backend.convert_to_tensor(expected)
    if input_mode == "both":
        kwargs["mps_inputs"] = _analog_mps_input(0.31)
    circuit = tc.AnalogCircuit(2, **kwargs)
    np.testing.assert_allclose(circuit.state(), expected, atol=1e-6)
    replacement = tc.backend.convert_to_tensor(np.array([0, 0, 0, 1], dtype=complex))
    circuit.current_digital_circuit.replace_inputs(replacement)
    np.testing.assert_allclose(circuit.state(), replacement, atol=1e-6)


@pytest.mark.parametrize("mode", ["global", "local", "mixed"])
def test_analog_mps_hybrid_evolution(jaxb, highp, mode):
    theta = 0.31
    expected = np.array([0, np.cos(theta), 1j * np.sin(theta), 0])
    circuit = tc.AnalogCircuit(2, mps_inputs=_analog_mps_input(theta))
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    local = 0.4 * x - 0.2 * z
    global_h = 0.3 * np.kron(x, x) + 0.7 * np.kron(z, np.eye(2))
    circuit.h(0)
    expected = np.kron(h, np.eye(2)) @ expected
    if mode == "local":
        circuit.add_analog_block(
            lambda t: tc.backend.convert_to_tensor(local),
            0.23,
            index=[1],
            atol=1e-10,
            rtol=1e-9,
        )
        expected = np.kron(np.eye(2), expm(-0.23j * local)) @ expected
    else:
        circuit.add_analog_block(
            lambda t: tc.backend.convert_to_tensor(global_h),
            0.23,
            atol=1e-10,
            rtol=1e-9,
        )
        expected = expm(-0.23j * global_h) @ expected
    if mode == "mixed":
        circuit.x(1)
        expected = np.kron(np.eye(2), x) @ expected
        circuit.add_analog_block(
            lambda t: tc.backend.convert_to_tensor(local),
            0.19,
            index=[1],
            atol=1e-10,
            rtol=1e-9,
        )
        expected = np.kron(np.eye(2), expm(-0.19j * local)) @ expected
    circuit.rz(0, theta=0.17)
    expected = np.kron(expm(-0.085j * z), np.eye(2)) @ expected
    np.testing.assert_allclose(circuit.state(), expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(circuit.state(), expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(circuit.effective_circuit.state(), expected, atol=1e-7)


@pytest.mark.parametrize("evolve", [False, True])
def test_analog_mps_initial_state_gradient(jaxb, highp, evolve):
    def loss(params):
        circuit = tc.AnalogCircuit(2, mps_inputs=_analog_mps_input(params[0]))
        if evolve:
            circuit.add_analog_block(
                lambda t: tc.gates.x().tensor,
                params[1],
                index=[0],
                atol=1e-10,
                rtol=1e-9,
            )
        return tc.backend.real(circuit.expectation_ps(z=[0]))

    theta, time = 0.31, 0.23
    params = tc.backend.convert_to_tensor(np.array([theta, time]))
    expected = np.cos(2 * theta)
    gradient = np.array([-2 * np.sin(2 * theta), 0.0])
    if evolve:
        expected *= np.cos(2 * time)
        gradient *= np.cos(2 * time)
        gradient[1] = -2 * np.cos(2 * theta) * np.sin(2 * time)
    value_and_grad = tc.backend.value_and_grad(loss)
    for function in [value_and_grad, tc.backend.jit(value_and_grad)]:
        actual, actual_gradient = function(params)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
        np.testing.assert_allclose(actual_gradient, gradient, atol=1e-7, rtol=0)
