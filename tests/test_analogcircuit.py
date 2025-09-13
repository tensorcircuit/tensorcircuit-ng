import os
import sys

import pytest
import numpy as np

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
        ac.cx(1, 0)
        return tc.backend.real(ac.expectation_ps(z=[1]))

    # Test JIT compilation
    gf = tc.backend.jit(tc.backend.grad(cost_fn))
    param = tc.backend.ones([3])
    param = tc.backend.cast(param, dtype="float32")

    eps = 1e-4 * np.array([1.0, 0, 0])
    num_grad = (cost_fn(param + eps) - cost_fn(param - eps)) / (2e-4)

    np.testing.assert_allclose(gf(param), gf(param), atol=1e-6)
    np.testing.assert_allclose(gf(param)[0], num_grad, atol=1e-3)
