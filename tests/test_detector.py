import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

import tensorcircuit as tc
from tensorcircuit import experimental


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_sampling_basic_lookback(backend):
    c = tc.Circuit(1)
    c.x(0)
    c.measure_instruction(0)  # rec[0] = 1
    c.reset_instruction(0)
    c.measure_instruction(0)  # rec[1] = 0
    c.detector_instruction([-1, -2])  # 0 xor 1 = 1

    samples = np.asarray(c.sample_detector(batch_size=64, seed=7))
    assert samples.shape == (64, 1)
    np.testing.assert_array_equal(samples[:, 0], np.ones(64, dtype=bool))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_current_m_count_freeze(backend):
    c1 = tc.Circuit(1)
    c1.x(0)
    c1.measure_instruction(0)
    c1.detector_instruction([-1])  # should bind to rec[0]
    c1.reset_instruction(0)
    c1.x(0)
    c1.measure_instruction(0)

    c2 = tc.Circuit(1)
    c2.x(0)
    c2.measure_instruction(0)
    c2.detector_instruction([0])  # absolute reference
    c2.reset_instruction(0)
    c2.x(0)
    c2.measure_instruction(0)

    s1 = np.asarray(c1.sample_detector(batch_size=32, seed=11))
    s2 = np.asarray(c2.sample_detector(batch_size=32, seed=11))
    np.testing.assert_array_equal(s1, s2)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_with_reset_instruction(backend):
    c = tc.Circuit(1)
    c.x(0)
    c.measure_instruction(0)  # 1
    c.reset_instruction(0)
    c.measure_instruction(0)  # 0
    c.detector_instruction([-1, -2])  # 1
    samples = np.asarray(c.sample_detector(batch_size=32, seed=5))
    np.testing.assert_array_equal(samples[:, 0], np.ones(32, dtype=bool))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_depolarizing_channel_statistics(backend):
    c = tc.Circuit(1)
    c.depolarizing(0, px=0.2, py=0.0, pz=0.0)
    c.measure_instruction(0)
    c.detector_instruction([-1])
    r = np.asarray(c.sample_detector(batch_size=4000, seed=123))
    p1 = np.mean(r[:, 0])
    assert abs(p1 - 0.2) < 0.05


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_depolarizing_channel_seed_reproducible(backend):
    c = tc.Circuit(1)
    c.depolarizing(0, px=0.2, py=0.0, pz=0.0)
    c.measure_instruction(0)
    c.detector_instruction([-1])
    r0 = np.asarray(c.sample_detector(batch_size=64, seed=42))
    r1 = np.asarray(c.sample_detector(batch_size=64, seed=42))
    np.testing.assert_array_equal(r0, r1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_invalid_lookback_raises(backend):
    c = tc.Circuit(1)
    c.measure_instruction(0)
    c.detector_instruction([-2])
    with pytest.raises(ValueError):
        c.sample_detector(batch_size=1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_measure_then_gate_without_reset_raises(backend):
    c = tc.Circuit(1)
    c.x(0)
    c.measure_instruction(0)
    c.x(0)
    c.detector_instruction([-1])
    with pytest.raises(NotImplementedError):
        c.sample_detector(batch_size=1)


def test_detector_invalid_trajectory_does_not_poison_jax_highp(jaxb):
    c = tc.Circuit(1)
    c.x(0)
    c.measure_instruction(0)
    c.x(0)
    c.detector_instruction([-1])
    with pytest.raises(NotImplementedError):
        c.sample_detector(batch_size=1)

    n = 4
    nlayers = 1
    with tc.runtime_dtype("complex128"):

        def state(params):
            params = tc.backend.reshape(params, [2 * nlayers, n])
            c2 = tc.Circuit(n)
            c2 = tc.templates.blocks.example_block(c2, params, nlayers=nlayers)
            return c2.state()

        params = tc.backend.cast(tc.backend.ones([2 * nlayers * n]), "float32")
        fim = experimental.qng(state)(params)
        assert tc.backend.shape_tuple(fim) == (2 * nlayers * n, 2 * nlayers * n)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_allow_state_branch(backend):
    c = tc.Circuit(2)
    c.x(0)
    c.measure_instruction(0)
    c.measure_instruction(1)
    c.detector_instruction([-2])  # q0
    c.detector_instruction([-1])  # q1
    samples = np.asarray(c.sample_detector(batch_size=16, allow_state=True, seed=9))
    assert samples.shape == (16, 2)
    np.testing.assert_array_equal(samples[:, 0], np.ones(16, dtype=bool))
    np.testing.assert_array_equal(samples[:, 1], np.zeros(16, dtype=bool))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_allow_state_matches_trajectory_on_deterministic_case(backend):
    c = tc.Circuit(1)
    c.x(0)
    c.measure_instruction(0)
    c.detector_instruction([-1])
    s0 = np.asarray(c.sample_detector(batch_size=32, allow_state=False, seed=123))
    s1 = np.asarray(c.sample_detector(batch_size=32, allow_state=True, seed=123))
    np.testing.assert_array_equal(s0, s1)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_dm_detector_allow_state(backend):
    c = tc.DMCircuit(1)
    c.x(0)
    c.measure_instruction(0)
    c.detector_instruction([-1])
    r = np.asarray(c.sample_detector(batch_size=16, allow_state=True, seed=12))
    assert r.shape == (16, 1)
    np.testing.assert_array_equal(r[:, 0], np.ones(16, dtype=bool))


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_detector_probabilities_jit(backend):
    @tc.backend.jit
    def p1(theta):
        c = tc.Circuit(1)
        c.rx(0, theta=theta)
        c.measure_instruction(0)
        c.detector_instruction([-1])
        p = c.detector_probabilities()
        return p[1]

    v = p1(tc.backend.convert_to_tensor(0.2))
    assert np.isfinite(float(np.asarray(tc.backend.numpy(v))))


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_detector_probabilities_ad(backend):
    def p1(theta):
        c = tc.Circuit(1)
        c.rx(0, theta=theta)
        c.measure_instruction(0)
        c.detector_instruction([-1])
        p = c.detector_probabilities()
        return p[1]

    vg = tc.backend.value_and_grad(p1)
    v, g = vg(tc.backend.convert_to_tensor(0.2))
    assert np.isfinite(float(np.asarray(tc.backend.numpy(v))))
    assert np.isfinite(float(np.asarray(tc.backend.numpy(g))))


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_detector_sampling_shots_batch_status_reuse(backend):
    c = tc.Circuit(1)
    c.h(0)
    c.measure_instruction(0)
    c.detector_instruction([-1])
    status = tc.backend.implicit_randu(shape=[17, 1])
    r0 = np.asarray(c.sample_detector(shots=17, batch=5, status=status))
    r1 = np.asarray(c.sample_detector(shots=17, batch=17, status=status))
    np.testing.assert_array_equal(r0, r1)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_outcome_probability_jit_complex(backend):
    # A more complex 3-qubit example with noise and JIT
    @tc.backend.jit
    def get_probs(theta):
        c = tc.Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.cx(1, 2)
        c.rx(1, theta=theta)
        c.depolarizing(1, px=0.05, py=0.05, pz=0.05)
        c.measure_instruction(0)
        c.measure_instruction(1)
        c.measure_instruction(2)
        # 3 detectors with different lookbacks
        c.detector_instruction([0, 1])  # parity of q0, q1
        c.detector_instruction([1, 2])  # parity of q1, q2
        c.detector_instruction([0, 2])  # parity of q0, q2

        full_p = c.detector_probabilities()
        p000 = c.outcome_probability([0, 0, 0])
        p111 = c.outcome_probability([1, 1, 1])
        p101 = c.outcome_probability([1, 0, 1])

        return full_p, p000, p111, p101

    theta = tc.backend.convert_to_tensor(0.6)
    full_p, p000, p111, p101 = get_probs(theta)
    full_p_np = tc.backend.numpy(full_p)

    # Verify consistency between full distribution and individual outcome probability
    assert np.isclose(full_p_np[0, 0, 0], tc.backend.numpy(p000), atol=1e-5)
    assert np.isclose(full_p_np[1, 1, 1], tc.backend.numpy(p111), atol=1e-5)
    assert np.isclose(full_p_np[1, 0, 1], tc.backend.numpy(p101), atol=1e-5)
    assert np.isclose(np.sum(full_p_np), 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_multi_qubit_various_gates(backend):
    c = tc.Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cz(1, 2)
    c.s(0)
    c.t(1)
    c.x(2)
    c.measure_instruction(0)
    c.measure_instruction(1)
    c.measure_instruction(2)
    # Detectors referencing multiple measurements
    c.detector_instruction([0, 1, 2])  # m0 ^ m1 ^ m2
    c.detector_instruction([-1, -3])  # m2 ^ m0

    samples = np.asarray(c.sample_detector(batch_size=100, seed=123))
    assert samples.shape == (100, 2)
    # m2 is always 1 because of c.x(2)
    # m0, m1 are correlated. m0^m1 is always 0 because of cx(0,1) and initial |00>
    # so m0^m1^m2 should be 1
    # m2^m0 should be 1^m0
    np.testing.assert_array_equal(samples[:, 0], np.ones(100, dtype=bool))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_reset_complex_sequence(backend):
    c = tc.Circuit(2)
    c.x(0)
    c.measure_instruction(0)  # m0 = 1
    c.reset_instruction(0)
    c.measure_instruction(0)  # m1 = 0
    c.x(1)
    c.measure_instruction(1)  # m2 = 1
    c.detector_instruction([0, 1])  # m0 ^ m1 = 1
    c.detector_instruction([1, 2])  # m1 ^ m2 = 1
    c.detector_instruction([0, 2])  # m0 ^ m2 = 0

    samples = np.asarray(c.sample_detector(batch_size=32, seed=42))
    assert samples.shape == (32, 3)
    np.testing.assert_array_equal(samples[:, 0], np.ones(32, dtype=bool))
    np.testing.assert_array_equal(samples[:, 1], np.ones(32, dtype=bool))
    np.testing.assert_array_equal(samples[:, 2], np.zeros(32, dtype=bool))


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb")])
def test_detector_sampling_prob_consistency(backend):
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    c.rx(0, theta=0.8)
    c.depolarizing(1, px=0.1, py=0.1, pz=0.1)
    c.measure_instruction(0)
    c.measure_instruction(1)
    c.detector_instruction([0, 1])

    probs = tc.backend.numpy(c.detector_probabilities())
    # probs has shape (2,) because there is only 1 detector
    p1 = probs[1]

    samples = np.asarray(c.sample_detector(batch_size=10000, seed=123))
    p1_sample = np.mean(samples[:, 0])
    assert np.isclose(p1, p1_sample, atol=0.03)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_detector_mr_reset_chain(backend):
    c = tc.Circuit(1)
    c.x(0)
    c.measure_instruction(0)  # m0=1
    c.reset_instruction(0)
    c.measure_instruction(0)  # m1=0
    c.reset_instruction(0)
    c.x(0)
    c.mr_instruction(0)  # m2=1, reset to 0
    c.measure_instruction(0)  # m3=0

    c.detector_instruction([0, 1])  # 1^0 = 1
    c.detector_instruction([1, 2])  # 0^1 = 1
    c.detector_instruction([2, 3])  # 1^0 = 1
    c.detector_instruction([0, 1, 2, 3])  # 1^0^1^0 = 0

    samples = np.asarray(c.sample_detector(batch_size=32, seed=123))
    assert samples.shape == (32, 4)
    np.testing.assert_array_equal(samples[:, 0], np.ones(32, dtype=bool))
    np.testing.assert_array_equal(samples[:, 1], np.ones(32, dtype=bool))
    np.testing.assert_array_equal(samples[:, 2], np.ones(32, dtype=bool))
    np.testing.assert_array_equal(samples[:, 3], np.zeros(32, dtype=bool))


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_detector_larger_circuit(backend):
    n = 5
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    for i in range(n - 1):
        c.cx(i, i + 1)
    for i in range(n):
        c.measure_instruction(i)

    # Parity of all qubits
    c.detector_instruction(list(range(n)))

    p = c.detector_probabilities()
    samples = np.asarray(c.sample_detector(batch_size=1000, seed=42))
    p_sample = np.mean(samples, axis=0)
    # p has shape (2,) because there is only 1 detector
    assert np.isclose(tc.backend.numpy(p)[1], p_sample[0], atol=0.1)


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_dm_detector_mr(backend):
    c = tc.DMCircuit(1)
    c.x(0)
    c.mr_instruction(0)  # m0=1, reset to 0
    c.x(0)
    c.detector_instruction([0])

    samples = np.asarray(c.sample_detector(batch_size=32, seed=123, allow_state=True))
    assert samples.shape == (32, 1)
    np.testing.assert_array_equal(samples[:, 0], np.ones(32, dtype=bool))
