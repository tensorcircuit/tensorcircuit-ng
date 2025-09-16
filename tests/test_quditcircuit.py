# pylint: disable=invalid-name

import os
import sys

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

# see https://stackoverflow.com/questions/56307329/how-can-i-parametrize-tests-to-run-with-different-fixtures-in-pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
import tensorcircuit.quantum as qu


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_basics(backend):
    c = tc.QuditCircuit(2, 3)
    c.x(0)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("10")), np.array(1.0))
    c.csum(0, 1)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("11")), np.array(1.0))
    c.csum(0, 1)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("12")), np.array(1.0))

    c = tc.QuditCircuit(2, 3)
    c.x(0)
    c.x(0)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("20")), np.array(1.0))
    c.csum(0, 1, cv=1)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("21")), np.array(0.0))
    c.csum(0, 1, cv=2)
    np.testing.assert_allclose(tc.backend.numpy(c.amplitude("21")), np.array(1.0))


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_measure(backend):
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    c.x(1)
    c.csum(0, 1)
    assert c.measure(1)[0] in [0, 1, 2]


@pytest.mark.parametrize("backend", [lf("jaxb")])
def test_large_scale_sample(backend):
    L = 30
    d = 3
    c = tc.QuditCircuit(L, d)
    c.h(0)
    for i in range(L - 1):
        c.csum(i, i + 1)

    batch = 1024
    results = c.sample(
        allow_state=False, batch=batch, format="count_dict_bin", jittable=False
    )

    k0, k1, k2 = "0" * L, "1" * L, "2" * L
    c0, c1, c2 = results.get(k0, 0), results.get(k1, 0), results.get(k2, 0)
    assert c0 + c1 + c2 == batch

    probs = np.array([c0, c1, c2], dtype=float) / batch
    np.testing.assert_allclose(probs, np.ones(3) / 3, rtol=0.2, atol=0.0)

    for a, b in [(c0, c1), (c1, c2), (c0, c2)]:
        ratio = (a + 1e-12) / (b + 1e-12)
        assert 0.8 <= ratio <= 1.25


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_expectation(backend):
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    np.testing.assert_allclose(
        tc.backend.numpy(c.expectation((tc.quditgates.z_matrix_func(3), [0]))),
        0,
        atol=1e-7,
    )


def test_complex128(highp, tfb):
    c = tc.QuditCircuit(2, 3)
    c.h(1)
    c.rx(0, theta=1j)
    c.wavefunction()
    np.testing.assert_allclose(
        c.expectation((tc.quditgates.z_matrix_func(3), [1])), 0, atol=1e-15
    )


def test_single_qubit():
    c = tc.QuditCircuit(1, 10)
    c.h(0)
    w = c.state()[0]
    np.testing.assert_allclose(
        w,
        np.array(
            [
                1,
            ]
            * 10
        )
        / np.sqrt(10),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_expectation_between_two_states_qudit(backend):
    dim = 3
    X3 = tc.quditgates.x_matrix_func(dim)
    # Y3 = tc.quditgates._y_matrix_func(dim)  # ZX/i
    Z3 = tc.quditgates.z_matrix_func(dim)
    H3 = tc.quditgates.h_matrix_func(dim)
    X3_dag = np.conjugate(X3.T)

    # e0 = np.array([1.0, 0.0, 0.0], dtype=np.complex64)
    # e1 = np.array([0.0, 1.0, 0.0], dtype=np.complex64)
    # val = tc.expectation((tc.gates.Gate(Y3), [0]), ket=e0, bra=e1, dim=dim)
    # omega = np.exp(2j * np.pi / dim)
    # expected = omega / 1j
    # np.testing.assert_allclose(tc.backend.numpy(val), expected, rtol=1e-6, atol=1e-6)

    c = tc.QuditCircuit(3, dim)
    c.unitary(0, unitary=tc.gates.Gate(H3))
    c.ry(1, theta=0.8, j=0, k=1)
    state = c.wavefunction()
    x1z2 = [(tc.gates.Gate(X3), [0]), (tc.gates.Gate(Z3), [1])]
    e1 = c.expectation(*x1z2)
    e2 = tc.expectation(*x1z2, ket=state, bra=state, normalization=True, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e2), tc.backend.numpy(e1))

    c = tc.QuditCircuit(3, dim)
    c.unitary(0, unitary=tc.gates.Gate(H3))
    c.ry(1, theta=0.8 + 0.7j, j=0, k=1)
    state = c.wavefunction()
    e1 = c.expectation(*x1z2) / (tc.backend.norm(state) ** 2)
    e2 = tc.expectation(*x1z2, ket=state, normalization=True, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e2), tc.backend.numpy(e1))

    c1 = tc.QuditCircuit(2, dim)
    c1.unitary(1, unitary=tc.gates.Gate(X3))
    s1 = c1.state()

    c2 = tc.QuditCircuit(2, dim)
    c2.unitary(0, unitary=tc.gates.Gate(X3))
    s2 = c2.state()

    c3 = tc.QuditCircuit(2, dim)
    c3.unitary(1, unitary=tc.gates.Gate(H3))
    s3 = c3.state()

    x1x2_fixed = [(tc.gates.Gate(X3), [0]), (tc.gates.Gate(X3_dag), [1])]
    e = tc.expectation(*x1x2_fixed, ket=s1, bra=s2, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e), 1.0)

    e2 = tc.expectation(*x1x2_fixed, ket=s3, bra=s2, dim=dim)
    np.testing.assert_allclose(tc.backend.numpy(e2), 1.0 / np.sqrt(3))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("cpb")])
def test_any_inputs_state_qudit_true_gates(backend):
    dim = 3
    Xd = tc.quditgates.x_matrix_func(dim)
    Zd = tc.quditgates.z_matrix_func(dim)
    omega = np.exp(2j * np.pi / dim)

    def idx(j0, j1):
        return dim * j0 + j1

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(2, 0)] = 1.0
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), 1.0 + 0j, rtol=1e-6, atol=1e-6)

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(0, 0)] = 1.0
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), omega, rtol=1e-6, atol=1e-6)

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(1, 0)] = 1.0
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), omega**2, rtol=1e-6, atol=1e-6)

    vec = np.zeros(dim * dim, dtype=np.complex64)
    vec[idx(0, 0)] = 1 / np.sqrt(2)
    vec[idx(1, 0)] = 1 / np.sqrt(2)
    c = tc.QuditCircuit(2, dim, inputs=tc.array_to_tensor(vec))
    c.unitary(0, unitary=tc.gates.Gate(Xd))
    z0 = c.expectation((tc.gates.Gate(Zd), [0]))
    np.testing.assert_allclose(tc.backend.numpy(z0), -0.5 + 0j, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("cpb")])
def test_postselection(backend):
    c = tc.QuditCircuit(3, 3)
    c.h(1)
    c.h(2)
    c.mid_measurement(1, 1)
    c.mid_measurement(2, 1)
    s = c.wavefunction()
    np.testing.assert_allclose(tc.backend.numpy(s[4]).real, 1.0 / 3.0, rtol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("cpb")])
def test_unitary(backend):
    c = tc.QuditCircuit(2, dim=3, inputs=np.eye(9))
    c.x(0)
    c.z(1)
    answer = tc.backend.numpy(
        np.kron(tc.quditgates.x_matrix_func(3), tc.quditgates.z_matrix_func(3))
    )
    np.testing.assert_allclose(
        tc.backend.numpy(c.wavefunction().reshape([9, 9])), answer, atol=1e-5
    )


def test_probability():
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    c.h(1)
    np.testing.assert_allclose(c.probability(), np.ones(9) / 9, atol=1e-5)


def test_circuit_add_demo():
    dim = 3
    c = tc.QuditCircuit(2, dim=dim)
    c.x(0)  # |00> -> |10>
    c2 = tc.QuditCircuit(2, dim=dim, mps_inputs=c.quvector())
    c2.x(0)  # |00> -> |20>
    answer = np.zeros(dim * dim, dtype=np.complex64)
    answer[dim * 2 + 0] = 1.0
    np.testing.assert_allclose(c2.wavefunction(), answer, atol=1e-5)
    c3 = tc.QuditCircuit(2, dim=dim)
    c3.x(0)
    c3.replace_mps_inputs(c.quvector())
    np.testing.assert_allclose(c3.wavefunction(), answer, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_circuit_matrix(backend):
    c = tc.QuditCircuit(2, 3)
    c.x(1)
    c.csum(0, 1)

    U = c.matrix()
    U_np = tc.backend.numpy(U)

    row_10 = U_np[3]
    expected_row_10 = np.zeros(9, dtype=row_10.dtype)
    expected_row_10[4] = 1.0
    np.testing.assert_allclose(row_10, expected_row_10, atol=1e-5)

    state = tc.backend.numpy(c.state())
    expected_state = np.zeros(9, dtype=state.dtype)
    expected_state[1] = 1.0
    np.testing.assert_allclose(state, expected_state, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_batch_sample(backend):
    c = tc.QuditCircuit(3, 3)
    c.h(0)
    c.csum(0, 1)
    print(c.sample())
    print(c.sample(batch=8, status=np.random.uniform(size=[8, 3])))
    print(c.sample(batch=8))
    print(c.sample(random_generator=tc.backend.get_random_state(42)))
    print(c.sample(allow_state=True))
    print(c.sample(batch=8, allow_state=True))
    print(
        c.sample(
            batch=8, allow_state=True, random_generator=tc.backend.get_random_state(42)
        )
    )
    print(
        c.sample(
            batch=8,
            allow_state=True,
            status=np.random.uniform(size=[8]),
            format="sample_bin",
        )
    )
    print(
        c.sample(
            batch=8,
            allow_state=False,
            status=np.random.uniform(size=[8, 3]),
            format="sample_bin",
        )
    )


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_sample_format(backend):
    c = tc.QuditCircuit(2, 3)
    c.h(0)
    c.csum(0, 1)
    key = tc.backend.get_random_state(42)
    for allow_state in [False, True]:
        print("allow_state: ", allow_state)
        for batch in [None, 1, 3]:
            print("  batch: ", batch)
            for format_ in [
                None,
                "sample_bin",
                "count_dict_bin",
            ]:
                print("    format: ", format_)
                print(
                    "      ",
                    c.sample(
                        batch=batch,
                        allow_state=allow_state,
                        format_=format_,
                        random_generator=key,
                    ),
                )


def test_sample_representation():
    _ALPHBET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    c = tc.QuditCircuit(1, 36)
    (result,) = c.sample(1, format="count_dict_bin").keys()
    assert result == _ALPHBET[0]

    for i in range(1, 35):
        c.x(0)
        (result,) = c.sample(1, format="count_dict_bin").keys()
        assert result == _ALPHBET[i]


def test_quditcircuit_set_dim_validation():
    with pytest.raises(ValueError):
        tc.QuditCircuit(1, 2)
    with pytest.raises(ValueError):
        tc.QuditCircuit(1, 2.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb"), lf("torchb")])
def test_qudit_minimal_ad_qudit(backend):
    r"""Minimal AD test on a single-qudit (d=3) circuit.
    We differentiate the expectation :math:`\langle Z\rangle` w.r.t. a single RY parameter and
    compare to a finite-difference estimate.
    """

    dim = 3

    def energy(theta):
        c = tc.QuditCircuit(1, dim)
        # rotate on the (0,1) subspace so that the observable is sensitive to theta
        c.ry(0, theta=theta, j=0, k=1)
        # measure Z on site 0 (qudit Z for d=3)
        E = c.expectation((tc.quditgates.z_matrix_func(dim), [0]))
        return tc.backend.real(E)

    # backend autodiff gradient
    grad_energy = tc.backend.grad(energy)

    theta0 = tc.num_to_tensor(0.37)
    g = grad_energy(theta0)

    # finite-difference check
    eps = 1e-3
    num = (energy(theta0 + eps) - energy(theta0 - eps)) / (2 * eps)

    np.testing.assert_allclose(
        tc.backend.numpy(g), tc.backend.numpy(num), rtol=1e-2, atol=1e-3
    )


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb"), lf("torchb")])
def test_qudit_minimal_jit_qudit(backend):
    """Minimal JIT test: jit-compiled energy matches eager energy."""

    dim = 3

    def energy(theta):
        c = tc.QuditCircuit(1, dim)
        c.ry(0, theta=theta, j=0, k=1)
        E = c.expectation((tc.quditgates.z_matrix_func(dim), [0]))
        return tc.backend.real(E)

    jit_energy = tc.backend.jit(energy)

    theta0 = tc.num_to_tensor(-0.91)
    e_eager = energy(theta0)
    e_jit = jit_energy(theta0)

    np.testing.assert_allclose(
        tc.backend.numpy(e_jit), tc.backend.numpy(e_eager), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("backend", [lf("jaxb"), lf("tfb"), lf("torchb")])
def test_qudit_minimal_vmap_qudit(backend):
    """Minimal VMAP test: vectorized energies equal per-element eager results."""

    dim = 3

    def energy(theta):
        c = tc.QuditCircuit(1, dim)
        c.ry(0, theta=theta, j=0, k=1)
        E = c.expectation((tc.quditgates.z_matrix_func(dim), [0]))
        return tc.backend.real(E)

    venergy = tc.backend.vmap(energy)

    thetas = tc.array_to_tensor(np.linspace(-1.0, 1.0, 7))
    vvals = venergy(thetas)
    eager_vals = np.array([energy(t) for t in thetas])

    np.testing.assert_allclose(
        tc.backend.numpy(vvals), eager_vals, rtol=1e-6, atol=1e-6
    )


def test_qudit_paths_and_sampling_wrappers():
    c = tc.QuditCircuit(2, 3)
    c.x(0)
    c.rzz(0, 1, theta=np.float64(0.2), j1=0, k1=1, j2=0, k2=1)
    c.cphase(0, 1, cv=1)
    qo = c.get_quoperator()
    assert qo is not None
    _ = c.sample(allow_state=False, batch=1, format="count_dict_bin")
    for bad in ["sample_int", "count_tuple", "count_dict_int", "count_vector"]:
        with pytest.raises(NotImplementedError):
            c.sample(allow_state=False, batch=1, format=bad)


def test_quditcircuit_amplitude_before_wrapper():
    c = tc.QuditCircuit(2, 3)
    c.x(0)
    nodes = c.amplitude_before("00")
    assert isinstance(nodes, list)
    assert len(nodes) == 5  # one gate (X on qudit 0) -> single node in the traced path


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_qudit_entanglement_measures_maximally_entangled(backend):
    r"""
    Prepare the two-qudit maximally entangled state
    :math:`\lvert \Phi_d \rangle = \frac{1}{\sqrt{d}} \sum_{j=0}^{d-1} \lvert j \rangle \otimes \lvert j \rangle`
    for :math:`d>2` using a generalized Hadamard :math:`H` on qudit-0 followed by
    :math:`\mathrm{CSUM}(0\!\to\!1)`. For this state,
    :math:`\rho_A = \operatorname{Tr}_B(\lvert \Phi_d \rangle\langle \Phi_d \rvert) = \mathbb{I}_d/d`.
    We check:
    - eigenvalues of :math:`\rho_A` are all :math:`1/d`;
    - purity :math:`\operatorname{Tr}(\rho_A^2) = 1/d` (via Rényi entropy);
    - linear entropy :math:`S_L = 1 - \operatorname{Tr}(\rho_A^2) = 1 - 1/d`.
    """
    d = 3
    c = tc.QuditCircuit(2, d)
    c.h(0)
    c.csum(0, 1)

    # Reduced density matrix of subsystem A (trace out qudit-1)
    rho_A = qu.reduced_density_matrix(c.state(), cut=[1], dim=d)

    # Spectrum check
    evals = tc.backend.eigh(rho_A)[0]
    np.testing.assert_allclose(evals, np.ones(d) / d, rtol=1e-6, atol=1e-7)

    # Purity from Rényi entropy of order 2
    S2 = qu.renyi_entropy(rho_A, k=2)
    purity = float(np.exp(-S2))
    np.testing.assert_allclose(purity, 1.0 / d, rtol=1e-6, atol=1e-7)

    # Linear entropy check
    linear_entropy = 1.0 - purity
    np.testing.assert_allclose(linear_entropy, 1.0 - 1.0 / d, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_qudit_mutual_information_product_vs_entangled(backend):
    r"""
    Compare quantum mutual information :math:`I(A\!:\!B) = S(\rho_A)+S(\rho_B)-S(\rho_{AB})`
    (with von Neumann entropy from built-in functions) for two two-qudit states (local dimension :math:`d>2`):

    1) **Product state** prepared *with gates* by applying single-qudit rotations
       :math:`\mathrm{RY}(\theta)` in the two-level subspace :math:`(j,k)=(0,1)` on **each** qudit.
       This yields a separable pure state, hence :math:`I(A\!:\!B)=0`.

    2) **Maximally entangled state** :math:`\lvert \Phi_d \rangle` prepared by :math:`H` on qudit-0
       then :math:`\mathrm{CSUM}(0\!\to\!1)`. For this pure bipartite state,
       :math:`\rho_A=\rho_B=\mathbb{I}_d/d`, so :math:`S(\rho_A)=S(\rho_B)=\ln d`,
       :math:`S(\rho_{AB})=0`, and :math:`I(A\!:\!B)=2\ln d`.
    """
    d = 3

    # Case 1: Product state
    c1 = tc.QuditCircuit(2, d)
    c1.ry(0, theta=0.37, j=0, k=1)
    c1.ry(1, theta=-0.59, j=0, k=1)
    I_AB_1 = qu.mutual_information(c1.state(), cut=[1], dim=d)
    np.testing.assert_allclose(I_AB_1, 0.0, atol=1e-7)

    # Case 2: Maximally entangled state
    c2 = tc.QuditCircuit(2, d)
    c2.h(0)
    c2.csum(0, 1)
    I_AB_2 = qu.mutual_information(c2.state(), cut=[1], dim=d)
    np.testing.assert_allclose(I_AB_2, 2.0 * np.log(d), rtol=1e-6, atol=1e-7)

    # Optional: confirm single-subsystem entropy equals log(d)
    rho_A = qu.reduced_density_matrix(c2.state(), cut=[1], dim=d)
    SA = qu.entropy(rho_A)
    np.testing.assert_allclose(SA, np.log(d), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_unitary_kraus_qutrit_single(backend):
    r"""
    Qutrit (d=3) deterministic unitary-kraus selection on a single site.
    Case A: prob=[0,1] -> pick X (shift), |0\rangle -> |1\rangle.
    Case B: prob=[1,0] -> pick I, state remains |0\rangle.
    """
    d = 3

    # Identity and qutrit shift X (|k\rangle -> |k+1 mod 3)\rangle
    I = tc.quditgates.i_matrix_func(d)
    X = tc.quditgates.x_matrix_func(d)

    # Case A: choose X branch deterministically
    c = tc.QuditCircuit(1, dim=d)
    idx = c.unitary_kraus([I, X], 0, prob=[0.0, 1.0])
    assert idx == 1
    np.testing.assert_allclose(c.amplitude("0"), 0.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c.amplitude("1"), 1.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c.amplitude("2"), 0.0 + 0j, atol=1e-6)

    # Case B: choose I branch deterministically
    c2 = tc.QuditCircuit(1, dim=d)
    idx2 = c2.unitary_kraus([I, X], 0, prob=[1.0, 0.0])
    assert idx2 == 0
    np.testing.assert_allclose(c2.amplitude("0"), 1.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c2.amplitude("1"), 0.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c2.amplitude("2"), 0.0 + 0j, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_general_kraus_qutrit_single(backend):
    r"""
    Qutrit (d=3) tests for general_kraus on a single site (Part B only).

    True general Kraus with normalization and `with_prob=True`:
      K0 = sqrt(p) * I, K1 = sqrt(1-p) * X
      (K0^\dagger K0 + K1^\dagger K1 = I)
      `status` controls which branch is sampled.
    """
    d = 3

    # Identity and qutrit shift X (|k> -> |k+1 mod 3)
    I = tc.quditgates.i_matrix_func(d)
    X = tc.quditgates.x_matrix_func(d)

    p = 0.7
    K0 = np.sqrt(p) * I
    K1 = np.sqrt(1.0 - p) * X

    # ---- completeness check in numpy space (works for all backends) ----
    np.testing.assert_allclose(
        tc.backend.transpose(tc.backend.conj(K0)) @ K0
        + tc.backend.transpose(tc.backend.conj(K1)) @ K1,
        I,
        atol=1e-6,
    )

    # ---- Case B1: status small -> pick K0 with prob ~ p; state remains |0\rangle ----
    c3 = tc.QuditCircuit(1, dim=d)
    idx3, prob3 = c3.general_kraus([K0, K1], 0, status=0.2, with_prob=True)
    assert idx3 == 0
    np.testing.assert_allclose(np.array(prob3), np.array([p, 1 - p]), atol=1e-6)
    np.testing.assert_allclose(np.array(prob3)[idx3], p, atol=1e-6)
    np.testing.assert_allclose(c3.amplitude("0"), 1.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c3.amplitude("1"), 0.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c3.amplitude("2"), 0.0 + 0j, atol=1e-6)

    # ---- Case B2: status large -> pick K1 with prob ~ (1-p); state becomes |1\rangle ----
    c4 = tc.QuditCircuit(1, dim=d)
    idx4, prob4 = c4.general_kraus([K0, K1], 0, status=0.95, with_prob=True)
    assert idx4 == 1
    np.testing.assert_allclose(np.array(prob4), np.array([p, 1 - p]), atol=1e-6)
    np.testing.assert_allclose(np.array(prob4)[idx4], 1.0 - p, atol=1e-6)
    np.testing.assert_allclose(c4.amplitude("0"), 0.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c4.amplitude("1"), 1.0 + 0j, atol=1e-6)
    np.testing.assert_allclose(c4.amplitude("2"), 0.0 + 0j, atol=1e-6)
