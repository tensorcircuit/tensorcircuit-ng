import numpy as np
from applications.bcomplex32_algebra import (
    _pair_tensordot,
    _pair_einsum,
    _complex_to_pair,
    _pair_to_complex,
)
import tensorcircuit.backends.numpy_backend as nb

be = nb.NumpyBackend()


def test_complex_to_pair_roundtrip_within_bf16_quant():
    t = np.array(
        [[1.0 + 2.0j, 3.0 - 1.0j], [0.25 + 0.5j, -2.0 + 7.0j]], dtype=np.complex64
    )
    pair = _complex_to_pair(be, t)
    back = _pair_to_complex(be, pair)
    np.testing.assert_allclose(np.asarray(back), t, rtol=2e-2)


def test_pair_tensordot_matches_complex_tensordot():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3)).astype(np.complex64)
    b = rng.standard_normal((3, 4)).astype(np.complex64)
    pa = _complex_to_pair(be, a)
    pb = _complex_to_pair(be, b)
    out = _pair_tensordot(be, pa, pb, axes=([1], [0]))
    ref = np.tensordot(a, b, axes=([1], [0]))
    np.testing.assert_allclose(
        np.asarray(_pair_to_complex(be, out)), ref, rtol=2e-2, atol=5e-3
    )


def test_pair_einsum_single_operand():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 2)).astype(np.complex64)
    pa = _complex_to_pair(be, a)
    out = _pair_einsum(
        be, "ab->a", pa
    )  # reduce over b (sum in complex -> here identity semantics)
    ref = np.einsum("ab->a", a)
    np.testing.assert_allclose(
        np.asarray(_pair_to_complex(be, out)), ref, rtol=2e-2, atol=5e-3
    )


# --- Task 12: end-to-end through real tc.Circuit + wall-avoidance canary ---

import tensorcircuit as tc
from applications.bcomplex32_algebra import bcomplex32


def test_bf16_end_to_end_matches_complex64():
    import numpy as np

    def build():
        c = tc.Circuit(3)
        c.H(0)
        c.cnot(0, 1)
        c.cnot(1, 2)
        return np.asarray(c.state())

    ref = build()  # default complex64
    with bcomplex32():
        got = build()  # bf16 pair path
    np.testing.assert_allclose(got, ref, rtol=2e-2)


def test_bf16_wall_avoidance_canary():
    import numpy as np

    with bcomplex32():
        c = tc.Circuit(4)
        for i in range(4):
            c.H(i)
        for i in range(3):
            c.cnot(i, i + 1)
        st = np.asarray(c.state())
    assert st.shape == (16,)  # ran cleanly, no axis==edge crash
    import tensorcircuit.cons as cons
    from tensorcircuit.contraction_algebra import StandardAlgebra

    assert isinstance(
        cons.get_contraction_algebra(), StandardAlgebra
    )  # CM restored algebra, no leak
    c2 = tc.Circuit(2)
    c2.H(0)
    c2.cnot(0, 1)  # subsequent native contraction
    assert np.asarray(c2.state()).shape == (4,)  # algebra restored, no leak


def test_pair_einsum_keeps_bfloat16_dtype():
    """T1: _pair_einsum must compute in bf16, not upcast to float32.

    numpy's np.einsum rejects bf16; the old _pair_einsum worked around it by
    upcasting to float32 (so its output pair was float32, not bf16). The rewrite
    uses manual tensordot decomposition, which stays bf16. This test locks that.
    """
    import ml_dtypes

    bf = ml_dtypes.bfloat16
    a = np.array([[1.0 + 2.0j, 3.0j], [-1.0j, 2.0 - 1.0j]], dtype=np.complex64)
    b = np.array([[0.5 + 0.5j, 1.0j], [2.0j, -1.0 + 1.0j]], dtype=np.complex64)
    pa = _complex_to_pair(be, a)  # bf16 pair, shape (2, 2, 2)
    pb = _complex_to_pair(be, b)
    out = np.asarray(_pair_einsum(be, "ij,jk->ik", pa, pb))
    assert out.dtype == bf, f"_pair_einsum upcast to {out.dtype}; expected bfloat16"


def test_bf16_ghz8_runs_and_matches_native():
    """T3: the 8-qubit GHZ that previously crashed (cotengra autoray transpose on
    a pair result) now runs under bcomplex32 and matches native within bf16
    tolerance. get_contractor_kwargs returns prefer_einsum=True, which avoids the transpose path; the genuine-bf16
    kernel keeps intermediates bf16 end-to-end.
    """

    def ghz(n):
        c = tc.Circuit(n)
        c.H(0)
        for i in range(n - 1):
            c.cnot(i, i + 1)
        return np.asarray(c.state())

    ref = ghz(8)
    with bcomplex32():
        got = ghz(8)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=5e-2), f"max abs diff = {np.abs(got - ref).max()}"
