import numpy as np
import tensornetwork as tn
import tensorcircuit as tc
import tensorcircuit.cons as cons
from applications.tropical_algebra import (
    MaxPlusAlgebra,
    tropical,
    recover_configuration,
)


def test_example_maxplus_tensordot_matches_brute():
    be = tc.backend
    rng = np.random.default_rng(0)
    anp, bnp = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = be.cast(be.convert_to_tensor(anp), "float64")
    b = be.cast(be.convert_to_tensor(bnp), "float64")
    got = np.array(MaxPlusAlgebra().tensordot(be, a, b, 1))
    # brute max-plus: Y[i,j] = max_k a[i,k] + b[k,j]
    ref = np.max(anp[:, :, None] + bnp[None, :, :], axis=1)
    np.testing.assert_allclose(got, ref)


def test_example_recover_configuration_on_tiny_ising():
    # 3-node chain (vector-matrix-vector) -> recover_configuration round-trips
    be = tc.backend
    n = tn.Node(be.cast(be.convert_to_tensor(np.array([0.0, 0.0])), "float64"))
    e = tn.Node(
        be.cast(be.convert_to_tensor(np.array([[1.0, -1.0], [-1.0, 1.0]])), "float64")
    )
    m = tn.Node(be.cast(be.convert_to_tensor(np.array([0.5, -0.5])), "float64"))
    tn.connect(n[0], e[0])
    tn.connect(e[1], m[0])
    nodes = [n, e, m]
    with tropical(track=True):
        float(cons.contractor(nodes, output_edge_order=[]).tensor)
        cfg = recover_configuration()
    assert isinstance(cfg, dict)
    assert len(cfg) >= 1
