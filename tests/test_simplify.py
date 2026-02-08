import os
import sys

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import numpy as np
import tensornetwork as tn
from tensorcircuit import simplify


def test_infer_shape():
    a = tn.Node(np.ones([2, 3, 5]))
    b = tn.Node(np.ones([3, 5, 7]))
    a[1] ^ b[0]
    a[2] ^ b[1]
    assert simplify.infer_new_shape(a, b) == ((2, 7), (2, 3, 5), (3, 5, 7))


def test_rank_simplify():
    a = tn.Node(np.ones([2, 2]), name="a")
    b = tn.Node(np.ones([2, 2]), name="b")
    c = tn.Node(np.ones([2, 2, 2, 2]), name="c")
    d = tn.Node(np.ones([2, 2, 2, 2, 2, 2]), name="d")
    e = tn.Node(np.ones([2, 2]), name="e")

    a[1] ^ c[0]
    b[1] ^ c[1]
    c[2] ^ d[0]
    c[3] ^ d[1]
    d[4] ^ e[0]

    nodes = simplify._full_rank_simplify([a, b, c, d, e])
    assert nodes[0].shape == tuple([2 for _ in range(6)])
    assert len(nodes) == 1

    f = tn.Node(np.ones([2, 2]), name="f")
    g = tn.Node(np.ones([2, 2, 2, 2]), name="g")
    h = tn.Node(np.ones([2, 2, 2, 2]), name="h")

    f[1] ^ g[0]
    g[2] ^ h[1]

    nodes = simplify._full_rank_simplify([f, g, h])
    assert len(nodes) == 2


def test_simplify_extra():
    from tensorcircuit import simplify
    import tensorcircuit as tc

    a = tn.Node(np.ones([2, 2]), name="a")
    b = tn.Node(np.ones([2, 2]), name="b")
    a[1] ^ b[0]
    nodes = simplify._full_rank_simplify([a, b])
    assert len(nodes) == 1

    # _full_light_cone_cancel
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    c.h(0)
    # usually used in expectation where the psi and its conj can cancel
    # but we can just call it on any nodes list
    qir = c.to_qir()
    nodes = [g["gate"] for g in qir]
    simplify._full_light_cone_cancel(nodes)
