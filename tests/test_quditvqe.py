# tests/test_quditvqe.py
# pylint: disable=invalid-name

import os
import sys

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))
sys.path.insert(0, modulepath)

import tensorcircuit as tc


def vqe_cost(params, d: int = 3):
    c = tc.QuditCircuit(2, dim=d)
    c.rx(0, params[0], j=0, k=1)
    c.ry(1, params[1], j=0, k=1)
    c.csum(0, 1)

    z0 = tc.backend.cast(
        tc.backend.convert_to_tensor(np.diag([1, -1, 0])),
        dtype=tc.cons.dtypestr,
    )
    z1 = tc.backend.cast(
        tc.backend.convert_to_tensor(np.diag([1, -1, 0])),
        dtype=tc.cons.dtypestr,
    )
    op = tc.backend.kron(z0, z1)

    wf = c.wavefunction()
    energy = tc.backend.real(
        tc.backend.einsum("i,ij,j->", tc.backend.adjoint(wf), op, wf)
    )
    return energy


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_autodiff(backend):
    params = tc.backend.cast(
        tc.backend.convert_to_tensor(np.array([0.1, 0.2])), dtype=tc.cons.rdtypestr
    )
    grad_fn = tc.backend.grad(lambda p: vqe_cost(p))
    g = grad_fn(params)
    assert g is not None
    assert tc.backend.shape_tuple(g) == tc.backend.shape_tuple(params)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_jit(backend):
    params = tc.backend.cast(
        tc.backend.convert_to_tensor(np.array([0.1, 0.2])), dtype=tc.cons.rdtypestr
    )
    f = lambda p: vqe_cost(p)
    f_jit = tc.backend.jit(f)
    e1 = f(params)
    e2 = f_jit(params)
    np.testing.assert_allclose(e1, e2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb"), lf("torchb")])
def test_vmap(backend):
    params_batch = tc.backend.cast(
        tc.backend.convert_to_tensor(np.array([[0.1, 0.2], [0.3, 0.4]])),
        dtype=tc.cons.rdtypestr,
    )
    f_batched = tc.backend.vmap(lambda p: vqe_cost(p))
    vals = f_batched(params_batch)
    assert tc.backend.shape_tuple(vals) == (2,)
