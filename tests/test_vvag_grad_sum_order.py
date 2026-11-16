# pylint: disable=invalid-name
import sys
import os

import pytest

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc

torch = pytest.importorskip("torch")

NQ = 4
NLAYER = 2
NPARAM = 6
BATCH = 8


def _theta_batch():
    return torch.arange(48, dtype=torch.float32).reshape(BATCH, NPARAM) * 0.1


def _w_shared():
    return torch.arange(NPARAM, dtype=torch.float32) * 0.2


def _energy(theta, w, nlayers=NLAYER, shift=0.0):
    """Real energy of the base n=4 two-layer RY+RX circuit.

    ``theta`` carries the RY angles and ``w`` the RX angles (plus an optional
    ``shift``); both are split into ``[NLAYER, 3]`` so that each layer rotates
    qubits 0-2 twice, with a CNOT entangler chain closing each layer. Only
    ever called on per-sample parameters (inside ``vmap``), never on batched
    ones.
    """
    t = tc.backend.reshape(theta, [NLAYER, 3])[:nlayers]
    wp = tc.backend.reshape(w, [NLAYER, 3])[:nlayers]
    c = tc.Circuit(NQ)
    for i in range(NQ):
        c.H(i)
    for j in range(nlayers):
        for i in range(3):
            c.ry(i, theta=t[j, i])
            c.rx(i, theta=wp[j, i] + shift)
        for i in range(NQ - 1):
            c.cnot(i, i + 1)
    e = 0.0
    for i in range(NQ):
        e += c.expectation((tc.gates.x(), [i]))
    for i in range(NQ - 1):
        e += c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [(i + 1) % NQ]))
    return tc.backend.real(e)


def _reference_vvag(f, theta, w, vectorized_w=False, **kws):
    """Per-sample reference: plain autograd loop, no torch.func/vmap.

    g_theta rows are d f(theta[i], w) / d theta[i]; g_w is summed over the
    batch when w is shared and stacked when w is vectorized (vvag contract).
    """
    values, g_theta_rows, g_w_rows = [], [], []
    for i in range(theta.shape[0]):
        ti = theta[i].detach().requires_grad_(True)
        wi = (w[i] if vectorized_w else w).detach().requires_grad_(True)
        v = f(ti, wi, **kws)
        values.append(v.detach())
        gv, gw = torch.autograd.grad(v, (ti, wi))
        g_theta_rows.append(gv)
        g_w_rows.append(gw)
    g_theta = torch.stack(g_theta_rows)
    g_w = torch.stack(g_w_rows) if vectorized_w else torch.stack(g_w_rows).sum(0)
    return torch.stack(values), g_theta, g_w


def _assert_allclose(a, b):
    assert a.shape == b.shape, f"{tuple(a.shape)} vs {tuple(b.shape)}"
    assert torch.allclose(
        a, b, rtol=1e-6, atol=1e-6
    ), f"max abs diff {float(torch.max(torch.abs(a - b)))}"


def test_vvag_argnums_int(torchb):
    theta = _theta_batch()
    w = _w_shared()
    v, g = tc.backend.vvag(_energy, argnums=0)(theta, w)
    ref_v, ref_g, _ = _reference_vvag(_energy, theta, w)
    _assert_allclose(v, ref_v)
    _assert_allclose(g, ref_g)
    assert tuple(v.shape) == (BATCH,)
    assert v.dtype == ref_v.dtype


def test_vvag_argnums_seq(torchb):
    theta = _theta_batch()
    w = _w_shared()
    v, (g_theta,) = tc.backend.vvag(_energy, argnums=(0,))(theta, w)
    ref_v, ref_g, _ = _reference_vvag(_energy, theta, w)
    _assert_allclose(v, ref_v)
    _assert_allclose(g_theta, ref_g)


def test_vvag_mixed_argnums(torchb):
    theta = _theta_batch()
    w = _w_shared()
    v, (g_theta, g_w) = tc.backend.vvag(_energy, argnums=(0, 1))(theta, w)
    ref_v, ref_g, ref_gw = _reference_vvag(_energy, theta, w)
    _assert_allclose(v, ref_v)
    _assert_allclose(g_theta, ref_g)
    _assert_allclose(g_w, ref_gw)


def test_vvag_multi_vectorized(torchb):
    theta = _theta_batch()
    w = _theta_batch() * 0.5
    v, (g_theta, g_w) = tc.backend.vvag(
        _energy, argnums=(0, 1), vectorized_argnums=(0, 1)
    )(theta, w)
    ref_v, ref_g, ref_gw = _reference_vvag(_energy, theta, w, vectorized_w=True)
    _assert_allclose(v, ref_v)
    _assert_allclose(g_theta, ref_g)
    _assert_allclose(g_w, ref_gw)


def test_vvag_kwargs(torchb):
    def shifted_energy(theta, w, shift=0.0):
        return _energy(theta, w, shift=shift)

    theta = _theta_batch()
    w = _w_shared()
    v_shift, g_shift = tc.backend.vvag(shifted_energy, argnums=0)(theta, w, shift=0.5)
    ref_v, ref_g, _ = _reference_vvag(shifted_energy, theta, w, shift=0.5)
    _assert_allclose(v_shift, ref_v)
    _assert_allclose(g_shift, ref_g)
    v_plain, _ = tc.backend.vvag(shifted_energy, argnums=0)(theta, w)
    assert not torch.allclose(v_shift, v_plain)


def test_vvag_has_aux_structured(torchb):
    def aux_energy(theta, w):
        e = _energy(theta, w)
        aux = (
            tc.backend.real(tc.backend.sum(theta * w)),
            {"s": tc.backend.sum(theta), "z": tc.backend.real(theta[0] * w[0])},
        )
        return e, aux

    theta = _theta_batch()
    w = _w_shared()
    (vs, aux), gs = tc.backend.vvag(aux_energy, argnums=0, has_aux=True)(theta, w)
    ref_v, ref_g, _ = _reference_vvag(lambda ti, wi: aux_energy(ti, wi)[0], theta, w)
    _assert_allclose(vs, ref_v)
    _assert_allclose(gs, ref_g)
    _assert_allclose(aux[0], torch.sum(theta * w, dim=1))
    _assert_allclose(aux[1]["s"], torch.sum(theta, dim=1))
    _assert_allclose(aux[1]["z"], theta[:, 0] * w[0])
    assert isinstance(aux[1], dict)
    assert tuple(aux[0].shape) == (BATCH,)


def test_vvag_non_tensor_arg(torchb):
    def layered_energy(theta, w, nlayers):
        return _energy(theta, w, nlayers=nlayers)

    theta = _theta_batch()
    w = _w_shared()
    v_two, _ = tc.backend.vvag(layered_energy, argnums=0)(theta, w, NLAYER)
    ref_two, _, _ = _reference_vvag(layered_energy, theta, w, nlayers=NLAYER)
    _assert_allclose(v_two, ref_two)
    v_one, _ = tc.backend.vvag(layered_energy, argnums=0)(theta, w, 1)
    ref_one, _, _ = _reference_vvag(layered_energy, theta, w, nlayers=1)
    _assert_allclose(v_one, ref_one)
    assert not torch.allclose(v_one, v_two)


def test_vvag_truncation_no_crash(torchb):
    n, k, layers, batch = 6, 2, 1, 4
    dim = 2**n
    mat_dim = 2 ** (n // 2)
    n_params = layers * 2 * n
    signs = torch.tensor([1.0] * (dim // 2) + [-1.0] * (dim // 2))

    def truncation_energy(th):
        u, s, vh = torch.linalg.svd(_truncation_state_matrix(th, n, layers, mat_dim))
        psi = torch.reshape((u[:, :k] * s[:k]) @ vh[:k, :], [dim])
        return torch.real(torch.sum(signs * psi.abs() ** 2))

    used_seed = None
    for seed in range(100):
        theta = (
            torch.randn(batch, n_params, generator=torch.Generator().manual_seed(seed))
            * 0.5
        )
        s0 = torch.linalg.svdvals(
            _truncation_state_matrix(theta[0], n, layers, mat_dim)
        )
        if float(s0[k - 1]) > 10 * float(s0[k]):
            used_seed = seed
            break
    assert used_seed is not None, "no seed in [0, 100) passes the spectral self-check"

    values, grads = tc.backend.vvag(truncation_energy, argnums=0, vectorized_argnums=0)(
        theta
    )
    assert tuple(values.shape) == (batch,)
    assert tuple(grads.shape) == (batch, n_params)
    assert _all_finite(values)
    assert _all_finite(grads)


def _truncation_state_matrix(th, n, layers, mat_dim):
    c = tc.Circuit(n)
    for j in range(layers):
        base = j * 2 * n
        for i in range(n):
            c.ry(i, theta=th[base + i])
        for i in range(n):
            c.rxx(i, (i + 1) % n, theta=th[base + n + i])
    return torch.reshape(c.state(), [mat_dim, mat_dim])


def _all_finite(t):
    if torch.is_complex(t):
        return bool(torch.all(torch.isfinite(torch.view_as_real(t))))
    return bool(torch.all(torch.isfinite(t)))
