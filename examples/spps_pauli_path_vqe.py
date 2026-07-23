"""
Stochastic Pauli-path simulator (SPPS) for VQE pre-training, following
"Stochastic Pauli-path simulator for large-scale quantum optimization"
(arXiv:2607.17804).

SPPS estimates unbiased stochastic gradients of a variational objective
    f(theta) = Tr[O U(theta) rho U(theta)^dag]
by sampling Pauli-propagation paths in the Heisenberg picture, correcting each
sample by its sampling probability (importance reweighting), and differentiating
the trigonometric path coefficients analytically -- "path automatic
differentiation" (PAD), which despite the name is a closed-form score, not
autodiff. The same sampled paths give both the energy and its gradient.

This example pre-trains a 1D transverse-field Ising model
    H = -J sum_i Z_i Z_{i+1} - g sum_i X_i
with a hardware-efficient ansatz (|+> preparation, then L layers of RZ / RY
rotations and an open nearest-neighbor CNOT chain), and shows that SPPS-driven
gradient descent tracks exact (state-vector autodiff) gradient descent.

Implementation notes:

* The gradient estimator is written with ``tc.backend`` ops only, so it is
  backend-agnostic: ``tc.set_backend("jax")`` gives a JIT + vmap kernel, while
  ``tc.set_backend("numpy")`` also runs (vmap falls back to a Python loop).
* Randomness is drawn outside the kernel with ``numpy`` and passed in, so the
  kernel is a pure deterministic function of (uniforms, angles) -- no
  backend-specific RNG, trivially jittable.
* Each Pauli operator's support is packed into two integer bitmasks (bit q =
  qubit q), so every gate is an O(1) bitwise update instead of an array scatter.
  This is both backend-agnostic and fast, but a single int64 mask caps this demo
  at n <= 63 qubits. Larger systems (the paper reaches 100 qubits) need a
  multi-word mask -- an array of int64 words with the same per-gate bit ops -- or
  a plain (batch, n) bit array; the sampling/PAD logic is unchanged either way.
"""

import numpy as np
import scipy.sparse.linalg

import tensorcircuit as tc

K = tc.set_backend("jax")  # "numpy" also works (vmap becomes a Python loop)
tc.set_dtype("complex128")


def build_ops(n, nlayers):
    """Ordered ansatz ops and parameter count.

    Each op is ("H", q), ("CNOT", c, t), or ("R", "Z"|"Y", q, param_index).
    """
    ops = [("H", q) for q in range(n)]
    idx = 0
    for _ in range(nlayers):
        for q in range(n):
            ops.append(("R", "Z", q, idx))
            idx += 1
        for q in range(n):
            ops.append(("R", "Y", q, idx))
            idx += 1
        for q in range(n - 1):
            ops.append(("CNOT", q, q + 1))
    return ops, idx


def tfim_terms(n, jcoup, g):
    """TFIM as a list of (coefficient, x-bitmask, z-bitmask) Pauli terms."""
    terms = []
    for i in range(n - 1):
        terms.append((-jcoup, 0, (1 << i) | (1 << (i + 1))))  # Z_i Z_{i+1}
    for i in range(n):
        terms.append((-g, 1 << i, 0))  # X_i
    return terms


def make_spps_kernel(ops_rev, nparams, smoothing):
    """Build a jitted, backend-agnostic SPPS gradient kernel.

    ``ops_rev`` is the op list in Heisenberg (reversed) order. The returned
    function maps pre-drawn uniforms of shape (n_terms, batch, n_rot) plus the
    Pauli-term bitmasks and rotation angles to per-term energy and gradient
    estimates, averaged over the path batch.
    """
    # Flatten the op list into a static schedule the traced loop can walk.
    schedule = []
    for op in ops_rev:
        if op[0] == "H":
            schedule.append(("H", op[1]))
        elif op[0] == "CNOT":
            schedule.append(("C", op[1], op[2]))
        else:
            _, gen, q, j = op
            gx = 1 if gen == "Y" else 0  # generator support: Y=(x,z), Z=(z)
            schedule.append(("R", q, gx, 1, gx, j))
    n_rot = sum(1 for o in schedule if o[0] == "R")
    real_i = K.convert_to_tensor(np.array([1.0, 0.0, -1.0, 0.0]))  # Re(i^phase)

    def single_path(us, xi, zi, thetas):
        """One sampled Heisenberg path: returns its energy and PAD gradient."""
        phase = 0  # power of i, mod 4
        psi = 1.0  # accumulated trigonometric weight
        nonzero_psi = 1.0  # psi with zero selected factors replaced by one
        prob = 1.0  # sampling probability of this path
        score = [0.0] * nparams  # PAD score, filled at each active rotation
        zero_derivative = [0.0] * nparams
        zero_count = 0
        r = 0
        for o in schedule:
            if o[0] == "H":
                q = o[1]
                xb = (xi >> q) & 1
                zb = (zi >> q) & 1
                swap = xb ^ zb  # H swaps X<->Z support on qubit q
                xi = xi ^ (swap << q)
                zi = zi ^ (swap << q)
                phase = (phase + 2 * (xb * zb)) % 4
            elif o[0] == "C":
                c, t = o[1], o[2]
                xi = xi ^ (((xi >> c) & 1) << t)  # x[t] ^= x[c]
                zi = zi ^ (((zi >> t) & 1) << c)  # z[c] ^= z[t]
            else:
                _, q, gx, gz, gphase, j = o
                th = thetas[j]
                cos_t = K.cos(th)
                sin_t = K.sin(th)
                q_cos = (K.abs(cos_t) + smoothing) / (
                    K.abs(cos_t) + K.abs(sin_t) + 2.0 * smoothing
                )
                xb = (xi >> q) & 1
                zb = (zi >> q) & 1
                anti = ((xb * gz + zb * gx) & 1) == 1  # commute test
                cos_branch = anti & (us[r] < q_cos)
                sin_branch = anti & (us[r] >= q_cos)
                factor = K.where(cos_branch, cos_t, K.where(sin_branch, sin_t, 1.0))
                derivative = K.where(
                    cos_branch, -sin_t, K.where(sin_branch, cos_t, 0.0)
                )
                is_zero = factor == 0.0
                psi = psi * factor
                nonzero_psi = nonzero_psi * K.where(is_zero, 1.0, factor)
                zero_count = zero_count + K.cast(is_zero, "int64")
                prob = prob * K.where(
                    cos_branch, q_cos, K.where(sin_branch, 1.0 - q_cos, 1.0)
                )
                score[j] = derivative / K.where(is_zero, 1.0, factor)
                zero_derivative[j] = K.where(is_zero, derivative, 0.0)
                flip = K.cast(sin_branch, "int64")  # sin branch: P -> i G P
                xi = xi ^ (flip * (gx << q))
                zi = zi ^ (flip * (gz << q))
                phase = K.where(
                    sin_branch, (phase + gphase + 2 * (gz * xb) + 1) % 4, phase
                )
                r += 1
        trace = real_i[phase] * K.cast(xi == 0, "float64")  # <0|P|0>, nonzero if pure-Z
        h_tilde = psi * trace / prob
        score = K.stack(score)
        zero_derivative = K.stack(zero_derivative)
        # A zero selected factor has a finite direct derivative but no log score.
        gradient_weight = K.where(
            zero_count == 0,
            psi * score,
            K.where(
                zero_count == 1,
                nonzero_psi * zero_derivative,
                K.zeros_like(score),
            ),
        )
        return h_tilde, trace * gradient_weight / prob

    over_batch = K.vmap(single_path, vectorized_argnums=0)
    over_terms = K.vmap(over_batch, vectorized_argnums=(0, 1, 2))

    @K.jit
    def kernel(us_all, term_xi, term_zi, thetas):
        h, grad = over_terms(us_all, term_xi, term_zi, thetas)
        return K.mean(h, axis=1), K.mean(grad, axis=1)  # per-term averages

    return kernel, n_rot


def spps_energy_grad(kernel, term_xi, term_zi, coefs, params, uniforms):
    """Combine per-term SPPS estimates into a scalar energy and gradient."""
    f, grad = kernel(uniforms, term_xi, term_zi, K.convert_to_tensor(params))
    f = np.asarray(K.numpy(f))
    grad = np.asarray(K.numpy(grad))
    energy = float(np.sum(coefs * f))
    gradient = (coefs[:, None] * grad).sum(axis=0)
    return energy, gradient


def exact_energy_fn(n, nlayers, terms):
    """State-vector energy of the ansatz, for the exact-GD baseline."""
    paulis = []
    for coef, xi, zi in terms:
        xs = [q for q in range(n) if (xi >> q) & 1 and not (zi >> q) & 1]
        ys = [q for q in range(n) if (xi >> q) & 1 and (zi >> q) & 1]
        zs = [q for q in range(n) if (zi >> q) & 1 and not (xi >> q) & 1]
        paulis.append((coef, xs, ys, zs))

    def energy(params):
        c = tc.Circuit(n)
        for q in range(n):
            c.h(q)
        idx = 0
        for _ in range(nlayers):
            for q in range(n):
                c.rz(q, theta=params[idx])
                idx += 1
            for q in range(n):
                c.ry(q, theta=params[idx])
                idx += 1
            for q in range(n - 1):
                c.cnot(q, q + 1)
        e = 0.0
        for coef, xs, ys, zs in paulis:
            e += coef * K.real(c.expectation_ps(x=xs, y=ys, z=zs))
        return K.real(e)

    return energy


def exact_ground_energy(n, jcoup, g):
    """Sparse ground-state energy of the open-boundary TFIM."""
    structures, weights = [], []
    for i in range(n - 1):
        s = [0] * n
        s[i] = s[i + 1] = 3  # Z Z
        structures.append(s)
        weights.append(-jcoup)
    for i in range(n):
        s = [0] * n
        s[i] = 1  # X
        structures.append(s)
        weights.append(-g)
    hmat = tc.quantum.PauliStringSum2COO(structures, weights, numpy=True).tocsr()
    return float(
        scipy.sparse.linalg.eigsh(hmat, k=1, which="SA", return_eigenvectors=False)[0]
    )


if __name__ == "__main__":
    n, nlayers = 15, 5
    jcoup, g = 1.0, 1.0
    steps, lr, batch, seed = 256, 0.05, 4000, 42

    ops, nparams = build_ops(n, nlayers)
    ops_rev = list(reversed(ops))
    terms = tfim_terms(n, jcoup, g)
    coefs = np.array([c for c, _, _ in terms])
    term_xi = K.convert_to_tensor(np.array([xi for _, xi, _ in terms]))
    term_zi = K.convert_to_tensor(np.array([zi for _, _, zi in terms]))

    e0 = exact_ground_energy(n, jcoup, g)
    print(f"exact ground energy E0 = {e0:.6f}  (n={n}, L={nlayers}, g={g})")

    kernel, n_rot = make_spps_kernel(ops_rev, nparams, smoothing=0.25 / nlayers)
    exact_vag = K.jit(K.value_and_grad(exact_energy_fn(n, nlayers, terms)))

    rng = np.random.default_rng(seed)
    theta_spps = rng.uniform(-0.25 * np.pi / nlayers, 0.25 * np.pi / nlayers, nparams)
    theta_exact = theta_spps.copy()

    print(f"{'step':>5} {'exact eps0':>12} {'spps eps0':>12}")
    for step in range(steps):
        uniforms = K.convert_to_tensor(rng.random((len(terms), batch, n_rot)))
        e_spps, g_spps = spps_energy_grad(
            kernel, term_xi, term_zi, coefs, theta_spps, uniforms
        )
        theta_spps = theta_spps - lr * g_spps

        e_exact, g_exact = exact_vag(tc.array_to_tensor(theta_exact, dtype="float64"))
        theta_exact = theta_exact - lr * np.asarray(K.numpy(g_exact))

        if step % 32 == 0 or step == steps - 1:
            eps_exact = abs(float(K.numpy(e_exact)) - e0) / n
            eps_spps = abs(e_spps - e0) / n
            print(f"{step:>5} {eps_exact:>12.2e} {eps_spps:>12.2e}")
