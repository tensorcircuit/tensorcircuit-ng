"""
Time evolution in a fixed particle-number sector with ``U1Operator.mvp``.

The XXZ chain conserves the number of up spins, so a Neel state evolves within
the half-filling sector of dimension C(n, n/2) instead of the full 2^n space.
``U1Operator`` compiles the Pauli strings once, and ``mvp`` binds their weights
to a matrix-free ``LinearOperator`` that works with ``krylov_evol`` and
``chebyshev_evol``. The weights stay differentiable.
"""

import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")
K = tc.backend

n = 16
k = n // 2

# XXZ chain: sum_i J (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1})
ps_list = []
for i in range(n - 1):
    for pauli in (1, 2, 3):
        ps = [0] * n
        ps[i] = ps[i + 1] = pauli
        ps_list.append(ps)
h_op = tc.u1circuit.U1Operator(n, k, ps_list)


def xxz_weights(params):
    j, delta = params[0], params[1]
    return K.stack([j, j, j * delta] * (n - 1))


# imbalance between even and odd sites, equal to 1 for the Neel state
z_list = [[3 if q == i else 0 for q in range(n)] for i in range(n)]
imbalance = tc.u1circuit.U1Operator(n, k, z_list).mvp(
    [(-1) ** (i + 1) / n for i in range(n)]
)


def expectation(op, psi):
    return K.real(K.sum(K.conj(psi) * op(psi)))


psi0 = tc.U1Circuit(n, k, filled=list(range(0, n, 2))).state()
params = K.convert_to_tensor([1.0, 0.5])
h = h_op.mvp(xxz_weights(params))
print(f"sector dimension {h_op.dimension} vs full dimension {2**n}")

# Krylov: one Krylov subspace serves several time points
times = [0.5, 1.0, 1.5, 2.0]
krylov_imbalance = tc.timeevol.krylov_evol(
    h,
    psi0,
    K.convert_to_tensor(times),
    subspace_dimension=60,
    callback=lambda psi: expectation(imbalance, psi),
)

# Chebyshev: needs spectral bounds, here from a short Lanczos run
emax, emin = tc.timeevol.estimate_spectral_bounds(h, n_iter=50)
pad = 0.05 * (float(emax) - float(emin))
bounds = (float(emax) + pad, float(emin) - pad)
print(f"spectral bounds ({bounds[1]:.3f}, {bounds[0]:.3f})")

for t, value in zip(times, krylov_imbalance):
    psi_t = tc.timeevol.chebyshev_evol(
        h, psi0, t, bounds, k=tc.timeevol.estimate_k(t, bounds)
    )
    print(
        f"t={t}: imbalance krylov {float(value):.6f}, "
        f"chebyshev {float(expectation(imbalance, psi_t)):.6f}"
    )

# cross-check against the full-space evolution
h_full = tc.quantum.PauliStringSum2MVP(ps_list, xxz_weights(params))
psi_full = tc.timeevol.krylov_evol(
    h_full,
    tc.quantum.u1_enlarge(psi0, n, k),
    K.convert_to_tensor([times[-1]]),
    subspace_dimension=60,
)[0]
psi_sector = tc.timeevol.krylov_evol(
    h, psi0, K.convert_to_tensor([times[-1]]), subspace_dimension=60
)[0]
overlap = K.sum(K.conj(tc.quantum.u1_project(psi_full, n, k)) * psi_sector)
print(f"fidelity with full-space evolution at t={times[-1]}: {abs(overlap) ** 2:.10f}")


# gradient of the final imbalance with respect to the Hamiltonian parameters
def final_imbalance(params):
    h = h_op.mvp(xxz_weights(params))
    psi = tc.timeevol.krylov_evol(
        h, psi0, K.convert_to_tensor([times[-1]]), subspace_dimension=60
    )[0]
    return expectation(imbalance, psi)


value, grad = K.jit(K.value_and_grad(final_imbalance))(params)
print(
    f"imbalance at t={times[-1]}: {float(value):.6f}, "
    f"d/dJ {float(grad[0]):.6f}, d/dDelta {float(grad[1]):.6f}"
)
