"""
One-site Time-Dependent Variational Principle (TDVP) implementation.
JAX-compatible, JIT-compilable, and GPU-accelerated.
"""

from functools import partial
import time
import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")


def apply_heff(L, W, R, M):
    T1 = tc.backend.einsum("abc,cde->abde", L, M)
    T2 = tc.backend.einsum("abde,bfdg->afeg", T1, W)
    M_out = tc.backend.einsum("afeg,hge->afh", T2, R)
    return M_out


def apply_keff(L, R, K):
    # K correctly contracts with top_L (c) and top_R (e)
    # Output is on bot_L (a) and bot_R (f)
    K_out = tc.backend.einsum("abc,ce,fbe->af", L, K, R)
    return K_out


def lanczos_expm(apply_fn, v0, dt, subspace_dim=10):
    shape = v0.shape
    v0_flat = tc.backend.reshape(v0, [-1])
    norm = tc.backend.norm(v0_flat)
    v0_flat = v0_flat / tc.backend.cast(norm, v0_flat.dtype)

    # Use jax.lax.scan for JIT-friendly Lanczos
    def lanczos_step(carry, i):
        V, alphas, betas, q = carry
        Aq = apply_fn(q)
        alpha = tc.backend.real(tc.backend.sum(tc.backend.conj(q) * Aq))

        # Gram-Schmidt re-orthogonalization against all previous vectors
        def ortho_step(q_target, j):
            v_k = V[:, j]
            proj = tc.backend.sum(tc.backend.conj(v_k) * q_target)
            # Only subtract if j < i (already processed)
            # Actually scan runs for fixed steps, we handle indices manually
            q_new = q_target - jnp.where(j < i, proj * v_k, 0.0)
            return q_new, None

        r, _ = jax.lax.scan(
            ortho_step,
            Aq - tc.backend.cast(alpha, q.dtype) * q,
            jnp.arange(subspace_dim),
        )

        beta = tc.backend.norm(r)
        new_q = r / tc.backend.cast(beta + 1e-15, r.dtype)

        # update V: V[:, i] = q
        new_V = V.at[:, i].set(q)
        new_alphas = alphas.at[i].set(alpha)
        new_betas = betas.at[i].set(beta)

        return (new_V, new_alphas, new_betas, new_q), q

    init_carry = (
        jnp.zeros((v0_flat.shape[0], subspace_dim), dtype=v0_flat.dtype),
        jnp.zeros(subspace_dim, dtype=v0_flat.dtype),
        jnp.zeros(subspace_dim, dtype=v0_flat.dtype),
        v0_flat,
    )

    (basis_mat, alphas, betas, _), _ = jax.lax.scan(
        lanczos_step, init_carry, jnp.arange(subspace_dim)
    )

    # Tridiagonal matrix
    T = tc.backend.diagflat(alphas)
    if subspace_dim > 1:
        off_diag = tc.backend.diagflat(betas[:-1], k=1)
        T = T + off_diag + tc.backend.adjoint(off_diag)

    expT = tc.backend.expm(T * dt)
    e1 = tc.backend.zeros((subspace_dim,), dtype=v0_flat.dtype).at[0].set(1.0)
    v_new_proj = tc.backend.matvec(expT, e1)
    v_new_flat = tc.backend.matvec(basis_mat, v_new_proj)

    return tc.backend.reshape(v_new_flat * tc.backend.cast(norm, v0_flat.dtype), shape)


def local_tdvp_step(L, W, R, M, dt, use_krylov=True, krylov_dim=10):
    shape = M.shape

    def apply_fn(v_flat):
        M_tensor = tc.backend.reshape(v_flat, shape)
        M_out = apply_heff(L, W, R, M_tensor)
        return tc.backend.reshape(M_out, [-1])

    if use_krylov:
        return lanczos_expm(apply_fn, M, -1j * dt, subspace_dim=krylov_dim)
    else:
        N = jnp.prod(jnp.array(shape))
        # Construction of dense matrix for small N
        eye = jnp.eye(N, dtype=M.dtype)
        H = jax.vmap(apply_fn)(eye).T
        return tc.backend.reshape(
            tc.backend.matvec(
                tc.backend.expm(-1j * H * dt), tc.backend.reshape(M, [-1])
            ),
            shape,
        )


def bond_tdvp_step(L, R, K, dt, use_krylov=True, krylov_dim=10):
    shape = K.shape

    def apply_fn(v_flat):
        K_tensor = tc.backend.reshape(v_flat, shape)
        K_out = apply_keff(L, R, K_tensor)
        return tc.backend.reshape(K_out, [-1])

    if use_krylov:
        return lanczos_expm(apply_fn, K, 1j * dt, subspace_dim=krylov_dim)
    else:
        N = jnp.prod(jnp.array(shape))
        eye = jnp.eye(N, dtype=K.dtype)
        H = jax.vmap(apply_fn)(eye).T
        return tc.backend.reshape(
            tc.backend.matvec(
                tc.backend.expm(1j * H * dt), tc.backend.reshape(K, [-1])
            ),
            shape,
        )


def left_canonicalize(M):
    chi_L, d, chi_R = M.shape
    M_mat = tc.backend.reshape(M, (chi_L * d, chi_R))
    Q, R_mat = tc.backend.qr(M_mat)
    Q_tensor = tc.backend.reshape(Q, (chi_L, d, chi_R))
    return Q_tensor, R_mat


def right_canonicalize(M):
    chi_L, d, chi_R = M.shape
    M_mat = tc.backend.reshape(M, (chi_L, d * chi_R))
    Q, R_mat = tc.backend.qr(tc.backend.adjoint(M_mat))
    Q_out = tc.backend.adjoint(Q)
    L_out = tc.backend.adjoint(R_mat)
    Q_tensor = tc.backend.reshape(Q_out, (chi_L, d, chi_R))
    return L_out, Q_tensor


def update_L(L, W, M):
    T1 = tc.backend.einsum("abc,cde->abde", L, M)
    T2 = tc.backend.einsum("abde,bfdg->afeg", T1, W)
    L_new = tc.backend.einsum("afeg,afh->hge", T2, tc.backend.conj(M))
    return L_new


def update_R(R, W, M):
    T1 = tc.backend.einsum("abc,dec->abde", R, M)
    T2 = tc.backend.einsum("abde,fgeb->adfg", T1, W)
    R_new = tc.backend.einsum("adfg,hga->hfd", T2, tc.backend.conj(M))
    return R_new


def sweep_left_to_right(
    M_list, W_list, R_env_list, init_L_env, dt, use_krylov, krylov_dim
):
    num_sites = M_list.shape[0]

    def scan_step(carry, i):
        L_curr, M_next_carry, M_list_carry = carry

        M_local = M_next_carry
        W_local = W_list[i]
        R_e = R_env_list[i]

        M_opt = local_tdvp_step(
            L_curr, W_local, R_e, M_local, dt, use_krylov, krylov_dim
        )

        def not_last():
            Q_tensor, R_mat = left_canonicalize(M_opt)
            L_new = update_L(L_curr, W_local, Q_tensor)

            R_mat_opt = bond_tdvp_step(L_new, R_e, R_mat, dt, use_krylov, krylov_dim)
            M_next_input = M_list[i + 1]  # We peek at the original to get next site
            M_next_new = tc.backend.einsum("ab,bcd->acd", R_mat_opt, M_next_input)

            return L_new, M_next_new, Q_tensor, L_curr

        def last():
            return L_curr, M_opt, M_opt, L_curr

        # Conditional logic based on index i
        L_next, M_next_val, M_out, L_env_val = jax.lax.cond(
            i < num_sites - 1, not_last, last
        )

        new_M_list = M_list_carry.at[i].set(M_out)
        return (L_next, M_next_val, new_M_list), (M_out, L_env_val)

    init_carry = (init_L_env, M_list[0], M_list)
    (_, final_M_at_last, final_M_list), (_, L_env_list) = jax.lax.scan(
        scan_step, init_carry, jnp.arange(num_sites)
    )

    # We must ensure the final M_list has the very last updated site
    final_M_list = final_M_list.at[num_sites - 1].set(final_M_at_last)

    return final_M_list, L_env_list


def sweep_right_to_left(
    M_list, W_list, L_env_list, init_R_env, dt, use_krylov, krylov_dim
):
    num_sites = M_list.shape[0]

    def scan_step(carry, i):
        # i goes from num_sites-1 down to 0
        R_curr, M_next_carry, M_list_carry = carry

        M_local = M_next_carry
        W_local = W_list[i]
        L_e = L_env_list[i]

        M_opt = local_tdvp_step(
            L_e, W_local, R_curr, M_local, dt, use_krylov, krylov_dim
        )

        def not_first():
            L_mat, Q_tensor = right_canonicalize(M_opt)
            R_new = update_R(R_curr, W_local, Q_tensor)

            L_mat_opt = bond_tdvp_step(L_e, R_new, L_mat, dt, use_krylov, krylov_dim)
            M_prev_input = M_list[i - 1]
            M_prev_new = tc.backend.einsum("abc,cd->abd", M_prev_input, L_mat_opt)

            return R_new, M_prev_new, Q_tensor, R_curr

        def first():
            return R_curr, M_opt, M_opt, R_curr

        R_next, M_next_val, M_out, R_env_val = jax.lax.cond(i > 0, not_first, first)

        new_M_list = M_list_carry.at[i].set(M_out)

        return (R_next, M_next_val, new_M_list), (M_out, R_env_val)

    init_carry = (init_R_env, M_list[num_sites - 1], M_list)
    (_, final_M_at_first, final_M_list), (_, R_env_list_rev) = jax.lax.scan(
        scan_step, init_carry, jnp.arange(num_sites - 1, -1, -1)
    )

    final_M_list = final_M_list.at[0].set(final_M_at_first)

    # R_env_list_rev is in reverse order (site L-1 to 0)
    # We want it in index order 0 to L-1
    # Actually site i in loop corresponded to R_env_list[i] = R_curr
    # The scan output order follows the jnp.arange(num_sites-1, -1, -1) output
    # So we need to reverse the R_env_list_rev to get it in 0..L-1 order
    return final_M_list, R_env_list_rev[::-1]


@partial(jax.jit, static_argnames=["num_steps", "use_krylov", "krylov_dim"])
def one_site_tdvp(
    M_list,
    W_list,
    init_L_env,
    init_R_env,
    dt,
    num_steps=10,
    use_krylov=True,
    krylov_dim=10,
):
    # Initial R_envs
    def step_rc(carry, xs):
        L_mat_next = carry
        M_local = xs
        M_absorbed = tc.backend.einsum("abc,cd->abd", M_local, L_mat_next)
        L_mat, Q_tensor = right_canonicalize(M_absorbed)
        return L_mat, Q_tensor

    L_mat_final, M_list_rc_rev = jax.lax.scan(
        step_rc, jnp.eye(M_list.shape[1], dtype=M_list.dtype), M_list[::-1]
    )
    M_list_init = M_list_rc_rev[::-1]
    # Absorb final residual into the first site (center of right-canonical MPS)
    M_list_init = M_list_init.at[0].set(
        tc.backend.einsum("ab,bcd->acd", L_mat_final, M_list_init[0])
    )

    def step_r_env(carry, xs):
        R_e = carry
        M_local, W_local = xs
        R_new = update_R(R_e, W_local, M_local)
        return R_new, R_e  # Output R_e so R_env_list[i] strictly excludes site i

    _, R_env_list_rev = jax.lax.scan(
        step_r_env, init_R_env, (M_list_init[::-1], W_list[::-1])
    )
    R_env_list_init = R_env_list_rev[::-1]

    def tdvp_step(carry, _):
        M, R_envs = carry
        M_l2r, L_envs = sweep_left_to_right(
            M, W_list, R_envs, init_L_env, dt / 2.0, use_krylov, krylov_dim
        )
        M_r2l, R_envs_new = sweep_right_to_left(
            M_l2r, W_list, L_envs, init_R_env, dt / 2.0, use_krylov, krylov_dim
        )
        return (M_r2l, R_envs_new), M_r2l

    (final_M, _), M_history = jax.lax.scan(
        tdvp_step, (M_list_init, R_env_list_init), jnp.arange(num_steps)
    )
    return final_M, M_history


def build_mpo_tensors(h_quimb, num_sites, d_mpo):
    """Convert a quimb MPO into a padded W_list tensor of shape (L, D, 2, 2, D)."""
    op = tc.quantum.quimb2qop(h_quimb)
    w_list = np.zeros((num_sites, d_mpo, 2, 2, d_mpo), dtype=np.complex128)
    for i in range(num_sites):
        t = tc.backend.numpy(op.nodes[i].tensor)
        if i == 0:
            for dr in range(t.shape[0]):
                w_list[i, d_mpo - 1, :, :, dr] = t[dr, :, :]
        elif i == num_sites - 1:
            for dl in range(t.shape[0]):
                w_list[i, dl, :, :, 0] = t[dl, :, :]
        else:
            for dl in range(t.shape[0]):
                for dr in range(t.shape[1]):
                    w_list[i, dl, :, :, dr] = t[dl, dr, :, :]
    return tc.backend.convert_to_tensor(w_list)


def build_neel_mps(num_sites, chi, noise_scale=1e-14):
    """Build a Neel state MPS |101010...> with small noise for bond growth.

    Applies OBC boundary masks: site 0 has effective shape (1, d, chi)
    and site L-1 has shape (chi, d, 1), so non-physical virtual bond
    indices are zeroed out to prevent noise leakage during canonicalization.
    """
    m_list = np.random.normal(
        0, noise_scale, (num_sites, chi, 2, chi)
    ) + 1j * np.random.normal(0, noise_scale, (num_sites, chi, 2, chi))
    for i in range(num_sites):
        if i % 2 == 0:
            m_list[i, 0, 1, 0] = 1.0
        else:
            m_list[i, 0, 0, 0] = 1.0
    # Boundary masks: zero out non-physical virtual bond dimensions
    m_list[0, 1:, :, :] = 0.0  # site 0: left bond is trivial (dim 1)
    m_list[-1, :, :, 1:] = 0.0  # site L-1: right bond is trivial (dim 1)
    return tc.backend.convert_to_tensor(m_list)


def get_initial_envs(chi, D, D_env_left, D_env_right):
    """Build boundary environment tensors."""
    L_env = np.zeros((chi, D, chi), dtype=np.complex128)
    L_env[0, D_env_left, 0] = 1.0
    R_env = np.zeros((chi, D, chi), dtype=np.complex128)
    R_env[0, D_env_right, 0] = 1.0
    return tc.backend.convert_to_tensor(L_env), tc.backend.convert_to_tensor(R_env)


def exact_diagonalization(H_quimb, tlist):
    H = tc.backend.convert_to_tensor(H_quimb.to_dense())
    L = H_quimb.L
    psi0 = np.zeros(2**L, dtype=np.complex128)
    # Neel state |101010...>
    idx = 0
    for i in range(L):
        if i % 2 == 0:
            idx += 2 ** (L - 1 - i)
    psi0[idx] = 1.0
    psi0 = tc.backend.convert_to_tensor(psi0)

    es, vs = tc.backend.eigh(H)
    utpsi0 = tc.backend.adjoint(vs) @ psi0

    results = []
    for t in tlist:
        psi_t = vs @ (tc.backend.exp(-1j * es * t) * utpsi0)
        results.append(psi_t)
    return tc.backend.stack(results)


def mps_to_state(M_list):
    L = M_list.shape[0]
    psi = M_list[0, 0, :, :]  # (2, chi)
    for i in range(1, L):
        psi = tc.backend.einsum("...a,abc->...bc", psi, M_list[i])
    # psi has shape (d, d, ..., d, chi)
    return tc.backend.reshape(psi[..., 0], [-1])


if __name__ == "__main__":
    L = 12
    chi = 16
    dt = 0.005
    num_steps = 600

    print(f"Running TDVP vs ED for L={L}, chi={chi}, dt={dt}, steps={num_steps}")

    H = qtn.SpinHam1D(S=0.5)
    H += 4.0, "Z", "Z"
    H += -2.1, "X"
    H += -1.0, "Z"
    H_quimb = H.build_mpo(L)
    D = 3
    W_list = build_mpo_tensors(H_quimb, L, D)
    L_e, R_e = get_initial_envs(chi, D, D - 1, 0)
    M_list = build_neel_mps(L, chi)

    t0 = time.time()
    final_M, M_history = one_site_tdvp(
        M_list, W_list, L_e, R_e, dt, num_steps=num_steps, krylov_dim=10
    )
    jax.block_until_ready(final_M)
    print(f"TDVP time (jit): {time.time() - t0:.3f}s")

    tlist = [dt * (i + 1) for i in range(num_steps)]
    psi_ed = exact_diagonalization(H_quimb, tlist)

    for i in range(num_steps):
        psi_tdvp = mps_to_state(M_history[i])
        fidelity = (
            tc.backend.norm(tc.backend.sum(tc.backend.conj(psi_ed[i]) * psi_tdvp)) ** 2
        )
        print(
            f"Step {i+1}, t={(i+1)*dt:.2f}, Fidelity: {tc.backend.numpy(fidelity):.8f}"
        )

    # L=100 Scalability Test
    print("\n" + "=" * 20)
    print("Running Scalability Test: L=100, chi=16, t=1.0")
    print("=" * 20)
    l_big = 100
    chi_big = 16
    dt_big = 0.01
    steps_big = 100

    h_mpo_big = H.build_mpo(l_big)  # reuse same SpinHam1D
    w_list_big = build_mpo_tensors(h_mpo_big, l_big, D)
    le_big, re_big = get_initial_envs(chi_big, D, D - 1, 0)
    m_list_big = build_neel_mps(l_big, chi_big)

    t_start = time.time()
    final_m_big, _ = one_site_tdvp(
        m_list_big,
        w_list_big,
        le_big,
        re_big,
        dt_big,
        num_steps=steps_big,
        krylov_dim=10,
    )
    jax.block_until_ready(final_m_big)
    print(f"L=100 TDVP time (jit + 1st call): {time.time() - t_start:.3f}s")

    t_start_2nd = time.time()
    final_m_big_2nd, _ = one_site_tdvp(
        m_list_big,
        w_list_big,
        le_big,
        re_big,
        dt_big,
        num_steps=steps_big,
        krylov_dim=10,
    )
    jax.block_until_ready(final_m_big_2nd)
    print(f"L=100 TDVP time (2nd call): {time.time() - t_start_2nd:.3f}s")
    print("Scalability test completed successfully.")
