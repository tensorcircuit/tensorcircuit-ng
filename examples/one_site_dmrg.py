import tensorcircuit as tc
import jax
import jax.numpy as jnp
import numpy as np

tc.set_backend("jax")
tc.set_dtype("complex128")
jax.config.update("jax_enable_x64", True)


def apply_heff(L, W, R, M):
    T1 = tc.backend.einsum("abc,cde->abde", L, M)
    T2 = tc.backend.einsum("abde,bfdg->afeg", T1, W)
    M_out = tc.backend.einsum("afeg,hge->afh", T2, R)
    return M_out


def local_eigen_solver(L, W, R, M_init, num_krylov=10):
    shape = M_init.shape
    M_flat = tc.backend.reshape(M_init, [-1])

    def heff_matvec(v_flat):
        M_tensor = tc.backend.reshape(v_flat, shape)
        M_out_tensor = apply_heff(L, W, R, M_tensor)
        return tc.backend.reshape(M_out_tensor, [-1])

    norm = tc.backend.norm(M_flat)
    v = M_flat / tc.backend.cast(norm, M_flat.dtype)

    V_list, alphas_list, betas_list = [], [], []
    v_prev = tc.backend.zeros_like(v)

    for i in range(num_krylov):
        V_list.append(v)
        w = heff_matvec(v)
        alpha = tc.backend.real(tc.backend.einsum("i,i->", tc.backend.conj(v), w))
        alphas_list.append(alpha)

        w = w - tc.backend.cast(alpha, w.dtype) * v
        if i > 0:
            w = w - tc.backend.cast(betas_list[-1], w.dtype) * v_prev

        beta = tc.backend.norm(w)
        eps = tc.backend.cast(tc.backend.ones_like(beta) * 1e-12, beta.dtype)
        beta_safe = tc.backend.where(beta < 1e-12, eps, beta)
        betas_list.append(tc.backend.real(beta_safe))

        v_next = w / tc.backend.cast(beta_safe, w.dtype)
        v_prev = v
        v = v_next

    alphas = tc.backend.stack(alphas_list)
    betas = tc.backend.stack(betas_list)
    V_mat = tc.backend.stack(V_list)

    H_krylov = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
    eigenvalues, eigenvectors = tc.backend.eigh(H_krylov)

    E0 = eigenvalues[0]
    vec0 = eigenvectors[:, 0]

    M_opt_flat = tc.backend.einsum("k,kd->d", tc.backend.cast(vec0, V_mat.dtype), V_mat)
    M_opt = tc.backend.reshape(M_opt_flat, shape)
    M_opt = M_opt / tc.backend.cast(tc.backend.norm(M_opt), M_opt.dtype)
    return E0, M_opt


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


def left_canonicalize(M):
    chi_L, d, chi_R = M.shape
    M_mat = tc.backend.reshape(M, (chi_L * d, chi_R))
    Q, R_mat = tc.backend.qr(M_mat)
    Q_tensor = tc.backend.reshape(Q, (chi_L, d, chi_R))
    return Q_tensor, R_mat


def right_canonicalize(M):
    chi_L, d, chi_R = M.shape
    M_mat = tc.backend.reshape(M, (chi_L, d * chi_R))
    Q, R_mat = tc.backend.qr(tc.backend.transpose(M_mat))
    Q_out = tc.backend.transpose(Q)
    L_out = tc.backend.transpose(R_mat)
    Q_tensor = tc.backend.reshape(Q_out, (chi_L, d, chi_R))
    return L_out, Q_tensor


def sweep_left_to_right(M_list, W_list, R_env_list, init_L_env, num_krylov):
    def step(carry, xs):
        L_e, R_mat_prev = carry
        M_local, W_local, R_e = xs

        M_local_absorbed = tc.backend.einsum("ab,bcd->acd", R_mat_prev, M_local)
        E0, M_opt = local_eigen_solver(L_e, W_local, R_e, M_local_absorbed, num_krylov)

        Q_tensor, R_mat = left_canonicalize(M_opt)
        L_new = update_L(L_e, W_local, Q_tensor)

        return (L_new, R_mat), (E0, Q_tensor, L_new)

    chi = M_list.shape[1]
    init_R_mat = tc.backend.eye(chi, dtype=M_list.dtype)
    init_carry = (init_L_env, init_R_mat)
    xs = (M_list, W_list, R_env_list)

    final_carry, (E0_list, Q_list, L_env_list) = jax.lax.scan(step, init_carry, xs)

    L_env_aligned = tc.backend.concat(
        [tc.backend.reshape(init_L_env, (1,) + init_L_env.shape), L_env_list[:-1]],
        axis=0,
    )

    return E0_list, Q_list, L_env_aligned, final_carry[1]


def sweep_right_to_left(M_list, W_list, L_env_list, init_R_env, num_krylov):
    def step(carry, xs):
        R_e, L_mat_next = carry
        M_local, W_local, L_e = xs

        M_local_absorbed = tc.backend.einsum("abc,cd->abd", M_local, L_mat_next)
        E0, M_opt = local_eigen_solver(L_e, W_local, R_e, M_local_absorbed, num_krylov)

        L_mat, Q_tensor = right_canonicalize(M_opt)
        R_new = update_R(R_e, W_local, Q_tensor)

        return (R_new, L_mat), (E0, Q_tensor, R_new)

    M_list_rev = M_list[::-1]
    W_list_rev = W_list[::-1]
    L_env_list_rev = L_env_list[::-1]

    chi = M_list.shape[1]
    init_L_mat = tc.backend.eye(chi, dtype=M_list.dtype)
    init_carry = (init_R_env, init_L_mat)
    xs = (M_list_rev, W_list_rev, L_env_list_rev)

    final_carry, (E0_list_rev, Q_list_rev, R_env_list_rev) = jax.lax.scan(
        step, init_carry, xs
    )

    E0_list = E0_list_rev[::-1]
    Q_list = Q_list_rev[::-1]
    R_env_list = R_env_list_rev[::-1]

    R_env_aligned = tc.backend.concat(
        [R_env_list[1:], tc.backend.reshape(init_R_env, (1,) + init_R_env.shape)],
        axis=0,
    )

    return E0_list, Q_list, R_env_aligned, final_carry[1]


# Make `chi`, `num_sweeps` and `num_krylov` static using `static_argnums`.
# We'll put them at the end. Or we can just use partial.
from functools import partial


@partial(jax.jit, static_argnames=["num_sweeps", "num_krylov"])
def one_site_dmrg(M_list, W_list, init_L_env, init_R_env, num_sweeps=4, num_krylov=10):
    """
    Jittable DMRG that runs num_sweeps.
    M_list should be randomly initialized before passing.
    """
    chi = M_list.shape[1]
    D = W_list.shape[1]
    L = W_list.shape[0]

    M_list = M_list / tc.backend.cast(tc.backend.norm(M_list), M_list.dtype)

    # Right canonicalize the initial MPS to build valid initial R_envs
    def step_rc(carry, M_local):
        L_mat_next = carry
        M_local_absorbed = tc.backend.einsum("abc,cd->abd", M_local, L_mat_next)
        L_mat, Q_tensor = right_canonicalize(M_local_absorbed)
        return L_mat, Q_tensor

    init_L_mat = tc.backend.eye(chi, dtype=M_list.dtype)
    _, M_list_rc_rev = jax.lax.scan(step_rc, init_L_mat, M_list[::-1])
    M_list = M_list_rc_rev[::-1]

    # Build initial R_envs
    def step_r_env(carry, xs):
        R_e = carry
        M_local, W_local = xs
        R_new = update_R(R_e, W_local, M_local)
        return R_new, R_new

    _, R_env_list_rev = jax.lax.scan(
        step_r_env, init_R_env, (M_list[::-1], W_list[::-1])
    )
    R_env_list = R_env_list_rev[::-1]
    R_env_aligned = tc.backend.concat(
        [R_env_list[1:], tc.backend.reshape(init_R_env, (1,) + init_R_env.shape)],
        axis=0,
    )

    def sweep_body(i, val):
        M, R_envs, _, _ = val
        E0_l2r, M_l2r, L_envs, _ = sweep_left_to_right(
            M, W_list, R_envs, init_L_env, num_krylov
        )
        E0_r2l, M_r2l, R_envs_new, _ = sweep_right_to_left(
            M_l2r, W_list, L_envs, init_R_env, num_krylov
        )
        return M_r2l, R_envs_new, L_envs, E0_r2l[0]

    init_val = (
        M_list,
        R_env_aligned,
        jnp.zeros((L, chi, D, chi), dtype=M_list.dtype),
        0.0,
    )
    final_M, final_R_envs, final_L_envs, final_E = jax.lax.fori_loop(
        0, num_sweeps, sweep_body, init_val
    )

    return final_E, final_M


def get_tfim_mpo(L, J=1.0, h=1.0, g=0.0):
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    W = np.zeros((3, 2, 2, 3), dtype=np.complex128)
    W[0, :, :, 0] = I
    W[1, :, :, 0] = -J * Z
    W[2, :, :, 0] = -h * X - g * Z
    W[2, :, :, 1] = Z
    W[2, :, :, 2] = I

    W_list = np.tile(W[np.newaxis, :, :, :, :], (L, 1, 1, 1, 1))
    W_first = np.zeros_like(W)
    W_first[2, :, :, :] = W[2, :, :, :]
    W_last = np.zeros_like(W)
    W_last[:, :, :, 0] = W[:, :, :, 0]

    W_list[0] = W_first
    W_list[-1] = W_last
    return tc.backend.convert_to_tensor(W_list)


def get_heisenberg_mpo(L, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0):
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    W = np.zeros((5, 2, 2, 5), dtype=np.complex128)
    W[0, :, :, 0] = I
    W[1, :, :, 0] = Jx * X
    W[2, :, :, 0] = Jy * Y
    W[3, :, :, 0] = Jz * Z
    W[4, :, :, 0] = h * Z
    W[4, :, :, 1] = X
    W[4, :, :, 2] = Y
    W[4, :, :, 3] = Z
    W[4, :, :, 4] = I

    W_list = np.tile(W[np.newaxis, :, :, :, :], (L, 1, 1, 1, 1))
    W_first = np.zeros_like(W)
    W_first[4, :, :, :] = W[4, :, :, :]
    W_last = np.zeros_like(W)
    W_last[:, :, :, 0] = W[:, :, :, 0]

    W_list[0] = W_first
    W_list[-1] = W_last
    return tc.backend.convert_to_tensor(W_list)


def get_initial_envs(chi, D, D_env_left, D_env_right):
    L_env = np.zeros((chi, D, chi), dtype=np.complex128)
    L_env[0, D_env_left, 0] = 1.0

    R_env = np.zeros((chi, D, chi), dtype=np.complex128)
    R_env[0, D_env_right, 0] = 1.0

    return tc.backend.convert_to_tensor(L_env), tc.backend.convert_to_tensor(R_env)


def generate_random_mps(L, chi, d):
    key = jax.random.PRNGKey(42)
    M_list = jax.random.normal(key, (L, chi, d, chi)) + 1j * jax.random.normal(
        jax.random.PRNGKey(43), (L, chi, d, chi)
    )
    return tc.backend.convert_to_tensor(M_list)


if __name__ == "__main__":
    L = 100
    chi = 8

    print(f"Running 1D TFIM DMRG for L={L}, chi={chi}")
    W_tfim = get_tfim_mpo(L)
    L_e_tfim, R_e_tfim = get_initial_envs(chi, 3, 2, 0)
    M_init_tfim = generate_random_mps(L, chi, 2)
    E_tfim, _ = one_site_dmrg(M_init_tfim, W_tfim, L_e_tfim, R_e_tfim, num_sweeps=4)
    print("Final Energy TFIM:", tc.backend.numpy(E_tfim))

    print(f"Running 1D Heisenberg DMRG for L={L}, chi={chi}")
    W_heis = get_heisenberg_mpo(L)
    L_e_heis, R_e_heis = get_initial_envs(chi, 5, 4, 0)
    M_init_heis = generate_random_mps(L, chi, 2)
    E_heis, _ = one_site_dmrg(M_init_heis, W_heis, L_e_heis, R_e_heis, num_sweeps=4)
    print("Final Energy Heisenberg:", tc.backend.numpy(E_heis))
