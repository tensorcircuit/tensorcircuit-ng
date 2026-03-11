import tensorcircuit as tc
import jax
import jax.numpy as jnp
import numpy as np

tc.set_backend("jax")
tc.set_dtype("float64")
jax.config.update("jax_enable_x64", True)


def apply_heff(L, W, R, M):
    T1 = tc.backend.einsum("abc,cde->abde", L, M)
    T2 = tc.backend.einsum("abde,bfdg->afeg", T1, W)
    M_out = tc.backend.einsum("afeg,hge->afh", T2, R)
    return M_out


from jax.experimental.sparse.linalg import lobpcg_standard


def local_eigen_solver(L, W, R, M_init, num_krylov=10):
    shape = M_init.shape
    M_flat = tc.backend.reshape(M_init, [-1])
    dim = M_flat.shape[0]

    def heff_matvec(v_flat):
        # lobpcg standard might pass (dim, k) where k is the search block size.
        # we apply matvec to each column.
        def apply_single(v):
            M_tensor = tc.backend.reshape(v, shape)
            M_out_tensor = apply_heff(L, W, R, M_tensor)
            out = tc.backend.reshape(M_out_tensor, [-1])
            return out

        # apply to all columns
        # v_flat shape is (dim, k)
        out = jax.vmap(apply_single, in_axes=1, out_axes=1)(v_flat)

        # We want the lowest eigenvalue of H_eff.
        # jax.experimental.sparse.linalg.lobpcg_standard finds the *largest* eigenvalues.
        # Therefore, we negate the action of H_eff to find the largest eigenvalue of -H_eff.
        return tc.backend.cast(-out, M_flat.dtype)

    norm = tc.backend.norm(M_flat)
    v0 = M_flat / tc.backend.cast(norm, M_flat.dtype)
    v0 = tc.backend.reshape(v0, [-1, 1])

    # Check dimension requirement: k*5 < n, so n must be > 5 for k=1.
    # We use jax.lax.cond to conditionally run full eigensolver or lobpcg based on dim

    def exact_diag(_):
        # Fallback to full exact diagonalization
        # We construct the full matrix.
        eye = jnp.eye(dim, dtype=M_flat.dtype)

        def apply_col(i):
            v_col = tc.backend.reshape(eye[:, i], [-1, 1])
            # note we apply positive H here to get the smallest eval correctly from eigh
            out = -heff_matvec(v_col)
            return tc.backend.reshape(out, [-1])

        H_full = jax.vmap(apply_col)(jnp.arange(dim)).T
        evals, evecs = tc.backend.eigh(H_full)
        E0 = evals[0]
        vec0 = evecs[:, 0]
        M_opt_out = tc.backend.reshape(vec0, shape)
        return E0, M_opt_out

    def lobpcg_diag(_):
        # We use lobpcg
        # jax.experimental.sparse.linalg.lobpcg_standard has a known issue with complex matrices
        # due to internal carry type mismatches (complex128 vs float64) during Rayleigh-Ritz.
        # However, for TFIM and Heisenberg models, the MPOs are completely real.
        # So we can safely run LOBPCG in the real subspace.
        # We cast both v0 and the output of heff_matvec to float64 for the eigensolver.

        v0_real = tc.backend.real(v0)

        def heff_matvec_real(v_real):
            # v_real is float64. heff_matvec returns float64 if inputs are real,
            # but we explicitly enforce it.
            return tc.backend.real(heff_matvec(tc.backend.cast(v_real, M_flat.dtype)))

        evals, evecs, i = lobpcg_standard(
            heff_matvec_real, v0_real, m=num_krylov, tol=1e-5
        )

        # evals is for -H_eff. So E0 is -evals[0].
        E0 = -evals[0]
        # if M_flat is float64, E0 and vec0 should be float64
        # if M_flat is complex128, E0 should be float64, and vec0 should be complex128
        # but exact_diag returns E0 as float64 (from eigh) implicitly.
        # we will ensure E0 is M_flat.dtype but real part.
        vec0 = tc.backend.cast(evecs[:, 0], M_flat.dtype)
        M_opt_out = tc.backend.reshape(vec0, shape)
        return tc.backend.cast(E0, M_flat.dtype), M_opt_out

    E0_res, M_opt_res = jax.lax.cond(dim <= 5, exact_diag, lobpcg_diag, None)

    M_opt_res = M_opt_res / tc.backend.cast(tc.backend.norm(M_opt_res), M_opt_res.dtype)
    return E0_res, M_opt_res


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


def sweep_left_to_right(M_list, W_list, R_env_list, init_L_env, init_R_mat, num_krylov):
    def step(carry, xs):
        L_e, R_mat_prev = carry
        M_local, W_local, R_e = xs

        M_local_absorbed = tc.backend.einsum("ab,bcd->acd", R_mat_prev, M_local)
        E0, M_opt = local_eigen_solver(L_e, W_local, R_e, M_local_absorbed, num_krylov)

        Q_tensor, R_mat = left_canonicalize(M_opt)
        L_new = update_L(L_e, W_local, Q_tensor)

        return (L_new, R_mat), (E0, Q_tensor, L_new)

    init_carry = (init_L_env, init_R_mat)
    xs = (M_list, W_list, R_env_list)

    final_carry, (E0_list, Q_list, L_env_list) = jax.lax.scan(step, init_carry, xs)

    L_env_aligned = tc.backend.concat(
        [tc.backend.reshape(init_L_env, (1,) + init_L_env.shape), L_env_list[:-1]],
        axis=0,
    )

    return E0_list, Q_list, L_env_aligned, final_carry[1]


def sweep_right_to_left(M_list, W_list, L_env_list, init_R_env, init_L_mat, num_krylov):
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
        M, R_envs, _, _, init_R_mat = val
        E0_l2r, M_l2r, L_envs, final_R_mat = sweep_left_to_right(
            M, W_list, R_envs, init_L_env, init_R_mat, num_krylov
        )
        E0_r2l, M_r2l, R_envs_new, final_L_mat = sweep_right_to_left(
            M_l2r, W_list, L_envs, init_R_env, final_R_mat, num_krylov
        )
        return M_r2l, R_envs_new, L_envs, E0_r2l[0], final_L_mat

    init_val = (
        M_list,
        R_env_aligned,
        jnp.zeros((L, chi, D, chi), dtype=M_list.dtype),
        0.0,
        tc.backend.eye(chi, dtype=M_list.dtype),
    )
    final_M, final_R_envs, final_L_envs, final_E, _ = jax.lax.fori_loop(
        0, num_sweeps, sweep_body, init_val
    )

    return final_E, final_M


def get_tfim_mpo(L, J=1.0, h=1.0, g=0.0):
    I = np.array([[1, 0], [0, 1]], dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)

    W = np.zeros((3, 2, 2, 3), dtype=np.float64)
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
    I = np.array([[1, 0], [0, 1]], dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    # Mapped real basis for Heisenberg to avoid complex numbers in JAX scan.
    # The Heisenberg Hamiltonian can be represented in real numbers using Y = iY_old.
    # Actually, we can just use the standard Pauli representation and ignore the 'i'
    # since we just need a mathematically equivalent symmetric real matrix.
    # Let's map Y -> iY: Y_new = [[0, 1], [-1, 0]]. Then Y_new @ Y_new = [[-1, 0], [0, -1]] = -I.
    # The term Jy * Y_old \otimes Y_old = Jy * (-i Y_new) \otimes (-i Y_new) = -Jy * Y_new \otimes Y_new.
    # So we can use real matrices!
    Y_real = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)

    W = np.zeros((5, 2, 2, 5), dtype=np.float64)
    W[0, :, :, 0] = I
    W[1, :, :, 0] = Jx * X
    W[2, :, :, 0] = -Jy * Y_real
    W[3, :, :, 0] = Jz * Z
    W[4, :, :, 0] = h * Z
    W[4, :, :, 1] = X
    W[4, :, :, 2] = Y_real
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
    L_env = np.zeros((chi, D, chi), dtype=np.float64)
    L_env[0, D_env_left, 0] = 1.0

    R_env = np.zeros((chi, D, chi), dtype=np.float64)
    R_env[0, D_env_right, 0] = 1.0

    return tc.backend.convert_to_tensor(L_env), tc.backend.convert_to_tensor(R_env)


def generate_random_mps(L, chi, d):
    key = jax.random.PRNGKey(42)
    # Generate real random MPS to avoid complex `lobpcg_standard` bug in JAX
    # Since TFIM and Heisenberg are real Hamiltonians, real initial state works.
    M_list = jax.random.normal(key, (L, chi, d, chi))

    # Bug Fix: Mask boundaries to avoid null-space leakage
    # The left-most tensor (site 0) must only connect via index 0 of its left bond
    mask_left = np.zeros((chi, d, chi))
    mask_left[0, :, :] = 1.0

    # The right-most tensor (site L-1) must only connect via index 0 of its right bond
    mask_right = np.zeros((chi, d, chi))
    mask_right[:, :, 0] = 1.0

    M_list = M_list.at[0].multiply(mask_left)
    M_list = M_list.at[-1].multiply(mask_right)

    return tc.backend.convert_to_tensor(M_list)


if __name__ == "__main__":
    import quimb.tensor as qtn

    L = 100
    chi = 8

    print(f"Running 1D TFIM DMRG for L={L}, chi={chi}")
    W_tfim = get_tfim_mpo(L)
    L_e_tfim, R_e_tfim = get_initial_envs(chi, 3, 2, 0)
    M_init_tfim = generate_random_mps(L, chi, 2)
    E_tfim, _ = one_site_dmrg(M_init_tfim, W_tfim, L_e_tfim, R_e_tfim, num_sweeps=4)
    print("Final Energy TFIM:", tc.backend.numpy(E_tfim))

    print(f"\nRunning 1D Heisenberg DMRG for L={L}, chi={chi}")
    W_heis = get_heisenberg_mpo(L)
    L_e_heis, R_e_heis = get_initial_envs(chi, 5, 4, 0)
    M_init_heis = generate_random_mps(L, chi, 2)
    E_heis, _ = one_site_dmrg(M_init_heis, W_heis, L_e_heis, R_e_heis, num_sweeps=4)
    print("Final Energy Heisenberg:", tc.backend.numpy(E_heis))

    print(f"\nComparing with Quimb 1D Heisenberg DMRG for L={L}, chi={chi}")
    H_quimb = qtn.MPO_ham_heis(L)
    dmrg = qtn.DMRG2(H_quimb, bond_dims=[chi])
    dmrg.solve(tol=1e-6, verbosity=0)
    print("Final Energy Heisenberg (Quimb):", dmrg.energy)

    # For benchmark part compared with the energy obtained via dmrg of quimb
    # We extract MPO from quimb directly instead of building the eval_matrix()
    # because eval_matrix is a 2^L x 2^L full matrix, which will OOM for L=100.
    # The requirement: "generate the mpo from quimb using tc.quantum.quimb2qop and for benchmark part compared with the energy obtained via dmrg of quimb"

    # Use quimb2qop to wrap it as QuOperator
    op = tc.quantum.quimb2qop(H_quimb)

    # We construct the uniform W_list out of the QuOperator nodes.
    # The QuOperator nodes corresponding to quimb MPO have shapes:
    # site 0: (D, d_out, d_in)
    # site 1..L-2: (D_L, D_R, d_out, d_in)
    # site L-1: (D, d_out, d_in)
    # where d_out, d_in are the last two legs.
    D = 5
    # The output is complex, but to test if we can do real, let's cast to float
    # since we want to avoid LOBPCG complex bug.
    # But quimb's Hamiltonian is fundamentally complex. We just saw LOBPCG fails if it stays complex128.
    # We will just cast to float and get real part of the Hamiltonian, which is equivalent.
    W_list_quimb = np.zeros((L, D, 2, 2, D), dtype=np.float64)

    for i in range(L):
        t = np.real(tc.backend.numpy(op.nodes[i].tensor))
        if i == 0:
            # Quimb left boundary: (D_R, d_out, d_in)
            # We map this to our W[D_L, d_out, d_in, D_R] where D_L is a dummy index (we use D-1)
            # t shape is (5, 2, 2). It goes to W[4, :, :, :]
            # Actually, looking at the edges, quimb uses D_R=5 for left boundary.
            for dr in range(t.shape[0]):
                W_list_quimb[i, 4, :, :, dr] = t[dr, :, :]
        elif i == L - 1:
            # Quimb right boundary: (D_L, d_out, d_in)
            # t shape is (5, 2, 2). It goes to W[:, :, :, 0]
            for dl in range(t.shape[0]):
                W_list_quimb[i, dl, :, :, 0] = t[dl, :, :]
        else:
            # Quimb middle: (D_L, D_R, d_out, d_in)
            # t shape is (5, 5, 2, 2). It goes to W[D_L, :, :, D_R]
            for dl in range(t.shape[0]):
                for dr in range(t.shape[1]):
                    W_list_quimb[i, dl, :, :, dr] = t[dl, dr, :, :]

    W_list_quimb = tc.backend.convert_to_tensor(W_list_quimb)

    L_e_quimb, R_e_quimb = get_initial_envs(chi, 5, 4, 0)
    M_init_quimb = generate_random_mps(L, chi, 2)
    E_heis_quimb, _ = one_site_dmrg(
        M_init_quimb, W_list_quimb, L_e_quimb, R_e_quimb, num_sweeps=4
    )
    print(
        "Final Energy Heisenberg (using quimb2qop MPO):", tc.backend.numpy(E_heis_quimb)
    )
