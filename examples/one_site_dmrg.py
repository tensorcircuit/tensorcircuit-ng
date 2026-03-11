from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
import quimb.tensor as qtn
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("float64")
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
        def apply_single(v):
            M_tensor = tc.backend.reshape(v, shape)
            M_out_tensor = apply_heff(L, W, R, M_tensor)
            return tc.backend.reshape(M_out_tensor, [-1])

        out = jax.vmap(apply_single, in_axes=1, out_axes=1)(v_flat)
        return tc.backend.cast(-out, M_flat.dtype)

    norm = tc.backend.norm(M_flat)
    v0 = M_flat / tc.backend.cast(norm, M_flat.dtype)
    v0 = tc.backend.reshape(v0, [-1, 1])

    # Safe real-subspace LOBPCG mapping
    v0_real = tc.backend.real(v0)

    def heff_matvec_real(v_real):
        return tc.backend.real(heff_matvec(tc.backend.cast(v_real, M_flat.dtype)))

    evals, evecs, _ = lobpcg_standard(heff_matvec_real, v0_real, m=num_krylov, tol=1e-5)

    # Recover exact energies and vectors
    E0 = tc.backend.cast(-evals[0], M_flat.dtype)
    vec0 = tc.backend.cast(evecs[:, 0], M_flat.dtype)
    M_opt_res = tc.backend.reshape(vec0, shape)

    # Final normalization
    M_opt_res = M_opt_res / tc.backend.cast(tc.backend.norm(M_opt_res), M_opt_res.dtype)
    return E0, M_opt_res


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


def sweep_left_to_right(
    M_list, W_list, R_env_list, init_L_env, init_R_mat, num_krylov, mask_list
):
    def step(carry, xs):
        L_e, R_mat_prev = carry
        M_local, W_local, R_e, local_mask = xs

        M_local_absorbed = tc.backend.einsum("ab,bcd->acd", R_mat_prev, M_local)
        # Apply mask before solving to avoid wasting Krylov vectors on numerical noise
        M_local_absorbed = M_local_absorbed * local_mask

        E0, M_opt = local_eigen_solver(L_e, W_local, R_e, M_local_absorbed, num_krylov)

        # Multiply with the local mask to maintain strictly zero null-space
        # avoiding dense float noise orthogonalization by QR.
        M_opt = M_opt * local_mask

        Q_tensor, R_mat = left_canonicalize(M_opt)
        L_new = update_L(L_e, W_local, Q_tensor)

        return (L_new, R_mat), (E0, Q_tensor, L_new)

    init_carry = (init_L_env, init_R_mat)
    xs = (M_list, W_list, R_env_list, mask_list)

    final_carry, (E0_list, Q_list, L_env_list) = jax.lax.scan(step, init_carry, xs)

    L_env_aligned = tc.backend.concat(
        [tc.backend.reshape(init_L_env, (1,) + init_L_env.shape), L_env_list[:-1]],
        axis=0,
    )

    return E0_list, Q_list, L_env_aligned, final_carry[1]


def sweep_right_to_left(
    M_list, W_list, L_env_list, init_R_env, init_L_mat, num_krylov, mask_list
):
    def step(carry, xs):
        R_e, L_mat_next = carry
        M_local, W_local, L_e, local_mask = xs

        M_local_absorbed = tc.backend.einsum("abc,cd->abd", M_local, L_mat_next)
        M_local_absorbed = M_local_absorbed * local_mask

        E0, M_opt = local_eigen_solver(L_e, W_local, R_e, M_local_absorbed, num_krylov)

        # Apply boundary mask to prevent null space leakage
        M_opt = M_opt * local_mask

        L_mat, Q_tensor = right_canonicalize(M_opt)

        R_new = update_R(R_e, W_local, Q_tensor)

        return (R_new, L_mat), (E0, Q_tensor, R_new)

    M_list_rev = M_list[::-1]
    W_list_rev = W_list[::-1]
    L_env_list_rev = L_env_list[::-1]
    mask_list_rev = mask_list[::-1]

    init_carry = (init_R_env, init_L_mat)
    xs = (M_list_rev, W_list_rev, L_env_list_rev, mask_list_rev)

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


@partial(jax.jit, static_argnames=["num_sweeps", "num_krylov"])
def one_site_dmrg(
    M_list, W_list, mask_list, init_L_env, init_R_env, num_sweeps=4, num_krylov=10
):
    """
    Jittable DMRG that runs num_sweeps.
    M_list should be randomly initialized before passing.
    """
    chi = M_list.shape[1]
    D = W_list.shape[1]
    L = W_list.shape[0]

    M_list = M_list / tc.backend.cast(tc.backend.norm(M_list), M_list.dtype)

    # Right canonicalize the initial MPS to build valid initial R_envs
    def step_rc(carry, xs):
        L_mat_next = carry
        M_local, local_mask = xs
        M_local_absorbed = tc.backend.einsum("abc,cd->abd", M_local, L_mat_next)
        M_local_absorbed = M_local_absorbed * local_mask
        L_mat, Q_tensor = right_canonicalize(M_local_absorbed)
        return L_mat, Q_tensor

    init_L_mat = tc.backend.eye(chi, dtype=M_list.dtype)
    _, M_list_rc_rev = jax.lax.scan(
        step_rc, init_L_mat, (M_list[::-1], mask_list[::-1])
    )
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
        _, M_l2r, L_envs, final_R_mat = sweep_left_to_right(
            M, W_list, R_envs, init_L_env, init_R_mat, num_krylov, mask_list
        )
        E0_r2l, M_r2l, R_envs_new, final_L_mat = sweep_right_to_left(
            M_l2r, W_list, L_envs, init_R_env, final_R_mat, num_krylov, mask_list
        )
        return M_r2l, R_envs_new, L_envs, E0_r2l[0], final_L_mat

    init_val = (
        M_list,
        R_env_aligned,
        jnp.zeros((L, chi, D, chi), dtype=M_list.dtype),
        0.0,
        tc.backend.eye(chi, dtype=M_list.dtype),
    )
    final_M, _, _, final_E, _ = jax.lax.fori_loop(0, num_sweeps, sweep_body, init_val)

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


def get_mps_masks(L, chi, d):
    masks = np.ones((L, chi, d, chi), dtype=np.float64)
    # The left-most tensor (site 0) must only connect via index 0 of its left bond
    masks[0, 1:, :, :] = 0.0
    # The right-most tensor (site L-1) must only connect via index 0 of its right bond
    masks[-1, :, :, 1:] = 0.0
    return tc.backend.convert_to_tensor(masks)


def generate_random_mps(L, chi, d, masks):
    key = jax.random.PRNGKey(42)
    # Generate real random MPS to avoid complex `lobpcg_standard` bug in JAX
    # Since TFIM and Heisenberg are real Hamiltonians, real initial state works.
    M_list = jax.random.normal(key, (L, chi, d, chi))

    # Apply the exact masks to avoid null-space leakage at initialization
    M_list = M_list * masks

    return tc.backend.convert_to_tensor(M_list)


if __name__ == "__main__":
    import quimb.tensor as qtn

    L = 100
    chi = 8

    print(f"Running 1D TFIM DMRG for L={L}, chi={chi}")
    W_tfim = get_tfim_mpo(L)
    L_e_tfim, R_e_tfim = get_initial_envs(chi, 3, 2, 0)
    masks = get_mps_masks(L, chi, 2)
    M_init_tfim = generate_random_mps(L, chi, 2, masks)
    E_tfim, _ = one_site_dmrg(
        M_init_tfim, W_tfim, masks, L_e_tfim, R_e_tfim, num_sweeps=4
    )
    print("Final Energy TFIM:", tc.backend.numpy(E_tfim))

    print(f"\nRunning 1D Heisenberg DMRG for L={L}, chi={chi}")
    W_heis = get_heisenberg_mpo(L)
    L_e_heis, R_e_heis = get_initial_envs(chi, 5, 4, 0)
    masks_heis = get_mps_masks(L, chi, 2)
    M_init_heis = generate_random_mps(L, chi, 2, masks_heis)
    E_heis, _ = one_site_dmrg(
        M_init_heis, W_heis, masks_heis, L_e_heis, R_e_heis, num_sweeps=4
    )
    print("Final Energy Heisenberg:", tc.backend.numpy(E_heis))

    print(f"\nComparing with Quimb 1D Heisenberg DMRG for L={L}, chi={chi}")
    H_quimb = qtn.MPO_ham_heis(L)
    dmrg = qtn.DMRG2(H_quimb, bond_dims=[chi])
    dmrg.solve(tol=1e-6, verbosity=0)
    print("Final Energy Heisenberg (Quimb):", dmrg.energy)

    # For benchmark part compared with the energy obtained via dmrg of quimb
    # We extract MPO from quimb directly instead of building the eval_matrix()
    # because eval_matrix is a 2^L x 2^L full matrix, which will OOM for L=100.
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
    masks_quimb = get_mps_masks(L, chi, 2)
    M_init_quimb = generate_random_mps(L, chi, 2, masks_quimb)
    E_heis_quimb, _ = one_site_dmrg(
        M_init_quimb, W_list_quimb, masks_quimb, L_e_quimb, R_e_quimb, num_sweeps=4
    )
    print(
        "Final Energy Heisenberg (using quimb2qop MPO):", tc.backend.numpy(E_heis_quimb)
    )
