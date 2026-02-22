"""
PEPS Contraction using Boundary MPS with JAX.

This module implements a high-performance, JIT-compilable contractor for
Projected Entangled Pair States (PEPS) using the Boundary Matrix Product State
(MPS) method. It specializes in contracting large-scale 2D tensor networks
(e.g., partition functions of statistical models) into a scalar or its logarithm.

Core Features:
1. Variational MPO-MPS row contraction with DMRG-like sweeps.
2. End-to-end JIT compilation and hardware acceleration via JAX.
3. Overflow-resistant log-partition function calculation for extreme system sizes.
4. Numerical stabilization using QR-based orthogonalization and norm extraction.
"""

from functools import partial
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.integrate as integrate
import tensornetwork as tn
import tensorcircuit as tc
from tensorcircuit.cons import contractor

tc.set_backend("jax")
tc.set_dtype("complex128")

# =====================================================================
# Module 1: Environment Builders
# =====================================================================


def update_L(L_prev, mps_new_node, mps_prev_node, grid_node):
    """
    Propagate the left environment to the right.
    L_prev: (n_L, g_L, p_L)      [n: new, g: grid, p: prev]
    mps_new_node (conjugated): (n_L, D_d, n_R)
    mps_prev_node: (p_L, D_u, p_R)
    grid_node: (D_u, D_d, g_L, g_R)
    Returns L_next: (n_R, g_R, p_R)
    """
    return jnp.einsum(
        "ngp, ndN, puP, udgG -> NGP",
        L_prev,
        jnp.conj(mps_new_node),
        mps_prev_node,
        grid_node,
        optimize=True,
    )


def update_R(R_next, mps_new_node, mps_prev_node, grid_node):
    """
    Propagate the right environment to the left.
    Returns R_prev: (n_L, g_L, p_L)
    """
    return jnp.einsum(
        "NGP, ndN, puP, udgG -> ngp",
        R_next,
        jnp.conj(mps_new_node),
        mps_prev_node,
        grid_node,
        optimize=True,
    )


# =====================================================================
# Module 2: The 1-Site Optimizer
# =====================================================================


def optimize_site(L_env, R_env, mps_prev_node, grid_node):
    """
    Contract local environments to compute the optimal target tensor E.
    Returns E: shape (n_L, D_d, n_R)
    """
    return jnp.einsum(
        "ngp, NGP, puP, udgG -> ndN",
        L_env,
        R_env,
        mps_prev_node,
        grid_node,
        optimize=True,
    )


# =====================================================================
# Precomputers using lax.scan
# =====================================================================


def precompute_R_envs(mps_new, mps_prev, grid_row, R_env_init):
    def scan_step(carry, inputs):
        mps_new_node, mps_prev_node, grid_node = inputs
        R_next = update_R(carry, mps_new_node, mps_prev_node, grid_node)
        return R_next, carry

    _, R_envs_tensor = jax.lax.scan(
        scan_step, R_env_init, (mps_new, mps_prev, grid_row), reverse=True
    )
    return R_envs_tensor


def precompute_L_envs(mps_new, mps_prev, grid_row, L_env_init):
    def scan_step(carry, inputs):
        mps_new_node, mps_prev_node, grid_node = inputs
        L_next = update_L(carry, mps_new_node, mps_prev_node, grid_node)
        return L_next, carry

    _, L_envs_tensor = jax.lax.scan(
        scan_step, L_env_init, (mps_new, mps_prev, grid_row)
    )
    return L_envs_tensor


# =====================================================================
# Module 3: JAX Scan Sweep Control (The Sweepers)
# =====================================================================


def sweep_right(mps_new, mps_prev, grid_row, L_env_init, R_envs):
    """
    Update MPS by sweeping from left to right.
    """
    chi_max = mps_new.shape[1]
    D_down = grid_row.shape[2]

    def scan_step(carry, inputs):
        L_env_curr, _ = carry
        R_env_curr, prev_node, grid_node, original_next_mps_node = inputs

        E = optimize_site(L_env_curr, R_env_curr, prev_node, grid_node)
        E_mat = E.reshape(chi_max * D_down, chi_max)

        Q, R_mat = jnp.linalg.qr(E_mat)
        Q_node = Q.reshape(chi_max, D_down, chi_max)

        updated_next_node = jnp.tensordot(
            R_mat, original_next_mps_node, axes=[[1], [0]]
        )
        L_env_next = update_L(L_env_curr, Q_node, prev_node, grid_node)

        return (L_env_next, updated_next_node), (Q_node, R_mat)

    dummy_next = jnp.zeros_like(mps_new[0])
    mps_new_shifted = jnp.concatenate([mps_new[1:], dummy_next[None, ...]], axis=0)

    inputs = (R_envs, mps_prev, grid_row, mps_new_shifted)
    init_carry = (L_env_init, mps_new[0])

    _, (Q_nodes_stack, R_mat_stack) = jax.lax.scan(scan_step, init_carry, inputs)

    # Right boundary handling: absorb residual R_mat back into the last node.
    last_Q = Q_nodes_stack[-1]
    last_R = R_mat_stack[-1]
    last_node = jnp.tensordot(last_Q, last_R, axes=[[2], [0]])

    return Q_nodes_stack.at[-1].set(last_node)


def sweep_left(mps_new, mps_prev, grid_row, L_envs, R_env_init):
    """
    Update MPS by sweeping from right to left.
    """
    chi_max = mps_new.shape[1]
    D_down = grid_row.shape[2]

    def scan_step(carry, inputs):
        R_env_curr, _ = carry
        L_env_curr, prev_node, grid_node, original_prev_mps_node = inputs

        E = optimize_site(L_env_curr, R_env_curr, prev_node, grid_node)

        # LQ decomposition via QR on transpose
        E_mat = E.reshape(chi_max, D_down * chi_max)
        Q, R_mat = jnp.linalg.qr(E_mat.T)
        L_mat = R_mat.T
        Q_node = (Q.T).reshape(chi_max, D_down, chi_max)

        updated_prev_node = jnp.tensordot(
            original_prev_mps_node, L_mat, axes=[[2], [0]]
        )
        R_env_next = update_R(R_env_curr, Q_node, prev_node, grid_node)

        return (R_env_next, updated_prev_node), (Q_node, L_mat)

    dummy_prev = jnp.zeros_like(mps_new[0])
    mps_new_shifted = jnp.concatenate([dummy_prev[None, ...], mps_new[:-1]], axis=0)

    inputs = (L_envs, mps_prev, grid_row, mps_new_shifted)
    init_carry = (R_env_init, mps_new[-1])

    _, (Q_nodes_stack, L_mat_stack) = jax.lax.scan(
        scan_step, init_carry, inputs, reverse=True
    )

    # Absorb left norm back to the first node
    first_Q = Q_nodes_stack[0]
    first_L = L_mat_stack[0]
    first_node = jnp.tensordot(first_L, first_Q, axes=[[1], [0]])

    return Q_nodes_stack.at[0].set(first_node)


# =====================================================================
# Module 4: The Variational Row Applier
# =====================================================================


def right_orthogonalize(mps):
    """
    Initial QR normalization from right to left to stabilize the DMRG state.
    """
    chi = mps.shape[1]
    D = mps.shape[2]

    def scan_step(carry, node):
        L_mat_prev = carry
        node_updated = jnp.tensordot(node, L_mat_prev, axes=[[2], [0]])
        mat = node_updated.reshape(chi, D * chi)
        Q, R = jnp.linalg.qr(mat.T)
        L_mat = R.T
        Q_mat = Q.T
        Q_node = Q_mat.reshape(chi, D, chi)
        return L_mat, Q_node

    init_carry = jnp.eye(chi, dtype=mps.dtype)
    final_L, Q_stack = jax.lax.scan(scan_step, init_carry, mps, reverse=True)

    first_node = jnp.tensordot(final_L, Q_stack[0], axes=[[1], [0]])

    mask = jnp.zeros(chi, dtype=mps.dtype).at[0].set(1.0)
    first_node = first_node * mask[:, None, None]

    return Q_stack.at[0].set(first_node)


@partial(jax.jit, static_argnames=["chi_max", "num_sweeps"])
def apply_grid_row_dmrg(mps_prev, grid_row, chi_max, key, num_sweeps=2):
    """
    Variational MPO-MPS application: mps_new ≈ mps_prev * grid_row using sweeps.
    """
    # 1. Initialize mps_new guess state with global noise
    noise_real = jax.random.normal(key, mps_prev.shape, dtype=mps_prev.dtype)
    noise_imag = jax.random.normal(
        jax.random.split(key)[0], mps_prev.shape, dtype=mps_prev.dtype
    )
    noise = (noise_real + 1j * noise_imag) * 1e-3

    mps_new = mps_prev + noise
    mps_new = right_orthogonalize(mps_new)

    # Initialize left and right environment boundaries for OBC (value 1 at bond 0)
    L_env_init = jnp.zeros((chi_max, grid_row.shape[2], chi_max), dtype=mps_new.dtype)
    R_env_init = jnp.zeros((chi_max, grid_row.shape[3], chi_max), dtype=mps_new.dtype)
    L_env_init = L_env_init.at[0, 0, 0].set(1.0)
    R_env_init = R_env_init.at[0, 0, 0].set(1.0)

    for _ in range(num_sweeps):
        R_envs = precompute_R_envs(mps_new, mps_prev, grid_row, R_env_init)
        mps_new = sweep_right(mps_new, mps_prev, grid_row, L_env_init, R_envs)

        L_envs = precompute_L_envs(mps_new, mps_prev, grid_row, L_env_init)
        mps_new = sweep_left(mps_new, mps_prev, grid_row, L_envs, R_env_init)

    # After sweep_left, the MPS is canonicalized to the first node.
    # We extract the complex 'weight' (scale + phase) using the maximum-amplitude element.
    # This 'Phase Anchoring' ensures military-grade stability against nodal points.
    left_node = mps_new[0]
    pivot = left_node.flatten()[jnp.argmax(jnp.abs(left_node))]
    row_norm = jnp.linalg.norm(left_node)

    # Track the complex phase of the pivot to avoid sign/phase masking
    phase = pivot / (jnp.abs(pivot) + 1e-300)
    row_complex_weight = phase * row_norm
    row_log_weight = jnp.log(row_complex_weight + 0j)

    # Normalize the MPS for the next layer (making it phase-canonical)
    mps_new = mps_new.at[0].set(left_node / (row_complex_weight + 1e-300))

    return mps_new, row_log_weight


# =====================================================================
# Module 5: End-to-End Contractors
# =====================================================================


@partial(jax.jit, static_argnames=["chi_max", "num_sweeps"])
def contract_peps_dmrg(mps_init, pe_grid, chi_max, key, num_sweeps=2):
    """
    Contract the entire PEPS grid layer-by-layer using jax.lax.scan.
    pe_grid: shape (L_y, L_x, D_u, D_d, D_l, D_r)
    mps_init: shape (L_x, chi_max, D_up, chi_max)
    """

    def scan_step(carry, grid_row):
        mps_curr, log_norm_acc, curr_key = carry

        # Split key for current step and future steps
        step_key, next_key = jax.random.split(curr_key)

        mps_new, row_log_weight = apply_grid_row_dmrg(
            mps_curr, grid_row, chi_max, step_key, num_sweeps
        )

        # Accumulate complex log-weight to prevent overflow and preserve phase
        new_log_weight_acc = log_norm_acc + row_log_weight

        return (mps_new, new_log_weight_acc, next_key), None

    init_carry = (mps_init, jnp.array(0.0, dtype=mps_init.dtype), key)
    (mps_final_stacked, total_log_norm, _), _ = jax.lax.scan(
        scan_step, init_carry, pe_grid
    )
    return mps_final_stacked, total_log_norm


@partial(jax.jit, static_argnames=["chi_max", "num_sweeps"])
def peps_partition_function(pe_grid, chi_max, key, num_sweeps=2):
    """
    Contract PEPS grid and return the scalar partition function value Z.
    pe_grid: shape (L_y, L_x, D_u, D_d, D_l, D_r)
    """
    L_x = pe_grid.shape[1]
    D = pe_grid.shape[2]

    # Infer appropriate complex dtype from input grid
    cdtype = jnp.result_type(pe_grid.dtype, jnp.complex64)

    # Initialize top boundary MPS
    init_node = jnp.zeros((chi_max, D, chi_max), dtype=cdtype)
    init_node = init_node.at[0, 0, 0].set(1.0)
    mps_init_stacked = jnp.stack([init_node] * L_x)

    # Perform end-to-end variational contraction
    mps_final_stacked, total_log_norm = contract_peps_dmrg(
        mps_init_stacked, pe_grid, chi_max, key, num_sweeps
    )

    # Final contraction of the bottom boundary
    mat_stack = mps_final_stacked[:, :, 0, :]

    def final_scan(v_curr, mat):
        v_next = v_curr @ mat
        return v_next, None

    v_init = jnp.zeros(chi_max, dtype=cdtype).at[0].set(1.0)
    v_final, _ = jax.lax.scan(final_scan, v_init, mat_stack)

    # Resulting Z = residual_scalar * exp(total_log_norm)
    return v_final[0] * jnp.exp(total_log_norm)


@partial(jax.jit, static_argnames=["chi_max", "num_sweeps"])
def peps_partition_function_log(pe_grid, chi_max, key, num_sweeps=2):
    """
    Contract PEPS grid and return the natural logarithm ln(Z) to avoid overflow.
    pe_grid: shape (L_y, L_x, D_u, D_d, D_l, D_r)
    """
    L_x = pe_grid.shape[1]
    D = pe_grid.shape[2]

    # Infer appropriate complex dtype from input grid
    cdtype = jnp.result_type(pe_grid.dtype, jnp.complex64)

    # Initialize top boundary MPS
    init_node = jnp.zeros((chi_max, D, chi_max), dtype=cdtype)
    init_node = init_node.at[0, 0, 0].set(1.0)
    mps_init_stacked = jnp.stack([init_node] * L_x)

    # Perform end-to-end variational contraction
    mps_final_stacked, total_log_norm = contract_peps_dmrg(
        mps_init_stacked, pe_grid, chi_max, key, num_sweeps
    )

    # Final contraction of the bottom boundary
    mat_stack = mps_final_stacked[:, :, 0, :]

    def final_scan(v_curr, mat):
        v_next = v_curr @ mat
        return v_next, None

    v_init = jnp.zeros(chi_max, dtype=cdtype).at[0].set(1.0)
    v_final, _ = jax.lax.scan(final_scan, v_init, mat_stack)

    # Final log-sum ln(Z). Correctly handles complex phase/sign without abs().
    return total_log_norm + jnp.log(v_final[0] + 0j)


# =====================================================================
# Verification and Benchmarking
# =====================================================================


def test_peps_contraction():
    """
    Benchmark the PEPS contractor against exact tensornetwork contraction
    for small-scale random grids.
    """
    np.random.seed(42)

    L_x = 4
    L_y = 4
    D = 4  # Physical bond dimension
    chi_max = 256  # Max MPS bond dimension

    # Build random grid tensors
    exact_grid_arrays = []
    dmrg_grid_np = np.zeros((L_y, L_x, D, D, D, D), dtype=np.complex128)

    for y in range(L_y):
        exact_row = []
        for x in range(L_x):
            D_u = 1 if y == 0 else D
            D_d = 1 if y == L_y - 1 else D
            D_l = 1 if x == 0 else D
            D_r = 1 if x == L_x - 1 else D

            node = (
                np.random.randn(D_u, D_d, D_l, D_r)
                + 1j * np.random.randn(D_u, D_d, D_l, D_r)
            ) / D
            exact_row.append(node)

            # Pad tensors to uniform size (D, D, D, D) for static JIT shapes
            dmrg_grid_np[y, x, :D_u, :D_d, :D_l, :D_r] = node

        exact_grid_arrays.append(exact_row)

    # Perform EXACT contraction using TensorNetwork Node
    tc_nodes = [
        [tn.Node(exact_grid_arrays[y][x], name=f"({x},{y})") for x in range(L_x)]
        for y in range(L_y)
    ]
    for y in range(L_y):
        for x in range(L_x):
            if x < L_x - 1:
                tc_nodes[y][x][3] ^ tc_nodes[y][x + 1][2]
            if y < L_y - 1:
                tc_nodes[y][x][1] ^ tc_nodes[y + 1][x][0]

    flat_nodes = [node for row in tc_nodes for node in row]
    dangling = tn.get_all_dangling(flat_nodes)
    exact_res = contractor(flat_nodes, output_edge_order=dangling).tensor.item()
    print(f"Exact Network Contraction Result: {exact_res}")

    print("Transferring grid to GPU Device (Single H2D transfer)...")
    dmrg_grid_tensor = jax.device_put(dmrg_grid_np)

    # Perform DMRG-PEPS contraction
    key = jax.random.PRNGKey(42)
    dmrg_res = peps_partition_function(
        dmrg_grid_tensor, chi_max=chi_max, key=key, num_sweeps=4
    )

    print(f"Boundary MPS DMRG Contraction Result: {dmrg_res}")
    print(f"Difference: {abs(exact_res - dmrg_res):.2e}")


# =====================================================================
# Applications: 2D Ising Model
# =====================================================================


def get_ising_local_tensor(beta, J, h, D_u, D_d, D_l, D_r):
    """
    Generate local PEPS tensors for the 2D Ising model.
    """
    # Interaction matrix W
    W = np.array(
        [[np.exp(beta * J), np.exp(-beta * J)], [np.exp(-beta * J), np.exp(beta * J)]]
    )

    # Eigen-decomposition W = V * D * V^T -> Q = V * sqrt(D)
    lam, V = np.linalg.eigh(W)
    Q = V @ np.diag(np.sqrt(lam))

    T = np.zeros((D_u, D_d, D_l, D_r), dtype=np.float64)

    # Loop over center site spin states s=0 (spin +1) and s=1 (spin -1)
    spins = [1.0, -1.0]
    for s, spin in enumerate(spins):
        weight = np.exp(beta * h * spin)

        # Apply Q matrices for internal bonds (D=2) or identity/scalar for OBC (D=1)
        vec_u = Q[s, :] if D_u == 2 else np.array([1.0])
        vec_d = Q[s, :] if D_d == 2 else np.array([1.0])
        vec_l = Q[s, :] if D_l == 2 else np.array([1.0])
        vec_r = Q[s, :] if D_r == 2 else np.array([1.0])

        # Outer product to assemble 4D tensor slice
        term = weight * np.einsum("u,d,l,r->udlr", vec_u, vec_d, vec_l, vec_r)
        T += term

    return T


def onsager_exact_free_energy_per_site(beta, J):
    """
    Compute exactly the free energy per site for infinite 2D Ising model (Onsager's solution).
    """

    K = beta * J
    kappa = 2 * np.sinh(2 * K) / (np.cosh(2 * K) ** 2)

    def integrand(theta):
        return np.log((1 + np.sqrt(1 - kappa**2 * np.sin(theta) ** 2)) / 2.0)

    integral, _ = integrate.quad(integrand, 0, np.pi / 2)

    f = -(1 / beta) * (np.log(2 * np.cosh(2 * K)) + (1 / np.pi) * integral)
    return f


def test_ising_peps_contraction():
    """
    Simulate a large 2D Ising model and compare with Onsager's analytical limit.
    """
    np.random.seed(42)

    L_x = 200
    L_y = 200
    D = 2  # Physical bond dimension for Ising
    chi_max = 16  # Max MPS bond dimension
    num_sweeps = 4

    # Critical temp beta ≈ 0.4406868
    beta = 0.4406
    J = 1.0
    h = 0.0

    print(f"Constructing Ising PEPS Grid ({L_x} x {L_y}) at Tc beta={beta:.4f}...")

    dmrg_grid_np = np.zeros((L_y, L_x, D, D, D, D))

    for y in range(L_y):
        for x in range(L_x):
            D_u = 1 if y == 0 else D
            D_d = 1 if y == L_y - 1 else D
            D_l = 1 if x == 0 else D
            D_r = 1 if x == L_x - 1 else D

            node = get_ising_local_tensor(beta, J, h, D_u, D_d, D_l, D_r)
            dmrg_grid_np[y, x, :D_u, :D_d, :D_l, :D_r] = node

    dmrg_grid_tensor = jax.device_put(dmrg_grid_np)

    print(
        f"Start End-to-End Variational JIT Sweeping (chi_max={chi_max}, sweeps={num_sweeps})..."
    )

    # Calculate log-partition function
    key = jax.random.PRNGKey(42)
    log_Z_dmrg = peps_partition_function_log(
        dmrg_grid_tensor, chi_max=chi_max, key=key, num_sweeps=num_sweeps
    )

    # Average free energy per site
    f_dmrg = -(1 / beta) * log_Z_dmrg / (L_x * L_y)

    # Onsager exact solution
    f_exact = onsager_exact_free_energy_per_site(beta, J)

    print("\n[Result Validations]")
    print(f"DMRG Free Energy per site : {f_dmrg:.8f} (with L={L_x}, chi={chi_max})")
    print(f"Onsager Infinite Limit    : {f_exact:.8f} (OBC Exact Limit L->inf)")
    print(f"Absolute Difference       : {abs(f_dmrg - f_exact):.2e}")


if __name__ == "__main__":
    print("=== Running Random PEPS Contraction Benchmark ===")
    test_peps_contraction()
    print("\n=== Running Ising PEPS Phase Transition Benchmark ===")
    test_ising_peps_contraction()
