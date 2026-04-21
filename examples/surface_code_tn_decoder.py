"""
Surface Code Tensor Network Decoder
This script implements a rigorous maximum likelihood decoder for the rotated surface code
using a unified tensor network, 4D qubit states (I, X, Y, Z), and the Twist/IWHT method.
It correctly handles depolarizing noise and outputs the most likely logical sector.
"""

from functools import partial
import tensornetwork as tn
import jax
import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")
# Use cotengra for efficient contraction path finding
tc.set_contractor("cotengra")


def generate_surface_code(d):
    """
    Generate the standard rotated surface code topology of distance d (d must be odd).
    Returns: data_qubits_count, z_checks, x_checks
    """
    assert d % 2 != 0, "Distance d must be odd (e.g., 3, 5, 7)"
    data_qubits_count = d * d
    z_checks = []
    x_checks = []

    for r in range(-1, d):
        for c in range(-1, d):
            qubits = []
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                qr, qc = r + dr, c + dc
                if 0 <= qr < d and 0 <= qc < d:
                    qubits.append(qr * d + qc)

            if len(qubits) <= 1:
                continue

            is_z_check = (r + c) % 2 == 0
            if is_z_check:
                if c != -1 and c != d - 1:
                    z_checks.append(qubits)
            else:
                if r != -1 and r != d - 1:
                    x_checks.append(qubits)

    return data_qubits_count, z_checks, x_checks


def get_logical_operators(d):
    """
    Generate logical X and Z operators for the rotated surface code.
    Logical X: A path of X operators connecting the X-boundaries (Left to Right).
    Logical Z: A path of Z operators connecting the Z-boundaries (Top to Bottom).
    """
    lx = [j for j in range(d)]
    lz = [i * d for i in range(d)]
    return lx, lz


def compute_partition_function(
    p, z_syndrome, x_syndrome, z_checks, x_checks, lx, lz, num_qubits, twist_x, twist_z
):
    """
    Compute the partition function Z under specific twist conditions.
    twist_x, twist_z are 1.0 to activate phase twist, 0.0 to disable.
    """
    nodes = []

    # 1. Depolarizing noise probabilities (I, X, Y, Z)
    p_I = 1 - p
    p_X = p / 3
    p_Y = p / 3
    p_Z = p / 3
    base_prob = jnp.array([p_I, p_X, p_Y, p_Z], dtype=tc.dtypestr)

    # Twist phases:
    # phase_z_twist: detects operators that anticommute with Z (i.e., X and Y)
    # phase_x_twist: detects operators that anticommute with X (i.e., Z and Y)
    phase_z_twist = jnp.array([1.0, -1.0, -1.0, 1.0], dtype=tc.dtypestr)
    phase_x_twist = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=tc.dtypestr)

    mask_lx = jnp.array([1 if q in lx else 0 for q in range(num_qubits)])
    mask_lz = jnp.array([1 if q in lz else 0 for q in range(num_qubits)])

    # 2. Feature projection matrices (Fourier basis)
    # H_Z (4x2): Z-check detects X and Y errors
    H_Z_mat = jnp.array(
        [[1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, 1.0]], dtype=tc.dtypestr
    )

    # H_X (4x2): X-check detects Z and Y errors
    H_X_mat = jnp.array(
        [[1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [1.0, -1.0]], dtype=tc.dtypestr
    )

    # 3. Build Nodes
    qubit_nodes = []
    for q in range(num_qubits):
        # Apply local twists:
        # lx (logical X path) detects logical Z errors (Z-twist)
        # lz (logical Z path) detects logical X errors (X-twist)
        q_prob = base_prob
        q_prob = q_prob * jnp.where(twist_x * mask_lz[q] == 1, phase_z_twist, 1.0)
        q_prob = q_prob * jnp.where(twist_z * mask_lx[q] == 1, phase_x_twist, 1.0)

        involved_z = [i for i, stab in enumerate(z_checks) if q in stab]
        involved_x = [i for i, stab in enumerate(x_checks) if q in stab]
        degree = len(involved_z) + len(involved_x) + 1

        cn = tn.CopyNode(degree, 4, name=f"Q{q}")
        qubit_nodes.append(cn)
        en = tn.Node(q_prob, name=f"E{q}")
        nodes.append(en)
        en[0] ^ cn[0]

    z_stab_nodes = []
    for i, stab in enumerate(z_checks):
        sn = tn.CopyNode(len(stab) + 1, 2, name=f"SZ{i}")
        z_stab_nodes.append(sn)
        factor = jnp.stack([1.0, (-1.0) ** z_syndrome[i]]).astype(tc.dtypestr)
        syn_node = tn.Node(factor, name=f"SynZ{i}")
        nodes.append(syn_node)
        syn_node[0] ^ sn[0]

    x_stab_nodes = []
    for i, stab in enumerate(x_checks):
        sn = tn.CopyNode(len(stab) + 1, 2, name=f"SX{i}")
        x_stab_nodes.append(sn)
        factor = jnp.stack([1.0, (-1.0) ** x_syndrome[i]]).astype(tc.dtypestr)
        syn_node = tn.Node(factor, name=f"SynX{i}")
        nodes.append(syn_node)
        syn_node[0] ^ sn[0]

    # 4. Connections
    q_leg_counters = [1] * num_qubits
    sz_leg_counters = [1] * len(z_checks)
    sx_leg_counters = [1] * len(x_checks)

    for i, stab in enumerate(z_checks):
        sn = z_stab_nodes[i]
        for q_idx in stab:
            cn = qubit_nodes[q_idx]
            proj = tn.Node(H_Z_mat, name=f"HZ_s{i}_q{q_idx}")
            nodes.append(proj)
            cn[q_leg_counters[q_idx]] ^ proj[0]
            proj[1] ^ sn[sz_leg_counters[i]]
            q_leg_counters[q_idx] += 1
            sz_leg_counters[i] += 1

    for i, stab in enumerate(x_checks):
        sn = x_stab_nodes[i]
        for q_idx in stab:
            cn = qubit_nodes[q_idx]
            proj = tn.Node(H_X_mat, name=f"HX_s{i}_q{q_idx}")
            nodes.append(proj)
            cn[q_leg_counters[q_idx]] ^ proj[0]
            proj[1] ^ sn[sx_leg_counters[i]]
            q_leg_counters[q_idx] += 1
            sx_leg_counters[i] += 1

    all_nodes = nodes + qubit_nodes + z_stab_nodes + x_stab_nodes
    result_node = tc.contractor(all_nodes, output_edge_order=[])
    return jnp.real(result_node.tensor)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def decode_jit(p, z_syndrome, x_syndrome, z_checks, x_checks, lx, lz, num_qubits):
    """
    Extract logical sector probabilities via 4 twist contractions and IWHT.
    """
    Z_00 = compute_partition_function(
        p, z_syndrome, x_syndrome, z_checks, x_checks, lx, lz, num_qubits, 0.0, 0.0
    )
    Z_10 = compute_partition_function(
        p, z_syndrome, x_syndrome, z_checks, x_checks, lx, lz, num_qubits, 1.0, 0.0
    )
    Z_01 = compute_partition_function(
        p, z_syndrome, x_syndrome, z_checks, x_checks, lx, lz, num_qubits, 0.0, 1.0
    )
    Z_11 = compute_partition_function(
        p, z_syndrome, x_syndrome, z_checks, x_checks, lx, lz, num_qubits, 1.0, 1.0
    )

    # Inverse Walsh-Hadamard Transform
    P_I = (Z_00 + Z_10 + Z_01 + Z_11) / 4.0
    P_X = (Z_00 - Z_10 + Z_01 - Z_11) / 4.0
    P_Z = (Z_00 + Z_10 - Z_01 - Z_11) / 4.0
    P_Y = (Z_00 - Z_10 - Z_01 + Z_11) / 4.0

    probs = jnp.array([P_I, P_X, P_Z, P_Y])
    best_sector = jnp.argmax(probs)

    # Map back to [decoded_lx, decoded_lz]
    d_lx = jnp.where((best_sector == 1) | (best_sector == 3), 1, 0)
    d_lz = jnp.where((best_sector == 2) | (best_sector == 3), 1, 0)

    return d_lx, d_lz


def main():
    d, p = 5, 0.05
    print(f"--- Surface Code TN Decoder (Unified 4D Implementation, d={d}) ---")

    num_qubits, z_checks, x_checks = generate_surface_code(d)
    lx, lz = get_logical_operators(d)

    z_checks_tuple = tuple(tuple(c) for c in z_checks)
    x_checks_tuple = tuple(tuple(c) for c in x_checks)
    lx_tuple, lz_tuple = tuple(lx), tuple(lz)

    # 1. Simulate Depolarizing Noise
    key = jax.random.PRNGKey(42)
    rands = jax.random.uniform(key, (num_qubits,))
    errors = jnp.zeros(num_qubits, dtype=jnp.int32)
    # Mapping: 1 -> X, 2 -> Y, 3 -> Z
    errors = jnp.where(rands < p, 1, errors)
    errors = jnp.where(rands < 2 * p / 3, 2, errors)
    errors = jnp.where(rands < p / 3, 3, errors)

    has_x_error = ((errors == 1) | (errors == 2)).astype(jnp.int32)
    has_z_error = ((errors == 3) | (errors == 2)).astype(jnp.int32)

    # 2. Compute syndromes
    z_syndrome = jnp.array(
        [jnp.sum(has_x_error[jnp.array(stab)]) % 2 for stab in z_checks]
    )
    x_syndrome = jnp.array(
        [jnp.sum(has_z_error[jnp.array(stab)]) % 2 for stab in x_checks]
    )

    actual_lx = jnp.sum(has_x_error[jnp.array(lz)]) % 2
    actual_lz = jnp.sum(has_z_error[jnp.array(lx)]) % 2

    print(f"\nActual logical X parity: {actual_lx}")
    print(f"Actual logical Z parity: {actual_lz}")

    print("\nDecoding...")
    decoded_lx, decoded_lz = decode_jit(
        p,
        z_syndrome,
        x_syndrome,
        z_checks_tuple,
        x_checks_tuple,
        lx_tuple,
        lz_tuple,
        num_qubits,
    )

    print(f"Decoded logical X parity: {decoded_lx}")
    print(f"Decoded logical Z parity: {decoded_lz}")

    sector_map = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    print(f"\nActual logical sector: {sector_map[(int(actual_lx), int(actual_lz))]}")
    print(f"Decoded logical sector: {sector_map[(int(decoded_lx), int(decoded_lz))]}")

    if decoded_lx == actual_lx and decoded_lz == actual_lz:
        print("\n[SUCCESS] Decoder correctly identified the logical sector.")


if __name__ == "__main__":
    main()
