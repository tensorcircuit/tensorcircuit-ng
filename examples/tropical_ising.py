"""Tropical (max-plus) Ising ground-state energy via TensorCircuit-NG contraction."""

import itertools
import numpy as np
import tensornetwork as tn
import tensorcircuit as tc
import tensorcircuit.cons as cons
from applications.tropical_algebra import tropical


def brute_force_ground_energy(n, edges, j_vals, h):
    best = np.inf
    for cfg in itertools.product([-1, 1], repeat=n):
        e = -sum(jij * cfg[i] * cfg[j] for (i, j), jij in zip(edges, j_vals)) - sum(
            hi * cfg[i] for i, hi in enumerate(h)
        )
        best = min(best, e)
    return best


def build_tn(n, edges, j_vals, h):
    be = tc.backend
    degree = [0] * n
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    nodes, copy_nodes = [], {}
    for i in range(n):
        cn = tn.CopyNode(degree[i] + 1, 2)  # rank (degree+1) delta, dim 2 -> hyperedge
        copy_nodes[i] = cn
        nodes.append(cn)
        tv = np.array([h[i], -h[i]], dtype=np.float64)
        tvn = tn.Node(be.cast(be.convert_to_tensor(tv), "float64"))
        tn.connect(cn[0], tvn[0])
        nodes.append(tvn)
    leg = [1] * n
    for (i, j), jij in zip(edges, j_vals):
        te = np.array([[jij, -jij], [-jij, jij]], dtype=np.float64)
        ten = tn.Node(be.cast(be.convert_to_tensor(te), "float64"))
        tn.connect(ten[0], copy_nodes[i][leg[i]])
        tn.connect(ten[1], copy_nodes[j][leg[j]])
        leg[i] += 1
        leg[j] += 1
        nodes.append(ten)
    return nodes


def main():
    n = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    J = [1.0, -1.0, 1.0, 1.0, -1.0]
    h = [0.5, -0.3, 0.2, 0.0, 0.4]
    nodes = build_tn(n, edges, J, h)
    with tropical():
        val = float(np.array(cons.contractor(nodes, output_edge_order=[]).tensor))
    e_ground = brute_force_ground_energy(n, edges, J, h)
    print(f"tropical contraction = {val:.6f}")
    print(f"-E_ground (brute)     = {-e_ground:.6f}")
    print(f"match: {np.isclose(val, -e_ground)}")


if __name__ == "__main__":
    main()
