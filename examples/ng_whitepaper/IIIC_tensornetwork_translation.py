"""
Tensor Network Format Translation Reference Script
==================================================

This script demonstrates the interoperability of TensorCircuit with other major
tensor network libraries: **TeNPy**, **Quimb**, and **TensorNetwork**.

It showcases how to translate Tensor Network objects (specifically Matrix Product States - MPS
and Matrix Product Operators - MPO) between these frameworks using built-in functions
in `tensorcircuit.quantum`.

Key functionalities demonstrated:
1.  **TensorNetwork Integration**: Constructing a valid MPS `QuOperator` (TensorCircuit's
    native TN operator class) from raw `tensornetwork` Nodes.
2.  **TeNPy Translation**: converting `QuOperator` to TeNPy's `MPS`/`MPO` and vice versa.
3.  **Quimb Translation**: converting `QuOperator` to Quimb's `MatrixProductState`/`MatrixProductOperator`
    and vice versa.
"""

import numpy as np
import tensornetwork as tn
import tensorcircuit as tc
from tensorcircuit import quantum as qu

tc.set_backend("tensorflow")


def separation_line(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def create_random_mps_quoperator(n, d=2, D=4):
    """
    Helper to manually construct a random MPS of n sites, physical dim d, bond dim D
    as a TensorCircuit `QuOperator`.

    This demonstrates the **TensorNetwork** integration: `QuOperator` is built directly
    on top of `tensornetwork.Node` objects.
    """
    print(
        f"Creating a random MPS (n={n}, d={d}, D={D}) using tensornetwork primitive Nodes..."
    )

    nodes = []

    # 1. Create Nodes
    for i in range(n):
        # Shape: (1, d, D) for left, (D, d, 1) for right, (D, d, D) for bulk
        # Check carefully about the dimension 1 legs.
        if i == 0:
            shape = (1, d, D)
        elif i == n - 1:
            shape = (D, d, 1)
        else:
            shape = (D, d, D)

        tensor = np.random.randn(*shape).astype(np.float32) + 1j * np.random.randn(
            *shape
        ).astype(np.float32)
        tensor /= np.linalg.norm(tensor)

        node = tn.Node(tensor, name=f"site_{i}", axis_names=["vL", "p", "vR"])
        nodes.append(node)

    # 2. Connect Nodes
    # Connect vR of site i to vL of site i+1
    for i in range(n - 1):
        tn.connect(nodes[i][2], nodes[i + 1][0])

    # 3. Define External Edges
    out_edges = [node[1] for node in nodes]
    in_edges = []

    # 4. Handle Boundary Edges
    # The left leg of first node and right leg of last node are size 1 dangling edges.
    left_edge = nodes[0][0]
    right_edge = nodes[-1][2]

    ignore_edges = [left_edge, right_edge]

    # 5. Construct QuOperator (MPS is technically a QuVector)
    # Note: qu.QuOperator is a factory function 'quantum_constructor' that returns correct class
    qop = qu.QuOperator(out_edges, in_edges, ignore_edges=ignore_edges)

    print("  -> QuOperator constructed successfully.")
    return qop


def tenpy_translation():
    separation_line("1. Translation: TensorCircuit <-> TeNPy")

    # 1. Create a TC QuOperator (MPS)
    qop = create_random_mps_quoperator(n=5)

    # 2. Convert TC QuOperator -> TeNPy MPS
    print("\n[TC -> TeNPy]")
    print("Converting QuOperator to TeNPy MPS...")
    # NOTE: qop2tenpy expects the QuOperator to have specific leg ordering and structure.
    tenpy_mps = qu.qop2tenpy(qop)

    print(f"  Result type: {type(tenpy_mps)}")
    print(f"  TeNPy MPS norm: {tenpy_mps.norm:.6f}")

    # 3. Modify in TeNPy
    tenpy_mps.canonical_form()

    # 4. Convert TeNPy MPS -> TC QuOperator
    print("\n[TeNPy -> TC]")
    qop_back = qu.tenpy2qop(tenpy_mps)

    # 5. Validation
    # We compare the contraction (dense vector)
    vec_original = qop.eval()
    vec_back = qop_back.eval()

    # Fix potential dtype mismatch (TeNPy uses float64/complex128 by default)
    vec_back = tc.backend.cast(vec_back, vec_original.dtype)

    # Normalize state vectors (canonical_form changes norm typically)
    vec_original /= tc.backend.norm(vec_original)
    vec_back /= tc.backend.norm(vec_back)

    overlap = tc.backend.abs(tc.backend.sum(tc.backend.conj(vec_original) * vec_back))
    print(f"  Overlap: {overlap:.6f}")


def quimb_translation():
    separation_line("2. Translation: TensorCircuit <-> Quimb")

    # 1. Create TC MPS
    qop = create_random_mps_quoperator(n=4)

    # 2. TC -> Quimb
    print("\n[TC -> Quimb]")
    quimb_tn = qu.qop2quimb(qop)
    print(f"  Result type: {type(quimb_tn)}")
    print(f"  Quimb TN info: \n{quimb_tn}")

    # 3. Quimb -> TC
    print("\n[Quimb -> TC]")
    qop_back = qu.quimb2qop(quimb_tn)

    # 4. Validation
    # Normalizing vectors to avoid large global scalar diffs
    vec_original = qop.eval()
    vec_original /= tc.backend.norm(vec_original)

    vec_back = qop_back.eval()
    if vec_back.dtype != vec_original.dtype:
        vec_back = tc.backend.cast(vec_back, vec_original.dtype)

    vec_back /= tc.backend.norm(vec_back)

    overlap = tc.backend.abs(tc.backend.sum(tc.backend.conj(vec_original) * vec_back))
    print(f"  Overlap: {overlap:.6f}")


def main():
    print("Starting Tensor Network Translation Reference Script...")

    tenpy_translation()

    quimb_translation()

    tn_translation()


def tn_translation():
    separation_line("3. Translation: TensorCircuit <-> TensorNetwork Nodes")

    # 1. Create TC MPS
    qop = create_random_mps_quoperator(n=4)

    # 2. TC -> TensorNetwork MPS
    print("\n[TC -> TN MPS]")
    tn_mps = qu.qop2tn(qop)
    print(f"  Result: {tn_mps}")

    # 3. TN MPS -> TC
    print("\n[TN MPS -> TC]")
    qop_back = qu.tn2qop(tn_mps)

    # 4. Validation
    vec_original = qop.eval()
    vec_original /= tc.backend.norm(vec_original)

    vec_back = qop_back.eval()
    vec_back /= tc.backend.norm(vec_back)

    overlap = tc.backend.abs(tc.backend.sum(tc.backend.conj(vec_original) * vec_back))
    print(f"  Overlap: {overlap:.6f}")

    print("\nDemo completed.")


if __name__ == "__main__":
    main()
