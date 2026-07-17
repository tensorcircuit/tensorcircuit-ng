"""Compute an exact 1000-site expectation and gradient with a spatial scan.

A vertical cut crosses ten rank-2 RZZ bonds in the ket, ten in the bra, and one
dimension-3 MPO bond, giving ``3 * 2**20`` complex numbers.  OMECo finds paths
for the left cap, reusable three-site bulk block, and right cap; JAX applies the
bulk contraction with ``lax.scan``.  The circuit is ``H -> (RZZ ladder -> RX)^10``
with no light cone, truncation, or gate fusion.

For the measured low-memory configuration, run with::

    NVIDIA_TF32_OVERRIDE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS=--xla_gpu_autotune_level=0 \
    python examples/omeco_block_scan_mpo.py
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import cotengra as ctg
import jax
import numpy as np
import omeco
import tensornetwork as tn
from opt_einsum import get_symbol

import tensorcircuit as tc

NQUBITS = 1000
LAYERS = 10
BLOCK_SIZE = 3
NTRIALS = 64
NITERS = 128
RW_WEIGHT = 64.0


def tfi_mpo_tensors(nqubits: int) -> Tuple[Any, ...]:
    coupling = np.ones(nqubits - 1, dtype=np.float32)
    field = -np.ones(nqubits, dtype=np.float32)
    mpo = tn.matrixproductstates.mpo.FiniteTFI(coupling, field, dtype=np.complex64)
    return tuple(tc.backend.convert_to_tensor(tensor) for tensor in mpo.tensors)


def rzz_split_tensors(params: Any) -> Tuple[Any, Any]:
    """Return the exact rank-2 factors of exp(-i theta Z.Z / 2)."""
    identity = tc.backend.convert_to_tensor(np.eye(2, dtype=np.complex64))
    pauli_z = tc.backend.convert_to_tensor(
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)
    )
    angles = params[:-1, :, 0] / 2.0
    left = tc.backend.stack(
        (
            tc.backend.cos(angles)[..., None, None] * identity,
            -tc.backend.i() * tc.backend.sin(angles)[..., None, None] * pauli_z,
        ),
        axis=-1,
    )
    right = tc.backend.stack((identity, pauli_z), axis=0)
    return left, right


def rx_tensors(params: Any) -> Any:
    identity = tc.backend.convert_to_tensor(np.eye(2, dtype=np.complex64))
    pauli_x = tc.backend.convert_to_tensor(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
    )
    angles = params[:, :, 1] / 2.0
    angles = tc.backend.reshape(angles, angles.shape + (1, 1))
    return (
        tc.backend.cos(angles) * identity
        - tc.backend.i() * tc.backend.sin(angles) * pauli_x
    )


def network_tensors(nqubits: int, mpo: Tuple[Any, ...], params: Any) -> List[Any]:
    """Emit ket, bra, and MPO tensors in the order used by the topology."""
    zero = tc.backend.convert_to_tensor(np.array([1.0, 0.0], dtype=np.complex64))
    hadamard = tc.backend.convert_to_tensor(
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex64) / np.sqrt(2.0)
    )
    split_left, split_right = rzz_split_tensors(params)
    rx = rx_tensors(params)
    ket = [zero] * nqubits + [hadamard] * nqubits
    for layer in range(LAYERS):
        for qubit in range(nqubits - 1):
            ket.extend((split_left[qubit, layer], split_right))
        ket.extend(rx[qubit, layer] for qubit in range(nqubits))
    return ket + [tc.backend.conj(tensor) for tensor in ket] + list(mpo)


def circuit_nodes(
    nqubits: int,
    tensors: List[Any],
    branch: int,
) -> Tuple[List[Any], List[int], Dict[Any, Tuple[int, int]], List[Any]]:
    """Connect one circuit branch and label its spatial RZZ cut bonds."""
    nodes: List[Any] = []
    sites: List[int] = []
    channels: Dict[Any, Tuple[int, int]] = {}
    tensor_iter = iter(tensors)

    def add(site: int, name: str) -> Any:
        node = tn.Node(next(tensor_iter), name=name)
        nodes.append(node)
        sites.append(site)
        return node

    front = [add(site, "zero")[0] for site in range(nqubits)]
    for site in range(nqubits):
        gate = add(site, "h")
        gate[1] ^ front[site]
        front[site] = gate[0]
    for layer in range(LAYERS):
        for site in range(nqubits - 1):
            left = add(site, "rzz_left")
            right = add(site + 1, "rzz_right")
            channels[left[2] ^ right[0]] = (branch, layer)
            left[1] ^ front[site]
            right[2] ^ front[site + 1]
            front[site], front[site + 1] = left[0], right[1]
        for site in range(nqubits):
            gate = add(site, "rx")
            gate[1] ^ front[site]
            front[site] = gate[0]
    return nodes, sites, channels, front


def network_nodes(
    nqubits: int, mpo: Tuple[Any, ...], params: Any
) -> Tuple[List[Any], List[int], Dict[Any, Tuple[int, int]]]:
    tensors = network_tensors(nqubits, mpo, params)
    branch_size = 2 * nqubits + LAYERS * (3 * nqubits - 2)
    ket_nodes, ket_sites, channels, ket_front = circuit_nodes(
        nqubits, tensors[:branch_size], 0
    )
    bra_nodes, bra_sites, bra_channels, bra_front = circuit_nodes(
        nqubits, tensors[branch_size : 2 * branch_size], 1
    )
    channels.update(bra_channels)
    mpo_nodes = [tn.Node(tensor, name="mpo") for tensor in mpo]
    for site in range(nqubits - 1):
        channels[mpo_nodes[site][1] ^ mpo_nodes[site + 1][0]] = (2, 0)
    for site in range(nqubits):
        bra_front[site] ^ mpo_nodes[site][-1]
        mpo_nodes[site][-2] ^ ket_front[site]
    nodes = ket_nodes + bra_nodes + mpo_nodes
    sites = ket_sites + bra_sites + list(range(nqubits))
    return nodes, sites, channels


@dataclass(frozen=True)
class Region:
    """A spatial subnetwork with optional left and right cut indices."""

    nqubits: int
    start: int
    stop: int
    tensors: Tuple[int, ...]
    inputs: Tuple[Tuple[int, ...], ...]
    closed: Tuple[int, ...]
    left: Tuple[int, ...]
    right: Tuple[int, ...]
    sizes: Dict[int, int]


def build_region(nqubits: int, start: int, stop: int) -> Region:
    """Extract a spatial region and expose its cut indices."""
    params = np.full((nqubits, LAYERS, 2), 0.1, dtype=np.float32)
    mpo = tfi_mpo_tensors(nqubits)
    nodes, sites, channels = network_nodes(nqubits, mpo, params)

    tensor_number = {node: i for i, node in enumerate(nodes)}
    labels: Dict[Any, int] = {}
    for node in nodes:
        for edge in node.edges:
            labels.setdefault(edge, len(labels))

    all_inputs = tuple(tuple(labels[edge] for edge in node.edges) for node in nodes)
    selected = tuple(i for i, site in enumerate(sites) if start <= site < stop)
    selected_nodes = {nodes[i] for i in selected}
    left: List[Tuple[Tuple[int, int], int]] = []
    right: List[Tuple[Tuple[int, int], int]] = []
    closed = []

    for edge, label in labels.items():
        node1_inside = edge.node1 in selected_nodes
        node2_inside = edge.node2 in selected_nodes
        if node1_inside == node2_inside:
            continue

        outside = edge.node2 if node1_inside else edge.node1
        if outside is None:
            if int(edge.dimension) != 1:
                raise RuntimeError("only dimension-1 dangling edges may be closed")
            closed.append(label)
            continue

        outside_site = sites[tensor_number[outside]]
        item = (channels[edge], label)
        if outside_site < start:
            left.append(item)
        else:
            right.append(item)

    left.sort()
    right.sort()

    return Region(
        nqubits=nqubits,
        start=start,
        stop=stop,
        tensors=selected,
        inputs=tuple(all_inputs[i] for i in selected),
        closed=tuple(sorted(closed)),
        left=tuple(label for _, label in left),
        right=tuple(label for _, label in right),
        sizes={label: int(edge.dimension) for edge, label in labels.items()},
    )


def find_path(name: str, region: Region) -> ctg.ContractionTree:
    """Use OMECo once, then freeze the resulting path for JAX."""
    inputs = list(region.inputs)
    inputs.extend((label,) for label in region.closed)
    if region.left:
        inputs.append(region.left)

    labels = {label for term in inputs for label in term}
    labels.update(region.right)
    sizes = {label: region.sizes[label] for label in labels}
    score = omeco.ScoreFunction(
        tc_weight=1.0,
        sc_weight=0.0,
        rw_weight=RW_WEIGHT,
        sc_target=64.0,
    )

    start = time.perf_counter()
    code = omeco.optimize_code(
        inputs,
        region.right,
        sizes,
        omeco.TreeSA(
            ntrials=NTRIALS,
            niters=NITERS,
            betas=[float(1.0 / x) for x in np.geomspace(2.0, 0.05, NITERS)],
            score=score,
        ),
    )
    search_seconds = time.perf_counter() - start
    path = tc.cons._omeco_tree_to_path(code.to_dict(), len(inputs))

    # Cotengra's execution core expects one-character einsum symbols.
    symbols = {label: get_symbol(i) for i, label in enumerate(sorted(labels))}
    tree = ctg.ContractionTree.from_path(
        [tuple(symbols[label] for label in term) for term in inputs],
        tuple(symbols[label] for label in region.right),
        {symbols[label]: size for label, size in sizes.items()},
        path=path,
    )
    print(
        f"{name:>5} path: search={search_seconds:.2f}s, "
        f"log10(flops)={math.log10(float(tree.total_flops())):.4f}, "
        f"log2(size)={math.log2(float(tree.max_size())):.3f}, "
        f"log2(write)={math.log2(float(tree.total_write())):.3f}"
    )
    return tree


def make_contraction(region: Region, tree: ctg.ContractionTree) -> Callable[..., Any]:
    """Create the numerical contraction for one fixed spatial region."""
    mpo = tfi_mpo_tensors(region.nqubits)

    def contract(params: Any, carry: Any = None) -> Any:
        full_params = tc.backend.concat(
            (
                tc.backend.zeros((region.start, LAYERS, 2), dtype=tc.rdtypestr),
                params,
                tc.backend.zeros(
                    (region.nqubits - region.stop, LAYERS, 2),
                    dtype=tc.rdtypestr,
                ),
            ),
            axis=0,
        )
        tensors = network_tensors(region.nqubits, mpo, full_params)
        arrays = [tensors[i] for i in region.tensors]
        arrays.extend(
            tc.backend.ones((region.sizes[label],), dtype=tc.dtypestr)
            for label in region.closed
        )
        if region.left:
            arrays.append(carry)
        return tree.contract_core(arrays, backend="jax", autojit=False)

    return contract


def main() -> None:
    tc.set_backend("numpy")
    tc.set_dtype("complex64")

    # 1 left site + 332 three-site blocks + 3 right sites = 1000 sites.
    nblocks = (NQUBITS - 2) // BLOCK_SIZE
    right_size = NQUBITS - 1 - nblocks * BLOCK_SIZE
    left_region = build_region(3, 0, 1)
    bulk_region = build_region(BLOCK_SIZE + 2, 1, BLOCK_SIZE + 1)
    right_region = build_region(right_size + 2, 2, right_size + 2)

    carry_shape = tuple(bulk_region.sizes[label] for label in bulk_region.left)
    carry_elements = math.prod(carry_shape)
    print(
        f"exact spatial carry: shape={carry_shape}, "
        f"elements={carry_elements:,}, complex64={carry_elements * 8 / 2**20:.1f} MiB"
    )

    left_tree = find_path("left", left_region)
    bulk_tree = find_path("bulk", bulk_region)
    right_tree = find_path("right", right_region)
    print(f"full scan: {nblocks} reusable {BLOCK_SIZE}-site blocks")

    tc.set_backend("jax")
    tc.set_dtype("complex64")
    left_contract = make_contraction(left_region, left_tree)
    # The full reverse pass stores cut tensors, but rematerializes block internals.
    bulk_contract = jax.checkpoint(make_contraction(bulk_region, bulk_tree))
    right_contract = make_contraction(right_region, right_tree)
    bulk_stop = 1 + nblocks * BLOCK_SIZE

    def expectation(params: Any) -> Any:
        carry = left_contract(params[:1])
        block_params = tc.backend.reshape(
            params[1:bulk_stop], (nblocks, BLOCK_SIZE, LAYERS, 2)
        )

        def step(current_carry: Any, current_params: Any) -> Tuple[Any, None]:
            # carry[k + 1] = exact_three_site_transfer(params[k], carry[k])
            return bulk_contract(current_params, current_carry), None

        carry, _ = jax.lax.scan(step, carry, block_params)
        return tc.backend.real(
            tc.backend.sum(right_contract(params[bulk_stop:], carry))
        )

    params = 0.1 * tc.backend.ones((NQUBITS, LAYERS, 2), dtype=tc.rdtypestr)
    value_and_grad = jax.jit(jax.value_and_grad(expectation))

    start = time.perf_counter()
    value, grad = jax.block_until_ready(value_and_grad(params))
    print(f"JIT compile + first run: {time.perf_counter() - start:.2f}s")
    print(
        f"expectation={float(np.asarray(value)):.8f}, "
        f"gradient shape={grad.shape}, norm={float(np.linalg.norm(np.asarray(grad))):.8f}"
    )

    start = time.perf_counter()
    value, grad = jax.block_until_ready(value_and_grad(params))
    print(f"second run (steady state): {time.perf_counter() - start:.2f}s")
    print(
        f"expectation={float(np.asarray(value)):.8f}, "
        f"gradient shape={grad.shape}, norm={float(np.linalg.norm(np.asarray(grad))):.8f}"
    )


if __name__ == "__main__":
    main()
