"""
Dynamic nearest-neighbor gate programs on uniformly padded dense MPS buffers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ...backends.jax_ops import adaware_qr, adaware_svd
from ...circuit import Circuit
from ...cons import backend
from ...mpscircuit import MPSCircuit
from ...symbolcircuit import SymbolCircuit

Array = Any

_NOOP = 5


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GateProgram:
    """
    Prepared circuit records with one gate per record.

    Record rows are int32 ``(kind, site)``; row ``i`` uses ``gates[i]``.
    Kinds are two-site right (0), two-site left (1), QR right (2), QR left
    (3), onsite (4), and no-op (5). Two-site records address the left site,
    and gate axes are outputs then inputs in increasing site order. Onsite
    rows store ``U ⊗ I``; QR and no-op rows store identities. Arrays are
    pytree leaves and ``n`` is static, so programs can be JIT arguments.
    :func:`stack_gate_programs` adds a leading batch axis to both arrays.

    :ivar records: Integer array of shape ``[L, 2]``.
    :ivar gates: Array of shape ``[L, d, d, d, d]``.
    :ivar n: Number of sites.
    """

    records: Array
    gates: Array
    n: int

    def tree_flatten(self) -> Tuple[Tuple[Array, Array], int]:
        """
        Flatten the program into array leaves and the static site count.

        :return: Record and gate arrays, and ``n``.
        """
        return (self.records, self.gates), self.n

    @classmethod
    def tree_unflatten(
        cls, aux_data: int, leaves: Tuple[Array, Array]
    ) -> "GateProgram":
        """
        Reconstruct a program from array leaves and the static site count.

        :param aux_data: Number of sites.
        :param leaves: Record and gate arrays.
        :return: Gate program.
        """
        return cls(leaves[0], leaves[1], aux_data)


class GateProgramResult(NamedTuple):
    """
    Final padded MPS buffers and the retained-weight estimate.

    :ivar buffers: Complex array of shape ``[n, chi, d, chi]``.
    :ivar fidelity: Product of relative retained weights over truncations.
    """

    buffers: Array
    fidelity: Array


def prepare_gate_program(circuit: Circuit, *, pad: bool = True) -> GateProgram:
    """
    Prepare a concrete adjacent unitary circuit outside JIT.

    Accept a ``Circuit`` with uniform local dimension and default zero input,
    containing only onsite and adjacent two-site unitary gates. Negative and
    reversed indices are normalized. Channels, MPOs, extra instructions, and
    custom inputs are rejected. Unitarity is checked on the host.

    Single-site gates are absorbed into incident two-site gates, which leaves
    every Schmidt spectrum and hence every truncation unchanged; gates on
    sites without two-site gates remain onsite records. Two-site gates are not
    fused with each other, since fusion can change finite-chi truncation. The
    program starts and ends with the canonical center at site zero.

    With ``pad=True`` the record count is padded with no-ops to the next power
    of two, so programs of similar length share one compiled runner. With
    ``pad=False`` the exact length is kept; :func:`stack_gate_programs` then
    pads a batch only to its longest member. Preparation is not
    differentiable; to differentiate gate values, replace entries of
    ``gates`` inside the differentiated function.

    :param circuit: Concrete circuit with default product input.
    :param pad: Pad the record count to a power of two.
    :return: Program for :func:`run_gate_program`.
    """
    if not isinstance(circuit, Circuit) or isinstance(circuit, SymbolCircuit):
        raise TypeError("prepare_gate_program requires a Circuit")
    if (
        any(
            circuit.circuit_param.get(key) is not None
            for key in ("inputs", "mps_inputs", "tensors")
        )
        or circuit._extra_qir
    ):
        raise ValueError(
            "gate programs require default input and no extra instructions"
        )
    n, d = circuit._nqubits, circuit._d
    ops = []
    for instruction in circuit.to_qir():
        if instruction.get("is_channel", False) or instruction.get("mpo", False):
            raise ValueError("gate programs do not support channels or MPO gates")
        index = tuple(int(q) % n for q in instruction["index"])
        if len(index) not in (1, 2) or (
            len(index) == 2 and abs(index[0] - index[1]) != 1
        ):
            raise ValueError("gate programs require onsite or adjacent two-site gates")
        gate = instruction.get("gate")
        if gate is None:
            gate = instruction["gatef"](**instruction.get("parameters", {}))
        matrix = np.asarray(backend.numpy(gate.tensor))
        if instruction.get("diagonal", False):
            matrix = np.diag(matrix.reshape(-1))
        matrix = matrix.reshape((d,) * (2 * len(index)))
        if len(index) == 2 and index[0] > index[1]:
            matrix = matrix.transpose(1, 0, 3, 2)
        matrix = matrix.reshape(d ** len(index), -1)
        if not np.allclose(
            matrix.conj().T @ matrix, np.eye(len(matrix)), atol=2e-5, rtol=2e-5
        ):
            raise ValueError("gate programs require unitary gates")
        ops.append((len(index), min(index), matrix))
    dtype = np.result_type(*(op[2].dtype for op in ops), np.complex64)
    eye = np.eye(d, dtype=dtype)
    pending = [eye] * n
    touched = [False] * n
    last: List[Optional[int]] = [None] * n
    folded: List[Any] = []
    for width, site, matrix in ops:
        if width == 1:
            pending[site] = matrix @ pending[site]
            touched[site] = True
        else:
            folded.append([site, matrix @ np.kron(pending[site], pending[site + 1])])
            for q in (site, site + 1):
                pending[q], touched[q], last[q] = eye, False, len(folded) - 1
    onsite = []
    for q in range(n):
        if not touched[q]:
            continue
        previous = last[q]
        if previous is None:
            onsite.append((4, q, np.kron(pending[q], eye)))
        else:
            site = folded[previous][0]
            local = np.kron(pending[q], eye) if q == site else np.kron(eye, pending[q])
            folded[previous][1] = local @ folded[previous][1]
    entries: List[Tuple[int, int, Optional[Array]]] = list(onsite)
    center = 0
    for number, (site, matrix) in enumerate(folded):
        target = site if abs(center - site) <= abs(center - site - 1) else site + 1
        while center != target:
            right = center < target
            entries.append((2 if right else 3, center if right else center - 1, None))
            center += 1 if right else -1
        right = number + 1 < len(folded) and folded[number + 1][0] > site
        entries.append((0 if right else 1, site, matrix))
        center = site + int(right)
    while center > 0:
        entries.append((3, center - 1, None))
        center -= 1
    length = 1 << (max(1, len(entries)) - 1).bit_length() if pad else len(entries)
    records = np.tile(np.asarray((_NOOP, 0), np.int32), (length, 1))
    gates = np.tile(np.eye(d * d, dtype=dtype), (length, 1, 1))
    for row, (kind, site, value) in enumerate(entries):
        records[row] = kind, site
        if value is not None:
            gates[row] = value
    return GateProgram(
        jnp.asarray(records), jnp.asarray(gates.reshape((length,) + (d,) * 4)), n
    )


def stack_gate_programs(programs: Sequence[GateProgram]) -> GateProgram:
    """
    Pad programs to the longest record count and stack them for ``vmap``.

    Padding appends no-op records, which do not change results. All programs
    must share ``n`` and the local dimension; gate dtypes are promoted. The
    result has a leading batch axis on both arrays; map
    :func:`run_gate_program` over it with ``jax.vmap``.

    :param programs: Non-empty sequence of unstacked programs.
    :return: Stacked program.
    """
    if not programs:
        raise ValueError("stack_gate_programs requires at least one program")
    n, d = programs[0].n, programs[0].gates.shape[-1]
    if any(p.n != n or p.gates.shape[-1] != d for p in programs):
        raise ValueError("stacked gate programs require equal n and local dimension")
    length = max(p.records.shape[0] for p in programs)
    dtype = jnp.result_type(*(p.gates.dtype for p in programs))
    identity = jnp.eye(d * d, dtype=dtype).reshape((d,) * 4)
    records, gates = [], []
    for p in programs:
        extra = length - p.records.shape[0]
        noop = jnp.tile(jnp.asarray((_NOOP, 0), jnp.int32), (extra, 1))
        records.append(jnp.concatenate((p.records, noop)))
        fill = jnp.broadcast_to(identity, (extra,) + identity.shape)
        gates.append(jnp.concatenate((p.gates.astype(dtype), fill)))
    return GateProgram(jnp.stack(records), jnp.stack(gates), n)


def _adjoint(value: Array) -> Array:
    return jnp.conj(value.T)


def _split_pair(
    matrix: Array, chi: int, right: Array, method: str
) -> Tuple[Array, Array, Array]:
    if method == "svd":
        u, s, vh = adaware_svd(matrix)
        total = jnp.sum(s**2)
        lost = jnp.sum(s[chi:] ** 2)
        u, s, vh = u[:, :chi], s[:chi], vh[:chi]
        left = jnp.where(right, u, u * s)
        other = jnp.where(right, s[:, None] * vh, vh)
    else:
        oriented = jnp.where(right, matrix, _adjoint(matrix))
        gram = oriented @ _adjoint(oriented)
        total = jnp.real(jnp.trace(gram))
        scale = jnp.where(total > 0, total, 1)
        gram = (gram + _adjoint(gram)) / (2 * scale)
        # Resolve the padded null-space degeneracy at machine precision.
        ramp = (
            8
            * jnp.finfo(gram.real.dtype).eps
            * jnp.linspace(1, 2, gram.shape[0], dtype=gram.real.dtype)
        )
        _, vectors = jnp.linalg.eigh(gram + jnp.diag(ramp))
        iso = vectors[:, -chi:][:, ::-1]
        center = _adjoint(iso) @ oriented
        residual = oriented - iso @ center
        lost = jnp.real(jnp.vdot(residual, residual))
        left = jnp.where(right, iso, _adjoint(center))
        other = jnp.where(right, center, _adjoint(iso))
    fraction = jnp.clip(lost / jnp.where(total > 0, total, 1), 0, 1)
    return left, other, jnp.log1p(-fraction)


def run_gate_program(
    program: GateProgram,
    *,
    chi: int,
    dtype: Any = jnp.complex64,
    truncation: str = "svd",
    truncation_dtype: Any = None,
) -> GateProgramResult:
    """
    Execute a program from ``|0...0>`` with one shared scan.

    This is a pure function without internal compilation: JIT it, or an
    enclosing function, with ``chi``, ``dtype``, and the truncation options
    static and the program as an argument. Compilation depends on ``n``,
    ``chi``, ``d``, the record count, dtypes, and truncation options, not on
    record or gate values, so different structures of equal record count
    reuse one executable. For a batch, ``jax.vmap`` this function over a
    program from :func:`stack_gate_programs`. Under ``vmap`` with differing
    records every branch of the record dispatch runs for every row, so no-op
    padding costs as much as a two-site gate.

    Buffers have padded shape ``[n, chi, d, chi]`` and are not normalized
    after truncation; ``fidelity`` is the product of relative retained
    weights and equals the final squared norm. Backend SVD failures are not
    silently masked.

    ``gram_eigh`` squares the condition number: use complex128 decomposition
    precision for quantitative work. It uses a machine-precision diagonal
    regularizer and measures discarded weight against the unperturbed matrix.
    Validate accuracy and finiteness on the target accelerator. complex128 in
    ``dtype`` or ``truncation_dtype`` requires x64 enabled around the complete
    JIT/AD invocation; the runner does not change global precision settings.

    SVD/QR reuse custom VJPs: reverse-mode gradients with respect to gate
    values can be used away from singular truncation boundaries, but
    forward-mode JVP is unsupported. Gram eigenvector gradients at degenerate
    spectra are not guaranteed.

    :param program: Unstacked program from :func:`prepare_gate_program`.
    :param chi: Static bond capacity.
    :param dtype: Complex dtype of the state buffers.
    :param truncation: Static choice ``svd`` or ``gram_eigh``.
    :param truncation_dtype: Optional complex decomposition dtype.
    :return: Final buffers and retained-weight estimate.
    """
    if truncation not in ("svd", "gram_eigh"):
        raise ValueError("truncation must be 'svd' or 'gram_eigh'")
    if chi < 1:
        raise ValueError("chi must be positive")
    records, n = program.records, program.n
    if records.ndim != 2:
        raise ValueError("run one program; vmap over stacked programs")
    d = program.gates.shape[-1]
    dtype = jnp.dtype(dtype)
    decomposition = dtype if truncation_dtype is None else jnp.dtype(truncation_dtype)
    if not (
        jnp.issubdtype(dtype, jnp.complexfloating)
        and jnp.issubdtype(decomposition, jnp.complexfloating)
    ):
        raise ValueError("dtype and truncation_dtype must be complex")
    if jnp.complex128 in (dtype, decomposition) and not jax.config.jax_enable_x64:  # type: ignore[attr-defined]
        raise ValueError("complex128 requires an enclosing x64 scope")
    log_dtype = jnp.empty((), dtype=decomposition).real.dtype
    state = jnp.zeros((n, chi, d, chi), dtype=dtype).at[:, 0, 0, 0].set(1)
    gates = program.gates.astype(dtype)

    def kernel(carry: Any, row: Any) -> Any:
        values, log_fidelity = carry
        (kind, site), gate = row
        start = jnp.minimum(site, max(0, n - 2))

        def onsite(operand: Any) -> Any:
            window, logf = operand
            offset = site - start
            tensor = jnp.einsum("ij,ajb->aib", gate[:, 0, :, 0], window[offset])
            return window.at[offset].set(tensor), logf

        if n == 1:
            carry = jax.lax.cond(kind == 4, onsite, lambda operand: operand, carry)
            return carry, None

        address = (start, np.int32(0), np.int32(0), np.int32(0))
        window = jax.lax.dynamic_slice(values, address, (2, chi, d, chi))

        def pair(operand: Any) -> Any:
            local, logf = operand
            theta = jnp.einsum("aik,kjb->aijb", local[0], local[1])
            theta = jnp.einsum("pqij,aijb->apqb", gate, theta)
            matrix = theta.reshape(chi * d, d * chi).astype(decomposition)
            left, right, log_kept = _split_pair(matrix, chi, kind == 0, truncation)
            update = jnp.stack(
                (left.reshape(chi, d, chi), right.reshape(chi, d, chi))
            ).astype(values.dtype)
            return update, logf + log_kept

        def shift(operand: Any) -> Any:
            local, logf = operand
            right = kind == 2
            matrix = jnp.where(
                right,
                local[0].reshape(chi * d, chi),
                local[1].reshape(chi, d * chi).T,
            )
            q, r = adaware_qr(matrix)
            left = jnp.where(
                right, q.reshape(chi, d, chi), jnp.einsum("aib,cb->aic", local[0], r)
            )
            other = jnp.where(
                right, jnp.einsum("ab,bic->aic", r, local[1]), q.T.reshape(chi, d, chi)
            )
            return jnp.stack((left, other)), logf

        branches = (pair, shift, onsite, lambda operand: operand)
        code = jnp.where(kind < 2, 0, jnp.where(kind < 4, 1, kind - 2))
        update, log_fidelity = jax.lax.switch(code, branches, (window, log_fidelity))
        values = jax.lax.dynamic_update_slice(values, update, address)
        return (values, log_fidelity), None

    (buffers, logf), _ = jax.lax.scan(
        kernel, (state, jnp.zeros((), log_dtype)), (records, gates)
    )
    return GateProgramResult(buffers, jnp.exp(logf))


def gate_program_to_mpscircuit(
    result: GateProgramResult, *, split: Optional[Dict[str, Any]] = None
) -> MPSCircuit:
    """
    Losslessly export runner output to a center-zero ``MPSCircuit``.

    Two static QR sweeps remove redundant padding and establish canonical
    tensors without assuming that arbitrary null-space bases occupy trailing
    coordinates. This costs O(n * d * chi**3) work and O(n * d * chi**2) storage.
    No additional truncation or normalization is performed; the result's
    fidelity is available from ``get_fidelity()``. The active TC backend must
    be JAX; construction can occur inside JIT/vmap, but the Python circuit
    object must be consumed there rather than returned by JIT.

    :param result: Output of :func:`run_gate_program`.
    :param split: Optional split policy for subsequent circuit operations.
    :return: Independent canonical circuit with physical boundary dimensions 1.
    """
    if backend.name != "jax":
        raise ValueError("gate program export requires the JAX backend")
    state = result.buffers
    n, _, d, _ = state.shape
    tensors = [state[i] for i in range(n)]
    tensors[0] = tensors[0][:1]
    tensors[-1] = tensors[-1][:, :, :1]
    for site in range(n - 1):
        tensor = tensors[site]
        q, r = adaware_qr(tensor.reshape(-1, tensor.shape[-1]))
        tensors[site] = q.reshape(tensor.shape[0], d, -1)
        tensors[site + 1] = jnp.einsum("ab,bic->aic", r, tensors[site + 1])
    for site in range(n - 1, 0, -1):
        tensor = tensors[site]
        q, r = adaware_qr(tensor.reshape(tensor.shape[0], -1).T)
        tensors[site] = q.T.reshape(-1, d, tensor.shape[-1])
        tensors[site - 1] = jnp.einsum("aib,cb->aic", tensors[site - 1], r)
    circuit = MPSCircuit(n, tensors=tensors, center_position=0, split=split, dim=d)
    circuit._fidelity = result.fidelity
    return circuit
