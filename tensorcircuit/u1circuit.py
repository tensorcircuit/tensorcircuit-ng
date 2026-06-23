"""
Circuit class for U(1) conserving circuits.
"""

from functools import lru_cache, partial
from itertools import combinations
import logging
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .abstractcircuit import AbstractCircuit
from .cons import backend, dtypestr, rdtypestr, idtypestr
from .quantum import sample2all
from .utils import arg_alias

logger = logging.getLogger(__name__)

Tensor = Any


@lru_cache(maxsize=128)
def _get_subsystem_basis_static(n: int, k: int) -> np.ndarray:  # type: ignore
    """
    Get U(1) basis for n qubits and k particles (Cached Static version).
    """
    basis = []
    for filled_combo in combinations(range(n), k):
        state = 0
        for bit_idx in filled_combo:
            state |= 1 << (n - 1 - bit_idx)
        basis.append(state)
    basis = sorted(basis)
    return np.array(basis, dtype=np.int64)


@lru_cache(maxsize=128)
def _extract_subsystem_bits_static(
    nqubits: int, k: int, subsystem: Tuple[int, ...]
) -> np.ndarray:  # type: ignore
    """
    Extract bits of subsystem and pack them into a contiguous integer (Cached Static version).
    """
    basis_np = _get_subsystem_basis_static(nqubits, k)
    n = len(subsystem)
    res = np.zeros(len(basis_np), dtype=np.int64)
    for i, q_idx in enumerate(subsystem):
        old_bp = nqubits - 1 - q_idx
        new_bp = n - 1 - i
        bits = (basis_np >> old_bp) & 1
        res |= bits << new_bp
    return res


def _normalize_ps_list(
    nqubits: int, ps_list: Sequence[Any]
) -> Tuple[Tuple[int, ...], ...]:
    """
    Normalize a list of Pauli strings (as dicts or sequences) into a tuple of tuples.
    """
    standard_ps_list = []
    for ps in ps_list:
        if isinstance(ps, dict):
            x = ps.get("x", [])
            y = ps.get("y", [])
            z = ps.get("z", [])
            s_ps = [0] * nqubits
            for idx in x:
                s_ps[idx] = 1
            for idx in y:
                s_ps[idx] = 2
            for idx in z:
                s_ps[idx] = 3
            standard_ps_list.append(tuple(s_ps))
        elif hasattr(ps, "__len__") and len(ps) == nqubits:
            standard_ps_list.append(tuple(ps))
        else:
            raise ValueError(f"Invalid Pauli string format: {ps}")
    return tuple(standard_ps_list)


_u1_operator_cache: Dict[Tuple[int, int, Tuple[Tuple[int, ...], ...]], "U1Operator"] = (
    {}
)


class U1Operator:
    """
    Operator helper class for fast expectation calculation in the U(1) subspace.
    """

    def __init__(
        self,
        nqubits: int,
        k: int,
        ps_list: Sequence[Any],
    ) -> None:
        self.nqubits = nqubits
        self.k = k

        self.ps_tuple = _normalize_ps_list(nqubits, ps_list)

        basis = _get_subsystem_basis_static(nqubits, k)
        dim = len(basis)
        basis_index = {int(state): i for i, state in enumerate(basis)}
        occupied = (basis[:, None] >> (nqubits - 1 - np.arange(nqubits)[None, :])) & 1

        diagonal_indices = []
        diagonal_vectors = []

        off_diagonal_indices = []
        off_diagonal_targets = []
        off_diagonal_phases = []

        for term_idx, ps in enumerate(self.ps_tuple):
            x_indices = []
            y_indices = []
            z_indices = []
            for i, val in enumerate(ps):
                if val == 1:
                    x_indices.append(i)
                elif val == 2:
                    y_indices.append(i)
                elif val == 3:
                    z_indices.append(i)

            xy_len = len(x_indices) + len(y_indices)
            if xy_len == 0:
                if len(z_indices) > 0:
                    D_a = np.prod(1.0 - 2.0 * occupied[:, z_indices], axis=1)
                else:
                    D_a = np.ones(dim, dtype=np.float64)
                diagonal_indices.append(term_idx)
                diagonal_vectors.append(D_a)
            else:
                if xy_len % 2 != 0:
                    # U(1) symmetry: odd number of X/Y flips -> zero expectation
                    continue

                swap_mask = 0
                for idx in x_indices + y_indices:
                    swap_mask |= 1 << (nqubits - 1 - idx)
                target_s_arr = basis ^ swap_mask
                target_idx_arr = np.array(
                    [basis_index.get(ts, -1) for ts in target_s_arr],
                    dtype=np.int64,
                )
                valid = target_idx_arr != -1
                target_idx_safe = np.where(valid, target_idx_arr, 0)

                phase = np.ones(dim, dtype=np.complex128)
                if len(z_indices) > 0:
                    phase = phase * np.prod(1.0 - 2.0 * occupied[:, z_indices], axis=1)
                if len(y_indices) > 0:
                    phase = phase * np.prod(
                        1j * (1.0 - 2.0 * occupied[target_idx_safe][:, y_indices]),
                        axis=1,
                    )
                phase = np.where(valid, phase, 0.0).astype(np.complex128)  # type: ignore

                off_diagonal_indices.append(term_idx)
                off_diagonal_targets.append(target_idx_safe)
                off_diagonal_phases.append(phase)

        self.diagonal_indices = np.array(diagonal_indices, dtype=np.int64)
        if len(diagonal_vectors) > 0:
            self.diagonal_vectors = np.stack(diagonal_vectors, axis=0)
        else:
            self.diagonal_vectors = np.empty((0, dim), dtype=np.float64)

        self.off_diagonal_indices = np.array(off_diagonal_indices, dtype=np.int64)
        if len(off_diagonal_targets) > 0:
            self.off_diagonal_targets = np.stack(off_diagonal_targets, axis=0)
            self.off_diagonal_phases = np.stack(off_diagonal_phases, axis=0)
        else:
            self.off_diagonal_targets = np.empty((0, dim), dtype=np.int64)
            self.off_diagonal_phases = np.empty((0, dim), dtype=np.complex128)


class U1Circuit(AbstractCircuit):
    """
    Circuit class for U(1) conserving circuits (particle number conservation).
    The simulation is performed entirely within the compressed U(1) subspace,
    enabling efficient simulation of systems where the total number of excitations
    is preserved.

    Note: Supports up to 64 qubits by utilizing 64-bit integer bitmasks for
    basis state representation.
    """

    def __init__(
        self,
        nqubits: int,
        k: Optional[int] = None,
        filled: Optional[Sequence[int]] = None,
        inputs: Optional[Any] = None,
    ) -> None:
        """
        Initialize a U(1) conserving circuit.

        :param nqubits: Number of qubits in the circuit.
        :type nqubits: int
        :param k: Total number of excitations (particles) to preserve.
            If None, it is inferred from the length of ``filled``.
        :type k: Optional[int], defaults to None
        :param filled: Initial indices of qubits that are occupied (|1> state).
            If provided and ``inputs`` is None, the initial state is set to this
            computational basis state.
        :type filled: Optional[Sequence[int]], defaults to None
        :param inputs: Initial state vector already defined in the U(1) subspace.
            If provided, must have shape (comb(nqubits, k),).
        :type inputs: Optional[Any], defaults to None
        """
        self._nqubits = nqubits
        if nqubits >= 64:
            raise ValueError(
                f"U1Circuit only supports nqubits < 64, but got {nqubits}. "
            )

        if k is None and filled is None:
            raise ValueError("Either 'k' or 'filled' must be provided")

        if filled is not None:
            if isinstance(filled, (list, tuple, range, set)):
                filled = list(filled)
                if k is not None and len(filled) != k:
                    raise ValueError(
                        f"Provided 'k' ({k}) does not match length of 'filled' ({len(filled)})"
                    )
                if len(set(filled)) != len(filled):
                    raise ValueError(f"Duplicate indices found in 'filled': {filled}")
                for i in filled:
                    if i < 0 or i >= nqubits:
                        raise ValueError(
                            f"Index {i} in 'filled' is out of range for {nqubits} qubits"
                        )
                if k is None:
                    k = len(filled)
            else:
                if k is None:
                    k = int(backend.shape_tuple(filled)[0])
        else:
            filled = list(range(k))  # type: ignore

        self._k = k
        self._d = 2
        self._qir: List[Dict[str, Any]] = []
        self._extra_qir: List[Dict[str, Any]] = []
        self.is_mps = False

        self.circuit_param = {
            "nqubits": nqubits,
            "k": k,
            "filled": filled,
            "inputs": inputs,
        }

        # Mapping helpers
        # TC uses qubit 0 as the leftmost (highest) bit
        # So qubit i corresponds to bit position (n-1-i) in the state integer
        assert k is not None
        self._dim = comb(nqubits, k)
        self._basis_np = _get_subsystem_basis_static(nqubits, k)
        self._basis = self._basis_np.tolist()

        self._idtypestr = "int64" if nqubits > 30 else idtypestr

        # Convert basis to backend tensor for faster lookup
        self._basis_tensor = backend.cast(
            backend.convert_to_tensor(self._basis_np), dtype=self._idtypestr
        )

        if inputs is not None:
            self._state = backend.cast(inputs, dtypestr)
        else:
            filled_tensor = backend.cast(
                backend.convert_to_tensor(filled), self._idtypestr
            )
            bit_positions = nqubits - 1 - filled_tensor
            bits = backend.left_shift(
                backend.cast(backend.convert_to_tensor(1), self._idtypestr),
                bit_positions,
            )
            filled_state = backend.sum(bits)

            # Find the index in our basis - JIT FRIENDLY
            # Use searchsorted and one_hot to stay within the graph
            fs_tensor = backend.reshape(
                backend.cast(filled_state, dtype=self._idtypestr), [1]
            )
            initial_idx = backend.searchsorted(self._basis_tensor, fs_tensor)[0]
            # Clip index for safety - searchsorted should always find exact match
            # but we clip for safety across backends
            initial_idx = backend.where(
                initial_idx >= self._dim, self._dim - 1, initial_idx
            )

            # Create state vector: all zeros except at initial_idx
            oh = backend.onehot(initial_idx, self._dim)
            self._state = backend.cast(oh, dtypestr)

    # -------------------------------------------------------------------------
    # Internal gate implementations (DRY - single source of truth)
    # -------------------------------------------------------------------------

    def _bit_position(self, i: int) -> int:
        """Convert qubit index to bit position (TC uses reversed ordering)."""
        return self._nqubits - 1 - i

    def _get_bit(self, i: int) -> Tensor:
        """Extract bit value (0 or 1) at position corresponding to qubit i."""
        bp = self._bit_position(i)
        one = backend.cast(backend.convert_to_tensor(1), self._idtypestr)
        mask = backend.left_shift(one, bp)
        return backend.right_shift(backend.bitwise_and(self._basis_tensor, mask), bp)

    def _get_swapped_info(self, i: int, j: int) -> Tuple[Tensor, Tensor]:
        """Get XOR difference and target indices for swapping qubits i and j."""
        bpi, bpj = self._bit_position(i), self._bit_position(j)
        bi = self._get_bit(i)
        bj = self._get_bit(j)
        diff = backend.bitwise_xor(bi, bj)
        one = backend.cast(backend.convert_to_tensor(1), self._idtypestr)
        mask_swap = backend.bitwise_or(
            backend.left_shift(one, bpi), backend.left_shift(one, bpj)
        )
        new_basis = backend.bitwise_xor(self._basis_tensor, diff * mask_swap)
        indices = backend.searchsorted(self._basis_tensor, new_basis)
        return diff, indices

    def _apply_rz(self, i: int, theta: Any) -> None:
        """Apply RZ(theta) on qubit i: |0> -> |0>, |1> -> e^{-i*theta/2}|1>."""
        bit_val = self._get_bit(i)
        phases = backend.cast(
            -0.5
            * theta
            * backend.cast(
                1 - 2 * backend.cast(bit_val, dtype=dtypestr), dtype=dtypestr
            ),
            dtype=dtypestr,
        )
        self._state = self._state * backend.exp(1j * phases)

    def _apply_rzz(self, i: int, j: int, theta: Any) -> None:
        """Apply RZZ(theta) on qubits i,j: exp(-i*theta/2 * Z_i Z_j)."""
        zi = self._get_bit(i)
        zj = self._get_bit(j)
        zz = 1 - 2 * backend.cast(backend.bitwise_xor(zi, zj), dtype=dtypestr)
        phases = backend.cast(-0.5 * theta * zz, dtype=dtypestr)
        self._state = self._state * backend.exp(1j * phases)

    def _apply_cz(self, i: int, j: int) -> None:
        """Apply CZ gate: |11> -> -|11>, others unchanged."""
        one = backend.cast(backend.convert_to_tensor(1), self._idtypestr)
        mask = backend.bitwise_or(
            backend.left_shift(one, self._bit_position(i)),
            backend.left_shift(one, self._bit_position(j)),
        )
        both_on = backend.cast(
            backend.bitwise_and(self._basis_tensor, mask) == mask, dtype=dtypestr
        )
        self._state = self._state * (1.0 - 2.0 * both_on)

    def _apply_cphase(self, i: int, j: int, theta: Any) -> None:
        """Apply controlled-phase: |11> -> e^{i*theta}|11>, others unchanged."""
        one = backend.cast(backend.convert_to_tensor(1), self._idtypestr)
        mask = backend.bitwise_or(
            backend.left_shift(one, self._bit_position(i)),
            backend.left_shift(one, self._bit_position(j)),
        )
        both_on = backend.cast(
            backend.bitwise_and(self._basis_tensor, mask) == mask, dtype=dtypestr
        )
        phase_val = backend.exp(
            1j * backend.cast(backend.convert_to_tensor(theta), dtypestr)
        )
        self._state = self._state * (1.0 + (phase_val - 1.0) * both_on)

    def _apply_swap(self, i: int, j: int) -> None:
        """Apply SWAP gate: exchange qubits i and j."""
        _, indices = self._get_swapped_info(i, j)
        self._state = backend.gather1d(self._state, indices)

    def _apply_iswap(self, i: int, j: int, theta: Any) -> None:
        """Apply iSWAP(theta): parameterized iSWAP with angle theta*pi/2."""
        diff, indices = self._get_swapped_info(i, j)
        theta_c = backend.cast(backend.convert_to_tensor(theta), dtypestr)
        cos_t = backend.cos(theta_c * np.pi / 2)
        isin_t = 1j * backend.sin(theta_c * np.pi / 2)
        diff_f = backend.cast(diff, dtypestr)
        swapped_state = backend.gather1d(self._state, indices)

        self._state = (1.0 - diff_f) * self._state + diff_f * (
            cos_t * self._state + isin_t * swapped_state
        )

    def _apply_diagonal(self, index: Sequence[int], diagonal: Any) -> None:
        """Apply diagonal gate on qubits specified by index."""
        # Qubit index[0] corresponds to the most significant bit in the diagonal vector
        config_val = backend.cast(
            backend.zeros_like(self._basis_tensor), self._idtypestr
        )
        m = len(index)
        for i, idx in enumerate(index):
            bit = self._get_bit(idx)
            config_val = backend.bitwise_xor(
                config_val, backend.left_shift(bit, m - 1 - i)
            )

        # config_val is now a tensor of indices [dim], each in [0, 2^m - 1]
        if hasattr(diagonal, "tensor"):
            diagonal = diagonal.tensor
        diag_tensor = backend.cast(backend.convert_to_tensor(diagonal), dtypestr)
        diag_tensor = backend.reshape(diag_tensor, [-1])
        phases = backend.gather1d(diag_tensor, config_val)
        self._state = self._state * phases

    # -------------------------------------------------------------------------
    # Public gate methods (delegate to internal implementations)
    # -------------------------------------------------------------------------

    def apply_general_gate(
        self,
        gate: Any,
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        diagonal: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Apply a gate by name. Called by _meta_apply generated methods.

        :param gate: Gate tensor (ignored, dispatch is by name)
        :param index: Qubit indices
        :param name: Gate name (rz, rzz, cz, cphase, swap, iswap)
        :param split: Split configuration (ignored in U1Circuit)
        :param mpo: MPO flag (ignored in U1Circuit)
        :param diagonal: Diagonal flag (ignored in U1Circuit)
        :param ir_dict: QIR dictionary for recording
        :param kwargs: Extra parameters
        """
        if name is None:
            name = ""
        non_u1 = {"x", "y", "h", "t", "s", "td", "sd", "rx", "ry"}
        if name and name.lower() in non_u1:
            raise ValueError(
                f"Gate {name} is not U(1) conserving and cannot be used in U1Circuit."
            )

        gate_name = name.lower() if name else None

        # Extract parameters
        params = {}
        if ir_dict is not None and "parameters" in ir_dict:
            params.update(ir_dict["parameters"])
        params.update(kwargs)

        # Record gate in QIR
        gate_dict = {
            "gate": gate,
            "index": index,
            "name": name,
            "split": split,
            "mpo": mpo,
            "diagonal": diagonal,
        }
        if params:
            gate_dict["parameters"] = params

        if ir_dict is not None:
            ir_dict.update(gate_dict)
        else:
            ir_dict = gate_dict
        self._qir.append(ir_dict)

        if gate_name == "rz":
            self._apply_rz(index[0], params.get("theta", 0))
        elif gate_name == "rzz":
            self._apply_rzz(index[0], index[1], params.get("theta", 0))
        elif gate_name == "cz":
            self._apply_cz(index[0], index[1])
        elif gate_name == "cphase":
            self._apply_cphase(index[0], index[1], params.get("theta", 0))
        elif gate_name == "swap":
            self._apply_swap(index[0], index[1])
        elif gate_name == "iswap":
            self._apply_iswap(index[0], index[1], params.get("theta", 1.0))
        elif gate_name == "diagonal":
            self._apply_diagonal(index, gate)
        else:
            raise ValueError(
                f"Gate {name} not implemented in U1Circuit. "
                "Supported: rz, rzz, cz, cphase, swap, iswap, diagonal."
            )

    # Note: Most gate methods (rz, rzz, cz, swap, iswap, cphase, etc.) are
    # auto-generated by _meta_apply() which calls apply_general_gate.

    # -------------------------------------------------------------------------
    # State and expectation methods
    # -------------------------------------------------------------------------

    def expectation_z(self, i: int) -> Tensor:
        """
        Compute expectation value of Z operator on qubit i.

        :param i: Qubit index
        :return: Expectation value <Z_i>
        """
        bit_i = backend.cast(self._get_bit(i), dtype=rdtypestr)
        z_vals = 1.0 - 2.0 * bit_i
        return backend.sum(
            backend.cast(backend.abs(self._state) ** 2, rdtypestr) * z_vals
        )

    def expectation(self, *ops: Any, **kws: Any) -> Tensor:
        """
        Compute expectation value of operators.

        Currently only supports single Z operator via expectation_z.
        For general operators, please use expectation_ps.

        :param ops: Operator specification
        :return: Expectation value
        """
        raise NotImplementedError(
            "General expectation not yet implemented for U1Circuit. Please use expectation_ps instead."
        )

    def expectation_ps(  # type: ignore
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        ps: Optional[Sequence[int]] = None,
        **kws: Any,
    ) -> Any:
        """
        Compute expectation value of a Pauli string.

        :param x: Qubit indices for X operators
        :param y: Qubit indices for Y operators
        :param z: Qubit indices for Z operators
        :param ps: Alternative Pauli string specification (0=I, 1=X, 2=Y, 3=Z)
        :return: Expectation value <P>
        """
        if ps is not None:
            from .quantum import ps2xyz

            d = ps2xyz(list(ps))  # type: ignore
            x = d.get("x", [])
            y = d.get("y", [])
            z = d.get("z", [])

        if x is not None:
            for i in x:
                if i < 0 or i >= self._nqubits:
                    raise ValueError(f"Index {i} in 'x' is out of range")
        if y is not None:
            for i in y:
                if i < 0 or i >= self._nqubits:
                    raise ValueError(f"Index {i} in 'y' is out of range")
        if z is not None:
            for i in z:
                if i < 0 or i >= self._nqubits:
                    raise ValueError(f"Index {i} in 'z' is out of range")

        x = x or []
        y = y or []
        z = z or []

        # Weight check: X and Y flip bits. To preserve U(1) symmetry,
        # the total number of flips must be even, and they must preserve particle number.
        # But here we just want to compute the expectation in the U(1) sector.
        # If the Pauli string moves the state out of the sector, the expectation is 0.

        all_xy = sorted(list(set(x) | set(y)))
        # For U(1) sector, expectation is 0 if total bit flips is odd
        if len(all_xy) % 2 != 0:
            return backend.cast(backend.convert_to_tensor(0.0), dtypestr)

        # Z contribution: Z_i |s> = (-1)^{s_i} |s>
        z_factor = backend.cast(backend.convert_to_tensor(1.0), dtypestr)
        for idx in z:
            bits = self._get_bit(idx)
            z_factor = z_factor * (1.0 - 2.0 * backend.cast(bits, dtypestr))

        if not all_xy:
            return backend.sum(backend.conj(self._state) * self._state * z_factor)

        swap_mask = backend.cast(backend.convert_to_tensor(0), self._idtypestr)
        one = backend.cast(backend.convert_to_tensor(1), self._idtypestr)
        for idx in all_xy:
            swap_mask = backend.bitwise_or(
                swap_mask, backend.left_shift(one, self._bit_position(idx))
            )

        new_basis = backend.bitwise_xor(self._basis_tensor, swap_mask)

        # Check if weight preserved (U(1) symmetry ensures this)
        indices = backend.searchsorted(self._basis_tensor, new_basis)
        # Clip indices for safe gather
        safe_indices = backend.where(indices >= self._dim, self._dim - 1, indices)
        valid = (indices < self._dim) & (
            backend.gather1d(self._basis_tensor, safe_indices) == new_basis
        )

        phase = z_factor
        for idx in y:
            bp = self._bit_position(idx)
            # Y phase depends on the bit value in the KET (target) state
            # Y|0> = i|1>, Y|1> = -i|0>
            # Read bit from new_basis (the ket side, before Y flips it)
            mask = backend.left_shift(one, bp)
            bit = backend.cast(
                backend.right_shift(backend.bitwise_and(new_basis, mask), bp),
                dtype=dtypestr,
            )
            # bit=0 -> Y|0>=i|1> -> phase=i, bit=1 -> Y|1>=-i|0> -> phase=-i
            phase = phase * (1j * (1.0 - 2.0 * bit))

        target_state = backend.gather1d(self._state, safe_indices)
        target_state = target_state * backend.cast(valid, dtypestr)

        return backend.sum(
            backend.conj(self._state) * target_state * backend.cast(phase, dtypestr)
        )

    def wavefunction(self) -> Any:
        """Return the state vector in the U(1) conserved subspace."""
        return self._state

    def state(self) -> Any:
        """Return the state vector in the U(1) conserved subspace."""
        return self._state

    # -------------------------------------------------------------------------
    # Dense state conversion
    # -------------------------------------------------------------------------

    def to_dense(self) -> Tensor:
        """
        Convert the U(1) conserved state to the full 2^n Hilbert space.

        Returns a state vector of shape [2^nqubits] where non-zero amplitudes
        appear only at positions corresponding to basis states with exactly k
        filled qubits.

        :return: State vector in full Hilbert space
        :rtype: Tensor
        """
        full_dim = 2**self._nqubits
        # Create zero vector of full size
        dense_state = backend.zeros([full_dim], dtype=dtypestr)
        # Scatter the U(1) state into the full space at basis positions
        # indices should be shape [n, 1] for 1D scatter
        indices = backend.reshape(
            backend.cast(self._basis_tensor, self._idtypestr), [self._dim, 1]
        )
        dense_state = backend.scatter(dense_state, indices, self._state)
        return dense_state

    # -------------------------------------------------------------------------
    # Probability and sampling
    # -------------------------------------------------------------------------

    def probability(self) -> Tensor:
        """
        Get the probability vector over the U(1) conserved basis states.

        :return: Probability vector of shape [dim] where dim = C(n, k)
        :rtype: Tensor
        """
        return backend.real(backend.abs(self._state) ** 2)

    def probability_full(self) -> Tensor:
        """
        Get the probability vector over the full 2^n computational basis.

        :return: Probability vector of shape [2^nqubits]
        :rtype: Tensor
        """
        dense_state = self.to_dense()
        return backend.real(backend.abs(dense_state) ** 2)

    @partial(arg_alias, alias_dict={"format": ["format_"]})
    def sample(
        self,
        batch: Optional[int] = None,
        allow_state: bool = True,
        format: Optional[str] = None,
        random_generator: Optional[Any] = None,
        status: Optional[Tensor] = None,
        jittable: bool = True,
    ) -> Any:
        """
        Sample from the U(1) circuit state.

        :param batch: Number of samples, defaults to None (single sample)
        :type batch: Optional[int], optional
        :param allow_state: If true, sample from the state vector directly
        :type allow_state: bool, optional
        :param format: Sample format. Options:
            - None: returns list of (binary_config_tensor, probability) tuples
            - "sample_int": np.array of integer samples
            - "sample_bin": list of binary arrays
            - "count_vector": count vector over full basis
            - "count_tuple": (unique_values, counts)
            - "count_dict_bin": {"01..": count, ...}
            - "count_dict_int": {int: count, ...}
        :type format: Optional[str]
        :param random_generator: Random generator, defaults to None
        :type random_generator: Optional[Any], optional
        :param status: External randomness tensor uniformly from [0, 1], shape [batch]
        :type status: Optional[Tensor]
        :param jittable: Keep full size for jit compatibility, defaults to True
        :type jittable: bool
        :return: Samples in the specified format
        :rtype: Any
        """
        if batch is None:
            nbatch = 1
        else:
            nbatch = batch

        # Get probabilities over U(1) basis
        p = self.probability()

        # Sample indices in the U(1) basis
        u1_indices = backend.probability_sample(nbatch, p, status, random_generator)

        # Map U(1) indices to full basis integers
        # u1_indices are indices into self._basis
        full_indices = backend.gather1d(self._basis_tensor, u1_indices)

        if format is None:
            # Return list of (binary_config, probability) tuples
            # Convert full_indices to binary configurations (TC ordering)
            results = []
            for i in range(nbatch):
                idx = full_indices[i]
                # Convert integer to binary representation (reversed ordering)
                binary = []
                for q in range(self._nqubits):
                    bp = self._bit_position(q)
                    bit = backend.bitwise_and(backend.right_shift(idx, bp), 1)
                    binary.append(backend.cast(bit, rdtypestr))
                binary_tensor = backend.stack(binary)
                results.append((binary_tensor, -1.0))
            if batch is None:
                return results[0]
            return results

        # Use sample2all for format conversion
        # full_indices are already integers representing the full basis states
        ch = backend.cast(full_indices, self._idtypestr)
        return sample2all(
            ch,
            n=self._nqubits,
            format=format,
            jittable=jittable,
            dim=self._d,
        )

    def measure(
        self,
        *index: int,
        with_prob: bool = False,
        status: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Any]:
        """
        Measure specific qubits in the computational basis.

        This is a simplified measurement that samples from the full state
        and returns the values at the specified qubit indices.

        :param index: Qubit indices to measure
        :type index: int
        :param with_prob: If true, return the probability of the outcome
        :type with_prob: bool, optional
        :param status: External randomness tensor, shape [1]
        :type status: Optional[Tensor]
        :return: Tuple of (measurement outcomes tensor, probability or -1.0)
        :rtype: Tuple[Tensor, Any]
        """
        # Get probabilities and sample one state
        p = self.probability()

        # Sample one index from U(1) basis
        u1_idx = backend.probability_sample(1, p, status, None)[0]

        # Get the full basis state integer
        full_state = backend.gather1d(self._basis_tensor, u1_idx[None])[0]

        # Extract bits at the specified indices (using TC ordering)
        outcomes = []
        for i in index:
            if i < 0 or i >= self._nqubits:
                raise ValueError(f"Index {i} is out of range for measurement")
            bp = self._bit_position(i)
            bit = backend.bitwise_and(backend.right_shift(full_state, bp), 1)
            outcomes.append(backend.cast(bit, rdtypestr))

        outcome_tensor = backend.stack(outcomes)

        if with_prob:
            # Get the probability of this outcome
            prob = backend.gather1d(p, u1_idx[None])[0]
            return outcome_tensor, prob
        else:
            return outcome_tensor, -1.0

    def _process_subsystem(
        self,
        subsystem_to_keep: Optional[Sequence[int]] = None,
        subsystem_to_traceout: Optional[Sequence[int]] = None,
    ) -> Tuple[
        Sequence[int],
        int,
        int,
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        int,
        int,
    ]:
        """
        Shared logic for subsystem validation and basis mapping.
        """
        if subsystem_to_keep is None and subsystem_to_traceout is None:
            raise ValueError(
                "Either subsystem_to_keep or subsystem_to_traceout must be provided"
            )
        if subsystem_to_keep is not None and subsystem_to_traceout is not None:
            raise ValueError(
                "Only one of subsystem_to_keep or subsystem_to_traceout should be provided"
            )

        if subsystem_to_keep is None:
            # Type ignore since we already checked both aren't None
            subsystem_to_keep = [
                i
                for i in range(self._nqubits)
                if i not in subsystem_to_traceout  # type: ignore
            ]

        nA = len(subsystem_to_keep)
        nB = self._nqubits - nA
        subsystem_B = [
            i for i in range(self._nqubits) if i not in subsystem_to_keep  # type: ignore
        ]

        # Use static mapping to stay JIT friendly
        # Now using cached global helper functions
        sA_all = _extract_subsystem_bits_static(
            self._nqubits, self._k, tuple(subsystem_to_keep)
        )
        sB_all = _extract_subsystem_bits_static(
            self._nqubits, self._k, tuple(subsystem_B)
        )

        # Calculate kA for each basis state statically
        kA_all = np.array([bin(x).count("1") for x in sA_all])

        k_min = max(0, self._k - nB)  # type: ignore
        k_max = min(self._k, nA)  # type: ignore

        return (  # type: ignore
            subsystem_to_keep,
            nA,
            nB,
            sA_all,
            sB_all,
            kA_all,
            k_min,
            k_max,
        )

    def reduced_density_matrix(
        self,
        subsystem_to_keep: Optional[Sequence[int]] = None,
        subsystem_to_traceout: Optional[Sequence[int]] = None,
        return_blocks: bool = False,
    ) -> Any:
        """
        Compute the reduced density matrix for the specified qubits.

        :param subsystem_to_keep: The qubits to keep (all others are traced out).
        :type subsystem_to_keep: Sequence[int], optional
        :param subsystem_to_traceout: The qubits to trace out.
        :type subsystem_to_traceout: Sequence[int], optional
        :param return_blocks: If true, return a list of RDM blocks for each k_A.
        :type return_blocks: bool
        :return: The reduced density matrix or a list of blocks.
        :rtype: Any
        """
        (
            _,
            nA,
            nB,
            sA_all,
            sB_all,
            kA_all,
            k_min,
            k_max,
        ) = self._process_subsystem(subsystem_to_keep, subsystem_to_traceout)

        blocks = []
        if not return_blocks:
            if nA > 14:
                raise ValueError(
                    f"Subsystem size {nA} is too large to return a dense RDM. "
                    "Please use return_blocks=True or compute entropy directly."
                )
            full_dim = 2**nA
            rho_full = backend.zeros([full_dim, full_dim], dtype=dtypestr)

        for kA in range(k_min, k_max + 1):  # type: ignore
            kB = self._k - kA  # type: ignore
            mask = kA_all == kA
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            sA_sector = sA_all[indices]
            sB_sector = sB_all[indices]

            # Move coefficients to backend
            coeffs_sector = backend.gather1d(
                self._state, backend.convert_to_tensor(indices)
            )

            # Map to local subspace indices
            basisA = _get_subsystem_basis_static(nA, kA)
            basisB = _get_subsystem_basis_static(nB, kB)

            idxA = np.searchsorted(basisA, sA_sector)
            idxB = np.searchsorted(basisB, sB_sector)

            dimA = comb(nA, kA)
            dimB = comb(nB, kB)

            # Form coefficient matrix M: dimA x dimB
            M = backend.zeros([dimA, dimB], dtype=dtypestr)
            # Use backend tensors for scatter
            scatter_indices = backend.convert_to_tensor(np.stack([idxA, idxB], axis=1))
            M = backend.scatter(M, scatter_indices, coeffs_sector)

            # Compute block RDM
            rho_k = backend.matmul(M, backend.conj(backend.transpose(M)))

            if return_blocks:
                blocks.append(rho_k)
            else:
                # Scatter rho_k into rho_full
                ii, jj = np.meshgrid(basisA, basisA, indexing="ij")
                scatter_idx_full = backend.convert_to_tensor(
                    np.stack([ii.flatten(), jj.flatten()], axis=1)
                )
                rho_full = backend.scatter(
                    rho_full, scatter_idx_full, backend.reshape(rho_k, [-1])
                )

        if return_blocks:
            return blocks
        return rho_full

    def entanglement_entropy(
        self,
        subsystem_to_keep: Optional[Sequence[int]] = None,
        subsystem_to_traceout: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """
        Compute the von Neumann entanglement entropy for the specified qubits.

        :param subsystem_to_keep: The qubits to keep.
        :type subsystem_to_keep: Sequence[int], optional
        :param subsystem_to_traceout: The qubits to trace out.
        :type subsystem_to_traceout: Sequence[int], optional
        :return: Entanglement entropy.
        :rtype: Tensor
        """
        (
            _,
            nA,
            nB,
            sA_all,
            sB_all,
            kA_all,
            k_min,
            k_max,
        ) = self._process_subsystem(subsystem_to_keep, subsystem_to_traceout)

        entropy = backend.cast(backend.convert_to_tensor(0.0), rdtypestr)
        eps = 1e-12

        for kA in range(k_min, k_max + 1):  # type: ignore
            kB = self._k - kA  # type: ignore
            mask = kA_all == kA
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            sA_sector = sA_all[indices]
            sB_sector = sB_all[indices]
            coeffs_sector = backend.gather1d(
                self._state, backend.convert_to_tensor(indices)
            )

            basisA = _get_subsystem_basis_static(nA, kA)
            basisB = _get_subsystem_basis_static(nB, kB)

            idxA = np.searchsorted(basisA, sA_sector)
            idxB = np.searchsorted(basisB, sB_sector)

            dimA = comb(nA, kA)
            dimB = comb(nB, kB)

            M = backend.zeros([dimA, dimB], dtype=dtypestr)
            scatter_indices = backend.convert_to_tensor(np.stack([idxA, idxB], axis=1))
            M = backend.scatter(M, scatter_indices, coeffs_sector)

            # Optimization: Use eigh on the smaller covariance matrix instead of SVD
            if dimA <= dimB:
                # rho_A = M @ M.H
                rho = backend.matmul(M, backend.conj(backend.transpose(M)))
            else:
                # rho_B = M.H @ M (same non-zero eigenvalues)
                rho = backend.matmul(backend.conj(backend.transpose(M)), M)

            lbd = backend.eigh(rho)[0]
            probs = backend.real(lbd)
            # Still use where for AD safety but now we avoid dynamic shape from nonzero
            entropy -= backend.sum(
                backend.where(probs > eps, probs * backend.log(probs + eps), 0.0)
            )

        return entropy

    def expectation_pss(
        self,
        ps_list: Sequence[Any],
        coefficients: Any,
    ) -> Tensor:
        """
        Compute the expectation value of a sum of Pauli strings in the U(1) subspace.

        :param ps_list: A sequence of Pauli strings. Each can be represented as a list of
            integers of length nqubits (0=I, 1=X, 2=Y, 3=Z) or a dict with 'x', 'y', 'z' keys.
        :type ps_list: Sequence[Any]
        :param coefficients: The coefficients for each Pauli string in the sum. Can be a list
            of floats or a backend tensor of coefficients.
        :type coefficients: Any
        :return: The expectation value of the sum of Pauli strings.
        :rtype: Tensor
        """
        nqubits = self._nqubits
        k = self._k
        assert k is not None

        # 1. Parse ps_list to a standard tuple of tuples for caching
        ps_tuple = _normalize_ps_list(nqubits, ps_list)

        key = (nqubits, k, ps_tuple)
        if key not in _u1_operator_cache:
            if len(_u1_operator_cache) >= 100:
                logger.warning(
                    "U1Operator cache size has reached %d. "
                    "This might indicate memory consumption issues if new operators are "
                    "continually created.",
                    len(_u1_operator_cache),
                )
            _u1_operator_cache[key] = U1Operator(nqubits, k, ps_tuple)

        op = _u1_operator_cache[key]
        state = self._state
        coefs = backend.cast(backend.convert_to_tensor(coefficients), dtype=dtypestr)

        total_energy = backend.cast(backend.convert_to_tensor(0.0), dtypestr)

        # A. Diagonal energy
        if len(op.diagonal_indices) > 0:
            coefs_diag = backend.gather1d(
                coefs, backend.convert_to_tensor(op.diagonal_indices)
            )
            diag_vecs = backend.cast(
                backend.convert_to_tensor(op.diagonal_vectors), dtype=dtypestr
            )
            coefs_diag_2d = backend.reshape(coefs_diag, [-1, 1])
            diag_vector_2d = backend.matmul(backend.transpose(diag_vecs), coefs_diag_2d)
            diag_vector = backend.reshape(diag_vector_2d, [-1])
            diagonal_energy = backend.tensordot(
                backend.conj(state), diag_vector * state, 1
            )
            total_energy = total_energy + diagonal_energy

        # B. Off-diagonal energy
        if len(op.off_diagonal_indices) > 0:
            coefs_off = backend.gather1d(
                coefs, backend.convert_to_tensor(op.off_diagonal_indices)
            )
            targets = backend.convert_to_tensor(op.off_diagonal_targets)
            phases = backend.cast(
                backend.convert_to_tensor(op.off_diagonal_phases), dtype=dtypestr
            )

            weighted_phases = phases * backend.reshape(coefs_off, [-1, 1])

            def term_expectation(target_idx: Tensor, phase_fact: Tensor) -> Tensor:
                target_state = backend.gather1d(state, target_idx)
                return backend.tensordot(
                    backend.conj(state), phase_fact * target_state, 1
                )

            vmapped_expect = backend.vmap(term_expectation, vectorized_argnums=(0, 1))
            off_diagonal_energy = backend.sum(vmapped_expect(targets, weighted_phases))
            total_energy = total_energy + off_diagonal_energy

        return backend.real(total_energy)


# Register gates via _meta_apply
U1Circuit._meta_apply()
