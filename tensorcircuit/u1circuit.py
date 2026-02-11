"""
Circuit class for U(1) conserving circuits.
"""

from functools import partial
from itertools import combinations
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .abstractcircuit import AbstractCircuit
from .cons import backend, dtypestr, rdtypestr, idtypestr
from .quantum import sample2all
from .utils import arg_alias

Tensor = Any


class U1Circuit(AbstractCircuit):
    """
    Circuit class for U(1) conserving circuits (particle number conservation).
    Note: Currently only supports nqubits < 64 due to the use of 64-bit integer
    bitmasks for basis state representation.
    """

    def __init__(
        self,
        nqubits: int,
        k: Optional[int] = None,
        filled: Optional[Sequence[int]] = None,
        inputs: Optional[Any] = None,
    ) -> None:
        self._nqubits = nqubits
        if nqubits >= 64:
            raise ValueError(
                f"U1Circuit only supports nqubits < 64, but got {nqubits}. "
            )

        # Infer k from filled if not provided
        if k is None and filled is None:
            raise ValueError("Either 'k' or 'filled' must be provided")
        if k is None:
            k = len(filled)  # type: ignore

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
        self._dim = comb(nqubits, k)
        self._basis: List[int] = []
        for filled_combo in combinations(range(nqubits), k):
            state = 0
            for bit in filled_combo:
                state |= 1 << (nqubits - 1 - bit)  # Reversed bit ordering
            self._basis.append(state)
        self._basis = sorted(self._basis)

        # Convert basis to backend tensor for faster lookup
        self._basis_tensor = backend.cast(
            backend.convert_to_tensor(self._basis), dtype=idtypestr
        )

        if inputs is not None:
            self._state = backend.cast(inputs, dtypestr)
        else:
            if filled is None:
                # Default: fill lowest-indexed qubits (highest bits in state integer)
                filled_state = ((1 << k) - 1) << (nqubits - k)
            else:
                filled_state = 0
                for i in filled:
                    filled_state |= 1 << (nqubits - 1 - i)  # Reversed bit ordering

            # Find the index in our basis - JIT FRIENDLY
            # Use searchsorted and one_hot to stay within the graph
            fs_tensor = backend.cast(
                backend.convert_to_tensor([filled_state]), dtype=idtypestr
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

    def _apply_rz(self, i: int, theta: Any) -> None:
        """Apply RZ(theta) on qubit i: |0> -> |0>, |1> -> e^{-i*theta/2}|1>."""
        bp = self._bit_position(i)
        bit_val = backend.right_shift(
            backend.bitwise_and(self._basis_tensor, (1 << bp)), bp
        )
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
        bpi, bpj = self._bit_position(i), self._bit_position(j)
        zi = backend.right_shift(
            backend.bitwise_and(self._basis_tensor, (1 << bpi)), bpi
        )
        zj = backend.right_shift(
            backend.bitwise_and(self._basis_tensor, (1 << bpj)), bpj
        )
        zz = 1 - 2 * backend.cast(backend.bitwise_xor(zi, zj), dtype=dtypestr)
        phases = backend.cast(-0.5 * theta * zz, dtype=dtypestr)
        self._state = self._state * backend.exp(1j * phases)

    def _apply_cz(self, i: int, j: int) -> None:
        """Apply CZ gate: |11> -> -|11>, others unchanged."""
        bpi, bpj = self._bit_position(i), self._bit_position(j)
        mask = (1 << bpi) | (1 << bpj)
        both_on = backend.cast(
            backend.bitwise_and(self._basis_tensor, mask) == mask, dtype=dtypestr
        )
        self._state = self._state * (1.0 - 2.0 * both_on)

    def _apply_cphase(self, i: int, j: int, theta: Any) -> None:
        """Apply controlled-phase: |11> -> e^{i*theta}|11>, others unchanged."""
        bpi, bpj = self._bit_position(i), self._bit_position(j)
        mask = (1 << bpi) | (1 << bpj)
        both_on = backend.cast(
            backend.bitwise_and(self._basis_tensor, mask) == mask, dtype=dtypestr
        )
        self._state = self._state * (
            1.0
            + (
                backend.exp(
                    1j * backend.cast(backend.convert_to_tensor(theta), dtypestr)
                )
                - 1.0
            )
            * both_on
        )

    def _apply_swap(self, i: int, j: int) -> None:
        """Apply SWAP gate: exchange qubits i and j."""
        bpi, bpj = self._bit_position(i), self._bit_position(j)
        mask_i = 1 << bpi
        mask_j = 1 << bpj
        bi = backend.right_shift(backend.bitwise_and(self._basis_tensor, mask_i), bpi)
        bj = backend.right_shift(backend.bitwise_and(self._basis_tensor, mask_j), bpj)
        diff = backend.bitwise_xor(bi, bj)
        mask_swap = (1 << bpi) | (1 << bpj)
        new_basis = backend.bitwise_xor(self._basis_tensor, diff * mask_swap)
        indices = backend.searchsorted(self._basis_tensor, new_basis)
        self._state = backend.gather1d(self._state, indices)

    def _apply_iswap(self, i: int, j: int, theta: Any) -> None:
        """Apply iSWAP(theta): parameterized iSWAP with angle theta*pi/2."""
        bpi, bpj = self._bit_position(i), self._bit_position(j)
        mask_i = 1 << bpi
        mask_j = 1 << bpj
        bi = backend.right_shift(backend.bitwise_and(self._basis_tensor, mask_i), bpi)
        bj = backend.right_shift(backend.bitwise_and(self._basis_tensor, mask_j), bpj)
        diff = backend.bitwise_xor(bi, bj)
        mask_swap = (1 << bpi) | (1 << bpj)
        new_basis = backend.bitwise_xor(self._basis_tensor, diff * mask_swap)
        indices = backend.searchsorted(self._basis_tensor, new_basis)

        theta_c = backend.cast(backend.convert_to_tensor(theta), dtypestr)
        cos_t = backend.cos(theta_c * np.pi / 2)
        isin_t = 1j * backend.sin(theta_c * np.pi / 2)
        diff_f = backend.cast(diff, dtypestr)
        swapped_state = backend.gather1d(self._state, indices)

        self._state = (1.0 - diff_f) * self._state + diff_f * (
            cos_t * self._state + isin_t * swapped_state
        )

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

        # Extract parameters: _meta_apply puts them in ir_dict['parameters']
        # Also check kwargs for direct calls
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
        else:
            raise ValueError(
                f"Gate {name} not implemented in U1Circuit. "
                "Supported: rz, rzz, cz, cphase, swap, iswap."
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
        bp = self._bit_position(i)
        bit_i = backend.cast(
            backend.right_shift(backend.bitwise_and(self._basis_tensor, (1 << bp)), bp),
            dtype=rdtypestr,
        )
        z_vals = 1.0 - 2.0 * bit_i
        return backend.sum(
            backend.cast(backend.abs(self._state) ** 2, rdtypestr) * z_vals
        )

    def expectation(self, *ops: Any, **kws: Any) -> Tensor:
        """
        Compute expectation value of operators.

        Currently only supports single Z operator. For Pauli strings, use expectation_ps.

        :param ops: Operator specification as (gate_tensor, [qubit_indices])
        :return: Expectation value
        """
        if len(ops) == 1 and isinstance(ops[0], tuple) and len(ops[0]) == 2:
            op, idx = ops[0]
            op_np = backend.numpy(op)
            if np.allclose(op_np, [[1, 0], [0, -1]]):
                return self.expectation_z(idx[0])

        raise NotImplementedError(
            "General expectation not yet implemented for U1Circuit. Use expectation_ps."
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

        x = x or []
        y = y or []
        z = z or []

        z_factor: Any = 1.0
        for idx in z:
            bp = self._bit_position(idx)
            bit = backend.cast(
                backend.right_shift(
                    backend.bitwise_and(self._basis_tensor, (1 << bp)), bp
                ),
                dtype=dtypestr,
            )
            z_factor = z_factor * (1.0 - 2.0 * bit)

        all_xy = list(x) + list(y)
        if not all_xy:
            return backend.sum(
                backend.conj(self._state)
                * self._state
                * backend.cast(z_factor, dtypestr)
            )

        swap_mask = 0
        for idx in all_xy:
            swap_mask |= 1 << self._bit_position(idx)

        new_basis = backend.bitwise_xor(self._basis_tensor, swap_mask)

        # Check if weight preserved (U(1) symmetry ensures this)
        indices = backend.searchsorted(self._basis_tensor, new_basis)
        # Clip indices for safe gather
        safe_indices = backend.where(indices >= self._dim, self._dim - 1, indices)
        valid = (indices < self._dim) & (
            backend.gather1d(self._basis_tensor, safe_indices) == new_basis
        )

        phase = backend.cast(backend.convert_to_tensor(z_factor), dtypestr)
        for idx in y:
            bp = self._bit_position(idx)
            # Y phase depends on the bit value in the KET (target) state
            # Y|0> = i|1>, Y|1> = -i|0>
            # Read bit from new_basis (the ket side, before Y flips it)
            bit = backend.cast(
                backend.right_shift(backend.bitwise_and(new_basis, (1 << bp)), bp),
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
            backend.cast(self._basis_tensor, idtypestr), [self._dim, 1]
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
        ch = backend.cast(full_indices, idtypestr)
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


# Register gates via _meta_apply
U1Circuit._meta_apply()
