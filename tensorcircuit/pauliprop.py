"""
Pauli Propagation Engine
========================

This module implements a Pauli Propagation Engine (PPE) that tracks the evolution
of Pauli observables through a quantum circuit using a global k-local representation.

Key Features:
    - Tracks all k-local Pauli strings globally, avoiding subset boundary issues.
    - Uses precomputed Neighbor Maps for O(1) fiber lookups.
    - JAX JIT and AD compatible.
    - Naturally handles truncation of terms exceeding locality k.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence
import logging
import numpy as np

from . import gates
from .cons import backend
from .circuit import Circuit

logger = logging.getLogger(__name__)

Tensor = Any


class PauliPropagationEngine:
    """
    A Pauli Propagation Engine that tracks observables in the global k-local space.

    The state is represented as a flat vector of size $|P_k| = \sum_{i=0}^k \binom{N}{i} 3^i$.
    """

    def __init__(self, N: int, k: int) -> None:
        """
        Initialize the Pauli Propagation Engine.

        :param N: Total number of qubits.
        :type N: int
        :param k: Maximum locality of the Pauli strings to track.
        :type k: int
        """
        self.N = N
        self.k = k
        self._build_basis()
        self._build_neighbor_map()
        self._cache_z_indices()

        # Pauli matrices for PTM construction
        self.pauli_mats = backend.cast(
            backend.convert_to_tensor(
                np.array(
                    [
                        [[1, 0], [0, 1]],  # I
                        [[0, 1], [1, 0]],  # X
                        [[0, -1j], [1j, 0]],  # Y
                        [[1, 0], [0, -1]],  # Z
                    ],
                    dtype=np.complex64,
                )
            ),
            "complex64",
        )

    def _build_basis(self) -> None:
        """
        Build the canonical list of all k-local Pauli strings.
        A string is represented as a tuple of ( (q_idx, ...), (pauli_code, ...) )
        where codes are 1:X, 2:Y, 3:Z. Identity is implicitly handled.
        """
        # index 0 is always the identity string (empty tuple)
        self.basis: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = [((), ())]
        self.string_to_idx: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {
            ((), ()): 0
        }

        import itertools

        for loc in range(1, self.k + 1):
            for qubits in itertools.combinations(range(self.N), loc):
                for codes in itertools.product([1, 2, 3], repeat=loc):
                    s = (qubits, codes)
                    self.string_to_idx[s] = len(self.basis)
                    self.basis.append(s)

        self.dim = len(self.basis)
        logger.info(
            f"Initialized Pauli basis with size {self.dim} for N={self.N}, k={self.k}"
        )

    def _build_neighbor_map(self) -> None:
        """
        Precompute the neighbor map.
        neighbor_map[idx, q, code] = new_idx
        where code in {0:I, 1:X, 2:Y, 3:Z}.
        Indices that exceed locality k point to self.dim (a SINK index).
        """
        # We use uint32 for indices to save memory, or int32 if backend preferred
        # SINK index is self.dim, we add one extra row for it in the map
        self.neighbor_map_np = np.full(
            (self.dim + 1, self.N, 4), self.dim, dtype=np.int32
        )

        for i, (qubits, codes) in enumerate(self.basis):
            q_to_code = dict(zip(qubits, codes))

            for q in range(self.N):
                # We want to know what happens if we change the base at q
                # Existing code at q
                current_code = q_to_code.get(q, 0)

                for target_code in range(4):
                    if target_code == current_code:
                        self.neighbor_map_np[i, q, target_code] = i
                        continue

                    # Construct new string
                    new_q_to_code = q_to_code.copy()
                    if target_code == 0:
                        del new_q_to_code[q]
                    else:
                        new_q_to_code[q] = target_code

                    if len(new_q_to_code) > self.k:
                        self.neighbor_map_np[i, q, target_code] = self.dim  # SINK
                    else:
                        # Canonicalize
                        new_qubits = tuple(sorted(new_q_to_code.keys()))
                        new_codes = tuple(new_q_to_code[q_] for q_ in new_qubits)
                        new_s = (new_qubits, new_codes)
                        self.neighbor_map_np[i, q, target_code] = self.string_to_idx[
                            new_s
                        ]

        # SINK row already initialized to point to SINK for all operations
        # Convert to backend tensor
        self.neighbor_map = backend.convert_to_tensor(
            self.neighbor_map_np, dtype="int32"
        )

    def _cache_z_indices(self) -> None:
        """
        Precompute indices of Pauli strings that consist only of I and Z.
        """
        z_indices = []
        for i, (_, codes) in enumerate(self.basis):
            if all(c == 3 for c in codes):
                z_indices.append(i)
        self.z_indices_np = np.array(z_indices, dtype=np.int32)
        self.z_indices = backend.convert_to_tensor(self.z_indices_np, dtype="int32")

    def get_ptm_1q(self, u: Any) -> Any:
        u = backend.convert_to_tensor(u)
        u_dag = backend.conj(backend.transpose(u))
        sigmas = self.pauli_mats  # (4, 2, 2)
        tmp = backend.matmul(u_dag, sigmas)
        rot_sigmas = backend.matmul(tmp, u)
        res = backend.matmul(
            backend.reshape(sigmas, [4, 4]),
            backend.transpose(backend.reshape(rot_sigmas, [4, 4])),
        )
        m = 0.5 * res
        return backend.real(m)

    def get_ptm_2q(self, u: Any) -> Any:
        u = backend.convert_to_tensor(u)
        u_dag = backend.conj(backend.transpose(u))
        s1 = self.pauli_mats
        sigmas_2q = backend.reshape(
            backend.reshape(s1, [4, 1, 2, 1, 2, 1])
            * backend.reshape(s1, [1, 4, 1, 2, 1, 2]),
            [16, 4, 4],
        )
        tmp = backend.matmul(u_dag, sigmas_2q)
        rot_sigmas = backend.matmul(tmp, u)
        res = backend.matmul(
            backend.reshape(sigmas_2q, [16, 16]),
            backend.transpose(backend.reshape(rot_sigmas, [16, 16])),
        )
        m = 0.25 * res
        return backend.real(m)

    def get_initial_state(self, structures: Any, weights: Any) -> Any:
        """
        Initialize the state vector from Hamiltonian terms.
        """
        structures = np.array(structures)
        indices = []
        updates = []

        for i in range(len(weights)):
            term = structures[i]
            w = weights[i]
            qubits = tuple(np.where(term != 0)[0])
            if len(qubits) > self.k:
                continue
            codes = tuple(term[q] for q in qubits)
            s = (qubits, codes)
            if s in self.string_to_idx:
                indices.append([self.string_to_idx[s]])
                updates.append(w)

        # Flat state vector + 1 for SINK
        state = backend.zeros((self.dim + 1,), dtype="complex64")
        if len(indices) > 0:
            indices_tensor = backend.convert_to_tensor(
                np.array(indices, dtype=np.int32), dtype="int32"
            )
            updates_tensor = backend.cast(backend.stack(updates), "complex64")
            state = backend.scatter(state, indices_tensor, updates_tensor, mode="add")

        return state

    def expectation(self, state: Any) -> Any:
        """
        Sum coefficients of purely Z observables in the final state on |0...0>.
        """
        z_coeffs = backend.gather1d(state, self.z_indices)
        return backend.real(backend.sum(z_coeffs))

    def apply_gate(
        self, state: Any, gate_name: str, wires: Any, params: Any = None
    ) -> Any:
        gate_name = gate_name.lower()
        gate_func = getattr(gates, gate_name)

        if params is not None:
            if isinstance(params, dict):
                u = gate_func(**params).tensor
            else:
                try:
                    u = gate_func(params).tensor
                except TypeError:
                    u = gate_func(theta=params).tensor
        else:
            u = gate_func().tensor

        u = backend.convert_to_tensor(u)
        dim = 2 ** len(wires)
        u = backend.reshape(u, [dim, dim])

        if len(wires) == 1:
            ptm = self.get_ptm_1q(u)  # (4, 4)
            q = wires[0]

            # Fiber partitioning for qubit q
            repr_indices = np.where(
                self.neighbor_map_np[: self.dim, q, 0] == np.arange(self.dim)
            )[0]
            repr_idx_tensor = backend.convert_to_tensor(repr_indices, dtype="int32")

            # Neighbor indices for these representatives
            fiber_indices = backend.gather1d(
                self.neighbor_map[:, q, :], repr_idx_tensor
            )

            def fiber_update(indices: Any) -> Any:
                # gather values including SINK (which is always 0)
                vals = backend.gather1d(state, indices)
                new_vals = backend.tensordot(
                    backend.cast(ptm, vals.dtype), vals, ([1], [0])
                )
                return new_vals

            # Apply PTM to all fibers
            new_vals_all = backend.vmap(fiber_update)(fiber_indices)

            # Flatten and scatter back
            state = backend.scatter(
                state,
                backend.reshape(fiber_indices, [-1, 1]),
                backend.reshape(new_vals_all, [-1]),
                mode="update",
            )
            # Ensure SINK stays 0
            sink_idx = backend.convert_to_tensor([[self.dim]], dtype="int32")
            sink_val = backend.zeros([1], dtype=state.dtype)
            state = backend.scatter(state, sink_idx, sink_val, mode="update")

        elif len(wires) == 2:
            ptm = self.get_ptm_2q(u)  # (16, 16)
            q1, q2 = wires

            # Representatives have I at both q1 AND q2
            # neighbor_map is indexed by (global_idx, qubit, code)
            # string at glob_idx has I at q1 IF neighbor_map[glob_idx, q1, 0] == glob_idx
            mask = (self.neighbor_map_np[: self.dim, q1, 0] == np.arange(self.dim)) & (
                self.neighbor_map_np[: self.dim, q2, 0] == np.arange(self.dim)
            )
            repr_indices = np.where(mask)[0]
            repr_idx_tensor = backend.convert_to_tensor(repr_indices, dtype="int32")

            # To get 16 indices, we use neighbor_map twice
            # indices at q1 (4) -> for each, indices at q2 (4) -> 16 total
            fiber4_q1 = backend.gather1d(
                self.neighbor_map[:, q1, :], repr_idx_tensor
            )  # (N_repr, 4)

            # Vmap over the 4 columns for q2 lookup
            def lookup_q2(idx_q1: Any) -> Any:
                return backend.gather1d(self.neighbor_map[:, q2, :], idx_q1)

            fiber16 = backend.vmap(lookup_q2, vectorized_argnums=0)(
                backend.transpose(fiber4_q1)
            )  # (4, N_repr, 4)
            fiber16 = backend.transpose(fiber16, (1, 0, 2))  # (N_repr, 4, 4)
            fiber16 = backend.reshape(fiber16, [-1, 16])

            def fiber_update_16(indices: Any) -> Any:
                vals = backend.gather1d(state, indices)
                new_vals = backend.tensordot(
                    backend.cast(ptm, vals.dtype), vals, ([1], [0])
                )
                return new_vals

            new_vals_all = backend.vmap(fiber_update_16)(fiber16)
            state = backend.scatter(
                state,
                backend.reshape(fiber16, [-1, 1]),
                backend.reshape(new_vals_all, [-1]),
                mode="update",
            )
            sink_idx = backend.convert_to_tensor([[self.dim]], dtype="int32")
            sink_val = backend.zeros([1], dtype=state.dtype)
            state = backend.scatter(state, sink_idx, sink_val, mode="update")

        return state

    def compute_expectation_scan(
        self,
        ham_structures: Any,
        ham_weights: Any,
        layer_fn: Callable[..., None],
        params_batch: Any,
        extra_inputs: Optional[Sequence[Any]] = None,
    ) -> Any:
        state = self.get_initial_state(ham_structures, ham_weights)

        def scan_body(state: Any, scan_inputs: Any) -> Any:
            # scan_inputs might be a tuple or a single tensor depending on backend and extra_inputs
            if isinstance(scan_inputs, (tuple, list)):
                p_l = scan_inputs[0]
                args = scan_inputs[1:]
            else:
                p_l = scan_inputs
                args = ()

            if isinstance(state, (tuple, list)):
                state = state[0]

            c_layer = Circuit(self.N)
            layer_fn(c_layer, p_l, *args)
            ops = c_layer.to_qir()

            for op in reversed(ops):
                gate_name = op["name"]
                wires = op["index"]
                p_dict = op.get("parameters", {})
                param_val = p_dict if p_dict else None
                state = self.apply_gate(state, gate_name, wires, param_val)

            return state

        scan_inputs = (
            params_batch if extra_inputs is None else (params_batch, *extra_inputs)
        )
        # Workaround for general reverse
        if isinstance(scan_inputs, (tuple, list)):
            scan_inputs_rev = tuple([backend.reverse(x) for x in scan_inputs])
        else:
            scan_inputs_rev = backend.reverse(scan_inputs)

        # Prepare inputs for scan consistently as a tuple
        scan_inputs = (
            (params_batch,) if extra_inputs is None else (params_batch, *extra_inputs)
        )
        scan_inputs_rev = tuple([backend.reverse(x) for x in scan_inputs])

        if backend.name in ["jax", "tensorflow"]:
            final_state = backend.scan(scan_body, scan_inputs_rev, state)
        else:
            final_state = state
            # Manual scan for backends like NumPy to avoid abstract_backend inconsistencies
            for i in range(backend.shape_tuple(scan_inputs_rev[0])[0]):
                scan_inputs_i = tuple([x[i] for x in scan_inputs_rev])
                final_state = scan_body(final_state, scan_inputs_i)

        return self.expectation(final_state)


def pauli_propagation(
    c: Circuit,
    observable: Any,
    weights: Optional[Any] = None,
    k: int = 3,
) -> Any:
    N = c._nqubits
    pp = PauliPropagationEngine(N, k)

    if weights is not None:
        state = pp.get_initial_state(observable, weights)
    elif (
        isinstance(observable, list)
        and len(observable) > 0
        and isinstance(observable[0], tuple)
    ):
        map_char = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        num_terms = len(observable)
        structures = np.zeros((num_terms, N), dtype=int)
        weights_arr = np.zeros(num_terms, dtype=np.complex64)
        for i, (coeff, p_str) in enumerate(observable):
            weights_arr[i] = coeff
            for j, char in enumerate(p_str):
                if j < N:
                    structures[i, j] = map_char[char]
        state = pp.get_initial_state(structures, weights_arr)
    elif isinstance(observable, tuple) and len(observable) == 2:
        state = pp.get_initial_state(observable[0], observable[1])
    else:
        raise ValueError("observable format not supported")

    ops = c.to_qir()
    for op in reversed(ops):
        gate_name = op["name"]
        wires = op["index"]
        params_dict = op.get("parameters", {})
        param_val = params_dict if params_dict else None
        state = pp.apply_gate(state, gate_name, wires, param_val)

    return pp.expectation(state)


class SparsePauliPropagationEngine:
    """
    A Truly Sparse Pauli Propagation Engine that tracks Pauli strings
    using bitpacked integers. No combinatorial basis is precomputed,
    making it suitable for hundreds of qubits.
    """

    def __init__(self, N: int, k: int, buffer_size: int = 2000) -> None:
        self.N = N
        self.k = k
        self.buffer_size = buffer_size
        # Number of int64 words to track 2 bits per qubit
        self.W = (N + 31) // 32

        self.pauli_mats = backend.convert_to_tensor(
            np.array(
                [
                    [[1, 0], [0, 1]],
                    [[0, 1], [1, 0]],
                    [[0, -1j], [1j, 0]],
                    [[1, 0], [0, -1]],
                ]
            )
        )

    def prepare_initial_state(self, structures: Any, weights: Any) -> Any:
        K = backend
        structures = K.convert_to_tensor(structures, dtype="int32")
        weights = K.convert_to_tensor(weights, dtype="complex64")
        M = K.shape_tuple(structures)[0]

        codes = []
        for w in range(self.W):
            word = K.zeros((M,), dtype="int64")
            for i in range(32):
                q = w * 32 + i
                if q < self.N:
                    op = K.cast(structures[:, q], "int64")
                    word = word | (op << (2 * i))
            codes.append(word)
        codes = K.stack(codes, axis=1)  # (M, W)

        num_pad = self.buffer_size - M
        if num_pad > 0:
            codes = K.concat([codes, K.zeros((num_pad, self.W), dtype="int64") - 1])
            weights = K.concat([weights, K.zeros((num_pad,), dtype="complex64")])
        else:
            codes = codes[: self.buffer_size]
            weights = weights[: self.buffer_size]

        return (codes, weights)

    def get_initial_state(self, structures: Any, weights: Any) -> Any:
        return self.prepare_initial_state(structures, weights)

    def expectation(self, state: Any) -> Any:
        codes, coeffs = state
        K = backend
        is_z = K.ones(K.shape_tuple(codes)[0], dtype="bool")
        m55 = 0x5555555555555555
        for w in range(self.W):
            word = codes[:, w]
            low_bits = word & m55
            high_bits = (word >> 1) & m55
            word_is_z = (low_bits == high_bits) & (word != -1)
            is_z = is_z & word_is_z
        return K.real(K.sum(K.where(is_z, coeffs, 0.0)))

    def _get_weight(self, codes: Any) -> Any:
        K = backend
        total_weight = K.zeros(K.shape_tuple(codes)[0], dtype="int32")
        m55 = 0x5555555555555555
        for w in range(self.W):
            word = codes[:, w]
            non_iden = (word & m55) | ((word >> 1) & m55)
            # Use hardware-accelerated popcount if available via backend
            total_weight += K.cast(K.popc(non_iden), "int32")
        return total_weight

    def _aggregate_and_truncate(self, codes: Any, coeffs: Any) -> Any:
        K = backend
        M = K.shape_tuple(coeffs)[0]
        # Sort by codes to group duplicates
        sort_idx = K.lexsort([codes[:, i] for i in range(self.W - 1, -1, -1)])
        codes_s = codes[sort_idx]
        coeffs_s = coeffs[sort_idx]

        # Identify boundaries of unique codes
        is_diff = K.zeros((M - 1,), dtype="bool")
        for w in range(self.W):
            is_diff = is_diff | (codes_s[1:, w] != codes_s[:-1, w])
        is_diff = K.concat([K.convert_to_tensor([True], dtype="bool"), is_diff])

        # Aggregate coefficients
        seg_ids = K.cumsum(K.cast(is_diff, "int32")) - 1
        unique_coeffs = K.zeros((M,), dtype="complex64")
        unique_coeffs = K.scatter(
            unique_coeffs, K.reshape(seg_ids, [-1, 1]), coeffs_s, mode="add"
        )

        # Use aggregated coefficients at boundary positions for truncation
        agg_coeffs_at_i = unique_coeffs[seg_ids]
        magnitudes = K.where(is_diff, K.abs(agg_coeffs_at_i), -1.0)

        # Keep top buffer_size terms
        _, top_idx = K.top_k(magnitudes, self.buffer_size)

        final_codes = codes_s[top_idx]
        final_coeffs = agg_coeffs_at_i[top_idx]

        # Nullify inactive/empty slots
        is_active = (K.abs(final_coeffs) > 1e-12) & (final_codes[:, 0] != -1)
        final_codes = K.where(K.reshape(is_active, [-1, 1]), final_codes, -1)
        final_coeffs = K.where(is_active, final_coeffs, 0.0)
        return (final_codes, final_coeffs)

    def apply_gate(
        self, state: Any, gate_name: str, wires: Any, params: Any = None
    ) -> Any:
        indices, coeffs = state
        K = backend
        ptm = self._get_ptm(gate_name, wires, params)
        ptm_complex = K.cast(ptm, "complex64")

        if len(wires) == 1:
            q = wires[0]
            w_idx, b_pos = q // 32, (q % 32) * 2

            # 1. Extract current ops (S,)
            curr_ops = K.where(
                indices[:, w_idx] != -1, (indices[:, w_idx] >> b_pos) & 3, 0
            )

            # 2. Get multipliers (S, 4)
            ptm_t = K.transpose(ptm_complex)
            multipliers = K.gather1d(ptm_t, curr_ops)

            # 3. New coeffs (S*4,)
            flat_coeffs = K.reshape(coeffs[:, None] * multipliers, [-1])

            # 4. New indices (S*4, W)
            mask = ~(K.convert_to_tensor(3, dtype="int64") << b_pos)
            t_ops = K.convert_to_tensor(np.arange(4), dtype="int64")
            new_words = (indices[:, w_idx, None] & mask) | (t_ops[None, :] << b_pos)
            flat_new_words = K.reshape(new_words, [-1])

            cols = []
            for i in range(self.W):
                # Use K.repeat for correct alignment during expansion [T0, T1] -> [T0, T0, T0, T0, T1, T1, T1, T1]
                col = K.where(
                    i == w_idx, flat_new_words, K.repeat(indices[:, i], 4, axis=0)
                )
                cols.append(col)
            flat_indices = K.stack(cols, axis=1)

            if self.k < self.N:
                w = self._get_weight(flat_indices)
                flat_coeffs = K.where(w <= self.k, flat_coeffs, 0.0)

            return self._aggregate_and_truncate(flat_indices, flat_coeffs)

        elif len(wires) == 2:
            q1, q2 = wires
            w1, b1 = q1 // 32, (q1 % 32) * 2
            w2, b2 = q2 // 32, (q2 % 32) * 2

            o1 = K.where(indices[:, w1] != -1, (indices[:, w1] >> b1) & 3, 0)
            o2 = K.where(indices[:, w2] != -1, (indices[:, w2] >> b2) & 3, 0)
            curr_ops12 = o1 * 4 + o2

            ptm_t = K.transpose(ptm_complex)
            multipliers = K.gather1d(ptm_t, curr_ops12)
            flat_coeffs = K.reshape(coeffs[:, None] * multipliers, [-1])

            t12 = K.convert_to_tensor(np.arange(16), dtype="int64")
            t1, t2 = t12 // 4, t12 % 4
            mask1, mask2 = ~(K.convert_to_tensor(3, dtype="int64") << b1), ~(
                K.convert_to_tensor(3, dtype="int64") << b2
            )

            # All possible target words regardless of w1==w2
            new_words1 = (indices[:, w1, None] & mask1) | (t1[None, :] << b1)
            new_words2 = (indices[:, w2, None] & mask2) | (t2[None, :] << b2)
            new_words_both = (
                (indices[:, w1, None] & mask1 & mask2)
                | (t1[None, :] << b1)
                | (t2[None, :] << b2)
            )

            flat_w1 = K.reshape(new_words1, [-1])
            flat_w2 = K.reshape(new_words2, [-1])
            flat_both = K.reshape(new_words_both, [-1])

            cols = []
            for i in range(self.W):
                # Use K.repeat for correct alignment [T0, T1] -> [T0... (16 times), T1... (16 times)]
                target_word = K.repeat(indices[:, i], 16, axis=0)
                # Correctly handle overlapping vs distinct word indices
                target_word = K.where(
                    i == w1, K.where(w1 == w2, flat_both, flat_w1), target_word
                )
                target_word = K.where((i == w2) & (w1 != w2), flat_w2, target_word)
                cols.append(target_word)
            flat_indices = K.stack(cols, axis=1)

            if self.k < self.N:
                w = self._get_weight(flat_indices)
                flat_coeffs = K.where(w <= self.k, flat_coeffs, 0.0)

            return self._aggregate_and_truncate(flat_indices, flat_coeffs)
        return state

    def _get_ptm(self, gate_name: str, wires: Sequence[int], params: Any) -> Any:
        from . import gates

        K = backend
        gate_func = getattr(gates, gate_name.lower())
        if params is None:
            params = {}
        if isinstance(params, dict):
            u = gate_func(**params).tensor
        else:
            try:
                u = gate_func(params).tensor
            except TypeError:
                u = gate_func(theta=params).tensor
        u = K.convert_to_tensor(u)
        u = K.reshape(u, [2 ** len(wires), 2 ** len(wires)])
        u_dag = K.conj(K.transpose(u))
        s1 = self.pauli_mats
        if len(wires) == 1:
            rot = K.matmul(K.matmul(u_dag, s1), u)
            res = K.matmul(K.reshape(s1, [4, 4]), K.transpose(K.reshape(rot, [4, 4])))
            return K.real(0.5 * res)
        else:
            s2 = K.reshape(
                K.reshape(s1, [4, 1, 2, 1, 2, 1]) * K.reshape(s1, [1, 4, 1, 2, 1, 2]),
                [16, 4, 4],
            )
            rot = K.matmul(K.matmul(u_dag, s2), u)
            res = K.matmul(
                K.reshape(s2, [16, 16]), K.transpose(K.reshape(rot, [16, 16]))
            )
            return K.real(0.25 * res)

    def compute_expectation_scan(
        self,
        structures: Any,
        weights: Any,
        layer: Callable[[Any, Any], None],
        params: Any,
    ) -> Any:
        K = backend
        state = self.get_initial_state(structures, weights)

        def step(s: Any, p: Any) -> Any:
            c = Circuit(self.N)
            layer(c, p)
            for op in reversed(c.to_qir()):
                s = self.apply_gate(s, op["name"], op["index"], op.get("parameters"))
            return s

        final_state = K.scan(step, params[::-1], state)
        return self.expectation(final_state)
