"""
Pauli Propagation Engine
========================

This module implements a Pauli Propagation Engine (PPE) that tracks the evolution
of Pauli observables through a quantum circuit.

Key Features:
    - Tracks k-local Pauli strings across all subsets of qubits.
    - Uses Pauli Transfer Matrices (PTMs) to propagate observables.
    - Supports exact evolution for k >= N.
    - Implements a "light-cone" optimization to drop Pauli strings that
      propagate outside their k-local support.
    - Provides an interface for variational algorithms (e.g., VQE/QAOA).

Usage Example:
    >>> import tensorcircuit as tc
    >>> from tensorcircuit.pauliprop import PauliPropagationEngine
    >>> N = 4
    >>> k = 2
    >>> pp = PauliPropagationEngine(N, k)
    >>> # H = 1.0 * Z0 Z1 + 0.5 * X1 X2
    >>> structures = np.array([[3, 3, 0, 0], [0, 1, 1, 0]])
    >>> weights = np.array([1.0, 0.5])
    >>> state = pp.get_initial_state(structures, weights)
    >>> # Apply CNOT(0, 1)
    >>> state = pp.apply_gate(state, "cnot", (0, 1))
    >>> print(pp.expectation(state))  # Expected value of H
"""

from itertools import combinations
from typing import Any, Callable, Optional, Sequence, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore

import tensorcircuit as tc

K = tc.backend

Tensor = Any

# TODO(@refraction-ray): backend agnostic implementation


class PauliPropagationEngine:
    """
    A Pauli Propagation Engine that tracks observables projected onto
    all possible k-local subspaces.

    The engine maintains a state of shape (num_subsets, 4^k), where each row
    corresponds to a k-qubit subset and contains the coefficients of the
    4^k Pauli strings supported on that subset.
    """

    def __init__(self, N: int, k: int) -> None:
        """
        Initialize the Pauli Propagation Engine.

        :param N: Total number of qubits.
        :type N: int
        :param k: Maximum locality of the Pauli strings to track.
        :type k: int
        """
        if jax is None:
            raise ImportError(
                "JAX is required for PauliPropagationEngine. "
                "Please install jax and jaxlib."
            )
        self.N = N
        self.k = k
        self.subsets = list(combinations(range(N), k))
        self.num_subsets = len(self.subsets)

        self.subset_arr = jnp.array(self.subsets, dtype=jnp.int32)

        # Basis matrices for PTM construction
        self.pauli_mats = jnp.array(
            [
                [[1, 0], [0, 1]],  # I
                [[0, 1], [1, 0]],  # X
                [[0, -1j], [1j, 0]],  # Y
                [[1, 0], [0, -1]],  # Z
            ],
            dtype=jnp.complex64,
        )
        # Indices corresponding to Pauli strings with only I and Z
        valid_indices = []
        for i in range(4**self.k):
            # Check digits
            temp = i
            is_valid = True
            for _ in range(self.k):
                digit = temp % 4
                if digit == 1 or digit == 2:  # X or Y
                    is_valid = False
                    break
                temp //= 4
            if is_valid:
                valid_indices.append(i)

        self.valid_indices = jnp.array(valid_indices, dtype=jnp.int32)

    def get_ptm_1q(self, u: Any) -> Any:
        """
        Compute 4x4 Pauli Transfer Matrix for a 1-qubit unitary U.
        R_ij = 0.5 * Tr(sigma_i U^dag sigma_j U)
        """
        u_dag = jnp.conjugate(u.T)

        # shape: (4, 2, 2)
        sigmas = self.pauli_mats

        # U^dag sigma_j U
        # (4, 2, 2)
        rot_sigmas = jnp.einsum("ab,jbc,cd->jad", u_dag, sigmas, u)

        # Tr(sigma_i * rot_sigmas)
        # (4, 4)
        m = 0.5 * jnp.einsum("iab,jba->ij", sigmas, rot_sigmas)
        return m.real  # PTM should be real for CPTP maps on Hermitian basis

    def get_ptm_2q(self, u: Any) -> Any:
        """
        Compute 16x16 Pauli Transfer Matrix for a 2-qubit unitary U.
        """
        u_dag = jnp.conjugate(u.T)

        # Tensor product of Paulis: (16, 4, 4)
        s1 = self.pauli_mats  # (4, 2, 2)
        # Kronecker product for basis
        # I_i x I_j -> index 4*i + j
        # shape (16, 4, 4)
        sigmas_2q = jnp.kron(s1[:, None, :, :], s1[None, :, :, :]).reshape(16, 4, 4)

        rot_sigmas = jnp.einsum("ab,jbc,cd->jad", u_dag, sigmas_2q, u)

        # Normalization factor for 2Q is 1/4
        m = 0.25 * jnp.einsum("iab,jba->ij", sigmas_2q, rot_sigmas)
        return m.real

    def get_initial_state(self, structures: Any, weights: Any) -> Any:
        """
        Initialize the propagation state from a Hamiltonian in TC internal format.

        :param structures: (M, N) integer array. 0:I, 1:X, 2:Y, 3:Z.
        :type structures: np.ndarray
        :param weights: (M,) float/complex array.
        :type weights: np.ndarray
        :return: Initial state tensor.
        :rtype: jnp.ndarray
        """
        # Ensure numpy for processing structures (must be static)
        structures = np.array(structures)

        # If weights is a tracer/JAX array, we use JAX-native accumulation
        # to avoid TracerArrayConversionError.
        state = jnp.zeros((self.num_subsets, 4**self.k), dtype=jnp.complex64)

        for i in range(len(weights)):
            term = structures[i]
            w = weights[i]

            # Identify non-identity qubits
            qubits = np.where(term != 0)[0]

            if len(qubits) > self.k:
                continue

            q_set = set(qubits)

            # Find target subset
            target_idx = -1
            for s_i, sub in enumerate(self.subsets):
                if q_set.issubset(sub):
                    target_idx = s_i
                    break

            if target_idx != -1:
                subset = self.subsets[target_idx]
                flat_idx = 0
                for local_pos, global_q in enumerate(subset):
                    code = term[global_q]
                    flat_idx += code * (4 ** (self.k - 1 - local_pos))

                state = state.at[target_idx, flat_idx].add(w)

        return state

    def expectation(self, state: Any) -> Any:
        # Sum coefficients at valid indices (pure Z/I terms)
        total = jnp.sum(state[:, self.valid_indices])
        return jnp.real(total).real

    def _apply_1q_kernel(self, state_subset: Any, ptm: Any, local_idx: Any) -> Any:
        """
        Apply 1Q PTM to a specific local axis of the state tensor (flattened).
        Dispatches to static versions using switch.
        """
        branches = []
        for i in range(self.k):

            def branch_fn(s: Any = state_subset, p: Any = ptm, idx: int = i) -> Any:
                return self._apply_1q_kernel_static(s, p, idx)

            branches.append(branch_fn)

        return jax.lax.switch(local_idx, branches)

    def _apply_1q_kernel_static(
        self, state_subset: Any, ptm: Any, static_idx: Any
    ) -> Any:
        shape = (4,) * self.k
        tensor = state_subset.reshape(shape)
        # tensordot axis 1 of PTM with static_idx of tensor
        new_tensor = jnp.tensordot(ptm, tensor, axes=([1], [static_idx]))

        # Permutation to restore order
        # tensordot puts new axis at 0
        perm = (
            list(range(1, static_idx + 1)) + [0] + list(range(static_idx + 1, self.k))
        )
        new_tensor = jnp.transpose(new_tensor, perm)
        return new_tensor.reshape(4**self.k)

    def _apply_2q_kernel(
        self, state_subset: Any, ptm: Any, idx1: Any, idx2: Any
    ) -> Any:
        """
        Apply 2Q PTM to local axes idx1, idx2.
        Dispatches via switch on flattened index (idx1 * k + idx2).
        """
        flat_idx = idx1 * self.k + idx2

        branches = []
        for i in range(self.k * self.k):
            # Decode i -> r, c
            r = i // self.k
            c = i % self.k

            if r == c:
                # Diagonal case: invalid for contraction (same axis twice).
                # Unreachable for valid distinct-wire gates.
                # Return dummy Identity.
                def branch_fn(
                    s: Any = state_subset,
                    p: Any = ptm,
                    r_static: Any = r,
                    c_static: Any = c,
                ) -> Any:
                    return s.reshape(4**self.k)

            else:

                def branch_fn(
                    s: Any = state_subset,
                    p: Any = ptm,
                    r_static: Any = r,
                    c_static: Any = c,
                ) -> Any:
                    return self._apply_2q_kernel_static(s, p, r_static, c_static)

            branches.append(branch_fn)

        return jax.lax.switch(flat_idx, branches)

    def _apply_2q_kernel_static(
        self, state_subset: Any, ptm: Any, idx1: Any, idx2: Any
    ) -> Any:
        shape = (4,) * self.k
        tensor = state_subset.reshape(shape)

        ptm_reshaped = ptm.reshape(4, 4, 4, 4)

        # Contract
        new_tensor = jnp.tensordot(ptm_reshaped, tensor, axes=([2, 3], [idx1, idx2]))

        # Permutation
        # new_tensor axes: (out1, out2, rest...)
        # We want mappings: 0->idx1, 1->idx2.

        target_perm = [0] * self.k
        target_perm[idx1] = 0
        target_perm[idx2] = 1

        current_rest_ptr = 2
        for i in range(self.k):
            if i != idx1 and i != idx2:
                target_perm[i] = current_rest_ptr
                current_rest_ptr += 1

        new_tensor = jnp.transpose(new_tensor, target_perm)
        return new_tensor.reshape(4**self.k)

    def _project_2q_boundary(
        self, state_subset: Any, ptm: Any, local_idx: Any, is_q1_in: Any
    ) -> Any:
        """
        Apply boundary update.
        PTM is 16x16 (q1, q2).
        If is_q1_in=True: q1 is in subset (at local_idx), q2 is out.
        We take submatrix M s.t. q2 input=I, q2 output=I.
        Indices: q1*4 + q2. I is 0.
        So we want Input indices where q2=0: 0, 4, 8, 12.
        Output indices where q2=0: 0, 4, 8, 12.
        This forms a 4x4 matrix.
        """
        # Reshape PTM to (4(out1), 4(out2), 4(in1), 4(in2))
        ptm_4 = ptm.reshape(4, 4, 4, 4)

        reduced_ptm = jax.lax.cond(
            is_q1_in, lambda _: ptm_4[:, 0, :, 0], lambda _: ptm_4[0, :, 0, :], None
        )

        return self._apply_1q_kernel(state_subset, reduced_ptm, local_idx)

    def apply_gate(
        self, state: Any, gate_name: str, wires: Any, params: Any = None
    ) -> Any:
        # 1. Get PTM
        # Use tc.gates to get matrix directly, supporting JAX types
        gate_name = gate_name.lower()
        gate_func = getattr(tc.gates, gate_name)

        if params is not None:
            if isinstance(params, dict):
                u = gate_func(**params).tensor
            else:
                try:
                    u = gate_func(params).tensor
                except TypeError:
                    # Fallback for gates requiring named args (e.g. rzz)
                    u = gate_func(theta=params).tensor
        else:
            u = gate_func().tensor

        # Ensure matrix shape
        dim = 2 ** len(wires)
        u = u.reshape(dim, dim)

        if len(wires) == 1:
            ptm = self.get_ptm_1q(u)
            q = wires[0]

            # Find subsets containing q
            # Use masking for vectorization
            # self.subset_arr shape (num_subsets, k)

            # Mask: (num_subsets,) boolean
            has_q = jnp.any(self.subset_arr == q, axis=1)

            # Local index: (num_subsets,) int. -1 if not present.
            # argmax returns first True index, which is what we want if q is present.
            local_indices = jnp.argmax(self.subset_arr == q, axis=1)

            # Define batched update function
            def update_fn_1q(s: Any, idx: Any, present: Any) -> Any:
                # if present, apply, else identity
                res = self._apply_1q_kernel(s, ptm, idx)
                return jnp.where(present, res, s)

            state = jax.vmap(update_fn_1q)(state, local_indices, has_q)  # type: ignore

        elif len(wires) == 2:
            ptm = self.get_ptm_2q(u)
            q1, q2 = wires

            # Identify subsets
            # Mask1: contains q1. Mask2: contains q2.
            mask1 = jnp.any(self.subset_arr == q1, axis=1)
            mask2 = jnp.any(self.subset_arr == q2, axis=1)

            idx1 = jnp.argmax(self.subset_arr == q1, axis=1)
            idx2 = jnp.argmax(self.subset_arr == q2, axis=1)

            # Case 1: Both in (mask1 & mask2) -> Full 2Q update
            # Case 2: Only q1 in (mask1 & !mask2) -> Boundary update q1
            # Case 3: Only q2 in (!mask1 & mask2) -> Boundary update q2

            def update_fn_2q(s: Any, m1: Any, m2: Any, i1: Any, i2: Any) -> Any:
                # Branchless logic using where

                # Case Both
                s_both = self._apply_2q_kernel(s, ptm, i1, i2)

                # Case q1 only
                s_q1 = self._project_2q_boundary(s, ptm, i1, True)

                # Case q2 only
                s_q2 = self._project_2q_boundary(s, ptm, i2, False)

                # Select
                # If m1 and m2: s_both
                # Elif m1: s_q1
                # Elif m2: s_q2
                # Else: s

                res = jnp.where(
                    m1 & m2, s_both, jnp.where(m1, s_q1, jnp.where(m2, s_q2, s))
                )
                return res

            state = jax.vmap(update_fn_2q)(state, mask1, mask2, idx1, idx2)  # type: ignore

        return state

    def compute_expectation_scan(
        self,
        ham_structures: Any,
        ham_weights: Any,
        layer_fn: Callable[..., None],
        params_batch: Any,
        extra_inputs: Optional[Sequence[Any]] = None,
    ) -> Any:
        """
        Compute expectation using scanning over layers in the Heisenberg picture.

        :param ham_structures: (M, N) integer array of Hamiltonian structures.
        :type ham_structures: np.ndarray
        :param ham_weights: (M,) array of Hamiltonian weights.
        :type ham_weights: np.ndarray
        :param layer_fn: Function that constructs one layer of the circuit.
        :type layer_fn: Callable[..., None]
        :param params_batch: Tensor of shape (layers, params_per_layer).
        :type params_batch: jnp.ndarray
        :param extra_inputs: Optional extra inputs to scan over or broadcast.
        :type extra_inputs: Optional[Sequence[Any]]
        :return: Scalar expectation value.
        :rtype: jnp.ndarray
        """
        # Initialize
        state = self.get_initial_state(ham_structures, ham_weights)

        # Define scan body
        def scan_body(state: Any, scan_inputs: Any) -> Tuple[Any, None]:
            if extra_inputs is None:
                p_l = scan_inputs
                args = ()
            else:
                p_l = scan_inputs[0]
                args = scan_inputs[1:]

            # 1. Trace the circuit for this layer to get operations
            # We assume structure is static, so we can trace once or every time (JIT compliant)
            # Circuit creation in JAX is fine as long as we extract metadata
            c_layer = tc.Circuit(self.N)
            layer_fn(c_layer, p_l, *args)

            ops = c_layer.to_qir()

            # 2. Key Step: Heisenberg Picture requires REVERSED order of gates
            # The last gate in the circuit is the first to act on the Observable.

            # ops is list of dicts.
            # We iterate reversed.

            def body_apply(s: Any, op: Any) -> Any:
                # Unpack op
                gate_name = op["name"]
                wires = op["index"]
                p_dict = op.get("parameters", {})
                param_val = None
                if p_dict:
                    # Heuristic: pass the dict or the first value?
                    # apply_gate handles dict.
                    param_val = p_dict

                return self.apply_gate(s, gate_name, wires, param_val)

            # Loop over ops in this layer
            for op in reversed(ops):
                state = body_apply(state, op)

            return state, None

        # Prepare inputs for scan
        scan_inputs = (
            params_batch if extra_inputs is None else (params_batch, *extra_inputs)
        )

        # Scan REVERSED over layers (Layer L, L-1... 1)
        final_state, _ = jax.lax.scan(scan_body, state, scan_inputs, reverse=True)

        return self.expectation(final_state)


def pauli_propagation(
    c: tc.Circuit,
    observable: Any,
    weights: Optional[Any] = None,
    k: int = 3,
) -> Any:
    """
    Computes the expectation value using Pauli Propagation.

    :param c: The Circuit object to simulate.
    :type c: tc.Circuit
    :param observable: The observable to measure. Can be a list of (coeff, Pauli_string)
                       or the Hamiltonian structures (M, N) array.
    :type observable: Union[List[Tuple[float, str]], Any]
    :param weights: Hamiltonian weights (M,) array. Required if observable is structures.
    :type weights: Optional[Any]
    :param k: Maximum locality of the Pauli strings to track.
    :type k: int
    :return: Expectation value.
    :rtype: Any
    """
    N = c._nqubits
    pp = PauliPropagationEngine(N, k)

    # Parse observables
    if weights is not None:
        # User passed (c, structures, weights, ...)
        state = pp.get_initial_state(observable, weights)
    elif (
        isinstance(observable, list)
        and len(observable) > 0
        and isinstance(observable[0], tuple)
    ):
        # User passed (c, list_of_tuples, ...)
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
        # User passed (c, (structures, weights), ...)
        state = pp.get_initial_state(observable[0], observable[1])
    else:
        raise ValueError(
            "observable must be list of (coeff, string), "
            "(structures, weights) tuple, or separate structures/weights args."
        )

    # Circuit Loop
    # We iterate reversed
    ops = c.to_qir()
    for op in reversed(ops):
        gate_name = op["name"]
        wires = op["index"]
        params_dict = op.get("parameters", {})
        # Pass params_dict directly if present, else None
        param_val = params_dict if params_dict else None

        state = pp.apply_gate(state, gate_name, wires, param_val)

    return pp.expectation(state)
