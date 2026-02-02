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

Note: This implementation is optimized for JAX and NumPy backends.
TensorFlow backend is not supported for this module.
"""

from itertools import combinations
from typing import Any, Callable, Optional, Sequence
import numpy as np

from . import gates
from .cons import backend
from .circuit import Circuit

Tensor = Any


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
        self.N = N
        self.k = k
        self.subsets = list(combinations(range(N), k))
        self.num_subsets = len(self.subsets)

        self.subset_arr = backend.convert_to_tensor(
            np.array(self.subsets, dtype=np.int32)
        )

        # Basis matrices for PTM construction
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

        self.valid_indices = backend.convert_to_tensor(
            np.array(valid_indices, dtype=np.int32)
        )

    def get_ptm_1q(self, u: Any) -> Any:
        """
        Compute 4x4 Pauli Transfer Matrix for a 1-qubit unitary U.
        R_ij = 0.5 * Tr(sigma_i U^dag sigma_j U)
        """
        u = backend.convert_to_tensor(u)
        u_dag = backend.conj(backend.transpose(u))

        # shape: (4, 2, 2)
        sigmas = self.pauli_mats

        # U^dag sigma_j U
        # (4, 2, 2)
        tmp = backend.matmul(u_dag, sigmas)
        rot_sigmas = backend.matmul(tmp, u)

        # Tr(sigma_i * rot_sigmas_j)
        # (4, 4)
        res = backend.matmul(
            backend.reshape(sigmas, [4, 4]),
            backend.transpose(backend.reshape(rot_sigmas, [4, 4])),
        )
        m = 0.5 * res
        return backend.real(m)  # PTM should be real for CPTP maps on Hermitian basis

    def get_ptm_2q(self, u: Any) -> Any:
        """
        Compute 16x16 Pauli Transfer Matrix for a 2-qubit unitary U.
        """
        u = backend.convert_to_tensor(u)
        u_dag = backend.conj(backend.transpose(u))

        # Tensor product of Paulis: (16, 4, 4)
        s1 = self.pauli_mats  # (4, 2, 2)

        # Vectorized broadcasting for JAX/NumPy
        sigmas_2q = backend.reshape(
            backend.reshape(s1, [4, 1, 2, 1, 2, 1])
            * backend.reshape(s1, [1, 4, 1, 2, 1, 2]),
            [16, 4, 4],
        )

        tmp = backend.matmul(u_dag, sigmas_2q)
        rot_sigmas = backend.matmul(tmp, u)

        # Normalization factor for 2Q is 1/4
        res = backend.matmul(
            backend.reshape(sigmas_2q, [16, 16]),
            backend.transpose(backend.reshape(rot_sigmas, [16, 16])),
        )
        m = 0.25 * res
        return backend.real(m)

    def get_initial_state(self, structures: Any, weights: Any) -> Any:
        """
        Initialize the propagation state from a Hamiltonian in TC internal format.

        :param structures: (M, N) integer array. 0:I, 1:X, 2:Y, 3:Z.
        :type structures: np.ndarray
        :param weights: (M,) float/complex array.
        :type weights: np.ndarray
        :return: Initial state tensor.
        :rtype: Tensor
        """
        # Ensure numpy for processing structures (must be static)
        structures = np.array(structures)

        indices = []
        updates = []

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

                indices.append([target_idx, flat_idx])
                updates.append(w)

        state = backend.zeros((self.num_subsets, 4**self.k), dtype="complex64")
        if len(indices) > 0:
            indices_tensor = backend.convert_to_tensor(
                np.array(indices, dtype=np.int32)
            )
            updates_tensor = backend.stack(updates)  # Use stack to handle tracers
            updates_tensor = backend.cast(updates_tensor, "complex64")
            state = backend.scatter(state, indices_tensor, updates_tensor, mode="add")

        return state

    def expectation(self, state: Any) -> Any:
        # Sum coefficients at valid indices (pure Z/I terms)
        # workaround for missing backend.gather with axis support
        valid_terms = backend.gather1d(backend.transpose(state), self.valid_indices)
        total = backend.sum(valid_terms)
        return backend.real(total)

    def _apply_1q_kernel(self, state_subset: Any, ptm: Any, local_idx: Any) -> Any:
        """
        Apply 1Q PTM to a specific local axis of the state tensor (flattened).
        Dispatches to static versions using backend.switch.
        """
        branches = []
        for i in range(self.k):

            def branch_fn(s: Any = state_subset, p: Any = ptm, idx: int = i) -> Any:
                return self._apply_1q_kernel_static(s, p, idx)

            branches.append(branch_fn)

        return backend.switch(local_idx, branches)

    def _apply_1q_kernel_static(
        self, state_subset: Any, ptm: Any, static_idx: Any
    ) -> Any:
        shape = (4,) * self.k
        tensor = backend.reshape(state_subset, shape)
        new_tensor = backend.tensordot(ptm, tensor, axes=([1], [static_idx]))

        # Permutation to restore order
        perm = (
            list(range(1, static_idx + 1)) + [0] + list(range(static_idx + 1, self.k))
        )
        new_tensor = backend.transpose(new_tensor, perm)
        return backend.reshape(new_tensor, [4**self.k])

    def _apply_2q_kernel(
        self, state_subset: Any, ptm: Any, idx1: Any, idx2: Any
    ) -> Any:
        """
        Apply 2Q PTM to local axes idx1, idx2.
        Dispatches via backend.switch on flattened index (idx1 * k + idx2).
        """
        flat_idx = idx1 * self.k + idx2

        branches = []
        for i in range(self.k * self.k):
            # Decode i -> r, c
            r = i // self.k
            c = i % self.k

            if r == c:

                def branch_fn(
                    s: Any = state_subset,
                    p: Any = ptm,
                    r_static: Any = r,
                    c_static: Any = c,
                ) -> Any:
                    return backend.reshape(s, [4**self.k])

            else:

                def branch_fn(
                    s: Any = state_subset,
                    p: Any = ptm,
                    r_static: Any = r,
                    c_static: Any = c,
                ) -> Any:
                    return self._apply_2q_kernel_static(s, p, r_static, c_static)

            branches.append(branch_fn)

        return backend.switch(flat_idx, branches)

    def _apply_2q_kernel_static(
        self, state_subset: Any, ptm: Any, idx1: Any, idx2: Any
    ) -> Any:
        shape = (4,) * self.k
        tensor = backend.reshape(state_subset, shape)
        ptm_reshaped = backend.reshape(ptm, [4, 4, 4, 4])

        # Contract
        new_tensor = backend.tensordot(
            ptm_reshaped, tensor, axes=([2, 3], [idx1, idx2])
        )

        # Permutation
        target_perm = [0] * self.k
        target_perm[idx1] = 0
        target_perm[idx2] = 1

        current_rest_ptr = 2
        for i in range(self.k):
            if i != idx1 and i != idx2:
                target_perm[i] = current_rest_ptr
                current_rest_ptr += 1

        new_tensor = backend.transpose(new_tensor, target_perm)
        return backend.reshape(new_tensor, [4**self.k])

    def _project_2q_boundary(
        self, state_subset: Any, ptm: Any, local_idx: Any, is_q1_in: Any
    ) -> Any:
        """
        Apply boundary update.
        PTM is 16x16 (q1, q2).
        If is_q1_in=True: q1 is in subset (at local_idx), q2 is out.
        We take submatrix M s.t. q2 input=I, q2 output=I.
        """
        # Reshape PTM to (4(out1), 4(out2), 4(in1), 4(in2))
        ptm_4 = backend.reshape(ptm, [4, 4, 4, 4])

        if is_q1_in:
            reduced_ptm = ptm_4[:, 0, :, 0]
        else:
            reduced_ptm = ptm_4[0, :, 0, :]

        return self._apply_1q_kernel(state_subset, reduced_ptm, local_idx)

    def apply_gate(
        self, state: Any, gate_name: str, wires: Any, params: Any = None
    ) -> Any:
        # 1. Get PTM
        gate_name = gate_name.lower()
        gate_func = getattr(gates, gate_name)

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

        # Ensure matrix shape and convert to backend tensor
        u = backend.convert_to_tensor(u)
        dim = 2 ** len(wires)
        u = backend.reshape(u, [dim, dim])

        if len(wires) == 1:
            ptm = self.get_ptm_1q(u)
            q = wires[0]

            has_q = backend.sum(backend.cast(self.subset_arr == q, "int32"), axis=1) > 0
            local_indices = backend.argmax(
                backend.cast(self.subset_arr == q, "int32"), axis=1
            )

            def update_fn_1q(s: Any, idx: Any, present: Any) -> Any:
                res = self._apply_1q_kernel(s, ptm, idx)
                return backend.where(present, res, s)

            state = backend.vmap(update_fn_1q, vectorized_argnums=(0, 1, 2))(
                state, local_indices, has_q
            )

        elif len(wires) == 2:
            ptm = self.get_ptm_2q(u)
            q1, q2 = wires

            mask1 = (
                backend.sum(backend.cast(self.subset_arr == q1, "int32"), axis=1) > 0
            )
            mask2 = (
                backend.sum(backend.cast(self.subset_arr == q2, "int32"), axis=1) > 0
            )

            idx1 = backend.argmax(backend.cast(self.subset_arr == q1, "int32"), axis=1)
            idx2 = backend.argmax(backend.cast(self.subset_arr == q2, "int32"), axis=1)

            def update_fn_2q(s: Any, m1: Any, m2: Any, i1: Any, i2: Any) -> Any:
                s_both = self._apply_2q_kernel(s, ptm, i1, i2)
                s_q1 = self._project_2q_boundary(s, ptm, i1, True)
                s_q2 = self._project_2q_boundary(s, ptm, i2, False)
                res = backend.where(
                    m1 & m2,
                    s_both,
                    backend.where(m1, s_q1, backend.where(m2, s_q2, s)),
                )
                return res

            state = backend.vmap(update_fn_2q, vectorized_argnums=(0, 1, 2, 3, 4))(
                state, mask1, mask2, idx1, idx2
            )

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
        :type params_batch: Tensor
        :param extra_inputs: Optional extra inputs to scan over or broadcast.
        :type extra_inputs: Optional[Sequence[Any]]
        :return: Scalar expectation value.
        :rtype: Tensor
        """
        # Initialize
        state = self.get_initial_state(ham_structures, ham_weights)

        # Define scan body
        def scan_body(state: Any, scan_inputs: Any) -> Any:
            if extra_inputs is None:
                p_l = scan_inputs
                args = ()
            else:
                p_l = scan_inputs[0]
                args = scan_inputs[1:]

            # Heisenberg Picture requires REVERSED order of gates
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

        # Prepare inputs for scan
        scan_inputs = (
            params_batch if extra_inputs is None else (params_batch, *extra_inputs)
        )

        # Simple reverse for JAX/NumPy (works for any rank tensor reverse on axis 0)
        if isinstance(scan_inputs, (tuple, list)):
            scan_inputs_rev = tuple([backend.reverse(x) for x in scan_inputs])
        else:
            scan_inputs_rev = backend.reverse(scan_inputs)

        final_state = backend.scan(scan_body, scan_inputs_rev, state)

        return self.expectation(final_state)


def pauli_propagation(
    c: Circuit,
    observable: Any,
    weights: Optional[Any] = None,
    k: int = 3,
) -> Any:
    """
    Computes the expectation value using Pauli Propagation.

    :param c: The Circuit object to simulate.
    :type c: Circuit
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
        raise ValueError(
            "observable must be list of (coeff, string), "
            "(structures, weights) tuple, or separate structures/weights args."
        )

    # Circuit Loop
    ops = c.to_qir()
    for op in reversed(ops):
        gate_name = op["name"]
        wires = op["index"]
        params_dict = op.get("parameters", {})
        param_val = params_dict if params_dict else None
        state = pp.apply_gate(state, gate_name, wires, param_val)

    return pp.expectation(state)
